// Tests for the router's kv-preserve preset keys (tools/server/server-router-groups.cpp and
// the wiring in tools/server/server-models.cpp): parse, the child argv through the real
// update_args() path, mode gating (download/estimate children get no flags), replicas,
// alias sanitisation, the slot-autosave conflict, and the stop-timeout floor through the real
// apply path. Driven through the real preset ini parser.

#include "server-router-groups.h"
#include "server-router-admission.h"
#include "server-models.h"

#include "arg.h"
#include "preset.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static common_presets load_presets(const std::string & path, std::vector<router_group_section> * sections) {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset global;
    const common_presets presets = ctx.load_from_ini(path, global);
    if (sections) {
        for (const auto & [name, preset] : presets) {
            sections->push_back(router_group_parse_section(preset, name));
        }
    }
    return presets;
}

// writes `ini` to a temp file, returns the error text of parsing it ("" when it parses)
static std::string parse_error(const std::string & ini) {
    const std::string path = "test-router-kv-preserve-tmp.ini";
    {
        std::ofstream f(path);
        f << ini;
    }
    std::string err;
    try {
        std::vector<router_group_section> sections;
        load_presets(path, &sections);
    } catch (const std::runtime_error & e) {
        err = e.what();
        assert(!err.empty());
    }
    std::remove(path.c_str());
    return err;
}

static const router_group_section & find(const std::vector<router_group_section> & v, const std::string & name) {
    for (const auto & s : v) {
        if (s.name == name) {
            return s;
        }
    }
    assert(false && "section not found");
    return v.front();
}

using argv_t = std::vector<std::string>;

// a model meta built the way the router builds one: preset from the ini, group keys parsed
static server_model_meta model_meta(const common_presets & presets, const std::string & name) {
    server_model_meta meta;
    meta.name   = name;
    meta.preset = presets.at(name);
    server_model_parse_group_keys(meta);
    return meta;
}

// the child argv for a mode, through the real update_args() path
static argv_t child_args(server_model_meta & meta, server_child_mode mode) {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    meta.update_args(ctx, "", mode);
    return meta.args;
}

static size_t count_of(const argv_t & a, const std::string & s) {
    return (size_t) std::count(a.begin(), a.end(), s);
}

// true when `flag` is immediately followed by `value`
static bool has_pair(const argv_t & a, const std::string & flag, const std::string & value) {
    for (size_t i = 0; i + 1 < a.size(); ++i) {
        if (a[i] == flag && a[i + 1] == value) {
            return true;
        }
    }
    return false;
}

static bool has_kv_flags(const argv_t & a) {
    return count_of(a, CHILD_ARG_SLOT_AUTOSAVE_DIR) > 0 || count_of(a, CHILD_ARG_SLOT_AUTOSAVE_BUDGET_MB) > 0;
}

int main() {
    std::vector<router_group_section> sections;
    const std::string fixture = "tests/router-fixtures/kv-preserve/kv-preserve.ini";
    const common_presets presets = load_presets(fixture, &sections);
    assert(sections.size() == 7);

    // parse: both keys are read verbatim into the section
    const auto & local = find(sections, "local");
    assert(local.kv_preserve_root == "/var/lib/kv");
    assert(local.kv_preserve_mb == 4096);
    assert(local.slot_autosave.empty());

    // plain section: no kv-preserve at all
    const auto & plain = find(sections, "plain");
    assert(plain.kv_preserve_root.empty());
    assert(plain.kv_preserve_mb == 0);

    // ---- child argv through the real update_args() path ----

    // local, serving child: dir and budget
    {
        auto meta = model_meta(presets, "local");
        const argv_t a = child_args(meta, SERVER_CHILD_MODE_NORMAL);
        assert(has_pair(a, "--slot-autosave-dir", "/var/lib/kv/local"));
        assert(has_pair(a, "--slot-autosave-budget-mb", "4096"));
        assert(count_of(a, "--slot-autosave-dir") == 1);

        // rendering again must not duplicate the flags
        const argv_t again = child_args(meta, SERVER_CHILD_MODE_NORMAL);
        assert(count_of(again, "--slot-autosave-dir") == 1);
    }

    // remote (machine=) serving child: the root is verbatim and there is no budget
    {
        auto meta = model_meta(presets, "remote");
        assert(meta.machine == "mad-lab-2026");
        const argv_t a = child_args(meta, SERVER_CHILD_MODE_NORMAL);
        assert(has_pair(a, "--slot-autosave-dir", "/var/lib/kv/remote"));
        assert(count_of(a, "--slot-autosave-budget-mb") == 0);
    }

    // group spine: the flags of the spine itself
    {
        auto meta = model_meta(presets, "spine");
        assert(meta.kind == ROUTER_KIND_MODEL && !meta.depends.empty());
        const argv_t a = child_args(meta, SERVER_CHILD_MODE_NORMAL);
        assert(has_pair(a, "--slot-autosave-dir", "/var/lib/kv/spine"));
        assert(has_pair(a, "--slot-autosave-budget-mb", "2048"));
    }

    // a worker of the group never gets the flags
    {
        auto meta = model_meta(presets, "spine-w1");
        assert(meta.is_external());
        assert(!has_kv_flags(child_args(meta, SERVER_CHILD_MODE_NORMAL)));
    }

    // replica: the router names it "alias~rK" and it gets its own subdirectory
    {
        auto alias = model_meta(presets, "local");
        server_model_meta replica = alias; // make_replica_meta_locked copies the alias meta
        replica.name = router_replica_name("local", 2);
        const argv_t a = child_args(replica, SERVER_CHILD_MODE_NORMAL);
        assert(has_pair(a, "--slot-autosave-dir", "/var/lib/kv/local_r2"));
        assert(has_pair(a, "--slot-autosave-budget-mb", "4096"));
    }

    // download and estimate children never get the flags, whatever the model
    {
        auto local_meta = model_meta(presets, "local");
        assert(!has_kv_flags(child_args(local_meta, SERVER_CHILD_MODE_DOWNLOAD)));
        assert(!has_kv_flags(child_args(local_meta, SERVER_CHILD_MODE_ESTIMATE)));

        auto spine_meta = model_meta(presets, "spine");
        assert(!has_kv_flags(child_args(spine_meta, SERVER_CHILD_MODE_DOWNLOAD)));
        assert(!has_kv_flags(child_args(spine_meta, SERVER_CHILD_MODE_ESTIMATE)));
    }

    // plain model: no flags
    {
        auto meta = model_meta(presets, "plain");
        assert(!has_kv_flags(child_args(meta, SERVER_CHILD_MODE_NORMAL)));
    }

    // ---- stop-timeout through the real apply path ----
    {
        auto meta = model_meta(presets, "local");
        meta.stop_timeout = 10; // the default the router builds models with
        server_model_apply_stop_timeout(meta);
        assert(meta.stop_timeout == ROUTER_KV_PRESERVE_STOP_TIMEOUT_S);
        assert(meta.stop_timeout == 60);
    }
    {
        auto meta = model_meta(presets, "explicit"); // stop-timeout = 15 in the preset
        meta.stop_timeout = 10;
        server_model_apply_stop_timeout(meta);
        assert(meta.stop_timeout == 15); // explicit wins over the floor
    }
    {
        auto meta = model_meta(presets, "plain");
        meta.stop_timeout = 10;
        server_model_apply_stop_timeout(meta);
        assert(meta.stop_timeout == 10);
    }
    {
        auto meta = model_meta(presets, "spine-w1"); // a worker: no floor
        meta.stop_timeout = 10;
        server_model_apply_stop_timeout(meta);
        assert(meta.stop_timeout == 10);
    }
    {
        // unparsable explicit value: falls back to the default and the floor is skipped
        auto meta = model_meta(presets, "badst");
        meta.stop_timeout = 10;
        server_model_apply_stop_timeout(meta);
        assert(meta.stop_timeout == 10);
    }

    // pure floor rules
    assert(router_stop_timeout_floor(10, false, true) == 60);
    assert(router_stop_timeout_floor(10, false, false) == 10);
    assert(router_stop_timeout_floor(120, false, true) == 120); // never lowered
    assert(router_stop_timeout_floor(15, true, true) == 15);

    // conflict: kv-preserve and slot-autosave in one section name both keys
    {
        const std::string err = parse_error(
            "[c]\nmodel = /m.gguf\nkv-preserve = /kv\nslot-autosave = /s.slot\n");
        assert(err.find("kv-preserve") != std::string::npos);
        assert(err.find("slot-autosave") != std::string::npos);
    }

    // kv-preserve-mb without kv-preserve is refused
    {
        const std::string err = parse_error("[m]\nmodel = /m.gguf\nkv-preserve-mb = 100\n");
        assert(err.find("kv-preserve-mb") != std::string::npos && err.find("kv-preserve") != std::string::npos);
    }

    // a budget that is not a whole number is refused, naming the section
    {
        const std::string err = parse_error("[m]\nmodel = /m.gguf\nkv-preserve = /kv\nkv-preserve-mb = 12x\n");
        assert(err.find("kv-preserve-mb") != std::string::npos && err.find("'m'") != std::string::npos);
    }

    // alias sanitisation: anything outside [A-Za-z0-9._-] becomes '_'
    assert(router_kv_preserve_dirname("qwen3.5-27b_Q4") == "qwen3.5-27b_Q4");
    assert(router_kv_preserve_dirname("my model/x:y") == "my_model_x_y");
    assert(router_kv_preserve_dirname("a..b") == "a..b");
    assert(router_kv_preserve_dirname(".") == "_");
    assert(router_kv_preserve_dirname("..") == "_");

    // replica naming used by the subdirectory
    assert(router_replica_name("qwen", 1) == "qwen");
    assert(router_kv_preserve_dirname(router_replica_name("qwen", 2)) == "qwen_r2");
    assert(router_kv_preserve_dirname(router_replica_name("qwen", 3)) == "qwen_r3");

    return 0;
}
