// Tests for the router's kv-preserve preset keys (tools/server/server-router-groups.cpp):
// parse, the child argv fragment, alias sanitisation, the slot-autosave conflict, the
// stop-timeout floor and replica subdirectories. Driven through the real preset ini parser.

#include "server-router-groups.h"
#include "server-router-admission.h"

#include "arg.h"
#include "preset.h"

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

int main() {
    std::vector<router_group_section> sections;
    const std::string fixture = "tests/router-fixtures/kv-preserve/kv-preserve.ini";
    const common_presets presets = load_presets(fixture, &sections);
    assert(sections.size() == 6);

    // parse: both keys are read verbatim into the section
    const auto & local = find(sections, "local");
    assert(local.kv_preserve_root == "/var/lib/kv");
    assert(local.kv_preserve_mb == 4096);
    assert(local.slot_autosave.empty());

    // plain section: no kv-preserve at all
    const auto & plain = find(sections, "plain");
    assert(plain.kv_preserve_root.empty());
    assert(plain.kv_preserve_mb == 0);

    // local child argv fragment: dir and budget
    assert((router_kv_preserve_child_args(local.kv_preserve_root, local.name, local.kv_preserve_mb) ==
            argv_t{ "--slot-autosave-dir", "/var/lib/kv/local", "--slot-autosave-budget-mb", "4096" }));

    // remote child: machine= does not change the fragment, and the root is used verbatim (no budget)
    const auto & remote = find(sections, "remote");
    assert(remote.machine == "mad-lab-2026");
    assert(remote.kv_preserve_mb == 0);
    assert((router_kv_preserve_child_args(remote.kv_preserve_root, remote.name, remote.kv_preserve_mb) ==
            argv_t{ "--slot-autosave-dir", "/var/lib/kv/remote" }));

    // group spine: the flags of the spine itself, under its own subdirectory
    const auto & spine = find(sections, "spine");
    assert(spine.kind == ROUTER_KIND_MODEL);
    assert(!spine.depends.empty());
    assert((router_kv_preserve_child_args(spine.kv_preserve_root, spine.name, spine.kv_preserve_mb) ==
            argv_t{ "--slot-autosave-dir", "/var/lib/kv/spine", "--slot-autosave-budget-mb", "2048" }));

    // a worker of the group parses the key too, but it is kind=external: the router never
    // applies kv-preserve to a worker (checked in server_model_meta::kv_preserve_applies())
    const auto & worker = find(sections, "spine-w1");
    assert(worker.kind == ROUTER_KIND_EXTERNAL);
    assert(worker.kv_preserve_root == "/var/lib/kv");

    // stop-timeout: no explicit value gets the 60 s floor; an explicit one always wins
    assert(router_stop_timeout_floor(10, false, true) == 60);
    assert(router_stop_timeout_floor(10, false, false) == 10);
    assert(router_stop_timeout_floor(120, false, true) == 120); // never lowered
    assert(router_stop_timeout_floor(15, true, true) == 15);     // explicit wins
    assert(router_stop_timeout_floor(10, true, true) == 10);     // explicit wins even at the default

    {
        const auto & explicit_sec = find(sections, "explicit");
        std::string val;
        const bool has_explicit = presets.at("explicit").get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val);
        assert(has_explicit && val == "15");
        assert(router_stop_timeout_floor(15, has_explicit, !explicit_sec.kv_preserve_root.empty()) == 15);

        std::string none;
        const bool plain_explicit = presets.at("plain").get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, none);
        assert(!plain_explicit);
        assert(router_stop_timeout_floor(10, plain_explicit, !plain.kv_preserve_root.empty()) == 10);
    }

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

    // replicas: each gets its own subdirectory, named from the router's replica name
    assert(router_replica_name("qwen", 1) == "qwen");
    assert(router_kv_preserve_dirname(router_replica_name("qwen", 2)) == "qwen_r2");
    assert(router_kv_preserve_dirname(router_replica_name("qwen", 3)) == "qwen_r3");
    assert((router_kv_preserve_child_args("/var/lib/kv", router_replica_name("qwen", 3), 0) ==
            argv_t{ "--slot-autosave-dir", "/var/lib/kv/qwen_r3" }));

    return 0;
}
