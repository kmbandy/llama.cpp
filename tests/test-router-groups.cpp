// Tests for the router's model-group preset keys (tools/server/server-router-groups.cpp),
// driven through the real preset ini parser.

#include "server-router-groups.h"

#include "preset.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static std::vector<router_group_section> load_sections(const std::string & path) {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset global;
    const common_presets presets = ctx.load_from_ini(path, global);
    std::vector<router_group_section> out;
    for (const auto & [name, preset] : presets) {
        out.push_back(router_group_parse_section(preset, name));
    }
    return out;
}

// writes `ini` to a temp file, returns the error text of resolving it ("" when it resolves)
static std::string resolve_error(const std::string & ini) {
    const std::string path = "test-router-groups-tmp.ini";
    {
        std::ofstream f(path);
        f << ini;
    }
    std::string err;
    try {
        router_groups_resolve(load_sections(path));
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

int main() {
    // fixture: one spine + two externals parse into exactly one group
    {
        const auto sections = load_sections("tests/router-fixtures/groups/groups.ini");
        assert(sections.size() == 3);

        const auto & spine = find(sections, "dsv41");
        assert(spine.kind == ROUTER_KIND_MODEL);
        assert((spine.depends == std::vector<std::string>{ "dsv41-w-main", "dsv41-w-2026" }));
        assert(spine.machine == "mad-lab-main");
        assert(spine.gpu == "ROCm1");
        assert(spine.slot_autosave == "/var/lib/router/dsv41.slot");
        assert(spine.startup_timeout_s == 600);

        const auto & w_main = find(sections, "dsv41-w-main");
        assert(w_main.kind == ROUTER_KIND_EXTERNAL);
        assert(w_main.launch == "/opt/bin/expert-worker --port 9001 --shard 0");
        assert(w_main.machine == "mad-lab-main");
        assert(w_main.gpu == "ROCm0");
        assert(w_main.park_file == "/var/lib/router/dsv41-w-main.park");
        assert(w_main.park_mode == ROUTER_PARK_OPT_IN);
        assert(w_main.startup_timeout_s == 300); // default

        const auto & w_2026 = find(sections, "dsv41-w-2026");
        assert(w_2026.kind == ROUTER_KIND_EXTERNAL);
        assert(w_2026.machine == "mad-lab-2026");
        assert(w_2026.park_mode == ROUTER_PARK_NONE); // default

        const auto groups = router_groups_resolve(sections);
        assert(groups.size() == 2);
        assert(groups.at("dsv41-w-main") == "dsv41");
        assert(groups.at("dsv41-w-2026") == "dsv41");
        // requestability and the /v1/models listing, decided from the parsed kinds
        assert(router_model_requestable(spine.kind));
        assert(!router_model_requestable(w_main.kind));
        assert(!router_model_requestable(w_2026.kind));
        assert(router_model_in_oai_listing(spine.kind));
        assert(!router_model_in_oai_listing(w_main.kind));
        assert(!router_model_in_oai_listing(w_2026.kind));
    }

    // a preset with no group keys is a plain model in no group
    {
        const auto groups = router_groups_resolve({ router_group_section{ "plain" } });
        assert(groups.empty());
    }

    // dangling depends: no such section
    {
        const std::string err = resolve_error(
            "[spine]\nmodel = /m.gguf\ndepends = ghost\n");
        assert(err.find("spine") != std::string::npos && err.find("ghost") != std::string::npos);
    }

    // depends on a section that exists but is not kind=external
    {
        const std::string err = resolve_error(
            "[spine]\nmodel = /m.gguf\ndepends = other\n\n[other]\nmodel = /o.gguf\n");
        assert(err.find("spine") != std::string::npos && err.find("other") != std::string::npos &&
               err.find("external") != std::string::npos);
    }

    // a worker referenced by two groups
    {
        const std::string err = resolve_error(
            "[a]\nmodel = /a.gguf\ndepends = w\n\n[b]\nmodel = /b.gguf\ndepends = w\n\n"
            "[w]\nkind = external\nlaunch = /bin/worker\n");
        assert(err.find("'w'") != std::string::npos);
        assert(err.find("a") != std::string::npos && err.find("b") != std::string::npos);
    }

    // kind=external without launch
    {
        const std::string err = resolve_error("[w]\nkind = external\n");
        assert(err.find("'w'") != std::string::npos && err.find("launch") != std::string::npos);
    }

    // a launch that is not an absolute path (PATH is never searched) fails naming the section
    {
        const std::string err = resolve_error("[w]\nkind = external\nlaunch = expert-worker --port 9001\n");
        assert(err.find("'w'") != std::string::npos && err.find("absolute path") != std::string::npos);
    }

    // a launch that is not a single command fails naming the section
    {
        const std::string err = resolve_error("[w]\nkind = external\nlaunch = /bin/worker > /tmp/log\n");
        assert(err.find("'w'") != std::string::npos && err.find("launch") != std::string::npos);
    }

    // bad enum values fail naming the section
    {
        const std::string path = "test-router-groups-tmp2.ini";
        {
            std::ofstream f(path);
            f << "[x]\nkind = bogus\n";
        }
        bool threw = false;
        try {
            load_sections(path);
        } catch (const std::runtime_error & e) {
            threw = std::string(e.what()).find("'x'") != std::string::npos;
        }
        std::remove(path.c_str());
        assert(threw);
    }

    // worker-port: parsed for an external, refused when out of range
    {
        const std::string path = "test-router-groups-tmp3.ini";
        {
            std::ofstream f(path);
            f << "[w]\nkind = external\nlaunch = /bin/worker\nworker-port = 9100\n";
        }
        const auto sections = load_sections(path);
        std::remove(path.c_str());
        assert(find(sections, "w").worker_port == 9100);

        {
            std::ofstream f(path);
            f << "[w]\nkind = external\nlaunch = /bin/worker\nworker-port = 70000\n";
        }
        bool threw = false;
        try {
            load_sections(path);
        } catch (const std::runtime_error & e) {
            threw = std::string(e.what()).find("worker-port") != std::string::npos;
        }
        std::remove(path.c_str());
        assert(threw);
    }

    return 0;
}
