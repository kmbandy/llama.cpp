#pragma once

// Model groups for the router: one spine (a normal router-spawned model) plus expert
// workers declared as `kind=external` preset sections. Pure parsing and validation of the
// preset keys, no router state, so it is unit-testable with literals or a real ini.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "preset.h"

static constexpr const char * ROUTER_ARG_KIND            = "LLAMA_ARG_ROUTER_KIND";
static constexpr const char * ROUTER_ARG_DEPENDS         = "LLAMA_ARG_ROUTER_DEPENDS";
static constexpr const char * ROUTER_ARG_LAUNCH          = "LLAMA_ARG_ROUTER_LAUNCH";
static constexpr const char * ROUTER_ARG_PARK_FILE       = "LLAMA_ARG_ROUTER_PARK_FILE";
static constexpr const char * ROUTER_ARG_PARK_MODE       = "LLAMA_ARG_ROUTER_PARK_MODE";
static constexpr const char * ROUTER_ARG_MACHINE         = "LLAMA_ARG_ROUTER_MACHINE";
static constexpr const char * ROUTER_ARG_STARTUP_TIMEOUT = "LLAMA_ARG_ROUTER_STARTUP_TIMEOUT";
// kind=external only: TCP port the worker listens on, when it cannot be read off `launch`
// (`--listen HOST:PORT` / `--port N`). Not `port`: the router's own --port is merged into
// every section, so LLAMA_ARG_PORT cannot tell a worker's port apart.
static constexpr const char * ROUTER_ARG_WORKER_PORT     = "LLAMA_ARG_ROUTER_WORKER_PORT";
// not router-only: forwarded to the spine child as-is (llama-server --slot-autosave)
static constexpr const char * ROUTER_ARG_SLOT_AUTOSAVE   = "LLAMA_ARG_SLOT_AUTOSAVE";

enum router_kind {
    ROUTER_KIND_MODEL    = 0, // a normal router-spawned llama-server
    ROUTER_KIND_EXTERNAL = 1, // an expert worker launched outside the router's model path
};

enum router_park_mode {
    ROUTER_PARK_NONE   = 0,
    ROUTER_PARK_OPT_IN = 1, // reserves the SIGUSR1/SIGUSR2 park path; nothing uses it yet
};

// The group-related preset keys of one section, as written (values verbatim).
struct router_group_section {
    std::string              name;
    router_kind              kind = ROUTER_KIND_MODEL;
    std::vector<std::string> depends;
    std::string              launch;   // full command line, stored verbatim
    std::string              machine;  // "" = local
    std::string              gpu;
    std::string              park_file;
    router_park_mode         park_mode = ROUTER_PARK_NONE;
    std::string              slot_autosave;
    int                      startup_timeout_s = 300;
    int                      worker_port = 0; // 0 = derive from launch
};

// Reads the group keys out of a preset. Must run before unset_reserved_args() strips the
// router-only keys. Throws std::runtime_error naming the section on a bad kind / park-mode /
// startup-timeout / worker-port value.
router_group_section router_group_parse_section(const common_preset & preset, const std::string & name);

// Validates all sections together and returns worker name -> spine name. Throws
// std::runtime_error naming the offending section when: a kind=external section has no
// launch, `depends` names a missing section or one that is not kind=external, a worker is
// referenced by two groups (or twice by one), or an external itself declares depends.
std::map<std::string, std::string> router_groups_resolve(const std::vector<router_group_section> & sections);

// A request may name only a real model; a kind=external worker answers like an unknown name.
inline bool router_model_requestable(router_kind kind) {
    return kind == ROUTER_KIND_MODEL;
}

// /v1/models is the OAI view: loadable models only. (/models lists every section.)
inline bool router_model_in_oai_listing(router_kind kind) {
    return kind == ROUTER_KIND_MODEL;
}
