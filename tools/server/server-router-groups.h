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
// router-only: root directory (on the child's machine) for per-model slot KV autosave. The
// child gets --slot-autosave-dir <root>/<model dirname>. Cannot be combined with slot-autosave.
static constexpr const char * ROUTER_ARG_KV_PRESERVE     = "LLAMA_ARG_ROUTER_KV_PRESERVE";
// router-only: budget in MiB for that directory (--slot-autosave-budget-mb); 0 = no budget.
static constexpr const char * ROUTER_ARG_KV_PRESERVE_MB  = "LLAMA_ARG_ROUTER_KV_PRESERVE_MB";
// child flags, implemented by llama-server; literal strings, not presets
static constexpr const char * CHILD_ARG_SLOT_AUTOSAVE_DIR       = "--slot-autosave-dir";
static constexpr const char * CHILD_ARG_SLOT_AUTOSAVE_BUDGET_MB = "--slot-autosave-budget-mb";
// stop-timeout floor for a kv-preserve child, so a multi-slot save is not SIGKILLed
static constexpr int ROUTER_KV_PRESERVE_STOP_TIMEOUT_S = 60;
// largest kv-preserve-mb: MiB -> bytes must fit in uint64 (the child's --slot-autosave-budget-mb
// is parsed with the same cap in common/arg.cpp)
static constexpr uint64_t ROUTER_KV_PRESERVE_MB_MAX = UINT64_MAX / (1024ull * 1024ull);

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
    std::string              kv_preserve_root; // "" = kv-preserve not set
    uint64_t                 kv_preserve_mb = 0; // 0 = no budget flag
};

// Reads the group keys out of a preset. Must run before unset_reserved_args() strips the
// router-only keys. Throws std::runtime_error naming the section on a bad kind / park-mode /
// startup-timeout / worker-port / kv-preserve-mb value, on kv-preserve-mb without kv-preserve,
// and on kv-preserve together with slot-autosave (both keys named).
router_group_section router_group_parse_section(const common_preset & preset, const std::string & name);

// The subdirectory name under a kv-preserve root for a router model name: every byte outside
// [A-Za-z0-9._-] becomes '_' (so a replica "alias~r2" -> "alias_r2"); "." and ".." become "_".
std::string router_kv_preserve_dirname(const std::string & router_name);

// The child argv fragment for a kv-preserve model: --slot-autosave-dir <root>/<dirname>, plus
// --slot-autosave-budget-mb <mb> when mb > 0. The root is used verbatim.
std::vector<std::string> router_kv_preserve_child_args(const std::string & root, const std::string & router_name,
                                                       uint64_t budget_mb);

// The stop-timeout of a model: an explicit preset stop-timeout always wins; otherwise a
// kv-preserve child is raised to at least ROUTER_KV_PRESERVE_STOP_TIMEOUT_S.
int router_stop_timeout_floor(int current_s, bool has_explicit, bool kv_preserve_applies);

// Validates all sections together and returns worker name -> spine name. Throws
// std::runtime_error naming the offending section when: a kind=external section has no
// launch or one router_parse_launch() refuses (e.g. not an absolute path), `depends` names a missing section or one that is not kind=external, a worker is
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
