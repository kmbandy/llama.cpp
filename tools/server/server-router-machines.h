#pragma once

// The machine registry: ~/.config/mad-lab-agents/machines.json, shared with other tools
// (Python / Rust / TUI readers), so unknown keys are ignored here too. Shape:
//
//   { "<name>": { "local": true?, "ssh": "user@host"?, "models_dir": "/path"?,
//                 "router_node": "http://host:port"? }, ... }
//
// `local` is per box: each box's copy marks itself. `router_node` is the URL of that
// machine's --router-node daemon (the leader drives remote machines through it).

#include <string>
#include <vector>

struct machine_entry {
    std::string name;
    bool        local = false;
    std::string ssh;         // "" when absent
    std::string models_dir;  // "" when absent
    std::string router_node; // "" when absent: no node daemon on that machine
};

struct machines_registry {
    std::vector<machine_entry> machines; // in file order
    std::string                error;    // "" when the file was read and parsed

    bool ok() const { return error.empty(); }

    // nullptr when no machine has that name
    const machine_entry * find(const std::string & name) const;

    // the first machine marked `local: true`; "" when none is
    std::string local_machine() const;

    // `router_node` of that machine; "" when the machine is unknown or has none
    std::string node_url(const std::string & name) const;
};

// ~/.config/mad-lab-agents/machines.json; "" when HOME is unset
std::string machines_default_path();

// Parses the JSON text. Entries that are not objects are skipped, keys of the wrong type
// count as absent, unknown keys are ignored. Not an object / not JSON -> error set, no machines.
machines_registry parse_machines(const std::string & json_text);

// Reads and parses `path` (machines_default_path() when empty). A missing or unreadable
// file gives an empty registry with error set; never throws.
machines_registry load_machines(const std::string & path = "");

// Name of this machine: the `local: true` entry of machines.json at `path` (default path
// when empty), else the short hostname, else "local".
std::string router_local_machine(const std::string & path = "");

// The `local: true` entry of a machines.json text; "" when none / unparsable.
std::string router_parse_local_machine(const std::string & machines_json);
