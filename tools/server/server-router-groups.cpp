#include "server-router-groups.h"

#include "common.h"
#include "preset.h"

#include <set>
#include <stdexcept>

router_group_section router_group_parse_section(const common_preset & preset, const std::string & name) {
    router_group_section s;
    s.name = name;

    std::string val;
    if (preset.get_option(ROUTER_ARG_KIND, val)) {
        val = string_strip(val);
        if (val == "external") {
            s.kind = ROUTER_KIND_EXTERNAL;
        } else if (!val.empty() && val != "model") {
            throw std::runtime_error(string_format(
                "preset '%s': invalid kind '%s' (expected 'model' or 'external')", name.c_str(), val.c_str()));
        }
    }
    if (preset.get_option(ROUTER_ARG_DEPENDS, val)) {
        for (auto dep : string_split<std::string>(val, ',')) {
            dep = string_strip(dep);
            if (!dep.empty()) {
                s.depends.push_back(dep);
            }
        }
    }
    if (preset.get_option(ROUTER_ARG_LAUNCH, val)) {
        s.launch = string_strip(val);
    }
    if (preset.get_option(ROUTER_ARG_MACHINE, val)) {
        s.machine = string_strip(val);
    }
    if (preset.get_option("LLAMA_ARG_ROUTER_GPU", val)) {
        s.gpu = string_strip(val);
    }
    if (preset.get_option(ROUTER_ARG_PARK_FILE, val)) {
        s.park_file = string_strip(val);
    }
    if (preset.get_option(ROUTER_ARG_PARK_MODE, val)) {
        val = string_strip(val);
        if (val == "opt-in") {
            s.park_mode = ROUTER_PARK_OPT_IN;
        } else if (!val.empty() && val != "none") {
            throw std::runtime_error(string_format(
                "preset '%s': invalid park-mode '%s' (expected 'none' or 'opt-in')", name.c_str(), val.c_str()));
        }
    }
    if (preset.get_option(ROUTER_ARG_SLOT_AUTOSAVE, val)) {
        s.slot_autosave = val;
    }
    if (preset.get_option(ROUTER_ARG_STARTUP_TIMEOUT, val) && !val.empty()) {
        int t = 0;
        try {
            t = std::stoi(val);
        } catch (...) {
            t = 0;
        }
        if (t <= 0) {
            throw std::runtime_error(string_format(
                "preset '%s': invalid startup-timeout '%s' (must be a positive number of seconds)",
                name.c_str(), val.c_str()));
        }
        s.startup_timeout_s = t;
    }
    if (preset.get_option(ROUTER_ARG_WORKER_PORT, val) && !val.empty()) {
        int p = 0;
        try {
            p = std::stoi(val);
        } catch (...) {
            p = 0;
        }
        if (p <= 0 || p > 65535) {
            throw std::runtime_error(string_format(
                "preset '%s': invalid worker-port '%s' (must be 1-65535)", name.c_str(), val.c_str()));
        }
        s.worker_port = p;
    }
    return s;
}

std::map<std::string, std::string> router_groups_resolve(const std::vector<router_group_section> & sections) {
    std::map<std::string, const router_group_section *> by_name;
    for (const auto & s : sections) {
        by_name[s.name] = &s;
    }

    std::map<std::string, std::string> worker_to_spine;
    for (const auto & s : sections) {
        if (s.kind == ROUTER_KIND_EXTERNAL) {
            if (s.launch.empty()) {
                throw std::runtime_error(string_format(
                    "preset '%s': kind=external requires a non-empty 'launch'", s.name.c_str()));
            }
            if (!s.depends.empty()) {
                throw std::runtime_error(string_format(
                    "preset '%s': kind=external sections cannot have 'depends'", s.name.c_str()));
            }
            continue;
        }
        for (const auto & dep : s.depends) {
            auto it = by_name.find(dep);
            if (it == by_name.end()) {
                throw std::runtime_error(string_format(
                    "preset '%s': depends on '%s', which is not a section in the preset", s.name.c_str(), dep.c_str()));
            }
            if (it->second->kind != ROUTER_KIND_EXTERNAL) {
                throw std::runtime_error(string_format(
                    "preset '%s': depends on '%s', which is not kind=external", s.name.c_str(), dep.c_str()));
            }
            auto ins = worker_to_spine.emplace(dep, s.name);
            if (!ins.second) {
                throw std::runtime_error(string_format(
                    "preset '%s': worker '%s' is already part of group '%s' (a worker may belong to one group only)",
                    s.name.c_str(), dep.c_str(), ins.first->second.c_str()));
            }
        }
    }
    return worker_to_spine;
}
