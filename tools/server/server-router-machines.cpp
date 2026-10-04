#include "server-router-machines.h"

#include "json.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#ifndef _WIN32
#include <unistd.h>
#endif

using json = common_json;

const machine_entry * machines_registry::find(const std::string & name) const {
    for (const auto & m : machines) {
        if (m.name == name) {
            return &m;
        }
    }
    return nullptr;
}

std::string machines_registry::local_machine() const {
    for (const auto & m : machines) {
        if (m.local) {
            return m.name;
        }
    }
    return "";
}

std::string machines_registry::node_url(const std::string & name) const {
    const machine_entry * m = find(name);
    return m ? m->router_node : "";
}

std::string machines_default_path() {
    const char * home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return "";
    }
    return std::string(home) + "/.config/mad-lab-agents/machines.json";
}

static std::string string_field(const json & entry, const char * key) {
    if (entry.contains(key) && entry.at(key).is_string()) {
        return entry.at(key).get<std::string>();
    }
    return "";
}

machines_registry parse_machines(const std::string & json_text) {
    machines_registry reg;
    try {
        const json j = json::parse(json_text);
        if (!j.is_object()) {
            reg.error = "machines.json: top level is not an object";
            return reg;
        }
        for (const auto & [name, entry] : j.items()) {
            if (!entry.is_object()) {
                continue;
            }
            machine_entry m;
            m.name        = name;
            m.local       = entry.contains("local") && entry.at("local").is_boolean() && entry.at("local").get<bool>();
            m.ssh         = string_field(entry, "ssh");
            m.models_dir  = string_field(entry, "models_dir");
            m.router_node = string_field(entry, "router_node");
            reg.machines.push_back(std::move(m));
        }
    } catch (const std::exception & e) {
        reg.machines.clear();
        reg.error = std::string("machines.json: ") + e.what();
    }
    return reg;
}

machines_registry load_machines(const std::string & path_in) {
    const std::string path = path_in.empty() ? machines_default_path() : path_in;
    if (path.empty()) {
        machines_registry reg;
        reg.error = "machines.json: no path (HOME is unset)";
        return reg;
    }
    std::ifstream f(path);
    if (!f) {
        machines_registry reg;
        reg.error = "cannot read " + path;
        return reg;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_machines(ss.str());
}

std::string router_parse_local_machine(const std::string & machines_json) {
    return parse_machines(machines_json).local_machine();
}

std::string router_local_machine(const std::string & path) {
    const std::string name = load_machines(path).local_machine();
    if (!name.empty()) {
        return name;
    }
#ifndef _WIN32
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0] != '\0') {
        std::string host = buf;
        return host.substr(0, host.find('.'));
    }
#endif
    return "local";
}
