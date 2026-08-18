#include "pipe-dense-segment-manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace pipe_dense_segment {
namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("dense segment manifest: " + message);
}

bool is_sha256(const std::string & value) {
    if (value.size() != 71 || value.compare(0, 7, "sha256:") != 0) {
        return false;
    }
    for (size_t i = 7; i < value.size(); ++i) {
        if (!std::isxdigit((unsigned char) value[i]) ||
            (value[i] >= 'A' && value[i] <= 'F')) {
            return false;
        }
    }
    return true;
}

uint64_t fnv1a64(const std::string & input) {
    uint64_t value = 14695981039346656037ull;
    for (unsigned char byte : input) {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}

std::string checksum_for(const json & root) {
    json unsigned_root = root;
    unsigned_root.erase("checksum");
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "fnv1a64:%016llx",
                  (unsigned long long) fnv1a64(unsigned_root.dump()));
    return buffer;
}

json parse_json(const std::string & json_text) {
    try {
        return json::parse(json_text);
    } catch (const json::exception & error) {
        fail(std::string("invalid JSON: ") + error.what());
    }
}

void require_object_keys(const json & object, const std::set<std::string> & allowed,
                         const char * name) {
    if (!object.is_object()) {
        fail(std::string(name) + " must be an object");
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (allowed.count(it.key()) == 0) {
            fail(std::string(name) + " contains unknown member '" + it.key() + "'");
        }
    }
}

const json & required(const json & object, const char * key, const char * name) {
    const auto it = object.find(key);
    if (it == object.end()) {
        fail(std::string(name) + " is missing '" + key + "'");
    }
    return *it;
}

std::string string_member(const json & object, const char * key, const char * name) {
    const json & value = required(object, key, name);
    if (!value.is_string()) {
        fail(std::string(name) + "." + key + " must be a string");
    }
    return value.get<std::string>();
}

int32_t i32_member(const json & object, const char * key, const char * name) {
    const json & value = required(object, key, name);
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        fail(std::string(name) + "." + key + " must be an integer");
    }
    if (value.is_number_unsigned()) {
        const uint64_t result = value.get<uint64_t>();
        if (result > (uint64_t) std::numeric_limits<int32_t>::max()) {
            fail(std::string(name) + "." + key + " is outside int32 range");
        }
        return (int32_t) result;
    }
    const int64_t result = value.get<int64_t>();
    if (result < std::numeric_limits<int32_t>::min() || result > std::numeric_limits<int32_t>::max()) {
        fail(std::string(name) + "." + key + " is outside int32 range");
    }
    return (int32_t) result;
}

uint32_t u32_member(const json & object, const char * key, const char * name) {
    const json & value = required(object, key, name);
    if (!value.is_number_unsigned() && !value.is_number_integer()) {
        fail(std::string(name) + "." + key + " must be an unsigned integer");
    }
    if (value.is_number_unsigned()) {
        const uint64_t result = value.get<uint64_t>();
        if (result > std::numeric_limits<uint32_t>::max()) {
            fail(std::string(name) + "." + key + " is outside uint32 range");
        }
        return (uint32_t) result;
    }
    const int64_t result = value.get<int64_t>();
    if (result < 0 || (uint64_t) result > std::numeric_limits<uint32_t>::max()) {
        fail(std::string(name) + "." + key + " is outside uint32 range");
    }
    return (uint32_t) result;
}

manifest parse_manifest_json(const json & root, const std::string & expected_model_identity_sha256) {
    require_object_keys(root, {
        "format", "version", "model_identity_sha256", "n_layer", "n_embd", "wire_precision",
        "segments", "checksum",
    }, "root");

    if (string_member(root, "format", "root") != MANIFEST_FORMAT) {
        fail("unsupported format");
    }
    if (u32_member(root, "version", "root") != MANIFEST_VERSION) {
        fail("unsupported version");
    }

    manifest result;
    result.model_identity_sha256 = string_member(root, "model_identity_sha256", "root");
    result.n_layer = i32_member(root, "n_layer", "root");
    result.n_embd = i32_member(root, "n_embd", "root");
    result.wire_precision = string_member(root, "wire_precision", "root");
    result.checksum = string_member(root, "checksum", "root");
    if (!is_sha256(result.model_identity_sha256)) {
        fail("model_identity_sha256 must be lowercase sha256:hex");
    }
    if (!expected_model_identity_sha256.empty() &&
        result.model_identity_sha256 != expected_model_identity_sha256) {
        fail("model identity does not match the local model");
    }
    if (result.n_layer <= 0 || result.n_embd <= 0 || result.wire_precision != "f32") {
        fail("invalid model dimensions or wire precision");
    }
    if (result.checksum != checksum_for(root)) {
        fail("checksum mismatch");
    }

    const json & json_segments = required(root, "segments", "root");
    if (!json_segments.is_array() || json_segments.empty()) {
        fail("segments must be a non-empty array");
    }
    std::set<uint32_t> ids;
    for (const json & entry : json_segments) {
        require_object_keys(entry, {
            "id", "layer_first", "layer_last", "host", "port", "stage_gguf", "device", "devices",
            "split_mode", "tensor_split", "tap_layers", "artifact_sha256",
        },
                            "segment");
        segment value;
        value.id = u32_member(entry, "id", "segment");
        value.layer_first = i32_member(entry, "layer_first", "segment");
        value.layer_last = i32_member(entry, "layer_last", "segment");
        value.target.host = string_member(entry, "host", "segment");
        value.stage_gguf = string_member(entry, "stage_gguf", "segment");

        const bool has_device  = entry.find("device") != entry.end();
        const bool has_devices = entry.find("devices") != entry.end();
        if (has_device == has_devices) {
            fail("segment " + std::to_string(value.id) + " must have exactly one of 'device' or 'devices'");
        }
        if (has_device) {
            value.devices.push_back(string_member(entry, "device", "segment"));
            if (value.devices.front().empty()) {
                fail("segment.device must be a non-empty string");
            }
        } else {
            const json & json_devices = entry["devices"];
            if (!json_devices.is_array() || json_devices.empty()) {
                fail("segment.devices must be a non-empty array");
            }
            std::set<std::string> seen;
            for (const json & dev : json_devices) {
                if (!dev.is_string() || dev.get<std::string>().empty()) {
                    fail("segment.devices entries must be non-empty strings");
                }
                if (!seen.insert(dev.get<std::string>()).second) {
                    fail("segment.devices contains a duplicate device");
                }
                value.devices.push_back(dev.get<std::string>());
            }
        }

        // Interior taps. Validated against this segment's own band below, once
        // layer_first/layer_last have themselves been range-checked.
        if (entry.find("tap_layers") != entry.end()) {
            const json & json_taps = entry["tap_layers"];
            if (!json_taps.is_array()) {
                fail("segment.tap_layers must be an array");
            }
            std::set<uint32_t> seen_taps;
            for (const json & tap : json_taps) {
                if (!tap.is_number_unsigned()) {
                    fail("segment.tap_layers entries must be unsigned integers");
                }
                const uint32_t lid = tap.get<uint32_t>();
                if (!seen_taps.insert(lid).second) {
                    fail("segment.tap_layers contains a duplicate layer");
                }
                value.tap_layers.push_back(lid);
            }
            // Ascending order is part of the wire contract: the head concatenates the
            // returned rows in this order and must be able to reproduce the mapping
            // without a second lookup table.
            std::sort(value.tap_layers.begin(), value.tap_layers.end());
        }

        if (entry.find("split_mode") != entry.end()) {
            value.split_mode = string_member(entry, "split_mode", "segment");
            if (value.split_mode != "tensor") {
                fail("segment.split_mode must be 'tensor' when present");
            }
            if (value.devices.size() < 2) {
                fail("segment.split_mode 'tensor' requires at least 2 devices");
            }
        }

        if (entry.find("tensor_split") != entry.end()) {
            const json & json_ts = entry["tensor_split"];
            if (!json_ts.is_array() || json_ts.size() != value.devices.size()) {
                fail("segment.tensor_split must be an array with one entry per device");
            }
            for (const json & frac : json_ts) {
                if (!frac.is_number() || frac.get<double>() <= 0.0) {
                    fail("segment.tensor_split entries must be positive numbers");
                }
                value.tensor_split.push_back((float) frac.get<double>());
            }
        }

        const uint32_t port = u32_member(entry, "port", "segment");
        value.artifact_sha256 = string_member(entry, "artifact_sha256", "segment");
        if (!ids.insert(value.id).second) {
            fail("duplicate segment id " + std::to_string(value.id));
        }
        if (value.layer_first < 0 || value.layer_last < value.layer_first ||
            value.layer_last >= result.n_layer || value.target.host.empty() ||
            value.stage_gguf.empty() || value.devices.empty() ||
            port == 0 || port > 65535 || !is_sha256(value.artifact_sha256)) {
            fail("invalid segment " + std::to_string(value.id));
        }
        // Taps are checked only now, because they are only meaningful once this
        // segment's own band is known to be sane. A tap outside the band could never
        // be produced: the segment's graph does not build that layer at all.
        for (const uint32_t lid : value.tap_layers) {
            if ((int32_t) lid < value.layer_first || (int32_t) lid > value.layer_last) {
                fail("segment " + std::to_string(value.id) + " declares tap layer " +
                     std::to_string(lid) + " outside its band [" +
                     std::to_string(value.layer_first) + ", " + std::to_string(value.layer_last) + "]");
            }
        }
        value.target.port = (uint16_t) port;
        result.segments.push_back(std::move(value));
    }

    std::sort(result.segments.begin(), result.segments.end(), [](const segment & a, const segment & b) {
        return a.layer_first != b.layer_first ? a.layer_first < b.layer_first : a.layer_last < b.layer_last;
    });
    int32_t expected_first = 0;
    for (const segment & value : result.segments) {
        if (value.layer_first != expected_first) {
            fail(value.layer_first < expected_first ? "segment layer ranges overlap" : "segment layer ranges have a gap");
        }
        expected_first = value.layer_last + 1;
    }
    if (expected_first != result.n_layer) {
        fail("segment layer ranges do not cover the model");
    }
    return result;
}

} // namespace

std::string manifest_checksum(const std::string & json_text) {
    const json root = parse_json(json_text);
    if (!root.is_object()) {
        fail("root must be an object");
    }
    return checksum_for(root);
}

manifest parse_manifest(const std::string & json_text, const std::string & expected_model_identity_sha256) {
    if (!expected_model_identity_sha256.empty() && !is_sha256(expected_model_identity_sha256)) {
        fail("expected model identity must be lowercase sha256:hex");
    }
    return parse_manifest_json(parse_json(json_text), expected_model_identity_sha256);
}

manifest load_manifest(const std::string & path, const std::string & expected_model_identity_sha256) {
    std::ifstream input(path);
    if (!input) {
        fail("cannot open " + path);
    }
    const std::string json_text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return parse_manifest(json_text, expected_model_identity_sha256);
}

} // namespace pipe_dense_segment
