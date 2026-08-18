#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pipe_dense_segment {

static constexpr const char * MANIFEST_FORMAT = "llama.cpp.dense-segment-manifest";
static constexpr uint32_t MANIFEST_VERSION = 1;

struct endpoint {
    std::string host;
    uint16_t    port = 0;
};

struct segment {
    uint32_t    id = 0;
    int32_t     layer_first = -1;
    int32_t     layer_last = -1;
    endpoint    target;
    std::string stage_gguf;
    // devices this segment runs on. Populated from either the scalar "device"
    // member (single device, v1 manifests) or the "devices" array (tensor-
    // parallel stages). Always non-empty after parsing.
    std::vector<std::string> devices;
    // "" (single device / layer semantics) or "tensor" (split every layer
    // across `devices` with per-layer AllReduce); requires >= 2 devices.
    std::string split_mode;
    // optional per-device split proportions; empty or devices.size() entries
    std::vector<float> tensor_split;
    // INTERIOR TAPS. Layers whose input hidden state this segment must extract and
    // ship back to the head alongside the forward response, in ascending order and
    // always inside [layer_first, layer_last].
    //
    // A DFlash/DSpark speculative draft conditions on the target's hidden states at
    // fixed layers (the draft GGUF's dflash.target_layers). When the target is split
    // across machines, a tap can land on a segment the head does not own, and the head
    // has no way to compute it. Declaring taps HERE rather than deriving them from the
    // draft is deliberate: both the head and the worker already parse this manifest, so
    // the worker can make the extraction a LOAD-TIME graph decision -- the same shape
    // as terminal_kind -- and HELLO only has to verify the two sides agree. Deriving
    // them from the draft is not possible on the head at the right time anyway: the
    // segment client performs its HELLO while constructing, long before the draft model
    // (and therefore its target_layers metadata) has been loaded.
    std::vector<uint32_t> tap_layers;
    std::string artifact_sha256;

    const std::string & device() const { return devices.front(); }
};

struct manifest {
    std::string          model_identity_sha256;
    int32_t              n_layer = 0;
    int32_t              n_embd = 0;
    std::string          wire_precision;
    std::vector<segment> segments;
    std::string          checksum;
};

// Return the v1 checksum for a JSON manifest. The root checksum member is
// excluded and the remaining JSON is serialized in nlohmann's stable order.
std::string manifest_checksum(const std::string & json_text);

// Parse and validate a dense-segment manifest. `expected_model_identity_sha256`
// is normally the local full-model identity; when supplied it must match the
// manifest's identity exactly.
manifest parse_manifest(const std::string & json_text,
                        const std::string & expected_model_identity_sha256 = {});

manifest load_manifest(const std::string & path,
                       const std::string & expected_model_identity_sha256 = {});

} // namespace pipe_dense_segment
