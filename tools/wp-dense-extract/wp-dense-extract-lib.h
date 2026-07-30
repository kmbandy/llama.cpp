#pragma once

#include <cstdint>
#include <string>

namespace wp_dense_extract {

constexpr const char * ROUTED_EXPERTS_EXTERNAL_KEY = "weight_pager.routed_experts_external";

struct result {
    int64_t  tensor_count        = 0;
    uint64_t tensor_bytes        = 0;
    int64_t  routed_tensor_count = 0;
    uint64_t routed_tensor_bytes = 0;
    uint64_t file_bytes          = 0;
    bool     verified            = false;
};

result extract(const std::string & model_path, const std::string & output_path, bool verify);

}  // namespace wp_dense_extract
