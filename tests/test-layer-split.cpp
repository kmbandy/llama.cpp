#include "common.h"

#include <algorithm>
#include <array>
#undef NDEBUG
#include <cassert>

int main() {
    common_params params;
    params.n_gpu_layers = 99;
    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    params.tensor_split[0] = 6.5f;
    params.tensor_split[1] = 6.5f;
    params.tensor_split[2] = 28.5f;
    params.tensor_split[3] = 13.0f;

    const llama_model_params mparams = common_model_params_to_llama(params);
    assert(mparams.pipeline_layer_first == -1);
    assert(mparams.pipeline_layer_last == -1);
    assert(mparams.tensor_split[0] == 6.5f);
    assert(mparams.tensor_split[1] == 6.5f);
    assert(mparams.tensor_split[2] == 28.5f);
    assert(mparams.tensor_split[3] == 13.0f);

    constexpr int n_layer_all = 65;
    const int act_gpu_layers = std::min(mparams.n_gpu_layers, n_layer_all + 1);
    auto count_layers = [act_gpu_layers](std::array<float, 4> splits) {
        float split_sum = 0.0f;
        for (float & split : splits) {
            split_sum += split;
            split = split_sum;
        }
        for (float & split : splits) {
            split /= split_sum;
        }

        std::array<int, 4> layer_counts = {};
        for (int il = 0; il < n_layer_all; ++il) {
            const size_t device = std::upper_bound(
                splits.begin(), splits.end(), (float) il/act_gpu_layers) - splits.begin();
            ++layer_counts[device];
        }
        return layer_counts;
    };

    const std::array<float, 4> splits = {
        mparams.tensor_split[0], mparams.tensor_split[1],
        mparams.tensor_split[2], mparams.tensor_split[3],
    };
    const std::array<int, 4> layer_counts = count_layers(splits);
    assert((layer_counts == std::array<int, 4> { 8, 8, 35, 14 }));

    const std::array<int, 4> reversed_counts = count_layers({ 6.5f, 6.5f, 13.0f, 28.5f });
    assert((reversed_counts == std::array<int, 4> { 8, 8, 16, 33 }));

    std::array<float, 4> split_points = splits;
    float split_sum = 0.0f;
    for (float & split : split_points) {
        split_sum += split;
        split = split_sum;
    }
    for (float & split : split_points) {
        split /= split_sum;
    }
    const size_t output_device = std::upper_bound(
        split_points.begin(), split_points.end(), (float) n_layer_all/act_gpu_layers) - split_points.begin();
    assert(output_device == 3);

    return 0;
}
