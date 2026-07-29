#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace wp {

struct PagePartitionInput {
    std::string  name;
    const void * device = nullptr;
    bool         paged  = true;
};

struct PagePartition {
    const void *        device = nullptr;
    std::vector<size_t> source_indices;
};

struct PagePartitionRoute {
    size_t partition_idx = 0;
    int    page_idx      = -1;
};

struct PartitionedPages {
    std::vector<PagePartition>                         partitions;
    std::unordered_map<std::string, PagePartitionRoute> routes;
    size_t                                             n_paged = 0;
};

// Stable partition by device. Pages retain source order within each
// partition, and partitions retain first-device-appearance order.
// Non-paged inputs are intentionally absent from both partitions and routes.
// Throws std::invalid_argument for a paged page with no device or a duplicate
// paged name.
PartitionedPages partition_pages_by_device(const std::vector<PagePartitionInput> & pages);

} // namespace wp
