#include "weight-pager/wp-partition.h"

#include <stdexcept>

namespace wp {

PartitionedPages partition_pages_by_device(const std::vector<PagePartitionInput> & pages) {
    PartitionedPages out;
    std::unordered_map<const void *, size_t> device_to_partition;

    for (size_t source_idx = 0; source_idx < pages.size(); ++source_idx) {
        const PagePartitionInput & page = pages[source_idx];
        if (!page.paged) {
            continue;
        }
        if (page.device == nullptr) {
            throw std::invalid_argument("paged page has no device: " + page.name);
        }

        auto [dev_it, inserted] =
            device_to_partition.emplace(page.device, out.partitions.size());
        if (inserted) {
            PagePartition partition;
            partition.device = page.device;
            out.partitions.push_back(std::move(partition));
        }

        PagePartition & partition = out.partitions[dev_it->second];
        const int local_page_idx = (int) partition.source_indices.size();
        partition.source_indices.push_back(source_idx);

        auto route = out.routes.emplace(
            page.name, PagePartitionRoute{dev_it->second, local_page_idx});
        if (!route.second) {
            throw std::invalid_argument("duplicate paged page name: " + page.name);
        }
        ++out.n_paged;
    }

    return out;
}

} // namespace wp
