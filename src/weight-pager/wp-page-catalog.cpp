#include "wp-page-catalog.h"

namespace wp {

int PageCatalog::add(const std::string & name, uint16_t file_idx,
                     uint64_t file_offset, size_t size) {
    const int idx = (int) pages_.size();
    pages_.push_back({ name, file_idx, file_offset, size });
    name_to_idx_.emplace(name, idx);
    if (size > max_size_) {
        max_size_ = size;
    }
    return idx;
}

int PageCatalog::find(const std::string & name) const {
    const auto it = name_to_idx_.find(name);
    return it == name_to_idx_.end() ? -1 : it->second;
}

void PageCatalog::clear() {
    pages_.clear();
    name_to_idx_.clear();
    max_size_ = 0;
}

}  // namespace wp
