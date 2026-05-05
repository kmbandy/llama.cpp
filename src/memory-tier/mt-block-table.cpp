#include "mt-block-table.h"

#include "mt-block-pool.h"  // kInvalidBlockId
#include "llama-impl.h"     // LLAMA_LOG_*

#include <cassert>

namespace mt {

void BlockTable::init(uint32_t max_seqs, uint32_t block_size) {
    max_seqs_   = max_seqs;
    block_size_ = block_size;
    table_.clear();
    table_.resize(max_seqs);
    LLAMA_LOG_INFO("mt::BlockTable: init max_seqs=%u block_size=%u\n",
                   max_seqs, block_size);
}

void BlockTable::append_block(llama_seq_id seq, uint32_t physical_block_id) {
    assert(seq >= 0 && (uint32_t) seq < max_seqs_ && "seq out of range");
    table_[seq].push_back(physical_block_id);
}

uint32_t BlockTable::swap_block(llama_seq_id seq, uint32_t logical_idx,
                                uint32_t new_physical_block_id) {
    assert(seq >= 0 && (uint32_t) seq < max_seqs_ && "seq out of range");
    auto & row = table_[seq];
    assert(logical_idx < row.size() && "logical_idx out of range");
    const uint32_t old = row[logical_idx];
    row[logical_idx] = new_physical_block_id;
    return old;
}

uint32_t BlockTable::get_physical(llama_seq_id seq, uint32_t logical_idx) const {
    if (seq < 0 || (uint32_t) seq >= max_seqs_) return kInvalidBlockId;
    const auto & row = table_[seq];
    if (logical_idx >= row.size()) return kInvalidBlockId;
    return row[logical_idx];
}

uint32_t BlockTable::get_physical_for_pos(llama_seq_id seq, llama_pos pos) const {
    if (pos < 0 || block_size_ == 0) return kInvalidBlockId;
    return get_physical(seq, (uint32_t) pos / block_size_);
}

uint32_t BlockTable::num_blocks(llama_seq_id seq) const {
    if (seq < 0 || (uint32_t) seq >= max_seqs_) return 0;
    return (uint32_t) table_[seq].size();
}

std::vector<uint32_t> BlockTable::clear_seq(llama_seq_id seq) {
    if (seq < 0 || (uint32_t) seq >= max_seqs_) return {};
    std::vector<uint32_t> freed = std::move(table_[seq]);
    table_[seq].clear();  // post-move state is unspecified; explicit clear
    return freed;
}

void BlockTable::reset() {
    for (auto & row : table_) {
        row.clear();
    }
}

}  // namespace mt
