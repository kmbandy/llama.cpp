#include "mt-block-pool.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cassert>
#include <cmath>

namespace mt {

void BlockPool::init(uint32_t n_gpu, uint32_t n_cpu, float watermark) {
    total_gpu_blocks_ = n_gpu;
    total_cpu_blocks_ = n_cpu;

    // Watermark is a fraction; round up so a small reserve never floors
    // to 0 (which would defeat the purpose). vLLM uses ceil() too.
    watermark_gpu_ = (uint32_t) std::ceil((double) n_gpu * (double) watermark);
    watermark_cpu_ = (uint32_t) std::ceil((double) n_cpu * (double) watermark);

    // Initialize free stacks. Push descending so pop() returns block 0
    // first — predictable in tests, no behavior cost.
    gpu_free_.clear();
    gpu_free_.reserve(n_gpu);
    for (uint32_t i = n_gpu; i-- > 0; ) {
        gpu_free_.push_back(i);
    }

    cpu_free_.clear();
    cpu_free_.reserve(n_cpu);
    for (uint32_t i = n_cpu; i-- > 0; ) {
        // CPU IDs live above the GPU range so is_gpu() is a single
        // comparison.
        cpu_free_.push_back(n_gpu + i);
    }

    // MAD-128: per-block refcount, all start at 0 (free).
    refcount_.assign((size_t) n_gpu + n_cpu, 0);

    LLAMA_LOG_INFO("mt::BlockPool: init n_gpu=%u n_cpu=%u watermark=%.2f "
                   "(reserve gpu=%u cpu=%u)\n",
                   n_gpu, n_cpu, (double) watermark,
                   watermark_gpu_, watermark_cpu_);
}

uint32_t BlockPool::alloc_gpu() {
    if (gpu_free_.empty()) return kInvalidBlockId;
    const uint32_t id = gpu_free_.back();
    gpu_free_.pop_back();
    assert(id < refcount_.size() && refcount_[id] == 0 && "alloc'd block had nonzero refcount");
    refcount_[id] = 1;
    return id;
}

uint32_t BlockPool::alloc_cpu() {
    if (cpu_free_.empty()) return kInvalidBlockId;
    const uint32_t id = cpu_free_.back();
    cpu_free_.pop_back();
    assert(id < refcount_.size() && refcount_[id] == 0 && "alloc'd block had nonzero refcount");
    refcount_[id] = 1;
    return id;
}

void BlockPool::bump_ref(uint32_t block_id) {
    if (block_id == kInvalidBlockId) {
        LLAMA_LOG_WARN("mt::BlockPool::bump_ref: kInvalidBlockId — ignoring\n");
        return;
    }
    assert(block_id < refcount_.size() && "bump_ref: block_id out of range");
    assert(refcount_[block_id] > 0 && "bump_ref: block is free (refcount==0)");
    ++refcount_[block_id];
}

uint32_t BlockPool::refcount(uint32_t block_id) const {
    if (block_id == kInvalidBlockId) return 0;
    if (block_id >= refcount_.size()) return 0;
    return refcount_[block_id];
}

void BlockPool::free_block(uint32_t block_id) {
    if (block_id == kInvalidBlockId) {
        LLAMA_LOG_WARN("mt::BlockPool::free_block: kInvalidBlockId — ignoring\n");
        return;
    }

    assert(block_id < refcount_.size() && "free_block: block_id out of range");
    if (refcount_[block_id] == 0) {
        LLAMA_LOG_WARN("mt::BlockPool::free_block: double-free of block %u\n", block_id);
        return;
    }
    --refcount_[block_id];
    if (refcount_[block_id] > 0) {
        // Still has other references — don't return to free stack yet.
        return;
    }

    if (is_gpu(block_id)) {
        assert(block_id < total_gpu_blocks_);
        gpu_free_.push_back(block_id);
    } else {
        assert(block_id < total_gpu_blocks_ + total_cpu_blocks_);
        cpu_free_.push_back(block_id);
    }
}

bool BlockPool::is_gpu(uint32_t block_id) const {
    return block_id < total_gpu_blocks_;
}

size_t BlockPool::n_free_gpu() const {
    return gpu_free_.size();
}

size_t BlockPool::n_free_cpu() const {
    return cpu_free_.size();
}

bool BlockPool::has_free_gpu_blocks(uint32_t n) const {
    const size_t avail = gpu_free_.size();
    if (avail <= watermark_gpu_) return false;
    return n <= (avail - watermark_gpu_);
}

bool BlockPool::has_free_cpu_blocks(uint32_t n) const {
    const size_t avail = cpu_free_.size();
    if (avail <= watermark_cpu_) return false;
    return n <= (avail - watermark_cpu_);
}

void BlockPool::reset() {
    gpu_free_.clear();
    gpu_free_.reserve(total_gpu_blocks_);
    for (uint32_t i = total_gpu_blocks_; i-- > 0; ) {
        gpu_free_.push_back(i);
    }
    cpu_free_.clear();
    cpu_free_.reserve(total_cpu_blocks_);
    for (uint32_t i = total_cpu_blocks_; i-- > 0; ) {
        cpu_free_.push_back(total_gpu_blocks_ + i);
    }
    // MAD-128: zero all refcounts on whole-pool reset.
    std::fill(refcount_.begin(), refcount_.end(), 0u);
}

}  // namespace mt
