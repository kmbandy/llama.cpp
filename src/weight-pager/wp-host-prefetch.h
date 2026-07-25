#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace wp {

class HostPrefetcher {
public:
    using ReadCallback = std::function<int64_t(int page_idx, void * dst, size_t capacity)>;
    using StoreCallback = std::function<bool(int page_idx, const void * bytes, size_t n)>;
    using SkipPredicate = std::function<bool(int page_idx)>;

    HostPrefetcher(ReadCallback read, StoreCallback store, SkipPredicate should_skip,
                   size_t max_queue_depth, size_t max_page_size);
    ~HostPrefetcher();

    HostPrefetcher(const HostPrefetcher &) = delete;
    HostPrefetcher & operator=(const HostPrefetcher &) = delete;

    void enqueue(int page_idx);
    void start();
    void stop();

    uint64_t enqueued() const { return enqueued_.load(std::memory_order_relaxed); }
    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
    uint64_t read_ok() const { return read_ok_.load(std::memory_order_relaxed); }
    uint64_t read_fail() const { return read_fail_.load(std::memory_order_relaxed); }
    uint64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }

private:
    void run_();

    ReadCallback read_;
    StoreCallback store_;
    SkipPredicate should_skip_;
    size_t max_queue_depth_ = 0;
    size_t max_page_size_ = 0;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<int> queue_;
    std::thread worker_;
    bool running_ = false;
    bool stopping_ = false;

    std::atomic<uint64_t> enqueued_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint64_t> read_ok_{0};
    std::atomic<uint64_t> read_fail_{0};
    std::atomic<uint64_t> skipped_{0};
};

}  // namespace wp
