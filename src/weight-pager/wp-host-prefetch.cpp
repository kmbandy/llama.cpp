#include "wp-host-prefetch.h"

#include <utility>

namespace wp {

HostPrefetcher::HostPrefetcher(ReadCallback read, StoreCallback store,
                               SkipPredicate should_skip, size_t max_queue_depth,
                               size_t max_page_size)
    : read_(std::move(read)),
      store_(std::move(store)),
      should_skip_(std::move(should_skip)),
      max_queue_depth_(max_queue_depth),
      max_page_size_(max_page_size) {
}

HostPrefetcher::~HostPrefetcher() {
    stop();
}

void HostPrefetcher::enqueue(int page_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_ || max_queue_depth_ == 0 || queue_.size() >= max_queue_depth_) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    queue_.push_back(page_idx);
    enqueued_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_one();
}

void HostPrefetcher::start() {
    std::lock_guard<std::mutex> lock(mu_);
    if (running_ || stopping_) {
        return;
    }
    stopping_ = false;
    running_ = true;
    worker_ = std::thread(&HostPrefetcher::run_, this);
}

void HostPrefetcher::stop() {
    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!running_) {
            return;
        }
        stopping_ = true;
        running_ = false;
        worker = std::move(worker_);
    }
    cv_.notify_one();
    if (worker.joinable()) {
        worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = false;
    }
}

void HostPrefetcher::run_() {
    std::vector<uint8_t> buffer(max_page_size_);
    for (;;) {
        int page_idx = -1;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;
            }
            page_idx = queue_.front();
            queue_.pop_front();
        }

        if (should_skip_ && should_skip_(page_idx)) {
            skipped_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!read_ || !store_ || buffer.empty()) {
            read_fail_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const int64_t n = read_(page_idx, buffer.data(), buffer.size());
        if (n < 0 || (uint64_t) n > buffer.size()) {
            read_fail_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        read_ok_.fetch_add(1, std::memory_order_relaxed);
        store_(page_idx, buffer.data(), (size_t) n);
    }
}

}  // namespace wp
