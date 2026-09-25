#pragma once

// Persistent worker-thread pool for the spine-side MoE plan/encode path
// (2026-09-25). WP_EXPERT_PLAN_THREADS / WP_EXPERT_ENCODE_THREADS parallelize
// routing and wire-encoding across CPU cores; a layer's issue phase runs
// every ~200 ms (40 layers x up to 3 ubatches per prefill), so spawning
// std::thread per call is real overhead at that cadence -- this pool creates
// its threads once (lazily, on first use) and reuses them for every
// parallel_for() call for the lifetime of the process.
//
// Design is deliberately small: one pool, one job at a time (parallel_for()
// blocks the calling thread until all workers finish that job), no queue.
// The spine dispatch thread never issues two parallel_for() calls
// concurrently against the same pool, so this is the simplest shape that
// removes spawn/join cost without adding a task queue nobody needs yet.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pipe_thread_pool_detail {

class pool {
public:
    // Lazily-constructed process-wide singleton. `n_threads` sizes the pool
    // on first call only; later calls with a different value are ignored --
    // callers that want a specific worker count pass it every time and the
    // pool is sized once from whichever call happens first (in practice the
    // knob value is fixed for the process lifetime, so this never matters in
    // production; the ctor-races-on-N case only shows up in tests that flip
    // thread counts, which is why the byte-identity test calls parallel_for
    // directly with each explicit thread count rather than relying on this
    // singleton's size to vary).
    static pool & instance(size_t n_threads) {
        static pool p(n_threads);
        return p;
    }

    ~pool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_worker_.notify_all();
        for (std::thread & t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    // Runs fn(begin, end) for `n_workers` contiguous, non-overlapping ranges
    // covering [0, n). Blocks until every range has completed. `n_workers` is
    // clamped to [1, threads actually available in this pool]; n_workers<=1
    // (or n too small to split -- caller's job) should be handled by the
    // caller taking the plain non-threaded path instead, but calling this
    // with n_workers==1 is also safe (runs fn(0, n) inline, no thread hop).
    void parallel_for(size_t n, size_t n_workers, const std::function<void(size_t, size_t)> & fn) {
        if (n == 0) {
            return;
        }
        n_workers = std::max<size_t>(1, std::min(n_workers, workers_.size() + 1));
        if (n_workers <= 1) {
            fn(0, n);
            return;
        }
        const size_t per = (n + n_workers - 1) / n_workers;
        const size_t n_jobs = (n + per - 1) / per; // <= n_workers
        // Job 0 runs on the CALLING thread (no hop for the first slice); jobs
        // 1..n_jobs-1 go to pool workers. This means a pool sized for
        // (hardware_concurrency - 1) background threads plus the caller
        // still uses all cores.
        pending_.store(n_jobs - 1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t j = 1; j < n_jobs; ++j) {
                const size_t b0 = j * per;
                const size_t b1 = std::min(n, b0 + per);
                job_queue_.push_back([&fn, b0, b1]() { fn(b0, b1); });
            }
        }
        cv_worker_.notify_all();
        fn(0, std::min(n, per));
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this]() { return pending_.load(std::memory_order_relaxed) == 0; });
        }
    }

private:
    explicit pool(size_t n_threads) {
        // n_threads counts TOTAL desired concurrency including the caller's
        // own thread (see parallel_for's job 0), so this pool spawns
        // n_threads-1 background workers.
        const size_t n_background = n_threads > 1 ? n_threads - 1 : 0;
        workers_.reserve(n_background);
        for (size_t i = 0; i < n_background; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    void worker_loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_worker_.wait(lock, [this]() { return stop_ || !job_queue_.empty(); });
                if (stop_ && job_queue_.empty()) {
                    return;
                }
                if (job_queue_.empty()) {
                    continue;
                }
                job = std::move(job_queue_.front());
                job_queue_.pop_front();
            }
            job();
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                cv_done_.notify_all();
            }
        }
    }

    std::vector<std::thread>            workers_;
    std::mutex                          mutex_;
    std::condition_variable             cv_worker_;
    std::condition_variable             cv_done_;
    std::deque<std::function<void()>>   job_queue_;
    std::atomic<size_t>                 pending_{0};
    bool                                 stop_ = false;
};

}  // namespace pipe_thread_pool_detail

// Runs fn(begin, end) across `n_threads` roughly-equal chunks of [0, n),
// using a process-wide persistent pool sized to n_threads on first call.
// n_threads<=1 runs fn(0, n) directly with no thread involvement at all
// (identical code path to calling fn once, so the n_threads<=1 case is
// exactly today's single-threaded behaviour byte-for-byte).
inline void pipe_parallel_for(size_t n, size_t n_threads, const std::function<void(size_t, size_t)> & fn) {
    if (n_threads <= 1 || n == 0) {
        if (n > 0) fn(0, n);
        return;
    }
    pipe_thread_pool_detail::pool::instance(n_threads).parallel_for(n, n_threads, fn);
}
