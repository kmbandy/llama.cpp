#include "pipe-expert-shm.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>

#if defined(__linux__)
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace {

constexpr uint32_t SHM_MAGIC = 0x31504D53u; // "SMP1"
constexpr uint32_t SHM_VERSION = 1;
constexpr uint32_t RECORD_WRAP = 0;
constexpr size_t RECORD_HEADER_SIZE = 8;
constexpr size_t RECORD_ALIGNMENT = 8;
constexpr size_t HEADER_ALIGNMENT = 4096;

size_t align_up(size_t value, size_t alignment) {
    if (value > SIZE_MAX - alignment + 1) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

size_t record_span(size_t length) {
    if (length > SIZE_MAX - RECORD_HEADER_SIZE) {
        return 0;
    }
    return align_up(RECORD_HEADER_SIZE + length, RECORD_ALIGNMENT);
}

} // namespace

// A nested class member cannot be defined inside an unnamed namespace.
#if defined(__linux__)
struct alignas(HEADER_ALIGNMENT) pipe_expert_shm_ring::Header {
    uint32_t magic = SHM_MAGIC;
    uint32_t version = SHM_VERSION;
    uint64_t mapping_size = 0;
    uint64_t request_offset = 0;
    uint64_t request_capacity = 0;
    uint64_t response_offset = 0;
    uint64_t response_capacity = 0;
    std::atomic<uint64_t> request_head{0};
    std::atomic<uint64_t> request_tail{0};
    std::atomic<uint64_t> response_head{0};
    std::atomic<uint64_t> response_tail{0};
    std::atomic<uint32_t> error{0};
};

static_assert(sizeof(pipe_expert_shm_ring::Header) <= HEADER_ALIGNMENT,
              "shm header must fit in the reserved header page");
#endif

namespace {

#if defined(__linux__)
char g_cleanup_name[256] = {};

void shm_cleanup_signal(int) {
    if (g_cleanup_name[0] != '\0') {
        shm_unlink(g_cleanup_name);
    }
    _exit(128);
}

void install_cleanup_signal(const std::string & name) {
    if (name.size() >= sizeof(g_cleanup_name)) {
        return;
    }
    std::memcpy(g_cleanup_name, name.c_str(), name.size() + 1);
    ::signal(SIGINT, shm_cleanup_signal);
    ::signal(SIGTERM, shm_cleanup_signal);
}
#endif

} // namespace

pipe_expert_shm_ring::~pipe_expert_shm_ring() {
#if defined(__linux__)
    if (mapping_ != nullptr && mapping_ != MAP_FAILED) {
        munmap(mapping_, mapping_size_);
    }
    if (owner_ && !name_.empty()) {
        shm_unlink(name_.c_str());
    }
#endif
}

std::unique_ptr<pipe_expert_shm_ring> pipe_expert_shm_ring::create(
        const std::string & name, uint32_t n_tokens, uint32_t n_embd,
        uint32_t n_assignments) {
    if (n_tokens == 0 || n_embd == 0 || n_assignments == 0) {
        return nullptr;
    }
    const uint64_t activation_bytes = (uint64_t) n_tokens * n_embd * sizeof(float);
    const uint64_t assignment_bytes = (uint64_t) n_assignments *
                                      (sizeof(int32_t) + (uint64_t) n_tokens * sizeof(float));
    const uint64_t request_bytes = 16ull + assignment_bytes + activation_bytes;
    const uint64_t response_bytes = 16ull + activation_bytes;
    if (request_bytes > SIZE_MAX / 2 || response_bytes > SIZE_MAX / 2) {
        return nullptr;
    }
    return create_for_test(name, align_up((size_t) request_bytes * 2, RECORD_ALIGNMENT),
                           align_up((size_t) response_bytes * 2, RECORD_ALIGNMENT));
}

std::unique_ptr<pipe_expert_shm_ring> pipe_expert_shm_ring::create_for_test(
        const std::string & name, size_t request_capacity, size_t response_capacity) {
#if defined(__linux__)
    if (name.empty() || name[0] != '/' || request_capacity < RECORD_HEADER_SIZE ||
        response_capacity < RECORD_HEADER_SIZE) {
        return nullptr;
    }
    request_capacity = align_up(request_capacity, RECORD_ALIGNMENT);
    response_capacity = align_up(response_capacity, RECORD_ALIGNMENT);
    if (request_capacity == 0 || response_capacity == 0 ||
        request_capacity > UINT32_MAX || response_capacity > UINT32_MAX) {
        return nullptr;
    }
    if (response_capacity > SIZE_MAX - HEADER_ALIGNMENT ||
        request_capacity > SIZE_MAX - HEADER_ALIGNMENT - response_capacity) {
        return nullptr;
    }
    auto result = std::unique_ptr<pipe_expert_shm_ring>(new pipe_expert_shm_ring);
    const size_t total = HEADER_ALIGNMENT + request_capacity + response_capacity;
    if (!result->open_region(name, true, total, request_capacity, response_capacity)) {
        return nullptr;
    }
    install_cleanup_signal(name);
    return result;
#else
    (void) name;
    (void) request_capacity;
    (void) response_capacity;
    return nullptr;
#endif
}

std::unique_ptr<pipe_expert_shm_ring> pipe_expert_shm_ring::attach(const std::string & name) {
#if defined(__linux__)
    if (name.empty() || name[0] != '/') {
        return nullptr;
    }
    auto result = std::unique_ptr<pipe_expert_shm_ring>(new pipe_expert_shm_ring);
    if (!result->open_region(name, false, 0, 0, 0)) {
        return nullptr;
    }
    return result;
#else
    (void) name;
    return nullptr;
#endif
}

bool pipe_expert_shm_ring::open_region(const std::string & name, bool owner,
                                       size_t total_size, size_t request_capacity,
                                       size_t response_capacity) {
#if defined(__linux__)
    int fd = -1;
    if (owner) {
        fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0 && errno == EEXIST) {
            shm_unlink(name.c_str());
            fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        }
    } else {
        fd = shm_open(name.c_str(), O_RDWR, 0600);
    }
    if (fd < 0) {
        fail(std::strerror(errno));
        return false;
    }
    if (owner && ftruncate(fd, (off_t) total_size) != 0) {
        fail(std::strerror(errno));
        close(fd);
        shm_unlink(name.c_str());
        return false;
    }
    if (!owner) {
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < (off_t) HEADER_ALIGNMENT) {
            fail("shm region is too small");
            close(fd);
            return false;
        }
        total_size = (size_t) st.st_size;
    }
    void * mapped = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fail(std::strerror(errno));
        if (owner) {
            shm_unlink(name.c_str());
        }
        return false;
    }
    mapping_ = mapped;
    mapping_size_ = total_size;
    owner_ = owner;
    name_ = name;
    header_ = reinterpret_cast<Header *>(mapped);
    if (owner) {
        header_->magic = SHM_MAGIC;
        header_->version = SHM_VERSION;
        header_->mapping_size = total_size;
        header_->request_offset = HEADER_ALIGNMENT;
        header_->request_capacity = request_capacity;
        header_->response_offset = HEADER_ALIGNMENT + request_capacity;
        header_->response_capacity = response_capacity;
        header_->request_head.store(0, std::memory_order_relaxed);
        header_->request_tail.store(0, std::memory_order_relaxed);
        header_->response_head.store(0, std::memory_order_relaxed);
        header_->response_tail.store(0, std::memory_order_relaxed);
        header_->error.store(0, std::memory_order_relaxed);
    } else if (header_->magic != SHM_MAGIC || header_->version != SHM_VERSION ||
               header_->mapping_size != total_size ||
               header_->request_offset < HEADER_ALIGNMENT ||
               header_->request_capacity < RECORD_HEADER_SIZE ||
               header_->response_capacity < RECORD_HEADER_SIZE ||
               header_->response_offset <= header_->request_offset ||
               header_->response_offset + header_->response_capacity > total_size ||
               header_->request_offset + header_->request_capacity > total_size) {
        fail("shm region header is invalid");
        munmap(mapping_, mapping_size_);
        mapping_ = nullptr;
        header_ = nullptr;
        return false;
    }
    return true;
#else
    (void) name;
    (void) owner;
    (void) total_size;
    (void) request_capacity;
    (void) response_capacity;
    return false;
#endif
}

bool pipe_expert_shm_ring::valid() const {
    return header_ != nullptr;
}

const std::string & pipe_expert_shm_ring::name() const {
    return name_;
}

void pipe_expert_shm_ring::fail(const char * message) {
    std::snprintf(error_, sizeof(error_), "%s", message != nullptr ? message : "shm error");
#if defined(__linux__)
    if (header_ != nullptr) {
        header_->error.store(1, std::memory_order_release);
    }
#endif
}

const char * pipe_expert_shm_ring::error() const {
    return error_[0] != '\0' ? error_ : "shm ring error";
}

bool pipe_expert_shm_ring::write_request(const void * data, size_t length,
                                         pipe_expert_shm_ref & ref) {
    return write(false, data, length, ref);
}

bool pipe_expert_shm_ring::write_response(const void * data, size_t length,
                                          pipe_expert_shm_ref & ref) {
    return write(true, data, length, ref);
}

bool pipe_expert_shm_ring::read_request(const pipe_expert_shm_ref & ref, const uint8_t ** data) {
    return read(false, ref, data);
}

bool pipe_expert_shm_ring::read_response(const pipe_expert_shm_ref & ref, const uint8_t ** data) {
    return read(true, ref, data);
}

bool pipe_expert_shm_ring::consume_request(const pipe_expert_shm_ref & ref) {
    return consume(false, ref);
}

bool pipe_expert_shm_ring::consume_response(const pipe_expert_shm_ref & ref) {
    return consume(true, ref);
}

bool pipe_expert_shm_ring::write(bool response, const void * data, size_t length,
                                 pipe_expert_shm_ref & ref) {
#if defined(__linux__)
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!valid() || data == nullptr || length == 0 || length > UINT32_MAX) {
        fail("invalid shm write");
        return false;
    }
    if (header_->error.load(std::memory_order_acquire) != 0) {
        fail("shm ring error is latched");
        return false;
    }
    std::atomic<uint64_t> & head = response ? header_->response_head : header_->request_head;
    const std::atomic<uint64_t> & tail = response ? header_->response_tail : header_->request_tail;
    const uint64_t base = response ? header_->response_offset : header_->request_offset;
    const uint64_t capacity = response ? header_->response_capacity : header_->request_capacity;
    const size_t span = record_span(length);
    uint64_t current = head.load(std::memory_order_relaxed);
    const uint64_t consumed = tail.load(std::memory_order_acquire);
    if (span == 0 || current < consumed || current - consumed > capacity ||
        span > capacity - (current - consumed)) {
        fail("shm ring is full");
        return false;
    }
    size_t position = (size_t) (current % capacity);
    const size_t remaining = (size_t) capacity - position;
    if (remaining < span) {
        if (remaining >= RECORD_HEADER_SIZE) {
            uint32_t * marker = reinterpret_cast<uint32_t *>
                (reinterpret_cast<uint8_t *>(mapping_) + base + position);
            marker[0] = RECORD_WRAP;
            marker[1] = (uint32_t) remaining;
        }
        current += remaining;
        position = 0;
        if (span > capacity - (current - consumed)) {
            fail("shm ring wrap has no space");
            return false;
        }
    }
    uint8_t * slot = reinterpret_cast<uint8_t *>(mapping_) + base + position;
    uint32_t * record = reinterpret_cast<uint32_t *>(slot);
    record[0] = (uint32_t) length;
    record[1] = (uint32_t) span;
    std::memcpy(slot + RECORD_HEADER_SIZE, data, length);
    const uint64_t committed = current + span;
    head.store(committed, std::memory_order_release);
    ref.offset = base + position + RECORD_HEADER_SIZE;
    ref.length = length;
    return true;
#else
    (void) response;
    (void) data;
    (void) length;
    (void) ref;
    fail("shared memory is not supported on this platform");
    return false;
#endif
}

bool pipe_expert_shm_ring::read(bool response, const pipe_expert_shm_ref & ref,
                                const uint8_t ** data) {
#if defined(__linux__)
    if (!valid() || data == nullptr) {
        fail("invalid shm read");
        return false;
    }
    if (header_->error.load(std::memory_order_acquire) != 0) {
        fail("shm ring error is latched");
        return false;
    }
    const std::atomic<uint64_t> & head = response ? header_->response_head : header_->request_head;
    std::atomic<uint64_t> & tail = response ? header_->response_tail : header_->request_tail;
    const uint64_t base = response ? header_->response_offset : header_->request_offset;
    const uint64_t capacity = response ? header_->response_capacity : header_->request_capacity;
    uint64_t current = tail.load(std::memory_order_relaxed);
    const uint64_t available = head.load(std::memory_order_acquire);
    if (current == available || current > available || available - current > capacity) {
        fail("shm ring has no committed record");
        return false;
    }
    for (;;) {
        const size_t position = (size_t) (current % capacity);
        const uint8_t * slot = reinterpret_cast<const uint8_t *>(mapping_) + base + position;
        const uint32_t length = reinterpret_cast<const uint32_t *>(slot)[0];
        const uint32_t span = reinterpret_cast<const uint32_t *>(slot)[1];
        if (length == RECORD_WRAP) {
            if (span < RECORD_HEADER_SIZE || span > capacity - position) {
                fail("invalid shm wrap record");
                return false;
            }
            current += span;
            tail.store(current, std::memory_order_release);
            if (current == available) {
                fail("shm ring wrap has no record");
                return false;
            }
            continue;
        }
        if (span != record_span(length) || span > capacity - position ||
            current + span > available || ref.offset != base + position + RECORD_HEADER_SIZE ||
            ref.length != length) {
            fail("shm record does not match control frame");
            return false;
        }
        *data = slot + RECORD_HEADER_SIZE;
        return true;
    }
#else
    (void) response;
    (void) ref;
    (void) data;
    fail("shared memory is not supported on this platform");
    return false;
#endif
}

bool pipe_expert_shm_ring::consume(bool response, const pipe_expert_shm_ref & ref) {
#if defined(__linux__)
    if (!valid()) {
        fail("invalid shm consume");
        return false;
    }
    if (header_->error.load(std::memory_order_acquire) != 0) {
        fail("shm ring error is latched");
        return false;
    }
    const uint64_t base = response ? header_->response_offset : header_->request_offset;
    const uint64_t capacity = response ? header_->response_capacity : header_->request_capacity;
    std::atomic<uint64_t> & tail = response ? header_->response_tail : header_->request_tail;
    const uint64_t current = tail.load(std::memory_order_relaxed);
    const size_t position = (size_t) (current % capacity);
    const uint8_t * slot = reinterpret_cast<const uint8_t *>(mapping_) + base + position;
    const uint32_t length = reinterpret_cast<const uint32_t *>(slot)[0];
    const uint32_t span = reinterpret_cast<const uint32_t *>(slot)[1];
    if (length == RECORD_WRAP || span != record_span(length) ||
        ref.offset != base + position + RECORD_HEADER_SIZE || ref.length != length) {
        fail("invalid shm consume");
        return false;
    }
    tail.store(current + span, std::memory_order_release);
    return true;
#else
    (void) response;
    (void) ref;
    fail("shared memory is not supported on this platform");
    return false;
#endif
}
