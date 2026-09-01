#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

// One shared region has independent single-producer/single-consumer byte rings
// for requests and responses. TCP carries only the control frame that names a
// committed record. The record bytes are otherwise the existing expert frame
// payload bytes, including their f32 arrays.
struct pipe_expert_shm_ref {
    uint64_t offset = 0;
    uint64_t length = 0;
};

class pipe_expert_shm_ring {
  public:
    struct Header;

    ~pipe_expert_shm_ring();

    pipe_expert_shm_ring(const pipe_expert_shm_ring &) = delete;
    pipe_expert_shm_ring & operator=(const pipe_expert_shm_ring &) = delete;

    static std::unique_ptr<pipe_expert_shm_ring> create(
            const std::string & name, uint32_t n_tokens, uint32_t n_embd,
            uint32_t n_assignments);
    static std::unique_ptr<pipe_expert_shm_ring> create_for_test(
            const std::string & name, size_t request_capacity, size_t response_capacity);
    static std::unique_ptr<pipe_expert_shm_ring> attach(const std::string & name);

    bool valid() const;
    const std::string & name() const;

    bool write_request(const void * data, size_t length, pipe_expert_shm_ref & ref);
    bool write_response(const void * data, size_t length, pipe_expert_shm_ref & ref);
    bool read_request(const pipe_expert_shm_ref & ref, const uint8_t ** data);
    bool read_response(const pipe_expert_shm_ref & ref, const uint8_t ** data);
    bool consume_request(const pipe_expert_shm_ref & ref);
    bool consume_response(const pipe_expert_shm_ref & ref);

    const char * error() const;

  private:
    pipe_expert_shm_ring() = default;

    bool open_region(const std::string & name, bool owner, size_t total_size,
                     size_t request_capacity, size_t response_capacity);
    bool write(bool response, const void * data, size_t length, pipe_expert_shm_ref & ref);
    bool read(bool response, const pipe_expert_shm_ref & ref, const uint8_t ** data);
    bool consume(bool response, const pipe_expert_shm_ref & ref);
    void fail(const char * message);

    std::string name_;
    void * mapping_ = nullptr;
    size_t mapping_size_ = 0;
    bool owner_ = false;
    struct Header * header_ = nullptr;
    char error_[192] = {};
    std::mutex write_mutex_;
};
