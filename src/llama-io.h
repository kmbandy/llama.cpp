#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // Write n_rows rows of `tensor` (starting at byte `offset`), re-encoded as
    // `dst_type`. Identical to write_tensor() when dst_type == tensor->type.
    // The generic implementation pulls the rows to host, quantizes, and writes
    // the bytes; a writer that only counts bytes overrides it to skip the pull.
    virtual void write_tensor_as(ggml_tensor * tensor, size_t offset, size_t n_rows, size_t n_per_row, ggml_type dst_type);

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // Inverse of write_tensor_as(): read n_rows rows encoded as `src_type` and
    // store them into `tensor` (whose own type is the decode target).
    virtual void read_tensor_as(ggml_tensor * tensor, size_t offset, size_t n_rows, size_t n_per_row, ggml_type src_type);

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
