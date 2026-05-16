// mt-locked-buffer.h
//
// LockedBuffer: a page-aligned byte buffer that can be optionally pinned in
// RAM via mlock() / VirtualLock(). Used by the tiered-KV warm staging tier so
// the kernel cannot page warm-tier KV out to swap under memory pressure (the
// failure mode that turned warm-tier reads into swap-disk reads, observed on
// long-context workloads with constrained host RAM).
//
// Owns its allocation via mmap (POSIX) or VirtualAlloc (Windows) — does NOT
// use the C++ heap, so it bypasses any allocator-level placement assumptions
// the kernel could apply to vector/new memory.
//
// Locking is BEST-EFFORT: if the kernel refuses mlock (e.g. RLIMIT_MEMLOCK
// too small or no CAP_IPC_LOCK), allocation still succeeds and the buffer
// is unlocked. A warning is logged with a concrete remediation hint.
//
// Drop-in replacement for `std::vector<uint8_t>` in the narrow API surface
// the warm-tier code uses today: data(), size(), and a single bulk allocate.
//
// Cross-platform:
//   Linux : mmap(MAP_PRIVATE|MAP_ANONYMOUS [|MAP_LOCKED]) + fallback mlock()
//   macOS : mmap(MAP_PRIVATE|MAP_ANONYMOUS) + mlock()  (no MAP_LOCKED)
//   Win32 : VirtualAlloc(MEM_COMMIT|MEM_RESERVE) + VirtualLock()
#pragma once

#include <cstddef>
#include <cstdint>

namespace mt {

class LockedBuffer {
public:
    LockedBuffer() = default;
    ~LockedBuffer();

    // Allocate `size_bytes` of zero-initialized memory.
    //
    // If `lock` is true, attempts to pin the memory in RAM. On failure the
    // buffer remains allocated (operation succeeds) but `is_locked()` will
    // return false and a warning is logged.
    //
    // Returns false only if the underlying allocation itself fails.
    bool allocate(size_t size_bytes, bool lock = true);

    // Release the buffer (munlock + munmap / VirtualUnlock + VirtualFree).
    // Safe to call on an empty buffer. Called automatically by the destructor.
    void reset();

    uint8_t *       data()         { return data_; }
    const uint8_t * data()   const { return data_; }
    size_t          size()   const { return size_; }
    bool            is_locked() const { return locked_; }

    // Disable copy. Allow move so we can reassign in the tiered cache class.
    LockedBuffer(const LockedBuffer &)             = delete;
    LockedBuffer & operator=(const LockedBuffer &) = delete;
    LockedBuffer(LockedBuffer && other) noexcept;
    LockedBuffer & operator=(LockedBuffer && other) noexcept;

private:
    uint8_t * data_   = nullptr;
    size_t    size_   = 0;
    bool      locked_ = false;
};

} // namespace mt
