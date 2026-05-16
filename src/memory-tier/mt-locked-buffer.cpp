// mt-locked-buffer.cpp — see header for rationale.
#include "mt-locked-buffer.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstring>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <unistd.h>
#endif

namespace mt {

LockedBuffer::~LockedBuffer() {
    reset();
}

LockedBuffer::LockedBuffer(LockedBuffer && other) noexcept
    : data_(other.data_), size_(other.size_), locked_(other.locked_) {
    other.data_   = nullptr;
    other.size_   = 0;
    other.locked_ = false;
}

LockedBuffer & LockedBuffer::operator=(LockedBuffer && other) noexcept {
    if (this != &other) {
        reset();
        data_         = other.data_;
        size_         = other.size_;
        locked_       = other.locked_;
        other.data_   = nullptr;
        other.size_   = 0;
        other.locked_ = false;
    }
    return *this;
}

void LockedBuffer::reset() {
    if (!data_) {
        size_   = 0;
        locked_ = false;
        return;
    }
#ifdef _WIN32
    if (locked_) {
        VirtualUnlock(data_, size_);
    }
    VirtualFree(data_, 0, MEM_RELEASE);
#else
    if (locked_) {
        munlock(data_, size_);
    }
    munmap(data_, size_);
#endif
    data_   = nullptr;
    size_   = 0;
    locked_ = false;
}

bool LockedBuffer::allocate(size_t size_bytes, bool lock) {
    reset();
    if (size_bytes == 0) {
        return true;
    }

#ifdef _WIN32
    data_ = static_cast<uint8_t *>(
        VirtualAlloc(nullptr, size_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!data_) {
        LLAMA_LOG_ERROR(
            "mt::LockedBuffer: VirtualAlloc(%zu bytes / %.1f MiB) failed (error %lu)\n",
            size_bytes, (double) size_bytes / (1024.0 * 1024.0),
            (unsigned long) GetLastError());
        return false;
    }
    size_ = size_bytes;
    if (lock) {
        if (VirtualLock(data_, size_bytes)) {
            locked_ = true;
        } else {
            LLAMA_LOG_WARN(
                "mt::LockedBuffer: VirtualLock(%.1f MiB) failed (error %lu). "
                "Buffer is allocated but eligible for page-out under RAM pressure. "
                "Raise the process working-set quota to fix.\n",
                (double) size_bytes / (1024.0 * 1024.0),
                (unsigned long) GetLastError());
        }
    }
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    bool tried_map_locked = false;

  #ifdef MAP_LOCKED
    if (lock) {
        flags |= MAP_LOCKED;
        tried_map_locked = true;
    }
  #endif

    void * p = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, flags, -1, 0);

  #ifdef MAP_LOCKED
    if (p == MAP_FAILED && tried_map_locked) {
        // MAP_LOCKED often fails on RLIMIT_MEMLOCK; retry without it and try
        // mlock() after, which gives a clearer error path.
        int saved_errno = errno;
        flags &= ~MAP_LOCKED;
        p = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (p == MAP_FAILED) {
            LLAMA_LOG_ERROR(
                "mt::LockedBuffer: mmap(%.1f MiB) failed: %s "
                "(also failed with MAP_LOCKED: %s)\n",
                (double) size_bytes / (1024.0 * 1024.0),
                strerror(errno), strerror(saved_errno));
            return false;
        }
        // fall through — we have an unlocked mapping; mlock() pass below.
        tried_map_locked = false;
    } else if (p != MAP_FAILED && tried_map_locked) {
        // MAP_LOCKED succeeded — pages are pinned.
        locked_ = true;
    }
  #endif

    if (p == MAP_FAILED) {
        LLAMA_LOG_ERROR(
            "mt::LockedBuffer: mmap(%.1f MiB) failed: %s\n",
            (double) size_bytes / (1024.0 * 1024.0), strerror(errno));
        return false;
    }

    data_ = static_cast<uint8_t *>(p);
    size_ = size_bytes;

    // If we asked for a lock but MAP_LOCKED wasn't used or didn't apply,
    // do a follow-up mlock() so we get a clear errno path.
    if (lock && !locked_) {
        if (mlock(data_, size_bytes) == 0) {
            locked_ = true;
        } else {
            LLAMA_LOG_WARN(
                "mt::LockedBuffer: mlock(%.1f MiB) failed: %s. "
                "Buffer is allocated but eligible for swap-out under RAM pressure. "
                "Raise RLIMIT_MEMLOCK (systemd unit: LimitMEMLOCK=infinity) "
                "or run with CAP_IPC_LOCK to fix.\n",
                (double) size_bytes / (1024.0 * 1024.0), strerror(errno));
        }
    }
#endif

    return true;
}

} // namespace mt
