#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using HsaExportDmaBufFn = uint32_t (*)(const void *, size_t, int *, uint64_t *);
using HsaCloseDmaBufFn  = uint32_t (*)(int);

const char * hsa_status_name(uint32_t status) {
    switch (status) {
        case 0x0000: return "SUCCESS";
        case 0x1004: return "INVALID_AGENT";
        case 0x1008: return "OUT_OF_RESOURCES";
        case 0x100B: return "NOT_INITIALIZED";
        default:     return "UNKNOWN";
    }
}

void print_rocm_version() {
    int rt_version = 0;
    if (hipRuntimeGetVersion(&rt_version) == hipSuccess) {
        const int major = rt_version / 10000000;
        const int minor = (rt_version / 100000) % 100;
        const int patch = (rt_version / 1000) % 100;
        std::printf("ROCm runtime: %d.%d.%d\n", major, minor, patch);
    } else {
        std::printf("ROCm runtime: unknown\n");
    }
    std::printf("P2P validated ROCm: 7.2.2\n");
}

}  // namespace

int main() {
    print_rocm_version();

    void * libhsa = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (libhsa == nullptr) {
        std::fprintf(stderr, "unsupported: dlopen libhsa-runtime64.so.1 failed: %s\n", dlerror());
        return 2;
    }

    void * export_ptr = dlsym(libhsa, "hsa_amd_portable_export_dmabuf");
    if (export_ptr == nullptr) {
        std::fprintf(stderr, "unsupported: dlsym hsa_amd_portable_export_dmabuf failed: %s\n", dlerror());
        dlclose(libhsa);
        return 2;
    }
    void * close_ptr = dlsym(libhsa, "hsa_amd_portable_close_dmabuf");
    if (close_ptr == nullptr) {
        std::fprintf(stderr, "unsupported: dlsym hsa_amd_portable_close_dmabuf failed: %s\n", dlerror());
        dlclose(libhsa);
        return 2;
    }

    auto hsa_export = reinterpret_cast<HsaExportDmaBufFn>(export_ptr);
    auto hsa_close  = reinterpret_cast<HsaCloseDmaBufFn>(close_ptr);

    constexpr size_t kProbeBytes = 4096;
    void * vram = nullptr;
    hipError_t err = hipMalloc(&vram, kProbeBytes);
    if (err != hipSuccess) {
        std::fprintf(stderr, "unsupported: hipMalloc(%zu) failed: %s\n",
                     kProbeBytes, hipGetErrorString(err));
        dlclose(libhsa);
        return 2;
    }

    int dmabuf_fd = -1;
    uint64_t dmabuf_offset = 0;
    const uint32_t status = hsa_export(vram, kProbeBytes, &dmabuf_fd, &dmabuf_offset);
    if (status != 0 || dmabuf_fd < 0) {
        std::fprintf(stderr, "unsupported: hsa_amd_portable_export_dmabuf failed: %s (0x%04x), fd=%d\n",
                     hsa_status_name(status), status, dmabuf_fd);
        (void) hipFree(vram);
        dlclose(libhsa);
        return 2;
    }

    struct stat st {};
    if (fstat(dmabuf_fd, &st) != 0) {
        std::fprintf(stderr, "unsupported: fstat(dmabuf) failed: %s\n", std::strerror(errno));
        hsa_close(dmabuf_fd);
        (void) hipFree(vram);
        dlclose(libhsa);
        return 2;
    }

    void * mapped = mmap(nullptr, kProbeBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                         dmabuf_fd, (off_t) dmabuf_offset);
    if (mapped == MAP_FAILED) {
        std::fprintf(stderr, "unsupported: mmap(dmabuf) failed: %s\n", std::strerror(errno));
        hsa_close(dmabuf_fd);
        (void) hipFree(vram);
        dlclose(libhsa);
        return 2;
    }

    munmap(mapped, kProbeBytes);
    const uint32_t close_status = hsa_close(dmabuf_fd);
    (void) hipFree(vram);
    dlclose(libhsa);

    if (close_status != 0) {
        std::fprintf(stderr, "unsupported: hsa_amd_portable_close_dmabuf failed: %s (0x%04x)\n",
                     hsa_status_name(close_status), close_status);
        return 2;
    }

    std::printf("supported: dma_buf export, fstat, and mmap succeeded\n");
    return 0;
}
