// MAD-238: pybind11 bindings for the weight pager (src/weight-pager/wp-*.h).
//
// First-cut minimal surface: catalog operations only (add_page, find_page,
// n_pages, page_meta). Init/shutdown/ensure (which need a ggml backend buffer
// type) land in iteration 2.
//
// The wp:: namespace symbols are exported from libllama.so (verified via nm),
// so this module links against libllama directly — no separate libwp needed.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/weight-pager/wp-pager.h"
#include "src/weight-pager/wp-page-catalog.h"
#include "ggml-cuda.h"

// AMD-only build (matches the rest of mad-lab's llama.cpp fork).
#include <hip/hip_runtime.h>
#define WP_DEVICE_MEMCPY(dst, src, n) hipMemcpy((dst), (src), (n), hipMemcpyDeviceToDevice)
#define WP_DEVICE_SYNCHRONIZE()       hipDeviceSynchronize()
#define WP_DEVICE_GET_ERROR_STRING(e) hipGetErrorString(e)

#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace wp;

PYBIND11_MODULE(wp_native, m) {
    m.doc() = "Weight pager bindings — pybind11 wrap of src/weight-pager.";

    // ─── PageMeta (read-only view of catalog entry) ──────────────────────
    py::class_<PageMeta>(m, "PageMeta")
        .def_readonly("tensor_name",     &PageMeta::tensor_name)
        .def_readonly("file_idx",        &PageMeta::file_idx)
        .def_readonly("file_offset",     &PageMeta::file_offset)
        .def_readonly("size",            &PageMeta::size)
        .def_readonly("block_idx",       &PageMeta::block_idx)
        .def_readonly("expert_idx",      &PageMeta::expert_idx)
        .def_readonly("is_expert",       &PageMeta::is_expert)
        .def_readonly("is_consolidated", &PageMeta::is_consolidated)
        .def_readonly("is_sub_expert",   &PageMeta::is_sub_expert)
        .def_readonly("parent_page_idx", &PageMeta::parent_page_idx)
        .def("__repr__", [](const PageMeta & m) {
            return "<PageMeta name=" + m.tensor_name +
                   " size=" + std::to_string(m.size) +
                   " block=" + std::to_string(m.block_idx) +
                   " expert=" + std::to_string(m.expert_idx) + ">";
        });

    // ─── WeightPager.Config ──────────────────────────────────────────────
    py::class_<WeightPager::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("n_slots",         &WeightPager::Config::n_slots)
        .def_readwrite("prefetch_depth",  &WeightPager::Config::prefetch_depth)
        .def_readwrite("prefer_async_io", &WeightPager::Config::prefer_async_io);

    // ─── WeightPager (catalog-only API for iteration 1) ──────────────────
    py::class_<WeightPager>(m, "WeightPager")
        .def(py::init<>())
        .def("add_page", &WeightPager::add_page,
             py::arg("name"), py::arg("file_idx"), py::arg("file_offset"),
             py::arg("size"), py::arg("n_experts") = 1,
             "Register a tensor for the catalog. Returns the assigned page index. "
             "Must be called before init(). n_experts > 1 marks a consolidated MoE tensor.")
        .def("find_page", &WeightPager::find_page,
             py::arg("name"),
             "Find a page index by tensor name. Returns -1 if not found.")
        .def("n_pages", &WeightPager::n_pages,
             "Total number of registered pages.")
        .def("max_page_size", &WeightPager::max_page_size,
             "Maximum page size in bytes across the catalog.")
        .def("is_initialized", &WeightPager::is_initialized,
             "True iff init() has completed successfully.")
        .def("page_meta", &WeightPager::page_meta,
             py::arg("page_idx"),
             py::return_value_policy::reference_internal,
             "Metadata for a registered page (name, file offsets, MoE classification).")
        // Iteration 2: init/ensure/shutdown for actual GPU paging.
        .def("init_for_device",
             [](WeightPager & self, const WeightPager::Config & cfg,
                int device_idx, const std::vector<std::string> & gguf_paths) -> bool {
                 // Resolve HIP/CUDA backend buffer type for this device.
                 ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device_idx);
                 if (buft == nullptr) {
                     throw std::runtime_error(
                         "ggml_backend_cuda_buffer_type returned null — is the HIP backend loaded?");
                 }
                 // Open fds for each GGUF path.
                 std::vector<int> fds;
                 fds.reserve(gguf_paths.size());
                 for (const auto & p : gguf_paths) {
                     int fd = ::open(p.c_str(), O_RDONLY);
                     if (fd < 0) {
                         // Clean up any opened fds before erroring.
                         for (int prev : fds) ::close(prev);
                         throw std::runtime_error("failed to open " + p);
                     }
                     fds.push_back(fd);
                 }
                 std::vector<int> devices_used{device_idx};
                 return self.init(cfg, buft, device_idx, std::move(fds), devices_used);
             },
             py::arg("cfg"), py::arg("device_idx"), py::arg("gguf_paths"),
             "Initialize the pager for a given HIP/CUDA device. Opens fds for "
             "each GGUF file internally and resolves the device's backend buffer "
             "type. Returns True on success.")
        .def("shutdown", &WeightPager::shutdown,
             "Tear down pool/transport/prefetch and close fds. Safe to call multiple times.")
        .def("ensure", [](WeightPager & self, int page_idx) -> uintptr_t {
                 void * ptr = self.ensure(page_idx);
                 return reinterpret_cast<uintptr_t>(ptr);
             },
             py::arg("page_idx"),
             "Page the tensor into VRAM (sync if not prefetched). Returns the "
             "VRAM device pointer as an integer (0 on failure). Wrap with torch via "
             "torch.from_blob or .as_strided on a tensor constructed at that address.")
        .def("prefetch_page", &WeightPager::prefetch_page, py::arg("page_idx"),
             "Async prefetch hint. No-op if already in flight or loaded.")
        .def("tick", &WeightPager::tick,
             "Drive the prefetch pipeline forward. Idempotent + non-blocking.")
        .def("slot_for_page", &WeightPager::slot_for_page, py::arg("page_idx"),
             "Slot index in the VRAM ring (or -1 if not currently loaded).");

    // ─── Device-to-device memcpy helper ──────────────────────────────────
    // Used by the Python wrapper to copy from a pager slot into a freshly
    // allocated torch CUDA tensor. This is the "safe copy-out" path for
    // torch tensor wrapping — the slot is free to evict immediately after
    // this returns. (Zero-copy view via __cuda_array_interface__ is fragile
    // because torch.as_tensor doesn't honor that protocol; revisit when
    // slot-pin lands in MAD-231.)
    m.def("device_memcpy",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, size_t nbytes) {
              if (dst_ptr == 0 || src_ptr == 0) {
                  throw std::runtime_error("device_memcpy: null pointer");
              }
              auto err = WP_DEVICE_MEMCPY(
                  reinterpret_cast<void *>(dst_ptr),
                  reinterpret_cast<void *>(src_ptr),
                  nbytes);
              if (err != 0) {
                  throw std::runtime_error(std::string("device_memcpy failed: ") +
                                           WP_DEVICE_GET_ERROR_STRING(err));
              }
              // Sync because the calibration code reads the dst tensor on the
              // host side via torch operations expecting the data to be ready.
              WP_DEVICE_SYNCHRONIZE();
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          "Device-to-device memcpy. dst_ptr + src_ptr are HIP/CUDA device "
          "pointers (uintptr_t). Synchronizes before returning.");
}
