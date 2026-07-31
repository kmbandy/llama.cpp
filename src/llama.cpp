#include "llama.h"
#include <fcntl.h>
#include <unistd.h>  // close() for weight-pager blob descriptors

#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"
#include "llama-weight-pager.h"
#include "weight-pager/wp-file-io.h"
#include "weight-pager/wp-pager-set.h"
#include "weight-pager/wp-partition.h"

#include <cerrno>
#include <climits>
#include <cstdlib>  // strtol
#include <map>     // MAD-420: page_size_histogram for size-class auto-sizing

#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

//
// Weight pager initialization helper
//

// Parse a HIP/CUDA backend device name like "ROCm0" / "ROCm1" / "CUDA0" /
// "CUDA1" into the integer device index. Returns -1 if the name doesn't
// match the expected pattern.
static int parse_backend_dev_idx(const char * dev_name) {
    if (dev_name == nullptr) return -1;
    // Skip any leading non-digit characters.
    const char * p = dev_name;
    while (*p != '\0' && (*p < '0' || *p > '9')) ++p;
    if (*p == '\0') return -1;
    char * end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p) return -1;
    return (int) v;
}

static int parse_positive_int_env(const char * var, int fallback) {
    const char * v = std::getenv(var);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }

    errno = 0;
    char * end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (errno != 0 || end == v || (end != nullptr && *end != '\0') || n <= 0 || n > INT_MAX) {
        LLAMA_LOG_WARN("%s: ignoring invalid %s=%s\n", __func__, var, v);
        return fallback;
    }
    return (int) n;
}

static int wp_prefetch_depth_from_env() {
    static const int value = parse_positive_int_env("WP_PREFETCH_DEPTH", 4);
    return value;
}

static int wp_iouring_depth_override_from_env() {
    static const int value = parse_positive_int_env("WP_IOURING_DEPTH", 0);
    return value;
}

// WP_RESIDENT_DENSE: page only routed experts, keep dense weights resident.
// Must mirror the same-named gate in llama-model.cpp (both read the env).
static bool wp_resident_dense_from_env() {
    const char * v = std::getenv("WP_RESIDENT_DENSE");
    return v != nullptr && v[0] == '1';
}


static bool init_weight_pager(llama_model & model, llama_model_loader & ml, const llama_model_params & params) {
    if (!params.weight_paging_enabled) {
        return true;
    }
    if (ml.weight_page_infos.empty()) {
        LLAMA_LOG_WARN("%s: weight paging enabled but no weight page info available\n", __func__);
        return true;
    }
    if (!model.weight_pager || model.weight_pager->weight_tensor_ptrs.empty()) {
        LLAMA_LOG_WARN("%s: no weight tensor pointers collected - nothing to page\n", __func__);
        return true;
    }
    if (model.weight_pager->weight_tensor_ptrs.size() !=
        model.weight_pager->weight_tensor_bufts.size()) {
        throw std::runtime_error(
            "weight pager: tensor/buft carrier size mismatch");
    }

    std::vector<ggml_backend_buffer_type_t> gpu_bufts;
    for (ggml_backend_buffer_type_t buft : model.weight_pager->weight_bufts) {
        if (buft == nullptr || ggml_backend_buft_is_host(buft)) {
            continue;
        }
        if (std::find(gpu_bufts.begin(), gpu_bufts.end(), buft) == gpu_bufts.end()) {
            gpu_bufts.push_back(buft);
        }
    }
    if (gpu_bufts.empty()) {
        throw std::runtime_error(
            "weight pager: no GPU buffer types found among paged weights");
    }

    auto buft_for_name = [&](const std::string & name) {
        for (size_t i = 0; i < model.weight_pager->weight_tensor_ptrs.size(); ++i) {
            ggml_tensor * t = model.weight_pager->weight_tensor_ptrs[i];
            if (t == nullptr || name != ggml_get_name(t)) {
                continue;
            }
            if (t->buffer != nullptr) {
                return ggml_backend_buffer_get_type(t->buffer);
            }
            return model.weight_pager->weight_tensor_bufts[i];
        }
        return (ggml_backend_buffer_type_t) nullptr;
    };

    // Build the complete catalog once. Its page_buft_ vector is the canonical
    // per-page device assignment used for the partition below.
    wp::WeightPager complete_catalog;
    for (const llama_weight_page_info & info : ml.weight_page_infos) {
        ggml_backend_buffer_type_t page_buft = buft_for_name(info.name);
        if (page_buft == nullptr || ggml_backend_buft_is_host(page_buft)) {
            throw std::runtime_error(
                "weight pager: catalog page has no GPU device: " + info.name);
        }
        complete_catalog.add_page(
            info.name, info.file_idx, info.offset, info.size,
            info.n_experts, page_buft);
    }

    std::vector<wp::PagePartitionInput> partition_inputs;
    partition_inputs.reserve((size_t) complete_catalog.n_pages());
    for (int page_idx = 0; page_idx < complete_catalog.n_pages(); ++page_idx) {
        partition_inputs.push_back({
            complete_catalog.page_meta(page_idx).tensor_name,
            complete_catalog.page_buft(page_idx),
            true,
        });
    }
    const wp::PartitionedPages partitioned =
        wp::partition_pages_by_device(partition_inputs);
    if (partitioned.n_paged != (size_t) complete_catalog.n_pages()) {
        throw std::runtime_error("weight pager: not every catalog page was partitioned");
    }

    model.wp_pager = std::make_unique<wp::WeightPagerSet>();
    for (const wp::PagePartition & partition : partitioned.partitions) {
        auto buft = (ggml_backend_buffer_type_t) partition.device;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == nullptr) {
            throw std::runtime_error(
                "weight pager: partition buffer type has no device");
        }
        const int device_idx = parse_backend_dev_idx(ggml_backend_dev_name(dev));
        if (device_idx < 0) {
            throw std::runtime_error(format(
                "weight pager: could not parse device index from '%s'",
                ggml_backend_dev_name(dev)));
        }
        model.wp_pager->add_pager(
            buft, device_idx, ggml_backend_dev_name(dev));
    }

    // Re-add each source tensor to exactly one local catalog. Consolidated
    // parents and all synthesized expert children share page_buft_, so this
    // preserves their local adjacency and parent indices.
    for (const llama_weight_page_info & info : ml.weight_page_infos) {
        const int full_idx = complete_catalog.find_page(info.name);
        if (full_idx < 0) {
            throw std::runtime_error(
                "weight pager: source page missing from complete catalog: " + info.name);
        }
        const ggml_backend_buffer_type_t buft =
            complete_catalog.page_buft(full_idx);
        wp::WeightPager * target = nullptr;
        for (wp::WeightPagerSet::Entry & entry : model.wp_pager->entries()) {
            if (entry.buft == buft) {
                target = entry.pager.get();
                break;
            }
        }
        if (target == nullptr) {
            throw std::runtime_error(
                "weight pager: no partition owner for page: " + info.name);
        }
        target->add_page(
            info.name, info.file_idx, info.offset, info.size,
            info.n_experts, buft);
    }

    model.wp_pager->build_routes((size_t) complete_catalog.n_pages());
    for (const auto & expected : partitioned.routes) {
        const wp::WeightPagerSet::Route actual =
            model.wp_pager->find_page(expected.first);
        const wp::PagePartitionRoute route = expected.second;
        if (!actual ||
            actual.pager != model.wp_pager->entries()[route.partition_idx].pager.get() ||
            actual.page_idx != route.page_idx) {
            throw std::runtime_error(
                "weight pager: partition route mismatch for page: " + expected.first);
        }
    }

    // wp-repack blobs — redirect routed-expert pages at their expert-major
    // copy. The catalog was built from the source GGUFs above, so a page with
    // no blob entry simply keeps reading the original file; the two sources
    // coexist and `file_idx` distinguishes them. Blob fds are appended AFTER
    // the model's own fds (see the fds loop below), so a blob's descriptor
    // index is ml.files.size() + blob_idx.
    const size_t n_blob_files = params.weight_paging_blob_files != nullptr
        ? params.weight_paging_n_blob_files : 0;
    if (n_blob_files > 0 && params.weight_paging_n_blob_entries > 0) {
        const size_t fd_base = ml.files.size();
        if (fd_base + n_blob_files > (size_t) UINT16_MAX) {
            throw std::runtime_error("weight pager: too many files to index with uint16 file_idx");
        }
        size_t n_remapped = 0;
        size_t n_absent   = 0;
        for (size_t i = 0; i < params.weight_paging_n_blob_entries; ++i) {
            const llama_wp_blob_entry & e = params.weight_paging_blob_entries[i];
            if (e.name == nullptr) {
                throw std::runtime_error("weight pager: blob entry with null name");
            }
            if (e.blob_idx >= n_blob_files) {
                throw std::runtime_error(format(
                    "weight pager: blob entry '%s' references blob %u of %zu",
                    e.name, e.blob_idx, n_blob_files));
            }
            const uint16_t file_idx = (uint16_t) (fd_base + e.blob_idx);
            // A page lives in exactly one device pager; ask each until one
            // owns it. NotFound everywhere is legitimate -- the blob set may
            // cover layers this process does not own (pipeline band) or
            // experts held resident rather than paged.
            bool handled = false;
            for (wp::WeightPagerSet::Entry & entry : model.wp_pager->entries()) {
                const wp::PageCatalog::RemapStatus st = entry.pager->remap_page_source(
                    e.name, file_idx, e.blob_offset, (size_t) e.size);
                if (st == wp::PageCatalog::RemapStatus::NotFound) {
                    continue;
                }
                if (st == wp::PageCatalog::RemapStatus::SizeMismatch) {
                    throw std::runtime_error(format(
                        "weight pager: blob entry '%s' is %" PRIu64 " bytes but the model's "
                        "page is a different size -- this blob set was built from a "
                        "different model or quantisation", e.name, e.size));
                }
                if (st == wp::PageCatalog::RemapStatus::NotPageable) {
                    throw std::runtime_error(format(
                        "weight pager: blob entry '%s' names a pinned or consolidated "
                        "page, which never reads from a file", e.name));
                }
                ++n_remapped;
                handled = true;
                break;
            }
            if (!handled) {
                ++n_absent;
            }
        }
        if (n_remapped == 0) {
            throw std::runtime_error(format(
                "weight pager: none of the %zu blob entries matched a catalog page -- "
                "the blob set does not describe this model",
                params.weight_paging_n_blob_entries));
        }
        LLAMA_LOG_INFO("%s: wp-repack blobs: %zu files, %zu pages remapped"
                       " (%zu entries not owned by this process)\n",
                       __func__, n_blob_files, n_remapped, n_absent);
    }

    if (wp_resident_dense_from_env()) {
        for (const wp::WeightPagerSet::Entry & entry : model.wp_pager->entries()) {
            for (int page_idx = 0; page_idx < entry.pager->n_pages(); ++page_idx) {
                const wp::PageMeta & meta = entry.pager->page_meta(page_idx);
                if (!meta.is_expert && !meta.is_consolidated &&
                    !meta.is_sub_expert && !meta.is_pinned) {
                    throw std::runtime_error(
                        "weight pager: resident-dense catalog contains dense page: " +
                        meta.tensor_name);
                }
            }
        }
    }

    const int prefetch_depth = wp_prefetch_depth_from_env();
    int io_uring_depth = wp_iouring_depth_override_from_env();
    if (io_uring_depth <= 0) {
        io_uring_depth = prefetch_depth;
    }
    io_uring_depth = std::max(io_uring_depth, prefetch_depth);

    for (wp::WeightPagerSet::Entry & entry : model.wp_pager->entries()) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(entry.buft);
        int n_slots = params.weight_paging_slots > 0
            ? params.weight_paging_slots : entry.pager->n_pages();
        if (n_slots <= 0) {
            n_slots = 1;
        }
        if (params.weight_paging_slots <= 0) {
            size_t free_vram = 0;
            size_t total_vram = 0;
            ggml_backend_dev_memory(dev, &free_vram, &total_vram);
            if (total_vram != 0) {
                const size_t reserve = 3ULL * 1024 * 1024 * 1024;
                const size_t usable = free_vram > reserve ? free_vram - reserve : 0;
                const size_t stride = entry.pager->max_page_size();
                // MAD-420 — when size-class slots are enabled, size the pool
                //   by the AVERAGE page size, not the max. The uniform
                //   assumption (usable / max_page_size) under-provisions by
                //   the waste factor on models whose expert sub-pages are
                //   strongly non-uniform (GLM-5.2 UD-Q2_K_XL: 97% of pages
                //   are 3.469/4.594 MiB but max is 6.375 MiB, so uniform
                //   counts 0.61x the slots that actually fit). Using the
                //   histogram's average sizes the arena budget to hold
                //   min(n_pages, usable/avg) pages, which is what the pre-
                //   carve solver then packs. --weight-paging-slots override
                //   is left alone (handled by the branch guard above).
                const char * sc_env = std::getenv("WP_SIZE_CLASS_SLOTS");
                const bool sc_on = (sc_env != nullptr && sc_env[0] == '1' && sc_env[1] == '\0');
                int fit = stride > 0 ? (int) (usable / stride) : 0;
                if (sc_on && stride > 0) {
                    std::map<size_t, int> hist = entry.pager->page_size_histogram();
                    long long total_pages = 0;
                    long long weighted    = 0;
                    for (const auto & kv : hist) {
                        total_pages += kv.second;
                        weighted    += (long long) kv.second * (long long) kv.first;
                    }
                    if (total_pages > 0 && weighted > 0) {
                        const double avg = (double) weighted / (double) total_pages;
                        // Budget needed to hold every page with size classes:
                        //   n_pages * avg. If that fits in usable, allocate
                        //   exactly that (no paging). Otherwise cap at usable
                        //   and let the pre-carve solver pack K = usable/avg.
                        const double need_all = (double) total_pages * avg;
                        const size_t budget = (need_all <= (double) usable)
                                              ? (size_t) need_all : usable;
                        const int fit_sc = (int) (budget / stride);
                        if (fit_sc >= 1) {
                            fit = fit_sc;
                        }
                    }
                }
                if (fit >= 1) {
                    n_slots = std::min(n_slots, fit);
                }
            }
        }

        std::vector<int> fds;
        fds.reserve(ml.files.size() + n_blob_files);
        for (const auto & file : ml.files) {
            const int fd = wp::dup_clear_o_direct(file->file_id());
            // Every page carries a file_idx that indexes THIS vector, so a
            // skipped entry would silently shift every later file's index and
            // read the wrong bytes from the wrong file. Fail instead.
            if (fd < 0) {
                for (int open_fd : fds) {
                    close(open_fd);
                }
                model.wp_pager.reset();
                throw std::runtime_error(
                    "weight pager: could not duplicate a model file descriptor");
            }
            fds.push_back(fd);
        }
        if (fds.empty()) {
            model.wp_pager.reset();
            throw std::runtime_error("weight pager: no usable file descriptors");
        }
        // Blob fds follow the model's, in blob_idx order -- this is the
        // ordering the remap above assumed. Each device pager owns its own
        // descriptors (init() takes ownership and closes them), so open a
        // fresh set per entry rather than sharing.
        for (size_t b = 0; b < n_blob_files; ++b) {
            const char * path = params.weight_paging_blob_files[b];
            const int fd = path != nullptr ? open(path, O_RDONLY) : -1;
            if (fd < 0) {
                for (int open_fd : fds) {
                    close(open_fd);
                }
                model.wp_pager.reset();
                throw std::runtime_error(format(
                    "weight pager: could not open repack blob '%s': %s",
                    path != nullptr ? path : "(null)", strerror(errno)));
            }
            fds.push_back(fd);
        }

        wp::WeightPager::Config cfg;
        cfg.n_slots             = n_slots;
        cfg.prefetch_depth      = prefetch_depth;
        cfg.io_uring_depth      = io_uring_depth;
        cfg.prefer_async_io     = params.weight_paging_prefetch;
        cfg.host_budget_divisor = model.wp_pager->size();

        const char * buft_name = ggml_backend_buft_name(entry.buft);
        if (buft_name != nullptr && std::strstr(buft_name, "Vulkan") != nullptr) {
            size_t alignment = 1;
            for (size_t i = 0; i < model.weight_pager->weight_tensor_ptrs.size(); ++i) {
                const ggml_tensor * t = model.weight_pager->weight_tensor_ptrs[i];
                if (t != nullptr && model.weight_pager->weight_tensor_bufts[i] == entry.buft) {
                    const size_t type_size = ggml_type_size(t->type);
                    if (type_size > 0) {
                        alignment = alignment / std::gcd(alignment, type_size) * type_size;
                    }
                }
            }
            cfg.block_alignment = alignment;
        }

        if (!entry.pager->init(
                cfg, entry.buft, entry.device_idx, std::move(fds),
                {entry.device_idx})) {
            model.wp_pager.reset();
            throw std::runtime_error(format(
                "weight pager: init failed for device %s with %d slots",
                entry.device_name.c_str(), n_slots));
        }
        LLAMA_LOG_INFO(
            "%s: device pager ready: %s pages=%d slots=%d max_page=%zu\n",
            __func__, entry.device_name.c_str(), entry.pager->n_pages(),
            n_slots, entry.pager->max_page_size());
    }

    size_t n_placed = 0;
    for (ggml_tensor * t : model.weight_pager->weight_tensor_ptrs) {
        if (t == nullptr || t->data != nullptr) {
            continue;
        }
        const wp::WeightPagerSet::Route route =
            model.wp_pager->find_page(ggml_get_name(t));
        if (!route) {
            continue;
        }
        ggml_backend_buffer_t pool_buf = route.pager->pool_buf(route.page_idx);
        void * placeholder =
            pool_buf != nullptr ? ggml_backend_buffer_get_base(pool_buf) : nullptr;
        if (placeholder == nullptr) {
            throw std::runtime_error("weight pager: pool buffer has null base");
        }
        t->data   = placeholder;
        t->buffer = pool_buf;
        ++n_placed;
    }

    int n_tid2eid = 0;
    for (int il = 0; il < (int) model.layers.size(); ++il) {
        ggml_tensor * t = model.layers[il].ffn_gate_tid2eid;
        if (t == nullptr || t->data == nullptr || t->type != GGML_TYPE_I32) {
            continue;
        }
        const int n_used = (int) t->ne[0];
        const int n_vocab = (int) t->ne[1];
        if (n_used <= 0 || n_vocab <= 0) {
            continue;
        }
        const size_t nbytes =
            (size_t) n_used * (size_t) n_vocab * sizeof(int32_t);
        std::vector<int32_t> host(nbytes / sizeof(int32_t));
        if (t->buffer != nullptr) {
            ggml_backend_tensor_get(t, host.data(), 0, nbytes);
        } else {
            std::memcpy(host.data(), t->data, nbytes);
        }
        model.wp_pager->register_tid2eid_host(
            il, n_used, n_vocab, host.data());
        ++n_tid2eid;
    }

    LLAMA_LOG_INFO(
        "%s: weight pager ready: devices=%zu pages=%d placeholders=%zu tid2eid=%d\n",
        __func__, model.wp_pager->size(), complete_catalog.n_pages(),
        n_placed, n_tid2eid);
    return true;
}

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

const char * llama_load_mode_name(enum llama_load_mode load_mode) {
    switch (load_mode) {
        case LLAMA_LOAD_MODE_NONE:
            return "none";
        case LLAMA_LOAD_MODE_MMAP:
            return "mmap";
        case LLAMA_LOAD_MODE_MLOCK:
            return "mlock";
        case LLAMA_LOAD_MODE_MMAP_MLOCK:
            return "mmap+mlock";
        case LLAMA_LOAD_MODE_DIRECT_IO:
            return "dio";
    }
    GGML_ABORT("fatal error");
}

enum llama_load_mode llama_load_mode_from_str(const char * str) {
    if (std::strcmp(str, "none") == 0)       { return LLAMA_LOAD_MODE_NONE;       }
    if (std::strcmp(str, "mmap") == 0)       { return LLAMA_LOAD_MODE_MMAP;       }
    if (std::strcmp(str, "mlock") == 0)      { return LLAMA_LOAD_MODE_MLOCK;      }
    if (std::strcmp(str, "mmap+mlock") == 0) { return LLAMA_LOAD_MODE_MMAP_MLOCK; }
    if (std::strcmp(str, "dio") == 0)        { return LLAMA_LOAD_MODE_DIRECT_IO;  }
    throw std::invalid_argument(std::string("unknown load mode: ") + str);
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// returns true on success
static bool llama_prepare_model_devices(const llama_model_params & params, llama_model * model) {
    // create list of devices to use with this model
    if (params.devices) {
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            size_t n_devs = 0;
            while (params.devices[n_devs]) {
                n_devs++;
            }
            if (n_devs == 0) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, n_devs);
            for (size_t i = 0; i < n_devs; ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s\n", __func__, i, ggml_backend_dev_name(params.devices[i]));
            }
            model->get_split_state_ud.n_devices = n_devs;
            model->get_split_state_ud.model = model;
            model->devices.push_back({
                true, ggml_backend_meta_device(
                params.devices, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
                model->devices.push_back({false, *dev});
            }
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<llama_device> gpus;
        std::vector<llama_device> igpus;
        std::vector<llama_device> rpc_servers;

        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(ggml_backend_dev_count());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type()) {
                    LLAMA_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    continue;
                }
                devs.push_back(dev);
            }
            if (devs.empty()) {
                LLAMA_LOG_ERROR("%s: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
                return false;
            }

            LLAMA_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, devs.size());
            for (size_t i = 0; i < devs.size(); ++i) {
                LLAMA_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, i, ggml_backend_dev_name(devs[i]), ggml_backend_dev_description(devs[i]));
            }

            GGML_ASSERT(!devs.empty());
            model->get_split_state_ud.n_devices = devs.size();
            model->get_split_state_ud.model     = model;
            gpus.push_back({
                true, ggml_backend_meta_device(
                devs.data(), devs.size(), llama_meta_device_get_split_state, &model->get_split_state_ud)
            });
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                switch (ggml_backend_dev_type(dev)) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                        // skip CPU backends since they are handled separately
                        break;

                    case GGML_BACKEND_DEVICE_TYPE_GPU: {
                        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                        if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                            rpc_servers.push_back({false, dev});
                        } else {
                            // check if there is already a GPU with the same device id
                            ggml_backend_dev_props props;
                            ggml_backend_dev_get_props(dev, &props);
                            auto it = std::find_if(gpus.begin(), gpus.end(), [&props](const llama_device & d) {
                                ggml_backend_dev_props d_props;
                                ggml_backend_dev_get_props(d.dev, &d_props);
                                if (props.device_id && d_props.device_id) {
                                    return strcmp(props.device_id, d_props.device_id) == 0;
                                }
                                return false;
                            });

                            if (it != gpus.end()) {
                                LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                        __func__,
                                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                        props.device_id ? props.device_id : "unknown id",
                                        ggml_backend_dev_name(it->dev), ggml_backend_dev_description(it->dev));
                            } else {
                                gpus.push_back({false, dev});
                            }
                        }
                        break;
                    }

                    case GGML_BACKEND_DEVICE_TYPE_IGPU:
                        if (igpus.empty()) {
                            igpus.push_back({false, dev});
                        }
                        break;
                    case GGML_BACKEND_DEVICE_TYPE_META:
                        GGML_ABORT("fatal error");
                }
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no discrete GPUs were found
        // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
        if (gpus.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE && !model->devices.empty()) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                return false;
            }
            llama_device main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (const auto & dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.load_mode,
            params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get());
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        // Create the weight pager carrier early so load_tensors can collect tensor ptrs into it.
        if (params.weight_paging_enabled) {
            model->weight_pager = std::make_unique<llama_weight_pager>();
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        // Initialize weight pager (pool alloc, page table, fds) after tensors are loaded.
        init_weight_pager(*model, ml, params);

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}

struct llama_model * llama_model_init_from_user(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        struct llama_model_params params) {
    GGML_ASSERT(metadata != nullptr);
    std::string path_model;
    std::vector<std::string> splits = {};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    return llama_model_load_from_file_impl(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, /*file*/ nullptr, params);
}
// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, splits.front(), splits, /*file*/ nullptr, params);
}

struct llama_model * llama_model_load_from_file_ptr(FILE * file, struct llama_model_params params) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }
    std::string path_model;
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, file, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}
