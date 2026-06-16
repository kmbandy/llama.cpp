// pm4_dispatch.cpp  (MAD-304 T2)
//
// Raw-PM4 dynamic-VGPR unlock harness for gfx1201 (RDNA4 / R9700).
//
// Launches a tiny hand-written compute wave (probe.bin) on a KFD compute queue
// via a raw PM4 indirect-buffer-style packet stream placed directly in the
// ring. Because WE write COMPUTE_PGM_RSRC2 via SET_SH_REG (the kfdtest / PAL /
// Vulkan path), the CP consumes it verbatim and the MES firmware is out of the
// loop -- so unlike HIP/AQL, our COMPUTE_PGM_RSRC2 bit 6 (== DYNAMIC_VGPR on
// GFX12) actually reaches the dispatch. The probe reads its own wave
// STATUS[30] (DYN_VGPR_EN) and stores it to an output buffer.
//
//   baseline (no flag) : RSRC2 bit6 clear -> MUST read DYN_VGPR_EN = 0  (proves
//                        the PM4 vehicle is correct before changing anything)
//   --dynvgpr          : RSRC2 bit6 set   -> read DYN_VGPR_EN; 1 == dynamic
//                        VGPR armed on RDNA4 compute (the goal).
//
// Dispatch/queue mechanics mirror kfdtest BaseQueue/PM4Queue/Dispatch verbatim
// (ROCm/ROCR-Runtime @ ba56a24c); packet encoders are the vendored kfdtest
// PM4Packet.cpp. SUPERVISED: a malformed dispatch can hang the compute queue.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <ctime>

#include "hsakmt/hsakmt.h"
#include "pm4_defs.h"

// ---- the dgpu flag the vendored PM4 encoder asks for ----------------------
static bool g_is_dgpu = true;
bool hsakmt_is_dgpu() { return g_is_dgpu; }

#define CHECK(call) do {                                                       \
    HSAKMT_STATUS _s = (call);                                                 \
    if (_s != HSAKMT_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "FATAL: %s -> status %d  (%s:%d)\n",                   \
                #call, (int)_s, __FILE__, __LINE__);                          \
        exit(2);                                                               \
    }                                                                          \
} while (0)

static const uint32_t CMD_NOP_TYPE_3 = 0xFFFF1002u;   // BaseQueue.hpp:95
static const uint32_t FENCE_VALUE     = 0xCAFEF00Du;

// ---------------------------------------------------------------------------
// GPU memory buffer: system/GTT, host-visible, mapped to the target node.
// Flag recipe mirrors kfdtest HsaMemoryBuffer (KFDTestUtil.cpp:286-326) for the
// isLocal=false path.
// ---------------------------------------------------------------------------
struct GpuBuf {
    void*    ptr  = nullptr;
    uint64_t size = 0;
    uint32_t node = 0;
};

static GpuBuf AllocGpu(uint32_t node, uint64_t size, bool isExec, bool isUncached) {
    HsaMemFlags f;
    f.Value = 0;
    f.ui32.PageSize    = HSA_PAGE_SIZE_4KB;
    f.ui32.HostAccess  = 1;
    f.ui32.NonPaged    = 0;
    f.ui32.CoarseGrain = 0;
    f.ui32.NoNUMABind  = 1;
    f.ui32.Uncached    = isUncached ? 1 : 0;
    if (isExec)
        f.ui32.ExecuteAccess = 1;

    GpuBuf b;
    b.size = size;
    b.node = node;
    CHECK(hsaKmtAllocMemory(node, size, f, &b.ptr));
    memset(b.ptr, 0, size);

    HsaMemMapFlags mf = {0};
    CHECK(hsaKmtMapMemoryToGPUNodes(b.ptr, size, nullptr, mf, 1, &node));
    return b;
}

static void FreeGpu(GpuBuf& b) {
    if (!b.ptr) return;
    hsaKmtUnmapMemoryToGPU(b.ptr);
    hsaKmtFreeMemory(b.ptr, b.size);
    b.ptr = nullptr;
}

// ---------------------------------------------------------------------------
// Minimal PM4 compute ring. Mirrors BaseQueue::PlacePacket +
// PM4Queue::SubmitPacket for FAMILY >= AI (64-bit wptr/doorbell).
// ---------------------------------------------------------------------------
struct Ring {
    GpuBuf            buf;
    uint32_t*         dw       = nullptr;  // ring as dword array
    uint32_t          sizeDw   = 0;
    HsaQueueResource  res      = {};
    uint32_t          wptr     = 0;        // wrapped, dwords
    uint64_t          wptr64   = 0;        // absolute, dwords
};

static void RingPlace(Ring& r, const BasePacket& pkt) {
    uint32_t ndw = pkt.SizeInDWords();
    // Wraparound: pad to buffer end with NOPs (BaseQueue::PlacePacket). With a
    // 64KB ring and a ~60-dword IB starting at 0 this never triggers, but keep
    // it correct.
    if (r.wptr + ndw >= r.sizeDw) {
        while (r.wptr + ndw > r.sizeDw) {
            r.dw[r.wptr] = CMD_NOP_TYPE_3;
            r.wptr = (r.wptr + 1) % r.sizeDw;
            r.wptr64++;
        }
    }
    memcpy(r.dw + r.wptr, pkt.GetPacket(), ndw * sizeof(uint32_t));
    r.wptr   = (r.wptr + ndw) % r.sizeDw;
    r.wptr64 += ndw;
}

static void RingSubmit(Ring& r) {
    // FAMILY_GFX12 >= FAMILY_AI -> 64-bit wptr + doorbell (PM4Queue::SubmitPacket)
    __sync_synchronize();
    *r.res.Queue_write_ptr_aql = r.wptr64;
    __sync_synchronize();
    *r.res.Queue_DoorBell_aql  = r.wptr64;
}

// ---------------------------------------------------------------------------
static uint32_t FindGfx1201Node() {
    HsaSystemProperties sys;
    CHECK(hsaKmtAcquireSystemProperties(&sys));
    for (uint32_t n = 0; n < sys.NumNodes; ++n) {
        HsaNodeProperties props;
        memset(&props, 0, sizeof(props));
        if (hsaKmtGetNodeProperties(n, &props) != HSAKMT_STATUS_SUCCESS)
            continue;
        if (props.NumFComputeCores == 0)
            continue;  // CPU-only node
        printf("  node %u: GFXIP %u.%u.%u  FCompute=%u  CPUcores=%u\n",
               n, props.EngineId.ui32.Major, props.EngineId.ui32.Minor,
               props.EngineId.ui32.Stepping, props.NumFComputeCores,
               props.NumCPUCores);
        if (props.EngineId.ui32.Major == 12) {   // gfx12xx -> gfx1201 R9700
            g_is_dgpu = (props.NumCPUCores == 0);
            return n;
        }
    }
    fprintf(stderr, "FATAL: no gfx12 (GFXIP major 12) compute node found\n");
    exit(1);
}

static uint8_t* ReadFile(const char* path, size_t* outLen) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "FATAL: cannot open ISA '%s'\n", path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(len);
    if (fread(buf, 1, len, fp) != (size_t)len) { fprintf(stderr, "FATAL: short read\n"); exit(1); }
    fclose(fp);
    *outLen = (size_t)len;
    return buf;
}

int main(int argc, char** argv) {
    bool dynVgpr = false;
    bool priv    = false;
    const char* isaPath = "probe.bin";
    uint32_t timeoutMs = 2000;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--dynvgpr")) dynVgpr = true;
        else if (!strcmp(argv[i], "--priv"))    priv    = true;
        else if (!strcmp(argv[i], "--isa") && i + 1 < argc) isaPath = argv[++i];
        else if (!strcmp(argv[i], "--timeout-ms") && i + 1 < argc) timeoutMs = (uint32_t)atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--dynvgpr] [--priv] [--isa probe.bin] [--timeout-ms N]\n", argv[0]); return 1; }
    }

    printf("=== RDNA4 raw-PM4 dynamic-VGPR probe ===\n");
    printf("  mode        : %s\n", dynVgpr ? "LIFT (RSRC2 bit6 = DYNAMIC_VGPR set)"
                                           : "BASELINE (RSRC2 bit6 clear)");
    printf("  priv        : %d\n", (int)priv);
    uint32_t rsrc1 = BuildPgmRsrc1(priv);
    uint32_t rsrc2 = BuildPgmRsrc2(dynVgpr);
    printf("  PGM_RSRC1   : 0x%08x\n", rsrc1);
    printf("  PGM_RSRC2   : 0x%08x  (DYNAMIC_VGPR bit6 = %d)\n",
           rsrc2, (rsrc2 & RSRC2_DYNAMIC_VGPR_MASK) ? 1 : 0);

    CHECK(hsaKmtOpenKFD());

    printf("  scanning topology:\n");
    uint32_t node = FindGfx1201Node();
    printf("  -> using gfx12 node %u (dgpu=%d)\n", node, (int)g_is_dgpu);

    // --- buffers ---
    size_t isaLen = 0;
    uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa   = AllocGpu(node, 0x1000, /*exec*/true,  /*uncached*/false);
    GpuBuf out   = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);
    GpuBuf fence = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);
    memcpy(isa.ptr, isaBytes, isaLen);
    free(isaBytes);
    volatile uint32_t* outWord   = (volatile uint32_t*)out.ptr;
    volatile uint32_t* fenceWord = (volatile uint32_t*)fence.ptr;
    *outWord   = 0xDEADBEEF;   // sentinel: shader must overwrite with 0/1
    *fenceWord = 0;

    // --- compute queue ---
    Ring ring;
    ring.buf    = AllocGpu(node, 0x10000, /*exec*/true, /*uncached*/true);  // 64KB ring
    ring.dw     = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100 /*percent*/,
                            HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size,
                            nullptr, &ring.res));
    printf("  queue created: id=%llu\n", (unsigned long long)ring.res.QueueId);

    // --- build the dispatch register values (mirror Dispatch::BuildIb) ---
    uint64_t isaVa      = (uint64_t)isa.ptr;
    uint64_t shiftedIsa = isaVa >> 8;
    uint64_t outVa      = (uint64_t)out.ptr;
    uint64_t fenceVa    = (uint64_t)fence.ptr;

    uint32_t dims[8]    = {0, 0, 0, 1, 1, 1, 0, 0};  // START_X/Y/Z, NUM_THREAD_X/Y/Z, PIPESTAT, PERFCOUNT
    uint32_t pgm[6]     = {  // gfx9+ layout; scratch_base = 0
        (uint32_t)shiftedIsa,
        (uint32_t)(shiftedIsa >> 32) | (g_is_dgpu ? 0u : (1u << 8)),
        0, 0, 0, 0
    };
    uint32_t rsrc[2]    = { rsrc1, rsrc2 };
    uint32_t reslim[1]  = {0};
    uint32_t tmpring[1] = {0};
    uint32_t restart[4] = {0, 0, 0, 0};
    uint32_t userdata[16] = {
        (uint32_t)outVa, (uint32_t)(outVa >> 32),  // USER_DATA_0/1 -> s0:s1 = out ptr
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    uint32_t dispInit = BuildDispatchInitiator();

    // --- place packets directly into the ring (Dispatch::BuildIb order) ---
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X,         dims,     8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,          pgm,      6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,       rsrc,     2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim,   1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,    tmpring,  1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,       restart,  4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,     userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(1, 1, 1, dispInit));
    // EOP release-mem fence: writes FENCE_VALUE to fenceVa after the dispatch
    // completes and caches flush (NV gcr_cntl). isPolling -> no interrupt.
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, /*isPolling*/true, fenceVa, FENCE_VALUE));

    printf("  IB built: %u dwords. Ringing doorbell...\n", (unsigned)ring.wptr64);
    RingSubmit(ring);

    // --- wait on the fence (CPU poll with timeout) ---
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    bool done = false;
    while (true) {
        if (*fenceWord == FENCE_VALUE) { done = true; break; }
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (ms > timeoutMs) break;
        struct timespec nap = {0, 200000};  // 0.2ms
        nanosleep(&nap, nullptr);
    }

    int rc;
    if (!done) {
        fprintf(stderr, "\n*** TIMEOUT: fence not signalled within %u ms ***\n", timeoutMs);
        fprintf(stderr, "    The compute queue may be hung. fence=0x%08x out=0x%08x\n",
                *fenceWord, *outWord);
        fprintf(stderr, "    Leaving the queue as-is for inspection; check dmesg for a ring/CP reset.\n");
        rc = 3;  // skip teardown to avoid blocking on a hung queue
        return rc;
    }

    uint32_t dynEn = *outWord;
    printf("\n  fence signalled. DYN_VGPR_EN (STATUS[30]) = %u\n", dynEn);
    if (!dynVgpr) {
        printf("  [BASELINE] expected 0 (proves PM4 vehicle). %s\n",
               dynEn == 0 ? "PASS" : "UNEXPECTED");
        rc = (dynEn == 0) ? 0 : 4;
    } else {
        if (dynEn == 1) {
            printf("  [LIFT] *** DYNAMIC VGPR ARMED ON RDNA4 COMPUTE *** \n");
            rc = 0;
        } else {
            printf("  [LIFT] bit accepted, wave did NOT enter dyn mode (DYN_VGPR_EN=0, no hang).\n");
            printf("         -> iterate rsrc1 block encoding / SQ_DYN_VGPR interaction.\n");
            rc = 5;
        }
    }

    // --- teardown ---
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf);
    FreeGpu(fence);
    FreeGpu(out);
    FreeGpu(isa);
    hsaKmtCloseKFD();
    return rc;
}
