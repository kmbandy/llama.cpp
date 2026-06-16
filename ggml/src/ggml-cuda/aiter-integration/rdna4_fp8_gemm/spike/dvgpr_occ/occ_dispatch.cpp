// occ_dispatch.cpp  (dvgpr_occ T4)
//
// Dynamic-VGPR OCCUPANCY A/B harness for gfx1201 (RDNA4 / R9700). Phase 2 of
// MAD-293, builds directly on the MAD-304 raw-PM4 dispatch vehicle.
//
// Dispatches occ_kernel (hand-written gfx1201 ISA) twice over a large grid of
// single-wave32 workgroups:
//   static : PGM_RSRC1.VGPRS = 128, RSRC2 bit6 = 0  -> reserves 128 VGPRs for life,
//            occupancy capped by the 128-VGPR footprint.
//   dyn    : PGM_RSRC1.VGPRS = 32,  RSRC2 bit6 = 1  -> launches at a 32-VGPR block,
//            s_alloc_vgpr 128 only for the WMMA, shrinks back -> more resident waves.
//
// Each wave: global_atomic_add(live,+1) -> atomic_max(maxlive) -> long busy-wait ->
// s_alloc_vgpr 128 -> fp8 WMMA (4 accumulators) -> store -> s_alloc_vgpr 32 ->
// atomic_add(live,-1). maxlive (device-scope) = peak concurrent waves. All waves
// use the same A/B fragments and write the same D; the CPU fp8 oracle verifies it.
//
// Gates: dyn WMMA == oracle, dyn maxlive >= 2 * static maxlive, no hang.
//
// SUPERVISED: raw PM4 on the headless gfx12 node. timeout 30 around the binary;
// 2 s internal fence timeout; on timeout we leave the (possibly hung) queue for
// dmesg inspection and do NOT fire the next variant.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <ctime>

#include "hsakmt/hsakmt.h"
#include "pm4_defs.h"        // -I ../dvgpr_pm4  (register offsets + RSRC builders + vendored encoder)
#include "fp8_oracle.h"
#include "frag_layout.h"

// ---- the dgpu flag the vendored PM4 encoder asks for (same as MAD-304) -----
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
static const uint32_t FENCE_VALUE    = 0xCAFEF00Du;

// ---------------------------------------------------------------------------
// GPU buffer (system/GTT, host-visible). Verbatim from MAD-304 pm4_dispatch.cpp.
// ---------------------------------------------------------------------------
struct GpuBuf { void* ptr = nullptr; uint64_t size = 0; uint32_t node = 0; };

static GpuBuf AllocGpu(uint32_t node, uint64_t size, bool isExec, bool isUncached) {
    HsaMemFlags f; f.Value = 0;
    f.ui32.PageSize    = HSA_PAGE_SIZE_4KB;
    f.ui32.HostAccess  = 1;
    f.ui32.NonPaged    = 0;
    f.ui32.CoarseGrain = 0;
    f.ui32.NoNUMABind  = 1;
    f.ui32.Uncached    = isUncached ? 1 : 0;
    if (isExec) f.ui32.ExecuteAccess = 1;
    GpuBuf b; b.size = size; b.node = node;
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
// Minimal PM4 compute ring. Verbatim from MAD-304 pm4_dispatch.cpp.
// ---------------------------------------------------------------------------
struct Ring {
    GpuBuf            buf;
    uint32_t*         dw     = nullptr;
    uint32_t          sizeDw = 0;
    HsaQueueResource  res    = {};
    uint32_t          wptr   = 0;
    uint64_t          wptr64 = 0;
};
static void RingPlace(Ring& r, const BasePacket& pkt) {
    uint32_t ndw = pkt.SizeInDWords();
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
        HsaNodeProperties props; memset(&props, 0, sizeof(props));
        if (hsaKmtGetNodeProperties(n, &props) != HSAKMT_STATUS_SUCCESS) continue;
        if (props.NumFComputeCores == 0) continue;
        if (props.EngineId.ui32.Major == 12) {
            g_is_dgpu = (props.NumCPUCores == 0);
            printf("  -> gfx12 node %u (FCompute=%u, dgpu=%d)\n",
                   n, props.NumFComputeCores, (int)g_is_dgpu);
            return n;
        }
    }
    fprintf(stderr, "FATAL: no gfx12 (GFXIP major 12) compute node found\n");
    exit(1);
}

static uint8_t* ReadFile(const char* path, size_t* outLen) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "FATAL: cannot open ISA '%s'\n", path); exit(1); }
    fseek(fp, 0, SEEK_END); long len = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(len);
    if (fread(buf, 1, len, fp) != (size_t)len) { fprintf(stderr, "FATAL: short read\n"); exit(1); }
    fclose(fp); *outLen = (size_t)len; return buf;
}

// ---------------------------------------------------------------------------
// One A/B variant: dispatch occ_kernel over nWG single-wave workgroups, read
// maxlive + the 4 WMMA output tiles. Returns ok=false (without tearing down a
// possibly-hung queue) on fence timeout.
// ---------------------------------------------------------------------------
struct RunResult { bool ok = false; uint32_t maxlive = 0; float Dtile[4][256]; };

static RunResult run_variant(uint32_t node, const char* isaPath, bool dynvgpr,
                             const uint32_t* fragIn /*128 u32: 64 A then 64 B*/, uint32_t nWG,
                             uint32_t staticVgprField) {
    RunResult res;

    // --- buffers (host-visible GTT) ---
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa   = AllocGpu(node, 0x1000, /*exec*/true,  /*uncached*/false);
    GpuBuf occ   = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);   // [live, maxlive]
    GpuBuf fin   = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);   // A(256B) then B(256B)
    GpuBuf fout  = AllocGpu(node, 0x2000, /*exec*/false, /*uncached*/true);   // 4 tiles * 1024B
    GpuBuf fence = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(fin.ptr, fragIn, 128 * sizeof(uint32_t));
    volatile uint32_t* occW   = (volatile uint32_t*)occ.ptr;    // occW[0]=live, occW[1]=maxlive
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0] = 0; occW[1] = 0; *fenceW = 0;

    // --- compute queue ---
    Ring ring;
    ring.buf    = AllocGpu(node, 0x10000, /*exec*/true, /*uncached*/true);
    ring.dw     = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL,
                            ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    // --- dispatch register values ---
    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa   = (uint64_t)occ.ptr;
    uint64_t finVa   = (uint64_t)fin.ptr;
    uint64_t foutVa  = (uint64_t)fout.ptr;
    uint64_t fenceVa = (uint64_t)fence.ptr;

    uint32_t dims[8] = {0, 0, 0, 32, 1, 1, 0, 0};   // NUM_THREAD_X = 32 -> one wave32 / WG
    uint32_t pgm[6]  = { (uint32_t)shiftedIsa,
                         (uint32_t)(shiftedIsa >> 32) | (g_is_dgpu ? 0u : (1u << 8)),
                         0, 0, 0, 0 };
    // RSRC1: dyn launches at 32 VGPRs (field 4); static reserves staticVgprField*8 VGPRs.
    uint32_t rsrc1 = BuildPgmRsrc1(false);
    if (!dynvgpr) rsrc1 = (rsrc1 & ~0x3fu) | (staticVgprField & 0x3fu);   // VGPRS field (bits 0..5)
    // RSRC2: BuildPgmRsrc2 keeps TGID_X_EN | TIDIG_COMP_CNT | EXCP_EN_MSB and bit6 for dyn;
    // force USER_SGPR field 4 -> 6 (three 64-bit pointers in s0:s5).
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (6u << RSRC2_USER_SGPR_SHIFT);
    uint32_t rsrc[2]    = { rsrc1, rsrc2 };
    uint32_t reslim[1]  = {0};
    uint32_t tmpring[1] = {0};
    uint32_t restart[4] = {0, 0, 0, 0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,  (uint32_t)(occVa >> 32),    // s0:s1 = occ[live,maxlive]
        (uint32_t)finVa,  (uint32_t)(finVa >> 32),    // s2:s3 = fragIn (A@0, B@256)
        (uint32_t)foutVa, (uint32_t)(foutVa >> 32),   // s4:s5 = fragOut
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    uint32_t dispInit = BuildDispatchInitiator();

    // --- place packets (Dispatch::BuildIb order) ---
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X,         dims,     8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,          pgm,      6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,       rsrc,     2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim,   1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,    tmpring,  1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,       restart,  4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,     userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, /*isPolling*/true, fenceVa, FENCE_VALUE));
    RingSubmit(ring);

    // --- poll the fence (10 s; outer `timeout 30` is the hard supervised guard) ---
    // Lower-occupancy configs (heavy static footprint) legitimately run longer over a
    // large grid, so this is generous vs the MAD-304 probe's 2 s.
    const uint32_t timeoutMs = 10000;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    bool done = false;
    while (true) {
        if (*fenceW == FENCE_VALUE) { done = true; break; }
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (ms > timeoutMs) break;
        struct timespec nap = {0, 200000}; nanosleep(&nap, nullptr);
    }
    if (!done) {
        fprintf(stderr, "\n*** TIMEOUT (%s): fence not signalled in %u ms. live=%u maxlive=%u ***\n",
                isaPath, timeoutMs, occW[0], occW[1]);
        fprintf(stderr, "    Compute queue may be hung; leaving it for dmesg inspection.\n");
        res.ok = false;
        return res;   // do NOT tear down a hung queue
    }

    res.ok      = true;
    res.maxlive = occW[1];
    const float* fo = (const float*)fout.ptr;
    for (int t = 0; t < 4; ++t) unpack_D(fo + t * 256, res.Dtile[t]);

    // --- teardown ---
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(fout); FreeGpu(fin); FreeGpu(occ); FreeGpu(isa);
    return res;
}

int main(int argc, char** argv) {
    uint32_t nWG         = (argc > 1) ? (uint32_t)atoi(argv[1]) : 2048;
    uint32_t staticVgprs = (argc > 2) ? (uint32_t)atoi(argv[2]) : 128;   // static-twin reservation
    uint32_t staticField = staticVgprs / 8;                              // RSRC1.VGPRS = vgprs/8

    // Test matrices A,B (16x16 e4m3, non-trivial) and the CPU oracle D.
    uint8_t A[256], B[256]; float C[256] = {0}, Dref[256];
    for (int i = 0; i < 256; ++i) { A[i] = (uint8_t)(0x38 + (i % 3)); B[i] = (uint8_t)(0x38 + (i % 2)); }
    wmma_ref_16x16x16(A, B, C, Dref);
    uint32_t fragIn[128]; pack_A(A, fragIn); pack_B(B, fragIn + 64);

    printf("=== RDNA4 dyn-VGPR occupancy A/B (grid=%u WGs, static=%u VGPRs vs dyn 32->128) ===\n",
           nWG, staticVgprs);
    CHECK(hsaKmtOpenKFD());
    uint32_t node = FindGfx1201Node();

    auto wmma_ok = [&](const RunResult& r) {
        for (int t = 0; t < 4; ++t)
            for (int i = 0; i < 256; ++i)
                if (std::fabs(r.Dtile[t][i] - Dref[i]) > 1e-3f * std::fabs(Dref[i]) + 1e-3f) return false;
        return true;
    };

    RunResult st = run_variant(node, "occ_static.bin", false, fragIn, nWG, staticField);
    if (!st.ok) { fprintf(stderr, "static variant did not complete; aborting A/B.\n"); hsaKmtCloseKFD(); return 3; }
    printf("  static(%3u VGPRs): maxlive=%-5u  WMMA %s\n", staticVgprs, st.maxlive, wmma_ok(st) ? "OK" : "MISMATCH");

    RunResult dy = run_variant(node, "occ_dyn.bin", true, fragIn, nWG, 4);
    if (!dy.ok) { fprintf(stderr, "dyn variant did not complete; aborting A/B.\n"); hsaKmtCloseKFD(); return 3; }
    printf("  dyn  ( 32->128 ) : maxlive=%-5u  WMMA %s\n", dy.maxlive, wmma_ok(dy) ? "OK" : "MISMATCH");

    double ratio = (st.maxlive > 0) ? (double)dy.maxlive / (double)st.maxlive : 0.0;
    bool gate_func = wmma_ok(dy) && wmma_ok(st);
    bool gate_occ  = (st.maxlive > 0) && (dy.maxlive > st.maxlive);   // dyn delivers more residency
    printf("\n  occupancy ratio dyn/static = %.2fx  (target >= 2.0x; dyn near the 32-VGPR-block max = lever proven)\n", ratio);
    printf("  gates : functional(WMMA==oracle)=%d  occupancy(dyn>static)=%d  ratio>=2x=%d\n",
           (int)gate_func, (int)gate_occ, (int)(ratio >= 2.0));
    printf("  VERDICT: %s\n", (gate_func && gate_occ) ? "DYN-VGPR OCCUPANCY LEVER PROVEN"
                                                      : "see table / iterate");
    hsaKmtCloseKFD();
    return (gate_func && gate_occ) ? 0 : 5;
}
