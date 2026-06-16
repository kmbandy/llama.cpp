// occ_dispatch.cpp  (dvgpr_occ Phase 3)
//
// Dynamic-VGPR GEMM-occupancy DE-RISK harness for gfx1201 (RDNA4 / R9700). Phase 3 of
// MAD-293 (MAD-305), built on the MAD-304/Phase-2 raw-PM4 vehicle.
//
// occ_kernel is now a timed WMMA-throughput chain (load A/B once -> KDEPTH iterations of
// NACC independent accumulating WMMAs). We host-time the PM4 submit->EOP-fence interval
// and report TFLOPS, using a KDEPTH=1 differential to cancel launch/memory/atomic/fence
// overhead so the number is the accumulating loop's compute throughput.
//
// Modes:
//   --prong1 : occupancy -> throughput curve. LIGHT kernel (NACC=8, occ_n8_d0.bin),
//              static, sweep RSRC1.VGPRS reservation {80..256} -> occ {~16..5}.
//              GREEN-1 if TFLOPS rises materially toward higher occupancy.
//   --prong2 : dyn-VGPR delivers occupancy over a long fat phase. HEAVY kernel (NACC=16),
//              static (occ_n16_d0.bin, reserve 144) vs dyn (occ_n16_d1.bin, lean 32 ->
//              s_alloc 144) over a KDEPTH sweep. GREEN-2 if dyn >= static (no serialization
//              penalty).
//   (default): correctness A/B -- KDEPTH=1, static & dyn heavy, WMMA==oracle + maxlive.
//
// SUPERVISED: raw PM4 on the headless gfx12 node. outer `timeout` is the hard guard;
// 10 s internal fence timeout; on timeout we leave the (possibly hung) queue for dmesg.
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

static double now_s() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

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
// One dispatch: occ_kernel over nWG single-wave workgroups at the given VGPR
// reservation (static) or lean-32 launch (dyn) and runtime KDEPTH. Host-timed
// submit->fence. Returns the acc0 tile (256 f32) for the oracle and maxlive.
// ---------------------------------------------------------------------------
struct RunResult { bool ok = false; uint32_t maxlive = 0; double secs = 0.0; float D[256]; };

static RunResult run_variant(uint32_t node, const char* isaPath, bool dynvgpr,
                             const uint32_t* fragIn /*128 u32: 64 A then 64 B*/,
                             uint32_t nWG, uint32_t staticVgprField, uint32_t kdepth) {
    RunResult res;

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa   = AllocGpu(node, 0x1000, /*exec*/true,  /*uncached*/false);
    GpuBuf occ   = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);   // [live, maxlive, KDEPTH]
    GpuBuf fin   = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);   // A(256B) then B(256B)
    GpuBuf fout  = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);   // acc0 tile (1024B)
    GpuBuf fence = AllocGpu(node, 0x1000, /*exec*/false, /*uncached*/true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(fin.ptr, fragIn, 128 * sizeof(uint32_t));
    volatile uint32_t* occW   = (volatile uint32_t*)occ.ptr;    // occW[0]=live, occW[1]=maxlive, occW[2]=KDEPTH
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0] = 0; occW[1] = 0; occW[2] = kdepth; *fenceW = 0;

    Ring ring;
    ring.buf    = AllocGpu(node, 0x10000, /*exec*/true, /*uncached*/true);
    ring.dw     = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL,
                            ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa   = (uint64_t)occ.ptr;
    uint64_t finVa   = (uint64_t)fin.ptr;
    uint64_t foutVa  = (uint64_t)fout.ptr;
    uint64_t fenceVa = (uint64_t)fence.ptr;

    uint32_t dims[8] = {0, 0, 0, 32, 1, 1, 0, 0};   // NUM_THREAD_X = 32 -> one wave32 / WG
    uint32_t pgm[6]  = { (uint32_t)shiftedIsa,
                         (uint32_t)(shiftedIsa >> 32) | (g_is_dgpu ? 0u : (1u << 8)),
                         0, 0, 0, 0 };
    // RSRC1.VGPRS: dyn launches lean at 32 VGPR (field 4) then s_alloc grows; static reserves
    // staticVgprField*8 VGPRs for life. Under-reserving below kernel usage = OOB = hang.
    uint32_t rsrc1 = BuildPgmRsrc1(false);
    uint32_t launchField = dynvgpr ? 4u : (staticVgprField & 0x3fu);
    rsrc1 = (rsrc1 & ~0x3fu) | (launchField & 0x3fu);
    // RSRC2: keep TGID_X_EN | TIDIG_COMP_CNT | EXCP and bit6 (dyn); force USER_SGPR 4 -> 6.
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (6u << RSRC2_USER_SGPR_SHIFT);
    uint32_t rsrc[2]    = { rsrc1, rsrc2 };
    uint32_t reslim[1]  = {0};
    uint32_t tmpring[1] = {0};
    uint32_t restart[4] = {0, 0, 0, 0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,  (uint32_t)(occVa >> 32),    // s0:s1 = occ[live,maxlive,KDEPTH]
        (uint32_t)finVa,  (uint32_t)(finVa >> 32),    // s2:s3 = fragIn (A@0, B@256)
        (uint32_t)foutVa, (uint32_t)(foutVa >> 32),   // s4:s5 = fragOut
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    uint32_t dispInit = BuildDispatchInitiator();

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

    // --- timed submit->fence (busy-poll for wall-clock precision) ---
    const double timeoutS = 10.0;
    double t0 = now_s();
    RingSubmit(ring);
    bool done = false;
    while (true) {
        if (*fenceW == FENCE_VALUE) { done = true; break; }
        if (now_s() - t0 > timeoutS) break;
    }
    double t1 = now_s();
    if (!done) {
        fprintf(stderr, "\n*** TIMEOUT (%s, KDEPTH=%u): fence not signalled in %.0f s. live=%u maxlive=%u ***\n",
                isaPath, kdepth, timeoutS, occW[0], occW[1]);
        fprintf(stderr, "    Compute queue may be hung; leaving it for dmesg inspection.\n");
        res.ok = false;
        return res;   // do NOT tear down a hung queue
    }

    res.ok      = true;
    res.maxlive = occW[1];
    res.secs    = t1 - t0;
    unpack_D((const float*)fout.ptr, res.D);

    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(fout); FreeGpu(fin); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---- run reps times, keep the fastest (min secs) ----
struct Timed { bool ok = false; uint32_t maxlive = 0; double secs = 1e30; float D[256]; };
static Timed run_timed(uint32_t node, const char* bin, bool dyn, const uint32_t* fragIn,
                       uint32_t nWG, uint32_t field, uint32_t kdepth, int reps) {
    Timed best;
    for (int r = 0; r < reps; ++r) {
        RunResult rr = run_variant(node, bin, dyn, fragIn, nWG, field, kdepth);
        if (!rr.ok) { best.ok = false; return best; }
        if (rr.secs < best.secs) { best.secs = rr.secs; best.maxlive = rr.maxlive; memcpy(best.D, rr.D, sizeof(best.D)); }
        best.ok = true;
    }
    return best;
}

// loop-only TFLOPS: (work(K) - work(1)) WMMAs over (t_K - t_1), cancelling fixed overhead.
static double tf_diff(uint32_t nWG, uint32_t K, uint32_t nacc, double t_K, double t_1) {
    double dwork = (double)nWG * (double)(K - 1u) * (double)nacc;   // accumulating-loop WMMAs
    double dt = t_K - t_1;
    if (dt <= 0.0) return 0.0;
    return dwork * (2.0 * 16 * 16 * 16) / dt / 1e12;
}

int main(int argc, char** argv) {
    enum { CORRECT, PRONG1, PRONG2 } mode = CORRECT;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--prong1")) mode = PRONG1;
        else if (!strcmp(argv[i], "--prong2")) mode = PRONG2;
    }

    // Test matrices A,B (16x16 e4m3, non-trivial) and the CPU oracle D = A.B.
    uint8_t A[256], B[256]; float C[256] = {0}, Dref[256];
    for (int i = 0; i < 256; ++i) { A[i] = (uint8_t)(0x38 + (i % 3)); B[i] = (uint8_t)(0x38 + (i % 2)); }
    wmma_ref_16x16x16(A, B, C, Dref);
    uint32_t fragIn[128]; pack_A(A, fragIn); pack_B(B, fragIn + 64);
    auto wmma_ok = [&](const float* D) {
        for (int i = 0; i < 256; ++i)
            if (std::fabs(D[i] - Dref[i]) > 1e-3f * std::fabs(Dref[i]) + 1e-3f) return false;
        return true;
    };

    CHECK(hsaKmtOpenKFD());
    uint32_t node = FindGfx1201Node();

    int rc = 0;
    if (mode == PRONG1) {
        // LIGHT kernel (NACC=8): occupancy -> throughput. Reservation floor = 80 (kernel usage).
        const char* BIN = "occ_n8_d0.bin"; const uint32_t NACC = 8, nWG = 8192, K = 16384;
        printf("=== PRONG 1: occupancy -> throughput (LIGHT NACC=8, grid=%u, KDEPTH=%u, static) ===\n", nWG, K);
        Timed corr = run_timed(node, BIN, false, fragIn, nWG, 128/8, 1, 1);
        if (!corr.ok) { fprintf(stderr, "prong1 correctness did not complete.\n"); hsaKmtCloseKFD(); return 3; }
        bool ok = wmma_ok(corr.D);
        printf("  correctness (KDEPTH=1, reserve=128): WMMA %s  maxlive=%u\n", ok ? "OK" : "MISMATCH", corr.maxlive);
        if (!ok) { fprintf(stderr, "prong1 WMMA mismatch; aborting.\n"); hsaKmtCloseKFD(); return 4; }
        const uint32_t reserves[] = {80, 96, 128, 160, 192, 256};
        printf("\n  reserveVGPR  maxlive  occ/SIMD   TFLOPS\n");
        for (uint32_t rv : reserves) {
            Timed tk = run_timed(node, BIN, false, fragIn, nWG, rv/8, K, 3);
            Timed t1 = run_timed(node, BIN, false, fragIn, nWG, rv/8, 1, 3);
            if (!tk.ok || !t1.ok) { fprintf(stderr, "  reserve=%u did not complete; aborting prong1.\n", rv); rc = 3; break; }
            double tf = tf_diff(nWG, K, NACC, tk.secs, t1.secs);
            printf("  %8u     %5u    %6.2f   %7.1f\n", rv, tk.maxlive, tk.maxlive / 128.0, tf);
        }
        printf("\n  GREEN-1 if TFLOPS rises materially toward higher occupancy (lower reserve).\n");
    } else if (mode == PRONG2) {
        // HEAVY kernel (NACC=16): static (reserve 144) vs dyn (lean 32 -> s_alloc 144).
        const uint32_t NACC = 16, nWG = 8192, SFIELD = 144/8;
        const uint32_t Ks[] = {1024, 4096, 8192};
        printf("=== PRONG 2: dyn vs static over a long fat phase (HEAVY NACC=16, grid=%u) ===\n", nWG);
        Timed cs = run_timed(node, "occ_n16_d0.bin", false, fragIn, nWG, SFIELD, 1, 1);
        Timed cd = run_timed(node, "occ_n16_d1.bin", true,  fragIn, nWG, 4,      1, 1);
        if (!cs.ok || !cd.ok) { fprintf(stderr, "prong2 correctness did not complete.\n"); hsaKmtCloseKFD(); return 3; }
        bool sok = wmma_ok(cs.D), dok = wmma_ok(cd.D);
        printf("  correctness (KDEPTH=1): static WMMA %s (maxlive=%u)  dyn WMMA %s (maxlive=%u)\n",
               sok ? "OK" : "MISMATCH", cs.maxlive, dok ? "OK" : "MISMATCH", cd.maxlive);
        if (!sok || !dok) { fprintf(stderr, "prong2 WMMA mismatch; aborting.\n"); hsaKmtCloseKFD(); return 4; }
        printf("\n  KDEPTH   static_TF  static_occ   dyn_TF  dyn_occ   dyn/static\n");
        for (uint32_t K : Ks) {
            Timed sk = run_timed(node, "occ_n16_d0.bin", false, fragIn, nWG, SFIELD, K, 2);
            Timed s1 = run_timed(node, "occ_n16_d0.bin", false, fragIn, nWG, SFIELD, 1, 2);
            Timed dk = run_timed(node, "occ_n16_d1.bin", true,  fragIn, nWG, 4,      K, 2);
            Timed d1 = run_timed(node, "occ_n16_d1.bin", true,  fragIn, nWG, 4,      1, 2);
            if (!sk.ok || !s1.ok || !dk.ok || !d1.ok) { fprintf(stderr, "  KDEPTH=%u did not complete; aborting prong2.\n", K); rc = 3; break; }
            double stf = tf_diff(nWG, K, NACC, sk.secs, s1.secs);
            double dtf = tf_diff(nWG, K, NACC, dk.secs, d1.secs);
            printf("  %6u   %8.1f   %7.2f   %7.1f  %6.2f   %6.2fx\n",
                   K, stf, sk.maxlive / 128.0, dtf, dk.maxlive / 128.0, stf > 0 ? dtf / stf : 0.0);
        }
        printf("\n  GREEN-2 if dyn >= static TFLOPS at realistic KDEPTH (no serialization penalty).\n");
    } else {
        // Default: correctness A/B (regression check on the throughput kernel).
        printf("=== correctness A/B (KDEPTH=1, HEAVY NACC=16) ===\n");
        Timed st = run_timed(node, "occ_n16_d0.bin", false, fragIn, 2048, 144/8, 1, 1);
        Timed dy = run_timed(node, "occ_n16_d1.bin", true,  fragIn, 2048, 4,     1, 1);
        if (!st.ok || !dy.ok) { fprintf(stderr, "correctness did not complete.\n"); hsaKmtCloseKFD(); return 3; }
        bool sok = wmma_ok(st.D), dok = wmma_ok(dy.D);
        printf("  static: WMMA %s  maxlive=%u\n", sok ? "OK" : "MISMATCH", st.maxlive);
        printf("  dyn   : WMMA %s  maxlive=%u\n", dok ? "OK" : "MISMATCH", dy.maxlive);
        rc = (sok && dok) ? 0 : 5;
    }

    hsaKmtCloseKFD();
    return rc;
}
