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
#include <vector>
#include <random>
#include <cstdarg>
#include <unistd.h>

#include "hsakmt/hsakmt.h"
#include "pm4_defs.h"        // -I ../dvgpr_pm4  (register offsets + RSRC builders + vendored encoder)
#include "pm4_perf.h"        // GL2C perfcounter encoders + verified gfx12 register map (in-ring DRAM meter)
#include "fp8_oracle.h"
#include "frag_layout.h"

// ---- the dgpu flag the vendored PM4 encoder asks for (same as MAD-304) -----
static bool g_is_dgpu = true;
bool hsakmt_is_dgpu() { return g_is_dgpu; }
static bool g_gl2c = false;   // --gl2c: bracket the dispatch with in-ring GL2C perfcounters (byte-exact DRAM read traffic)

// ---- crash-survivable progress log: write a line to a disk file + fflush + fsync BEFORE each GPU
// dispatch, so a hard hang/reboot leaves the last "STARTING ..." line ON DISK = exactly which config
// wedged the queue. Also echoes to stdout. Open g_prog before the run; nullptr = stdout only. ----
static FILE* g_prog = nullptr;
static void prog(const char* fmt, ...) {
    va_list ap; char line[512];
    va_start(ap, fmt); vsnprintf(line, sizeof line, fmt, ap); va_end(ap);
    printf("%s\n", line);
    if (g_prog) { fprintf(g_prog, "%s\n", line); fflush(g_prog); fsync(fileno(g_prog)); }
}

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
struct GpuBuf { void* ptr = nullptr; uint64_t size = 0; uint32_t node = 0; bool vram = false; };

static GpuBuf AllocGpu(uint32_t node, uint64_t size, bool isExec, bool isUncached, bool deviceLocal=false) {
    HsaMemFlags f; f.Value = 0;
    f.ui32.PageSize    = HSA_PAGE_SIZE_4KB;
    f.ui32.HostAccess  = 1;
    f.ui32.NonPaged    = deviceLocal ? 1 : 0;   // 1 = device-local VRAM (640 GB/s); 0 = system/GTT (PCIe ~25 GB/s)
    f.ui32.CoarseGrain = deviceLocal ? 1 : 0;   // VRAM is coarse-grained
    f.ui32.NoNUMABind  = 1;
    f.ui32.Uncached    = isUncached ? 1 : 0;
    if (isExec) f.ui32.ExecuteAccess = 1;
    GpuBuf b; b.size = size; b.node = node; b.vram = deviceLocal;
    CHECK(hsaKmtAllocMemory(node, size, f, &b.ptr));
    memset(b.ptr, 0, size);
    HsaMemMapFlags mf = {0};
    CHECK(hsaKmtMapMemoryToGPUNodes(b.ptr, size, nullptr, mf, 1, &node));
    return b;
}
// Bytes to map for a kernel .text image: page-round the ISA, then add ONE TRAILING GUARD PAGE.
//   The SQC instruction prefetcher reads ahead past s_endpgm. With an exactly-page-rounded mapping, a bin
//   whose .text happens to end near the page boundary makes that prefetch walk into the next, UNMAPPED page
//   -> [gfxhub] SQC(inst) page fault (RW=0, MAPPING_ERROR=1) -> MES cannot REMOVE_QUEUE -> MODE1 brick.
//   Whether a given build bricks is a lottery on .text size. Measured 2026-07-12 (gfx1201):
//     16316B bin (68B slack in a 16KiB mapping) -> BRICKED;  12568/13136/17140B (3+KiB slack) -> clean.
//   AllocGpu memsets the whole allocation, so the guard page is mapped and zero-filled.
static inline uint64_t IsaMapBytes(size_t isaLen) {
    return (((uint64_t)isaLen + 0xFFFull) & ~0xFFFull) + 0x1000ull;
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
struct RunResult { bool ok = false; uint32_t maxlive = 0; uint32_t total = 0; uint64_t wall = 0; double secs = 0.0; float D[256]; };

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
    volatile uint32_t* occW   = (volatile uint32_t*)occ.ptr;    // [0]=live [1]=maxlive [2]=min(start) [3]=max(end)
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0] = 0; occW[1] = 0; occW[2] = 0xFFFFFFFFu; occW[3] = 0; occW[4] = 0; *fenceW = 0;
    // KDEPTH travels in a user SGPR (s6), not memory: the scalar K-cache is not invalidated by
    // AcquireMem, so a memory KDEPTH reads stale across dispatches (buffer VAs get reused).

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
    // RSRC2: keep TGID_X_EN | TIDIG_COMP_CNT | EXCP and bit6 (dyn); force USER_SGPR 4 -> 7
    // (3 pointers in s0:s5 + KDEPTH in s6; TGID_X then lands in s7, leaving s8/s9 free scratch).
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (7u << RSRC2_USER_SGPR_SHIFT);
    uint32_t rsrc[2]    = { rsrc1, rsrc2 };
    uint32_t reslim[1]  = {0};
    uint32_t tmpring[1] = {0};
    uint32_t restart[4] = {0, 0, 0, 0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,  (uint32_t)(occVa >> 32),    // s0:s1 = occ[live,maxlive]
        (uint32_t)finVa,  (uint32_t)(finVa >> 32),    // s2:s3 = fragIn (A@0, B@256)
        (uint32_t)foutVa, (uint32_t)(foutVa >> 32),   // s4:s5 = fragOut
        kdepth,                                       // s6     = KDEPTH (runtime loop count)
        0, 0, 0, 0, 0, 0, 0, 0, 0
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
    // DISPATCH_DIRECT DIM_X is in THREADS (not workgroups): one wave32/WG -> DIM_X = nWG*32.
    // (Passing nWG directly only launched nWG/32 waves -- the 32x work over-count bug.)
    RingPlace(ring, PM4DispatchDirectPacket(nWG * 32u, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, /*isPolling*/true, fenceVa, FENCE_VALUE));

    // --- submit, then wait for TRUE completion ---
    // The EOP fence can fire before all waves retire on this raw-PM4 path, which truncates the
    // in-kernel span counters. The live counter (occ[0]: +1 at admit, -1 at exit) returns to 0
    // only once every wave has retired -> that, together with the fence, is the real completion.
    const double timeoutS = 10.0;
    double t0 = now_s();
    RingSubmit(ring);
    bool done = false, admitted = false;
    uint32_t lastEnd = 0; double lastEndChange = t0;
    while (true) {
        double now = now_s();
        if (occW[1] > 0) admitted = true;                 // some wave reached the counter
        uint32_t end = occW[3];
        if (end != lastEnd) { lastEnd = end; lastEndChange = now; }   // track max(end) advancing
        bool fenceFired = (*fenceW == FENCE_VALUE);
        // True completion: admitted, fence fired, live drained, AND max(end) stable for 25 ms
        // (guards against transient live==0 between batches truncating the span at scale).
        if (admitted && occW[0] == 0 && fenceFired && end != 0 && (now - lastEndChange) > 0.025) { done = true; break; }
        if (now - t0 > timeoutS) break;
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
    res.total   = occW[4];   // total waves that actually launched (vs nWG requested)
    {   // whole-dispatch loop wall span in GPU clock cycles = max(end) - min(start) over all waves
        uint32_t gs = occW[2], ge = occW[3];
        res.wall = (ge >= gs) ? (uint64_t)(ge - gs)
                              : ((uint64_t)ge + 0x100000000ull - (uint64_t)gs);   // 32-bit wrap
    }
    res.secs    = t1 - t0;
    unpack_D((const float*)fout.ptr, res.D);

    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(fout); FreeGpu(fin); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---- run reps times, keep the best (min host completion time = peak throughput) ----
// host secs is now robust: we gate completion on live==0 (all waves retired), so t1-t0 is the
// true kernel duration. We also carry the in-kernel span for cross-check.
struct Timed { bool ok = false; uint32_t maxlive = 0; uint32_t total = 0; uint64_t wall = ~0ull; double secs = 1e30; float D[256]; };
static Timed run_timed(uint32_t node, const char* bin, bool dyn, const uint32_t* fragIn,
                       uint32_t nWG, uint32_t field, uint32_t kdepth, int reps) {
    Timed best; bool have = false;
    for (int r = 0; r < reps; ++r) {
        RunResult rr = run_variant(node, bin, dyn, fragIn, nWG, field, kdepth);
        if (!rr.ok) { best.ok = false; return best; }
        if (!have || rr.wall < best.wall) { best.secs = rr.secs; best.wall = rr.wall; best.maxlive = rr.maxlive; best.total = rr.total; memcpy(best.D, rr.D, sizeof(best.D)); have = true; }
        best.ok = true;
    }
    return best;
}
// Throughput from host completion time (robust via live==0 gating).
static double tf_host(uint32_t nWG, uint32_t K, uint32_t nacc, double secs) {
    if (secs <= 0) return 0.0;
    return (double)nWG * (double)(K - 1u) * (double)nacc * (2.0 * 16 * 16 * 16) / secs / 1e12;
}

// Throughput = (whole-grid loop WMMAs) / (whole-dispatch wall span). nWG*(K-1)*NACC WMMAs in
// (wall/freq) seconds. Bounded by the matrix-unit ceiling by construction (total work / total wall).
static double tf_span(uint32_t nWG, uint32_t K, uint32_t nacc, uint64_t wall, double freq_hz) {
    if (wall == 0) return 0.0;
    double work = (double)nWG * (double)(K - 1u) * (double)nacc;
    return work * (2.0 * 16 * 16 * 16) * freq_hz / (double)wall / 1e12;
}

// ---------------------------------------------------------------------------
// MICRO-BATCH dynamic-queue dispatch: launch nWaves PERSISTENT waves that pull output-tiles
// from a global atomic counter (occ[nextTile@20]) and process them until totalTiles drains.
// Per tile the dyn kernel s_alloc-grows to the accumulator footprint, computes, ships C[ti],
// and s_alloc-shrinks. Validates every tile == kdepth*Dref. C buffer = totalTiles*1024 B.
// ---------------------------------------------------------------------------
struct MbResult { bool ok=false; uint32_t maxlive=0, total=0, okTiles=0, nTiles=0; uint64_t wall=0; double secs=0; };
static MbResult run_microbatch(uint32_t node, const char* isaPath, bool dynvgpr,
                               const uint32_t* fragIn, uint32_t nWaves, uint32_t staticVgprField,
                               uint32_t kdepth, uint32_t totalTiles, const float* Dref) {
    MbResult res; res.nTiles = totalTiles;
    size_t isaLen=0; uint8_t* isaBytes=ReadFile(isaPath,&isaLen);
    GpuBuf isa  = AllocGpu(node, 0x1000, true,  false);
    GpuBuf occ  = AllocGpu(node, 0x1000, false, true);
    GpuBuf fin  = AllocGpu(node, 0x1000, false, true);
    uint64_t cbytes = ((uint64_t)totalTiles*1024 + 0xFFF) & ~0xFFFull;
    GpuBuf C    = AllocGpu(node, cbytes, false, true);
    GpuBuf fence= AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(fin.ptr, fragIn, 128*sizeof(uint32_t));
    volatile uint32_t* occW=(volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW=(volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;  // occ[5]=nextTile

    Ring ring; ring.buf=AllocGpu(node,0x10000,true,true); ring.dw=(uint32_t*)ring.buf.ptr;
    ring.sizeDw=(uint32_t)(ring.buf.size/sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node,HSA_QUEUE_COMPUTE,100,HSA_QUEUE_PRIORITY_NORMAL,ring.buf.ptr,ring.buf.size,nullptr,&ring.res));

    uint64_t shiftedIsa=((uint64_t)isa.ptr)>>8;
    uint64_t occVa=(uint64_t)occ.ptr, finVa=(uint64_t)fin.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8]={0,0,0,32,1,1,0,0};
    uint32_t pgm[6]={(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1=BuildPgmRsrc1(false);
    uint32_t launchField=dynvgpr?4u:(staticVgprField&0x3fu);
    rsrc1=(rsrc1&~0x3fu)|(launchField&0x3fu);
    uint32_t rsrc2=(BuildPgmRsrc2(dynvgpr)&~0x3eu)|(8u<<RSRC2_USER_SGPR_SHIFT);   // 8 user SGPRs (s0..s7)
    uint32_t rsrc[2]={rsrc1,rsrc2};
    uint32_t reslim[1]={0},tmpring[1]={0},restart[4]={0,0,0,0};
    uint32_t userdata[16]={
        (uint32_t)occVa,(uint32_t)(occVa>>32),       // s0:s1 occ
        (uint32_t)finVa,(uint32_t)(finVa>>32),       // s2:s3 fragIn
        (uint32_t)cVa, (uint32_t)(cVa>>32),          // s4:s5 C base
        kdepth, totalTiles, 0,0,0,0,0,0,0,0 };       // s6 KDEPTH, s7 TOTAL_TILES
    uint32_t dispInit=BuildDispatchInitiator();
    RingPlace(ring,PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_START_X,dims,8));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,pgm,6));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,rsrc,2));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS,reslim,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,tmpring,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,restart,4));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,userdata,16));
    RingPlace(ring,PM4DispatchDirectPacket(nWaves*32u,1,1,dispInit));            // nWaves persistent wave32
    RingPlace(ring,PM4ReleaseMemoryPacket(FAMILY_GFX12,true,fenceVa,FENCE_VALUE));

    const double timeoutS=20.0;
    double t0=now_s(); RingSubmit(ring);
    bool done=false,admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while(true){ double now=now_s();
        if(occW[1]>0) admitted=true;
        uint32_t end=occW[3]; if(end!=lastEnd){lastEnd=end;lastEndChange=now;}
        bool ff=(*fenceW==FENCE_VALUE);
        if(admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025){done=true;break;}
        if(now-t0>timeoutS) break;
    }
    double t1=now_s();
    if(!done){ fprintf(stderr,"\n*** MB TIMEOUT (%s): live=%u maxlive=%u nextTile=%u (queue may be hung) ***\n",
                       isaPath,occW[0],occW[1],occW[5]); res.ok=false; return res; }
    res.ok=true; res.maxlive=occW[1]; res.total=occW[4];
    { uint32_t gs=occW[2],ge=occW[3]; res.wall=(ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.secs=t1-t0;
    const float* Cf=(const float*)C.ptr; uint32_t okc=0;
    for(uint32_t t=0;t<totalTiles;t++){ float D[256]; unpack_D(Cf+(size_t)t*256, D);
        bool good=true; for(int i=0;i<256;i++){ float want=(float)kdepth*Dref[i];
            if(std::fabs(D[i]-want) > 5e-3f*std::fabs(want)+1e-2f){good=false;break;} }
        if(good) okc++; }
    res.okTiles=okc;
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(C);FreeGpu(fin);FreeGpu(occ);FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// FED FAT-TILE MICRO-BATCH GEMM (MAD-305): the occ_kernel_mbgemm.s vehicle on a REAL fp8 GEMM.
// Persistent waves pull output-tiles from the atomic queue; per tile dyn-VGPR grows, STREAMS the
// K-reduction with real A (direct) + B (global_load_tr from pre-shuffled) feed, ships acc[0][0],
// shrinks. Oracle = per-tile chained wmma_ref over the K-tiles. C buffer = totalTiles*1024 B.
// ---------------------------------------------------------------------------
static inline int mbg_trperm(int L, int s) {            // verified closed form (Phase-0 contract)
    int base = (L & 7) + ((L >> 3) & 1) * 32 + ((L >> 4) & 1) * 128;
    return base + (s & 3) * 8 + ((s >> 2) & 1) * 64;
}
static void mbg_preshuffle_B(const uint8_t* B, uint8_t* Bshuf, int K, int N) {
    int KT = K / 16, NT = N / 16;
    for (int kt = 0; kt < KT; ++kt) for (int nt = 0; nt < NT; ++nt) {
        uint8_t* tile = Bshuf + (size_t)(kt * NT + nt) * 256;
        for (int L = 0; L < 32; ++L) for (int s = 0; s < 8; ++s) {
            int kl = ((L >> 4) & 1) * 8 + s, nl = L & 15;
            tile[mbg_trperm(L, s)] = B[(size_t)(kt * 16 + kl) * N + (nt * 16 + nl)];
        }
    }
}
// A-shuf for the ANOLDSTR LDS-free-A path: SYMMETRIC mirror of mbg_preshuffle_B (M<->N). Layout = [kt][mt][256]
// (MT m-frags per kt-row). global_load_tr_b64 reads tile (kt,mt) -> the A-frag (lane=row=ml, K-bytes [colhi*8..+7]),
// the same hardware coalesced-transpose path B uses. Indexed in-kernel by s2:3=Ashuf base, s14=MT*256 (A kt stride).
static void mbg_preshuffle_A(const uint8_t* A, uint8_t* Ashuf, int M, int K) {
    int KT = K / 16, MT = M / 16;
    for (int kt = 0; kt < KT; ++kt) for (int mt = 0; mt < MT; ++mt) {
        uint8_t* tile = Ashuf + (size_t)(kt * MT + mt) * 256;
        for (int L = 0; L < 32; ++L) for (int s = 0; s < 8; ++s) {
            int kl = ((L >> 4) & 1) * 8 + s, ml = L & 15;
            tile[mbg_trperm(L, s)] = A[(size_t)(mt * 16 + ml) * K + (kt * 16 + kl)];
        }
    }
}
// FRAG-READY B for the 128-bit B feed (MAD-305 B128). RDNA4 has NO 128-bit transpose for 8-bit data
// (tr_b128 is 16-bit only), so we do the transpose HERE and lay B out lane-linear: block
// [(ktp*NT + nt)]*512; byte [L*16 + kk*8 + s] = B[(2*ktp+kk)*16 + colhi*8 + s][nt*16 + nl], colhi=(L>>4)&1,
// nl=L&15. The device then does a PLAIN global_load_b128 (vaddr=lane*16) delivering 2 K=16 B-frags/instr
// (kk0=low 8B, kk1=high 8B). Same total bytes as mbg_preshuffle_B; CPU-proven byte-identical frag values
// to the tr_b64 path. Requires K%32==0 (paired K-tiles) and N%16==0.
static void mbg_preshuffle_B128(const uint8_t* B, uint8_t* Bshuf, int K, int N) {
    int KTP = K / 32, NT = N / 16;
    for (int ktp = 0; ktp < KTP; ++ktp) for (int nt = 0; nt < NT; ++nt) {
        uint8_t* blk = Bshuf + (size_t)(ktp * NT + nt) * 512;
        for (int L = 0; L < 32; ++L) {
            int colhi = (L >> 4) & 1, nl = L & 15;
            for (int kk = 0; kk < 2; ++kk) for (int s = 0; s < 8; ++s)
                blk[L * 16 + kk * 8 + s] = B[(size_t)((2 * ktp + kk) * 16 + colhi * 8 + s) * N + (nt * 16 + nl)];
        }
    }
}
struct MbgResult { bool ok=false; uint32_t maxlive=0, total=0, okTiles=0, nChecked=0; uint64_t wall=0; double secs=0; uint32_t phase[6]={0,0,0,0,0,0};
                   uint64_t wallSum=0, wallMin=0, wallMax=0; uint32_t repsDone=0; };  // sustained mode: per-rep span stats
// When set (DYNFAT1 single-shot): run_mbgemm builds EVERYTHING (operands, queue, all PM4
// packets) then BLOCKS on this gate file before RingSubmit, so the volatile umr cap-flip can
// be applied <1s before dispatch and cannot revert in an idle gap. Cleared by the caller.
// g_readyFile is created the instant prep is done, so an operator one-liner can WAIT for it,
// then flip the cap + touch g_gateFile atomically -> cap is fresh (<100ms) at dispatch.
static const char* g_gateFile  = nullptr;
static const char* g_readyFile = nullptr;

static MbgResult run_mbgemm(uint32_t node, const char* isaPath, bool dynvgpr, uint32_t nWaves,
                            int M, int N, int K, int FM, int FN, bool fullCheck, bool useGenDiv=false,
                            uint32_t reps=1, double targetSecs=0.0) {   // targetSecs>0: loop reps until that much wall
    MbgResult res;
    int TMr = 16 * FM, TNc = 16 * FN, MTL = M / TMr, NTL = N / TNc, KT = K / 16, NT = N / 16;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if (!useGenDiv && (NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not power of two (need GENDIV bin)\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;                 // pow2 path only
    uint32_t magic = (uint32_t)((0x100000000ULL + NTL - 1) / NTL);          // GENDIV: ceil(2^32/NTL)

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};   // 1, 2, .5, -1, -2, -.5
    std::vector<uint8_t> Ah((size_t)M * K), Bh((size_t)K * N), Bshufh((size_t)K * N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i * 7 + i / (size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i * 5 + (i / (size_t)N) * 3) % 6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, (Ah.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true);    // VRAM: the A feed
    GpuBuf Bd  = AllocGpu(node, (Bshufh.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true); // VRAM: the B feed
    uint64_t cbytes = ((uint64_t)TOTAL * 1024 + 0xFFF) & ~0xFFFull;
    GpuBuf C   = AllocGpu(node, cbytes, false, true, /*deviceLocal*/true);   // VRAM: the C store traffic
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    // ---- VRAM GUARD: perf operands MUST be device-local (the PCIe-fed bug must not recur silently) ----
    if (!(Ad.vram && Bd.vram && C.vram)) {
        fprintf(stderr, "\n*** VRAM GUARD FAILED (%s): operands not device-local (A=%d B=%d C=%d) -> PCIe-fed, PERF INVALID ***\n",
                isaPath, Ad.vram, Bd.vram, C.vram);
        abort();
    }
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ah.data(), Ah.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,32,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t fatregs = (uint32_t)((32 + FM*FN*8 + FM*4 + FN*4 + 15) & ~15);  // double-buffered A/B frags
    uint32_t rsrc1 = BuildPgmRsrc1(false);
    uint32_t launchField = dynvgpr ? 4u : ((fatregs / 8) & 0x3fu);   // dyn launches lean 32; static reserves fat
    rsrc1 = (rsrc1 & ~0x3fu) | (launchField & 0x3fu);
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT);   // 15 user SGPRs s0..s14
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32),                 // s0:s1 occ
        (uint32_t)aVa, (uint32_t)(aVa>>32),                    // s2:s3 A
        (uint32_t)bVa, (uint32_t)(bVa>>32),                    // s4:s5 Bshuf
        (uint32_t)cVa, (uint32_t)(cVa>>32),                    // s6:s7 C
        (uint32_t)KT, (uint32_t)K, (uint32_t)(NT*256), TOTAL,  // s8 KT, s9 K, s10 NTx256, s11 TOTAL
        useGenDiv?magic:(uint32_t)(NTL-1), useGenDiv?(uint32_t)NTL:(uint32_t)log2NTL, (uint32_t)(FN*256), 0 };  // s12 magic|mask, s13 NTL|log2, s14 FNx256
    uint32_t dispInit = BuildDispatchInitiator();
    // SUSTAINED: re-place packets + re-submit back-to-back `reps` times (buffers reused, no host refill),
    // resetting the occ counters + fence each rep. Accumulate per-rep in-kernel GPU-clock spans so the
    // steady-state is read over tens of seconds, not a single warmup-dominated millisecond dispatch.
    const double timeoutS = 25.0;
    uint64_t spanSum=0, spanMin=~0ull, spanMax=0; uint32_t lastMaxlive=0, lastTotal=0; bool allok=true;
    double loopStart = now_s();
    // ---- COMPOSITOR YIELD (wall-time based; the proper fix per task #97) ----
    // The R9700 (card1) ALSO drives 2 of the 3 desktop monitors (DP-1 + DP-4); Thunderbolt login order
    // pins them here -- they CANNOT move to the 6900XT. Sustained back-to-back dispatch starves Hyprland's
    // gfx ring -> SQC(data) page fault -> gfx_0.0.0 ring reset -> desktop dies (the 2026-05-27 starvation
    // signature; this is what bricked the dynfull run at ~100 min on o_pf 4x2 b32, the fattest 128-VGPR
    // grow held longest). Between reps the dispatch is fully DRAINED (fence signaled, waves exited, dyn-VGPR
    // pool released), so a short host sleep here gives the compositor an unconditional render + VGPR window.
    // TF is measured from per-rep IN-KERNEL gpu-clock spans, so this host sleep does NOT skew throughput.
    // Env: ML8_YIELD_DISABLE=1 (genuinely headless box) / ML8_YIELD_MS (sleep, default 5) /
    //      ML8_YIELD_EVERY_MS (render-window cadence, default 100).
    const char* ydis = getenv("ML8_YIELD_DISABLE");
    bool  yieldOff   = ydis && ydis[0]=='1';
    int   yieldMs    = getenv("ML8_YIELD_MS")       ? atoi(getenv("ML8_YIELD_MS"))       : 5;
    double yieldEvery= (getenv("ML8_YIELD_EVERY_MS")? atoi(getenv("ML8_YIELD_EVERY_MS")): 100) / 1000.0;
    if (yieldMs < 0) yieldMs = 0;
    if (yieldEvery <= 0.0) yieldEvery = 0.1;
    double lastYield = loopStart;
    for (uint32_t rep=0; ; ++rep) {
        if (targetSecs > 0.0) { if (rep >= 4 && (now_s()-loopStart) >= targetSecs) break; if (rep >= 200000u) break; }
        else if (rep >= reps) break;
        occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;
        RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
        RingPlace(ring, PM4DispatchDirectPacket(nWaves * 32u, 1, 1, dispInit));
        RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));
        if (g_gateFile && rep==0) {   // DYNFAT1 single-shot gate (only first rep; null in normal runs)
            remove(g_gateFile);
            if (g_readyFile) { FILE* rf = fopen(g_readyFile, "w"); if (rf) fclose(rf); }
            fprintf(stderr, "\n[DYNFAT1] PREP COMPLETE. waiting for gate...\n");
            for (;;) { if (access(g_gateFile, F_OK) == 0) break; struct timespec ts={0,20000000}; nanosleep(&ts,nullptr); }
            remove(g_gateFile); if (g_readyFile) remove(g_readyFile);
        }
        double t0 = now_s(); RingSubmit(ring);
        bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
        while (true) { double now = now_s();
            if (occW[1] > 0) admitted = true;
            uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
            bool ff = (*fenceW == FENCE_VALUE);
            if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
            if (now - t0 > timeoutS) break;
        }
        if (!done) {
            fprintf(stderr, "\n*** MBGEMM TIMEOUT (%s rep %u): live=%u maxlive=%u nextTile=%u ***\n",
                    isaPath, rep, occW[0], occW[1], occW[5]);
            allok=false; break;
        }
        uint32_t gs=occW[2], ge=occW[3];
        uint64_t span = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs);
        spanSum += span; if (span<spanMin) spanMin=span; if (span>spanMax) spanMax=span;
        lastMaxlive=occW[1]; lastTotal=occW[4]; res.repsDone = rep+1;
        // ---- compositor yield: dispatch is drained (fence signaled) -> hand the gfx ring a window ----
        if (!yieldOff && yieldMs > 0 && (now_s() - lastYield) >= yieldEvery) {
            struct timespec ts = { yieldMs/1000, (long)(yieldMs%1000)*1000000L };
            nanosleep(&ts, nullptr);
            lastYield = now_s();
        }
    }
    if (!allok) {
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = lastMaxlive; res.total = lastTotal;
    res.wall = spanSum / (res.repsDone ? res.repsDone : 1);   // mean per-rep span (== single span when reps=1)
    res.wallSum = spanSum; res.wallMin = spanMin; res.wallMax = spanMax;

    // ---- per-tile oracle: reference D = sum_kt A_block . B_block (chained wmma_ref, D=A*B+C) ----
    const float* Cf = (const float*)C.ptr; uint32_t okc=0, checked=0;
    uint32_t stride = fullCheck ? 1u : (TOTAL > 256 ? TOTAL / 256u : 1u);
    for (uint32_t ti = 0; ti < TOTAL; ti += stride) {
        int tc = ti % NTL, tr = ti / NTL;   // general division: correct for pow2 AND GENDIV (non-pow2) NTL
        float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
        uint8_t Ablk[256], Bblk[256]; float Dout[256];
        for (int kt = 0; kt < KT; ++kt) {
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) {   // acc[0][0] origin: rows tr*16*FM, cols tc*16*FN
                Ablk[i*16+j] = Ah[(size_t)(tr*16*FM+i)*K + (kt*16+j)];
                Bblk[i*16+j] = Bh[(size_t)(kt*16+i)*N + (tc*16*FN+j)];
            }
            wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
            for (int i=0;i<256;i++) Cacc[i]=Dout[i];
        }
        float D[256]; unpack_D(Cf + (size_t)ti * 256, D);
        bool good=true; for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
        if (good) ++okc; ++checked;
    }
    res.okTiles = okc; res.nChecked = checked;
    for (int i = 0; i < 6; ++i) res.phase[i] = occW[6 + i];   // PROFILE: per-phase tick totals (occ[24..44], wg0)
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// WAVE-GROUP skeleton dispatch (MAD-305 Phase 1): nWG persistent TWM*TWN-wave (128-thread) workgroups
// grid-stride over TGID_X; one logical 128x128 C tile per workgroup. The SMOKE kernel (no compute)
// writes a decode mark per wave to C[ti*WAVES + wid]; the host verifies every wave of every tile
// wrote the right (tile_row,tile_col,wave_m,wave_n) -> proves (a) the 4-wave workgroup forms, (b) the
// lane/wave mapping is correct, (c) grid-stride tile decode covers each tile exactly once.
// ---------------------------------------------------------------------------
struct WgResult { bool ok=false; uint32_t maxlive=0, total=0; uint32_t okMarks=0, badMarks=0, missMarks=0; uint64_t wall=0; double secs=0; };

static WgResult run_wggemm_smoke(uint32_t node, const char* isaPath, int M, int N, uint32_t nWG,
                                 int TWM, int TWN) {
    WgResult res;
    const int TM = 128, TN = 128;                       // logical tile (per-wave 4x4 frags x TWMxTWN)
    int MTL = M / TM, NTL = N / TN;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not power of two\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;
    int WAVES = TWM * TWN;

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    uint64_t cbytes = ((uint64_t)TOTAL * WAVES * 4 + 0xFFF) & ~0xFFFull;   // WAVES u32 marks per tile
    GpuBuf C   = AllocGpu(node, cbytes, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;
    uint32_t* Cw = (uint32_t*)C.ptr;
    for (uint64_t i = 0; i < (uint64_t)TOTAL * WAVES; ++i) Cw[i] = 0xFFFFFFFFu;   // sentinel

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES*32),1,1,0,0};   // NUM_THREAD_X = WAVES*32 = 128 -> 4 waves/WG
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false);
    rsrc1 = (rsrc1 & ~0x3fu) | (4u & 0x3fu);                    // static 32 VGPR (field 4); smoke uses ~8
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT);   // 15 user SGPRs -> s15=TGID_X
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32),                 // s0:s1 occ
        (uint32_t)cVa, (uint32_t)(cVa>>32),                    // s2:s3 C
        TOTAL, nWG, (uint32_t)(NTL-1), (uint32_t)log2NTL,      // s4 TOTAL, s5 nWG, s6 NTL_MASK, s7 NTL_LOG2
        0,0,0,0, 0,0,0,0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * (uint32_t)(WAVES*32), 1, 1, dispInit));   // DIM_X = nWG*128 threads
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    const double timeoutS = 20.0;
    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > timeoutS) break;
    }
    double t1 = now_s();
    if (!done) {
        fprintf(stderr, "\n*** WGGEMM SMOKE TIMEOUT (%s): live=%u maxlive=%u (queue may be hung) ***\n",
                isaPath, occW[0], occW[1]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[4];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.secs = t1 - t0;

    if (getenv("WG_DIAG")) {
        fprintf(stderr, "  [DIAG] occ: live=%u maxlive=%u tstart=%u tend=%u total=%u  TOTAL=%u nWG=%u\n",
                occW[0], occW[1], occW[2], occW[3], occW[4], TOTAL, nWG);
        uint32_t shown = 0;
        for (uint32_t i = 0; i < (uint32_t)TOTAL * WAVES && shown < 24; ++i) {
            if (Cw[i] == 0xFFFFFFFFu) continue;
            uint32_t v = Cw[i], slot_ti = i / WAVES, slot_wid = i % WAVES;
            fprintf(stderr, "  [DIAG] C[%u] (slot ti=%u wid=%u) = 0x%08x -> row=%u col=%u wm=%u wn=%u\n",
                    i, slot_ti, slot_wid, v, (v>>20)&0xFFF, (v>>8)&0xFFF, (v>>4)&0xF, v&0xF);
            ++shown;
        }
        uint32_t nonsentinel = 0; for (uint32_t i = 0; i < (uint32_t)TOTAL*WAVES; ++i) if (Cw[i]!=0xFFFFFFFFu) ++nonsentinel;
        fprintf(stderr, "  [DIAG] %u/%u C slots written (rest still sentinel)\n", nonsentinel, TOTAL*WAVES);
    }
    // ---- verify the decode marks: every wave of every tile present + correct ----
    for (uint32_t ti = 0; ti < TOTAL; ++ti) {
        uint32_t trow = ti >> log2NTL, tcol = ti & (uint32_t)(NTL - 1);
        for (int wid = 0; wid < WAVES; ++wid) {
            uint32_t got = Cw[(uint64_t)ti * WAVES + wid];
            if (got == 0xFFFFFFFFu) { res.missMarks++; continue; }
            uint32_t wm = (uint32_t)(wid / TWN), wn = (uint32_t)(wid % TWN);
            uint32_t want = (trow << 20) | (tcol << 8) | (wm << 4) | wn;
            if (got == want) res.okMarks++; else res.badMarks++;
        }
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// WAVE-GROUP fp8 GEMM COMPUTE (MAD-305 Phase 2): 4-wave cooperative, A-in-LDS + B-global_load_tr,
// per-wave static 4x4. Verifies every frag of every tile vs a chained-wmma_ref reference. C is the
// FLAT diagnostic layout: C[ti*65536 + wid*16384 + frag*1024] = each wave's 16 frags (256 f32 each).
// ---------------------------------------------------------------------------
struct WgcResult { bool ok=false; uint32_t maxlive=0, total=0; uint64_t okFrags=0, badFrags=0; uint64_t wall=0; };

static uint32_t ldsRsrc2Bits(uint32_t ldsBytes, uint32_t* outUnits, uint32_t* outAlloc, uint32_t* outGranule);

static WgcResult run_wggemm_compute(uint32_t node, const char* isaPath, int M, int N, int K,
                                    uint32_t nWG, bool fullCheck,
                                    int FMt = 4, uint32_t ldsBytes = 8196u, uint32_t vgprField = 26u,
                                    int useAtr = 0, int TWN = 2, int FNt = -1, int TWMt = 2,
                                    int useB128 = 0, int useTileord = 0, int edgeData = 0,
                                    bool useGenDiv = false) {
    WgcResult res;
    const int FM = FMt, FN = (FNt < 0 ? FMt : FNt), TWM = TWMt, WAVES = TWM*TWN, TBK = 32;
    const int TM = TWM*FM*16, TN = TWN*FN*16;     // claimed tile = (TWM*FM*16)x(TWN*FN*16)
    int NTL = N / TN, MTL = M / TM, NT = N / 16, NTILES = K / TBK;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if (!useGenDiv && (NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not pow2\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Bshufh((size_t)K*N);
    if (edgeData) {
        // FULL-DOMAIN edge-encoding gate (MAD-305 safety floor): A heavy on DENORMALS (0x01-0x07 = exp0,
        // subnormal ~2^-9, both signs) + small/mid/max + zero; B heavy on MAX-NORMALS (0x7E=448, 0xFE=-448).
        // -> dot products are dominated by denorm(A)*max(B) ~ 0.002*448 summed over K=512 (~hundreds), so if the
        // GPU FLUSHES input denormals (default FP_DENORM) while the e4m3fn ref does not, output diverges LOUDLY
        // (>> 0.5%+1e-2 tol). Also exercises 0x7E saturation + signs. NO NaN (0x7F/0xFF): ref maps NaN->0 but HW
        // WMMA would emit NaN -> a ref limitation, not a HW bug; tested separately.
        static const uint8_t EA[16] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08, 0x81,0x83,0x85,0x87, 0x10,0x38,0x7E,0x00};
        static const uint8_t EB[16] = {0x7E,0x7C,0x7A,0x78,0xFE,0xFC,0x40,0x38, 0x7E,0x7C,0x40,0x38,0x7E,0x7C,0x40,0x7E};
        for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = EA[(i*7 + i/(size_t)K) % 16];
        for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = EB[(i*5 + (i/(size_t)N)*3) % 16];
    } else {
        for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
        for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    }
    if (useB128) mbg_preshuffle_B128(Bh.data(), Bshufh.data(), K, N);   // frag-ready 512B blocks for plain b128
    else         mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);
    std::vector<uint8_t> Ashufh;   // ANOLDSTR oracle: A-shuf feed (plain Ah kept for the chained wmma_ref reference)
    if (useAtr) { Ashufh.resize((size_t)M*K); mbg_preshuffle_A(Ah.data(), Ashufh.data(), M, K); }

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, (Ah.size() + 0xFFF) & ~0xFFFull, false, true);
    GpuBuf Bd  = AllocGpu(node, (Bshufh.size() + 0xFFF) & ~0xFFFull, false, true);
    GpuBuf AshufD{}; if (useAtr) { AshufD = AllocGpu(node, (Ashufh.size() + 0xFFF) & ~0xFFFull, false, true); memcpy(AshufD.ptr, Ashufh.data(), Ashufh.size()); }
    uint64_t cbytes = ((uint64_t)TOTAL * (uint64_t)(WAVES*FM*FN*256*4) + 0xFFF) & ~0xFFFull;  // WAVES * (FM*FN frags) * 256 floats
    GpuBuf C   = AllocGpu(node, cbytes, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ah.data(), Ah.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES*32),1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);   // static vgprField*8 VGPR
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    printf("  [wggemm2] %dx%dx%d  tile=%dx%d (FM=FN=%d)  TOTAL=%u tiles  nWG=%u  LDS=%uB(units=%u alloc=%u) VGPR~%u RSRC2=0x%x\n",
           M,N,K, TM,TN,FMt, TOTAL, nWG, ldsBytes, ldsU, ldsA, vgprField*8, rsrc2);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    int log2MTL = 0; while ((1 << log2MTL) < MTL) ++log2MTL;
    if (useTileord && (MTL & (MTL - 1)) != 0) { fprintf(stderr, "  MTL=%d not pow2 (N_STATIONARY needs it)\n", MTL); return res; }
    uint32_t s10v = useTileord ? (uint32_t)(MTL-1) : (uint32_t)(NTL-1);   // TILEORD=1 swaps decode to MTL mask/shift
    uint32_t s11v = useTileord ? (uint32_t)log2MTL : (uint32_t)log2NTL;
    if (useGenDiv && !useTileord) {                                       // GENDIV=1: s10=magic=ceil(2^32/NTL), s11=NTL
        s10v = (uint32_t)((0x100000000ULL + (uint64_t)NTL - 1) / (uint64_t)NTL);
        s11v = (uint32_t)NTL;
    }
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, (uint32_t)(NT*256), s10v, s11v,
        (uint32_t)NTILES, TOTAL, 0, 0 };
    if (useAtr) { uint64_t asVa=(uint64_t)AshufD.ptr; userdata[2]=(uint32_t)asVa; userdata[3]=(uint32_t)(asVa>>32); userdata[14]=(uint32_t)((uint64_t)M*16); } // ANOLDSTR oracle: s2:3=Ashuf, s14=MT*256
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * (uint32_t)(WAVES*32), 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 20.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** WGGEMM2 TIMEOUT (%s): live=%u maxlive=%u claim=%u ***\n", isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        if (useAtr) FreeGpu(AshufD);
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }

    // ---- verify every frag of every tile (or a stride sample) vs chained wmma_ref ----
    const float* Cf = (const float*)C.ptr;
    uint32_t tstride = fullCheck ? 1u : (TOTAL > 64 ? TOTAL/64u : 1u);
    for (uint32_t ti = 0; ti < TOTAL; ti += tstride) {
        int trow, tcol;
        if (useTileord)     { trow = ti & (MTL - 1); tcol = ti >> log2MTL; }   // N_STATIONARY: mirror kernel TILEORD=1
        else if (useGenDiv) { trow = (int)(ti / (uint32_t)NTL); tcol = (int)(ti % (uint32_t)NTL); } // GENDIV: exact div/mod
        else                { trow = ti >> log2NTL;  tcol = ti & (NTL - 1); }
        for (int wid = 0; wid < WAVES; ++wid) {
            int wm = wid / TWN, wn = wid % TWN;
            for (int mi = 0; mi < FM; ++mi) for (int ni = 0; ni < FN; ++ni) {
                int rowbase = trow*TM + wm*(FM*16) + mi*16;
                int colbase = tcol*TN + wn*(FN*16) + ni*16;
                float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
                float CaccHalf[256]; for (int i=0;i<256;i++) CaccHalf[i]=0.f;  // GEMM over FIRST HALF of K-slices
                uint8_t Ablk[256], Bblk[256]; float Dout[256];
                for (int kt = 0; kt < K/16; ++kt) {
                    for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                        Ablk[i*16+j] = Ah[(size_t)(rowbase+i)*K + (kt*16+j)];
                        Bblk[i*16+j] = Bh[(size_t)(kt*16+i)*N + (colbase+j)];
                    }
                    wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
                    for (int i=0;i<256;i++) Cacc[i]=Dout[i];
                    if (kt == K/32 - 1) for (int i=0;i<256;i++) CaccHalf[i]=Cacc[i];   // snapshot at first-half-K boundary
                }
                int frag = mi*FN + ni;
                size_t foff = (size_t)ti*(size_t)(WAVES*FM*FN*256) + (size_t)wid*(size_t)(FM*FN*256) + (size_t)frag*256;   // float index
                float D[256]; unpack_D(Cf + foff, D);
                bool good=true;
                for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
                if (good) res.okFrags++; else res.badFrags++;
                if (!good && getenv("WGC_DBG")) {
                    static int dbg = 0;
                    if (dbg < 24 && ti==0 && wid==0 && frag==0) {
                        // disambiguate the half: 0.5x-full (magnitude halve) vs first-half-K-slices vs even-kt-only vs even-K-within-WMMA
                        float CaccEvenKt[256]; for (int i=0;i<256;i++) CaccEvenKt[i]=0.f;
                        float CaccEvenK[256];  for (int i=0;i<256;i++) CaccEvenK[i]=0.f;
                        for (int kt=0; kt<K/16; ++kt) {
                            for (int i=0;i<16;i++) for (int j=0;j<16;j++) { Ablk[i*16+j]=Ah[(size_t)(rowbase+i)*K+(kt*16+j)]; Bblk[i*16+j]=Bh[(size_t)(kt*16+i)*N+(colbase+j)]; }
                            if ((kt&1)==0) { wmma_ref_16x16x16(Ablk,Bblk,CaccEvenKt,Dout); for(int i=0;i<256;i++) CaccEvenKt[i]=Dout[i]; }   // even kt only
                            uint8_t Ah2[256],Bh2[256]; for(int i=0;i<16;i++) for(int kk=0;kk<16;kk++){ uint8_t av=(kk&1)?0:Ablk[i*16+kk]; Ah2[i*16+kk]=av; uint8_t bv=(kk&1)?0:Bblk[kk*16+i]; Bh2[kk*16+i]=bv; }
                            wmma_ref_16x16x16(Ah2,Bh2,CaccEvenK,Dout); for(int i=0;i<256;i++) CaccEvenK[i]=Dout[i];   // even-K within each WMMA (odd K zeroed)
                        }
                        auto matchpct=[&](const float* R){ int m=0; for(int i=0;i<256;i++) if(std::fabs(D[i]-R[i])<=5e-3f*std::fabs(R[i])+1e-2f) m++; return m*100/256; };
                        std::vector<float> half(256); for(int i=0;i<256;i++) half[i]=0.5f*Cacc[i];
                        float amaxD=0.f,amaxC=0.f; for(int i=0;i<256;i++){amaxD=std::max(amaxD,std::fabs(D[i]));amaxC=std::max(amaxC,std::fabs(Cacc[i]));}
                        fprintf(stderr, "  [WGC_DBG] K=%d wid=%d frag=%d ratio=%.3f | match%%: 0.5xFull=%d firstHalfK=%d evenKt=%d evenK-in-WMMA=%d\n",
                                K, wid, frag, amaxC>0?amaxD/amaxC:0.f, matchpct(half.data()), matchpct(CaccHalf), matchpct(CaccEvenKt), matchpct(CaccEvenK));
                        dbg++;
                    }
                }
            }
        }
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    if (useAtr) FreeGpu(AshufD);
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// WAVE-GROUP PERF (MAD-305 Phase 3 G2 gate): time the STORE=0 (minimal-store) wave-group GEMM at a
// wgrad shape; report TF, % of 307 ceiling, WMMA/cycle, maxlive, claims, sampled acc[0][0] correctness.
// ---------------------------------------------------------------------------
struct WgpResult { bool ok=false; uint32_t maxlive=0, total=0; uint64_t wall=0; double tf=0; uint32_t okSamp=0, badSamp=0; };

static WgpResult run_wggemm_perf(uint32_t node, const char* isaPath, int M, int N, int K,
                                 uint32_t nWG, double freq_hz,
                                 int FMt = 4, uint32_t ldsBytes = 8196u, uint32_t vgprField = 26u,
                                 int TWN = 2, int useAtr = 0, int FNt = -1, int TWMt = 2,
                                 int useB128 = 0, int useTileord = 0, bool useGenDiv = false) {
    WgpResult res;
    if (FNt < 0) FNt = FMt;                        // FNt<0 -> square per-wave tile (FN=FM); set explicitly for 8x2 etc.
    const int TWM = TWMt, WAVES = TWM*TWN;        // TWM*TWN waves/WG (16 @ TWM=4 TWN=4 lean, or 8 @ TWM=2 TWN=4)
    const int TM = TWM*FMt*16, TN = TWN*FNt*16;   // claimed tile = (TWM*FM*16)x(TWN*FN*16)  (128x256 @ TWN=4)
    int NTL = N / TN, MTL = M / TM, NT = N / 16, NTILES = K / 32;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if (!useGenDiv && (NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not pow2\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Bshufh((size_t)K*N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    if (useB128) mbg_preshuffle_B128(Bh.data(), Bshufh.data(), K, N);   // frag-ready 512B blocks for plain b128
    else         mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);
    std::vector<uint8_t> Ashufh;   // ANOLDSTR: A-shuf feed (mbg_preshuffle_A); plain Ah kept for the acc00 reference
    if (useAtr) { Ashufh.resize((size_t)M*K); mbg_preshuffle_A(Ah.data(), Ashufh.data(), M, K); }

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, (Ah.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true);   // VRAM: the A feed
    GpuBuf Bd  = AllocGpu(node, (Bshufh.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true);// VRAM: the B feed
    GpuBuf AshufD{}; if (useAtr) { AshufD = AllocGpu(node, (Ashufh.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true); memcpy(AshufD.ptr, Ashufh.data(), Ashufh.size()); }
    uint64_t cbytes = ((uint64_t)TOTAL * (uint64_t)(WAVES*1024) + 0xFFF) & ~0xFFFull;  // STORE=0: 1 frag/wave -> WAVES*1024 B/tile
    GpuBuf C   = AllocGpu(node, cbytes, false, true, /*deviceLocal*/true);   // VRAM: the C store traffic
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    // ---- VRAM GUARD: perf operands MUST be device-local (the PCIe-fed bug must not recur silently) ----
    if (!(Ad.vram && Bd.vram && C.vram)) {
        fprintf(stderr, "\n*** VRAM GUARD FAILED (%s): operands not device-local (A=%d B=%d C=%d) -> PCIe-fed, PERF INVALID ***\n",
                isaPath, Ad.vram, Bd.vram, C.vram);
        abort();
    }
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ah.data(), Ah.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES*32),1,1,0,0};   // workgroup = WAVES waves (256 threads @ TWN=4)
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    int log2MTL = 0; while ((1 << log2MTL) < MTL) ++log2MTL;
    if (useTileord && (MTL & (MTL - 1)) != 0) { fprintf(stderr, "  MTL=%d not pow2 (N_STATIONARY needs it)\n", MTL); return res; }
    uint32_t s10v = useTileord ? (uint32_t)(MTL-1) : (uint32_t)(NTL-1);   // TILEORD=1 swaps decode to MTL mask/shift
    uint32_t s11v = useTileord ? (uint32_t)log2MTL : (uint32_t)log2NTL;
    if (useGenDiv && !useTileord) {                                       // GENDIV=1: s10=magic=ceil(2^32/NTL), s11=NTL
        s10v = (uint32_t)((0x100000000ULL + (uint64_t)NTL - 1) / (uint64_t)NTL);
        s11v = (uint32_t)NTL;
    }
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, (uint32_t)(NT*256), s10v, s11v,
        (uint32_t)NTILES, TOTAL, 0, 0 };
    if (useAtr) { uint64_t asVa=(uint64_t)AshufD.ptr; userdata[2]=(uint32_t)asVa; userdata[3]=(uint32_t)(asVa>>32); userdata[14]=(uint32_t)((uint64_t)M*16); } // ANOLDSTR: s2:3=Ashuf, s14=MT*256
    uint32_t dispInit = BuildDispatchInitiator();
    // (a) MULTI-ITERATION timing: the wall is a GPU-tick span (occ[2]->occ[3]), so sub-ms shapes (small M) are
    //   dominated by shader-clock RAMP -- a single cold dispatch is noisy (a byte-identical bin swung +-60% at
    //   nWG=32). Run WARM warm-up dispatches (ramp the clock) then take the MIN ticks over TIMED runs = steady
    //   state -> reproducible. Each rep resets occ (claim counter occ[5] + start/end ticks occ[2]/[3]) + fence.
    const int WARM = 2, TIMED = 4;
    uint64_t bestWall = ~0ull;
    for (int rep = 0; rep < WARM + TIMED; ++rep) {
        occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;
        RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
        RingPlace(ring, PM4DispatchDirectPacket(nWG * (uint32_t)(WAVES*32), 1, 1, dispInit));
        RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

        double t0 = now_s(); RingSubmit(ring);
        bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
        while (true) { double now = now_s();
            if (occW[1] > 0) admitted = true;
            uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
            bool ff = (*fenceW == FENCE_VALUE);
            if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
            if (now - t0 > 40.0) break;
        }
        if (!done) {
            fprintf(stderr, "\n*** WGGEMM PERF TIMEOUT (%s): live=%u maxlive=%u claim=%u ***\n", isaPath, occW[0], occW[1], occW[5]);
            CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
            if (useAtr) FreeGpu(AshufD);
            FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
            return res;
        }
        res.maxlive = occW[1]; res.total = occW[5];
        uint32_t gs=occW[2], ge=occW[3];
        uint64_t w = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs);
        if (rep >= WARM && w > 0 && w < bestWall) bestWall = w;   // min over the TIMED runs = warmed-up steady state
    }
    res.ok = true; res.wall = (bestWall==~0ull) ? 0 : bestWall;
    res.tf = (res.wall > 0) ? (2.0*(double)M*N*K * freq_hz / (double)res.wall / 1e12) : 0.0;

    // ---- sampled acc[0][0] correctness (cheap; ~16 tiles) ----
    const float* Cf = (const float*)C.ptr;
    uint32_t tstride = TOTAL > 16 ? TOTAL/16u : 1u;
    for (uint32_t ti = 0; ti < TOTAL; ti += tstride) {
        int trow, tcol;
        if (useTileord)     { trow = ti & (MTL - 1); tcol = ti >> log2MTL; }   // N_STATIONARY: mirror kernel TILEORD=1
        else if (useGenDiv) { trow = (int)(ti / (uint32_t)NTL); tcol = (int)(ti % (uint32_t)NTL); } // GENDIV: exact div/mod
        else                { trow = ti >> log2NTL;  tcol = ti & (NTL - 1); }
        for (int wid = 0; wid < WAVES; ++wid) {
            int wm = wid/TWN, wn = wid%TWN, rowbase = trow*TM + wm*(FMt*16), colbase = tcol*TN + wn*(FNt*16);
            float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
            uint8_t Ablk[256], Bblk[256]; float Dout[256];
            for (int kt = 0; kt < K/16; ++kt) {
                for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                    Ablk[i*16+j] = Ah[(size_t)(rowbase+i)*K + (kt*16+j)];
                    Bblk[i*16+j] = Bh[(size_t)(kt*16+i)*N + (colbase+j)];
                }
                wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
                for (int i=0;i<256;i++) Cacc[i]=Dout[i];
            }
            float D[256]; unpack_D(Cf + (size_t)ti*(size_t)(WAVES*256) + (size_t)wid*256, D);
            bool good=true; for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
            if (good) res.okSamp++; else res.badSamp++;
        }
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    if (useAtr) FreeGpu(AshufD);
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ===========================================================================
// MAD-305 #323 WAVE-SPECIALIZED fp8 GEMM (occ_kernel_wavespec.s).
//   Claimed tile = (NCOMP*FM*16) x (FN*16): NCOMP compute waves stacked in M, ONE shared B-panel.
//   NLOAD lean loader waves global_load_tr A+B into an LDS slot; compute ds_load + WMMA only.
//   Both operands preshuffled (Ashuf via mbg_preshuffle_A, Bshuf via mbg_preshuffle_B). NTILES=K/16.
//   userdata: s0:1=occ s2:3=Ashuf s4:5=Bshuf s6:7=C s8=K s9=NT*256 s10=NTL_MASK s11=NTL_LOG2
//             s12=NTILES(K/16) s13=TOTAL s14=MT*256.  dynvgpr arms RSRC2 bit6 (T4).
//   C (STORE=1): C[ti*(NCOMP*FM*FN*256) + cid*(FM*FN*256) + frag*256] floats; frag=mi*FN+ni.
// ---------------------------------------------------------------------------
static int wavespec_vgpr_field(int FM, int FN) {        // matches the kernel's NFV = FB + 2*FN + 16
    int ACC = 32, FA = ACC + 8*FM*FN, FB = FA + 2*FM, NFV = FB + 2*FN + 16;
    return (NFV + 7) / 8;                                 // rsrc1 VGPR granule = field*8
}
static uint32_t wavespec_lds_bytes(int FM, int FN, int NCOMP) {
    return (uint32_t)((NCOMP*FM + FN) * 256 + 12);        // SLOT + ti + bar_cnt + bar_sense (matches kernel
                                                          //   LDS_TOTAL; the +8 is the BUSYWAIT barrier slots,
                                                          //   harmless/unused on the non-busy-wait path).
}

static WgcResult run_wavespec_compute(uint32_t node, const char* isaPath, int M, int N, int K,
                                      uint32_t nWG, bool fullCheck,
                                      int FM, int FN, int NLOAD, int NCOMP, bool dynvgpr) {
    WgcResult res;
    const int WAVES_LAUNCH = NLOAD + NCOMP;
    const int TM = NCOMP*FM*16, TN = FN*16;               // claimed tile (one N-panel, NCOMP M-bands)
    int NTL = N / TN, MTL = M / TM, NT = N / 16, MT = M / 16, NTILES = K / 16;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not pow2\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Ashufh((size_t)M*K), Bshufh((size_t)K*N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    mbg_preshuffle_A(Ah.data(), Ashufh.data(), M, K);
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);

    uint32_t ldsBytes = wavespec_lds_bytes(FM, FN, NCOMP);
    // dyn: launch LEAN (32 VGPR = 4 granules; covers the identity/claim prologue v0..v24), then each wave
    //   s_alloc_vgpr's to its role size (loaders stay 32, compute grow to NFV). static: full NFV-sized launch.
    uint32_t vgprField = dynvgpr ? 4u : (uint32_t)wavespec_vgpr_field(FM, FN);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, (Ashufh.size() + 0xFFF) & ~0xFFFull, false, true);
    GpuBuf Bd  = AllocGpu(node, (Bshufh.size() + 0xFFF) & ~0xFFFull, false, true);
    uint64_t cbytes = ((uint64_t)TOTAL * (uint64_t)(NCOMP*FM*FN*256*4) + 0xFFF) & ~0xFFFull;
    GpuBuf C   = AllocGpu(node, cbytes, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ashufh.data(), Ashufh.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES_LAUNCH*32),1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(dynvgpr); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    printf("  [wavespec] %dx%dx%d  tile=%dx%d (FM=%d FN=%d NLOAD=%d NCOMP=%d)  TOTAL=%u nWG=%u  LDS=%uB(units=%u alloc=%u) VGPR~%u dyn=%d RSRC2=0x%x\n",
           M,N,K, TM,TN, FM,FN,NLOAD,NCOMP, TOTAL, nWG, ldsBytes, ldsU, ldsA, vgprField*8, dynvgpr, rsrc2);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t s10v = (uint32_t)(NTL-1), s11v = (uint32_t)log2NTL;
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, (uint32_t)(NT*256), s10v, s11v,
        (uint32_t)NTILES, TOTAL, (uint32_t)(MT*256), 0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * (uint32_t)(WAVES_LAUNCH*32), 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 20.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** WAVESPEC TIMEOUT (%s): live=%u maxlive=%u claim=%u ***\n", isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }

    // ---- oracle: every frag of every (tile, compute-wave) vs chained wmma_ref ----
    const float* Cf = (const float*)C.ptr;
    uint32_t tstride = fullCheck ? 1u : (TOTAL > 64 ? TOTAL/64u : 1u);
    for (uint32_t ti = 0; ti < TOTAL; ti += tstride) {
        int trow = ti >> log2NTL, tcol = ti & (NTL - 1);
        for (int cid = 0; cid < NCOMP; ++cid) {
            for (int mi = 0; mi < FM; ++mi) for (int ni = 0; ni < FN; ++ni) {
                int rowbase = trow*TM + cid*(FM*16) + mi*16;
                int colbase = tcol*TN + ni*16;
                float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
                uint8_t Ablk[256], Bblk[256]; float Dout[256];
                for (int kt = 0; kt < K/16; ++kt) {
                    for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                        Ablk[i*16+j] = Ah[(size_t)(rowbase+i)*K + (kt*16+j)];
                        Bblk[i*16+j] = Bh[(size_t)(kt*16+i)*N + (colbase+j)];
                    }
                    wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
                    for (int i=0;i<256;i++) Cacc[i]=Dout[i];
                }
                int frag = mi*FN + ni;
                size_t foff = (size_t)ti*(size_t)(NCOMP*FM*FN*256) + (size_t)cid*(size_t)(FM*FN*256) + (size_t)frag*256;
                float D[256]; unpack_D(Cf + foff, D);
                bool good=true;
                for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
                if (good) res.okFrags++; else res.badFrags++;
            }
        }
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// WAVESPEC PERF: STORE=0 bin (no C store) -> pure TF timing. Correctness gate is run_wavespec_compute.
static WgpResult run_wavespec_perf(uint32_t node, const char* isaPath, int M, int N, int K,
                                   uint32_t nWG, double freq_hz,
                                   int FM, int FN, int NLOAD, int NCOMP, bool dynvgpr) {
    WgpResult res;
    const int WAVES_LAUNCH = NLOAD + NCOMP;
    const int TM = NCOMP*FM*16, TN = FN*16;
    int NTL = N / TN, MTL = M / TM, NT = N / 16, MT = M / 16, NTILES = K / 16;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not pow2\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Ashufh((size_t)M*K), Bshufh((size_t)K*N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    mbg_preshuffle_A(Ah.data(), Ashufh.data(), M, K);
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);

    uint32_t ldsBytes = wavespec_lds_bytes(FM, FN, NCOMP);
    // dyn: launch LEAN (32 VGPR = 4 granules; covers the identity/claim prologue v0..v24), then each wave
    //   s_alloc_vgpr's to its role size (loaders stay 32, compute grow to NFV). static: full NFV-sized launch.
    uint32_t vgprField = dynvgpr ? 4u : (uint32_t)wavespec_vgpr_field(FM, FN);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, (Ashufh.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true);
    GpuBuf Bd  = AllocGpu(node, (Bshufh.size() + 0xFFF) & ~0xFFFull, false, true, /*deviceLocal*/true);
    GpuBuf C   = AllocGpu(node, 0x1000, false, true, /*deviceLocal*/true);   // STORE=0: tiny (no C traffic)
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    if (!(Ad.vram && Bd.vram)) {
        fprintf(stderr, "\n*** WAVESPEC VRAM GUARD FAILED (%s): A=%d B=%d -> PCIe-fed, PERF INVALID ***\n", isaPath, Ad.vram, Bd.vram);
        abort();
    }
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ashufh.data(), Ashufh.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES_LAUNCH*32),1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(dynvgpr); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t s10v = (uint32_t)(NTL-1), s11v = (uint32_t)log2NTL;
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, (uint32_t)(NT*256), s10v, s11v,
        (uint32_t)NTILES, TOTAL, (uint32_t)(MT*256), 0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * (uint32_t)(WAVES_LAUNCH*32), 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 40.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** WAVESPEC PERF TIMEOUT (%s): live=%u maxlive=%u claim=%u ***\n", isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.tf = (res.wall > 0) ? (2.0*(double)M*N*K * freq_hz / (double)res.wall / 1e12) : 0.0;
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 HYBRID COOPERATIVE (B1 harness — HYBRID_DESIGN.md Steps 1-7). One lean FEED/CLAIM wave + P dyn
// COMPUTE waves per (1+P)-wave workgroup, shared-B through LDS. Feed wave: atomic claim -> global_load_tr
// B[k] -> ds_store into a depth-RINGD ring + publish via LDS counters (NO s_barrier). Compute wave c:
// busy-wait the counter, ds_load_b128 B from LDS + global_load_b128 its own A, WMMA x(FM*FN), deferred fp32
// store of its 32-row M-band. dyn = heterogeneous sizing: feed launches+stays lean 32, compute grow-per-batch
// to ~120 (2x4). This fuses run_wavespec_compute's (1+P) geometry + cooperative oracle partition with
// run_mbgemm's sustained-reps + compositor-yield + GENDIV (non-pow2 ml8 N) + VRAM guard. Operand prep mirrors
// run_mbgemm: A plain row-major (compute direct-loads), B preshuffled for global_load_tr (feed wave).
//
// The kernel binary (occ_kernel_coop.s, B2) bakes COOP/P/RINGD/FM/FN/BATCH/GENDIV/DYNVGPR as defsyms; this
// harness only needs them to pick the bin name, size the WG/grid/LDS/C buffers, and drive the oracle. The
// userdata layout is IDENTICAL to run_mbgemm (occ,A,B,C, KT,K,NTx256,TOTAL, magic|mask, NTL|log2, FNx256) so
// the COOP kernel reuses mbgemm's scalar-arg ABI. Per-tile = (P*FM*16) M-rows x (FN*16) shared N-cols.
// ---------------------------------------------------------------------------
struct CoopResult { bool ok=false; uint32_t maxlive=0, total=0; uint64_t okFrags=0, badFrags=0;
                    uint64_t wall=0, wallSum=0, wallMin=0, wallMax=0; uint32_t repsDone=0; };

static CoopResult run_mbcoop(uint32_t node, const char* isaPath, bool dynvgpr, uint32_t pool,
                             int M, int N, int K, int FM, int FN, int P, int RINGD,
                             bool fullCheck, bool useGenDiv=false,
                             uint32_t reps=1, double targetSecs=0.0,
                             int totalWaves=0, uint32_t ldsBytesOverride=0) {
    // DSWS (MAD-305): totalWaves>0 launches N=NCOMP+NAFEED+NBFEED waves/WG (P=NCOMP for the C-store/oracle
    //   partition); ldsBytesOverride carries the larger DSWS LDS_TOTAL_DSWS. Both default to the proven 2-role
    //   coop behavior (1+P waves, coop LDS) so every existing caller is byte-identical.
    CoopResult res;
    const int WAVES_LAUNCH = totalWaves > 0 ? totalWaves : (1 + P);   // 1 feed + P compute (coop) | N (DSWS)
    const int TM = P*FM*16, TN = FN*16;                   // WG tile: P M-bands (each FM rows-of-16) x shared FN N-cols
    int MTL = M / TM, NTL = N / TN, NT = N / 16, KT = K / 16;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if (TM == 0 || TN == 0 || MTL == 0 || NTL == 0 || (M % TM) || (N % TN)) {
        fprintf(stderr, "  [coop] tile %dx%d does not divide %dx%d (P=%d FM=%d FN=%d)\n", TM,TN, M,N, P,FM,FN); return res; }
    if (!useGenDiv && (NTL & (NTL - 1)) != 0) { fprintf(stderr, "  [coop] NTL=%d not pow2 (need GENDIV)\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;
    uint32_t magic = (uint32_t)((0x100000000ULL + (uint64_t)NTL - 1) / (uint64_t)NTL);   // GENDIV: ceil(2^32/NTL)

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};   // 1, 2, .5, -1, -2, -.5
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Bshufh((size_t)K*N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    if (!isaBytes) { fprintf(stderr, "  [coop] cannot read kernel bin '%s'\n", isaPath); return res; }
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    // ---- SAFETY PADDING (ML8_COOP_PAD_MB, default 64): map a guard tail AFTER each operand so a small dyn
    //   off-by-one global access lands in mapped VRAM (observable wrong answer) instead of a page-fault brick.
    //   The data stays at offset 0 (valid indices unchanged); the tail is a canary for OOB STORES (C). ----
    uint64_t padB = (uint64_t)(getenv("ML8_COOP_PAD_MB") ? atoi(getenv("ML8_COOP_PAD_MB")) : 64) * 1024ull * 1024ull;
    GpuBuf Ad  = AllocGpu(node, ((Ah.size() + 0xFFF) & ~0xFFFull) + padB, false, true, /*deviceLocal*/true);     // VRAM: A feed + guard
    GpuBuf Bd  = AllocGpu(node, ((Bshufh.size() + 0xFFF) & ~0xFFFull) + padB, false, true, /*deviceLocal*/true);  // VRAM: B feed + guard
    uint64_t cbytes = ((uint64_t)TOTAL * (uint64_t)(P*FM*FN*256*4) + 0xFFF) & ~0xFFFull;  // P waves x FM*FN frags x 256 f32
    GpuBuf C   = AllocGpu(node, cbytes + padB, false, true, /*deviceLocal*/true);   // VRAM: C store traffic + guard canary
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    // ---- VRAM GUARD: perf operands MUST be device-local (the PCIe-fed bug must not recur silently) ----
    if (!(Ad.vram && Bd.vram && C.vram)) {
        fprintf(stderr, "\n*** COOP VRAM GUARD FAILED (%s): operands not device-local (A=%d B=%d C=%d) -> PERF INVALID ***\n",
                isaPath, Ad.vram, Bd.vram, C.vram); abort(); }
    // ---- ADDRESS BOUNDS GATE (MANDATORY, established RULE after the 2026-06-19 page-fault incident):
    //   mirror the kernel's EXACT global load/store offset formulas and assert the worst-case (last-byte)
    //   access of A (global_load_b64), B (global_load_tr_b64), C (global_store_b128) lands IN-BUFFER for
    //   this geometry BEFORE dispatch. A bad address on the R9700 is NOT isolated -- it wedges the 6900XT
    //   desktop through the shared amdgpu driver. A formula mismatch (cf. the FM*FN hardcode bug) is caught
    //   here on the CPU, never on silicon. (This validates VALID indices; the kernel's own ti<TOTAL guards
    //   handle race-garbage runtime indices.) ----
    {
        uint64_t Asize = (uint64_t)Ah.size(), Bsize = (uint64_t)Bshufh.size(), Csz = cbytes;
        // A: A_saddr = rowblk*16*FM*K + mi*16*K + k*16 ; v8 = (lane&15)*K + colhi*8 ; b64 -> +7
        uint64_t rowblkMax = (uint64_t)(MTL-1)*P + (P-1);
        uint64_t Amax = rowblkMax*(uint64_t)16*FM*K + (uint64_t)(FM-1)*16*K + (uint64_t)(KT-1)*16
                        + (uint64_t)15*K + 8 + 7;
        // B: B_saddr = tcol*FN*256 + k*NT*256 ; v9 = lane*8 ; offset (FN-1)*256 ; b64 -> +7
        uint64_t Bmax = (uint64_t)(NTL-1)*FN*256 + (uint64_t)(KT-1)*NT*256
                        + (uint64_t)31*8 + (uint64_t)(FN-1)*256 + 7;
        // C: Cbase = ti*(P*FM*FN*1024) + cid*(FM*FN*1024) + frag*1024 ; v10 = lane*32 ; hi b128 +16 ; b128 -> +15
        uint64_t Cmax = (uint64_t)(TOTAL-1)*(uint64_t)(P*FM*FN*1024) + (uint64_t)(P-1)*(FM*FN*1024)
                        + (uint64_t)(FM*FN-1)*1024 + (uint64_t)31*32 + 16 + 15;
        bool aok = Amax < Asize, bok = Bmax < Bsize, cok = Cmax < Csz;
        printf("  [coop bounds] A last=%llu/%llu %s  B last=%llu/%llu %s  C last=%llu/%llu %s\n",
               (unsigned long long)Amax,(unsigned long long)Asize, aok?"OK":"*OOB*",
               (unsigned long long)Bmax,(unsigned long long)Bsize, bok?"OK":"*OOB*",
               (unsigned long long)Cmax,(unsigned long long)Csz,  cok?"OK":"*OOB*");
        if (!(aok && bok && cok)) {
            fprintf(stderr, "\n*** COOP ADDRESS BOUNDS GATE FAILED (%s): a kernel offset formula exceeds its "
                    "buffer for this geometry -> REFUSING to dispatch (a bad GPU address wedges the desktop). ***\n", isaPath);
            FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);  // ring not yet created
            return res;
        }
    }
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ah.data(), Ah.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    // pre-TOUCH the A/B guard tails (commit VRAM pages + clear) so an OOB LOAD returns 0 instead of first-touch-faulting
    memset((char*)Ad.ptr + ((Ah.size()+0xFFF)&~0xFFFull), 0, padB);
    memset((char*)Bd.ptr + ((Bshufh.size()+0xFFF)&~0xFFFull), 0, padB);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0; occW[6]=occW[7]=occW[8]=occW[9]=occW[10]=occW[11]=occW[12]=occW[13]=0;
    memset((char*)C.ptr + cbytes, 0, padB);   // CANARY: zero the C guard tail; any nonzero after = an OOB store landed there

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES_LAUNCH*32),1,1,0,0};   // NUM_THREAD_X = (1+P)*32 -> (1+P) waves/WG
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    // dyn: ALL waves launch lean 32 (field 4); only compute waves s_alloc_vgpr-grow. static fallback: uniform fat.
    uint32_t fatregs = (uint32_t)((32 + FM*FN*8 + FM*4 + FN*4 + 15) & ~15);   // 2x4 compute: ~120 VGPR
    uint32_t vgprField = dynvgpr ? 4u : ((fatregs / 8) & 0x3fu);
    uint32_t rsrc1 = BuildPgmRsrc1(dynvgpr); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    // LDS (Step 3 byte layout): B_ring[RINGD*FN*256] + prod_count(u32) + cons_count[P](u32) + tile_slot[3](u32)
    uint32_t ldsBytes = ldsBytesOverride > 0 ? ldsBytesOverride        // DSWS: the full LDS_TOTAL_DSWS
                        : (uint32_t)(RINGD * FN * 256 + 4 * (1 + P + 3));   // coop: B-ring + prod/cons/ti/epoch/initflag
    ldsBytes = (ldsBytes + 0x1FFu) & ~0x1FFu;             // round to 512B LDS granule
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(dynvgpr) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    // per-SIMD residency check (brick invariant, [1,1,1,1] placement): one grower/SIMD/WG -> demand = max grow.
    printf("  [coop] %dx%dx%d  WGtile=%dx%d (P=%d FM=%d FN=%d)  TOTAL=%u  pool=%u WG  RINGD=%d LDS=%uB  VGPR~%u dyn=%d RSRC2=0x%x%s\n",
           M,N,K, TM,TN, P,FM,FN, TOTAL, pool, RINGD, ldsBytes, vgprField*8, dynvgpr, rsrc2, useGenDiv?" GENDIV":"");
    uint32_t s10v = useGenDiv ? magic : (uint32_t)(NTL-1);
    uint32_t s11v = useGenDiv ? (uint32_t)NTL : (uint32_t)log2NTL;
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),   // s0:1 occ, s2:3 A
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),       // s4:5 Bshuf, s6:7 C
        (uint32_t)KT, (uint32_t)K, (uint32_t)(NT*256), TOTAL,                       // s8 KT, s9 K, s10 NTx256, s11 TOTAL
        s10v, s11v, (uint32_t)(FN*256), 0 };                                        // s12 magic|mask, s13 NTL|log2, s14 FNx256
    uint32_t dispInit = BuildDispatchInitiator();

    // SUSTAINED loop + compositor yield (verbatim run_mbgemm scaffolding; the R9700 drives 2 desktop monitors).
    const char* ydis = getenv("ML8_YIELD_DISABLE");
    bool  yieldOff   = ydis && ydis[0]=='1';
    int   yieldMs    = getenv("ML8_YIELD_MS")       ? atoi(getenv("ML8_YIELD_MS"))       : 5;
    double yieldEvery= (getenv("ML8_YIELD_EVERY_MS")? atoi(getenv("ML8_YIELD_EVERY_MS")): 100) / 1000.0;
    if (yieldMs < 0) yieldMs = 0; if (yieldEvery <= 0.0) yieldEvery = 0.1;
    const double timeoutS = 25.0;
    uint64_t spanSum=0, spanMin=~0ull, spanMax=0; uint32_t lastMaxlive=0, lastTotal=0; bool allok=true;
    double loopStart = now_s(), lastYield = loopStart; (void)lastYield; (void)yieldEvery;
    // COMPOSITOR-SAFE CHUNKING (2026-06-26): one persistent dispatch over ALL tiles hogs the GPU for seconds and
    //   starves the desktop compositor's gfx ring -> brick (the R9700 drives the displays, like the single-GPU
    //   box every ml8 user has). FIX: bound each dispatch to ML8_COOP_CHUNK output tiles (claim starts at `base`,
    //   s11=hi terminal) and YIELD between dispatches -- exactly how llama.cpp launches bounded GEMM kernels.
    //   Sub-second dispatches are proven brick-safe (every oracle run). TF sums per-chunk in-kernel gpu-clock
    //   spans (= full-shape compute time; host yields excluded). CHUNK=0 (default) -> whole shape, one dispatch
    //   (legacy; the sub-second oracle stays single-chunk too unless CHUNK < its TOTAL).
    uint32_t chunkTiles = getenv("ML8_COOP_CHUNK") ? (uint32_t)atoi(getenv("ML8_COOP_CHUNK")) : 0u;
    if (chunkTiles == 0u || chunkTiles > TOTAL) chunkTiles = TOTAL;
    uint32_t nChunks = (TOTAL + chunkTiles - 1u) / chunkTiles;
    double chunkMaxS = getenv("ML8_COOP_CHUNK_MAXS") ? atof(getenv("ML8_COOP_CHUNK_MAXS")) : 0.75;
    if (chunkTiles < TOTAL) printf("  [coop] compositor-safe: %u tiles/dispatch x %u chunks (yield %dms between; abort chunk > %.2fs)\n", chunkTiles, nChunks, yieldMs, chunkMaxS);
    for (uint32_t rep=0; ; ++rep) {
        if (targetSecs > 0.0) { if (rep >= 4 && (now_s()-loopStart) >= targetSecs) break; if (rep >= 200000u) break; }
        else if (rep >= reps) break;
        uint64_t repSpan = 0; bool repFail = false; uint32_t repMaxlive = 0;
      for (uint32_t base = 0; base < TOTAL; base += chunkTiles) {
        uint32_t chunkHi = (base + chunkTiles < TOTAL) ? (base + chunkTiles) : TOTAL;
        userdata[11] = chunkHi;   // s11 = this chunk's terminal tile (feed exits at claim>=hi; POOLTERM compute follows)
        occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=base; *fenceW=0; occW[6]=occW[7]=occW[8]=occW[9]=occW[10]=occW[11]=occW[12]=occW[13]=0;
        for (int qi=14; qi<52; ++qi) occW[qi]=0;          // DIAGFINE: 14-21 hot-step, 22 raw-ti, 23-27 init, 28-31 A-feed, 32-36 sensors, 39 gate would-win count, 40-47 DSWS2_BAILMARK per-wave bail epochs, 48 conversion-commit count
        RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
        RingPlace(ring, PM4DispatchDirectPacket(pool * (uint32_t)(WAVES_LAUNCH*32), 1, 1, dispInit));   // grid = pool WGs
        RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));
        // ML8_COOP_NOFENCE: complete on the kernel's own done-signal (occ[0]==0, feed retired) instead of the
        //   PM4 EOP fence. The cooperative kernel's lone terminal store doesn't drain without concurrent WG
        //   traffic, so s_endpgm's implicit drain stalls the EOP fence -- but the store IS issued; a longer
        //   settle lets it land before the oracle bit-checks C (a stale read fails the oracle, never a false CLEAN).
        bool nofence = getenv("ML8_COOP_NOFENCE") != nullptr;
        double settle = nofence ? (getenv("ML8_COOP_SETTLE") ? atof(getenv("ML8_COOP_SETTLE")) : 0.30) : 0.025;
        double t0 = now_s(); RingSubmit(ring);
        // STREAM occ snapshots to disk every 200ms during the poll (ML8_COOP_STREAM or any oracle run). A dyn HANG
        //   freezes the markers; the LAST snapshot before a ring-timeout/MODE1 lands on disk = the exact wedge point,
        //   surviving the GPU reset (VRAM is lost, but the log isn't). This is the brick-data-capture mechanism.
        bool streamOn = getenv("ML8_COOP_STREAM") || fullCheck;
        bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0, lastSnap=t0;
        while (true) { double now = now_s();
            if (streamOn && (now - lastSnap) >= 0.2) { lastSnap = now;
                fprintf(stderr, "[occ +%5.2fs] live%u maxlive%u claim%u end%u | INIT adm%u tmr%u lds%u flag%u rdv%u | "
                        "feedPh%u compPh%u cons%u tiles%u | feed:tr%u pub%u comp:dsB%u wm%u rawTi%u | SENS occ_b%u occ_a%u roles[%u/%u/%u] | BAIL[w1=%u w2=%u w3=%u w4=%u w5=%u w6=%u w7=%u] | fence=%s\n",
                        now-t0, occW[0],occW[1],occW[5],occW[3],
                        occW[23],occW[24],occW[25],occW[26],occW[27],
                        occW[6],occW[7],occW[10],occW[11], occW[19],occW[21],occW[15],occW[17], occW[22],
                        occW[32],occW[33],occW[34],occW[35],occW[36],
                        occW[41],occW[42],occW[43],occW[44],occW[45],occW[46],occW[47],   // DSWS2_BAILMARK per-wave bail epochs (wid1..7; wid0=claimer=occ[40])
                        (*fenceW==FENCE_VALUE)?"FIRED":"--"); fflush(stderr);
            }
            if (occW[1] > 0) admitted = true;
            uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
            bool ff = (*fenceW == FENCE_VALUE) || nofence;
            if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>settle) { done=true; break; }
            if (now - t0 > timeoutS) break;
        }
        if (!done) {
            fprintf(stderr, "\n*** COOP TIMEOUT (%s rep %u): live=%u maxlive=%u claim=%u ***\n",
                    isaPath, rep, occW[0], occW[1], occW[5]);
            fprintf(stderr, "    occ[0..11]:");
            for (int qi = 0; qi < 14; ++qi) fprintf(stderr, " [%d]=%u", qi, occW[qi]);
            fprintf(stderr, "\n    DIAG: feedPhase(occ6)=%u computePhase(occ7)=%u feedTi(occ8)=%u computeTi(occ9)=%u\n",
                    occW[6], occW[7], occW[8], occW[9]);
            // DIAGFINE per-instruction wedge localization (max GLOBAL step each sub-step reached). The FIRST place
            //   feed's chain stalls (slotgate->loadtr->dsstore->publish) and compute's (prodwait->dsloadB->Arelease->wmma)
            //   pinpoints the exact instruction the dyn brick wedges on.
            fprintf(stderr, "    DIAGFINE feed   : slotok[18]=%u loadtr[19]=%u dsstore[20]=%u publish[21]=%u\n",
                    occW[18], occW[19], occW[20], occW[21]);
            fprintf(stderr, "    DIAGFINE compute: prodwait[14]=%u dsloadB[15]=%u Aload[16]=%u wmma[17]=%u consRel[10]=%u  rawTiMax[22]=%u\n",
                    occW[14], occW[15], occW[16], occW[17], occW[10], occW[22]);
            fprintf(stderr, "    DIAGFINE Afeed  : reached[28]=%u Aload[29]=%u dsstore[30]=%u publish[31]=%u  | Bphase(occ6)=%u Bti(occ8)=%u\n",
                    occW[28], occW[29], occW[30], occW[31], occW[6], occW[8]);
            fprintf(stderr, "    DIAGINIT        : adm[23]=%u tmr[24]=%u ldsinit[25]=%u initflag[26]=%u rdv[27]=%u\n",
                    occW[23], occW[24], occW[25], occW[26], occW[27]);
            fprintf(stderr, "    DSWS sensors    : occ_b[32]=%u occ_a[33]=%u nComp[34]=%u nAfeed[35]=%u nBfeed[36]=%u  gateWin[39]=%u\n",
                    occW[32], occW[33], occW[34], occW[35], occW[36], occW[39]);
            fprintf(stderr, "    DSWS2 BAILMARK  : per-wave last-bailed epoch  w1=%u w2=%u w3=%u w4=%u w5=%u w6=%u w7=%u  (all==hung epoch => visibility; ONE stale => that wave is the STRAGGLER)\n",
                    occW[41], occW[42], occW[43], occW[44], occW[45], occW[46], occW[47]);
            repFail=true; break;
        }
        uint32_t gs=occW[2], ge=occW[3];
        uint64_t span = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs);
        repSpan += span; if (occW[1] > repMaxlive) repMaxlive = occW[1]; lastTotal=occW[4];
        // YIELD to the compositor between every bounded dispatch -- THE brick fix. Each chunk is sub-second;
        //   this gap is the compositor's unconditional render window. Cheap (~yieldMs); excluded from the TF span.
        if (!yieldOff && yieldMs > 0) { struct timespec ts = { yieldMs/1000, (long)(yieldMs%1000)*1000000L }; nanosleep(&ts, nullptr); }
        // COMPOSITOR SAFETY (predict-and-stop): a dispatch that ran longer than chunkMaxS is a brick risk ->
        //   abort the remaining chunks rather than keep hammering multi-second dispatches on the display card.
        if ((now_s() - t0) > chunkMaxS) {
            fprintf(stderr, "  [coop] WARN chunk @base%u wall %.2fs > %.2fs cap -> ABORT remaining (raise ML8_COOP_CHUNK_MAXS or lower ML8_COOP_CHUNK)\n", base, now_s()-t0, chunkMaxS);
            repFail = true; break;
        }
      }
        if (repFail) { allok = false; break; }
        spanSum += repSpan; if (repSpan<spanMin) spanMin=repSpan; if (repSpan>spanMax) spanMax=repSpan;
        lastMaxlive = repMaxlive; res.repsDone = rep+1;
    }
    if (!allok) {
        // BRICK FIX (multi-WG teardown): a timed-out coop run = hung/non-idle queue. Do NOT
        //   hsaKmtDestroyQueue it -- that returns status 1 AND wedges the GPU (the recurring brick).
        //   Mirror run_dvgpr_occ's hang path: leave the queue + buffers for process-exit reclaim.
        fprintf(stderr, "    [teardown] coop run did not complete -> NOT destroying queue (brick-avoidance; process-exit reclaims).\n");
        return res;
    }
    // TEARDOWN SAFETY (multi-WG brick fix): the queue is IDLE only once the EOP RELEASE_MEMORY fence has
    //   fired. ML8_COOP_NOFENCE completes on live0 for ORACLE TIMING, but destroying a queue whose EOP is
    //   still pending = hsaKmtDestroyQueue status 1 + GPU wedge (the pool=2 brick). Wait (bounded) for the
    //   real fence -- this also guarantees C is fully drained before the oracle reads it below. If it never
    //   fires (this kernel's terminal store drains the EOP unreliably), leak the queue + let process-exit
    //   reclaim it (the proven-safe path for a non-idle queue).
    { double tw = now_s(); struct timespec ts = {0, 2000000L};
      while (*fenceW != FENCE_VALUE && (now_s() - tw) < 5.0) nanosleep(&ts, nullptr); }
    bool queueIdle = (*fenceW == FENCE_VALUE);
    if (!queueIdle) fprintf(stderr, "  [teardown] WARN: EOP fence never fired in 5s; queue NON-IDLE -> NOT destroying (process-exit reclaims). Brick-avoidance.\n");
    // DSWS Phase-2 sensor readback (always print on clean exit -- a fast oracle finishes before the 200ms
    //   stream OR the timeout dump ever fires, so this is the guaranteed observation of the LAST chunk's
    //   final sensor sample). occ_b/occ_a must sit in [0,RINGD] (not pinned 0/RINGD); roles must = launch mix.
    if (totalWaves > 0) {
        fprintf(stderr, "  [dsws sensors @clean-exit] occ_b=%u occ_a=%u  roles[nComp=%u nAfeed=%u nBfeed=%u]  gateWin(c->Afeed)[39]=%u  (last chunk)\n",
                occW[32], occW[33], occW[34], occW[35], occW[36], occW[39]);
        fprintf(stderr, "  [dsws CONVERSIONS] committed role-switches this run = %u  (occ[48]; >0 => waves ADAPTIVELY switched role)\n",
                occW[48]);
    }
    res.ok = true; res.maxlive = lastMaxlive; res.total = lastTotal;
    res.wall = spanSum / (res.repsDone ? res.repsDone : 1);   // mean per-rep span
    res.wallSum = spanSum; res.wallMin = spanMin; res.wallMax = spanMax;

    // ---- CANARY scan: did any C store land past the data region (in the guard tail)? The byte offset of the
    //   first OOB word reveals the OVERRUN UNIT: ~32B=per-lane, ~1KB=per-frag, ~8KB=per-tile (P*FM*FN*1024). ----
    { const uint32_t* tail = (const uint32_t*)((const char*)C.ptr + cbytes);
      uint64_t words = padB/4, firstNZ = ~0ull, lastNZ = 0, nzCount = 0;
      for (uint64_t w=0; w<words; ++w) if (tail[w]) { if (firstNZ==~0ull) firstNZ=w; lastNZ=w; ++nzCount; }
      if (nzCount) fprintf(stderr, "  [canary] *** C OOB STORE detected: %llu words written into guard tail; first at +%llu B past C-end (%.1f tiles / %.1f frags), last +%llu B ***\n",
                           (unsigned long long)nzCount, (unsigned long long)(firstNZ*4),
                           (double)(firstNZ*4)/(double)(P*FM*FN*1024), (double)(firstNZ*4)/1024.0, (unsigned long long)(lastNZ*4));
      else fprintf(stderr, "  [canary] C guard tail clean (no OOB store past C-end).\n"); }

    // ---- ORACLE (Step 7d): every frag of every (tile, compute-wave) vs chained wmma_ref. The cooperative
    //      partition: compute wave c owns M-rows [c*FM*16 .. ), ALL share the FN*16 N-cols. A fence/ordering
    //      bug in the busy-wait protocol surfaces here as a STALE-B read -> BAD frags (Step 3 race class). ----
    const float* Cf = (const float*)C.ptr;
    std::vector<int> tbad(TOTAL, 0);   // per-tile bad-frag count -> reveals terminal-only vs pattern vs warmup
    std::vector<int> fbad(FM*FN, 0);   // per-frag-index bad count -> reveals WHICH (mi,ni) frag is wrong
    uint32_t tstride = fullCheck ? 1u : (TOTAL > 64 ? TOTAL/64u : 1u);
    for (uint32_t ti = 0; ti < TOTAL; ti += tstride) {
        int trow, tcol;
        if (useGenDiv) { trow = (int)(ti / (uint32_t)NTL); tcol = (int)(ti % (uint32_t)NTL); }   // exact div/mod
        else           { trow = ti >> log2NTL;  tcol = ti & (NTL - 1); }
        for (int cid = 0; cid < P; ++cid) {
            for (int mi = 0; mi < FM; ++mi) for (int ni = 0; ni < FN; ++ni) {
                int rowbase = trow*TM + cid*(FM*16) + mi*16;
                int colbase = tcol*TN + ni*16;
                float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
                uint8_t Ablk[256], Bblk[256]; float Dout[256];
                for (int kt = 0; kt < KT; ++kt) {
                    for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                        Ablk[i*16+j] = Ah[(size_t)(rowbase+i)*K + (kt*16+j)];
                        Bblk[i*16+j] = Bh[(size_t)(kt*16+i)*N + (colbase+j)];
                    }
                    wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
                    for (int i=0;i<256;i++) Cacc[i]=Dout[i];
                }
                int frag = mi*FN + ni;
                size_t foff = (size_t)ti*(size_t)(P*FM*FN*256) + (size_t)cid*(size_t)(FM*FN*256) + (size_t)frag*256;
                float D[256]; unpack_D(Cf + foff, D);
                bool good=true;
                for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
                if (good) res.okFrags++; else { res.badFrags++; tbad[ti]++; fbad[mi*FN+ni]++; }
            }
        }
    }
    if (res.badFrags > 0) {   // which tiles failed? terminal-only (ti=TOTAL-1) vs pattern vs warmup (ti=0..)
        fprintf(stderr, "  [badmap] bad tiles (ti:cnt):");
        for (uint32_t ti = 0; ti < TOTAL; ++ti) if (tbad[ti]) fprintf(stderr, " %u:%d", ti, tbad[ti]);
        fprintf(stderr, "\n  [badmap] bad frags (mi,ni:cnt):");
        for (int mi=0;mi<FM;++mi) for (int ni=0;ni<FN;++ni) if (fbad[mi*FN+ni]) fprintf(stderr, " (%d,%d):%d", mi, ni, fbad[mi*FN+ni]);
        fprintf(stderr, "\n");
    }
    // GUARDED TEARDOWN: only destroy a CONFIRMED-IDLE queue (queueIdle = EOP fence fired in the wait above).
    //   Destroying a non-idle queue = hsaKmtDestroyQueue status 1 + GPU wedge; if not idle, leak the queue +
    //   buffers and let process-exit reclaim them (the proven-safe path for a non-idle queue).
    if (queueIdle) {
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    }
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 DSWS v2 substrate launch (Task A8, PLAN_DSWS_SUBSTRATE_V2.md). Dispatches occ_kernel_dsws.s: a
// pinned wid-0 claimer broadcasts super-tiles from a pool; B-feed/A-feed/compute waves drain LDS atomic
// claim counters against resident-in-LDS A/B for the current super-tile; compute flushes fp32 partials via
// global_atomic_add_f32 (split-K segments accumulate into the SAME C cell, ksi-independent address).
//
// This MIRRORS run_mbcoop's PM4 launch/chunk/teardown/canary infrastructure byte-for-byte where the
// protocols line up (buffer alloc, VRAM guard, address bounds gate, compositor-safe chunking, fence/settle
// poll, guarded teardown, C-guard-tail canary) -- re-keyed to the v2 contract:
//   * occ[20] (=occW[5], byte offset 20) is the SAME global claim counter coop uses for output-tile claims;
//     v2 claims super-tile ids (`sti`) through it instead.
//   * The v2 C address formula (ti*(G*FM*FN*1024) + r*(FM*FN*1024) + frag*1024 + lane*32 + e*4) is coop's
//     formula with P (compute-wave count) replaced by G (super-tile M-extent) and cid replaced by r
//     (claimed rowblk) -- so the oracle reuses unpack_D/oracle_compare exactly, decoding r where coop
//     decoded cid.
//
// *** FIX 1 (round-table Opus+Codex pass) ***
//   The original v2 kernarg contract called for COMPUTE_PGM_RSRC2.USER_SGPR=18 (s0..s17 hardware-preloaded)
//   so n_kseg/TOTAL_super/magic_kseg could ride in as s15/s16/s17, written via a SECOND SET_SH_REG packet
//   at register (COMPUTE_USER_DATA_0 + 16). That was never deliverable: every OTHER kernel in this harness
//   uses <=15 user SGPRs, and the project's own pinned PM4 register reference (dvgpr_pm4/ref/gfx_7_2_d.h)
//   defines COMPUTE_USER_DATA_0..15 ONLY (16 registers, covering s0..s15) -- there is no register defined
//   for s16/s17 anywhere in this raw-PM4 path, and RESULT_WGGEMM.md's "raw-PM4 TGID is unavailable" probe
//   already found SGPR delivery beyond s15 under raw-PM4 (CP-direct dispatch, MES bypassed) reads constant
//   garbage, not a controllable value. On top of that, the per-chunk override of kernarg slot s16 (the
//   chunk terminal bound) silently collided with s16 also being the would-be TOTAL_super kernarg -- two
//   unrelated meanings on the same undeliverable slot.
//   FIX 1 drops s15/s16/s17 from the kernarg contract entirely (USER_SGPR=15, s0..s14 only, ONE SET_SH_REG
//   packet of 16 registers like every other proven path here, index 15 unused/padding). n_kseg is now
//   DERIVED in-kernel from KT (s8); the chunk terminal bound is now MEMORY-CARRIED via occ[24] (occW[6],
//   written once per chunk below) instead of riding in an undeliverable kernarg slot. See the KERNARG
//   CONTRACT block at the top of occ_kernel_dsws.s for the full new scheme.
// ---------------------------------------------------------------------------
struct Dsws2Result {
    bool ok = false;                 // true iff the run completed cleanly (fence fired, occ[0]==0) AND badFrags==0
    uint64_t okFrags = 0, badFrags = 0;
    double maxRel = 0.0;
    uint32_t occ0 = 0;                // live-counter readback at last clean completion (expect 0)
    uint32_t occClaim = 0;            // occ[20] readback at last clean completion (global claim counter)
    double tf = 0.0;                  // TFPROBE throughput (2*M*N*K / summed GPU-tick span); 0 if the bin has no tick capture
    uint64_t wall = 0;                // summed per-chunk GPU-tick span (occ[3]-occ[2], device busy ticks, excl host gaps)
};

static Dsws2Result run_dsws2(uint32_t node, const char* isaPath,
                              uint32_t nComp, uint32_t nAfeed, uint32_t nBfeed,
                              int Gv, int SEGKv, int FMc, int FNc,
                              int Mo, int No, int Ko,
                              float orel, float oabs, double freq_hz) {
    Dsws2Result res;
    const uint32_t WAVES_LAUNCH = nComp + nAfeed + nBfeed;
    const int TMsuper = Gv*16*FMc, TN = FNc*16;          // super-tile M rows, N-panel cols
    if (TMsuper == 0 || TN == 0 || SEGKv <= 0 || Ko <= 0 || (Mo % TMsuper) || (No % TN) || (Ko % SEGKv)) {
        fprintf(stderr, "  [dsws2] geometry %dx%dx%d does not divide cleanly (G=%d SEGK=%d FM=%d FN=%d)\n",
                Mo, No, Ko, Gv, SEGKv, FMc, FNc);
        return res;
    }
    const int MTLsuper = Mo / TMsuper, NTL = No / TN, NT = No / 16, KT = Ko / 16;
    const int KSEG_STEPS = SEGKv / 16;
    const int n_kseg = Ko / SEGKv;
    if (MTLsuper == 0 || NTL == 0 || n_kseg == 0 || KSEG_STEPS == 0) {
        fprintf(stderr, "  [dsws2] degenerate geometry (MTLsuper=%d NTL=%d n_kseg=%d KSEG_STEPS=%d)\n",
                MTLsuper, NTL, n_kseg, KSEG_STEPS);
        return res;
    }
    // FIX 1(k): the kernel now DERIVES n_kseg in-kernel as KT >> NKSEG_SHIFT (a plain shift) and uses it
    //   *** LIFTED 2026-07-14: the kernel no longer needs a power-of-two n_kseg. ***
    //   It used to derive shift = s_ff1(n_kseg) (the EXACT log2), which only exists for a power of two.
    //   It now uses shift = CEIL(log2 n_kseg) and packs ksi into a power-of-2-SIZED field big enough to
    //   HOLD it: sti = (t<<shift) | ksi with ksi < n_kseg <= 2^shift. Decode stays a pure AND/SHIFT.
    //   That one line was making 10 of our 18 REAL shapes illegal -- every mlambaformer GEMM but the
    //   router (K=768 -> n_kseg=24, K=1536 -> 48) and most of ml8 dense (K=9216 -> 288, K=2560 -> 80).
    //   What IS still required: JDEPTH must be a power of two AND divide n_kseg (a J-carrier walks J
    //   consecutive segments of one tile; if J does not divide n_kseg it walks off the tile's end).
    if (n_kseg <= 0) {
        fprintf(stderr, "  [dsws2] *** REFUSE: n_kseg=%d (K=%d must be a positive multiple of SEGK=%d) ***\n",
                n_kseg, Ko, SEGKv);
        return res;
    }
    const uint32_t TOTAL = (uint32_t)MTLsuper * (uint32_t)NTL;                       // coop-compat output-tile count (C sizing)
    const uint64_t TOTAL_super = (uint64_t)MTLsuper * (uint64_t)NTL * (uint64_t)n_kseg;  // super-tile pool size
    const uint32_t magic = (uint32_t)((0x100000000ULL + (uint64_t)NTL - 1) / (uint64_t)NTL);          // ceil(2^32/NTL)
    const uint32_t magicTotal = (uint32_t)((0x100000000ULL + (uint64_t)TOTAL - 1) / (uint64_t)TOTAL); // ceil(2^32/TOTAL) for KMAJOR ksi=sti/TOTAL

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)Mo*Ko), Bh((size_t)Ko*No), Bshufh((size_t)Ko*No);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)Ko) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)No)*3) % 6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), Ko, No);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    if (!isaBytes) { fprintf(stderr, "  [dsws2] cannot read kernel bin '%s'\n", isaPath); return res; }
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    // ---- TRACE: per-super-tile time-series buffer (DSWS2_TRACE=1; requires a TRACE=1 kernel bin + single chunk).
    //   The claimer appends a 16-u32 row per super-tile (indexed by SEGCNT) capturing the live role mix, ring
    //   occupancy peak, conversions, and vresv. Host reads it back to CSV after the run. ----
    const bool traceOn = getenv("DSWS2_TRACE") != nullptr;
    uint32_t traceMaxRows = 0; GpuBuf traceBuf{}; volatile uint32_t* traceW = nullptr;
    if (traceOn) {
        uint64_t want = TOTAL_super + 64; traceMaxRows = (uint32_t)(want > (1u<<21) ? (1u<<21) : want);
        traceBuf = AllocGpu(node, ((uint64_t)traceMaxRows*64 + 0xFFF)&~0xFFFull, false, true);
        traceW = (volatile uint32_t*)traceBuf.ptr;
        if (!traceBuf.ptr) { fprintf(stderr, "  [dsws2 trace] buffer alloc FAILED -> trace disabled\n"); traceW = nullptr; }
        else memset((void*)traceW, 0, traceBuf.size);
    }
    // SAFETY PADDING (mirrors run_mbcoop): a guard tail after each operand so a small dyn off-by-one global
    //   access lands in mapped VRAM (observable wrong answer) instead of a page-fault brick.
    uint64_t padB = (uint64_t)(getenv("ML8_COOP_PAD_MB") ? atoi(getenv("ML8_COOP_PAD_MB")) : 64) * 1024ull * 1024ull;
    GpuBuf Ad = AllocGpu(node, ((Ah.size()+0xFFF)&~0xFFFull) + padB, false, true, /*deviceLocal*/true);
    GpuBuf Bd = AllocGpu(node, ((Bshufh.size()+0xFFF)&~0xFFFull) + padB, false, true, /*deviceLocal*/true);
    uint64_t cbytes = ((uint64_t)TOTAL * (uint64_t)((uint32_t)Gv*FMc*FNc*1024) + 0xFFF) & ~0xFFFull;  // TOTAL output tiles x G*FM*FN frags x 256 f32
    GpuBuf C = AllocGpu(node, cbytes + padB, false, true, /*deviceLocal*/true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    if (!(Ad.vram && Bd.vram && C.vram)) {
        fprintf(stderr, "\n*** DSWS2 VRAM GUARD FAILED (%s): operands not device-local -> PERF/SAFETY INVALID ***\n", isaPath);
        abort();
    }
    // ---- ADDRESS BOUNDS GATE (MANDATORY, mirrors run_mbcoop's gate). Formulas re-derived from
    //   occ_kernel_dsws.s's ASTAGE/BSTAGE/.Lcompute address math (G replaces coop's P; r replaces cid). ----
    {
        uint64_t Asize = (uint64_t)Ah.size(), Bsize = (uint64_t)Bshufh.size(), Csz = cbytes;
        uint64_t rowblkAbsMax = (uint64_t)MTLsuper * (uint64_t)Gv - 1ull;
        uint64_t Amax = rowblkAbsMax*(uint64_t)16*FMc*Ko + (uint64_t)(FMc-1)*16*Ko
                        + (uint64_t)(n_kseg-1)*SEGKv + (uint64_t)(KSEG_STEPS-1)*16
                        + (uint64_t)15*Ko + 8 + 7;
        uint64_t Bmax = (uint64_t)(NTL-1)*FNc*256 + (uint64_t)(n_kseg-1)*KSEG_STEPS*(uint64_t)NT*256
                        + (uint64_t)(FNc-1)*256 + (uint64_t)(KSEG_STEPS-1)*(uint64_t)NT*256
                        + (uint64_t)31*8 + 7;
        uint64_t Cmax = (uint64_t)(TOTAL-1)*(uint64_t)Gv*FMc*FNc*1024 + (uint64_t)(Gv-1)*(uint64_t)FMc*FNc*1024
                        + (uint64_t)(FMc*FNc-1)*1024 + (uint64_t)31*32 + (uint64_t)7*4 + 3;
        bool aok = Amax < Asize, bok = Bmax < Bsize, cok = Cmax < Csz;
        printf("  [dsws2 bounds] A last=%llu/%llu %s  B last=%llu/%llu %s  C last=%llu/%llu %s\n",
               (unsigned long long)Amax,(unsigned long long)Asize, aok?"OK":"*OOB*",
               (unsigned long long)Bmax,(unsigned long long)Bsize, bok?"OK":"*OOB*",
               (unsigned long long)Cmax,(unsigned long long)Csz,  cok?"OK":"*OOB*");
        if (!(aok && bok && cok)) {
            fprintf(stderr, "\n*** DSWS2 ADDRESS BOUNDS GATE FAILED (%s) -> REFUSING to dispatch. ***\n", isaPath);
            FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
            return res;
        }
    }
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    memcpy(Ad.ptr, Ah.data(), Ah.size());
    memcpy(Bd.ptr, Bshufh.data(), Bshufh.size());
    memset((char*)Ad.ptr + ((Ah.size()+0xFFF)&~0xFFFull), 0, padB);
    memset((char*)Bd.ptr + ((Bshufh.size()+0xFFF)&~0xFFFull), 0, padB);
    volatile uint32_t* occW   = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    memset((void*)occW, 0, occ.size);   // host zero-init: occ[0] live-count, occ[20] claim-counter, all reserved words
    *fenceW = 0;
    memset((char*)C.ptr + cbytes, 0, padB);   // CANARY: zero the C guard tail (any nonzero after run = an OOB store)
    // FIX 2: the kernel's compute role accumulates into C via global_atomic_add_f32 (split-K segments add
    //   into the SAME C cell) -- it never initializes a cell, so the host MUST zero the C data region before
    //   ANY dispatch (occ_kernel_dsws.s's KERNARG CONTRACT comment: "HOST MUST MEMSET C=0"). This was
    //   missing entirely (only the guard-tail canary was zeroed above). ONCE here, before the chunk loop --
    //   NOT per chunk, so split-K segments claimed across separate chunk dispatches still accumulate.
    memset((char*)C.ptr, 0, cbytes);

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES_LAUNCH*32),1,1,0,0};   // NUM_THREAD_X = WAVES_LAUNCH*32 -> WAVES_LAUNCH waves/WG
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    // DYNVGPR is baked DYNVGPR=1 into this bin's compute role (build_dsws.sh mk2 never overrides it) -- dyn-VGPR
    //   MUST be armed (RSRC2 bit6) to match; there is no static v2 bin to fall back to if it weren't.
    const uint32_t vgprField = 4u;   // lean 32-VGPR launch; compute waves s_alloc_vgpr-grow per claimed rowblk
    uint32_t rsrc1 = BuildPgmRsrc1(true); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    // FIX 1 pools: DSWS2_FLOW=1 -> N-deep flow pool (LDS_TOTAL_FLOW = 256 + POOL_N*OPSTRIDE);
    //   DSWS2_RING=1 -> D=2 ring (33024); neither -> single-slot occ_kernel_dsws.s (16640, byte-identical).
    uint32_t poolSlots = 1u;
    if (getenv("DSWS2_FLOW"))      poolSlots = getenv("FLOW_POOL_N") ? (uint32_t)atoi(getenv("FLOW_POOL_N")) : 3u;
    else if (getenv("DSWS2_RING")) poolSlots = 2u;
    uint32_t operandBytes = (uint32_t)(FNc*16*SEGKv) + (uint32_t)((uint32_t)Gv*16*FMc*SEGKv);   // per-slot = 16384
    // FIX 1 STAGGER: flow adds a per-rowblk fp32 reduction accumulator pool (ACC_N banks x FM*FN*1024B) AFTER
    //   the operand pool. Must match the kernel's ACC_BASE/ACC_STRIDE/ACC_N (DSWS2_ACC_N, default 1; 0 for ring/single).
    uint32_t accN = getenv("DSWS2_FLOW") ? (getenv("DSWS2_ACC_N") ? (uint32_t)atoi(getenv("DSWS2_ACC_N")) : 1u) : 0u;
    uint32_t accBytes = accN * (uint32_t)(FMc*FNc*1024);
    // *** CO-CHANGE: kOpBase MUST MATCH OP_BASE in occ_kernel_dsws_flow.s. ***
    //   Raised 256->512 (2026-07-13). At 256 the kernel's per-slot control blocks (SLOTC_BASE=148 + N*32)
    //   overran the operand pool for N>3, hard-capping POOL_N at 3 -- and POOL_N is the ceiling on how far
    //   ASSIGN may lead DRAIN, i.e. it caps ALL in-flight work. If this disagrees with the kernel the host
    //   under-allocates LDS and the workgroup SILENTLY NEVER LAUNCHES (all counters read 0 -- looks like a
    //   hang, is really a dispatch that could not fit). Do not "fix" one side alone.
    constexpr uint32_t kOpBase = 512u;
    static_assert(kOpBase >= 148u + 4u*32u, "kOpBase must clear SLOTC_BASE + POOL_N*SLOTC_STRIDE");
    uint32_t ldsBytesRaw = kOpBase + poolSlots * operandBytes + accBytes;   // WOFLUSH POOL4: 512 + 4*8192 = 33280
    if (ldsBytesRaw > 65536u) {   // the kernel .errors on this at assemble time; the host must not sail past it
        fprintf(stderr, "  [dsws2] FATAL: LDS %uB > 65536 (POOL_N=%u accN=%u). The WG would silently never launch.\n"
                        "          Under WOFLUSH=1 pass DSWS2_ACC_N=0 -- the kernel allocates NO accumulator banks.\n",
                ldsBytesRaw, poolSlots, accN);
        return res;
    }
    uint32_t ldsU=0, ldsA=0, ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytesRaw, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(true) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;   // USER_SGPR=15 (FIX 1h: dropped s15..s17)
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    printf("  [dsws2] %dx%dx%d  super-tile=%dx%d (G=%d FM=%d FN=%d)  TOTAL=%u TOTAL_super=%llu n_kseg=%d  "
           "waves/WG=%u(=%uc%ua%ub)  LDS=%uB(alloc %uB)  VGPR~%u dyn=1 RSRC2=0x%x\n",
           Mo,No,Ko, TMsuper,TN, Gv,FMc,FNc, TOTAL, (unsigned long long)TOTAL_super, n_kseg,
           WAVES_LAUNCH, nComp,nAfeed,nBfeed, ldsBytesRaw, ldsA, vgprField*8, rsrc2);

    // FIX 1(i): 15 kernargs (s0..s14) only -- n_kseg/TOTAL_super/magic_kseg dropped (derived in-kernel /
    //   memory-carried via occ[24], see occ_kernel_dsws.s KERNARG CONTRACT). Array is still 16 wide to match
    //   every other proven path's single 16-register SET_SH_REG packet; index 15 is unused padding (lands
    //   in the hardware's TGID_X slot, which this kernel does not read).
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),   // s0:1 occ, s2:3 A
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),       // s4:5 Bshuf, s6:7 C
        (uint32_t)KT, (uint32_t)Ko, (uint32_t)(NT*256), TOTAL,                      // s8 KT, s9 K(bytes/row), s10 NTx256, s11 TOTAL
        magic, (uint32_t)NTL, (uint32_t)(FNc*256), 0u };                            // s12 magic, s13 NTL, s14 FNx256, [15] unused
    uint32_t dispInit = BuildDispatchInitiator();

    const uint32_t poolD = getenv("ML8_POOL") ? (uint32_t)atoi(getenv("ML8_POOL")) : 64u;
    const uint32_t pool = poolD < 64u ? poolD : 64u;
    const char* ydis = getenv("ML8_YIELD_DISABLE"); bool yieldOff = ydis && ydis[0]=='1';
    int yieldMs = getenv("ML8_YIELD_MS") ? atoi(getenv("ML8_YIELD_MS")) : 5;
    if (yieldMs < 0) yieldMs = 0;
    double yieldEvery = (getenv("ML8_YIELD_EVERY_MS") ? atoi(getenv("ML8_YIELD_EVERY_MS")) : 100) / 1000.0;  // proven run_mbgemm cadence
    if (yieldEvery <= 0.0) yieldEvery = 0.1;
    const double timeoutS = 25.0;
    // COMPOSITOR-SAFE CHUNKING (mirrors run_mbcoop): bound each dispatch to ML8_COOP_CHUNK super-tiles
    //   (claim starts at occ[20]=base, terminal bound occ[24]/occW[6]=chunkHi -- FIX 1(j), memory-carried
    //   since there is no deliverable kernarg slot for it) and yield between dispatches.
    uint64_t chunkTilesEnv = getenv("ML8_COOP_CHUNK") ? (uint64_t)atoll(getenv("ML8_COOP_CHUNK")) : 0ull;
    // FIX 1 STAGGER: the flow write-once kernel claims occ[20] as whole TILES (a WG owns a tile's n_kseg
    //   segments so its per-WG LDS banks sum a full tile); every other dsws2 path claims super-tiles.
    const uint64_t claimTotal = getenv("DSWS2_FLOW") ? (uint64_t)TOTAL : TOTAL_super;
    uint64_t chunkTiles = (chunkTilesEnv == 0ull || chunkTilesEnv > claimTotal) ? claimTotal : chunkTilesEnv;
    uint64_t nChunks = (claimTotal + chunkTiles - 1ull) / chunkTiles;
    double chunkMaxS = getenv("ML8_COOP_CHUNK_MAXS") ? atof(getenv("ML8_COOP_CHUNK_MAXS")) : 0.75;
    if (chunkTiles < TOTAL_super) printf("  [dsws2] compositor-safe: %llu super-tiles/dispatch x %llu chunks (yield %dms between; abort chunk > %.2fs)\n",
                                          (unsigned long long)chunkTiles, (unsigned long long)nChunks, yieldMs, chunkMaxS);
    bool streamOn = getenv("ML8_COOP_STREAM") != nullptr;
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    bool allok = true; uint32_t lastOcc0 = 0, lastOcc20 = 0; uint32_t totalConv = 0;  // occ[48] conv-commit count, summed across chunks (reset per chunk)
    uint64_t sumSpan = 0; uint32_t spanChunks = 0; bool tfMissing = false;  // TFPROBE: summed per-chunk GPU-tick span (occ[3]-occ[2]); tfMissing => bin has no tick capture
    // SUSTAINED (DSWS2_REPS>1): re-run the whole chunked GEMM back-to-back, buffers reused, C re-zeroed per rep
    //   (split-K atomic-adds into C, so a repeated pass without reset would double it). Spans sum across ALL
    //   reps -> TF is over reps*(2MNK) work / total busy ticks (warm-clock steady state, not a cold ms blip).
    //   Per-rep span min/max -> the TF spread (glass-flat vs jittery), the trustworthiness signal.
    uint32_t dswsReps = getenv("DSWS2_REPS") ? (uint32_t)atoi(getenv("DSWS2_REPS")) : 1u;
    if (dswsReps < 1u) dswsReps = 1u;
    double dswsTarget = getenv("DSWS2_TARGET_SECS") ? atof(getenv("DSWS2_TARGET_SECS")) : 0.0;  // >0: rep until this many wall-secs
    double repT0 = now_s();
    uint64_t repSpanMin = ~0ull, repSpanMax = 0; uint32_t repsDone = 0;
    for (uint32_t rep = 0; ; ++rep) {
      if (dswsTarget > 0.0) { if (rep > 0 && (now_s() - repT0) >= dswsTarget) break; }   // duration-bounded
      else                  { if (rep >= dswsReps) break; }                              // count-bounded
      if (rep > 0) memset((char*)C.ptr, 0, cbytes);   // split-K accumulation reset before each repeated pass
      uint64_t repSpanBase = sumSpan;
      // ML8_CHUNK_DIAG: per-chunk wall + STAGINSTR delta (coast/computed/feed/grow-fail). occ[70..73] are
      //   OUTSIDE the per-chunk memset (occ[0..63]) so they accumulate; snapshot before each chunk for the delta.
      //   A slow chunk with grow-fail/coast spiking + computed crawling == VGPR-starvation churn (compositor
      //   held the SIMD pool during the inter-chunk yield); a slow chunk that is mostly `computed` == real work.
      const bool chunkDiag = getenv("ML8_CHUNK_DIAG") != nullptr;
    for (uint64_t base = 0; base < claimTotal; base += chunkTiles) {
        uint64_t chunkHi = (base + chunkTiles < claimTotal) ? (base + chunkTiles) : claimTotal;
        uint32_t diagPrevCoast = occW[70], diagPrevComp = occW[71], diagPrevFeed = occW[72], diagPrevGF = occW[73];
        memset((void*)occW, 0, 0x100);        // re-zero the control region (occ[0] live, occ[20] claim, reserved) each chunk
        occW[5] = (uint32_t)base;             // occ[20] (=occW[5]) claim counter starts at this chunk's base sti
        occW[6] = (uint32_t)chunkHi;          // FIX 1(j): occ[24] (=occW[6]) = this chunk's terminal sti bound (memory-carried)
        occW[2] = 0xFFFFFFFFu;                // TFPROBE: min-sentinel for the entry-tick atomic_min (occ[2]); occ[3] stays 0 (max)
        occW[62] = magicTotal;                // KMAJOR: magic(TOTAL) for the ksi=sti/TOTAL decode (ignored unless KMAJOR bin)
        if (traceW) { uint64_t tva=(uint64_t)traceBuf.ptr;   // TRACE: (re-)publish buffer VA + cap (memset above wiped occ[52..54])
                      occW[52]=(uint32_t)tva; occW[53]=(uint32_t)(tva>>32); occW[54]=traceMaxRows; }
        *fenceW = 0;
        RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
        RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));          // s0..s14 (FIX 1i: ONE packet, like every proven path)
        RingPlace(ring, PM4DispatchDirectPacket(pool * (uint32_t)(WAVES_LAUNCH*32), 1, 1, dispInit));   // grid = pool WGs
        // NOTE: CS_PARTIAL_FLUSH here was tried (2026-07-05) and STALLS too -- the CP-level terminal-store/wave
        //   drain is genuinely stuck at 16 waves; no host packet (drop-EOP, drop-ACQUIRE, PARTIAL_FLUSH) fixes it.
        //   The fix is shader-side (terminal store / dyn-VGPR wave retirement). See KG.
        // Codex fix (2026-07-05): EOP RELEASE_MEM only on the FINAL chunk. The terminal-store-drain quirk stalls
        //   the EOP fence with no post-kernel traffic; a per-chunk stalled EOP sits in the in-order queue and
        //   BLOCKS the next chunk's DispatchDirect from ever launching (that was the chunk1 "occ20 stuck at base,
        //   0 claims" hang). Non-last chunks serialize via the kernel-done gate (occ0==0 + settle) in the poll
        //   loop below; only the last chunk arms the real fence (guarded teardown handles it if it too stalls).
        const bool lastChunk = (chunkHi >= claimTotal);
        if (lastChunk) RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));
        double t0 = now_s(); RingSubmit(ring);
        bool done = false, admitted = false; double lastSnap = t0, lastYield = t0;
        // Complete on the KERNEL'S OWN done-signal (occ0==0 + settle), NOT the EOP fence. The terminal C store's
        //   s_endpgm implicit drain stalls the EOP fence on this raw-PM4 path (COOP_STATUS.md; the proven coop
        //   path completes the same way via ML8_COOP_NOFENCE). The store IS issued -- the settle lets it land
        //   before the oracle reads C (a stale read fails the oracle, never a false CLEAN). The guarded teardown
        //   below still refuses to destroy a non-idle queue (brick-avoidance), so a lingering wave is reclaimed
        //   by process-exit, never a forced destroy. A real hang still trips the timeoutS bail -> forensics.
        uint32_t lastEnd = 0; double lastEndChange = t0;
        double settle = getenv("DSWS2_SETTLE") ? atof(getenv("DSWS2_SETTLE")) : 0.30;
        while (true) { double now = now_s();
            // COMPOSITOR YIELD (proven run_mbgemm mechanism, was MISSING on the flow path): hand the gfx ring
            //   (Hyprland) an unconditional render+VGPR window every yieldEvery ms DURING the wait -- so a long or
            //   stuck dispatch can't starve the desktop's gfx ring into a ring-timeout MODE1 reset. Host sleep
            //   only; never enters the in-kernel TF span. This is the "let hyprland through every so often" logic.
            if (!yieldOff && yieldMs > 0 && (now - lastYield) >= yieldEvery) {
                struct timespec yts = { yieldMs/1000, (long)(yieldMs%1000)*1000000L };
                nanosleep(&yts, nullptr); lastYield = now_s();
            }
            if (streamOn && (now - lastSnap) >= 0.2) { lastSnap = now;
                // FREEZE-FRAME every 200ms -> real disk: the ONLY forensics that survives a MES-wedge brick
                //   (timeout/final readouts never fire on a brick). STAGINSTR counters + flow_snapshot frontier.
                fprintf(stderr, "[dsws2 +%5.2fs] occ0=%u claim=%u fence=%s | comp=%u coast=%u gf=%u | FRONTIER ASSIGN=%u STAGE=%u DRAIN=%u slot[RB=%u BF=%u AR=%u] barrier=%u | SPIN[sh=%u st=%u ta=%u] tick[e=%u x=%u] bank[w=%u a=%u]\n",
                        now-t0, occW[0], occW[5], (*fenceW==FENCE_VALUE)?"FIRED":"--",
                        occW[71], occW[70], occW[73],
                        occW[74], occW[75], occW[76], occW[77], occW[78], occW[79], occW[80],
                        occW[81], occW[82], occW[83], occW[2], occW[3], occW[84], occW[85]); fflush(stderr); }
            if (occW[0] > 0) admitted = true;
            uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
            bool ff = (*fenceW == FENCE_VALUE);
            // done = fence fired, OR kernel-done: occ0==0 (all counted waves retired) + a wave stamped its exit
            //   tick (occ[3]!=0) + settled (store landed). Matches the coop gate.
            if (admitted && occW[0]==0 && (ff || (end != 0 && (now - lastEndChange) > settle))) { done = true; break; }
            if (now - t0 > timeoutS) break;
        }
        if (!done) {
            fprintf(stderr, "\n*** DSWS2 TIMEOUT (chunk base=%llu hi=%llu): occ0(live)=%u occ20(claim)=%u fence=%s ***\n",
                    (unsigned long long)base, (unsigned long long)chunkHi, occW[0], occW[5], (*fenceW==FENCE_VALUE)?"FIRED":"--");
            // EMERGENT-economy timeout forensics: WHERE did it stall? (all slots stream live during the run)
            //   computed>0 & rising -> compute progressed (slow/contention); computed~0 & coast huge -> LIVELOCK.
            fprintf(stderr, "    [timeout forensics] residentPeak occ[1]=%u  fatPeak occ[58]=%u  fatResidual occ[57]=%u  alllive-net occ[60]=%u (TRACE: >0 w/ occ0=0 == waves stuck PRE-LIVE at .Lflow_alloc)\n",
                    occW[1], occW[58], occW[57], occW[60]);
            fprintf(stderr, "    [timeout forensics] STAGINSTR  coast occ[70]=%u  computed occ[71]=%u  feed-stages occ[72]=%u  grow-fail occ[73]=%u\n",
                    occW[70], occW[71], occW[72], occW[73]);
            fprintf(stderr, "    [timeout forensics] FRONTIER  ASSIGN=%u STAGE=%u DRAIN=%u  drain-slot[RBDONE=%u BFDONE=%u ARDONE=%u]  barrier=%u\n"
                            "        (read: where DRAIN<STAGE<ASSIGN pinpoints the stalled stage; RBDONE<ACC_N=never computed, BFDONE<FN|ARDONE<G=never staged, DRAIN<ASSIGN w/ all-full=completer/drain bug, barrier<WAVES=exit-barrier never closed)\n",
                    occW[74], occW[75], occW[76], occW[77], occW[78], occW[79], occW[80]);
            allok = false; break;
        }
        if (chunkDiag) {
            double cwall = now_s() - t0;
            fprintf(stderr, "  [chunk diag] base=%llu hi=%llu wall=%.3fs claim=%u  STAGINSTR d: coast=%u computed=%u feed=%u grow-fail=%u  fatPk=%u%s\n",
                    (unsigned long long)base, (unsigned long long)chunkHi, cwall, occW[5],
                    occW[70]-diagPrevCoast, occW[71]-diagPrevComp, occW[72]-diagPrevFeed, occW[73]-diagPrevGF,
                    occW[58],   // FATMAX: running peak concurrent fat waves (absolute, not a delta -- max isn't additive)
                    (cwall > 0.5) ? "   <-- SLOW" : "");
            fflush(stderr);
        }
        lastOcc0 = occW[0]; lastOcc20 = occW[5]; totalConv += occW[48];   // accumulate this chunk's role-switch commits (DIAG conv counter)
        // TFPROBE: read this chunk's device-busy span BEFORE the next iteration re-zeros occ[2]/occ[3]. A stamped
        //   chunk has occ[2] != 0xFFFFFFFF (entry min written) AND occ[3] != 0 (exit max written). Sum spans across
        //   chunks -> total GPU busy ticks for the whole GEMM (host inter-chunk gaps excluded). If unstamped, the
        //   bin lacks TFPROBE tick capture -> flag and skip (no bogus TF from the 0xFFFFFFFF sentinel).
        { uint32_t gs = occW[2], ge = occW[3];
          if (gs != 0xFFFFFFFFu && ge != 0) {
              sumSpan += (ge >= gs) ? (uint64_t)(ge - gs) : ((uint64_t)ge + 0x100000000ull - (uint64_t)gs);
              spanChunks++;
          } else tfMissing = true; }
        if (!yieldOff && yieldMs > 0) { struct timespec ts = { yieldMs/1000, (long)(yieldMs%1000)*1000000L }; nanosleep(&ts, nullptr); }
        if ((now_s() - t0) > chunkMaxS) {
            fprintf(stderr, "  [dsws2] WARN chunk @base%llu wall %.2fs > %.2fs cap -> ABORT remaining chunks\n",
                    (unsigned long long)base, now_s()-t0, chunkMaxS);
            allok = false; break;
        }
    }
      if (!allok) break;                               // rep loop: bail on any chunk failure/timeout
      { uint64_t rs = sumSpan - repSpanBase;            // this rep's busy-tick span (across its chunks)
        if (rs > 0) { if (rs < repSpanMin) repSpanMin = rs; if (rs > repSpanMax) repSpanMax = rs; repsDone++; } }
    }   // ---- end SUSTAINED rep loop ----
    if (!allok) {
        fprintf(stderr, "    [teardown] dsws2 run did not complete cleanly -> NOT destroying queue (brick-avoidance; process-exit reclaims).\n");
        return res;
    }
    { double tw = now_s(); struct timespec ts = {0, 2000000L};
      while (*fenceW != FENCE_VALUE && (now_s() - tw) < 5.0) nanosleep(&ts, nullptr); }
    bool queueIdle = (*fenceW == FENCE_VALUE);
    if (!queueIdle) fprintf(stderr, "  [teardown] WARN: EOP fence never fired in 5s; queue NON-IDLE -> NOT destroying (process-exit reclaims).\n");
    res.occ0 = lastOcc0; res.occClaim = lastOcc20;
    printf("  [dsws2 alllive-net] occ[60]=%u  peak-resident occ[1]=%u  (TRACE build: occ[60]>0 w/ occ0=0 == waves stuck PRE-LIVE at .Lflow_alloc)\n", occW[60], occW[1]);
    printf("  [dsws2 completion] occ[0](live)=%u (0=clean)  occ[20](claim)=%u  (NOTE: with pool>=1, each WG's pinned\n"
           "    claimer makes exactly one extra terminal over-claim past the bound, so the expected clean value is\n"
           "    chunkHi(last chunk)+#WGs-that-raced-the-last-claim, NOT exactly TOTAL_super=%llu -- treat occ[20] as a\n"
           "    'did every WG's claimer reach a terminal claim' liveness signal, occ[0]==0 as the real completion gate)\n",
           lastOcc0, lastOcc20, (unsigned long long)TOTAL_super);
    printf("  [dsws2 CONVERSIONS] committed role-switches (occ[48], summed over chunks) = %u  (>0 => waves ADAPTIVELY switched role)\n", totalConv);
    {   // STAGINSTR: FULL COAST DECOMPOSITION (2026-07-14). `coast` was ONE bucket with FOUR doors into it,
        //   and we spent a day tuning doors 3+4 (which are 0.008% of it) because we could not see doors 1+2.
        //   coast == CNOSTG + CLEAD + FATFULL + GROWFAIL  (the sum is a self-check: if it does not close, a
        //   door is miscounted). And occ[88] JWAIT is the one that was NEVER counted: a FAT carrier holding
        //   ACC in registers, spinning for its next segment to be staged. That is the only place a fat wave
        //   burns time, and it was invisible in every prior measurement.
        uint32_t coastIt = occW[70], compIt = occW[71], feedIt = occW[72], growFail = occW[73];
        uint32_t feedMT  = occW[86], fatFull = occW[87];
        uint32_t jWait   = occW[88], cLead = occW[89], cNoStg = occW[90];
        if (coastIt + compIt > 0) {
            double tot = (double)(coastIt + compIt);
            printf("  [dsws2 STAGINSTR] computed=%u  coast=%u  (coast-frac=%.1f%%)  feed-stages=%u\n",
                   compIt, coastIt, 100.0 * (double)coastIt / tot, feedIt);
            uint32_t doors = cNoStg + cLead + fatFull + growFail;
            printf("  [dsws2 COAST DECOMP]  (door sum=%u vs coast=%u : %s)\n"
                   "      door1 NOTHING-STAGED (DRAIN>=STAGE) = %-12u %5.1f%% of coast\n"
                   "      door2 LEAD-GATE      (ksi%%J != 0)   = %-12u %5.1f%% of coast   <- STRUCTURAL: (J-1)/J by construction\n"
                   "      door3 FAT-PEAK-FULL  (stagger cap)  = %-12u %5.1f%% of coast\n"
                   "      door4 GROW-FAIL      (VGPR budget)  = %-12u %5.1f%% of coast\n",
                   doors, coastIt,
                   (doors == coastIt ? "CLOSES" : "*** DOES NOT CLOSE -- a door is miscounted ***"),
                   cNoStg,   coastIt ? 100.0*(double)cNoStg/(double)coastIt : 0.0,
                   cLead,    coastIt ? 100.0*(double)cLead/(double)coastIt  : 0.0,
                   fatFull,  coastIt ? 100.0*(double)fatFull/(double)coastIt: 0.0,
                   growFail, coastIt ? 100.0*(double)growFail/(double)coastIt:0.0);
            // THE CARRIER STALL. jWait is NOT a coast -- the wave is FAT and cannot do anything else.
            //   jWait >> comp  => the carriers are starved: staging cannot keep a fat wave fed. STAGE-BOUND, and
            //                     no amount of admission control / grow-fail elimination can touch it.
            //   jWait << comp  => carriers run to completion; the cost is elsewhere.
            // *** DUTYPROBE: peak/cycle -- the number the whole traveling-peak design rests on. ***
            //   A wave's fat window vs its full cycle. If duty is LOW the peaks can be phase-offset and the
            //   resident budget becomes the AVERAGE footprint, not the max (kmbandy's governing rule) -- that is
            //   where the headroom for a much larger G lives. If duty is ~100% the wave is a SQUARE WAVE and no
            //   amount of staggering can help (that is what a big JDEPTH does: it re-creates full-K).
            uint32_t dFat = occW[93], dCyc = occW[94];
            if (dCyc > 0) {
                double duty  = (double)dFat / (double)dCyc;   // peak / cycle -- EVERY burst measured, no sampling
                double lanes = duty > 0 ? 1.0 / duty : 0.0;                       // waves that can SHARE one peak slot
                uint32_t accN = getenv("DSWS2_G") ? (uint32_t)atoi(getenv("DSWS2_G")) : 15u;
                printf("  [dsws2 DUTY] peak/cycle = %.1f%%   (fat=%u cyc=%u shader-cycles>>12, every burst)\n"
                       "      -> %.1f waves can SHARE one peak slot (1/duty)\n"
                       "      -> %u carriers x %.3f duty = %.1f peaks needed CONCURRENTLY (vs %u if unstaggered)\n"
                       "      -> %s\n",
                       100.0*duty, dFat, dCyc, lanes,
                       accN, duty, accN*duty, accN,
                       duty > 0.75 ? "SQUARE WAVE -- staggering CANNOT help here (peak ~= average)"
                     : duty > 0.35 ? "moderate duty -- a stagger buys some, not a lot"
                                   : "*** LOW DUTY -- TRAPEZOID. The traveling peak has real headroom here. ***");
            }
            uint32_t dmFat = occW[91], tokLeak = occW[92];
            if (tokLeak > 0u)
                fprintf(stderr, "  [dsws2 STAGGER] %u wave(s) retired HOLDING a fat token (occ[92]) -- leak caught+returned\n"
                                "      (unfixed, each leak permanently burns one of MAXFAT slots -> FATTOK saturates -> WG wedges)\n", tokLeak);
            if (dmFat > 0u) {
                // .Lflow_retire assumes "ACC dead, wave lean". A carrier force-retired out of .Lflow_jwait is FAT with an
                //   UNFLUSHED ACC -> its partial sum is dropped AND the slot's RBDONE never advances. The C matrix is wrong.
                //   This ate 34% of the computed segments at J=64 on 2026-07-14 and the 1-tile oracle sample never saw it.
                fprintf(stderr,
                    "\n  *** [dsws2 INVALID RUN] DEADMAN FORCE-RETIRED %u FAT CARRIER(S) (occ[91]) ***\n"
                    "      A fat carrier holds its split-K partial sum IN REGISTERS. .Lflow_retire does NOT flush it.\n"
                    "      => C IS WRONG and `computed` UNDERCOUNTS. DO NOT USE THIS RUN'S TF OR COUNTERS.\n"
                    "      Raise DEADMAN_TICKS (currently ~10s) or shorten the chunk (ML8_COOP_CHUNK_MAXS).\n\n", dmFat);
            }
            printf("  [dsws2 CARRIER STALL] occ[88] .Lflow_jwait spins (FAT, ACC live, waiting for a STAGE) = %u\n"
                   "                    -> %.2f spin-iters per computed rowblk-segment  => %s\n",
                   jWait,
                   compIt ? (double)jWait / (double)compIt : 0.0,
                   (compIt && (double)jWait > (double)compIt)
                       ? "*** CARRIERS ARE STAGE-STARVED (fat waves spend more time waiting than computing) ***"
                       : "carriers are fed (stall is not the wall)");
            // THE BATON (2026-07-16): a carrier that refused a fat-token but HAD staged work waited on the
            //   per-SIMD VGPR-budget pool (.Lflow_batonwait) instead of coasting, then grew into the registers
            //   a shrinking carrier freed at shrink-START. occ[98] > 0 is the proof the traveling peak ENGAGED
            //   (distinct from occ[87] FATFULL, which counts refusals that COASTED). Compare STAGGER=1 vs =0.
            uint32_t batonWait = occW[98];
            printf("  [dsws2 BATON] occ[98] .Lflow_batonwait spins (carrier waited on the VGPR-budget pool, then grew) = %u\n"
                   "                    -> %s\n",
                   batonWait,
                   batonWait > 0u ? "the traveling peak ENGAGED -- a shrinking carrier handed its budget to a waiter"
                                  : "no baton handoff (no carrier-with-work ever hit a full pool -- or STAGGER=0)");
            // feedMT is emitted by BOTH lean feed waves AND coasting compute waves, so it is NOT a subset of
            //   coast -- dividing by coast printed 196.2% on 2026-07-13. Correct denominator = all feed-path iters.
            double feedTot = (double)feedIt + (double)feedMT;
            printf("  [dsws2 STARVATION] feed-path iters with NOTHING ASSIGNED (occ[86]) = %u\n"
                   "                    -> %.1f%% of ALL feed-path iters found an empty ASSIGN frontier  => %s\n",
                   feedMT,
                   feedTot > 0 ? 100.0 * (double)feedMT / feedTot : 0.0,
                   (feedTot > 0 && (double)feedMT > 0.5 * feedTot)
                       ? "ASSIGN-BOUND (coordinator cannot publish fast enough)"
                       : "STAGE-BOUND (work is assigned; feeds/pool cannot stage it fast enough)");
            // DECENTASN CLAIM-PERSISTENCE DIAGNOSTIC (sol gpt-5.6-sol, 2026-07-15): both reviews converged that
            //   the seed is a PHANTOM claim (claim CAS reports success but does not persist to LDS). Measure it
            //   directly at the claim, upstream of every propagation story.
            if (occW[95] > 0u || occW[96] > 0u || occW[97] > 0u) {
                printf("  [dsws2 DECENTASN CLAIM-DIAG]\n"
                       "      occ[95] exec lane0 INACTIVE at claim CAS (lds_cas_rtn false-'won' precondition) = %u\n"
                       "      occ[96] won-claim did NOT persist (immediate re-read pending|inflight==0) = PHANTOM = %u\n"
                       "      occ[97] release bailed on inflight==0 (containment, no underflow)          = %u\n"
                       "      *** occ[96]>0 confirms the phantom-claim seed; occ[95]>0 too => it is the exec-mask path (931/939) ***\n",
                       occW[95], occW[96], occW[97]);
            }
        }
    }
    if (traceOn || occW[58] > 0u) {   // fat gauge populates occ[58]/[57] under STAGINSTR (or TRACE); print whenever data exists
        uint32_t fatPeak = occW[58], fatResidual = occW[57];   // FATMAX / FATLIVE (should end ~0 if balanced)
        const uint32_t nfv = (uint32_t)(((32 + 8*FMc*FNc + 2*FMc) + 2*FNc + 15) & ~15);  // matches kernel NFV (=112 @ FM2FN4, 80 @ FM1FN4)
        printf("  [dsws2 VGPR-BUDGET PROBE] peak concurrent FAT compute waves (occ[58]) = %u  -> ~%u VGPR in flight (x NFV=%u)"
               "   [residual live=%d]\n", fatPeak, fatPeak*nfv, nfv, (int)fatResidual);
        printf("      (per-SIMD B estimate = peak/128 SIMDs x %u; raise DSWS2_BUDGET/pool until this plateaus or s_alloc stalls)\n", nfv);
        uint32_t peakWaves = occW[1];   // occ[1] = peak concurrent RESIDENT waves (all roles), vs 2048 HW ceiling (16/SIMD)
        printf("  [dsws2 OCCUPANCY] peak concurrent resident waves (occ[1]) = %u of 2048 HW max (%.1f%%, %.2f/SIMD)  "
               "launched = %u WGs x %u waves = %u\n", peakWaves, peakWaves/2048.0*100.0, peakWaves/128.0,
               pool, WAVES_LAUNCH, pool*WAVES_LAUNCH);
    }

    // ---- TFPROBE THROUGHPUT: total useful work / total device-busy span. Work = 2*M*N*K (split-K independent;
    //   the n_kseg segments reduce the SAME K, so total MACs = M*N*K regardless of how K is partitioned). Span
    //   = summed per-chunk (occ[3]-occ[2]) GPU ticks -> the on-chip busy time, immune to host launch/fence/poll
    //   overhead (the reason a host wall-clock is useless at these <1ms shapes). TF = 2*M*N*K*freq / span / 1e12. ----
    if (spanChunks > 0 && sumSpan > 0) {
        res.wall = sumSpan;
        double reps_eff = (repsDone > 0) ? (double)repsDone : 1.0;   // work = reps_eff * (2MNK); span = sum over reps
        double workAll = 2.0 * (double)Mo * (double)No * (double)Ko * reps_eff;
        res.tf = workAll * freq_hz / (double)sumSpan / 1e12;
        double perRepWork = 2.0 * (double)Mo * (double)No * (double)Ko;
        double tfHi = (repSpanMax > 0) ? perRepWork * freq_hz / (double)repSpanMin / 1e12 : res.tf;  // min span -> peak TF
        double tfLo = (repSpanMax > 0) ? perRepWork * freq_hz / (double)repSpanMax / 1e12 : res.tf;  // max span -> trough TF
        printf("  [dsws2 THROUGHPUT] %dx%dx%d  TF=%.1f  (%.1f%% of 307 TF fp8 peak)  span=%llu ticks / %u chunk(s) @ %.0f MHz\n",
               Mo, No, Ko, res.tf, res.tf / 307.0 * 100.0, (unsigned long long)sumSpan, spanChunks, freq_hz / 1e6);
        if (repsDone > 1)
            printf("  [dsws2 SUSTAINED] reps=%u  TF=%.1f mean  (per-rep %.1f-%.1f, spread %.1f%%)  -- glass-flat=trustworthy\n",
                   repsDone, res.tf, tfLo, tfHi, (tfHi > 0 ? (tfHi - tfLo) / tfHi * 100.0 : 0.0));
    } else {
        printf("  [dsws2 THROUGHPUT] n/a -- bin has no TFPROBE tick capture (occ[2]/occ[3] unstamped%s). "
               "Rebuild the bin with -Wa,-defsym,TFPROBE=1 to measure TF.\n", tfMissing ? "" : "; no chunks completed");
    }

    // ---- PHASEPROBE: in-kernel per-phase tick breakdown of the COMPUTE wave (the critical path).
    //   Accumulators at occ[64..69] (bytes 256..276) live ABOVE the 0x100 per-chunk memset -> they SUM
    //   over the whole run. u32 slots -> keep PHASEPROBE runs short (single/few passes) to avoid wrap; the
    //   DISTRIBUTION (%) is stable regardless. Ticks are summed across ALL compute waves (aggregate time
    //   in each phase), so % shows WHERE compute-wave time goes -- measured, not inferred. ----
    {
        const char* phName[6] = {"FOLLOW_WAIT","STAGE_WAIT","GROW","WMMA","FLUSH","SHRINK"};
        uint64_t ph[6] = {0,0,0,0,0,0}, phSum = 0;
        for (int i = 0; i < 6; i++) { ph[i] = (uint64_t)occW[64 + i]; phSum += ph[i]; }
        if (phSum > 0) {
            printf("  [dsws2 PHASE breakdown] compute-wave ticks by phase (summed over all waves+chunks):\n");
            printf("      %-12s %14s   %6s   %s\n", "phase", "ticks", "share", "what");
            const char* phWhat[6] = {"idle: waiting for claimer to publish next super-tile",
                                     "idle: waiting for A/B feeds to stage operands",
                                     "dyn-VGPR grow 32->112 (+ rowblk claim)",
                                     "the actual fp8 WMMA compute",
                                     "split-K C reduction (global_atomic_add_f32)",
                                     "dyn-VGPR shrink 112->32"};
            for (int i = 0; i < 6; i++) {
                double pct = 100.0 * (double)ph[i] / (double)phSum;
                char bar[41]; int nb = (int)(pct / 2.5 + 0.5); if (nb > 40) nb = 40;
                for (int k = 0; k < nb; k++) bar[k] = '#'; bar[nb] = 0;
                printf("      %-12s %14llu   %5.1f%%  %-40s %s\n", phName[i],
                       (unsigned long long)ph[i], pct, bar, phWhat[i]);
            }
            printf("      %-12s %14llu\n", "TOTAL", (unsigned long long)phSum);
        }
    }

    // ---- TRACE dump: read the per-super-tile rows back to CSV (real disk). Rows are indexed by SEGCNT
    //   (1-based), so row 0 stays zero; skip all-zero (unwritten) rows. Single-chunk runs only (chunked
    //   runs reset SEGCNT per chunk -> rows overwrite). ----
    if (traceW) {
        const char* csv = getenv("DSWS2_TRACE_CSV");
        char path[600];
        if (!csv) { snprintf(path, sizeof path, "/home/kmbandy/dsws_gpu_logs/trace_%dx%dx%d.csv", Mo, No, Ko); csv = path; }
        FILE* tf = fopen(csv, "w");
        if (tf) {
            fprintf(tf, "row,tick_lo,segcnt,epoch,nComp,nAfeed,nBfeed,occA,occB,convCount,vresv,sti,quiesce,tick_hi,chunkHi,wg_id\n");
            uint32_t nrows = 0;
            for (uint32_t r = 0; r < traceMaxRows; ++r) {
                const volatile uint32_t* row = traceW + (size_t)r*16;
                bool nz = false; for (int i = 0; i < 16; ++i) if (row[i]) { nz = true; break; }
                if (!nz) continue;
                fprintf(tf, "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                        r, row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14]);
                ++nrows;
            }
            fclose(tf);
            printf("  [dsws2 trace] wrote %u rows -> %s\n", nrows, csv);
        } else fprintf(stderr, "  [dsws2 trace] cannot open '%s' for write\n", csv);
    }

    // ---- CANARY scan (verbatim pattern from run_mbcoop): did any C store land past the data region? ----
    { const uint32_t* tail = (const uint32_t*)((const char*)C.ptr + cbytes);
      uint64_t words = padB/4, firstNZ = ~0ull, lastNZ = 0, nzCount = 0;
      for (uint64_t w=0; w<words; ++w) if (tail[w]) { if (firstNZ==~0ull) firstNZ=w; lastNZ=w; ++nzCount; }
      if (nzCount) fprintf(stderr, "  [canary] *** C OOB STORE detected: %llu words written into guard tail; first at +%llu B past C-end ***\n",
                            (unsigned long long)nzCount, (unsigned long long)(firstNZ*4));
      else fprintf(stderr, "  [canary] C guard tail clean (no OOB store past C-end).\n"); (void)lastNZ; }

    // ---- ORACLE (A2 tiered compare): for EACH output tile ti=mblk*NTL+tcol and EACH claimed rowblk r in
    //   [0,G), the reference is the full-K chained 16x16x16 wmma_ref over all KT k-steps. Decodes the C
    //   buffer EXACTLY like the coop oracle (unpack_D), with cid->r and P->G (the v2 G-extent). ----
    const float* Cf = (const float*)C.ptr;
    // SAMPLED ORACLE: full-K CPU reference is ~O(TOTAL*G*KT) MACs -> minutes at training M (640 tiles, KT=128).
    //   DSWS2_ORACLE_STRIDE>1 checks every Nth output tile (still every rowblk/frag within it) so big-shape
    //   perf runs verify a representative subset cheaply. Default 1 = full check (unchanged for small shapes).
    int ostride = getenv("DSWS2_ORACLE_STRIDE") ? atoi(getenv("DSWS2_ORACLE_STRIDE")) : 1;
    if (ostride < 1) ostride = 1;
    int nTilesChecked = 0;
    for (int ti = 0; ti < (int)TOTAL; ti += ostride) {
        nTilesChecked++;
        int mblk = ti / NTL, tcol = ti % NTL;
        for (int r = 0; r < Gv; ++r) {
            for (int mi = 0; mi < FMc; ++mi) for (int ni = 0; ni < FNc; ++ni) {
                int rowbase = mblk*TMsuper + r*(FMc*16) + mi*16;
                int colbase = tcol*TN + ni*16;
                float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
                uint8_t Ablk[256], Bblk[256]; float Dout[256];
                for (int kt = 0; kt < KT; ++kt) {
                    for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                        Ablk[i*16+j] = Ah[(size_t)(rowbase+i)*Ko + (kt*16+j)];
                        Bblk[i*16+j] = Bh[(size_t)(kt*16+i)*No + (colbase+j)];
                    }
                    wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
                    for (int i=0;i<256;i++) Cacc[i]=Dout[i];
                }
                int frag = mi*FNc + ni;
                size_t foff = (size_t)ti*(size_t)((uint32_t)Gv*FMc*FNc*256) + (size_t)r*(size_t)(FMc*FNc*256) + (size_t)frag*256;
                float D[256]; unpack_D(Cf + foff, D);
                OracleCmp cmp = oracle_compare(D, Cacc, 256, orel, oabs);
                if (cmp.ok) res.okFrags++; else res.badFrags++;
                if (cmp.max_rel > res.maxRel) res.maxRel = cmp.max_rel;
            }
        }
    }
    printf("  [dsws2 oracle] ok=%llu bad=%llu max_rel=%.4g  tier=%s (rel=%.0e abs=%.0e)  [%d/%u tiles checked, stride=%d]\n",
           (unsigned long long)res.okFrags, (unsigned long long)res.badFrags, res.maxRel,
           n_kseg==1?"TIGHT":"LOOSE", orel, oabs, nTilesChecked, TOTAL, ostride);

    if (queueIdle) {
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        if (traceOn && traceBuf.ptr) FreeGpu(traceBuf);
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    }
    res.ok = allok && (res.badFrags == 0);
    return res;
}

// ===========================================================================================
// run_grind: the NON-split-K CONTROL. Launches occ_kernel_grind.bin (1 wave = 1 WG owns one
//   FM*16 x FN*16 output tile, full-K in registers, writes C ONCE with plain global_store, NO
//   split-K, NO C-atomic reduction). Directly comparable to run_dsws2's TF: does avoiding split-K's
//   32x C-write amplification beat the split-K kernel's ~2.1 TF? Static VGPR (the anti-moat control).
//   Self-instrumented tick span (occ[2]/occ[3]) + maxlive (occ[1]) live in the kernel already.
// ===========================================================================================
struct GrindResult { bool ok=false; uint64_t okFrags=0, badFrags=0; double maxRel=0.0, tf=0.0; uint64_t wall=0; uint32_t maxlive=0, occ0=0; };

static GrindResult run_grind(uint32_t node, const char* isaPath, int FMc, int FNc,
                             int Mo, int No, int Ko, float orel, float oabs, double freq_hz) {
    GrindResult res;
    const int TM = FMc*16, TN = FNc*16;                 // grind output tile = FM*16 x FN*16 (32x64 @ 2x4)
    if (TM==0 || TN==0 || Ko<=0 || (Mo%TM) || (No%TN) || (Ko%16)) {
        fprintf(stderr, "  [grind] geometry %dx%dx%d not tile-aligned (TM=%d TN=%d, K%%16)\n", Mo,No,Ko,TM,TN); return res; }
    const int MTL = Mo/TM, NTL = No/TN, KT = Ko/16, NT = No/16;
    const uint32_t TOTAL = (uint32_t)MTL*(uint32_t)NTL;   // grind tile count (finer: no G, no ksi)
    const uint32_t magic = (uint32_t)((0x100000000ULL + (uint64_t)NTL - 1)/(uint64_t)NTL);
    const int KCHUNK = getenv("GRIND_KCHUNK") ? atoi(getenv("GRIND_KCHUNK")) : 4;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)Mo*Ko), Bh((size_t)Ko*No), Bshufh((size_t)Ko*No);
    for (size_t i=0;i<Ah.size();++i) Ah[i]=NICE[(i*7 + i/(size_t)Ko)%6];
    for (size_t i=0;i<Bh.size();++i) Bh[i]=NICE[(i*5 + (i/(size_t)No)*3)%6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), Ko, No);

    size_t isaLen=0; uint8_t* isaBytes=ReadFile(isaPath,&isaLen);
    if (!isaBytes) { fprintf(stderr,"  [grind] cannot read '%s'\n",isaPath); return res; }
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node,0x1000,false,true);
    uint64_t padB = (uint64_t)(getenv("ML8_COOP_PAD_MB")?atoi(getenv("ML8_COOP_PAD_MB")):64)*1024ull*1024ull;
    GpuBuf Ad = AllocGpu(node,((Ah.size()+0xFFF)&~0xFFFull)+padB,false,true,true);
    GpuBuf Bd = AllocGpu(node,((Bshufh.size()+0xFFF)&~0xFFFull)+padB,false,true,true);
    uint64_t cbytes = ((uint64_t)TOTAL*(uint64_t)(FMc*FNc*1024) + 0xFFF) & ~0xFFFull;   // TOTAL tiles x FM*FN frags x 256 f32
    GpuBuf C = AllocGpu(node,cbytes+padB,false,true,true);
    GpuBuf fence = AllocGpu(node,0x1000,false,true);
    if (!(Ad.vram && Bd.vram && C.vram)) { fprintf(stderr,"\n*** GRIND VRAM GUARD FAILED -> abort ***\n"); abort(); }
    // address bounds gate: last A/B/C element the kernel can touch must be in-buffer.
    { uint64_t Amax = (uint64_t)(MTL*FMc*16-1)*Ko + (uint64_t)(KT-1)*16 + 8 + 7;   // max row=(MTL*FM*16-1), max kcol=(KT-1)*16+8+7
      uint64_t Bmax = (uint64_t)(NTL-1)*FNc*256 + (uint64_t)(KT-1)*(uint64_t)NT*256 + (uint64_t)(FNc-1)*256 + (uint64_t)31*8 + 7;
      uint64_t Cmax = (uint64_t)(TOTAL-1)*(uint64_t)(FMc*FNc*1024) + (uint64_t)(FMc*FNc-1)*1024 + (uint64_t)31*32 + 7*4 + 3;
      bool aok=Amax<Ah.size(), bok=Bmax<Bshufh.size(), cok=Cmax<cbytes;
      printf("  [grind bounds] A %llu/%zu %s  B %llu/%zu %s  C %llu/%llu %s\n",
             (unsigned long long)Amax,Ah.size(),aok?"OK":"*OOB*",(unsigned long long)Bmax,Bshufh.size(),bok?"OK":"*OOB*",
             (unsigned long long)Cmax,(unsigned long long)cbytes,cok?"OK":"*OOB*");
      if (!(aok&&bok&&cok)) { fprintf(stderr,"\n*** GRIND BOUNDS GATE FAILED -> REFUSE ***\n");
        FreeGpu(fence);FreeGpu(C);FreeGpu(Bd);FreeGpu(Ad);FreeGpu(occ);FreeGpu(isa); return res; } }
    memcpy(isa.ptr,isaBytes,isaLen); free(isaBytes);
    memcpy(Ad.ptr,Ah.data(),Ah.size()); memcpy(Bd.ptr,Bshufh.data(),Bshufh.size());
    memset((char*)Ad.ptr+((Ah.size()+0xFFF)&~0xFFFull),0,padB);
    memset((char*)Bd.ptr+((Bshufh.size()+0xFFF)&~0xFFFull),0,padB);
    volatile uint32_t* occW=(volatile uint32_t*)occ.ptr; volatile uint32_t* fenceW=(volatile uint32_t*)fence.ptr;
    memset((void*)occW,0,occ.size); *fenceW=0;
    memset((char*)C.ptr,0,cbytes+padB);   // grind writes each cell once; zero anyway (canary tail + clean)

    Ring ring; ring.buf=AllocGpu(node,0x10000,true,true); ring.dw=(uint32_t*)ring.buf.ptr; ring.sizeDw=(uint32_t)(ring.buf.size/sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node,HSA_QUEUE_COMPUTE,100,HSA_QUEUE_PRIORITY_NORMAL,ring.buf.ptr,ring.buf.size,nullptr,&ring.res));
    uint64_t shiftedIsa=((uint64_t)isa.ptr)>>8;
    uint64_t occVa=(uint64_t)occ.ptr,aVa=(uint64_t)Ad.ptr,bVa=(uint64_t)Bd.ptr,cVa=(uint64_t)C.ptr,fenceVa=(uint64_t)fence.ptr;
    const uint32_t WAVES_LAUNCH=1u;                       // 1 wave/WG (the control)
    uint32_t dims[8]={0,0,0,WAVES_LAUNCH*32,1,1,0,0};
    uint32_t pgm[6]={(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    const uint32_t vgprField=15u;                         // static 120 VGPR (NFV~108); anti-moat control
    uint32_t rsrc1=BuildPgmRsrc1(false); rsrc1=(rsrc1 & ~0x3fu)|(vgprField & 0x3fu);
    uint32_t ldsBytesRaw=(uint32_t)(KCHUNK*(FMc+FNc)*256); // 6144 @ KCHUNK=4,FM=2,FN=4
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits=ldsRsrc2Bits(ldsBytesRaw,&ldsU,&ldsA,&ldsG);
    uint32_t rsrc2=(BuildPgmRsrc2(false) & ~0x3eu)|(15u<<RSRC2_USER_SGPR_SHIFT)|ldsBits;
    uint32_t rsrc[2]={rsrc1,rsrc2};
    uint32_t userdata[16]={
        (uint32_t)occVa,(uint32_t)(occVa>>32),(uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32),(uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)KT,(uint32_t)Ko,(uint32_t)(NT*256),TOTAL,
        magic,(uint32_t)NTL,(uint32_t)(FNc*256),0u };
    uint32_t dispInit=BuildDispatchInitiator();
    const uint32_t pool=getenv("ML8_POOL")?(uint32_t)atoi(getenv("ML8_POOL")):256u;   // NO 64-clamp: grind is 1 wave/WG, needs many WGs
    printf("  [grind] %dx%dx%d  tile=%dx%d (FM=%d FN=%d)  TOTAL=%u tiles  waves/WG=1  pool=%u WGs  LDS=%uB(alloc %uB) VGPR=%u static RSRC2=0x%x\n",
           Mo,No,Ko,TM,TN,FMc,FNc,TOTAL,pool,ldsBytesRaw,ldsA,vgprField*8,rsrc2);
    uint32_t reslim[1]={0},tmpring[1]={0},restart[4]={0,0,0,0};
    memset((void*)occW,0,0x100); occW[20]=0; occW[2]=0xFFFFFFFFu; *fenceW=0;   // claim base 0; min-tick sentinel
    RingPlace(ring,PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_START_X,dims,8));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,pgm,6));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,rsrc,2));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS,reslim,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,tmpring,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,restart,4));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,userdata,16));
    RingPlace(ring,PM4DispatchDirectPacket(pool*WAVES_LAUNCH*32,1,1,dispInit));
    RingPlace(ring,PM4ReleaseMemoryPacket(FAMILY_GFX12,true,fenceVa,FENCE_VALUE));
    double t0=now_s(); RingSubmit(ring);
    bool done=false, admitted=false; const double timeoutS=25.0;
    while (true) { double now=now_s();
        if (occW[0]>0) admitted=true;
        if (admitted && occW[0]==0 && *fenceW==FENCE_VALUE) { done=true; break; }
        if (now-t0>timeoutS) break; }
    if (!done) { fprintf(stderr,"\n*** GRIND TIMEOUT: occ0=%u fence=%s ***\n",occW[0],(*fenceW==FENCE_VALUE)?"FIRED":"--");
        fprintf(stderr,"    [teardown] grind did not complete -> NOT destroying queue (process-exit reclaims).\n"); return res; }
    res.occ0=occW[0]; res.maxlive=occW[1];
    { uint32_t gs=occW[2],ge=occW[3];
      if (gs!=0xFFFFFFFFu && ge!=0) { res.wall=(ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs);
        res.tf=2.0*(double)Mo*(double)No*(double)Ko*freq_hz/(double)res.wall/1e12; } }
    printf("  [grind completion] occ[0]=%u (0=clean)  maxlive=%u (of 2048, %.1f%%, %.2f/SIMD)\n",
           res.occ0,res.maxlive,res.maxlive/2048.0*100.0,res.maxlive/128.0);
    if (res.wall) printf("  [grind THROUGHPUT] %dx%dx%d  TF=%.1f  (%.1f%% of 307 TF fp8 peak)  span=%llu ticks @ %.0f MHz\n",
                         Mo,No,Ko,res.tf,res.tf/307.0*100.0,(unsigned long long)res.wall,freq_hz/1e6);
    else printf("  [grind THROUGHPUT] n/a (occ[2]/occ[3] unstamped)\n");
    // canary
    { const uint32_t* tail=(const uint32_t*)((const char*)C.ptr+cbytes); uint64_t words=padB/4,nz=0;
      for (uint64_t w=0;w<words;++w) if (tail[w]) ++nz;
      fprintf(stderr,"  [grind canary] C guard tail %s\n",nz?"*** OOB STORE ***":"clean"); }
    // ---- ORACLE: each tile ti=(mblk,tcol), each (mi,ni) frag = full-K chained wmma_ref ----
    const float* Cf=(const float*)C.ptr;
    for (int ti=0; ti<(int)TOTAL; ++ti) {
        int mblk=ti/NTL, tcol=ti%NTL;
        for (int mi=0; mi<FMc; ++mi) for (int ni=0; ni<FNc; ++ni) {
            int rowbase=mblk*TM + mi*16, colbase=tcol*TN + ni*16;
            float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
            uint8_t Ablk[256],Bblk[256]; float Dout[256];
            for (int kt=0; kt<KT; ++kt) {
                for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                    Ablk[i*16+j]=Ah[(size_t)(rowbase+i)*Ko + (kt*16+j)];
                    Bblk[i*16+j]=Bh[(size_t)(kt*16+i)*No + (colbase+j)]; }
                wmma_ref_16x16x16(Ablk,Bblk,Cacc,Dout); for (int i=0;i<256;i++) Cacc[i]=Dout[i]; }
            int frag=mi*FNc+ni;
            size_t foff=(size_t)ti*(size_t)(FMc*FNc*256) + (size_t)frag*256;
            float D[256]; unpack_D(Cf+foff,D);
            OracleCmp cmp=oracle_compare(D,Cacc,256,orel,oabs);
            if (cmp.ok) res.okFrags++; else res.badFrags++;
            if (cmp.max_rel>res.maxRel) res.maxRel=cmp.max_rel;
        }
    }
    printf("  [grind oracle] ok=%llu bad=%llu max_rel=%.4g (rel=%.0e abs=%.0e)\n",
           (unsigned long long)res.okFrags,(unsigned long long)res.badFrags,res.maxRel,orel,oabs);
    if (*fenceW==FENCE_VALUE) { CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(C);FreeGpu(Bd);FreeGpu(Ad);FreeGpu(occ);FreeGpu(isa); }
    res.ok = (res.occ0==0) && (res.badFrags==0);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 Step A phase-timer: dispatch the PROFILE build of the real BLADDER FEEDONLY kernel and read the
// per-phase realtime tick-sums (occ[8..14]) + K-tile count (occ[15]) accumulated by the single profiler
// wave. Same real-GEMM setup as run_wggemm_perf (real A/Bshuf, claim loop, 512 K-tiles/tile). No oracle.
// ---------------------------------------------------------------------------
struct ProfResult { bool ok=false; uint32_t maxlive=0; uint64_t wall=0; double tf=0; uint32_t sum[7]={0}; uint32_t ktiles=0; uint32_t okSamp=0, badSamp=0; };

static ProfResult run_feedprof(uint32_t node, const char* isaPath, int M, int N, int K, uint32_t nWG, double freq_hz) {
    ProfResult res;
    const int FMt=4; const uint32_t ldsBytes=8196u, vgprField=26u;
    const int TM=FMt*32, TN=FMt*32;
    int NTL=N/TN, MTL=M/TM, NT=N/16, NTILES=K/32;
    uint32_t TOTAL=(uint32_t)MTL*NTL;
    int log2NTL=0; while ((1<<log2NTL) < NTL) ++log2NTL;
    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Bshufh((size_t)K*N);
    for (size_t i=0;i<Ah.size();++i) Ah[i]=NICE[(i*7 + i/(size_t)K)%6];
    for (size_t i=0;i<Bh.size();++i) Bh[i]=NICE[(i*5 + (i/(size_t)N)*3)%6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);

    size_t isaLen=0; uint8_t* isaBytes=ReadFile(isaPath,&isaLen);
    GpuBuf isa=AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ=AllocGpu(node,0x1000,false,true);
    GpuBuf Ad=AllocGpu(node,(Ah.size()+0xFFF)&~0xFFFull,false,true);
    GpuBuf Bd=AllocGpu(node,(Bshufh.size()+0xFFF)&~0xFFFull,false,true);
    uint64_t cbytes=((uint64_t)TOTAL*4096+0xFFF)&~0xFFFull;
    GpuBuf C=AllocGpu(node,cbytes,false,true);
    GpuBuf fence=AllocGpu(node,0x1000,false,true);
    memcpy(isa.ptr,isaBytes,isaLen); free(isaBytes);
    memcpy(Ad.ptr,Ah.data(),Ah.size()); memcpy(Bd.ptr,Bshufh.data(),Bshufh.size());
    volatile uint32_t* occW=(volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW=(volatile uint32_t*)fence.ptr;
    for (int i=0;i<17;i++) occW[i]=0; occW[2]=0xFFFFFFFFu; *fenceW=0;

    Ring ring; ring.buf=AllocGpu(node,0x10000,true,true); ring.dw=(uint32_t*)ring.buf.ptr;
    ring.sizeDw=(uint32_t)(ring.buf.size/sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node,HSA_QUEUE_COMPUTE,100,HSA_QUEUE_PRIORITY_NORMAL,ring.buf.ptr,ring.buf.size,nullptr,&ring.res));
    uint64_t shiftedIsa=((uint64_t)isa.ptr)>>8;
    uint64_t occVa=(uint64_t)occ.ptr,aVa=(uint64_t)Ad.ptr,bVa=(uint64_t)Bd.ptr,cVa=(uint64_t)C.ptr,fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8]={0,0,0,128,1,1,0,0};
    uint32_t pgm[6]={(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1=BuildPgmRsrc1(false); rsrc1=(rsrc1&~0x3fu)|(vgprField&0x3fu);
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits=ldsRsrc2Bits(ldsBytes,&ldsU,&ldsA,&ldsG);
    uint32_t rsrc2=(BuildPgmRsrc2(false)&~0x3eu)|(15u<<RSRC2_USER_SGPR_SHIFT)|ldsBits;
    uint32_t rsrc[2]={rsrc1,rsrc2};
    uint32_t reslim[1]={0},tmpring[1]={0},restart[4]={0,0,0,0};
    uint32_t userdata[16]={
        (uint32_t)occVa,(uint32_t)(occVa>>32),(uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32),(uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K,(uint32_t)(NT*256),(uint32_t)(NTL-1),(uint32_t)log2NTL,
        (uint32_t)NTILES,TOTAL,0,0 };
    uint32_t dispInit=BuildDispatchInitiator();
    RingPlace(ring,PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_START_X,dims,8));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,pgm,6));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,rsrc,2));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS,reslim,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,tmpring,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,restart,4));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,userdata,16));
    RingPlace(ring,PM4DispatchDirectPacket(nWG*128u,1,1,dispInit));
    RingPlace(ring,PM4ReleaseMemoryPacket(FAMILY_GFX12,true,fenceVa,FENCE_VALUE));

    double t0=now_s(); RingSubmit(ring);
    bool done=false,admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true){ double now=now_s();
        if (occW[1]>0) admitted=true;
        uint32_t end=occW[3]; if (end!=lastEnd){lastEnd=end;lastEndChange=now;}
        bool ff=(*fenceW==FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025){done=true;break;}
        if (now-t0>40.0) break;
    }
    if (!done){ fprintf(stderr,"\n*** FEEDPROF TIMEOUT (%s): live=%u maxlive=%u ***\n",isaPath,occW[0],occW[1]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(C);FreeGpu(Bd);FreeGpu(Ad);FreeGpu(occ);FreeGpu(isa); return res; }
    res.ok=true; res.maxlive=occW[1];
    { uint32_t gs=occW[2],ge=occW[3]; res.wall=(ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.tf=(res.wall>0)?(2.0*(double)M*N*K*freq_hz/(double)res.wall/1e12):0.0;
    for (int i=0;i<7;i++) res.sum[i]=occW[8+i];
    res.ktiles=occW[15];
    // ---- rung 9 verification: did the PROFILE build actually COMPUTE the GEMM? (same sampled acc[0][0]
    //      oracle as run_wggemm_perf). If bad -> the 96 TF was a bogus wall from incomplete/garbage work. ----
    { const float* Cf=(const float*)C.ptr;
      uint32_t tstride = TOTAL>16 ? TOTAL/16u : 1u;
      for (uint32_t ti=0; ti<TOTAL; ti+=tstride) {
        int trow=ti>>log2NTL, tcol=ti&(NTL-1);
        for (int wid=0; wid<4; ++wid) {
          int wm=wid/2, wn=wid%2, rowbase=trow*TM+wm*(FMt*16), colbase=tcol*TN+wn*(FMt*16);
          float Cacc[256]; for (int i=0;i<256;i++) Cacc[i]=0.f;
          uint8_t Ablk[256], Bblk[256]; float Dout[256];
          for (int kt=0; kt<K/16; ++kt) {
            for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
              Ablk[i*16+j]=Ah[(size_t)(rowbase+i)*K + (kt*16+j)];
              Bblk[i*16+j]=Bh[(size_t)(kt*16+i)*N + (colbase+j)];
            }
            wmma_ref_16x16x16(Ablk, Bblk, Cacc, Dout);
            for (int i=0;i<256;i++) Cacc[i]=Dout[i];
          }
          float D[256]; unpack_D(Cf + (size_t)ti*1024 + (size_t)wid*256, D);
          bool good=true; for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
          if (good) res.okSamp++; else res.badSamp++;
        }
      }
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(C);FreeGpu(Bd);FreeGpu(Ad);FreeGpu(occ);FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 STACK LADDER rung 1: load-only truthful base. 1-wave (32-thread) WGs stream FRAGS b64 from real
// A and real B buffers per claimed slice, sum every loaded dword -> occ[7], CPU-verify == expected (no DCE /
// no early-exit can fake the BW). Reports the three figures: TF-equiv, GB/s, checksum proof (+ maxlive,
// claims). totalBytes = CLAIM_CEIL * FRAGS * 256 * 2 (A+B).
// ---------------------------------------------------------------------------
struct StkResult { bool ok=false; uint32_t maxlive=0, claims=0, consumed=0; uint64_t wall=0; double gbps=0, tfeq=0;
                   bool proof=false; uint32_t got=0, exp=0; };

static StkResult run_stack(uint32_t node, const char* isaPath, double freq_hz,
                           uint32_t FRAGS=8, uint32_t CLAIMCHUNK=256, uint32_t SLICE_STRIDE=8192,
                           uint64_t windowBytes=64ull*1024*1024, uint32_t CLAIM_CEIL=1048576, uint32_t nWG=1024) {
    StkResult res;
    const uint32_t BUF_MASK = (uint32_t)(windowBytes - 1);    // window must be a power of 2; slices wrap within it
    const uint64_t BUFSZ = windowBytes + 65536;               // window + slack for the +2KB slice span
    std::vector<uint8_t> Ah(BUFSZ), Bh(BUFSZ);
    { std::mt19937 rg(0xA5A5u); for (uint64_t i=0;i<BUFSZ;i++) Ah[i]=(uint8_t)rg(); }   // random -> sum robustly != 0
    { std::mt19937 rg(0x5A5Au); for (uint64_t i=0;i<BUFSZ;i++) Bh[i]=(uint8_t)rg(); }

    size_t isaLen=0; uint8_t* isaBytes=ReadFile(isaPath,&isaLen);
    GpuBuf isa=AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ=AllocGpu(node,0x1000,false,true);
    GpuBuf Ad=AllocGpu(node,(BUFSZ+0xFFF)&~0xFFFull,false,true);
    GpuBuf Bd=AllocGpu(node,(BUFSZ+0xFFF)&~0xFFFull,false,true);
    GpuBuf C =AllocGpu(node,0x1000,false,true);              // unused at rung 1 (valid s[6:7])
    GpuBuf fence=AllocGpu(node,0x1000,false,true);
    memcpy(isa.ptr,isaBytes,isaLen); free(isaBytes);
    memcpy(Ad.ptr,Ah.data(),BUFSZ); memcpy(Bd.ptr,Bh.data(),BUFSZ);
    volatile uint32_t* occW=(volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW=(volatile uint32_t*)fence.ptr;
    for (int i=0;i<20;i++) occW[i]=0; occW[2]=0xFFFFFFFFu; *fenceW=0;

    Ring ring; ring.buf=AllocGpu(node,0x10000,true,true); ring.dw=(uint32_t*)ring.buf.ptr;
    ring.sizeDw=(uint32_t)(ring.buf.size/sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node,HSA_QUEUE_COMPUTE,100,HSA_QUEUE_PRIORITY_NORMAL,ring.buf.ptr,ring.buf.size,nullptr,&ring.res));
    uint64_t shiftedIsa=((uint64_t)isa.ptr)>>8;
    uint64_t occVa=(uint64_t)occ.ptr,aVa=(uint64_t)Ad.ptr,bVa=(uint64_t)Bd.ptr,cVa=(uint64_t)C.ptr,fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8]={0,0,0,32,1,1,0,0};                     // WG = 32 threads = 1 wave
    uint32_t pgm[6]={(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1=BuildPgmRsrc1(false); rsrc1=(rsrc1&~0x3fu)|(8u&0x3fu);   // 64 VGPR (v0..v63)
    uint32_t rsrc2=(BuildPgmRsrc2(false)&~0x3eu)|(15u<<RSRC2_USER_SGPR_SHIFT);   // no LDS
    uint32_t rsrc[2]={rsrc1,rsrc2};
    uint32_t reslim[1]={0},tmpring[1]={0},restart[4]={0,0,0,0};
    uint32_t userdata[16]={
        (uint32_t)occVa,(uint32_t)(occVa>>32),(uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32),(uint32_t)cVa,(uint32_t)(cVa>>32),
        0,(uint32_t)SLICE_STRIDE,(uint32_t)BUF_MASK,0, 0,(uint32_t)CLAIM_CEIL,0,0 };  // s9=stride s10=mask s13=CEIL
    uint32_t dispInit=BuildDispatchInitiator();
    RingPlace(ring,PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_START_X,dims,8));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,pgm,6));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,rsrc,2));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS,reslim,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,tmpring,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,restart,4));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,userdata,16));
    RingPlace(ring,PM4DispatchDirectPacket(nWG*32u,1,1,dispInit));
    RingPlace(ring,PM4ReleaseMemoryPacket(FAMILY_GFX12,true,fenceVa,FENCE_VALUE));

    double t0=now_s(); RingSubmit(ring);
    bool done=false,admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true){ double now=now_s();
        if (occW[1]>0) admitted=true;
        uint32_t end=occW[3]; if (end!=lastEnd){lastEnd=end;lastEndChange=now;}
        bool ff=(*fenceW==FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025){done=true;break;}
        if (now-t0>40.0) break;
    }
    if (!done){ fprintf(stderr,"\n*** STACK TIMEOUT (%s): live=%u maxlive=%u claims=%u ***\n",isaPath,occW[0],occW[1],occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(C);FreeGpu(Bd);FreeGpu(Ad);FreeGpu(occ);FreeGpu(isa); return res; }
    res.ok=true; res.maxlive=occW[1]; res.claims=occW[5]; res.consumed=occW[6]; res.got=occW[7];
    { uint32_t gs=occW[2],ge=occW[3]; res.wall=(ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    double secs = (double)res.wall / freq_hz;
    uint64_t totalBytes = (uint64_t)CLAIM_CEIL * FRAGS * 256ull * 2ull;   // A+B
    res.gbps = secs>0 ? (double)totalBytes / secs / 1e9 : 0.0;
    res.tfeq = res.gbps * (1.4/2.7);                                       // real-kernel yardstick (1.4 TF @ 2.7 GB/s)
    // ---- CPU re-sum over EXACTLY the streamed slices [0,CEIL) (u32 wraparound) -> traffic proof. Each slice
    //      reads [base, base+FRAGS*256) contiguous from A and B (32 lanes x 8B span every 256B frag). claims
    //      over-counts (each WG does one atomic before the s17>=CEIL exit check) so the PROOF is the checksum. --
    const uint32_t* A32=(const uint32_t*)Ah.data();
    const uint32_t* B32=(const uint32_t*)Bh.data();
    uint32_t expSum=0;
    const uint32_t DW = FRAGS*256/4;                          // dwords per slice per buffer (= FRAGS*64)
    for (uint32_t s=0; s<CLAIM_CEIL; ++s) {
        uint32_t base = (uint32_t)((uint64_t)s * SLICE_STRIDE) & BUF_MASK;
        const uint32_t* a=A32 + base/4; const uint32_t* b=B32 + base/4;
        for (uint32_t d=0; d<DW; ++d) expSum += a[d] + b[d];
    }
    // proof = slice-count exact (loop ran for every slice) AND data checksum matches AND is non-trivial (!=0)
    res.exp = expSum; res.proof = (res.consumed == CLAIM_CEIL) && (res.got == expSum) && (expSum != 0);
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(C);FreeGpu(Bd);FreeGpu(Ad);FreeGpu(occ);FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 CLEAN PM4 STREAMING BANDWIDTH PROBE (occ_kernel_bw.s). One atomic/wave for a dense worker id, then
// a pure streaming hot loop. Prove the raw-PM4 vehicle moves data near the ~640 GB/s spec before any GEMM.
// ---------------------------------------------------------------------------
struct BwResult { bool ok=false; uint32_t maxlive=0, workers=0, steps=0; uint64_t wall=0, bytes=0; double gbps=0;
                  bool proof=false; uint32_t chk=0; };

// ===========================================================================
// MAD-305 in-ring GL2C DRAM meter (gfx1201). Brackets a dispatch with GL2C
// hardware perfcounters read from WITHIN our own KFD ring -- the only profiler
// that can see our raw-PM4 kernels. Register map + sequence: pm4_perf.h +
// GFX12_PERFCOUNTER_REGISTERS.md. Result BO layout: instance-major, 8 dwords
// per instance = 4 counters x (LO,HI). Counters: 0=RDREQ_32B 1=RDREQ_64B
// 2=RDREQ_128B 3=WRREQ.
// ===========================================================================
static const uint32_t GL2C_DW_PER_INSTANCE = 8;   // 4 counters * 2 dwords (LO,HI)

// Emit right AFTER ACQUIRE_MEM, before the dispatch: reset, program the 4 GL2C
// counters (broadcast -- GLOBAL block), start counting.
static void gl2c_emit_setup_start(Ring& ring) {
    RingPlace(ring, PM4WriteRegPacket(mmCP_PERFMON_CNTL, CpPerfmonCntl(CP_PERFMON_STATE_DISABLE_RESET, false)));
    RingPlace(ring, PM4WriteRegPacket(mmGRBM_GFX_INDEX, GrbmBroadcastAll()));
    RingPlace(ring, PM4WriteRegPacket(mmGL2C_PERFCOUNTER0_SELECT + 0*GL2C_SELECT_STRIDE, Gl2cSelect(GL2C_EVT_EA_RDREQ_32B)));
    RingPlace(ring, PM4WriteRegPacket(mmGL2C_PERFCOUNTER0_SELECT + 1*GL2C_SELECT_STRIDE, Gl2cSelect(GL2C_EVT_EA_RDREQ_64B)));
    RingPlace(ring, PM4WriteRegPacket(mmGL2C_PERFCOUNTER0_SELECT + 2*GL2C_SELECT_STRIDE, Gl2cSelect(GL2C_EVT_EA_RDREQ_128B)));
    RingPlace(ring, PM4WriteRegPacket(mmGL2C_PERFCOUNTER0_SELECT + 3*GL2C_SELECT_STRIDE, Gl2cSelect(GL2C_EVT_EA_WRREQ)));
    RingPlace(ring, PM4WriteRegPacket(mmCP_PERFMON_CNTL, CpPerfmonCntl(CP_PERFMON_STATE_START, false)));
}

// Emit right AFTER the dispatch, before RELEASE_MEM: drain waves, stop+latch,
// then per-instance read the 4 counters (64-bit LO+HI) into resultVa.
static void gl2c_emit_stop_read(Ring& ring, uint64_t resultVa) {
    RingPlace(ring, PM4PartialFlushPacket());   // CS_PARTIAL_FLUSH: wait for all waves to retire
    RingPlace(ring, PM4WriteRegPacket(mmCP_PERFMON_CNTL, CpPerfmonCntl(CP_PERFMON_STATE_STOP, /*sample*/true)));
    for (uint32_t i = 0; i < NUM_GL2C_INSTANCES; i++) {
        RingPlace(ring, PM4WriteRegPacket(mmGRBM_GFX_INDEX, GrbmSelectGl2cInstance(i)));
        uint64_t base = resultVa + (uint64_t)i * GL2C_DW_PER_INSTANCE * sizeof(uint32_t);
        for (uint32_t c = 0; c < 4; c++) {
            RingPlace(ring, PM4CopyDataPacket(COPY_DATA_SRC_PERFCOUNTERS,
                                              mmGL2C_PERFCOUNTER0_LO + c * GL2C_RESULT_STRIDE, 0,
                                              base + (uint64_t)c * 2 * sizeof(uint32_t),
                                              /*count64*/true));
        }
    }
    RingPlace(ring, PM4WriteRegPacket(mmGRBM_GFX_INDEX, GrbmBroadcastAll()));   // restore broadcast
}

// CPU-side reduce + report. `secs` is the measured dispatch wall time; `logicalBytes`
// is the probe's own intended traffic for a sanity cross-check.
static void gl2c_reduce_report(const volatile uint32_t* R, double secs, uint64_t logicalBytes, const char* tag) {
    uint64_t s_rd32=0, s_rd64=0, s_rd128=0, s_wr=0;
    uint32_t nonzero_inst = 0;
    for (uint32_t i = 0; i < NUM_GL2C_INSTANCES; i++) {
        const volatile uint32_t* p = R + (size_t)i * GL2C_DW_PER_INSTANCE;
        auto v64 = [&](int c)->uint64_t { return (uint64_t)p[c*2] | ((uint64_t)p[c*2+1] << 32); };
        uint64_t a=v64(0), b=v64(1), d=v64(2), w=v64(3);
        if (a|b|d|w) nonzero_inst++;
        s_rd32 += a; s_rd64 += b; s_rd128 += d; s_wr += w;
    }
    uint64_t fetch = Gl2cFetchBytes(s_rd32, s_rd64, s_rd128);
    double fetch_gbps = secs > 0 ? (double)fetch / secs / 1e9 : 0.0;
    printf("  [GL2C %s] instances_nonzero=%u/%u  RDREQ 32B=%llu 64B=%llu 128B=%llu  WRREQ=%llu\n",
           tag, nonzero_inst, NUM_GL2C_INSTANCES,
           (unsigned long long)s_rd32, (unsigned long long)s_rd64,
           (unsigned long long)s_rd128, (unsigned long long)s_wr);
    printf("  [GL2C %s] FETCH = %llu bytes (%.2f MB) -> %.1f GB/s effective DRAM-read  | probe logical=%llu B (%.2f MB)\n",
           tag, (unsigned long long)fetch, fetch/1e6, fetch_gbps,
           (unsigned long long)logicalBytes, logicalBytes/1e6);
    if (nonzero_inst == 0)
        printf("  [GL2C %s] *** ALL COUNTERS ZERO -- plain register writes likely dropped; switch SELECT/CNTL to COPY_DATA->perfcounters (perf window). See GFX12_PERFCOUNTER_REGISTERS.md risk note. ***\n", tag);
}

static BwResult run_bw(uint32_t node, const char* isaPath, double freq_hz, int mode, uint32_t LDW,
                       uint64_t windowBytes, uint32_t STEPS, uint32_t WGSIZE, uint32_t nWG) {
    BwResult res;
    const uint32_t STEP = 32u*LDW;                            // bytes per coalesced wavefront load
    const uint32_t BUF_MASK = (uint32_t)(windowBytes - 1);
    const uint64_t SPAN = (uint64_t)STEPS * STEP;             // per-worker contiguous span
    const uint64_t BUFSZ = windowBytes + SPAN + 65536;        // window + worst-case worker span + slack
    std::vector<uint8_t> src(BUFSZ);
    { std::mt19937 rg(0xBEEF); for (uint64_t i=0;i<BUFSZ;i++) src[i]=(uint8_t)rg(); }

    size_t isaLen=0; uint8_t* isaBytes=ReadFile(isaPath,&isaLen);
    GpuBuf isa=AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ=AllocGpu(node,0x1000,false,true);
    GpuBuf Sd =AllocGpu(node,(BUFSZ+0xFFF)&~0xFFFull,false,true,/*deviceLocal*/true);   // VRAM stream src
    GpuBuf Dd =AllocGpu(node,(BUFSZ+0xFFF)&~0xFFFull,false,true,/*deviceLocal*/true);   // VRAM write/copy dst
    GpuBuf sink=AllocGpu(node,0x1000,false,true);
    GpuBuf fence=AllocGpu(node,0x1000,false,true);
    GpuBuf gl2c=AllocGpu(node,0x1000,false,true);   // GL2C perfcounter result BO (32 inst * 8 dw), host-visible
    if (!(Sd.vram && Dd.vram)) {   // VRAM GUARD: the BW probe MUST measure device-local memory, not PCIe/GTT
        fprintf(stderr, "\n*** BW VRAM GUARD FAILED (%s): src/dst not device-local (S=%d D=%d) ***\n", isaPath, Sd.vram, Dd.vram);
        abort();
    }
    memcpy(isa.ptr,isaBytes,isaLen); free(isaBytes);
    memcpy(Sd.ptr,src.data(),BUFSZ);
    volatile uint32_t* occW=(volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW=(volatile uint32_t*)fence.ptr;
    for (int i=0;i<20;i++) occW[i]=0; occW[2]=0xFFFFFFFFu; *fenceW=0;
    memset(gl2c.ptr,0,0x1000);   // zero the perfcounter result BO before the run

    Ring ring; ring.buf=AllocGpu(node,0x10000,true,true); ring.dw=(uint32_t*)ring.buf.ptr;
    ring.sizeDw=(uint32_t)(ring.buf.size/sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node,HSA_QUEUE_COMPUTE,100,HSA_QUEUE_PRIORITY_NORMAL,ring.buf.ptr,ring.buf.size,nullptr,&ring.res));
    uint64_t shiftedIsa=((uint64_t)isa.ptr)>>8;
    uint64_t occVa=(uint64_t)occ.ptr,sVa=(uint64_t)Sd.ptr,dVa=(uint64_t)Dd.ptr,kVa=(uint64_t)sink.ptr,fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8]={0,0,0,WGSIZE,1,1,0,0};
    uint32_t pgm[6]={(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1=BuildPgmRsrc1(false); rsrc1=(rsrc1&~0x3fu)|(8u&0x3fu);   // 64 VGPR
    uint32_t rsrc2=(BuildPgmRsrc2(false)&~0x3eu)|(15u<<RSRC2_USER_SGPR_SHIFT);   // no LDS
    uint32_t rsrc[2]={rsrc1,rsrc2};
    uint32_t reslim[1]={0},tmpring[1]={0},restart[4]={0,0,0,0};
    uint32_t userdata[16]={
        (uint32_t)occVa,(uint32_t)(occVa>>32),(uint32_t)sVa,(uint32_t)(sVa>>32),
        (uint32_t)dVa,(uint32_t)(dVa>>32),(uint32_t)kVa,(uint32_t)(kVa>>32),
        0,0,0,(uint32_t)BUF_MASK, 0,(uint32_t)STEPS,0,0 };   // s11=BUF_MASK s12=NWORKERS(0=no cap) s13=STEPS
    uint32_t dispInit=BuildDispatchInitiator();
    RingPlace(ring,PM4AcquireMemoryPacket(FAMILY_GFX12));
    if (g_gl2c) gl2c_emit_setup_start(ring);   // reset+program+start GL2C counters before the dispatch
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_START_X,dims,8));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,pgm,6));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,rsrc,2));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS,reslim,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,tmpring,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,restart,4));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,userdata,16));
    RingPlace(ring,PM4DispatchDirectPacket(nWG*WGSIZE,1,1,dispInit));
    if (g_gl2c) gl2c_emit_stop_read(ring, (uint64_t)gl2c.ptr);   // drain+stop+read counters before the fence
    RingPlace(ring,PM4ReleaseMemoryPacket(FAMILY_GFX12,true,fenceVa,FENCE_VALUE));

    double t0=now_s(); RingSubmit(ring);
    bool done=false,admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true){ double now=now_s();
        if (occW[1]>0) admitted=true;
        uint32_t end=occW[3]; if (end!=lastEnd){lastEnd=end;lastEndChange=now;}
        bool ff=(*fenceW==FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025){done=true;break;}
        if (now-t0>40.0) break;
    }
    if (!done){ fprintf(stderr,"\n*** BW TIMEOUT (%s): live=%u maxlive=%u workers=%u ***\n",isaPath,occW[0],occW[1],occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf);FreeGpu(gl2c);FreeGpu(fence);FreeGpu(sink);FreeGpu(Dd);FreeGpu(Sd);FreeGpu(occ);FreeGpu(isa); return res; }
    res.ok=true; res.maxlive=occW[1]; res.workers=occW[5]; res.steps=occW[6]; res.chk=occW[7];
    { uint32_t gs=occW[2],ge=occW[3]; res.wall=(ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    double secs=(double)res.wall/freq_hz;
    uint64_t per = (uint64_t)res.workers * STEPS * STEP;       // bytes moved per direction
    res.bytes = (mode==1) ? 2u*per : per;                     // copy moves 2x (load+store)
    res.gbps = secs>0 ? (double)res.bytes/secs/1e9 : 0.0;
    // proof: every worker did exactly STEPS steps (occ[6]==workers*STEPS) AND (read/copy) checksum != 0
    bool cntOK = ((uint64_t)res.steps == (uint64_t)res.workers * STEPS) && res.workers>0;
    res.proof = cntOK && ((mode==2) ? true : (res.chk != 0));
    if (g_gl2c) gl2c_reduce_report((volatile uint32_t*)gl2c.ptr, secs, res.bytes, isaPath);
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf);FreeGpu(gl2c);FreeGpu(fence);FreeGpu(sink);FreeGpu(Dd);FreeGpu(Sd);FreeGpu(occ);FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 Step 2.4 residency probe: BARRIER-FREE NOFEED. Tiny buffers (operands are garbage), no LDS,
// no barriers. Launch with `threads` per WG (128 = 4-wave probe, 32 = one-wave clone). claim_ceil =
// M*N/4096 so total WMMA = claim_ceil*K = M*N*K/4096 -> TF = 2*M*N*K/wall stays calibrated.
// ---------------------------------------------------------------------------
struct BfResult { bool ok=false; uint32_t maxlive=0, claims=0; uint64_t wall=0; double tf=0; };

static BfResult run_nofeed_bf(uint32_t node, const char* isaPath, int M, int N, int K,
                              uint32_t threads, uint32_t nWG, double freq_hz) {
    BfResult res;
    int NTILES = K / 32;
    uint32_t CLAIM_CEIL = (uint32_t)(((uint64_t)M * N) / 4096);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, 0x1000, false, true);   // operands are garbage -> tiny
    GpuBuf Bd  = AllocGpu(node, 0x1000, false, true);
    GpuBuf C   = AllocGpu(node, 0x1000, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[5]=0; occW[16]=0; *fenceW=0;  // occ[5]=claim ctr (byte 20)

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,threads,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (24u & 0x3fu);   // 192 VGPR
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT); // LDS=0
    printf("  [bf] %dx%dx%d  threads=%u/WG nWG=%u  NTILES=%d claim_ceil=%u  VGPR=192 LDS=0 RSRC2=0x%x\n",
           M,N,K, threads, nWG, NTILES, CLAIM_CEIL, rsrc2);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, 0, 0, 0, (uint32_t)NTILES, CLAIM_CEIL, 0, 0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * threads, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 40.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** NOFEED-BF TIMEOUT (%s): live=%u maxlive=%u claims=%u ***\n", isaPath, occW[0], occW[1], occW[20]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.claims = occW[5];   // claim counter lives at byte 20 = occW[5]
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.tf = (res.wall > 0) ? (2.0*(double)M*N*K * freq_hz / (double)res.wall / 1e12) : 0.0;
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 Step A: FEED-ONLY depth-P pipeline BANDWIDTH probe (occ_kernel_feedpipe.s). No WMMA/LDS/barrier.
// Streams a 64 MiB buffer keeping PDEPTH slices of FRAGS b64 loads in flight. Reports effective feed
// bandwidth (GB/s) so we can see if it scales above the proven ~2.7 GB/s serialized baseline.
//   bytes moved = CLAIM_CEIL slices * (FRAGS loads * 32 lanes * 8 B) = CLAIM_CEIL * FRAGS * 256.
// ---------------------------------------------------------------------------
struct FpResult { bool ok=false; uint32_t maxlive=0, claims=0; uint64_t wall=0; double gbps=0; };

static FpResult run_feedpipe(uint32_t node, const char* isaPath, uint32_t threads, uint32_t nWG,
                             uint32_t claimCeil, uint32_t frags, uint32_t vgprField, double freq_hz) {
    FpResult res;
    const uint64_t STREAM = 64u*1024*1024;             // 64 MiB stream buffer (BUF_MASK=0x3FFFFFF)
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, STREAM, false, false);  // cached -> realistic L2/DRAM feed path
    GpuBuf Bd  = AllocGpu(node, 0x1000, false, true);    // unused by the probe
    GpuBuf C   = AllocGpu(node, 0x1000, false, true);    // sink
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[5]=0; occW[16]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,threads,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT); // LDS=0
    printf("  [fp] threads=%u/WG nWG=%u  frags=%u claim_ceil=%u  VGPR=%u LDS=0\n",
           threads, nWG, frags, claimCeil, vgprField*8);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        0, 0, 0, 0, 0, claimCeil, 0, 0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * threads, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 60.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** FEEDPIPE TIMEOUT (%s): live=%u maxlive=%u claims=%u ***\n", isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.claims = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    double secs = (res.wall > 0) ? ((double)res.wall / freq_hz) : 0.0;
    double bytes = (double)claimCeil * (double)frags * 256.0;
    res.gbps = (secs > 0) ? (bytes / secs / 1e9) : 0.0;
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 Step A localization ladder (occ_kernel_feedladder.s). Adds the 4-wave / barrier / LDS-A-share
// couplings back onto the 123 GB/s feed baseline one at a time to find the collapse point. Fixed NSLICES
// per wave (no atomic-claim), launched waves = nWG*(threads/32). bytes = launched_waves*NSLICES*FRAGS*256.
// ---------------------------------------------------------------------------
static FpResult run_feedladder(uint32_t node, const char* isaPath, uint32_t threads, uint32_t nWG,
                               uint32_t nslices, uint32_t frags, uint32_t vgprField, uint32_t ldsBytes, double freq_hz) {
    FpResult res;
    const uint64_t STREAM = 64u*1024*1024;
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, STREAM, false, false);
    GpuBuf Bd  = AllocGpu(node, 0x1000, false, true);
    GpuBuf C   = AllocGpu(node, 0x1000, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,threads,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    uint32_t ldsU=0, ldsA=0, ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT);
    rsrc2 = (rsrc2 & ~(0x1FFu << 15)) | ldsBits;
    uint64_t lw = (uint64_t)nWG * (threads / 32u);
    printf("  [fl] threads=%u/WG nWG=%u (waves=%llu)  nslices=%u frags=%u  VGPR=%u LDS=%uB\n",
           threads, nWG, (unsigned long long)lw, nslices, frags, vgprField*8, ldsA);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        0, 0, 0, 0, 0, nslices, 0, 0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * threads, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 60.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** FEEDLADDER TIMEOUT (%s): live=%u maxlive=%u ***\n", isaPath, occW[0], occW[1]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.claims = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    double secs = (res.wall > 0) ? ((double)res.wall / freq_hz) : 0.0;
    double bytes = (double)lw * (double)nslices * (double)frags * 256.0;
    res.gbps = (secs > 0) ? (bytes / secs / 1e9) : 0.0;
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// MAD-305 Step A rung 6: B global_load_tr_b64 probe (occ_kernel_btr.s). 64 MiB stream as Bshuf (s4:5),
// s9=nt256=(N/16)*256. 8 frags (2 kk * FN=4) per slice. bytes = launched_waves*NSLICES*8*256.
// ---------------------------------------------------------------------------
static FpResult run_btr(uint32_t node, const char* isaPath, uint32_t threads, uint32_t nWG,
                        uint32_t nslices, uint32_t vgprField, uint32_t ldsBytes, uint32_t nt256, double freq_hz) {
    FpResult res;
    // 64 MiB addressed range (BUF_MASK=0x3FFFFFF) + slack: real-Bshuf reads up to s20+s9+frag span (~66 KB)
    // past the wrap, so the allocation must extend beyond 64 MiB or those loads fault OOB and the wave hangs.
    const uint64_t STREAM = 96u*1024*1024;
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf Ad  = AllocGpu(node, 0x1000, false, true);    // unused
    GpuBuf Bd  = AllocGpu(node, STREAM, false, false);    // Bshuf stream (cached)
    GpuBuf C   = AllocGpu(node, 0x1000, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[5]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,threads,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (vgprField & 0x3fu);
    uint32_t ldsU=0, ldsA=0, ldsG=0; uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsU, &ldsA, &ldsG);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT);
    rsrc2 = (rsrc2 & ~(0x1FFu << 15)) | ldsBits;
    uint64_t lw = (uint64_t)nWG * (threads / 32u);
    printf("  [btr] threads=%u/WG nWG=%u (waves=%llu)  nslices=%u nt256=%u  VGPR=%u LDS=%uB\n",
           threads, nWG, (unsigned long long)lw, nslices, nt256, vgprField*8, ldsA);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        0, nt256, 0, 0, 0, nslices, 0, 0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * threads, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > 60.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** BTR TIMEOUT (%s): live=%u maxlive=%u ***\n", isaPath, occW[0], occW[1]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.claims = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    double secs = (res.wall > 0) ? ((double)res.wall / freq_hz) : 0.0;
    double bytes = (double)lw * (double)nslices * 8.0 * 256.0;
    res.gbps = (secs > 0) ? (bytes / secs / 1e9) : 0.0;
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE = bits[15:23] (9-bit), per amd_hsa_kernel_code.h:122 /
// LLVM AMDHSAKernelDescriptor.h. gfx10/11/12 LDS encode granule = 512 bytes; units = ceil(bytes/512).
static uint32_t ldsRsrc2Bits(uint32_t ldsBytes, uint32_t* outUnits, uint32_t* outAlloc, uint32_t* outGranule) {
    const uint32_t GRANULE = 512;
    uint32_t units = (ldsBytes + GRANULE - 1) / GRANULE;
    if (outGranule) *outGranule = GRANULE;
    if (outUnits)   *outUnits   = units;
    if (outAlloc)   *outAlloc   = units * GRANULE;
    return (units & 0x1FFu) << 15;
}

// ---------------------------------------------------------------------------
// MAD-305 Lever A micro-oracle (GPT step 1): PROVE the fp8 fragment semantics of global_load_tr_b128
// before believing it. Single wave, occ_kernel_btr128.s. B is encoded so each byte VALUE = its global
// K-row index (tile0=K0..15 @ [0,256), tile1=K16..31 @ [256,512)). The kernel dumps, per lane,
//   [ tr_b64(tile0) 8B | tr_b64(tile1) 8B | tr_b128(both) 16B ].
// Verified frag map (frag_layout.h pack_B): lane L -> col=L&0xF, rowhi=(L>>4)&1, kbase=rowhi*8; lo/hi =
//   K[kbase..kbase+7] of that col. With byte==K, expected f0[i]=kbase+i, f1[i]=16+kbase+i.
// Lever A is fp8-correct iff tr_b128 is two adjacent fp8 frags: b128[0..7]==f0 AND b128[8..15]==f1.
// (If b128 is a 16-bit-oriented transpose, the bytes scramble -> the dump shows the real layout.)
// returns 0 = probe valid (b128 verdict printed), 2 = probe self-check failed, 3 = hang/error.
// ---------------------------------------------------------------------------
static int run_btr128_oracle(uint32_t node, const char* isaPath) {
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    if (!isaBytes) { fprintf(stderr, "  [btr128] cannot read %s\n", isaPath); return 3; }
    GpuBuf isa  = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ  = AllocGpu(node, 0x1000,  false, true);   // unused by kernel
    GpuBuf Ad   = AllocGpu(node, 0x1000,  false, true);   // unused
    GpuBuf Bd   = AllocGpu(node, 0x10000, false, true);   // 64 KB host-visible (only 512B used)
    GpuBuf sink = AllocGpu(node, 0x1000,  false, true);   // 32B/lane * 32 lanes = 1024B
    GpuBuf fence= AllocGpu(node, 0x1000,  false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);

    // encode B: byte at offset o (0..511): value = (o/256)*16 + (o%256)/16 = its global K-row index.
    uint8_t* B = (uint8_t*)Bd.ptr; memset(B, 0, 0x10000);
    for (int o = 0; o < 512; ++o) { int tile=o/256, klocal=(o%256)/16; B[o] = (uint8_t)(tile*16 + klocal); }
    uint8_t* S = (uint8_t*)sink.ptr; memset(S, 0xEE, 0x1000);
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr; *fenceW = 0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, aVa=(uint64_t)Ad.ptr, bVa=(uint64_t)Bd.ptr, sVa=(uint64_t)sink.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,32,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | (8u & 0x3fu);   // 64 VGPR (kernel uses up to v47)
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)sVa,(uint32_t)(sVa>>32),
        0,0,0,0,0,0,0,0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(32, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    bool done=false;
    while (true) { double now = now_s();
        if (*fenceW == FENCE_VALUE) { done=true; break; }
        if (now - t0 > 10.0) break;
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    if (!done) {
        fprintf(stderr, "  [btr128] *** TIMEOUT (kernel hang? tr_b128 OOB?) ***\n");
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(sink); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return 3;
    }
    // decode: f0/f1 = known-good frags; b128 = candidate. Expected f0[i]=kbase+i, f1[i]=16+kbase+i.
    int knownGoodOK = 1, b128OK = 1;
    for (int L = 0; L < 32; ++L) {
        const uint8_t* f0   = S + L*32 + 0;
        const uint8_t* f1   = S + L*32 + 8;
        const uint8_t* b128 = S + L*32 + 16;
        int kbase = ((L >> 4) & 1) * 8;
        for (int i = 0; i < 8; ++i) {
            if (f0[i] != (uint8_t)(kbase+i))    knownGoodOK = 0;
            if (f1[i] != (uint8_t)(16+kbase+i)) knownGoodOK = 0;
            if (b128[i]   != f0[i])             b128OK = 0;
            if (b128[8+i] != f1[i])             b128OK = 0;
        }
    }
    printf("\n=== MAD-305 Lever A micro-oracle: global_load_tr_b128 fp8 fragment semantics ===\n");
    printf("  known-good tr_b64 (sanity): %s\n",
           knownGoodOK ? "OK (probe + frag map verified)"
                       : "*** WRONG -- probe/addressing bug; b128 verdict inconclusive ***");
    printf("  tr_b128 == two adjacent fp8 frags: %s\n",
           b128OK ? "*** YES -- Lever A is fp8-correct; integrate b128 (halves B feed instrs) ***"
                  : "NO -- b128 is NOT a drop-in 2-frag fp8 load (likely 16-bit transpose). Do NOT integrate B b128.");
    if (!b128OK || !knownGoodOK) {
        printf("  --- per-lane forensic dump (values = K-row index)  L: f0[8] | f1[8] | b128[16] ---\n");
        for (int L = 0; L < 32; ++L) {
            const uint8_t* p = S + L*32;
            printf("   L%02d: ", L);
            for (int i=0;i<8;i++)  printf("%2u ", p[i]);     printf("| ");
            for (int i=0;i<8;i++)  printf("%2u ", p[8+i]);   printf("| ");
            for (int i=0;i<16;i++) printf("%2u ", p[16+i]);  printf("\n");
        }
    }
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(sink); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
    return knownGoodOK ? 0 : 2;
}

// ---------------------------------------------------------------------------
// WAVE-GROUP ATOMIC-CLAIM + LDS-BROADCAST smoke (MAD-305 Phase 1 pivot): nWG persistent 128-thread
// workgroups; each leader atomic-claims a tile, broadcasts it to the 4 waves via LDS+barrier, every
// wave writes its decode mark. Verifies each tile covered once with 4 correct wave marks. No TGID.
// ---------------------------------------------------------------------------
static WgResult run_wglds_smoke(uint32_t node, const char* isaPath, int M, int N, uint32_t nWG,
                                int TWM, int TWN, uint32_t ldsBytes) {
    WgResult res;
    const int TM = 128, TN = 128;
    int MTL = M / TM, NTL = N / TN;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not power of two\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;
    int WAVES = TWM * TWN;

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    uint64_t cbytes = ((uint64_t)TOTAL * WAVES * 4 + 0xFFF) & ~0xFFFull;
    GpuBuf C   = AllocGpu(node, cbytes, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    occW[0]=0; occW[1]=0; occW[2]=0xFFFFFFFFu; occW[3]=0; occW[4]=0; occW[5]=0; *fenceW=0;   // occ[5]=claim counter
    uint32_t* Cw = (uint32_t*)C.ptr;
    for (uint64_t i = 0; i < (uint64_t)TOTAL * WAVES; ++i) Cw[i] = 0xFFFFFFFFu;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, cVa=(uint64_t)C.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,(uint32_t)(WAVES*32),1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | 4u;        // static 32 VGPR
    uint32_t ldsUnits=0, ldsAlloc=0, ldsGranule=0;
    uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsUnits, &ldsAlloc, &ldsGranule);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    printf("  [LDS] request=%u B  granule=%u B  units=%u -> alloc=%u B  RSRC2.LDS_SIZE bits=0x%x  RSRC2=0x%x\n",
           ldsBytes, ldsGranule, ldsUnits, ldsAlloc, ldsBits, rsrc2);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32),
        (uint32_t)cVa, (uint32_t)(cVa>>32),
        TOTAL, 0u, (uint32_t)(NTL-1), (uint32_t)log2NTL,
        0,0,0,0, 0,0,0,0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * (uint32_t)(WAVES*32), 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    const double timeoutS = 20.0;
    double t0 = now_s(); RingSubmit(ring);
    bool done=false, admitted=false; uint32_t lastEnd=0; double lastEndChange=t0;
    while (true) { double now = now_s();
        if (occW[1] > 0) admitted = true;
        uint32_t end = occW[3]; if (end != lastEnd) { lastEnd = end; lastEndChange = now; }
        bool ff = (*fenceW == FENCE_VALUE);
        if (admitted && occW[0]==0 && ff && end!=0 && (now-lastEndChange)>0.025) { done=true; break; }
        if (now - t0 > timeoutS) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** WGLDS SMOKE TIMEOUT (%s): live=%u maxlive=%u claim=%u (queue may be hung) ***\n",
                isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[5];   // total = claim counter end
    for (uint32_t ti = 0; ti < TOTAL; ++ti) {
        uint32_t trow = ti >> log2NTL, tcol = ti & (uint32_t)(NTL - 1);
        for (int wid = 0; wid < WAVES; ++wid) {
            uint32_t got = Cw[(uint64_t)ti * WAVES + wid];
            if (got == 0xFFFFFFFFu) { res.missMarks++; continue; }
            uint32_t wm = (uint32_t)(wid / TWN), wn = (uint32_t)(wid % TWN);
            uint32_t want = (trow << 20) | (tcol << 8) | (wm << 4) | wn;
            if (got == want) res.okMarks++; else res.badMarks++;
        }
    }
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(occ); FreeGpu(isa);
    return res;
}

// ---------------------------------------------------------------------------
// SGPR PROBE: launch nWG 128-thread workgroups (USER_SGPR=15); each leader dumps entry s8..s23 to
// occ[256 + ord*64]. Finds which SGPR carries the per-workgroup id. Completion = atomic counter
// occ[4] reaches nWG (the probe kernel has no live/timer bookkeeping).
// ---------------------------------------------------------------------------
static void run_sgpr_probe(uint32_t node, const char* isaPath, uint32_t nWG) {
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x4000, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    for (int i = 0; i < 0x1000; ++i) occW[i] = 0; *fenceW = 0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,128,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | 8u;   // 64 VGPR (probe uses s40..s65)
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT);
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = { (uint32_t)occVa,(uint32_t)(occVa>>32), 0,0,0,0,0,0, 0,0,0,0,0,0,0,0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(nWG * 128u, 1, 1, dispInit));
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    while (occW[4] < nWG && now_s() - t0 < 10.0) { /* spin */ }
    printf("  [SGPR-PROBE] nWG=%u  counter occ[4]=%u  (USER_SGPR=15 -> s15 expected = TGID_X)\n", nWG, occW[4]);
    printf("    ord |    s8       s9      s10      s11      s12      s13      s14    *s15*      s16      s17      s18      s19      s20      s21      s22      s23\n");
    for (uint32_t ord = 0; ord < nWG; ++ord) {
        printf("    %3u |", ord);
        for (int k = 0; k < 16; ++k) printf(" %8x", occW[64 + ord*16 + k]);
        printf("\n");
    }
    printf("  -> the column with values {0..%u} as a permutation across ords IS the workgroup-id SGPR.\n", nWG-1);
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(occ); FreeGpu(isa);
}

// ---------------------------------------------------------------------------
// LDS BOUNDARY smoke: one 4-wave workgroup writes/reads LDS[0] and LDS[ldsBytes-4]; verifies the
// raw-PM4 RSRC2.GRANULATED_LDS_SIZE encoding actually allocates the requested bytes at a real size.
// ---------------------------------------------------------------------------
static bool run_lds_bound(uint32_t node, const char* isaPath, uint32_t ldsBytes) {
    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ = AllocGpu(node, 0x1000, false, true);
    GpuBuf fence = AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    for (int i = 0; i < 64; ++i) occW[i] = 0; *fenceW = 0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw = (uint32_t*)ring.buf.ptr;
    ring.sizeDw = (uint32_t)(ring.buf.size / sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,128,1,1,0,0};
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(false); rsrc1 = (rsrc1 & ~0x3fu) | 4u;
    uint32_t ldsUnits=0, ldsAlloc=0, ldsGranule=0;
    uint32_t ldsBits = ldsRsrc2Bits(ldsBytes, &ldsUnits, &ldsAlloc, &ldsGranule);
    uint32_t rsrc2 = (BuildPgmRsrc2(false) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;
    uint32_t rsrc[2] = {rsrc1, rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = { (uint32_t)occVa,(uint32_t)(occVa>>32), 0,0, (ldsBytes-4),0,0,0, 0,0,0,0,0,0,0,0 };
    uint32_t dispInit = BuildDispatchInitiator();
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(128u, 1, 1, dispInit));   // 1 workgroup = 4 waves
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));

    double t0 = now_s(); RingSubmit(ring);
    while (occW[0] != 0xD0u && now_s() - t0 < 10.0) { /* spin for done flag */ }
    bool ok = (occW[0] == 0xD0u);
    int goodWaves = 0;
    for (int wid = 0; wid < 4; ++wid) {
        uint32_t lo = occW[8 + wid*2], hi = occW[8 + wid*2 + 1];
        bool w = (lo == 0xAAAA1111u && hi == 0xBBBB2222u);
        if (w) ++goodWaves;
    }
    printf("  ldsBytes=%5u (units=%u alloc=%u RSRC2=0x%x): done=%s  waves OK=%d/4  [w0 first=0x%x last=0x%x]\n",
           ldsBytes, ldsUnits, ldsAlloc, rsrc2, ok?"yes":"NO", goodWaves, occW[8], occW[9]);
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(occ); FreeGpu(isa);
    return ok && goodWaves == 4;
}

// ---------------------------------------------------------------------------------------------------
// DYNSMOKE: 2-wave dyn-VGPR coordination isolation probe (occ_kernel_dynsmoke.s, NO GEMM, occ-only).
// Replicates the coop dyn arming (lean-32 launch, RSRC2 bit6, PRIV) with a 2-wave WG, but strips the
// GEMM: wave0=feed (optional s_alloc 32 + park), wave1=compute (elastic grow GROWSZ -> write top VGPR ->
// store -> shrink -> signal). Only writes its own occ buffer (in-bounds: a brick can ONLY come from the
// s_alloc coordination itself). Bin via env ML8_SMOKE_BIN (default occ_dynsmoke_fa1.bin). PRIV via
// ML8_SMOKE_NOPRIV. This is the brick-SAFE way to characterize the 2-wave dyn rule the coop GEMM hit.
static void run_dynsmoke(uint32_t node) {
    const char* bin = getenv("ML8_SMOKE_BIN") ? getenv("ML8_SMOKE_BIN") : "occ_dynsmoke_fa1.bin";
    bool priv = !getenv("ML8_SMOKE_NOPRIV");
    { FILE* fb=fopen(bin,"rb"); if(!fb){ printf("*** dynsmoke bin '%s' not built -> REFUSING.\n", bin); return; } fclose(fb); }
    size_t isaLen=0; uint8_t* isaBytes = ReadFile(bin, &isaLen);
    GpuBuf isa  = AllocGpu(node, IsaMapBytes(isaLen), true, false);
    GpuBuf occ  = AllocGpu(node, 0x1000, false, true);
    GpuBuf fence= AllocGpu(node, 0x1000, false, true);
    memcpy(isa.ptr, isaBytes, isaLen); free(isaBytes);
    volatile uint32_t* occW = (volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW = (volatile uint32_t*)fence.ptr;
    for (int i=0;i<8;i++) occW[i]=0; *fenceW=0;

    Ring ring; ring.buf = AllocGpu(node, 0x10000, true, true); ring.dw=(uint32_t*)ring.buf.ptr;
    ring.sizeDw=(uint32_t)(ring.buf.size/sizeof(uint32_t));
    CHECK(hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, 100, HSA_QUEUE_PRIORITY_NORMAL, ring.buf.ptr, ring.buf.size, nullptr, &ring.res));

    uint64_t shiftedIsa = ((uint64_t)isa.ptr) >> 8;
    uint64_t occVa=(uint64_t)occ.ptr, fenceVa=(uint64_t)fence.ptr;
    uint32_t dims[8] = {0,0,0,64,1,1,0,0};                                   // NUM_THREAD_X=64 -> 2 waves/WG
    uint32_t pgm[6] = {(uint32_t)shiftedIsa,(uint32_t)(shiftedIsa>>32)|(g_is_dgpu?0u:(1u<<8)),0,0,0,0};
    uint32_t rsrc1 = BuildPgmRsrc1(priv); rsrc1 = (rsrc1 & ~0x3fu) | 4u;     // lean 32 launch (field 4)
    uint32_t ldsBytes = getenv("ML8_SMOKE_LDS") ? (uint32_t)atoi(getenv("ML8_SMOKE_LDS")) : 0u;  // LDSWAIT ping-pong needs 512
    uint32_t ldsU=0,ldsA=0,ldsG=0; uint32_t ldsBits = ldsBytes ? ldsRsrc2Bits(ldsBytes,&ldsU,&ldsA,&ldsG) : 0u;
    uint32_t rsrc2 = (BuildPgmRsrc2(true) & ~0x3eu) | (15u << RSRC2_USER_SGPR_SHIFT) | ldsBits;  // dyn bit6 + 15 user sgprs + LDS
    uint32_t rsrc[2]={rsrc1,rsrc2};
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16]={ (uint32_t)occVa,(uint32_t)(occVa>>32), 0,0,0,0,0,0, 0,0,0,0, 0,0,0,0 };
    uint32_t dispInit = BuildDispatchInitiator();

    printf("=== DYNSMOKE 2-wave dyn-VGPR probe: bin=%s priv=%d lds=%uB(units=%u) (occ+LDS-only, NO GEMM, in-bounds) ===\n", bin, priv, ldsBytes, ldsU);
    RingPlace(ring, PM4AcquireMemoryPacket(FAMILY_GFX12));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_START_X, dims, 8));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO, pgm, 6));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, rsrc, 2));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS, reslim, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE, tmpring, 1));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X, restart, 4));
    RingPlace(ring, PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0, userdata, 16));
    RingPlace(ring, PM4DispatchDirectPacket(64u, 1, 1, dispInit));           // 1 WG, 64 threads = 2 waves
    RingPlace(ring, PM4ReleaseMemoryPacket(FAMILY_GFX12, true, fenceVa, FENCE_VALUE));
    double t0=now_s(); RingSubmit(ring);
    bool done=false;
    while (true) { if (*fenceW == FENCE_VALUE) { done=true; break; } if (now_s()-t0 > 20.0) break; }
    printf("  DYNSMOKE %s\n  occ: feedAlive=0x%x grown=0x%x sentinel=0x%x shrunk=0x%x DONE=0x%x feedExit=0x%x\n",
           done ? "COMPLETED (fence fired) -> 2-wave elastic dyn WORKS" : "TIMEOUT (hang) -> 2-wave dyn wedged",
           occW[0],occW[1],occW[2],occW[3],occW[4],occW[5]);
    printf("  read: feedAlive should=0xFEE0  grown=0xC0E0  sentinel=0x5A5A5A5A  shrunk=0x5417  DONE=0xDEAD  feedExit=0xF00D\n");
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(occ); FreeGpu(isa);
}

// ===== MAD-305 DSWS adaptive wave-role controller (SPEC_DSWS_CONTROLLER.md / PLAN_DSWS_CONTROLLER.md).
// v1 = static 3-role substrate {compute / A-feed / B-feed}; the controller layers on in Phases 2-4.
// Config is env-driven so build_dsws.sh / supervised runs stay parameterizable. =====
struct DswsCfg {
    uint32_t nComp, nAfeed, nBfeed;   // role partition; N = nComp+nAfeed+nBfeed (wave count never changes)
    uint32_t ringd;                   // A-ring / B-ring depth
    uint32_t low, high;               // watermark band: occ<low = starved, occ>high = over-served
    uint32_t epochShift;              // decision cadence: E = (segments_processed >> epochShift)
    bool     dyn;                     // arm s_alloc_vgpr dyn-VGPR (COMPUTE_PGM_RSRC2 bit6)
    uint32_t N() const { return nComp + nAfeed + nBfeed; }
};
static DswsCfg parse_dsws_cfg() {
    DswsCfg c;
    c.nComp      = getenv("DSWS_NCOMP")      ? (uint32_t)atoi(getenv("DSWS_NCOMP"))      : 4u;
    c.nAfeed     = getenv("DSWS_NAFEED")     ? (uint32_t)atoi(getenv("DSWS_NAFEED"))     : 2u;
    c.nBfeed     = getenv("DSWS_NBFEED")     ? (uint32_t)atoi(getenv("DSWS_NBFEED"))     : 2u;
    c.ringd      = getenv("DSWS_RINGD")      ? (uint32_t)atoi(getenv("DSWS_RINGD"))      : 2u;
    c.low        = getenv("DSWS_LOW")        ? (uint32_t)atoi(getenv("DSWS_LOW"))        : 1u;
    c.high       = getenv("DSWS_HIGH")       ? (uint32_t)atoi(getenv("DSWS_HIGH"))       : (c.ringd > 1u ? c.ringd - 1u : 1u);
    c.epochShift = getenv("DSWS_EPOCHSHIFT") ? (uint32_t)atoi(getenv("DSWS_EPOCHSHIFT")) : 3u;
    c.dyn        = getenv("DSWS_DYN")        ? atoi(getenv("DSWS_DYN")) != 0             : false;
    return c;
}

// FIX 3(m): a positional non-flag argv token (e.g. "4c2a2b" after --dsws2/--dsws) used to be silently
//   ignored by the argv loop below -- nothing ever read argv[i] once it failed every `--foo` strcmp. A user
//   passing a role-mix that disagreed with the actual (env-derived) NCOMP/NAFEED/NBFEED got the WRONG
//   config with zero warning. Captured here so the DSWS2 mode handler can validate it instead.
static char g_posMixArg[64] = {0};

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered: if a raw-PM4 run hangs and gets SIGKILL'd, the log still shows WHERE it died
    enum { CORRECT, PRONG1, PRONG2, PRONG3, COMBINED, TIMERCHECK, PROBE, MICROBATCH, MBGEMM, MBSAT, DYNFAT1, MBPROF, MERGE, WGGEMM, SGPRPROBE, WGLDS, LDSBOUND, WGGEMM2, WGPERF, WG2X2, NFUNROLL, NFOCC, NFBF, BANDSWP, FEEDPIPE, FEEDLADDER, FEEDBTR, FEEDPROF, FEEDSTAG, FEEDPB, STACK, BW, BASELINES, SUSTAIN, KWIN, KWINORACLE, TILEPROBE, BLDSPROBE, BTR128, ANOLDS, ANOLDSTR, WAVESWEEP, OCCSWEEP, REUSE82, REUSE82TW2, REUSE82KW2, VGPR82, BLDS82, BPF82, SP82, ALD82, WALL82, TW8, TW4LEAN, B128MODE, TILEORDMODE, FP8EDGE, LDSTRIMMODE, VGPRPROBE, LEAN, DECOMP, WAVESPEC, MBML8, MBML8LONG, MBML8GAUNT, MBML8DYN, MBML8GATE, MBML8NF, MBML8PROF, MBML8BATCH, MBML8MATCH, MBML8COOP, DYNSMOKE, DSWS, DSWS2, GRIND } mode = CORRECT;
    bool fat = false;   // --fat: include >128-VGPR shapes (require umr SQ_DYN_VGPR.BLOCK_SIZE=1, cap 256)
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--prong1"))    mode = PRONG1;
        else if (!strcmp(argv[i], "--prong2"))    mode = PRONG2;
        else if (!strcmp(argv[i], "--prong3"))    mode = PRONG3;
        else if (!strcmp(argv[i], "--combined"))  mode = COMBINED;
        else if (!strcmp(argv[i], "--timercheck"))mode = TIMERCHECK;
        else if (!strcmp(argv[i], "--probe"))     mode = PROBE;
        else if (!strcmp(argv[i], "--microbatch"))mode = MICROBATCH;
        else if (!strcmp(argv[i], "--mbgemm"))    mode = MBGEMM;
        else if (!strcmp(argv[i], "--mbml8"))     mode = MBML8;
        else if (!strcmp(argv[i], "--mbml8long")) mode = MBML8LONG;
        else if (!strcmp(argv[i], "--mbml8gaunt")) mode = MBML8GAUNT;
        else if (!strcmp(argv[i], "--mbml8dyn"))  mode = MBML8DYN;
        else if (!strcmp(argv[i], "--mbml8nf"))   mode = MBML8NF;
        else if (!strcmp(argv[i], "--mbml8prof")) mode = MBML8PROF;
        else if (!strcmp(argv[i], "--mbml8batch")) mode = MBML8BATCH;
        else if (!strcmp(argv[i], "--mbml8match")) mode = MBML8MATCH;
        else if (!strcmp(argv[i], "--mbml8coop")) mode = MBML8COOP;
        else if (!strcmp(argv[i], "--dynsmoke"))  mode = DYNSMOKE;
        else if (!strcmp(argv[i], "--dsws"))      mode = DSWS;
        else if (!strcmp(argv[i], "--dsws2"))     mode = DSWS2;
        else if (!strcmp(argv[i], "--grind"))     mode = GRIND;
        else if (!strcmp(argv[i], "--mbml8gate")) mode = MBML8GATE;
        else if (!strcmp(argv[i], "--mbsat"))     mode = MBSAT;
        else if (!strcmp(argv[i], "--dynfat1"))   mode = DYNFAT1;
        else if (!strcmp(argv[i], "--mbprof"))    mode = MBPROF;
        else if (!strcmp(argv[i], "--merge"))     mode = MERGE;
        else if (!strcmp(argv[i], "--wggemm-smoke")) mode = WGGEMM;
        else if (!strcmp(argv[i], "--sgpr-probe"))   mode = SGPRPROBE;
        else if (!strcmp(argv[i], "--wglds-smoke"))  mode = WGLDS;
        else if (!strcmp(argv[i], "--lds-bound"))    mode = LDSBOUND;
        else if (!strcmp(argv[i], "--wggemm-compute")) mode = WGGEMM2;
        else if (!strcmp(argv[i], "--wggemm-perf"))  mode = WGPERF;
        else if (!strcmp(argv[i], "--wavespec"))     mode = WAVESPEC;
        else if (!strcmp(argv[i], "--wggemm-2x2"))   mode = WG2X2;
        else if (!strcmp(argv[i], "--nofeed-unroll")) mode = NFUNROLL;
        else if (!strcmp(argv[i], "--nofeed-occ"))    mode = NFOCC;
        else if (!strcmp(argv[i], "--nofeed-bf"))     mode = NFBF;
        else if (!strcmp(argv[i], "--band-sweep"))    mode = BANDSWP;
        else if (!strcmp(argv[i], "--feedpipe"))      mode = FEEDPIPE;
        else if (!strcmp(argv[i], "--feedladder"))    mode = FEEDLADDER;
        else if (!strcmp(argv[i], "--feedbtr"))       mode = FEEDBTR;
        else if (!strcmp(argv[i], "--feedprof"))      mode = FEEDPROF;
        else if (!strcmp(argv[i], "--feedstag"))      mode = FEEDSTAG;
        else if (!strcmp(argv[i], "--feedpb"))        mode = FEEDPB;
        else if (!strcmp(argv[i], "--stack"))         mode = STACK;
        else if (!strcmp(argv[i], "--bw"))            mode = BW;
        else if (!strcmp(argv[i], "--gl2c"))          g_gl2c = true;   // bracket dispatch w/ in-ring GL2C perfcounters
        else if (!strcmp(argv[i], "--baselines"))     mode = BASELINES;
        else if (!strcmp(argv[i], "--sustain"))       mode = SUSTAIN;
        else if (!strcmp(argv[i], "--kwin"))          mode = KWIN;
        else if (!strcmp(argv[i], "--kwin-oracle"))   mode = KWINORACLE;
        else if (!strcmp(argv[i], "--tileprobe"))     mode = TILEPROBE;
        else if (!strcmp(argv[i], "--bldsprobe"))     mode = BLDSPROBE;
        else if (!strcmp(argv[i], "--btr128"))        mode = BTR128;
        else if (!strcmp(argv[i], "--anolds"))        mode = ANOLDS;
        else if (!strcmp(argv[i], "--anoldstr"))      mode = ANOLDSTR;
        else if (!strcmp(argv[i], "--wavesweep"))     mode = WAVESWEEP;
        else if (!strcmp(argv[i], "--occsweep"))      mode = OCCSWEEP;
        else if (!strcmp(argv[i], "--reuse82"))       mode = REUSE82;
        else if (!strcmp(argv[i], "--reuse82tw2"))    mode = REUSE82TW2;
        else if (!strcmp(argv[i], "--reuse82kw2"))    mode = REUSE82KW2;
        else if (!strcmp(argv[i], "--vgpr82"))        mode = VGPR82;
        else if (!strcmp(argv[i], "--blds82"))        mode = BLDS82;
        else if (!strcmp(argv[i], "--bpf82"))         mode = BPF82;
        else if (!strcmp(argv[i], "--sp82"))          mode = SP82;
        else if (!strcmp(argv[i], "--ald82"))         mode = ALD82;
        else if (!strcmp(argv[i], "--wall82"))        mode = WALL82;
        else if (!strcmp(argv[i], "--tw8"))           mode = TW8;
        else if (!strcmp(argv[i], "--tw4lean"))       mode = TW4LEAN;
        else if (!strcmp(argv[i], "--b128"))          mode = B128MODE;
        else if (!strcmp(argv[i], "--tileord"))       mode = TILEORDMODE;
        else if (!strcmp(argv[i], "--fp8edge"))       mode = FP8EDGE;
        else if (!strcmp(argv[i], "--ldstrim"))       mode = LDSTRIMMODE;
        else if (!strcmp(argv[i], "--vgprprobe"))     mode = VGPRPROBE;
        else if (!strcmp(argv[i], "--lean"))          mode = LEAN;
        else if (!strcmp(argv[i], "--decomp"))        mode = DECOMP;
        else if (!strcmp(argv[i], "--fat"))       fat = true;
        else if (argv[i][0] != '-') {
            // FIX 3(m): a non-flag positional token (e.g. a "4c2a2b" role-mix string) -- capture it for
            // DSWS2's validation instead of silently dropping it.
            snprintf(g_posMixArg, sizeof g_posMixArg, "%s", argv[i]);
        }
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
    // Self-validating loop check: after KDEPTH iterations acc0 = KDEPTH * (A.B) = KDEPTH*Dref.
    // Confirms the timed run actually executed the right loop count (catches a stale/garbled
    // loop counter -- the exact bug a memory-loaded KDEPTH caused). Loose tol for f32 accum.
    auto loop_ok = [&](const float* D, uint32_t K) {
        for (int i = 0; i < 256; ++i) {
            float want = (float)K * Dref[i];
            if (std::fabs(D[i] - want) > 5e-3f * std::fabs(want) + 1e-2f) return false;
        }
        return true;
    };

    CHECK(hsaKmtOpenKFD());
    uint32_t node = FindGfx1201Node();

    // Derive the GPU clock-counter frequency (ticks/s) to convert in-kernel Dmax wall-cycles
    // to seconds. Two samples over a 200 ms host interval. (Relative throughput maxlive/Dmax is
    // frequency-independent and is what the GREEN gate needs; freq only sets the absolute TFLOPS.)
    double freq_hz = 1e8;
    {
        HsaClockCounters c0, c1;
        if (hsaKmtGetClockCounters(node, &c0) == HSAKMT_STATUS_SUCCESS) {
            double h0 = now_s();
            struct timespec nap = {0, 200000000}; nanosleep(&nap, nullptr);
            hsaKmtGetClockCounters(node, &c1);
            double h1 = now_s();
            double dg = (double)(c1.GPUClockCounter - c0.GPUClockCounter);
            if (dg > 0 && h1 > h0) freq_hz = dg / (h1 - h0);
        }
        printf("  GPU clock-counter freq ~= %.2f MHz (Dmax cycles -> seconds)\n", freq_hz / 1e6);
    }

    int rc = 0;
    if (mode == PRONG1) {
        // LIGHT kernel (NACC=8): occupancy -> throughput. Reservation floor = 80 (kernel usage).
        // grid 65536 so occupancy is VGPR-bound (not launch-rate-bound at grid/32); KDEPTH 2048
        // keeps each dispatch well under the 10 s fence timeout.
        const char* BIN = "occ_n8_d0.bin"; const uint32_t NACC = 8, nWG = 16384, K = 2048;
        printf("=== PRONG 1: occupancy -> throughput (LIGHT NACC=8, grid=%u, KDEPTH=%u, static) ===\n", nWG, K);
        Timed corr = run_timed(node, BIN, false, fragIn, nWG, 128/8, 1, 1);
        if (!corr.ok) { fprintf(stderr, "prong1 correctness did not complete.\n"); hsaKmtCloseKFD(); return 3; }
        bool ok = wmma_ok(corr.D);
        printf("  correctness (KDEPTH=1, reserve=128): WMMA %s  maxlive=%u\n", ok ? "OK" : "MISMATCH", corr.maxlive);
        if (!ok) { fprintf(stderr, "prong1 WMMA mismatch; aborting.\n"); hsaKmtCloseKFD(); return 4; }
        // reserve 256 (occ 5) is dropped: it hits the known Phase-2 "256-VGPR slow-retire >10s
        // on the raw-PM4 path" pathology (benign, not a hang, but it aborts the sweep).
        const uint32_t reserves[] = {96, 112, 128, 160, 192};   // floor = light-kernel usage (96 VGPR)
        printf("  (requested grid nWG=%u)\n", nWG);
        printf("\n  reserveVGPR  maxlive  occ/SIMD  launched  span_ms  TFLOPS  %%of307  loop\n");
        for (uint32_t rv : reserves) {
            Timed tk = run_timed(node, BIN, false, fragIn, nWG, rv/8, K, 3);
            if (!tk.ok) { fprintf(stderr, "  reserve=%u did not complete; aborting prong1.\n", rv); rc = 3; break; }
            uint32_t W = tk.total ? tk.total : nWG;          // use ACTUAL launched waves for work
            double tf = tf_span(W, K, NACC, tk.wall, freq_hz);
            double sms = (double)tk.wall / freq_hz * 1e3;
            bool lk = loop_ok(tk.D, K);
            printf("  %8u     %5u    %6.2f  %8u  %7.3f  %6.1f  %5.1f   %s\n",
                   rv, tk.maxlive, tk.maxlive / 128.0, tk.total, sms, tf, 100.0*tf/307.0, lk ? "OK" : "BAD");
        }
        printf("\n  GREEN-1 if TFLOPS rises materially toward higher occupancy (lower reserve).\n");
    } else if (mode == PRONG2) {
        // HEAVY kernel (NACC=16): static (reserve 144) vs dyn (lean 32 -> s_alloc 144).
        // nWG = WAVES now (dispatch multiplies by 32); 8192 gives several fills for steady state.
        const uint32_t NACC = 16, nWG = 8192, SFIELD = 160/8;   // match dyn's s_alloc 160 (apples-to-apples)
        const uint32_t Ks[] = {256, 1024, 4096};
        printf("=== PRONG 2: dyn vs static over a long fat phase (HEAVY NACC=16, grid=%u) ===\n", nWG);
        Timed cs = run_timed(node, "occ_n16_d0.bin", false, fragIn, nWG, SFIELD, 1, 1);
        Timed cd = run_timed(node, "occ_n16_d1.bin", true,  fragIn, nWG, 4,      1, 1);
        if (!cs.ok || !cd.ok) { fprintf(stderr, "prong2 correctness did not complete.\n"); hsaKmtCloseKFD(); return 3; }
        bool sok = wmma_ok(cs.D), dok = wmma_ok(cd.D);
        printf("  correctness (KDEPTH=1): static WMMA %s (maxlive=%u)  dyn WMMA %s (maxlive=%u)\n",
               sok ? "OK" : "MISMATCH", cs.maxlive, dok ? "OK" : "MISMATCH", cd.maxlive);
        if (!sok || !dok) {
            printf("  DUMP[0..7]  oracle:");  for (int i=0;i<8;i++) printf(" %.2f", Dref[i]);
            printf("\n              static:"); for (int i=0;i<8;i++) printf(" %.2f", cs.D[i]);
            printf("\n              dyn   :"); for (int i=0;i<8;i++) printf(" %.2f", cd.D[i]);
            printf("\n");
            fprintf(stderr, "prong2 WMMA mismatch; aborting.\n"); hsaKmtCloseKFD(); return 4;
        }
        printf("\n  KDEPTH  static_TF static_occ   dyn_TF dyn_occ   dyn/static  loop(s/d)\n");
        for (uint32_t K : Ks) {
            Timed sk = run_timed(node, "occ_n16_d0.bin", false, fragIn, nWG, SFIELD, K, 2);
            Timed dk = run_timed(node, "occ_n16_d1.bin", true,  fragIn, nWG, 4,      K, 2);
            if (!sk.ok || !dk.ok) { fprintf(stderr, "  KDEPTH=%u did not complete; aborting prong2.\n", K); rc = 3; break; }
            double stf = tf_span(sk.total ? sk.total : nWG, K, NACC, sk.wall, freq_hz);
            double dtf = tf_span(dk.total ? dk.total : nWG, K, NACC, dk.wall, freq_hz);
            printf("  %6u  %8.1f   %6.2f   %7.1f  %5.2f    %6.2fx     %s/%s\n",
                   K, stf, sk.maxlive / 128.0, dtf, dk.maxlive / 128.0,
                   stf > 0 ? dtf / stf : 0.0, loop_ok(sk.D, K) ? "OK" : "BAD", loop_ok(dk.D, K) ? "OK" : "BAD");
        }
        printf("\n  GREEN-2 if dyn >= static TFLOPS at realistic KDEPTH (no serialization penalty).\n");
    } else if (mode == PRONG3) {
        // THE wide-feed x dyn-VGPR test (MAD-305 #287): does dyn-VGPR occupancy convert to
        // throughput for a MATRIX kernel WITH an operand-feed gap (the wide-feed lever's essence,
        // FEED=1: re-fetch B each iter)? Static reserves its footprint for life (occupancy-capped);
        // dyn launches lean -> s_alloc -> full occ. If occupancy hides the feed gap, dyn > static.
        // NACC=12 (128 VGPR) fits the current dyn cap; NACC=16 (160 = the real acc[4][4]) needs
        // SQ_DYN_VGPR.BLOCK_SIZE=1 (will show loop=BAD until the cap is lifted).
        struct Cfg { uint32_t nacc, sfield; const char* fed0; const char* fed1; };
        const Cfg cfgs[] = {
            {12, 128/8, "occ_n12fed_d0.bin", "occ_n12fed_d1.bin"},   // SAFE: 128 VGPR, no umr
            // {16,160/8,"occ_n16fed_d0.bin","occ_n16fed_d1.bin"},   // NACC=16=160 VGPR over the 128 dyn cap (umr) -- DISABLED (brick-safety)
        };
        const uint32_t nWG = 8192;
        const uint32_t Ks[] = {256, 1024, 4096};
        for (const Cfg& c : cfgs) {
            printf("\n=== PRONG 3: wide-feed gap x dyn-VGPR (NACC=%u, %u VGPR, grid=%u) ===\n",
                   c.nacc, c.sfield*8, nWG);
            Timed cs = run_timed(node, c.fed0, false, fragIn, nWG, c.sfield, 1, 1);
            Timed cd = run_timed(node, c.fed1, true,  fragIn, nWG, 4,        1, 1);
            if (!cs.ok || !cd.ok) { fprintf(stderr, "  NACC=%u correctness did not complete; skipping.\n", c.nacc); continue; }
            bool sok = wmma_ok(cs.D), dok = wmma_ok(cd.D);
            printf("  correctness (KDEPTH=1): static %s (occ=%.1f)  dyn %s (occ=%.1f)%s\n",
                   sok?"OK":"MISMATCH", cs.maxlive/128.0, dok?"OK":"MISMATCH", cd.maxlive/128.0,
                   dok?"":"  <- dyn over cap; flip SQ_DYN_VGPR.BLOCK_SIZE=1");
            printf("  KDEPTH  static_TF static_occ   dyn_TF dyn_occ   dyn/static  loop(s/d)\n");
            for (uint32_t K : Ks) {
                Timed sk = run_timed(node, c.fed0, false, fragIn, nWG, c.sfield, K, 3);
                Timed dk = run_timed(node, c.fed1, true,  fragIn, nWG, 4,        K, 3);
                if (!sk.ok || !dk.ok) { fprintf(stderr, "  NACC=%u KDEPTH=%u did not complete.\n", c.nacc, K); break; }
                double stf = tf_span(sk.total?sk.total:nWG, K, c.nacc, sk.wall, freq_hz);
                double dtf = tf_span(dk.total?dk.total:nWG, K, c.nacc, dk.wall, freq_hz);
                printf("  %6u  %8.1f   %6.2f   %7.1f  %5.2f    %6.2fx     %s/%s\n",
                       K, stf, sk.maxlive/128.0, dtf, dk.maxlive/128.0,
                       stf>0?dtf/stf:0.0, loop_ok(sk.D,K)?"OK":"BAD", loop_ok(dk.D,K)?"OK":"BAD");
            }
        }
        printf("\n  GREEN-3 if FED dyn/static > 1 (occupancy hides the feed gap), loops OK.\n");
    } else if (mode == COMBINED) {
        // THE combined kernel (MAD-305 #287): unroll x ILP(NACC) x feed x dyn-VGPR, all stacked.
        // s6 = TRIP COUNT; each trip does UNROLL accumulate-rounds; effK = 1 + trip*UNROLL.
        // Rows: NACC=8 no-feed (the operands-in-register ceiling, ~the 307 microbench), NACC=8 fed,
        // NACC=16 fed (the real acc[4][4] tile; dyn arm valid only after SQ_DYN_VGPR.BLOCK_SIZE=1).
        const uint32_t UNROLL = 8, nWG = 8192;
        const uint32_t trips[] = {64, 256, 512};
        printf("=== COMBINED: unroll(%u) x ILP x feed x dyn-VGPR  (grid=%u, ceiling=307 TF) ===\n", UNROLL, nWG);
        struct Cfg { const char* name; uint32_t nacc, sfield; const char* d0; const char* d1; };
        const Cfg cfgs[] = {
            {"NACC=8  no-feed (ceiling)", 8,  96/8,  "occ_combnf_n8_d0.bin", "occ_combnf_n8_d1.bin"},
            {"NACC=8  fed",              8,  96/8,  "occ_comb_n8_d0.bin",   "occ_comb_n8_d1.bin"},
            {"NACC=16 fed (real tile)",  16, 160/8, "occ_comb_n16_d0.bin",  "occ_comb_n16_d1.bin"},
        };
        for (const Cfg& c : cfgs) {
            printf("\n--- %s ---\n", c.name);
            Timed cs = run_timed(node, c.d0, false, fragIn, nWG, c.sfield, 0, 1);   // trip=0 -> peel only
            Timed cd = run_timed(node, c.d1, true,  fragIn, nWG, 4,        0, 1);
            if (!cs.ok || !cd.ok) { fprintf(stderr, "  %s correctness did not complete; skipping.\n", c.name); continue; }
            printf("  correctness (trip=0): static %s (occ=%.1f)  dyn %s (occ=%.1f)%s\n",
                   wmma_ok(cs.D)?"OK":"MISMATCH", cs.maxlive/128.0,
                   wmma_ok(cd.D)?"OK":"MISMATCH", cd.maxlive/128.0,
                   wmma_ok(cd.D)?"":"  <- dyn over cap (flip SQ_DYN_VGPR.BLOCK_SIZE=1)");
            printf("  trip  effK   static_TF s_occ %%307   dyn_TF d_occ %%307  dyn/st  loop(s/d)\n");
            for (uint32_t trip : trips) {
                uint32_t effK = 1 + trip*UNROLL;
                Timed sk = run_timed(node, c.d0, false, fragIn, nWG, c.sfield, trip, 3);
                Timed dk = run_timed(node, c.d1, true,  fragIn, nWG, 4,        trip, 3);
                if (!sk.ok || !dk.ok) { fprintf(stderr, "  %s trip=%u did not complete.\n", c.name, trip); break; }
                double stf = tf_span(sk.total?sk.total:nWG, effK, c.nacc, sk.wall, freq_hz);
                double dtf = tf_span(dk.total?dk.total:nWG, effK, c.nacc, dk.wall, freq_hz);
                printf("  %4u %6u   %7.1f %5.2f %5.1f  %7.1f %5.2f %5.1f  %5.2fx  %s/%s\n",
                       trip, effK, stf, sk.maxlive/128.0, 100*stf/307.0,
                       dtf, dk.maxlive/128.0, 100*dtf/307.0,
                       stf>0?dtf/stf:0.0, loop_ok(sk.D,effK)?"OK":"BAD", loop_ok(dk.D,effK)?"OK":"BAD");
            }
        }
        printf("\n  static %%307 = how close unroll+ILP gets to the matrix ceiling; dyn/static = the\n");
        printf("  dyn-VGPR occupancy contribution (dyn valid <=128 VGPR until SQ_DYN_VGPR.BLOCK_SIZE=1).\n");
    } else if (mode == TIMERCHECK) {
        // Resolve the s_sendmsg REALTIME tick rate. Busy-wait T ticks at grid=1; host-time it.
        // Two targets cancel the fixed submit/fence/poll overhead: freq = dT / d(host_secs).
        printf("=== TIMER CHECK: actual s_sendmsg REALTIME tick rate ===\n");
        printf("  assumed freq_hz (hsaKmt GPUClockCounter) = %.2f MHz ; rocprof shader clock ~= 2340 MHz\n",
               freq_hz/1e6);
        const uint32_t targets[] = {40000000u, 80000000u, 120000000u};
        double secs[3] = {0,0,0}; uint64_t span[3] = {0,0,0}; bool ok = true;
        printf("\n  target_ticks   host_secs   realtime_span(ticks)   span/host(MHz)\n");
        for (int i = 0; i < 3; ++i) {
            RunResult rr = run_variant(node, "occ_timercheck.bin", false, fragIn, 1, 8, targets[i]);
            if (!rr.ok) { fprintf(stderr, "  timercheck target=%u did not complete.\n", targets[i]); ok = false; rc = 3; break; }
            secs[i] = rr.secs; span[i] = rr.wall;
            printf("  %11u   %8.4f   %18llu   %8.1f\n",
                   targets[i], rr.secs, (unsigned long long)rr.wall, (double)rr.wall/rr.secs/1e6);
        }
        if (ok) {
            double f01 = (double)(targets[1]-targets[0])/(secs[1]-secs[0]);
            double f12 = (double)(targets[2]-targets[1])/(secs[2]-secs[1]);
            double f   = 0.5*(f01+f12);
            printf("\n  overhead-cancelled REALTIME freq (dticks / d(host_secs)):  %.1f MHz  (%.1f / %.1f)\n",
                   f/1e6, f01/1e6, f12/1e6);
            printf("  -> PM4-timed TFLOPS must be scaled by (actual/assumed) = %.2fx.\n", f/freq_hz);
            printf("     e.g. combined no-feed NACC=8 reported 64 TF -> really ~%.0f TF;\n", 64.0*f/freq_hz);
            printf("          combined NACC=16 reported 127 TF -> really ~%.0f TF.\n", 127.0*f/freq_hz);
        }
    } else if (mode == PROBE) {
        // Single wave -- no occupancy/grid confound. span_ms (in-kernel, min-start..max-end of the
        // ONE wave) is its exact loop wall-time, free of host/launch overhead. ns/WMMA below the
        // ~2.7ns issue floor (@2.5GHz) would mean the WMMAs are not really executing.
        const char* BIN = "occ_n8_d0.bin"; const uint32_t NACC = 8;
        const uint32_t Ks[] = {2000, 4000, 8000, 16000};
        printf("=== PROBE: single wave (nWG=1, NACC=8, reserve=128) raw per-WMMA cost ===\n");
        printf("  KDEPTH   host_ms   span_ms   span_ns/iter  span_ns/WMMA  loop\n");
        for (uint32_t K : Ks) {
            Timed t = run_timed(node, BIN, false, fragIn, 1, 128/8, K, 3);
            if (!t.ok) { fprintf(stderr, "  probe KDEPTH=%u did not complete.\n", K); rc = 3; break; }
            double hms = t.secs * 1e3, sms = (double)t.wall / freq_hz * 1e3;
            double ns_iter = (double)t.wall / freq_hz / (double)(K - 1) * 1e9;
            double ns_wmma = ns_iter / NACC;
            printf("  %6u  %8.2f  %8.4f  %12.3f  %12.4f  %s\n",
                   K, hms, sms, ns_iter, ns_wmma, loop_ok(t.D, K) ? "OK" : "BAD");
        }
        printf("  (physical issue floor ~2.7 ns/WMMA @ 2.5 GHz; below that => WMMAs elided/not real.)\n");
    } else if (mode == MICROBATCH) {
        // SATURATION SWEEP: hold occupancy fixed, crank work-per-tile (KDEPTH). If TF rises with
        // KDEPTH we were UNDER-FEEDING (per-tile s_alloc/atomic/reload/store overhead dominated and
        // the matrix unit idled between tiles); the asymptote is this hand-asm vehicle's saturated
        // rate (~64 TF, the load-once occ_kernel ceiling). If flat-low -> stall-bound regardless.
        const uint32_t NACC=8;
        // PART 1: push work/tile to find the saturated asymptote (occ 8, clean). 307 = matrix ceiling.
        printf("=== PART 1: saturation asymptote (pool=1024 = occ 8, %u tiles) ===\n", 2048);
        printf("  KDEPTH  variant  okTiles    span_ms  TFLOPS  %%307\n");
        { const uint32_t Ks[]={65536, 131072, 262144, 524288, 1048576}, TOTAL=2048, POOL=1024;  // push past 65536: was 276 the ceiling or still climbing?
          for (uint32_t K : Ks) for (int dv=0; dv<2; ++dv) {
              MbResult r=run_microbatch(node,dv?"occ_mb_d1.bin":"occ_mb_d0.bin",dv!=0,fragIn,POOL,96/8,K,TOTAL,Dref);
              if(!r.ok){ fprintf(stderr,"  K=%u %s incomplete.\n",K,dv?"dyn":"static"); rc=3; continue; }
              double tf=(double)TOTAL*K*NACC*(2.0*16*16*16)*freq_hz/(double)r.wall/1e12;
              printf("  %6u  %-6s   %4u/%u  %8.3f  %6.1f  %5.1f\n",
                     K,dv?"dyn":"static",r.okTiles,TOTAL,(double)r.wall/freq_hz*1e3,tf,100*tf/307.0); } }
        // PART 2: the real dyn-VGPR throughput test, finally on a SATURATED vehicle:
        // static-96 caps at occ 12; dyn lean->s_alloc reaches occ 16. At high KDEPTH (saturated),
        // does the extra 33% occupancy now convert to throughput? (pool 2048 >> ... so all waves busy.)
        printf("\n=== PART 2: occupancy-at-saturation -- static(occ12) vs dyn(occ16), KDEPTH=16384 ===\n");
        printf("  pool   variant  maxlive occ/SIMD  okTiles    span_ms  TFLOPS  %%307\n");
        { const uint32_t K=16384, TOTAL=4096;
          for (int dv=0; dv<2; ++dv) {
              const uint32_t POOL = dv ? 2560u : 1536u;   // dyn:2560 -> occ16 (above the 2048 exact-fill race); static:1536 -> occ12 (its 96-VGPR reserve cap). NOW the occ16-vs-occ12 contrast is real.
              MbResult r=run_microbatch(node,dv?"occ_mb_d1.bin":"occ_mb_d0.bin",dv!=0,fragIn,POOL,96/8,K,TOTAL,Dref);
              if(!r.ok){ fprintf(stderr,"  %s incomplete.\n",dv?"dyn":"static"); rc=3; continue; }
              double tf=(double)TOTAL*K*NACC*(2.0*16*16*16)*freq_hz/(double)r.wall/1e12;
              printf("  %5u  %-6s   %6u  %6.2f   %4u/%u  %8.3f  %6.1f  %5.1f\n",
                     POOL,dv?"dyn":"static",r.maxlive,r.maxlive/128.0,r.okTiles,TOTAL,
                     (double)r.wall/freq_hz*1e3,tf,100*tf/307.0); } }
        printf("\n  PART1: does TF reach ~300 (hand-asm CAN saturate)?  PART2: dyn TF > static TF =>\n");
        printf("  dyn-VGPR occupancy converts to throughput when saturated (the lever finally pays).\n");
    } else if (mode == MBGEMM) {
        // COMPUTE:LOAD (reuse) SWEEP -- the REAL lever. reuse = FM*FN/(FM+FN) = WMMAs per operand
        // load; only a bigger TILE raises it (K can't). The 128-VGPR "cap" only binds the DYNAMIC
        // s_alloc path; STATIC reservation reaches the 256-VGPR wave max with NO umr, and static
        // can't dyn-deadlock -> 4x4(192)/5x4(240) run right now. Cross the barrier we never crossed.
        struct T { int fm, fn, M, N; bool dyn; const char* tag; bool needUmr; };
        T tiles[] = {
            {1,1, 2048,2048, true,  "1x1", false},             // reuse 0.50, 128-cap dyn (ref)
            {2,2, 2048,2048, true,  "2x2", false},             // reuse 1.00
            {2,4, 2048,2048, true,  "2x4", false},             // reuse 1.33, the dyn ceiling
            {4,4, 2048,2048, false, "4x4", false},             // reuse 2.00, 192 VGPR STATIC (no umr)
            {5,4, 2560,2048, false, "5x4", false},             // reuse 2.22, 240 VGPR STATIC
            {4,4, 2048,2048, true,  "4x4", true },             // DYN 192 -- big tile + lean occ (THE thesis); umr
            {5,4, 2560,2048, true,  "5x4", true },             // DYN 240 -- umr
        };
        const int batch = 32, K = 2048;
        printf("=== COMPUTE:LOAD (reuse) SWEEP -- bigger tile = more WMMAs/load. K=%d batch=%d ===\n", K, batch);
        printf("  static (d0): big tile, low occ, no umr. dyn (d1, --fat): big tile + lean occ, needs umr.\n");
        printf("  tile  reuse  vgpr  mode    oracle    span_ms  TFLOPS  %%307  x143  pool  maxlive  claims\n");
        for (auto& t : tiles) {
            if (fat ? (t.fm < 4) : t.needUmr) continue;        // --fat: big tiles only (static+dyn head-to-head)
            int fatregs = (32 + t.fm*t.fn*8 + t.fm*4 + t.fn*4 + 15) & ~15;
            uint32_t pool;
            if (t.dyn) { pool = (uint32_t)((1152*128)/fatregs); if (pool>1536u) pool=1536u; pool=(pool/128u)*128u; }
            else       { pool = 768u; }                        // static: HW caps occ at file/fatregs; no deadlock
            char bn[80]; snprintf(bn, sizeof bn, "occ_mbgemm_%s_b%d_d%d.bin", t.tag, batch, t.dyn?1:0);
            double reuse = (double)(t.fm*t.fn)/(t.fm+t.fn);
            MbgResult o = run_mbgemm(node, bn, t.dyn, pool, 512,512,512, t.fm,t.fn, true);     // oracle gate
            MbgResult r = run_mbgemm(node, bn, t.dyn, pool, t.M,t.N,K, t.fm,t.fn, false);      // perf
            if (!o.ok || !r.ok) { fprintf(stderr, "  %s incomplete\n", t.tag); rc=3; continue; }
            uint32_t TOTAL = (uint32_t)(t.M/(16*t.fm)) * (t.N/(16*t.fn)); int KT = K/16;
            double work = (double)TOTAL * t.fm * t.fn * KT;
            double tf = work * (2.0*16*16*16) * freq_hz / (double)r.wall / 1e12;
            printf("  %-5s %.2f  %4d  %-6s  %s%u/%u  %8.3f  %6.1f  %4.1f  %.2f  %4u  %7u  %6u\n",
                   t.tag, reuse, fatregs, t.dyn?"dyn":"static",
                   o.okTiles==o.nChecked ? "OK " : "BAD", o.okTiles, o.nChecked,
                   (double)r.wall/freq_hz*1e3, tf, 100*tf/307.0, tf/143.0, pool, r.maxlive, r.total);
        }
        // ISOLATION: same kernel, operands loaded ONCE & reused (no per-K feed) -> framework ceiling.
        if (!fat) {
            printf("\n  [NO-FEED 2x4 b32 -- operands reused, ZERO per-K load]    K     KT   TFLOPS  span_ms  maxlive  claims\n");
            const int nfKs[] = {2048, 8192, 32768};
            for (int K : nfKs) {
                MbgResult r = run_mbgemm(node, "occ_mbgemm_2x4_b32_nf.bin", true, 1152u, 2048,2048,K, 2,4, false);
                if (!r.ok) continue;
                uint32_t TOTAL = (uint32_t)(2048/32)*(2048/64); int KT = K/16;
                double work = (double)TOTAL * 8 * KT;
                double tf = work * (2.0*16*16*16) * freq_hz / (double)r.wall / 1e12;
                printf("                                                       %6d %5d  %6.1f  %.3f  %7u  %6u\n",
                       K, KT, tf, (double)r.wall/freq_hz*1e3, r.maxlive, r.total);
            }
            printf("    CLIMBS with K => per-tile FRAMEWORK overhead is the wall (amortized by work/tile), NOT the\n");
            printf("    issue port. The fed K-sweep stayed flat only because every K-step re-pays the feed.\n");
        }
        printf("\n  TF climbs with reuse -> tile size is the lever; push bigger (umr->8x8, reuse 4.0).\n");
    } else if (mode == MBSAT) {
        // SATURATION + FAT-TILE SWEEP. grabs(=TOTAL/BATCH) MUST be >> pool or the reading is meaningless
        // (BATCH=1 here; the old BATCH=32 @2048^2 gave only 64 grabs -> ~64 of 1152 waves worked).
        // TOTAL / grabs-per-pool / maxlive are STANDARD columns now. Tests: (a) PLATEAU -- 2x4/4x4 @8192
        // vs @16384; (b) FAT-TILE thesis -- does more reuse move the saturated plateau? 5x4 (reuse 2.22,
        // 240 VGPR) is the fattest single-wave tile that fits the 256-VGPR wave max. dyn 5x4 needs the umr
        // SQ_DYN_VGPR.BLOCK_SIZE=1 flip (cap 256); without it s_alloc 240 corrupts and the oracle gate
        // catches it (BAD -> perf skipped, no false number). C/A/B all VRAM (guard in run_mbgemm).
        struct SC { int fm,fn; bool dyn; const char* tag; int M,N; uint32_t pool; double reuse; };
        SC cfgs[] = {
            {2,4,false,"2x4", 8192, 8192,  768u, 1.33},   // 2x4 STATIC control (same-tile dyn isolation)
            {2,4,false,"2x4",16384,16384,  768u, 1.33},   // 2x4 STATIC plateau
            {2,4,true, "2x4", 8192, 8192, 1152u, 1.33},   // 2x4 DYN (lean->grow 128, SAFE no-umr) -- the lever
            {2,4,true, "2x4",16384,16384,1152u, 1.33},   // 2x4 DYN plateau
            {4,4,false,"4x4", 8192, 8192,  768u, 2.00},   // saturated ref (static)
            {4,4,false,"4x4",16384,16384,  768u, 2.00},   // plateau point
            {5,4,false,"5x4",10240, 8192,  768u, 2.22},   // fattest single-wave, STATIC (low occ)
            {4,4,true, "4x4", 8192, 8192, 1152u, 2.00},   // DYN same-tile head-to-head (--fat; umr cap256)
            {5,4,true, "5x4",10240, 8192, 1152u, 2.22},   // fattest single-wave, DYN (--fat; umr cap256)
        };
        const int Ks = 8192;
        printf("=== MBGEMM SATURATION + FAT-TILE SWEEP (BATCH=1, K=%d, fed/VRAM%s) ===\n",
               Ks, fat?", --fat: dyn tiles need umr SQ_DYN_VGPR.BLOCK_SIZE=1":"");
        printf("  tile reuse mode      M     N    TOTAL  grabs/pool  maxlive  oracle       TF   %%307  span_ms\n");
        for (auto& c : cfgs) {
            int satFat = (32 + c.fm*c.fn*8 + c.fm*4 + c.fn*4 + 15) & ~15;    // dyn grow target
            if (c.dyn && satFat > 128 && !fat) continue;                     // only umr-needing dyn (>128) gated behind --fat; safe dyn runs
            char bn[80]; snprintf(bn, sizeof bn, "occ_mbgemm_%s_b1_d%d.bin", c.tag, c.dyn?1:0);
            int Mo=16*c.fm*4, No=16*c.fn*4;                                   // tile-valid oracle dims (NTL=4 pow2)
            MbgResult o = run_mbgemm(node, bn, c.dyn, c.pool, Mo,No,512, c.fm,c.fn, true);
            if (!o.ok || o.okTiles!=o.nChecked) {
                printf("  %-4s %.2f %-6s %6d %6d    --        --         --    %s%u/%u (perf skipped)\n",
                       c.tag, c.reuse, c.dyn?"dyn":"static", c.M, c.N,
                       o.ok?"BAD ":"ERR ", o.okTiles, o.nChecked);
                continue;
            }
            MbgResult r = run_mbgemm(node, bn, c.dyn, c.pool, c.M,c.N,Ks, c.fm,c.fn, false);
            if (!r.ok) { printf("  %-4s %.2f %-6s  (perf incomplete)\n", c.tag, c.reuse, c.dyn?"dyn":"static"); continue; }
            uint32_t TOTAL = (uint32_t)(c.M/(16*c.fm)) * (c.N/(16*c.fn)); int KT = Ks/16;
            double work = (double)TOTAL * c.fm * c.fn * KT;
            double tf = work * (2.0*16*16*16) * freq_hz / (double)r.wall / 1e12;
            printf("  %-4s %.2f %-6s %6d %6d  %7u  %7.1fx   %6u   OK %u/%u  %5.1f  %4.1f  %.3f\n",
                   c.tag, c.reuse, c.dyn?"dyn":"static", c.M, c.N, TOTAL,
                   (double)TOTAL/c.pool, r.maxlive, o.okTiles, o.nChecked, tf, 100*tf/307.0,
                   (double)r.wall/freq_hz*1e3);
        }
        printf("\n  PLATEAU: 2x4/4x4 @8192 vs @16384 flat => saturated ceiling reached.\n");
        printf("  FAT thesis: 4x4(2.0)->5x4(2.22) static, and 5x4 static vs dyn => does fatter tile / dyn-VGPR move it?\n");
    } else if (mode == MBML8) {
        // REAL ml8 TRAINING SHAPES, dyn-vs-static, single-wave mbgemm with GENDIV (non-pow2 NTL).
        // down-train  : M=2048 K=9216 N=2560  (NTL @FN4=40, @FN2=80)
        // gateup-train: M=2048 K=2560 N=9216  (NTL @FN4=144, @FN2=288)  -- all non-pow2 -> GENDIV bins.
        // ONE call per cfg gives BOTH a sampled oracle (okTiles/nChecked) AND the perf wall. BATCH=1 so
        // grabs >> pool (saturated). dyn arms <=128 VGPR (no umr); static 4x4 reserves 192 (no umr).
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = { {"down  ",2048,9216,2560}, {"gateup",2048,2560,9216} };
        struct MC { int fm,fn; bool dyn; const char* tag; uint32_t pool; double reuse; };
        MC cfgs[] = {
            {2,2,false,"2x2", 768u,1.00}, {2,2,true,"2x2",1152u,1.00},     // same-tile dyn-vs-static
            {2,4,false,"2x4", 768u,1.33}, {2,4,true,"2x4",1152u,1.33},     // same-tile dyn-vs-static
            {4,4,false,"4x4", 768u,2.00},                                  // reuse reference (static)
        };
        // feed=0: PREFETCHED (double-buffered, _gd); feed=1: NAIVE exposed (load->wait->compute, _ndgd).
        // The naive arm puts prong3's "occupancy is the only feed-hider" condition on the REAL ml8 GEMM.
        struct FM2 { int naive; const char* sfx; const char* label; };
        FM2 feeds[] = { {0,"gd","prefetch"}, {1,"ndgd","naive   "} };
        printf("=== REAL ml8 SHAPES -- single-wave mbgemm dyn-vs-static (GENDIV, BATCH=1, fed/VRAM) ===\n");
        for (auto& s : shapes) {
            printf("  --- %s  M=%d K=%d N=%d ---\n", s.name, s.M, s.K, s.N);
            printf("    feed      tile reuse mode    NTL   TOTAL  maxlive  oracle        TF    %%307  span_ms\n");
            for (auto& f : feeds) {
                for (auto& c : cfgs) {
                    if (f.naive && c.fm==4) continue;                          // naive arm: same-tile pairs only (2x2,2x4)
                    int satFat = (32 + c.fm*c.fn*8 + c.fm*4 + c.fn*4 + 15) & ~15;
                    if (c.dyn && satFat > 128 && !fat) continue;              // umr-needing dyn gated behind --fat
                    char bn[96]; snprintf(bn, sizeof bn, "occ_mbgemm_%s_b1_d%d_%s.bin", c.tag, c.dyn?1:0, f.sfx);
                    MbgResult r = run_mbgemm(node, bn, c.dyn, c.pool, s.M,s.N,s.K, c.fm,c.fn, /*fullCheck*/false, /*useGenDiv*/true);
                    if (!r.ok) { printf("    %-8s %-4s %.2f %-6s (incomplete/timeout)\n", f.label, c.tag, c.reuse, c.dyn?"dyn":"static"); rc=3; continue; }
                    int NTL = s.N/(16*c.fn); uint32_t TOTAL=(uint32_t)(s.M/(16*c.fm))*NTL; int KT=s.K/16;
                    double work=(double)TOTAL*c.fm*c.fn*KT;
                    double tf = work*(2.0*16*16*16)*freq_hz/(double)r.wall/1e12;
                    printf("    %-8s %-4s %.2f %-6s %5d %7u  %7u  %s%u/%u  %6.1f  %4.1f  %.3f\n",
                           f.label, c.tag, c.reuse, c.dyn?"dyn":"static", NTL, TOTAL, r.maxlive,
                           r.okTiles==r.nChecked?"OK ":"BAD", r.okTiles, r.nChecked,
                           tf, 100*tf/307.0, (double)r.wall/freq_hz*1e3);
                }
            }
        }
        printf("  prefetch=double-buffered (occ redundant); naive=exposed feed (occ is the only hider -> the prong3 condition).\n");
    } else if (mode == MBML8LONG) {
        // SUSTAINED steady-state: the real ml8 GEMM with M blown up to a big token count (128K rows),
        // each dispatch re-submitted REPS times back-to-back -> tens of seconds of continuous GPU work,
        // so the reading is steady-state (not warmup/clock-ramp/noise dominated). meanTF is the number;
        // min/max over reps shows the spread. Same-tile 2x4 dyn-vs-static on the REAL shapes.
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = { {"down  ",131072,9216,2560}, {"gateup",131072,2560,9216} };
        struct MC { int fm,fn; bool dyn; const char* tag; uint32_t pool; double reuse; };
        MC cfgs[] = {                                                    // safe tiles (<=128 VGPR dyn); 4x4 static-only
            {2,2,false,"2x2",768u,1.00}, {2,2,true,"2x2",1152u,1.00},
            {4,2,false,"4x2",768u,1.33}, {4,2,true,"4x2",1152u,1.33},
            {2,4,false,"2x4",768u,1.33}, {2,4,true,"2x4",1152u,1.33},
            {4,4,false,"4x4",768u,2.00},                                 // 192 VGPR static reference (dyn needs umr)
        };
        const uint32_t REPS=80;
        printf("=== SUSTAINED ml8 (M=131072 tokens, REPS=%u back-to-back, prefetch, GENDIV) -- full tile sweep ===\n", REPS);
        printf("    each cell runs ~12-30s of continuous GPU time; meanTF is the steady-state number.\n");
        for (auto& s : shapes) {
            printf("  --- %s  M=%d K=%d N=%d ---\n", s.name, s.M, s.K, s.N);
            printf("    tile reuse mode   GPUsec reps maxlive  oracle      meanTF  minTF  maxTF\n");
            for (auto& c : cfgs) {
                char bn[96]; snprintf(bn, sizeof bn, "occ_mbgemm_%s_b1_d%d_gd.bin", c.tag, c.dyn?1:0);
                MbgResult r = run_mbgemm(node, bn, c.dyn, c.pool, s.M,s.N,s.K, c.fm,c.fn, /*fullCheck*/false, /*useGenDiv*/true, REPS);
                if (!r.ok) { printf("    %-4s %.2f %-6s (incomplete/timeout at rep %u)\n", c.tag, c.reuse, c.dyn?"dyn":"static", r.repsDone); rc=3; continue; }
                int NTL=s.N/(16*c.fn); uint32_t TOTAL=(uint32_t)(s.M/(16*c.fm))*NTL; int KT=s.K/16;
                double wmma=(double)TOTAL*c.fm*c.fn*KT, flop=wmma*(2.0*16*16*16);
                double meanTF=flop*freq_hz/(double)r.wall/1e12;          // r.wall = mean per-rep span
                double maxTF =flop*freq_hz/(double)r.wallMin/1e12;       // fastest rep
                double minTF =flop*freq_hz/(double)r.wallMax/1e12;       // slowest rep
                double gpusec=(double)r.wallSum/freq_hz;
                printf("    %-4s %.2f %-6s %6.1f %4u  %7u  %s%u/%u  %6.1f %6.1f %6.1f\n",
                       c.tag, c.reuse, c.dyn?"dyn":"static", gpusec, r.repsDone, r.maxlive,
                       r.okTiles==r.nChecked?"OK ":"BAD", r.okTiles, r.nChecked, meanTF, minTF, maxTF);
            }
        }
        printf("  Steady-state over 30s+ -- if dyn still trails static across all tiles, the per-tile grow overhead is real, not noise.\n");
    } else if (mode == MBML8GATE) {
        // SAFE 512^3 ORACLE GATE for the COMPLETE dyn field: all 8 dyn-able tiles <=128 VGPR x
        // {gd=prefetch, dg=DEFERGROW grow-first} x BATCH{1,8,32} = 48 bins, single sub-1s dispatch
        // each, fullCheck. Catches any tile that computes wrong / page-faults BEFORE the at-scale
        // run. A failed grow can't run on unallocated VGPRs (SCC-retry guard). Grow targets disasm-
        // verified <=128 (1x1=48 .. 2x4/4x2 pf=128, dg=96).
        struct T { int fm,fn; } tiles[] = {{1,1},{1,2},{2,1},{2,2},{1,4},{4,1},{2,4},{4,2}};
        const char* feeds[] = {"gd","dg"}; int batches[] = {1,8,32};
        printf("=== SAFE 512^3 ORACLE GATE: complete dyn field, 48 bins (fullCheck, <1s each) ===\n");
        int npass=0, ntot=0;
        for (auto& t : tiles) for (auto* fd : feeds) for (int b : batches) {
            char bn[96]; snprintf(bn,sizeof bn,"occ_mbgemm_%dx%d_b%d_d1_%s.bin", t.fm,t.fn,b,fd);
            ntot++;
            MbgResult r = run_mbgemm(node, bn, true, 256u, 512,512,512, t.fm,t.fn, /*fullCheck*/true, /*useGenDiv*/true, 1, 0.0);
            if (!r.ok) { printf("  %-34s INCOMPLETE/TIMEOUT (maxlive=%u)\n", bn, r.maxlive); rc=3; continue; }
            bool ok = (r.okTiles==r.nChecked); if (ok) npass++; else rc=3;
            printf("  %-34s %s %u/%u  maxlive=%u\n", bn, ok?"PASS":"FAIL", r.okTiles, r.nChecked, r.maxlive);
        }
        printf("=== gate done: %d/%d PASS. ALL must PASS before at-scale --mbml8dyn. ===\n", npass, ntot);
    } else if (mode == MBML8DYN) {
        // THE FAIR RACE: dyn on its PROPER kernel vs static on its proper kernel, all 10 ml8 shapes.
        // dyn variants: prefetch (pf) BATCH{1,8,32} = grow-amortization track; DEFERGROW (dg) BATCH{1,8,32}
        // = lean-block frags + accumulators-only 96-VGPR grow + single-buffer occupancy-hidden feed.
        // static refs: the fat tiles that won stage 1 (2x8, 4x4). Oracle-gated per cell (dg correctness check).
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = {
            {"down   ",2048,9216,2560}, {"gate/up",2048,2560,9216}, {"attn_q ",2048,2560,4096},
            {"attn_kv",2048,2560,1024}, {"attn_o ",2048,4096,2560},
            {"down_pf",512,9216,2560},  {"gtup_pf",512,2560,9216},  {"q_pf   ",512,2560,4096},
            {"kv_pf  ",512,2560,1024},  {"o_pf   ",512,4096,2560},
        };
        // COMPLETE DYN FIELD: every dyn-able tile <=128 VGPR {1x1,1x2,2x1,2x2,1x4,4x1,2x4,4x2}
        // x {pf=prefetch double-buffer, dg=DEFERGROW grow-first single-buffer} x BATCH{1,8,32}
        // = 48 dyn configs/shape, + 2 static refs (2x8, 4x4). The only fair-and-complete dyn field.
        struct DV { int fm,fn; bool dyn; int batch; bool dg; };
        // static refs first
        DV stref[] = { {2,8,false,1,false}, {4,4,false,1,false} };
        struct T { int fm,fn; } tiles[] = {{1,1},{1,2},{2,1},{2,2},{1,4},{4,1},{2,4},{4,2}};
        bool dgs[] = {false,true}; int batches[] = {1,8,32};
        const double TGT = 12.0;
        printf("=== ml8 COMPLETE DYN FIELD: 8 dyn tiles x {pf,dg} x BATCH{1,8,32} (+static refs), per shape (~%.0fs/cell) ===\n", TGT);
        auto runcell = [&](const char* lbl, int fm, int fn, bool dyn, int batch, bool dg, const SH& s){
            if ((s.M/16)%fm || (s.N/16)%fn) { printf("  %-15s (tile doesn't divide shape)\n", lbl); return; }
            uint32_t pool = dyn?1152u:768u;
            char bn[96]; snprintf(bn,sizeof bn,"occ_mbgemm_%dx%d_b%d_d%d_%s.bin", fm,fn,batch,dyn?1:0,dg?"dg":"gd");
            MbgResult r = run_mbgemm(node,bn,dyn,pool,s.M,s.N,s.K,fm,fn,/*fullCheck*/false,/*useGenDiv*/true,1,TGT);
            if(!r.ok){ printf("  %-15s (timeout/incomplete)\n",lbl); rc=3; return; }
            int NTL=s.N/(16*fn); uint32_t TOTAL=(uint32_t)(s.M/(16*fm))*NTL; int KT=s.K/16;
            double flop=(double)TOTAL*fm*fn*KT*(2.0*16*16*16);
            double meanTF=flop*freq_hz/(double)r.wall/1e12;
            double maxTF=flop*freq_hz/(double)r.wallMin/1e12, minTF=flop*freq_hz/(double)r.wallMax/1e12;
            printf("  %-15s %5.1f %7u %s%u/%u %6.1f[%.0f..%.0f]\n",
                lbl,(double)r.wallSum/freq_hz,r.maxlive,
                r.okTiles==r.nChecked?"OK ":"BAD",r.okTiles,r.nChecked,meanTF,minTF,maxTF);
        };
        const char* onlyShape = getenv("ML8_ONLY");   // e.g. ML8_ONLY=o_pf -> run just that shape
        for (auto& s : shapes) {
            if (onlyShape && !strstr(s.name, onlyShape)) continue;
            printf("\n#### %s  M=%d K=%d N=%d ####\n", s.name, s.M, s.K, s.N);
            printf("  config            GPUs maxlive oracle    TF[min..max]\n");
            for (auto& st : stref) { char lbl[16]; snprintf(lbl,sizeof lbl,"stat %dx%d",st.fm,st.fn); runcell(lbl,st.fm,st.fn,false,1,false,s); }
            for (auto& t : tiles) for (bool dg : dgs) for (int b : batches) {
                char lbl[24]; snprintf(lbl,sizeof lbl,"dyn %dx%d %s b%d",t.fm,t.fn,dg?"dg":"pf",b);
                runcell(lbl,t.fm,t.fn,true,b,dg,s);
            }
        }
        printf("\n=== complete dyn field done. dg/dyn cells MUST be oracle-OK to count. ===\n");
    } else if (mode == MBML8NF) {
        // NO-FEED FRAMEWORK CEILING: per shape, run that shape's TOP-4 STATIC + TOP-4 DYN on their OWN
        // kernels with the per-K feed REMOVED (operands loaded once, reused for all KT WMMAs). Reveals
        // each config's compute "current potential" with DRAM bandwidth off the table -> the no-feed TF
        // ranking selects the TOP-2 of each group per shape that advance to the rocprof/RGA dive.
        // Respective kernels (no cross-pollination): static=d0_gd_nf, dyn-pf=d1_gd_nf, dyn-dg=d1_dg_nf.
        // Oracle BAD by design (operands garbage) -> fullCheck=false, no oracle column.
        // Per-shape lists: static top-4 from §4 (--mbml8gaunt); dyn top-4 from §6 complete field.
        struct SH { const char* name; int M,K,N; };
        struct CF { int fm,fn; bool dyn; int batch; bool dg; }; // dyn=false => static (batch=1,dg=false)
        struct SF { SH s; CF stat[4]; CF dyn[4]; };
        SF F[] = {
          {{"down   ",2048,9216,2560},
            {{2,8,0,1,0},{2,4,0,1,0},{4,4,0,1,0},{1,4,0,1,0}},
            {{2,4,1,8,0},{1,4,1,32,0},{2,4,1,8,1},{1,4,1,8,0}}},
          {{"gate/up",2048,2560,9216},
            {{2,8,0,1,0},{4,4,0,1,0},{8,2,0,1,0},{2,4,0,1,0}},
            {{2,4,1,32,0},{2,4,1,32,1},{2,4,1,8,0},{1,4,1,32,0}}},
          {{"attn_q ",2048,2560,4096},
            {{2,8,0,1,0},{4,4,0,1,0},{8,2,0,1,0},{2,4,0,1,0}},
            {{2,4,1,8,0},{1,4,1,32,0},{2,4,1,32,0},{2,4,1,32,1}}},
          {{"attn_kv",2048,2560,1024},
            {{2,8,0,1,0},{4,4,0,1,0},{8,2,0,1,0},{2,4,0,1,0}},
            {{2,4,1,8,0},{4,2,1,8,0},{4,1,1,8,0},{2,2,1,32,0}}},
          {{"attn_o ",2048,4096,2560},
            {{2,8,0,1,0},{4,4,0,1,0},{2,4,0,1,0},{4,2,0,1,0}},
            {{2,4,1,8,0},{2,4,1,8,1},{1,4,1,32,0},{1,4,1,8,0}}},
          {{"down_pf",512,9216,2560},
            {{2,8,0,1,0},{4,4,0,1,0},{2,4,0,1,0},{4,2,0,1,0}},
            {{2,2,1,8,0},{2,4,1,1,0},{1,4,1,8,0},{2,4,1,8,0}}},
          {{"gtup_pf",512,2560,9216},
            {{2,8,0,1,0},{4,4,0,1,0},{8,2,0,1,0},{2,4,0,1,0}},
            {{2,4,1,8,0},{2,4,1,8,1},{4,2,1,8,0},{2,4,1,32,0}}},
          {{"q_pf   ",512,2560,4096},
            {{4,4,0,1,0},{2,8,0,1,0},{8,2,0,1,0},{4,2,0,1,0}},
            {{2,4,1,8,0},{4,2,1,8,0},{4,1,1,8,0},{2,2,1,32,0}}},
          {{"kv_pf  ",512,2560,1024},
            {{2,8,0,1,0},{8,2,0,1,0},{4,4,0,1,0},{4,2,0,1,0}},
            {{2,4,1,8,1},{2,2,1,1,1},{2,2,1,8,0},{4,2,1,1,0}}},
          {{"o_pf   ",512,4096,2560},
            {{4,4,0,1,0},{2,8,0,1,0},{8,2,0,1,0},{2,4,0,1,0}},
            {{2,4,1,8,0},{4,1,1,8,0},{1,4,1,8,1},{2,2,1,8,0}}},
        };
        const double TGT = 12.0;
        auto runcell = [&](const char* grp, const CF& c, const SH& s){
            char lbl[28];
            if (c.dyn) snprintf(lbl,sizeof lbl,"%s %dx%d %s b%d",grp,c.fm,c.fn,c.dg?"dg":"pf",c.batch);
            else       snprintf(lbl,sizeof lbl,"%s %dx%d",grp,c.fm,c.fn);
            if ((s.M/16)%c.fm || (s.N/16)%c.fn) { printf("  %-22s (tile doesn't divide)\n",lbl); return; }
            uint32_t pool = c.dyn?1152u:768u;
            char bn[110]; snprintf(bn,sizeof bn,"occ_mbgemm_%dx%d_b%d_d%d_%s_nf.bin",
                                   c.fm,c.fn,c.batch,c.dyn?1:0,c.dg?"dg":"gd");
            MbgResult r = run_mbgemm(node,bn,c.dyn,pool,s.M,s.N,s.K,c.fm,c.fn,/*fullCheck*/false,/*useGenDiv*/true,1,TGT);
            if(!r.ok){ printf("  %-22s (timeout/incomplete)\n",lbl); rc=3; return; }
            int NTL=s.N/(16*c.fn); uint32_t TOTAL=(uint32_t)(s.M/(16*c.fm))*NTL; int KT=s.K/16;
            double flop=(double)TOTAL*c.fm*c.fn*KT*(2.0*16*16*16);
            double meanTF=flop*freq_hz/(double)r.wall/1e12;
            double maxTF=flop*freq_hz/(double)r.wallMin/1e12, minTF=flop*freq_hz/(double)r.wallMax/1e12;
            printf("  %-22s %5.1f %7u  %6.1f[%.0f..%.0f]\n",
                   lbl,(double)r.wallSum/freq_hz,r.maxlive,meanTF,minTF,maxTF);
        };
        const char* onlyShape = getenv("ML8_ONLY");   // e.g. ML8_ONLY=o_pf -> just that shape
        printf("=== ml8 NO-FEED ceiling: top-4 static + top-4 dyn per shape, feed removed (~%.0fs/cell). Oracle BAD by design. ===\n", TGT);
        for (auto& f : F) {
            if (onlyShape && !strstr(f.s.name, onlyShape)) continue;
            printf("\n#### %s  M=%d K=%d N=%d  (NO-FEED ceiling) ####\n", f.s.name, f.s.M, f.s.K, f.s.N);
            printf("  config                 GPUs maxlive  TF[min..max]\n");
            for (auto& c : f.stat) runcell("stat",c,f.s);
            for (auto& c : f.dyn)  runcell("dyn ",c,f.s);
        }
        printf("\n=== no-feed ceiling done. Top-2 TF of each group per shape -> the 40 rocprof cells. ===\n");
    } else if (mode == MBML8GAUNT) {
        // GAUNTLET stage 1: per ml8 shape (REAL dims, no artificial), sustained tile sweep dyn+static,
        // rank by steady-state TF -> print TOP 4 static + TOP 4 dyn per shape. targetSecs/cell makes it
        // time-adaptive across the 10x shape-size range. All bins GENDIV + prefetch + BATCH=1.
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = {
            {"down   ",2048,9216,2560}, {"gate/up",2048,2560,9216}, {"attn_q ",2048,2560,4096},
            {"attn_kv",2048,2560,1024}, {"attn_o ",2048,4096,2560},
            {"down_pf",512,9216,2560},  {"gtup_pf",512,2560,9216},  {"q_pf   ",512,2560,4096},
            {"kv_pf  ",512,2560,1024},  {"o_pf   ",512,4096,2560},
        };
        struct TL { int fm,fn; bool canDyn; const char* tag; double reuse; };
        TL tiles[] = {
            {1,1,true,"1x1",0.50},{2,2,true,"2x2",1.00},{1,4,true,"1x4",0.80},{4,1,true,"4x1",0.80},
            {2,4,true,"2x4",1.33},{4,2,true,"4x2",1.33},
            {4,4,false,"4x4",2.00},{8,2,false,"8x2",1.60},{2,8,false,"2x8",1.60},{8,1,false,"8x1",0.89},
        };
        const double TGT = 12.0;   // ~12s sustained/cell for stage-1 ranking (finalists re-run 30s in stage 2)
        struct R { char tag[8]; bool dyn; double tf; };
        printf("=== ml8 GAUNTLET stage 1: sustained tile sweep dyn+static, per shape (~%.0fs/cell, GENDIV prefetch B1) ===\n", TGT);
        for (auto& s : shapes) {
            printf("\n#### %s  M=%d K=%d N=%d ####\n", s.name, s.M, s.K, s.N);
            printf("  tile reuse mode   GPUs maxlive oracle    TF[min..max]\n");
            R res[24]; int nr=0;
            for (auto& t : tiles) {
                if ((s.M/16) % t.fm || (s.N/16) % t.fn) continue;          // tile must divide the shape
                for (int d=0; d<2; ++d) {
                    bool dyn=(d==1); if (dyn && !t.canDyn) continue;
                    uint32_t pool = dyn?1152u:768u;
                    char bn[96]; snprintf(bn,sizeof bn,"occ_mbgemm_%s_b1_d%d_gd.bin",t.tag,dyn?1:0);
                    MbgResult r = run_mbgemm(node,bn,dyn,pool,s.M,s.N,s.K,t.fm,t.fn,/*fullCheck*/false,/*useGenDiv*/true,1,TGT);
                    if(!r.ok){ printf("  %-4s %.2f %-6s (timeout)\n",t.tag,t.reuse,dyn?"dyn":"static"); rc=3; continue; }
                    int NTL=s.N/(16*t.fn); uint32_t TOTAL=(uint32_t)(s.M/(16*t.fm))*NTL; int KT=s.K/16;
                    double flop=(double)TOTAL*t.fm*t.fn*KT*(2.0*16*16*16);
                    double meanTF=flop*freq_hz/(double)r.wall/1e12;
                    double maxTF=flop*freq_hz/(double)r.wallMin/1e12, minTF=flop*freq_hz/(double)r.wallMax/1e12;
                    bool ok=(r.okTiles==r.nChecked);
                    printf("  %-4s %.2f %-6s %5.1f %7u %s%u/%u %6.1f[%.0f..%.0f]\n",
                        t.tag,t.reuse,dyn?"dyn":"static",(double)r.wallSum/freq_hz,r.maxlive,
                        ok?"OK ":"BAD",r.okTiles,r.nChecked,meanTF,minTF,maxTF);
                    if(ok && nr<24){ snprintf(res[nr].tag,8,"%s",t.tag); res[nr].dyn=dyn; res[nr].tf=meanTF; nr++; }
                }
            }
            // top-4 static + top-4 dyn by simple repeated-max selection
            for (int grp=0; grp<2; ++grp) { bool wantDyn=(grp==1);
                printf("  TOP4 %-6s:", wantDyn?"DYN":"STATIC");
                bool used[24]={false};
                for (int k=0;k<4;++k){ int best=-1; for(int i=0;i<nr;++i) if(res[i].dyn==wantDyn && !used[i] && (best<0||res[i].tf>res[best].tf)) best=i;
                    if(best<0) break; used[best]=true; printf("  %s(%.1f)",res[best].tag,res[best].tf); }
                printf("\n");
            }
        }
        printf("\n=== stage 1 done. Next: top-2 each -> RGA+rocprof -> adjust -> 30s rerun -> top-1. ===\n");
    } else if (mode == DYNFAT1) {
        // SINGLE-SHOT dyn-VGPR fat-tile measurement (GPT safe protocol; no sweep -> volatile cap can't revert):
        //   (1) GPU health smoke (4x4 STATIC, no dyn, no gate);
        //   (2) prebuild operands+queue+ring for ONE 4x4 DYN config (192 VGPR);
        //   (3) BLOCK on /tmp/dynfat_go until operator flips umr cap + touches in ONE command;
        //   (4) dispatch <1s after flip; (5) report TF/correctness/maxlive/claims/grabs/pool; (6) STOP.
        printf("=== DYNFAT1: single-shot dyn-VGPR fat-tile measurement (gated, no sweep) ===\n");
        MbgResult sm = run_mbgemm(node, "occ_mbgemm_4x4_b1_d0.bin", false, 768u, 2048,2048,2048, 4,4, false);
        if (!sm.ok) {
            fprintf(stderr, "*** GPU HEALTH SMOKE FAILED -- GPU not usable; NO dyn dispatch. Reboot/recover first. ***\n");
            rc = 3;
        } else {
            printf("  GPU health OK (4x4 static 2048^3: maxlive=%u, span=%.2f ms)\n",
                   sm.maxlive, (double)sm.wall/freq_hz*1e3);
            const int FM=4, FN=4, M=8192, N=8192, K=8192; const uint32_t pool=1152u;
            const char* gate = "/tmp/dynfat_go"; const char* ready = "/tmp/dynfat_ready";
            remove(gate); remove(ready); g_gateFile = gate; g_readyFile = ready;
            MbgResult r = run_mbgemm(node, "occ_mbgemm_4x4_b1_d1.bin", true, pool, M,N,K, FM,FN, false);
            g_gateFile = nullptr; g_readyFile = nullptr;
            if (!r.ok) {
                fprintf(stderr, "*** DYNFAT1 dyn dispatch did NOT complete (hang/timeout). Cap likely not 256 at dispatch. ***\n");
                rc = 3;
            } else {
                uint32_t TOTAL = (uint32_t)(M/(16*FM)) * (N/(16*FN)); int KT = K/16;
                double work = (double)TOTAL * FM * FN * KT;
                double tf = work * (2.0*16*16*16) * freq_hz / (double)r.wall / 1e12;
                printf("\n  === DYN-FAT RESULT (4x4 dyn, 192 VGPR, %dx%d K%d, BATCH=1) ===\n", M, N, K);
                printf("    TOTAL(tiles=claims)=%u  grabs/pool=%.1fx  maxlive=%u  launched=%u\n",
                       TOTAL, (double)TOTAL/pool, r.maxlive, r.total);
                printf("    oracle %s%u/%u   TF=%.1f  (%.1f%% of 307)   span=%.2f ms\n",
                       r.okTiles==r.nChecked?"OK ":"BAD", r.okTiles, r.nChecked, tf, 100*tf/307.0,
                       (double)r.wall/freq_hz*1e3);
                printf("    COMPARE: static 4x4 @8192 (same shape, 768 waves) = ~32 TF.\n");
                printf("    dyn ~= static => occupancy did not convert; dyn >> static => dyn-fat jumps.\n");
            }
        }
    } else if (mode == MBPROF) {
        // IN-KERNEL PHASE TIMING -- the rocprof-equivalent. Workgroup 0 accumulates REALTIME (100 MHz)
        // ticks spent in each phase of the per-tile cycle; we table the breakdown. 2x4, batch32, 2048^3.
        printf("=== IN-KERNEL PHASE BREAKDOWN (REALTIME 100MHz, workgroup 0, acc[2x4] b32, 2048^3) ===\n");
        const char* names[6] = { "ATOMIC (grab)", "GROW (s_alloc up)", "SETUP (decode+zero)",
                                 "COMPUTE (K-loop+feed)", "STORE", "SHRINK (s_alloc down)" };
        for (int nf = 0; nf < 2; ++nf) {
            const char* bn = nf ? "occ_mbgemm_2x4_b32_prof1.bin" : "occ_mbgemm_2x4_b32_prof0.bin";
            MbgResult r = run_mbgemm(node, bn, true, 1152u, 2048,2048,2048, 2,4, false);
            if (!r.ok) { fprintf(stderr, "  %s incomplete\n", bn); rc = 3; continue; }
            uint64_t tot = 0; for (int i = 0; i < 6; ++i) tot += r.phase[i];
            printf("\n  [%s feed]  wg0 total = %llu ticks = %.3f ms   (whole-dispatch wall %.3f ms)\n",
                   nf ? "NO" : "REAL", (unsigned long long)tot, tot * 10.0 / 1e6, (double)r.wall/freq_hz*1e3);
            printf("    phase                       ticks         us     %%\n");
            for (int i = 0; i < 6; ++i)
                printf("    %-24s %10u  %9.2f  %5.1f\n", names[i], r.phase[i], r.phase[i] * 10.0 / 1e3,
                       tot ? 100.0 * r.phase[i] / (double)tot : 0.0);
        }
        printf("\n  (10 ns/tick. COMPUTE = the K-loop incl feed waits; the rest = per-tile/per-batch overhead.)\n");
    } else if (mode == MBML8PROF) {
        // ml8 FRAMEWORK PHASE SPLIT: run the static winner (2x8) + dyn winner (2x4 b8) PROFILE builds
        // on the REAL ml8 dims, table the ATOMIC/GROW/SETUP/COMPUTE/STORE/SHRINK breakdown. Tells us
        // which bookkeeping phase dominates the framework wall (what a wave-spec redesign must hide),
        // and quantifies dyn's GROW/SHRINK tax. Realtime timers carry a ~70x perturbation -> read the
        // RATIOS between phases, NOT absolute time. (occ[24..44], wg0; fed; useGenDiv=true for non-pow2 N.)
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = {
            {"down   ",2048,9216,2560},  // deep-K, the feed-bound end (COMPUTE should dominate)
            {"gate/up",2048,2560,9216},  // shallow-K big-N, the lever shape
            {"q_pf   ",512,2560,4096},   // prefill, the framework-bound end (overhead should be worst)
        };
        struct CF { const char* lbl; const char* bin; bool dyn; uint32_t pool; int fm,fn; };
        CF cfgs[] = {
            {"static 2x8", "occ_mbgemm_2x8_b1_d0_prof.bin", false, 768u,  2,8},
            {"dyn 2x4 b8", "occ_mbgemm_2x4_b8_d1_prof.bin", true,  1152u, 2,4},
        };
        const char* names[6] = { "ATOMIC", "GROW", "SETUP", "COMPUTE", "STORE", "SHRINK" };
        printf("=== ml8 FRAMEWORK PHASE SPLIT (realtime 100MHz, wg0, fed). READ RATIOS, not absolute (timers ~70x perturb). ===\n");
        printf("    COMPUTE = productive K-loop. Everything else = bookkeeping bubbles the wave-spec redesign would hide.\n");
        for (auto& s : shapes) {
            printf("\n#### %s  M=%d K=%d N=%d ####\n", s.name, s.M, s.K, s.N);
            printf("  config        %8s %6s %7s %9s %7s %7s   (wg0 ticks)\n",
                   names[0],names[1],names[2],names[3],names[4],names[5]);
            for (auto& c : cfgs) {
                if ((s.M/16)%c.fm || (s.N/16)%c.fn) { printf("  %-12s (tile doesn't divide)\n", c.lbl); continue; }
                MbgResult r = run_mbgemm(node, c.bin, c.dyn, c.pool, s.M,s.N,s.K, c.fm,c.fn, /*fullCheck*/false, /*useGenDiv*/true);
                if (!r.ok) { printf("  %-12s incomplete (maxlive=%u)\n", c.lbl, r.maxlive); rc=3; continue; }
                uint64_t tot=0; for(int i=0;i<6;++i) tot+=r.phase[i];
                printf("  %-12s", c.lbl);
                for (int i=0;i<6;++i) printf(" %6.1f%%", tot?100.0*r.phase[i]/(double)tot:0.0);
                printf("   %llu\n", (unsigned long long)tot);
            }
        }
        printf("\n=== Higher COMPUTE%% = less wave-spec headroom. High ATOMIC/SETUP%% = the phases a claim/setup wave can hide. ===\n");
    } else if (mode == MBML8BATCH) {
        // ATOMIC-CLAIM IMPACT (clean, timer-free): the phase split said the framework wall is ~entirely the
        // ONE device-scope atomic claim. Static was only ever measured at b1 (full atomic toll). Sweep
        // BATCH{1,8,32} on the BEST tile (2x8) and the WORST tile (1x1 = max tiles = max claim density).
        // The b1->b32 TF lift = the size of the atomic wall. The worst tile should lift MOST (most claims to
        // amortize); 2x8 lifts less (already fat/few-claims). Real sustained TF, NO PROFILE perturbation.
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = {
            {"down   ",2048,9216,2560}, {"gate/up",2048,2560,9216}, {"attn_q ",2048,2560,4096},
            {"attn_kv",2048,2560,1024}, {"attn_o ",2048,4096,2560},
            {"down_pf",512,9216,2560},  {"gtup_pf",512,2560,9216},  {"q_pf   ",512,2560,4096},
            {"kv_pf  ",512,2560,1024},  {"o_pf   ",512,4096,2560},
        };
        struct T { const char* lbl; int fm,fn; } tiles[] = { {"2x8 (best)",2,8}, {"1x1 (worst)",1,1} };
        int batches[] = {1,8,32};
        const double TGT = 8.0;
        const char* onlyShape = getenv("ML8_ONLY");
        printf("=== ml8 STATIC ATOMIC-CLAIM BATCH SWEEP: 2x8 (best) + 1x1 (worst) x BATCH{1,8,32}, real TF (~%.0fs/cell) ===\n", TGT);
        printf("    b1->b32 lift = atomic-claim wall size. Worst tile (most claims) should lift most.\n");
        for (auto& s : shapes) {
            if (onlyShape && !strstr(s.name, onlyShape)) continue;
            printf("\n#### %s  M=%d K=%d N=%d ####\n", s.name, s.M, s.K, s.N);
            for (auto& t : tiles) {
                if ((s.M/16)%t.fm || (s.N/16)%t.fn) { printf("  %-12s (tile doesn't divide)\n", t.lbl); continue; }
                double tf[3] = {0,0,0};
                for (int bi = 0; bi < 3; ++bi) {
                    char bn[96]; snprintf(bn,sizeof bn,"occ_mbgemm_%dx%d_b%d_d0_gd.bin", t.fm,t.fn,batches[bi]);
                    MbgResult r = run_mbgemm(node,bn,false,768u,s.M,s.N,s.K,t.fm,t.fn,/*fullCheck*/false,/*useGenDiv*/true,1,TGT);
                    if (!r.ok) { rc=3; continue; }
                    int NTL=s.N/(16*t.fn); uint32_t TOTAL=(uint32_t)(s.M/(16*t.fm))*NTL; int KT=s.K/16;
                    double flop=(double)TOTAL*t.fm*t.fn*KT*(2.0*16*16*16);
                    tf[bi]=flop*freq_hz/(double)r.wall/1e12;
                }
                double lift = tf[0]>0 ? 100.0*(tf[2]-tf[0])/tf[0] : 0.0;
                printf("  %-12s  b1=%6.2f  b8=%6.2f  b32=%6.2f   b1->b32 %+5.0f%%\n", t.lbl, tf[0],tf[1],tf[2], lift);
            }
        }
        printf("\n=== Bigger b1->b32 lift = bigger atomic-claim wall. If 2x8 lifts too, that's a free win on the existing kernel. ===\n");
    } else if (mode == MBML8MATCH) {
        // MATCHED-BATCH FAIR RACE (steps 1+2): static 2x8 AND dyn 2x4, BOTH at BATCH{1,8,32}, real fed TF,
        // same run. The static-b1 confound (gate/up looked like a dyn win) is removed -> the honest question:
        // does dyn's BEST batch ever beat static's BEST batch? Plus it banks the per-shape static batch win.
        // Bins all pre-gated (static 2x8 from the atomic sweep; dyn 2x4 gd = the 48/48 dynfull set). Single-
        // wave dyn (zero s_barrier, SCC-retry, <=128 VGPR) -> brick-safe; compositor yield on; batch only
        // changes grow FREQUENCY not the grow target.
        struct SH { const char* name; int M,K,N; };
        SH shapes[] = {
            {"down   ",2048,9216,2560}, {"gate/up",2048,2560,9216}, {"attn_q ",2048,2560,4096},
            {"attn_kv",2048,2560,1024}, {"attn_o ",2048,4096,2560},
            {"down_pf",512,9216,2560},  {"gtup_pf",512,2560,9216},  {"q_pf   ",512,2560,4096},
            {"kv_pf  ",512,2560,1024},  {"o_pf   ",512,4096,2560},
        };
        struct CF { const char* lbl; int fm,fn; bool dyn; uint32_t pool; } cfgs[] = {
            {"stat 2x8", 2,8, false, 768u },
            {"dyn  2x4", 2,4, true, 1152u },
        };
        int batches[] = {1,8,32};
        const double TGT = 8.0;
        const char* onlyShape = getenv("ML8_ONLY");
        printf("=== ml8 MATCHED-BATCH FAIR RACE: static 2x8 vs dyn 2x4, BOTH x BATCH{1,8,32}, real fed TF (~%.0fs/cell) ===\n", TGT);
        printf("    Honest question: does dyn's BEST batch beat static's BEST batch once static is also batched?\n");
        for (auto& s : shapes) {
            if (onlyShape && !strstr(s.name, onlyShape)) continue;
            printf("\n#### %s  M=%d K=%d N=%d ####\n", s.name, s.M, s.K, s.N);
            double best[2] = {0,0};
            for (int ci = 0; ci < 2; ++ci) {
                CF& c = cfgs[ci];
                if ((s.M/16)%c.fm || (s.N/16)%c.fn) { printf("  %-8s (tile doesn't divide)\n", c.lbl); continue; }
                double tf[3] = {0,0,0};
                for (int bi = 0; bi < 3; ++bi) {
                    char bn[96]; snprintf(bn,sizeof bn,"occ_mbgemm_%dx%d_b%d_d%d_gd.bin", c.fm,c.fn,batches[bi],c.dyn?1:0);
                    MbgResult r = run_mbgemm(node,bn,c.dyn,c.pool,s.M,s.N,s.K,c.fm,c.fn,/*fullCheck*/false,/*useGenDiv*/true,1,TGT);
                    if (!r.ok) { rc=3; continue; }
                    int NTL=s.N/(16*c.fn); uint32_t TOTAL=(uint32_t)(s.M/(16*c.fm))*NTL; int KT=s.K/16;
                    double flop=(double)TOTAL*c.fm*c.fn*KT*(2.0*16*16*16);
                    tf[bi]=flop*freq_hz/(double)r.wall/1e12;
                    if (tf[bi] > best[ci]) best[ci] = tf[bi];
                }
                printf("  %-8s  b1=%6.2f  b8=%6.2f  b32=%6.2f   best=%6.2f\n", c.lbl, tf[0],tf[1],tf[2], best[ci]);
            }
            if (best[0] > 0 && best[1] > 0) {
                double gap = 100.0*(best[1]-best[0])/best[0];
                printf("  --> matched best: static %.2f vs dyn %.2f  =>  dyn %+5.1f%%  %s\n",
                       best[0], best[1], gap, gap > 0 ? "*** DYN WINS ***" : "(static wins)");
            }
        }
        printf("\n=== Matched-batch gap is REUSE (2x8@208VGPR reuse1.6 vs dyn 2x4 capped at 128, reuse1.33), not a dyn deficit.\n");
        printf("    Single-wave vehicle reuse-caps dyn -> the cooperative hybrid (lean feed + bounded dyn compute cluster) is the fix. ===\n");
    } else if (mode == DYNSMOKE) {
        // ===== MAD-305 2-wave dyn-VGPR coordination isolation probe (NO GEMM). Brick-safe: occ-only writes. =====
        run_dynsmoke(node);
    } else if (mode == MBML8COOP) {
        // ===== MAD-305 HYBRID COOPERATIVE harness (B1 — HYBRID_DESIGN.md Step 5c/8). Per ml8 shape: 512-ish
        // ORACLE gate (must be frag-exact) -> sustained real-shape TF, on the (1+P)-wave shared-B cooperative
        // kernel. dyn-armed (RSRC2 bit6), pool-capped, GENDIV (non-pow2 N). Config via env: ML8_P (compute
        // waves/WG, default 3), ML8_RINGD (B-ring depth, 2), ML8_FM/ML8_FN (per-wave tile, 2x4), ML8_BATCH (1),
        // ML8_POOL (resident WGs), ML8_DYN (1), ML8_ONLY (substring filter). Reports TF vs the batched-static
        // FLOOR (Step 0) and toward the 250-300 NORTH STAR.
        //
        // BRICK-GUARD: the COOP kernel bin (occ_kernel_coop.s -> occ_coop_*.bin, built in B2) does not exist
        // yet at B1. If it's absent we REFUSE to dispatch (file-not-found = safe stop; a geometry/bin mismatch
        // under bit6-armed dyn is exactly what bricks gfx1201). This makes B1 inherently GPU-safe to run.
        struct SH { const char* name; int M,K,N; double floor; };
        SH shapes[] = {   // floor = batched static 2x8 (HYBRID_DESIGN.md FLOOR line); decode shapes are mem-bound (sep regime)
            {"down   ",2048,9216,2560, 21.0}, {"gate/up",2048,2560,9216, 20.4}, {"attn_q ",2048,2560,4096, 11.0},
            {"attn_kv",2048,2560,1024,  3.0}, {"attn_o ",2048,4096,2560, 11.3},
            {"down_pf",512,9216,2560,   6.6}, {"gtup_pf",512,2560,9216,  6.7}, {"q_pf   ",512,2560,4096,  3.0},
            {"kv_pf  ",512,2560,1024,   0.8}, {"o_pf   ",512,4096,2560,  3.0},
        };
        const int    P     = getenv("ML8_P")     ? atoi(getenv("ML8_P"))     : 3;
        const int    RINGD = getenv("ML8_RINGD") ? atoi(getenv("ML8_RINGD")) : 2;
        const int    FM    = getenv("ML8_FM")    ? atoi(getenv("ML8_FM"))    : 2;
        const int    FN    = getenv("ML8_FN")    ? atoi(getenv("ML8_FN"))    : 4;
        const int    BATCH = getenv("ML8_BATCH") ? atoi(getenv("ML8_BATCH")) : 1;
        const bool   dyn   = getenv("ML8_DYN")   ? atoi(getenv("ML8_DYN"))   : 1;
        const uint32_t pool= getenv("ML8_POOL")  ? (uint32_t)atoi(getenv("ML8_POOL")) : 256u;   // resident (1+P)-wave WGs
        const double TGT   = getenv("ML8_TGT")   ? atof(getenv("ML8_TGT"))   : 8.0;
        const char* onlyShape = getenv("ML8_ONLY");
        const int TM = P*FM*16, TN = FN*16;        // WG tile
        char bn[128]; snprintf(bn, sizeof bn, "occ_coop_%dx%d_p%d_r%d_b%d_d%d_gd.bin", FM,FN,P,RINGD,BATCH,dyn?1:0);
        printf("=== ml8 HYBRID COOPERATIVE (1 feed + P=%d compute, shared-B; FM=%d FN=%d RINGD=%d BATCH=%d dyn=%d pool=%u) ===\n",
               P, FM, FN, RINGD, BATCH, dyn, pool);
        printf("    WG tile = %dx%d (P*FM*16 M-rows x FN*16 shared N-cols).  kernel bin = %s\n", TM, TN, bn);
        printf("    FLOOR = batched static 2x8 (must not regress); NORTH STAR = 250-300 TF compute (train/prefill).\n");
        // ---- BRICK-GUARD: refuse to dispatch if the B2 kernel bin is not built ----
        { FILE* fb = fopen(bn, "rb");
          if (fb) fclose(fb);
          else { printf("\n*** COOP kernel bin '%s' NOT BUILT (this is the expected B1 state) -- REFUSING to dispatch.\n"
                        "    Build it in B2 (occ_kernel_coop.s + build_coop.sh) then re-run --mbml8coop. ***\n", bn);
                 rc = 4; }
        }
        if (rc == 0) for (auto& s : shapes) {
            if (onlyShape && !strstr(s.name, onlyShape)) continue;
            if ((s.M % TM) || (s.N % TN) || (s.K % 16)) { printf("\n#### %s  (WG tile %dx%d doesn't divide M=%d N=%d) skipped\n", s.name, TM,TN, s.M, s.N); continue; }
            printf("\n#### %s  M=%d K=%d N=%d  (floor %.1f) ####\n", s.name, s.M, s.K, s.N, s.floor);
            // 1) ORACLE gate first: smallest tile-multiple shape (sub-second dispatch; wedge = free reboot).
            //    ML8_ORACLE_MTL/NTL shrink the tile grid for last-tile-store-hang isolation: MTL=NTL=1 -> TOTAL=1
            //    (single tile, feed-terminal, writes buffer START -> separates "feed-terminal" from "buffer-end").
            int oMTL = getenv("ML8_ORACLE_MTL") ? atoi(getenv("ML8_ORACLE_MTL")) : 4;
            int oNTL = getenv("ML8_ORACLE_NTL") ? atoi(getenv("ML8_ORACLE_NTL")) : 8;
            int Mo = TM*oMTL, No = TN*oNTL, Ko = 512;
            CoopResult o = run_mbcoop(node, bn, dyn, pool<64u?pool:64u, Mo,No,Ko, FM,FN,P,RINGD, /*fullCheck*/true, /*GENDIV*/true, 1, 0.0);
            if (!o.ok) { printf("  oracle INCOMPLETE (hang/timeout) -> protocol/grow bug; NOT proceeding to perf\n"); rc = 3; continue; }
            bool clean = (o.badFrags == 0 && o.okFrags > 0);
            printf("  oracle %dx%dx%d: %s  ok=%llu bad=%llu\n", Mo,No,Ko, clean?"CLEAN":"*** BAD (stale-B/math) ***",
                   (unsigned long long)o.okFrags, (unsigned long long)o.badFrags);
            if (!clean) { rc = 3; continue; }   // never report TF on an unproven kernel
            if (getenv("ML8_ORACLE_ONLY")) {     // B4 gate: stop after the sub-second oracle (no sustained hold)
                printf("  [oracle-only] correctness GREEN; skipping sustained perf (unset ML8_ORACLE_ONLY for B5)\n");
                continue;
            }
            // 2) sustained real-shape perf (compositor yield on; ~TGT s).
            CoopResult r = run_mbcoop(node, bn, dyn, pool, s.M,s.N,s.K, FM,FN,P,RINGD, /*fullCheck*/false, /*GENDIV*/true, 1, TGT);
            if (!r.ok) { printf("  perf INCOMPLETE (hang/timeout)\n"); rc = 3; continue; }
            double flop = 2.0*(double)s.M*s.N*s.K;
            double tf = (r.wall>0) ? flop*freq_hz/(double)r.wall/1e12 : 0.0;
            double vsFloor = s.floor>0 ? 100.0*(tf-s.floor)/s.floor : 0.0;
            printf("  PERF %6.2f TF  (%4.1f%% of 307 peak)  vs floor %.1f => %+5.0f%%  maxlive=%u WGs claims=%u\n",
                   tf, 100.0*tf/307.0, s.floor, vsFloor, r.maxlive, r.total);
        }
        printf("\n=== COOP harness: oracle-gated per shape; reuse climbs with P (P=2->2.0, P=3->2.4, P=4->2.67). ===\n");
    } else if (mode == MERGE) {
        // ===== MAD-305 MERGE (T2 G2 gate): the hand-asm 4x4 with the fine s_wait_loadcnt ladder
        // (WMMABUF_WAIT) must reach hipcc parity (~147-155 TF) at a wgrad shape, bit-exact, BEFORE the
        // fat tile goes on. Square big-K only: the power-of-2 tile-grid decode needs N/(16*FN) a power of
        // two (4096/64=64 OK); the realistic 14336 wgrad dim is non-pow2 and waits on a general-divide
        // kernel change (T8). Reuses the existing occ_mbgemm_4x4_b32_d0.bin (static 192 VGPR), which now
        // carries the ladder since occ_kernel_mbgemm.s was rebuilt. =====
        g_prog = fopen("/tmp/occ_merge_progress.log", "w");
        prog("=== MAD-305 MERGE (lockstep-stagger, BATCH=1, pool=768) -- crash log: /tmp/occ_merge_progress.log ===");
        const int FM = 4, FN = 4;
        // Staged smallest->biggest so a hang is ISOLATED, and each STARTING line is fsync'd to disk BEFORE its
        // GPU dispatch -> a hard reboot leaves the wedging config named on disk. oracle 512^3 -> smoke 2048^3
        // (KG-proven shape; isolates BATCH=1) -> the real 4096^2 x K8192 stagger sweep (KG 50147c07).
        prog("STARTING oracle st0 @512^3");
        MbgResult o = run_mbgemm(node, "occ_mbgemm_4x4_b1_st0_d0.bin", false, 768u, 512,512,512, FM,FN, true);
        if (!o.ok) { prog("  oracle INCOMPLETE -> BATCH=1 path hangs even at 512^3"); rc = 3; }
        else {
            bool bitexact = (o.okTiles == o.nChecked);
            prog("DONE oracle 512^3: %s %u/%u bit-exact", bitexact ? "OK" : "BAD", o.okTiles, o.nChecked);

            prog("STARTING smoke st0 @2048^3 (proven shape -- isolates BATCH=1 from the big shape)");
            { MbgResult s = run_mbgemm(node, "occ_mbgemm_4x4_b1_st0_d0.bin", false, 768u, 2048,2048,2048, FM,FN, false);
              if (!s.ok) prog("  smoke 2048^3 INCOMPLETE -> the BATCH=1 path itself hangs (not the big shape, not the stagger)");
              else { uint32_t T=(uint32_t)(2048/(16*FM))*(2048/(16*FN)); int KT=2048/16; double w=(double)T*FM*FN*KT;
                     double tf=w*(2.0*16*16*16)*freq_hz/(double)s.wall/1e12;
                     prog("DONE smoke 2048^3: OK  %.3f ms  %.1f TF  maxlive=%u", (double)s.wall/freq_hz*1e3, tf, s.maxlive); } }

            if (getenv("MERGE_BIG")) {
                // The fair test of the lockstep-stagger fix: does MORE occupancy let it pay off? pool ascending
                // 768/1152/1536 = ~3/4.5/6 waves/SIMD at 192-VGPR static (37%/56%/75% of the VGPR file -- all
                // safely BELOW the 2048=100%-file zero-slack deadlock). Lower-risk pools run + log first.
                prog("-- pool x stagger sweep @ 4096^2 x K8192 (does occupancy unlock the lockstep-stagger fix?) --");
                const uint32_t pools[] = { 768u, 1152u, 1536u };
                const int sts[] = { 0, 16, 64 };
                for (uint32_t pool : pools) {
                  for (int st : sts) {
                    char bn[80]; snprintf(bn, sizeof bn, "occ_mbgemm_4x4_b1_st%d_d0.bin", st);
                    int K = 8192;
                    prog("STARTING pool=%u st%d @4096^2 x K%d", pool, st, K);
                    MbgResult r = run_mbgemm(node, bn, false, pool, 4096,4096,K, FM,FN, false);
                    if (!r.ok) { prog("  pool=%u st%d INCOMPLETE", pool, st); rc = 3; continue; }
                    uint32_t TOTAL = (uint32_t)(4096/(16*FM)) * (4096/(16*FN)); int KT = K/16;
                    double work = (double)TOTAL * FM * FN * KT;
                    double tf = work * (2.0*16*16*16) * freq_hz / (double)r.wall / 1e12;
                    double wmma_cyc = 15.9 * tf / 307.0;
                    prog("DONE pool=%-4u st%-3d maxlive=%u  %.3f ms  %.1f TF  %.1f%%  %.2f WMMA/cyc",
                         pool, st, r.maxlive, (double)r.wall/freq_hz*1e3, tf, 100*tf/307.0, wmma_cyc);
                  }
                }
                prog("Reading: TF rises with pool+stagger => lockstep fix works once it has co-resident waves; flat => single-wave occupancy-dead (cd407a9b).");
            } else {
                prog("-- big 4096^2 x K8192 sweep SKIPPED (set MERGE_BIG=1 to run it once the smoke is confirmed clean) --");
            }
            if (!bitexact) prog("NOTE: oracle NOT bit-exact (%u/%u).", o.okTiles, o.nChecked);
        }
        if (g_prog) { fclose(g_prog); g_prog = nullptr; }
    } else if (mode == SGPRPROBE) {
        // ===== Find which SGPR carries the per-workgroup id under raw PM4 / USER_SGPR=15. =====
        printf("\n=== SGPR PROBE (raw PM4, 128-thread WG, USER_SGPR=15) ===\n");
        run_sgpr_probe(node, "occ_wgdiag.bin", 8u);
    } else if (mode == WGPERF) {
        // ===== MAD-305 Phase 3: wave-group G2 BASELINE perf (STORE=0 minimal-store, NO tuning yet). =====
        printf("\n=== MAD-305 Phase 3: wave-group G2 BASELINE perf (STORE=0 minimal store, no tuning) ===\n");
        printf("    ceiling 307 TF = 15.9 WMMA/cyc; HIP winner @4096^2 x K16384 = 161.1 TF (hard pass >=153)\n");
        struct Sm { const char* name; int M, N, K; uint32_t nWG; };
        Sm cases[] = {
            { "1024^2 x K2048  (sanity)",  1024, 1024,  2048, 256u },
            { "2048^2 x K4096  (medium)",  2048, 2048,  4096, 256u },
            { "4096^2 x K16384 (TARGET)",  4096, 4096, 16384, 256u },
        };
        for (auto& c : cases) {
            WgpResult r = run_wggemm_perf(node, "occ_wggemm2_perf.bin", c.M, c.N, c.K, c.nWG, freq_hz);
            if (!r.ok) { printf("  %-28s INCOMPLETE (hang/timeout)\n", c.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0, wpc = 15.9*r.tf/307.0;
            printf("  %-28s %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs  claims=%u  acc00 OK=%u/%u\n",
                   c.name, r.tf, pct, wpc, r.maxlive, r.total, r.okSamp, r.okSamp + r.badSamp);
        }
    } else if (mode == WAVESPEC) {
        // ===== MAD-305 #323: lean WAVE-SPECIALIZED fp8 GEMM. Oracle (STORE=1) then perf (STORE=0). =====
        // Config via env: WS_FM WS_FN WS_NLOAD WS_NCOMP WS_DYN (defaults 2/2/1/4/0).
        int FM = getenv("WS_FM") ? atoi(getenv("WS_FM")) : 2;
        int FN = getenv("WS_FN") ? atoi(getenv("WS_FN")) : 2;
        int NLOAD = getenv("WS_NLOAD") ? atoi(getenv("WS_NLOAD")) : 1;
        int NCOMP = getenv("WS_NCOMP") ? atoi(getenv("WS_NCOMP")) : 4;
        bool dyn = getenv("WS_DYN") && atoi(getenv("WS_DYN"));
        printf("\n=== MAD-305 #323: lean wave-specialized fp8 GEMM (FM=%d FN=%d NLOAD=%d NCOMP=%d dyn=%d) ===\n",
               FM, FN, NLOAD, NCOMP, dyn);
        // ---- correctness gate: STORE=1 bin, small shape (tile = NCOMP*FM*16 x FN*16) ----
        int TM = NCOMP*FM*16, TN = FN*16;
        int Mc = TM*4, Nc = TN*16, Kc = 512;          // 4 M-tiles x 16 N-tiles (NTL pow2), K=512
        // the kernel BINARY must match (FM,FN,NLOAD,NCOMP,dyn) EXACTLY: DYNVGPR=1 bins carry s_alloc_vgpr +
        //   the lean streamed loader, and NLOAD/NCOMP are baked into the kernel's wave-role split. Loading a
        //   mismatched bin under bit6-armed dispatch is the hang/brick. Names are fully qualified per cell
        //   (build.sh stanza [1h2]): occ_ws_<FM>x<FN>_l<NLOAD>_c<NCOMP>[_dyn][_st].bin.
        char stbin[128], perfbin[128];
        // WS_BW=1 (dyn only): the BUSYWAIT variant that swaps the 4 asymmetric K-slice s_barrier for an
        //   LDS busy-wait (T6 BRICK #4 fix: s_barrier deadlocks under dyn-VGPR at mixed allocations).
        bool bw = dyn && getenv("WS_BW") && atoi(getenv("WS_BW"));
        const char* dtag = dyn ? "_dyn" : "";
        const char* btag = bw ? "_bw" : "";
        snprintf(perfbin, sizeof perfbin, "occ_ws_%dx%d_l%d_c%d%s%s.bin",    FM, FN, NLOAD, NCOMP, dtag, btag);
        snprintf(stbin,   sizeof stbin,   "occ_ws_%dx%d_l%d_c%d%s%s_st.bin", FM, FN, NLOAD, NCOMP, dtag, btag);
        if (bw) printf("  [BUSYWAIT] s_barrier->LDS busy-wait on the 4 K-slice barriers (claim barrier kept)\n");
        // BRICK-GUARD: if the EXACT bin for this config was not built, REFUSE to dispatch. A file-not-found
        //   is a safe stop; a geometry/bin mismatch under bit6-armed dyn is what bricked gfx1201. Build first.
        { const char* miss = nullptr;
          FILE* fa = fopen(stbin, "rb");   if (fa) fclose(fa);   else miss = stbin;
          FILE* fb = fopen(perfbin, "rb"); if (fb) fclose(fb);   else if (!miss) miss = perfbin;
          if (miss) { printf("  *** bin '%s' not built -- run ./build.sh (stanza [1h2]); REFUSING to dispatch (mismatch=brick) ***\n", miss); rc = 4; } }
        if (rc == 0) {
        printf("  [oracle] %dx%dx%d  tile=%dx%d  bin=%s\n", Mc, Nc, Kc, TM, TN, stbin);
        WgcResult o = run_wavespec_compute(node, stbin, Mc, Nc, Kc, 64u, true, FM, FN, NLOAD, NCOMP, dyn);
        if (!o.ok) { printf("  ORACLE INCOMPLETE (hang/timeout)\n"); rc = 3; }
        else {
            uint64_t tot = o.okFrags + o.badFrags;
            printf("  ORACLE %s  (%llu/%llu frags OK, bad=%llu)  maxlive=%u claims=%u\n",
                   o.badFrags==0 ? "OK" : "*** MISMATCH ***",
                   (unsigned long long)o.okFrags, (unsigned long long)tot, (unsigned long long)o.badFrags, o.maxlive, o.total);
            if (o.badFrags) rc = 3;
        }
        }
        // ---- perf: STORE=0 bin, real shape ----
        if (rc == 0) {
            struct Sm { const char* name; int M, N, K; uint32_t nWG; };
            Sm cases[] = {
                { "2048x2048xK4096  (medium)", 2048, 2048,  4096, 256u },
                { "4096x4096xK16384 (TARGET)", 4096, 4096, 16384, 256u },
            };
            for (auto& c : cases) {
                WgpResult r = run_wavespec_perf(node, perfbin, c.M, c.N, c.K, c.nWG, freq_hz, FM, FN, NLOAD, NCOMP, dyn);
                if (!r.ok) { printf("  %-28s INCOMPLETE (hang/timeout)\n", c.name); rc = 3; continue; }
                double pct = 100.0*r.tf/307.0;
                printf("  %-28s %6.1f TF  %4.1f%% of 307  maxlive=%u WGs  claims=%u\n",
                       c.name, r.tf, pct, r.maxlive, r.total);
            }
        }
    } else if (mode == WGGEMM2) {
        // ===== MAD-305 Phase 2: wave-group fp8 GEMM compute (A-LDS + B-trfeed + static 4x4). Oracle. =====
        printf("\n=== MAD-305 Phase 2: wave-group fp8 GEMM COMPUTE (A-LDS + B-global_load_tr + static 4x4) ===\n");
        struct Sm { const char* name; int M, N, K; uint32_t nWG; };
        Sm cases[] = {
            { "256x256x256   (4 tiles)",         256, 256,  256,  4u },
            { "512x512x512   (16 tiles)",        512, 512,  512,  8u },
            { "512x512x2048  (16 tiles, big-K)", 512, 512, 2048,  8u },
        };
        for (auto& c : cases) {
            WgcResult r = run_wggemm_compute(node, "occ_wggemm2.bin", c.M, c.N, c.K, c.nWG, true);
            if (!r.ok) { printf("  %-34s INCOMPLETE (hang/timeout)\n", c.name); rc = 3; continue; }
            uint64_t expect = (uint64_t)(c.M/128) * (c.N/128) * 4 * 16;   // tiles * waves * frags
            bool pass = (r.okFrags == expect && r.badFrags == 0);
            printf("  %-34s maxlive=%u claims=%u  frags OK=%llu/%llu bad=%llu  %s\n",
                   c.name, r.maxlive, r.total, (unsigned long long)r.okFrags,
                   (unsigned long long)expect, (unsigned long long)r.badFrags, pass ? "PASS" : "FAIL");
            if (!pass) rc = 3;
        }
    } else if (mode == WG2X2) {
        // ===== MAD-305 Step 2.3: per-wave 2x2 tile (64x64 claimed) -> ~80 VGPR -> ~16 waves/SIMD static.
        //   Decisive occupancy/ceiling test: (1) oracle 2x2 BLADDER bit-exact, (2) perf 2x2 BLADDER fed,
        //   (3) NOFEED@2x2 vs NOFEED@4x4 -- does the 104 TF compute ceiling RISE with occupancy? =====
        const uint32_t LDS2 = 2052u, VF2 = 12u;     // 2x2: As[64*32]=2048 + ti; ~96 VGPR reservation
        printf("\n=== MAD-305 Step 2.3: 2x2 per-wave tile + NOFEED@2x2 (occupancy vs WMMA-schedule ceiling) ===\n");
        printf("    307 TF ceiling; G2 bar 161 TF; prior NOFEED@4x4 = 104 TF @ ~6 waves/SIMD.\n");
        // ---- (1) oracle: 2x2 BLADDER must stay bit-exact ----
        printf("  -- oracle (2x2 BLADDER, must be bit-exact) --\n");
        struct Sm { const char* name; int M, N, K; uint32_t nWG; };
        Sm ocases[] = {
            { "256x256x256   (16 tiles)",        256, 256,  256,  8u },
            { "512x512x2048  (64 tiles, big-K)", 512, 512, 2048, 16u },
        };
        bool oracleOK = true;
        for (auto& c : ocases) {
            WgcResult r = run_wggemm_compute(node, "occ_wggemm2_blad2.bin", c.M, c.N, c.K, c.nWG, true, 2, LDS2, VF2);
            if (!r.ok) { printf("  %-34s INCOMPLETE\n", c.name); rc = 3; oracleOK = false; continue; }
            uint64_t expect = (uint64_t)(c.M/64) * (c.N/64) * 4 * 4;   // tiles * waves * (FM*FN=4) frags
            bool pass = (r.okFrags == expect && r.badFrags == 0);
            printf("  %-34s maxlive=%u claims=%u  frags OK=%llu/%llu bad=%llu  %s\n",
                   c.name, r.maxlive, r.total, (unsigned long long)r.okFrags,
                   (unsigned long long)expect, (unsigned long long)r.badFrags, pass ? "PASS" : "FAIL");
            if (!pass) { rc = 3; oracleOK = false; }
        }
        if (!oracleOK) { printf("  ORACLE FAILED -- not trusting perf. Fix correctness first.\n"); }
        // ---- (2) perf 2x2 BLADDER fed, (3) NOFEED ceiling pair ----
        printf("  -- perf @4096^2 x K16384 (TARGET) --\n");
        struct Pf { const char* name; const char* bin; int FMt; uint32_t lds, vf; };
        Pf pcases[] = {
            { "2x2 BLADDER fed",   "occ_wggemm2_blad2_perf.bin",   2, LDS2,  VF2  },
            { "NOFEED @ 2x2",      "occ_wggemm2_nofeed2_perf.bin", 2, LDS2,  VF2  },
            { "NOFEED @ 4x4 (ctl)","occ_wggemm2_nofeed4_perf.bin", 4, 8196u, 26u },
        };
        for (auto& c : pcases) {
            WgpResult r = run_wggemm_perf(node, c.bin, 4096, 4096, 16384, 256u, freq_hz, c.FMt, c.lds, c.vf);
            if (!r.ok) { printf("  %-22s INCOMPLETE (hang/timeout)\n", c.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0, wpc = 15.9*r.tf/307.0;
            printf("  %-22s %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs  claims=%u  acc00 OK=%u/%u\n",
                   c.name, r.tf, pct, wpc, r.maxlive, r.total, r.okSamp, r.okSamp + r.badSamp);
        }
    } else if (mode == NFUNROLL) {
        // ===== MAD-305 Step 2.4: NOFEED@KUNROLL sweep (4x4). Does the 98 TF ceiling climb with longer
        //   back-to-back WMMA runs (-> backedge/issue-density bound) or stay flat (-> the 32-WMMA run is
        //   itself sub-peak vs HIP, diff against /tmp/cg/winner_hotloop_4x4.s)? =====
        printf("\n=== MAD-305 Step 2.4: NOFEED@KUNROLL sweep (4x4, 4096^2 x K16384) ===\n");
        printf("    307 TF ceiling (15.9 WMMA/cyc); NOFEED@U1 baseline ~98 TF; HIP NOFEED ~272.\n");
        struct U { int u; const char* bin; };
        U us[] = {
            { 1, "occ_wggemm2_nofeed4_u1.bin" },
            { 2, "occ_wggemm2_nofeed4_u2.bin" },
            { 4, "occ_wggemm2_nofeed4_u4.bin" },
            { 8, "occ_wggemm2_nofeed4_u8.bin" },
        };
        for (auto& c : us) {
            WgpResult r = run_wggemm_perf(node, c.bin, 4096, 4096, 16384, 256u, freq_hz, 4, 8196u, 26u);
            if (!r.ok) { printf("  U=%-2d  INCOMPLETE (hang/timeout)\n", c.u); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0, wpc = 15.9*r.tf/307.0;
            printf("  KUNROLL=%-2d (%3d WMMA/backedge)  %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs\n",
                   c.u, c.u*32, r.tf, pct, wpc, r.maxlive);
        }
    } else if (mode == NFOCC) {
        // ===== MAD-305 Step 2.4b: NOFEED@4x4 OCCUPANCY sweep at the SAME tile (u8, longest runs).
        //   The WMMA stream is structurally identical to HIP yet caps at ~100 TF. Does pushing resident
        //   WGs (tighter VGPR field, smaller LDS request, bigger grid) lift the SAME-tile ceiling? If yes,
        //   the 4x4 ceiling is occupancy/latency-hiding bound (not stream-malformed, not run-length). =====
        printf("\n=== MAD-305 Step 2.4b: NOFEED@4x4 occupancy sweep (same tile, u8, 4096^2 x K16384) ===\n");
        printf("    baseline (field26/lds8196/256WG) was ~100 TF @ maxlive 192. NOFEED needs only 4100B LDS / 192 VGPR.\n");
        struct Oc { const char* name; uint32_t vf, lds, nWG; };
        Oc ocs[] = {
            { "field26 lds8196 256WG (base)", 26u, 8196u,  256u },
            { "field24 lds4100 256WG",        24u, 4100u,  256u },
            { "field24 lds4100 512WG",        24u, 4100u,  512u },
            { "field24 lds4100 1024WG",       24u, 4100u, 1024u },
        };
        for (auto& c : ocs) {
            WgpResult r = run_wggemm_perf(node, "occ_wggemm2_nofeed4_u8.bin", 4096, 4096, 16384, c.nWG, freq_hz, 4, c.lds, c.vf);
            if (!r.ok) { printf("  %-30s INCOMPLETE\n", c.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0, wpc = 15.9*r.tf/307.0;
            printf("  %-30s %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs\n",
                   c.name, r.tf, pct, wpc, r.maxlive);
        }
    } else if (mode == NFBF) {
        // ===== MAD-305 Step 2.4c: BARRIER-FREE NOFEED residency probe (GPT plan steps 1+2).
        //   Same 4x4 WMMA stream, NO LDS / NO barriers, per-wave independent claim. 128-thread WG = the
        //   4-wave probe (does removing barriers/LDS lift the 192-WG wall?); 32-thread WG = one-wave clone
        //   (can PM4 itself reach HIP-like NOFEED when WGs aren't 128-thread/barrier-shaped?). =====
        printf("\n=== MAD-305 Step 2.4c: barrier-free NOFEED residency probe (4096^2 x K16384) ===\n");
        printf("    307 TF ceiling; HIP 4x4 NOFEED 284-289; barrier+LDS 4-wave wall was 192 WGs / ~99 TF.\n");
        struct B { const char* name; const char* bin; uint32_t threads, nWG; };
        B bs[] = {
            // chunk=1 (per-tile atomic): throughput falls as resident waves rise -> atomic contention?
            { "1w c1  nWG=256",  "occ_wgnofeed_bf.bin",     32u,  256u },
            { "1w c1  nWG=896",  "occ_wgnofeed_bf.bin",     32u,  896u },
            // chunk=16 (1 atomic / 16 tiles): if contention is the cap, high-occupancy throughput recovers
            { "1w c16 nWG=256",  "occ_wgnofeed_bf_c16.bin", 32u,  256u },
            { "1w c16 nWG=896",  "occ_wgnofeed_bf_c16.bin", 32u,  896u },
            { "4w c16 nWG=256",  "occ_wgnofeed_bf_c16.bin",128u,  256u },
        };
        for (auto& c : bs) {
            BfResult r = run_nofeed_bf(node, c.bin, 4096, 4096, 16384, c.threads, c.nWG, freq_hz);
            if (!r.ok) { printf("  %-18s INCOMPLETE\n", c.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0, wpc = 15.9*r.tf/307.0;
            printf("  %-18s %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs  claims=%u\n",
                   c.name, r.tf, pct, wpc, r.maxlive, r.claims);
        }
    } else if (mode == BANDSWP) {
        // ===== MAD-305 Step 2.5: BAND-CLAIM sweep on the real 4x4 GEMM. One atomic per BAND tiles.
        //   NOFEED: does the ~99 TF ceiling rise toward HIP 285?  FED (BLADDER): does ~1.3 TF lift? =====
        printf("\n=== MAD-305 Step 2.5: band-claim sweep (4x4, 4096^2 x K16384, TOTAL=1024 tiles) ===\n");
        printf("    per-tile-atomic baselines: NOFEED ~99 TF, FED(BLADDER) ~1.3 TF. HIP 4x4 NOFEED 284-289.\n");
        struct Bn { const char* name; const char* bin; };
        printf("  -- NOFEED (compute ceiling vs atomic granularity) --\n");
        Bn nf[] = {
            { "NOFEED band=1",  "occ_wggemm2_nf_b1.bin"  },
            { "NOFEED band=2",  "occ_wggemm2_nf_b2.bin"  },
            { "NOFEED band=4",  "occ_wggemm2_nf_b4.bin"  },
            { "NOFEED band=8",  "occ_wggemm2_nf_b8.bin"  },
            { "NOFEED band=16", "occ_wggemm2_nf_b16.bin" },
            { "NOFEED band=32", "occ_wggemm2_nf_b32.bin" },
        };
        for (auto& c : nf) {
            WgpResult r = run_wggemm_perf(node, c.bin, 4096, 4096, 16384, 256u, freq_hz, 4, 8196u, 26u);
            if (!r.ok) { printf("  %-16s INCOMPLETE\n", c.name); rc = 3; continue; }
            printf("  %-16s %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs  claims=%u\n",
                   c.name, r.tf, 100.0*r.tf/307.0, 15.9*r.tf/307.0, r.maxlive, r.total);
        }
        printf("  -- FED (BLADDER, real feed) --\n");
        Bn fd[] = {
            { "FED band=1",  "occ_wggemm2_fd_b1.bin"  },
            { "FED band=4",  "occ_wggemm2_fd_b4.bin"  },
            { "FED band=16", "occ_wggemm2_fd_b16.bin" },
        };
        for (auto& c : fd) {
            WgpResult r = run_wggemm_perf(node, c.bin, 4096, 4096, 16384, 256u, freq_hz, 4, 8196u, 26u);
            if (!r.ok) { printf("  %-16s INCOMPLETE\n", c.name); rc = 3; continue; }
            printf("  %-16s %6.1f TF  %4.1f%%  %.2f WMMA/cyc  maxlive=%u WGs  claims=%u  acc00 OK=%u/%u\n",
                   c.name, r.tf, 100.0*r.tf/307.0, 15.9*r.tf/307.0, r.maxlive, r.total, r.okSamp, r.okSamp+r.badSamp);
        }
    } else if (mode == FEEDPIPE) {
        // ===== MAD-305 Step A: FEED-ONLY depth-P pipeline bandwidth probe. No WMMA/LDS/barrier.
        //   Keeps PDEPTH slices of FRAGS b64 loads in flight; measures effective feed GB/s.
        //   P=1 reproduces the proven ~2.7 GB/s serialized baseline; does it scale with P? =====
        printf("\n=== MAD-305 Step A: feed-only depth-P pipeline bandwidth probe (32-thread waves, 64 MiB stream) ===\n");
        printf("    proven serialized FED feed = ~2.7 GB/s (<0.5%% of ~640 GB/s). Does keeping P slices in flight scale BW?\n");
        const uint32_t CEIL = 393216u;   // total slices (mult. of CLAIMCHUNK=256); 805 MB @F=8, 402 MB @F=4
        struct Fp { const char* name; const char* bin; uint32_t frags, P; };
        Fp fps[] = {
            { "P=1  F=8 (baseline)", "occ_feedpipe_p1_f8.bin",  8u,  1u },
            { "P=2  F=8",           "occ_feedpipe_p2_f8.bin",  8u,  2u },
            { "P=4  F=8",           "occ_feedpipe_p4_f8.bin",  8u,  4u },
            { "P=8  F=8",           "occ_feedpipe_p8_f8.bin",  8u,  8u },
            { "P=16 F=4",           "occ_feedpipe_p16_f4.bin", 4u, 16u },
        };
        for (auto& c : fps) {
            uint32_t field = (32u + c.P*c.frags*2u + 7u) / 8u;
            FpResult r = run_feedpipe(node, c.bin, 32u, 1024u, CEIL, c.frags, field, freq_hz);
            if (!r.ok) { printf("  %-20s INCOMPLETE\n", c.name); rc = 3; continue; }
            printf("  %-20s %8.1f GB/s  (%5.1fx baseline)  maxlive=%u waves  claims=%u\n",
                   c.name, r.gbps, r.gbps/2.7, r.maxlive, r.claims);
        }
    } else if (mode == FEEDLADDER) {
        // ===== MAD-305 Step A localization ladder: add 4-wave / barrier / LDS-A-share couplings back onto
        //   the 123 GB/s baseline ONE AT A TIME (GPT rungs 1-5). Find which rung collapses 123 -> ~2.7. =====
        printf("\n=== MAD-305 Step A ladder: localize the FED collapse (1024 launched waves, NSLICES=4096, FRAGS=8) ===\n");
        printf("    rung1 = re-baseline of feedpipe P=1 (~123 GB/s). Watch which coupling collapses it toward 2.7 GB/s.\n");
        struct R { const char* name; const char* bin; uint32_t threads, nWG, field, lds; };
        R rs[] = {
            { "r1 1w  none           ", "occ_feedladder_r1.bin", 32u, 1024u, 6u,    0u },
            { "r2 4w  none           ", "occ_feedladder_r2.bin",128u,  256u, 6u,    0u },
            { "r3 4w  +barrier/slice  ","occ_feedladder_r3.bin",128u,  256u, 6u,    0u },
            { "r4 4w  +LDS roundtrip  ","occ_feedladder_r4.bin",128u,  256u,10u, 8192u },
            { "r5 4w  +globalA->LDS    ","occ_feedladder_r5.bin",128u, 256u,10u, 8192u },
        };
        for (auto& c : rs) {
            FpResult r = run_feedladder(node, c.bin, c.threads, c.nWG, 4096u, 8u, c.field, c.lds, freq_hz);
            if (!r.ok) { printf("  %-24s INCOMPLETE\n", c.name); rc = 3; continue; }
            printf("  %-24s %8.1f GB/s  (%5.1fx baseline 2.7)  maxlive=%u waves\n",
                   c.name, r.gbps, r.gbps/2.7, r.maxlive);
        }
    } else if (mode == FEEDBTR) {
        // ===== MAD-305 Step A rung 6: is B's global_load_tr_b64 the FED wall? (GPT 6a-6d) =====
        printf("\n=== MAD-305 rung 6: B global_load_tr_b64 (1024 launched waves, NSLICES=2048, 8 frags/slice, nt256=65536) ===\n");
        printf("    ladder peers all ran 1000+ GB/s. Does the transpose load and/or real Bshuf address pattern collapse it -> ~2.7?\n");
        struct R { const char* name; const char* bin; uint32_t field, lds; };
        R rs[] = {
            { "6a tr  synthetic stride ", "occ_btr_6a.bin",  6u,    0u },
            { "6b tr  real Bshuf addr  ", "occ_btr_6b.bin",  6u,    0u },
            { "6c tr  Bshuf +real resid","occ_btr_6b.bin",  24u, 8192u },  // 192 VGPR + 8 KB LDS -> ~real residency
            { "6d ld  Bshuf (neg ctrl) ", "occ_btr_6d.bin",  6u,    0u },
        };
        for (auto& c : rs) {
            FpResult r = run_btr(node, c.bin, 128u, 256u, 512u, c.field, c.lds, 65536u, freq_hz);
            if (!r.ok) { printf("  %-26s INCOMPLETE\n", c.name); rc = 3; continue; }
            printf("  %-26s %8.1f GB/s  (%6.1fx baseline 2.7)  maxlive=%u waves\n",
                   c.name, r.gbps, r.gbps/2.7, r.maxlive);
        }
    } else if (mode == BTR128) {
        // ===== MAD-305 Lever A micro-oracle (GPT step 1): prove fp8 fragment semantics of global_load_tr_b128 =====
        rc = run_btr128_oracle(node, "occ_btr128.bin");
    } else if (mode == ANOLDS) {
        // ===== MAD-305 LDS-FREE A: each wave loads its own A frags straight from global (no LDS publish, no
        //   barriers, no A-tile LDS). A staged-in-LDS optimizes BANDWIDTH (cross-wave reuse) but we are
        //   ISSUE-bound -> it's overhead for the wrong bottleneck. THE decisive test: instruction ledger
        //   predicts ~169 TF (58 instr/slice = 55.2% WMMA). measured >> 169 => the barriers / LDS-occupancy
        //   were a hidden wall (the big chunk); ~169 => pure issue-count (no big chunk, 52% ecosystem ceiling). =====
        printf("\n=== MAD-305 LDS-FREE A vs KWIN=4 pw4 baseline @65536^2 x K16384 (card pinned) ===\n");
        printf("    ledger: ANOLDS 58 instr/slice = 55.2%% WMMA -> predicted ~169 TF (baseline 66.25 = 48.3%% -> 145).\n");
        printf("    >>169 => barrier/occupancy was a hidden wall (BIG CHUNK); ~169 => pure issue-count (no big chunk).\n");
        // (1) full-fragment correctness gate (every frag of every tile vs chained wmma_ref). vgprField 26 (match
        //     baseline) so the ONLY variable vs baseline is LDS (512 vs 16388) + barriers.
        WgcResult o = run_wggemm_compute(node, "occ_wggemm2_anolds_st1.bin", 512, 512, 2048, 8u, true, 4, 512u, 26u);
        uint64_t expect = (uint64_t)(512/128)*(512/128)*4*16;
        bool opass = o.ok && o.okFrags == expect && o.badFrags == 0;
        printf("    [oracle] LDS-free full-frag 512x512x2048: OK=%llu/%llu bad=%llu  %s\n",
               (unsigned long long)o.okFrags, (unsigned long long)expect, (unsigned long long)o.badFrags,
               opass ? "PASS" : "*** FAIL -> perf aborted");
        if (!opass) { rc = 3; }
        else {
            printf("    %-26s %8s %7s %7s %8s %s\n", "config","TF","%307","fill%","span_ms","resW/correct");
            struct A { const char* name; const char* bin; uint32_t lds; bool nofeed; bool correct; };
            A rows[] = {
                { "128x128 NOFEED ceil",    "occ_wggemm2_nofeed4_perf.bin", 8196u, true,  false },
                { "128x128 FED baseline pw4","occ_wggemm2_kwin4_pw4.bin",   16388u,false, true  },
                { "128x128 LDS-FREE A (strided)","occ_wggemm2_anolds_perf.bin",512u,false, true  },
                { "128x128 LDS-FREE A COALESCED-diag","occ_wggemm2_anolds_coal_perf.bin",512u,false,false }, // wrong data; isolates the strided/coalescing cost
            };
            const int M=65536, N=65536, Ks=16384;   // saturated -> card pins 2350; deltas real
            double nf = 0;
            for (auto& a : rows) {
                WgpResult r = run_wggemm_perf(node, a.bin, M, N, Ks, 256u, freq_hz, 4, a.lds, 26u, 2);
                if (!r.ok) { printf("    %-26s INCOMPLETE (hang/timeout)\n", a.name); rc = 3; continue; }
                if (a.nofeed) nf = r.tf;
                double fill = nf>0 ? 100.0*r.tf/nf : 0.0;
                const char* corr = a.correct ? (r.badSamp ? "*** acc00 BAD" : "acc00 OK") : "(ceiling)";
                printf("    %-26s %8.1f %6.1f%% %6.1f%% %8.1f  resW=%u %s\n",
                       a.name, r.tf, 100.0*r.tf/307.0, fill, (double)r.wall/freq_hz*1e3, r.maxlive*4u, corr);
                if (a.correct && r.badSamp) rc = 3;
            }
            printf("    [resW = resident waves = maxlive_wg * 4 (4 waves/WG). ANOLDS frees 16KB A-tile LDS -> watch resW.]\n");
        }
    } else if (mode == ANOLDSTR) {
        // ===== MAD-305 LDS-FREE A via global_load_tr (COALESCED) -- THE FIX for ANOLDS's strided 61 TF. A fed
        //   exactly like B from an A-shuf. Coalescing diag PROVED ~152 is reachable; this is the real, correct
        //   kernel. Predicted ~158 (62 instr/slice). Full-frag oracle (Ashuf) first, then baseline vs ANOLDSTR. =====
        printf("\n=== MAD-305 LDS-FREE A via global_load_tr (coalesced A-shuf) vs KWIN=4 pw4 @65536^2 x K16384 ===\n");
        printf("    coalescing diag proved ~152 reachable; ledger 62 instr/slice = 51.6%% -> ~158 TF (baseline 145).\n");
        WgcResult o = run_wggemm_compute(node, "occ_wggemm2_anoldstr_st1.bin", 512, 512, 2048, 8u, true, 4, 512u, 26u, /*useAtr*/1);
        uint64_t expect = (uint64_t)(512/128)*(512/128)*4*16;
        bool opass = o.ok && o.okFrags == expect && o.badFrags == 0;
        printf("    [oracle] A-tr full-frag 512x512x2048: OK=%llu/%llu bad=%llu  %s\n",
               (unsigned long long)o.okFrags, (unsigned long long)expect, (unsigned long long)o.badFrags,
               opass ? "PASS" : "*** FAIL -> perf aborted (debug A-shuf/addressing)");
        if (!opass) { rc = 3; }
        else {
            printf("    %-26s %8s %7s %7s %8s %s\n", "config","TF","%307","fill%","span_ms","resW/correct");
            struct A { const char* name; const char* bin; uint32_t lds; bool nofeed; bool correct; int atr; };
            A rows[] = {
                { "128x128 NOFEED ceil",    "occ_wggemm2_nofeed4_perf.bin", 8196u, true,  false, 0 },
                { "128x128 FED baseline pw4","occ_wggemm2_kwin4_pw4.bin",   16388u,false, true,  0 },
                { "baseline FEED-ONLY (no WMMA)","occ_wggemm2_kwin4_pw4_feedonly.bin",16388u,false,false,0 }, // localize baseline wall: feed-bound vs compute/issue-bound
                { "baseline FEED-ONLY NO-BARRIER","occ_wggemm2_kwin4_pw4_feedonly_nobar.bin",16388u,false,false,0 }, // isolate barrier cost in the feed
                { "128x128 LDS-FREE A (tr)", "occ_wggemm2_anoldstr_perf.bin",  512u,false, true,  1 },
                { "ANOLDSTR FEED-ONLY (no WMMA)","occ_wggemm2_anoldstr_feedonly.bin",512u,false,false,1 }, // 16 tr/slice, WMMA stripped -> feed-throughput probe
            };
            const int M=65536, N=65536, Ks=16384;
            double nf = 0;
            for (auto& a : rows) {
                WgpResult r = run_wggemm_perf(node, a.bin, M, N, Ks, 256u, freq_hz, 4, a.lds, 26u, 2, a.atr);
                if (!r.ok) { printf("    %-26s INCOMPLETE (hang/timeout)\n", a.name); rc = 3; continue; }
                if (a.nofeed) nf = r.tf;
                double fill = nf>0 ? 100.0*r.tf/nf : 0.0;
                const char* corr = a.correct ? (r.badSamp ? "*** acc00 BAD" : "acc00 OK") : "(ceiling)";
                printf("    %-26s %8.1f %6.1f%% %6.1f%% %8.1f  resW=%u %s\n",
                       a.name, r.tf, 100.0*r.tf/307.0, fill, (double)r.wall/freq_hz*1e3, r.maxlive*4u, corr);
                if (a.correct && r.badSamp) rc = 3;
            }
            printf("    [if ANOLDSTR ~150-158 & acc00 OK -> LDS-free A WINS (no LDS, no barriers, coalesced); beats HIP path.]\n");
        }
    } else if (mode == OCCSWEEP) {
        // ===== MAD-305 OCCUPANCY / "throw more waves" saturation sweep: crank nWG hard on the best geometries,
        //   watch TF + resident waves. Does it keep climbing or flatten? (the saturation proof, not a guess). =====
        printf("\n=== MAD-305 OCCUPANCY SWEEP @65536^2 x K16384 -- crank nWG, watch TF + resident waves ===\n");
        printf("    %-14s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; int twn; };
        G geoms[] = {
            { "128x256 TWN=4", "occ_wggemm2_tw4_kwin4_pw4.bin", 4 },
            { "128x128 TWN=2", "occ_wggemm2_kwin4_pw4.bin",     2 },
        };
        uint32_t nwgs[] = { 128u, 256u, 512u, 1024u, 2048u, 4096u };
        for (auto& g : geoms) {
            printf("  --- %s (%d waves/WG) ---\n", g.name, 2*g.twn);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 4, 16388u, 26u, g.twn);
                if (!r.ok) { printf("    %-14s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-14s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*(uint32_t)(2*g.twn),
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** BAD":"OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [residWv = maxlive_WG * waves/WG. If TF flattens while nWG climbs -> occupancy-capped (resident wall),\n");
        printf("     not under-fed -> the lever past it is MORE RESIDENT waves (leaner VGPR / dyn-VGPR feed phase), not more nWG.]\n");
    } else if (mode == LEAN) {
        // ===== MAD-305 L4: LEAN single-wave register-blocked GEMM. TWM=1 TWN=1 (1 wave/WG = the SAFE regime: no
        //   co-residency barrier, cannot deadlock), register-blocked FM x FN tile, direct-global ANOLDSTR feed (no
        //   LDS A-tile, no barriers). A/B frags packed tight just past the accumulators -> VGPR alloc tracks the
        //   lean tile. Persistent: nWG single-wave WGs claim output tiles. Sweeps (FM,FN) x nWG to DECOUPLE
        //   occupancy (nWG -> residWv) from B-reuse (FM,FN) -- the high-occupancy-WITH-strong-reuse corner the
        //   4-wave confound (192 resident but ~149 TF) never measured. Win: residWv >> 64 AND TF > 165.7. =====
        printf("\n=== MAD-305 L4 LEAN single-wave register-blocked GEMM (TWM=1 TWN=1, ANOLDSTR direct-global feed) ===\n");
        printf("    winner ref: 8x2 8-wave WG = 165.7 TF @ ~64 resident waves; NOFEED ceiling ~282. closing that gap is the goal.\n");
        struct LC { const char* tag; int fm, fn; uint32_t vgpr, vgpr_bpf; };  // vgpr=naive ceil((32+8fmfn+4fm+4fn)/8); bpf adds 2nd feed buffer
        LC cfgs[] = {
            { "2x2", 2, 2, 10u, 12u },   // naive 80 / bpf 96
            { "4x2", 4, 2, 15u, 18u },   // naive 120 / bpf 144
            { "2x4", 2, 4, 15u, 18u },   // naive 120 / bpf 144
            { "4x4", 4, 4, 24u, 28u },   // naive 192 / bpf 224
            { "8x2", 8, 2, 25u, 30u },   // naive 200 / bpf 240
        };
        struct Var { const char* name; const char* prefix; bool bpf; };   // naive lean vs LEANBPF (prefetch-pipelined)
        Var vars[] = { {"naive", "occ_lean_", false}, {"bpf", "occ_leanbpf_", true} };
        // ---- oracle gate @512^3 (EVERY frag of the single wave vs chained CPU wmma_ref) BEFORE any perf; naive + LEANBPF ----
        printf("  --- oracle gate @512x512x512 (all FM*FN frags vs chained wmma_ref; naive + LEANBPF) ---\n");
        bool allpass = true;
        for (auto& c : cfgs) {
            for (auto& v : vars) {
                bool bpf = v.bpf;
                char bin[80]; snprintf(bin, sizeof bin, "%s%s_st1.bin", v.prefix, c.tag);
                uint32_t vg = bpf ? c.vgpr_bpf : c.vgpr;
                WgcResult r = run_wggemm_compute(node, bin, 512,512,512, 64u, true, c.fm, 512u, vg,
                                                 /*useAtr*/1, /*TWN*/1, /*FNt*/c.fn, /*TWMt*/1);
                bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
                printf("    oracle %-5s lean_%-4s okFrags=%u badFrags=%u  %s\n", v.name, c.tag, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
                if (!pass) { allpass = false; rc = 3; }
            }
        }
        if (!allpass) { printf("    *** oracle FAIL -> perf aborted (debug lean feed/addressing/register-pack/LEANBPF pipeline)\n"); }
        else {
            // ---- DURATION-GUARDED SHAPE RAMP. BRICK-#2 LESSON: the old cap was POST-HOC -- it checked span AFTER
            //   running the dispatch, so it RAN the ~10s 65536^2 shape (which starved the gfx ring) before "stopping".
            //   A post-hoc check is NOT a guard. Fixed: PREDICT each shape's span from the previous (work ratio) and
            //   SKIP -- never RUN a dispatch projected over the cap. The smallest shape (4096^2 x 2048) is safe-by-
            //   construction (<~100ms even at 1 TF). Compares naive vs LEANBPF (the RGA-diagnosed feed-prefetch fix).
            //   NOTE: still assumes a GPU genuinely ISOLATED from the compositor; a hang is only bounded by the 20s
            //   harness timeout, which is fatal if the desktop shares the device (see the gfx1201 brick KGs). ----
            const double SPAN_CAP_MS = 1200.0;   // never RUN a dispatch PREDICTED to exceed this
            const uint32_t NW = 1024u;
            struct Shape { int M, N, K; };
            Shape shapes[] = { {4096,4096,2048}, {8192,8192,2048}, {16384,16384,4096}, {32768,32768,8192}, {65536,65536,16384} };
            printf("  --- perf RAMP (smallest-first, nWG=%u, PREDICT-AND-SKIP guard @ %.0f ms; naive vs LEANBPF) ---\n", NW, SPAN_CAP_MS);
            printf("    %-6s %-6s %-18s %8s %7s %9s %9s %s\n","cfg","var","shape(MxNxK)","TF","%307","residWv","span_ms","correct");
            for (auto& c : cfgs) {
                for (auto& v : vars) {
                    bool bpf = v.bpf;
                    char bin[80]; snprintf(bin, sizeof bin, "%s%s.bin", v.prefix, c.tag);
                    uint32_t vg = bpf ? c.vgpr_bpf : c.vgpr;
                    double prev_span = 0.0, prev_work = 0.0;
                    for (auto& sh : shapes) {
                        double work = 2.0*(double)sh.M*sh.N*sh.K;
                        char shp[24]; snprintf(shp, sizeof shp, "%dx%dx%d", sh.M, sh.N, sh.K);
                        if (prev_span > 0.0) {                                  // PREDICT before running; skip (and stop) if over cap
                            double pred = prev_span * (work / prev_work);
                            if (pred > SPAN_CAP_MS) {
                                printf("    %-6s %-6s %-18s SKIP (pred ~%.0f ms > %.0f cap; larger shapes also skipped)\n",
                                       c.tag, v.name, shp, pred, SPAN_CAP_MS);
                                break;
                            }
                        }
                        WgpResult r = run_wggemm_perf(node, bin, sh.M,sh.N,sh.K, NW, freq_hz, c.fm, 512u, vg,
                                                      /*TWN*/1, /*useAtr*/1, /*FNt*/c.fn, /*TWMt*/1);
                        if (!r.ok) { printf("    %-6s %-6s %-18s INCOMPLETE -> STOP\n", c.tag, v.name, shp); rc=3; break; }
                        double span_ms = (double)r.wall/freq_hz*1e3;
                        printf("    %-6s %-6s %-18s %8.1f %6.1f%% %9u %9.1f  %s\n",
                               c.tag, v.name, shp, r.tf, 100.0*r.tf/307.0, r.maxlive*1u, span_ms, r.badSamp?"*** BAD":"OK");
                        if (r.badSamp) rc=3;
                        prev_span = span_ms; prev_work = work;
                        if (span_ms > SPAN_CAP_MS) { printf("    [measured %.0f ms > cap -> STOP (predictor undershot)]\n", span_ms); break; }
                    }
                }
            }
            printf("    [Per shape: LEANBPF (bpf) should lift TF vs naive if feed-latency-bound (the RGA diagnosis).\n");
            printf("     TF rises with shape -> compute-bound. residWv >> 64 (single-wave) + TF > 165.7 = L4 beats the winner.]\n");
        }
    } else if (mode == DECOMP) {
        // ===== MAD-305: cooperative 4x4 vs 8x2 BOTTLENECK DECOMPOSITION. For each tile: FED (real GEMM), NOFEED
        //   (WMMA-only compute ceiling), FEEDONLY (feed, no WMMA = feed wall). Reads: FED near NOFEED => compute-
        //   bound/tapped out; FED near FEEDONLY => feed-bound, headroom = NOFEED-FED. Answers "what binds 4x4, and is
        //   it fixable to overtake 8x2". Cooperative ~1s winner-class regime (proven safe). 65536^2 x K16384, nWG=256. =====
        printf("\n=== MAD-305 cooperative 4x4 vs 8x2 BOTTLENECK DECOMPOSITION @65536^2 x K16384 (TWN=4, nWG=256) ===\n");
        printf("    %-14s %8s %7s %9s %9s\n","config","TF","%307","residWv","span_ms");
        struct D { const char* name; const char* bin; int fm, fn; uint32_t lds; };
        D rows[] = {
            { "4x4 FED",      "occ_wggemm2_tw4_kwin4_pw4.bin",             4, 4, 16388u },
            { "4x4 NOFEED",   "occ_wggemm2_tw4_nofeed.bin",                4, 4, 16388u },
            { "4x4 FEEDONLY", "occ_wggemm2_tw4_kwin4_pw4_feedonly.bin",    4, 4, 16388u },
            { "8x2 FED",      "occ_wggemm2_82_tw4_kwin4_pw4.bin",          8, 2, 32772u },
            { "8x2 NOFEED",   "occ_wggemm2_82_tw4_nofeed.bin",             8, 2, 32772u },
            { "8x2 FEEDONLY", "occ_wggemm2_82_tw4_kwin4_pw4_feedonly.bin", 8, 2, 32772u },
        };
        const int M=65536, N=65536, K=16384; const uint32_t NW=256u;
        for (auto& d : rows) {
            WgpResult r = run_wggemm_perf(node, d.bin, M,N,K, NW, freq_hz, d.fm, d.lds, 26u,
                                          /*TWN*/4, /*useAtr*/0, /*FNt*/d.fn, /*TWMt*/2);
            if (!r.ok) { printf("    %-14s INCOMPLETE (hang/timeout)\n", d.name); rc=3; continue; }
            printf("    %-14s %8.1f %6.1f%% %9u %9.1f\n",
                   d.name, r.tf, 100.0*r.tf/307.0, r.maxlive*8u, (double)r.wall/freq_hz*1e3);
        }
        printf("    [FED near NOFEED -> compute-bound (no headroom). FED near FEEDONLY -> feed-bound (headroom = NOFEED-FED).\n");
        printf("     4x4 with a BIG NOFEED->FED gap AND FED~FEEDONLY => feed-bound w/ headroom => a B-bandwidth lever could lift it past 8x2.]\n");
    } else if (mode == REUSE82) {
        // ===== MAD-305 REUSE-TILE B-FEED lever: 8x2 per-wave tile HALVES the binding per-wave B global_load_tr
        //   feed (B-tr/MAC 0.25 -> 0.125; A-LDS-rd 0.25 -> 0.50) vs 4x4. Both @ TWN=4 (8-wave WG, vgprField 26 =
        //   208 VGPR -> OCCUPANCY-MATCHED, max v207 vs v203), so any TF delta isolates the B-tr-feed wall.
        //   If 8x2 >> 4x4: per-wave B-tr feed IS the ~148 wall (new winner). If flat/worse: total-issue / A-rd bound. =====
        printf("\n=== MAD-305 REUSE-TILE 8x2 vs 4x4 @65536^2 x K16384 (TWN=4, occupancy-matched) ===\n");
        printf("    4x4: B-tr/MAC=0.25  A-rd/MAC=0.25 (tot 0.50);  8x2: B-tr/MAC=0.125 A-rd/MAC=0.50 (tot 0.625).\n");
        // ---- full-fragment oracle gate (small size, EVERY frag of every wave vs CPU wmma_ref) BEFORE perf ----
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) ---\n");
        { struct O { const char* name; const char* bin; int fm,fn; uint32_t lds; };
          O ors[] = {
            { "4x4 TWN=4", "occ_wggemm2_tw4_kwin4_st1.bin",    4, 4, 16388u },   // sanity: known-good, validates the TWN/FN-generalized oracle harness
            { "8x2 TWN=4", "occ_wggemm2_82_tw4_kwin4_st1.bin", 8, 2, 32772u },   // THE gate: 8x2 all-frag correctness
          };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, o.fm, o.lds, 26u, 0, 4, o.fn);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-10s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    %-16s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; int fm, fn; uint32_t lds; };
        G geoms[] = {
            { "4x4 TWN=4", "occ_wggemm2_tw4_kwin4_pw4.bin",    4, 4, 16388u },
            { "8x2 TWN=4", "occ_wggemm2_82_tw4_kwin4_pw4.bin", 8, 2, 32772u },
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            printf("  --- %s (8 waves/WG, B-tr/MAC=%.3f) ---\n", g.name, 1.0/(double)g.fm);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, g.fm, g.lds, 26u, 4, 0, g.fn);
                if (!r.ok) { printf("    %-16s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-16s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [acc00 = full-K dot of frag(0,0) per wave, sampled 16 tiles x 8 waves. BAD -> 8x2 addressing wrong, ignore TF.]\n");
    } else if (mode == REUSE82TW2) {
        // ===== MAD-305 8x2 @ TWN=2 RESIDENCY lever: the 162 TF 8x2@TWN4 winner exposed WMMA (FED/FO 0.887) -- feed
        //   ceiling rose to 182 but latency-hiding at 64-WG/512-resident-wave became the new wall. TWN=2 -> 4-wave WGs
        //   (NBANDS=FM/TWN=4 A-fill bands) gives +50% resident waves (768 vs 512) to RE-HIDE that WMMA. Same per-wave
        //   B-tr/MAC=0.125. If TWN2 > 162: residency was the wall (new winner). If flat/worse: WMMA is intrinsic, not
        //   latency -- pivot to publish-width / KWINBPF overlap. 4x4@TWN2 (149.7) + 8x2@TWN4 (162) are the controls. =====
        printf("\n=== MAD-305 8x2 @ TWN=2 (4-wave, NBANDS=4) vs 8x2@TWN4 (162) and 4x4@TWN2 (149.7) ===\n");
        // ---- oracle gate: 8x2@TWN2 NBANDS=4 publish correctness (all FM*FN frags, all 4 waves) BEFORE perf ----
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) ---\n");
        { struct O { const char* name; const char* bin; int fm,fn,twn; uint32_t lds; };
          O ors[] = {
            { "8x2 TWN=2", "occ_wggemm2_82_kwin4_st1.bin", 8, 2, 2, 32772u },   // THE gate: NBANDS=4 A-fill correctness
          };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, o.fm, o.lds, 26u, 0, o.twn, o.fn);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-10s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    %-16s %5s %6s %8s %8s %9s %8s %s\n","geom","waves","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; int fm, fn, twn; uint32_t lds; };
        G geoms[] = {
            { "4x4 TWN=2", "occ_wggemm2_kwin4_pw4.bin",        4, 4, 2, 16388u },
            { "8x2 TWN=4", "occ_wggemm2_82_tw4_kwin4_pw4.bin", 8, 2, 4, 32772u },
            { "8x2 TWN=2", "occ_wggemm2_82_kwin4_pw4.bin",     8, 2, 2, 32772u },
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            int waves = 2*g.twn;
            printf("  --- %s (%d waves/WG, B-tr/MAC=%.3f) ---\n", g.name, waves, 1.0/(double)g.fm);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, g.fm, g.lds, 26u, g.twn, 0, g.fn);
                if (!r.ok) { printf("    %-16s %5d %6u INCOMPLETE\n", g.name, waves, nw); rc=3; continue; }
                printf("    %-16s %5d %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, waves, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*(uint32_t)waves,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [residWv = maxlive*waves. 8x2 TWN=2 target: re-hide the WMMA -> recover toward the 182 FEEDONLY ceiling.]\n");
    } else if (mode == REUSE82KW2) {
        // ===== MAD-305 8x2 KWIN=2 OCCUPANCY lever: today's TWN=2 result proved 8x2 residency is LDS-BOUND at 64 WGs
        //   (LDS = KWIN*ATILE, TWN-invariant). The fix is NOT fewer waves/WG (TWN) but a SMALLER per-WG LDS footprint:
        //   halve the A-ring (KWIN 4->2 -> LDS 32772->16388) -> ~2x WGs -> ~1024 resident waves (vs 512) to re-hide the
        //   WMMA exposed at the 182 FEEDONLY ceiling. KWINPW=2. Tradeoff: 2x A-publish barrier freq. KWIN=4 = control. =====
        printf("\n=== MAD-305 8x2 @ TWN=4 KWIN=2 (half LDS ring -> ~2x residency) vs KWIN=4 winner (162) ===\n");
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) ---\n");
        { struct O { const char* name; const char* bin; int fm,fn; uint32_t lds; };
          O ors[] = {
            { "8x2 KWIN=2", "occ_wggemm2_82_tw4_kwin2_st1.bin", 8, 2, 16388u },   // THE gate: KWIN=2 publish/consume correctness
          };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, o.fm, o.lds, 26u, 0, 4, o.fn);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-10s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    %-16s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; uint32_t lds; };
        G geoms[] = {
            { "8x2 KWIN=4", "occ_wggemm2_82_tw4_kwin4_pw4.bin", 32772u },   // the 162 winner (control)
            { "8x2 KWIN=2", "occ_wggemm2_82_tw4_kwin2_pw2.bin", 16388u },   // half LDS ring
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            printf("  --- %s (8 waves/WG, TWN=4, LDS=%u) ---\n", g.name, g.lds);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, g.lds, 26u, 4, 0, 2);
                if (!r.ok) { printf("    %-16s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-16s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [LDS-residency hypothesis: KWIN=2 should ~2x residWv (512->~1024). If TF rises -> latency-bound confirmed,\n");
        printf("     the 162->182 gap was occupancy. If flat/worse -> 2x barrier cost ate the occupancy win (try KWINNOTAIL).]\n");
    } else if (mode == VGPR82) {
        // ===== MAD-305 8x2 VGPR-RESIDENCY PROBE: is the 64-WG cap VGPR-allocation-bound or structural (tile geometry)?
        //   The 8x2 winner genuinely uses 208 VGPR (v0..v207) so we can't reserve FEWER (would corrupt). Instead sweep
        //   vgprField UPWARD (over-reserve, always safe -- extra regs unused) on the SAME winner bin and watch maxlive:
        //     - maxlive DROPS below 64 at/near field 26 -> VGPR reservation is the binding occ limiter at 208 ->
        //       dyn-VGPR (lean launch, reserve <208 most of the wave's life) WOULD lift residency (2nd payoff).
        //     - maxlive holds 64 well past 26 (VGPR headroom) -> the 64-WG cap is STRUCTURAL (tile shape, not VGPR) ->
        //       dyn-VGPR's value is purely the feed-density lever, not residency.
        //   No kernel rebuild: vgprField is a dispatch-time RSRC1 field. acc00 stays OK (kernel still uses only v0..207). =====
        printf("\n=== MAD-305 8x2 VGPR-residency probe (sweep RSRC1 vgprField UP on the 162 winner, nWG=512) ===\n");
        printf("    8x2 genuinely needs 208 VGPR (v207); field>=26 is over-reservation. Watching where maxlive falls off 64.\n");
        printf("    %6s %8s %9s %9s %9s %8s %s\n","field","~VGPR","maxlvWG","residWv","TF","span_ms","correct");
        const int M=65536, N=65536, Ks=16384; const uint32_t nWG=512u;
        const char* bin = "occ_wggemm2_82_tw4_kwin4_pw4.bin";
        uint32_t fields[] = { 26u, 27u, 28u, 29u, 30u, 32u, 36u, 40u, 48u, 56u, 63u };
        uint32_t base_maxlive = 0;
        for (uint32_t vf : fields) {
            WgpResult r = run_wggemm_perf(node, bin, M,N,Ks, nWG, freq_hz, 8, 32772u, vf, 4, 0, 2);
            if (!r.ok) { printf("    %6u %8u %9s %9s %9s %8s  %s\n", vf, vf*8u, "INCOMPLETE","-","-","-","(VGPR over-reserve too high to launch)"); continue; }
            if (vf == 26u) base_maxlive = r.maxlive;
            const char* tag = "";
            if (base_maxlive && r.maxlive < base_maxlive) tag = "  <- WG cap dropping (VGPR-bound)";
            printf("    %6u %8u %9u %9u %9.1f %8.1f  %s%s\n",
                   vf, vf*8u, r.maxlive, r.maxlive*8u, r.tf,
                   (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK", tag);
            if (r.badSamp) rc=3;
        }
        printf("    [VERDICT: maxlive falls immediately past field 26 -> VGPR-allocation-bound -> dyn-VGPR lifts residency too.\n");
        printf("     maxlive holds at 64 with VGPR headroom -> 64-WG cap is STRUCTURAL (tile geometry); dyn-VGPR = feed-density only.]\n");
    } else if (mode == BLDS82) {
        // ===== MAD-305 B-in-LDS DEDUP on the 8x2 winner: both wave_m of a wave_n load IDENTICAL B columns today
        //   (redundant global_load_tr). BLDS=1 -> wave_m==0 loads B -> LDS B-ring, both wave_m read from LDS ->
        //   HALVES the binding global B-tr feed (B-tr/MAC 0.125->0.0625). Static, no umr, no VGPR change. Cost: a
        //   B-ring in LDS (49156 B total) + a dedup ds_load in the consume. If TF rises -> binding feed re-cut
        //   below the 182 ceiling (composable with dyn-VGPR later). If flat -> we were already WMMA-exposed, not
        //   feed-bound at the margin (then the win needs latency-hiding, not less feed). =====
        printf("\n=== MAD-305 8x2 + B-in-LDS dedup (halve binding B-tr feed) vs 8x2 winner (162) ===\n");
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) ---\n");
        { struct O { const char* name; const char* bin; int fm,fn; uint32_t lds; };
          O ors[] = {
            { "8x2 BLDS", "occ_wggemm2_82_tw4_kwin4_blds_st1.bin", 8, 2, 49156u },   // THE gate: B-in-LDS dedup correctness
          };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, o.fm, o.lds, 26u, 0, 4, o.fn);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-10s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    %-16s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; uint32_t lds; double btr; };
        G geoms[] = {
            { "8x2 winner",  "occ_wggemm2_82_tw4_kwin4_pw4.bin",  32772u, 0.125 },
            { "8x2 B-in-LDS", "occ_wggemm2_82_tw4_kwin4_blds.bin", 49156u, 0.0625 },
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            printf("  --- %s (8 waves/WG, B-tr/MAC=%.4f) ---\n", g.name, g.btr);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, g.lds, 26u, 4, 0, 2);
                if (!r.ok) { printf("    %-16s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-16s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [B-in-LDS halves the binding global B-tr feed. >162 -> feed re-cut wins (stacks with dyn later);\n");
        printf("     flat -> margin is WMMA-exposure not feed (the B-ring ds_load + dedup barrier ate the feed save).]\n");
    } else if (mode == BPF82) {
        // ===== MAD-305 KWINBPF on 8x2 = the RDNA equivalent of CDNA rung-7 DOUBLE-BUFFERING: prefetch next slice's
        //   B (the binding feed) into the other of 2 ping-pong slots while WMMA runs on the current slice. Overlaps
        //   B-load latency behind compute -> directly attacks the 162->182 WMMA-exposure gap. No new instrs, no extra
        //   barrier (descending s_wait_loadcnt 8 keeps next-slice B in flight). >162 -> latency-hiding wins. =====
        printf("\n=== MAD-305 8x2 + KWINBPF B-prefetch (CDNA rung-7 double-buffer equiv) vs 8x2 winner (162) ===\n");
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) ---\n");
        { struct O { const char* name; const char* bin; int fm,fn; uint32_t lds; };
          O ors[] = {
            { "4x4 BPF", "occ_wggemm2_tw4_kwin4_bpf_st1.bin",    4, 4, 16388u },   // gate: 4x4 KWINBPF correctness
            { "8x2 BPF", "occ_wggemm2_82_tw4_kwin4_bpf_st1.bin", 8, 2, 32772u },   // gate: 8x2 symbolized ping-pong slots correctness
          };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, o.fm, o.lds, 26u, 0, 4, o.fn);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-10s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    [TILE x LEVER GRID: does the B-prefetch double-buffer help a higher-B-feed tile (4x4, B-tr 0.25) MORE\n");
        printf("     than the winner (8x2, B-tr 0.125)? -> reveals whether a different tile could overtake under the lever.]\n");
        printf("    %-16s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; int fm,fn; uint32_t lds; double btr; };
        G geoms[] = {
            { "4x4 base",    "occ_wggemm2_tw4_kwin4_pw4.bin",    4, 4, 16388u, 0.25  },
            { "4x4 KWINBPF", "occ_wggemm2_tw4_kwin4_bpf.bin",    4, 4, 16388u, 0.25  },
            { "8x2 base",    "occ_wggemm2_82_tw4_kwin4_pw4.bin", 8, 2, 32772u, 0.125 },
            { "8x2 KWINBPF", "occ_wggemm2_82_tw4_kwin4_bpf.bin", 8, 2, 32772u, 0.125 },
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            printf("  --- %s (8 waves/WG, B-tr/MAC=%.3f) ---\n", g.name, g.btr);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, g.fm, g.lds, 26u, 4, 0, g.fn);
                if (!r.ok) { printf("    %-16s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-16s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [KWINBPF overlaps next-slice B-load behind current WMMA. >162 -> latency-hiding wins (CDNA rung-7 lands);\n");
        printf("     flat -> B-load already hidden by KWIN amortization, the exposure is intrinsic WMMA result-latency.]\n");
    } else if (mode == SP82) {
        // ===== MAD-305 s_setprio scheduling on the 165 winner (KWINBPF) = CDNA rung-9 gap-filler equivalent: s_setprio 1
        //   around the per-slice WMMA burst, s_setprio 0 during feed -> bias the shared issue port toward WMMA-phase
        //   waves so they issue dense back-to-back while feed-phase waves yield. STACKS on the double-buffer. =====
        printf("\n=== MAD-305 ml8-shape TILE SWEEP (wggemm2 KWIN=4 KWINBPF; winner 256x128 + smaller tiles) ===\n");
        // ml8-shape re-targeting: override M/N/K via env (WG_M/WG_N/WG_K) to bench REAL ml8 FFN shapes vs the
        //   square 65536^2 default. Smaller tiles -> more tiles -> fill the 64 CUs at the small real shapes.
        //   Constraints: N = pow2 * (TWN*FN*16);  M % (TWM*FM*16) == 0;  K % (32*KWIN) == 0.
        //   per-tile lds = (TWM*FM*16)*32*KWIN + 4 ; vf = ceil(RGA-livereg-peak / 8). (exact N=9216/2560 needs
        //   the pow2->div/mod NTL decode generalization; N=8192/2048 are pow2 proxies.)
        const int M  = getenv("WG_M") ? atoi(getenv("WG_M")) : 65536;
        const int N  = getenv("WG_N") ? atoi(getenv("WG_N")) : 65536;
        const int Ks = getenv("WG_K") ? atoi(getenv("WG_K")) : 16384;
        printf("    [shape: M=%d N=%d K=%d]\n", M, N, Ks);
        struct T { const char* name; const char* pbin; const char* obin;
                   int FMt, FNt, TWN, TWMt; uint32_t lds, vf; };
        T tiles[] = {
            { "82_tw4 256x128(164.9)", "occ_wggemm2_82_tw4_kwin4_bpf_gd.bin", "occ_wggemm2_82_tw4_kwin4_bpf_gd_st1.bin", 8,2,4,2, 32772u, 26u },
            // LEAN vf after the publish-register relocation fix (PUBW slots now tile-relative = FA+{0,8,16,24}):
            //   lean tiles drop from the forced 208 VGPR to their true footprint. vf = ceil((max_vgpr_idx+1)/8):
            //   42_tw4/42_tw2 idx127->16, 24_tw2 idx135->17, 22_tw2 idx91->12. LDS=KWIN*ATILE+4 unchanged.
            //   GENDIV bins (_gd infix): magic-reciprocal NTL decode -> exact non-pow2 NTL (N=9216/2560).
            { "42_tw4 128x128",        "occ_wggemm2_42_tw4_kwin4_bpf_gd.bin", "occ_wggemm2_42_tw4_kwin4_bpf_gd_st1.bin", 4,2,4,2, 16388u, 16u },
            { "42_tw2 128x64",         "occ_wggemm2_42_tw2_kwin4_bpf_gd.bin", "occ_wggemm2_42_tw2_kwin4_bpf_gd_st1.bin", 4,2,2,2, 16388u, 16u },
            { "24_tw2 64x128",         "occ_wggemm2_24_tw2_kwin4_bpf_gd.bin", "occ_wggemm2_24_tw2_kwin4_bpf_gd_st1.bin", 2,4,2,2,  8196u, 17u },
            { "22_tw2 64x64",          "occ_wggemm2_22_tw2_kwin4_bpf_gd.bin", "occ_wggemm2_22_tw2_kwin4_bpf_gd_st1.bin", 2,2,2,2,  8196u, 12u },
            // lever probe (option 4): s_setprio on the two real-dim winners (same vf/lds; just the scheduling hint)
            { "82_sp 256x128+setpr",   "occ_wggemm2_82_tw4_kwin4_bpf_sp_gd.bin", "occ_wggemm2_82_tw4_kwin4_bpf_sp_gd_st1.bin", 8,2,4,2, 32772u, 26u },
            { "42_sp 128x128+setpr",   "occ_wggemm2_42_tw4_kwin4_bpf_sp_gd.bin", "occ_wggemm2_42_tw4_kwin4_bpf_sp_gd_st1.bin", 4,2,4,2, 16388u, 16u },
            // lever probe (option 4): B-in-LDS (BLDS=1, alternative to KWINBPF) -- B staged in LDS, M-dedup feed.
            //   lds = KWIN*ATILE + KWIN*BTILE + 4 (82: 32768+16384+4=49156; 42: 16384+16384+4=32772). vf = PUB_TOP/8.
            { "82_blds 256x128 B-LDS", "occ_wggemm2_82_tw4_kwin4_blds_gd.bin", "occ_wggemm2_82_tw4_kwin4_blds_gd_st1.bin", 8,2,4,2, 49156u, 24u },
            { "42_blds 128x128 B-LDS", "occ_wggemm2_42_tw4_kwin4_blds_gd.bin", "occ_wggemm2_42_tw4_kwin4_blds_gd_st1.bin", 4,2,4,2, 32772u, 16u },
            // CEILING probes (128x128): NOFEED = compute ceiling (load once, pure WMMA); FEEDONLY = feed ceiling
            //   (full feed, zero WMMA). acc00 BAD is EXPECTED (garbage/no output). TF is the ceiling number. oracle bin
            //   reuses base _gd_st1 (the gate is meaningless for these; ignore its result).
            { "42_NOFEED compute-ceil","occ_wggemm2_42_tw4_kwin4_nofeed_gd.bin",   "occ_wggemm2_42_tw4_kwin4_bpf_gd_st1.bin", 4,2,4,2, 16388u, 16u },
            { "42_FEEDONLY feed-ceil", "occ_wggemm2_42_tw4_kwin4_feedonly_gd.bin", "occ_wggemm2_42_tw4_kwin4_bpf_gd_st1.bin", 4,2,4,2, 16388u, 16u },
            // lever A probe: 16-frag (256x128) NOFEED -- does deeper accumulator ILP raise the compute ceiling vs 8-frag 170?
            { "82_NOFEED compute-ceil","occ_wggemm2_82_tw4_kwin4_nofeed_gd.bin",   "occ_wggemm2_82_tw4_kwin4_bpf_gd_st1.bin", 8,2,4,2, 32772u, 26u },
        };
        printf("  --- oracle gate @512x512x512 (per-tile, all frags) ---\n");
        for (auto& t : tiles) {
            WgcResult r = run_wggemm_compute(node, t.obin, 512,512,512, 64u, true, t.FMt, t.lds, t.vf, 0, t.TWN, t.FNt, t.TWMt, 0, 0, 0, /*useGenDiv*/true);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-22s okFrags=%u badFrags=%u  %s\n", t.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
        }
        printf("    %-22s %6s %8s %8s %9s %8s %s\n","tile","nWG","TF","%307","residWv","span_ms","correct");
        // nWG sweep extended DOWNWARD (MAD-305 ml8): at small real shapes TF climbs monotonically as nWG drops
        //   (persistent kernel -> fewer WGs claim more tiles each -> less aggregate prologue/setup overhead).
        //   gfx1201 = 64 CUs, so 32/64/128 = 0.5/1/2 WG-per-CU. WG_NWG env overrides the whole set with one value.
        uint32_t nwgs[] = { 32u, 48u, 64u, 80u, 96u };   // option 2: finer around the 64 (=#CU) train optimum
        for (auto& t : tiles) {
            printf("  --- %s ---\n", t.name);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, t.pbin, M,N,Ks, nw, freq_hz, t.FMt, t.lds, t.vf, t.TWN, 0, t.FNt, t.TWMt, 0, 0, /*useGenDiv*/true);
                if (!r.ok) { printf("    %-22s %6u INCOMPLETE\n", t.name, nw); rc=3; continue; }
                printf("    %-22s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       t.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [smaller tiles -> more tiles -> fill the 64 CUs at small real shapes. winner row = 164.9-basis baseline.]\n");
    } else if (mode == B128MODE) {
        // ===== MAD-305 128-bit B FEED on the 165.7 winner = the ISSUE-SLOT axis via WIDE LOADS. RDNA4 has NO
        //   128-bit transpose for fp8 (tr_b128 is 16-bit only), so the transpose is moved to the CPU preshuffle
        //   (mbg_preshuffle_B128 -> frag-ready, lane-linear 512B blocks). The device then does a PLAIN
        //   global_load_b128 (vaddr=lane*16) delivering 2 K=16 B-frags/instr -> B-feed slots 16->8/K-window
        //   (HALVED, proven in the disasm). CPU-proven byte-identical frag values to the tr_b64 winner. =====
        printf("\n=== MAD-305 8x2 KWINBPF+SETPRIO + 128-bit B feed (plain b128 over frag-ready preshuffle) vs the 165.7 winner ===\n");
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) -- THE correctness gate for the wide B load ---\n");
        { struct O { const char* name; const char* bin; };
          O ors[] = { { "bpf+sp+b128", "occ_wggemm2_82_tw4_kwin4_bpf_sp_b128_st1.bin" } };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, 8, 32772u, 26u, 0, 4, 2, 2, /*useB128*/1);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-12s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
          if (rc==3) { printf("    [B128 oracle FAILED -> NOT running perf. Frag-ready layout or kernel b128 indexing wrong.]\n"); }
        }
        if (rc!=3) {
          printf("    %-20s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
          const int M=65536, N=65536, Ks=16384;
          struct G { const char* name; const char* bin; int b128; };
          G geoms[] = {
              { "bpf+sp (165.7 win)",  "occ_wggemm2_82_tw4_kwin4_bpf_sp.bin",      0 },
              { "bpf+sp + b128 feed",  "occ_wggemm2_82_tw4_kwin4_bpf_sp_b128.bin", 1 },
          };
          uint32_t nwgs[] = { 256u, 512u, 1024u };
          for (auto& g : geoms) {
              printf("  --- %s ---\n", g.name);
              for (uint32_t nw : nwgs) {
                  WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, 32772u, 26u, 4, 0, 2, 2, g.b128);
                  if (!r.ok) { printf("    %-20s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                  printf("    %-20s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                         g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                         (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                  if (r.badSamp) rc=3;
              }
          }
          printf("    [128-bit B load halves B-feed issue slots (16->8/K-window). >165.7 -> the issue-slot model holds and B\n");
          printf("     feed was part of the wall; flat -> B feed wasn't issue-bound (look to LDS A-reads / WMMA-result latency).]\n");
        }
    } else if (mode == TILEORDMODE) {
        // ===== MAD-305 L1: persistent tile-order / B-panel L2 locality. Default claim order is A-stationary
        //   (tile_col = ti & NTL_MASK, N fastest) so the B/N panel changes every tile. N_STATIONARY (TILEORD=1)
        //   makes consecutive ti share tile_col and sweep tile_row -> a B panel stays hot in L2 across the
        //   M-sweep. Pure claim-order change (no WMMA math); correctness is order-invariant (oracle mirrors it).
        //   FED-faster -> B-feed latency was partly L2 miss (cache lever real); flat -> not cache. =====
        printf("\n=== MAD-305 L1: N_STATIONARY tile order (B-panel L2 locality) vs the 165.7 winner ===\n");
        printf("  --- oracle gate @512x512x512 (mirror-decoded) -- THE correctness gate for the claim-order swap ---\n");
        { WgcResult r = run_wggemm_compute(node, "occ_wggemm2_82_tw4_kwin4_bpf_sp_nstat_st1.bin", 512,512,512, 64u, true, 8, 32772u, 26u, 0, 4, 2, 2, /*useB128*/0, /*useTileord*/1);
          bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
          printf("    oracle nstat  okFrags=%llu badFrags=%llu  %s\n", (unsigned long long)r.okFrags, (unsigned long long)r.badFrags, pass?"PASS":"*** FAIL");
          if (!pass) { rc=3; printf("    [N_STATIONARY oracle FAILED -> NOT running perf. decode swap mismatched.]\n"); }
        }
        if (rc!=3) {
          printf("    %-22s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
          const int M=65536, N=65536, Ks=16384;
          struct G { const char* name; const char* bin; int tileord; };
          G geoms[] = {
              { "A-stat (165.7 win)",  "occ_wggemm2_82_tw4_kwin4_bpf_sp.bin",       0 },
              { "N_STATIONARY (L2)",   "occ_wggemm2_82_tw4_kwin4_bpf_sp_nstat.bin", 1 },
          };
          uint32_t nwgs[] = { 256u, 512u, 1024u };
          for (auto& g : geoms) {
              printf("  --- %s ---\n", g.name);
              for (uint32_t nw : nwgs) {
                  WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, 32772u, 26u, 4, 0, 2, 2, /*useB128*/0, g.tileord);
                  if (!r.ok) { printf("    %-22s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                  printf("    %-22s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                         g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                         (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                  if (r.badSamp) rc=3;
              }
          }
          printf("    [N_STATIONARY keeps a B panel hot in L2 across the M-sweep. >165.7 -> B-feed latency was partly L2\n");
          printf("     miss (cache locality lever is real, make it shape-dependent); flat -> not cache, go to wave-spec (L2).]\n");
        }
    } else if (mode == FP8EDGE) {
        // ===== MAD-305 SAFETY FLOOR: full-fp8-domain edge-encoding gate. The RDNA4 ISA (F8_Mode table) shows e4m3
        //   has denormals (exp0, ~2^-9) + max-normal 0x7E=448, and FP_DENORM (MODE reg) "affects float ops in VALU"
        //   (WMMA runs on VALU). Our prior oracles only fed NORMAL mid-range bytes -> never probed whether the GPU's
        //   default modes interpret denormals/max-normals like our OCP e4m3fn CPU ref. This gate feeds A=denormals,
        //   B=max-normals so denorm*max dominates the dot product -> any input-denormal FLUSH diverges LOUDLY. =====
        printf("\n=== MAD-305 fp8 FULL-DOMAIN edge gate: denormals x max-normals vs CPU OCP-e4m3fn ref ===\n");
        const char* bin = "occ_wggemm2_82_tw4_kwin4_bpf_sp_st1.bin";   // the production 165.7 winner oracle binary
        printf("  --- control: NICE mid-range data (must PASS -- sanity that binary+harness are clean) ---\n");
        { WgcResult r = run_wggemm_compute(node, bin, 512,512,512, 64u, true, 8, 32772u, 26u, 0, 4, 2, 2, 0, 0, /*edgeData*/0);
          bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
          printf("    NICE   okFrags=%llu badFrags=%llu  %s\n", (unsigned long long)r.okFrags, (unsigned long long)r.badFrags, pass?"PASS":"*** FAIL");
          if (!pass) rc=3; }
        printf("  --- EDGE: A=denormals(0x01-0x07,signed)+0x7E, B=max-normals(0x7E=448,0xFE=-448)+mid ---\n");
        { WgcResult r = run_wggemm_compute(node, bin, 512,512,512, 64u, true, 8, 32772u, 26u, 0, 4, 2, 2, 0, 0, /*edgeData*/1);
          bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
          printf("    EDGE   okFrags=%llu badFrags=%llu  %s\n", (unsigned long long)r.okFrags, (unsigned long long)r.badFrags, pass?"PASS":"*** EDGE MISMATCH");
          if (!pass) rc=3; }
        printf("    [EDGE PASS -> GPU WMMA decodes denormals + 0x7E=448 IDENTICALLY to OCP e4m3fn; default FP_DENORM\n");
        printf("     allows input denorms -> NO HW_REG_MODE setreg needed across the full fp8 magnitude domain. Settled.\n");
        printf("     EDGE FAIL while NICE PASSES -> a real edge divergence (likely input-denormal flush); rerun with\n");
        printf("     WGC_DBG=1 to dump the failing values, then decide s_setreg(FP_DENORM) vs ref-fix vs quantizer-clamp.]\n");
    } else if (mode == VGPRPROBE) {
        // ===== MAD-305 VGPR sensitivity probe (RGA follow-up): is the 64-wave residency cap VGPR-bound? The kernel
        //   NEEDS 207 regs (max index v206) so we can't go BELOW the baseline without a repack -- but we can RAISE the
        //   reservation safely (over-allocation never corrupts) and watch residWv. DROP as VGPR climbs -> we're on the
        //   VGPR-occupancy boundary, the 192 repack would pay. FLAT -> NOT VGPR-bound (like LDS) -> the cap is the
        //   SE/persistent-dispatch, go to dyn-VGPR / more persistence. Runs on the LDSTRIM substrate (LDS 32768). =====
        printf("\n=== MAD-305 VGPR sensitivity probe (raise reservation, watch residWv) on the LDSTRIM substrate ===\n");
        printf("    %-26s %6s %8s %9s %8s %s\n","reservation","nWG","TF","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        const char* bin = "occ_wggemm2_82_tw4_kwin4_bpf_sp_ldstrim.bin";
        struct V { const char* name; uint32_t f; };
        V vs[] = { {"208->216 (field26 base)",26u}, {"232->240 (field29)",29u}, {"256 max (field32)",32u} };
        for (auto& v : vs) {
            WgpResult r = run_wggemm_perf(node, bin, M,N,Ks, 512u, freq_hz, 8, 32768u, v.f, 4, 0, 2);
            if (!r.ok) { printf("    %-26s %6u INCOMPLETE\n", v.name, 512u); rc=3; continue; }
            const char* tag = r.tf > 307.0 ? "*** timer glitch" : (r.badSamp?"*** acc00 BAD":"acc00 OK");
            printf("    %-26s %6u %8.1f %9u %8.1f  %s\n", v.name, 512u, r.tf, r.maxlive*8u, (double)r.wall/freq_hz*1e3, tag);
            if (r.badSamp) rc=3;
        }
        printf("    [residWv DROPS as reservation climbs -> VGPR-bound (the 192 repack pays). FLAT across 216..256 ->\n");
        printf("     NOT VGPR-bound; the 64-wave cap is the SE/persistent-dispatch -> dyn-VGPR (lean launch) is the lever.]\n");
    } else if (mode == LDSTRIMMODE) {
        // ===== MAD-305 RGA-surfaced LDS-cliff trim: the winner reserves 32772B LDS (4B over the 32768 boundary) ->
        //   rounds a full 512B granule -> alloc 33280 -> only 1 WG fits a 64KB WGP. LDSTRIM overlaps the 4B
        //   ti-broadcast into A-ring slot 0 -> 32768 = alloc 32768 -> 2 WGs/WGP. WATCH residWv: ~2x (64->128 waves)
        //   = the trim unlocked occupancy; if residWv flat -> occupancy wasn't LDS-bound (it's VGPR/SE/dispatch). =====
        printf("\n=== MAD-305 LDS-cliff trim: 32772 (alloc 33280, 1 WG/WGP) -> 32768 (alloc 32768, 2 WG/WGP) ===\n");
        printf("  --- oracle gate @512x512x512 (the ti-broadcast/A-fill overlap + race-closing barrier) ---\n");
        { WgcResult r = run_wggemm_compute(node, "occ_wggemm2_82_tw4_kwin4_bpf_sp_ldstrim_st1.bin", 512,512,512, 64u, true, 8, 32768u, 26u, 0, 4, 2);
          bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
          printf("    oracle ldstrim  okFrags=%llu badFrags=%llu  %s\n", (unsigned long long)r.okFrags, (unsigned long long)r.badFrags, pass?"PASS":"*** FAIL (overlap race / barrier)");
          if (!pass) { rc=3; printf("    [oracle FAILED -> NOT running perf. The slot-0 overlap raced; barrier placement wrong.]\n"); }
        }
        if (rc!=3) {
          printf("    %-26s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
          const int M=65536, N=65536, Ks=16384;
          struct G { const char* name; const char* bin; uint32_t lds; };
          G geoms[] = {
              { "winner (LDS 33280 alloc)",  "occ_wggemm2_82_tw4_kwin4_bpf_sp.bin",         32772u },
              { "LDSTRIM (LDS 32768 alloc)", "occ_wggemm2_82_tw4_kwin4_bpf_sp_ldstrim.bin", 32768u },
          };
          uint32_t nwgs[] = { 256u, 512u, 1024u };
          for (auto& g : geoms) {
              printf("  --- %s ---\n", g.name);
              for (uint32_t nw : nwgs) {
                  WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, g.lds, 26u, 4, 0, 2);
                  if (!r.ok) { printf("    %-26s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                  printf("    %-26s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                         g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                         (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                  if (r.badSamp) rc=3;
              }
          }
          printf("    [residWv ~2x (e.g. 512->1024 waves) -> the 4-byte trim unlocked 2 WGs/WGP = occupancy was LDS-bound;\n");
          printf("     and if the wall was occupancy, TF rises. residWv flat -> occupancy NOT LDS-bound (VGPR / SE / dispatch).]\n");
        }
    } else if (mode == ALD82) {
        // ===== MAD-305 WIDE A-READ on the 165.7 winner = the ISSUE-SLOT axis ("fewer dispatch slots on feed"):
        //   ds_load_2addr_stride64_b64 loads 2 M-frags/instr (offset*512 matches the mi*512 frag stride) -> A-reads
        //   16->8/slice -> more dispatch bandwidth for WMMA. Same LDS bytes (oracle is the arbiter). >165.7 = the
        //   issue-slot model holds (the win the old ledger predicted from wider loads, finally on silicon). =====
        printf("\n=== MAD-305 8x2 KWINBPF+SETPRIO + wide A-read (ds_load_2addr_stride64) vs the 165.7 winner ===\n");
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) -- THE correctness gate for the wide read ---\n");
        { struct O { const char* name; const char* bin; };
          O ors[] = { { "bpf+sp+ald2", "occ_wggemm2_82_tw4_kwin4_bpf_sp_a2_st1.bin" } };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, 8, 32772u, 26u, 0, 4, 2);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-12s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    %-20s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; };
        G geoms[] = {
            { "bpf+sp (165.7 win)",  "occ_wggemm2_82_tw4_kwin4_bpf_sp.bin" },
            { "bpf+sp + wide-A",     "occ_wggemm2_82_tw4_kwin4_bpf_sp_a2.bin" },
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            printf("  --- %s ---\n", g.name);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, 32772u, 26u, 4, 0, 2);
                if (!r.ok) { printf("    %-20s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-20s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*8u,
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [wide A-read halves A-read issue slots (16->8/slice). >165.7 -> the issue-slot axis pays (mimic the\n");
        printf("     separate pipe by spending fewer dispatch slots on feed); flat -> the base-adds/bank-conflicts ate it.]\n");
    } else if (mode == TW8) {
        // ===== MAD-305 BIG-TILE lever (CDNA rung-8 256x256 equiv): grow the COOPERATIVE tile, not the per-wave tile.
        //   TWN=8 -> 16-wave WG computing a 256x256 SQUARE region (vs 256x128 @ TWN=4). The 165.7 wall is WMMA-result
        //   latency hidden by cross-wave WMMA interleave at a structural WG-residency cap; doubling waves/WG puts 2x WMMA
        //   in flight per WG to hide more of it. Per-wave shape unchanged (FM=8 FN=2, ~128 VGPR static -> co-resides);
        //   LDS unchanged (TM=TWM*FM*16 is TWN-independent -> 32KB KWIN ring); A-strip reused by 8 N-waves (NBANDS=1). =====
        printf("\n=== MAD-305 8x2 TWN=8 256x256 SQUARE cooperative tile (16-wave WG) vs the 165.7 TWN=4 winner ===\n");
        printf("  --- oracle gate @512x512x512 (all FM*FN frags, all waves) -- THE correctness gate for the 16-wave WG ---\n");
        { struct O { const char* name; const char* bin; int twn; };
          O ors[] = {
            { "tw4 win (ref)", "occ_wggemm2_82_tw4_kwin4_bpf_sp_st1.bin",    4 },
            { "tw8 bpf+sp",    "occ_wggemm2_82_tw8_kwin4_bpf_sp_st1.bin",    8 },
            { "tw8 +wide-A",   "occ_wggemm2_82_tw8_kwin4_bpf_sp_a2_st1.bin", 8 },
          };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, 8, 32772u, 26u, 0, o.twn, 2);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-14s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          }
        }
        printf("    %-20s %6s %8s %8s %9s %8s %s\n","geom","nWG","TF","%307","residWv","span_ms","correct");
        const int M=65536, N=65536, Ks=16384;
        struct G { const char* name; const char* bin; int twn; };
        G geoms[] = {
            { "tw4 bpf+sp (165.7)", "occ_wggemm2_82_tw4_kwin4_bpf_sp.bin",    4 },
            { "tw8 bpf+sp (sq)",    "occ_wggemm2_82_tw8_kwin4_bpf_sp.bin",    8 },
            { "tw8 +wide-A (sq)",   "occ_wggemm2_82_tw8_kwin4_bpf_sp_a2.bin", 8 },
        };
        uint32_t nwgs[] = { 256u, 512u, 1024u };
        for (auto& g : geoms) {
            printf("  --- %s ---\n", g.name);
            for (uint32_t nw : nwgs) {
                WgpResult r = run_wggemm_perf(node, g.bin, M,N,Ks, nw, freq_hz, 8, 32772u, 26u, g.twn, 0, 2);
                if (!r.ok) { printf("    %-20s %6u INCOMPLETE\n", g.name, nw); rc=3; continue; }
                printf("    %-20s %6u %8.1f %6.1f%% %9u %8.1f  %s\n",
                       g.name, nw, r.tf, 100.0*r.tf/307.0, r.maxlive*(uint32_t)(2*g.twn),
                       (double)r.wall/freq_hz*1e3, r.badSamp?"*** acc00 BAD":"acc00 OK");
                if (r.badSamp) rc=3;
            }
        }
        printf("    [TWN=8 doubles waves/WG (8->16) -> 2x in-flight WMMA per WG to hide WMMA-result latency. >165.7 -> the\n");
        printf("     big-tile lever lands; flat/down -> residency cap counts WGs not waves (fewer 16-wave WGs fit -> same\n");
        printf("     total in-flight) or the 16-wave barrier serializes. residWv = maxlive(WGs) * WAVES.]\n");
    } else if (mode == TW4LEAN) {
        // ===== MAD-305 LEAN-16-WAVE (TWM=4 TWN=4 FM=4 FN=2, 256x128, 16 lean waves) — SAFE oracle-GATED run. The C-store
        //   FM*FN=16 hardcode that page-faulted the R9700 (and starved the desktop GPU) is FIXED (FMFN_LOG2 + TROW_SH).
        //   SAFETY PROTOCOL: run the TINY 512^3 oracle FIRST per vgprField; the big 65536^2 perf runs ONLY where that oracle
        //   PASSED (no fault, admitted, correct). vf=26 dropped (known 16-wave co-residency deadlock at 208). Goals: (1) confirm
        //   the page-fault fix (oracle PASS, no fault), (2) find the VGPR-reservation floor where 16 lean waves co-reside,
        //   (3) compare TF vs the 165.7 winner. Growing TWM also amortizes the BINDING B-feed (M-waves share B). =====
        printf("\n=== MAD-305 LEAN-16-WAVE (TWM=4 TWN=4 FM=4 FN=2, 256x128) — oracle-GATED, page-fault fix verify ===\n");
        const int Mo=512, No=512, Ko=512;
        const int M=65536, N=65536, Ks=16384;
        const char* obin = "occ_wggemm2_42_tw4x4_kwin4_bpf_sp_st1.bin";
        const char* pbin = "occ_wggemm2_42_tw4x4_kwin4_bpf_sp.bin";
        // STEP 0 -- BISECTION CONTROL: same per-wave FM=4 FN=2 but only 8 waves (TWM=2, 128x128). PASS -> the all-wrong bug
        //   is in the TWM=4/16-wave path; FAIL -> it's the FM=4xFN=2 per-wave/frag math. vf=26 (208) is plenty for 8 waves.
        printf("  --- STEP 0: bisection control TWM=2 FM=4 FN=2 (8 waves, 128x128) @512^3 ---\n");
        { WgcResult c = run_wggemm_compute(node, "occ_wggemm2_42_tw2x4_kwin4_bpf_sp_st1.bin", Mo,No,Ko, 64u, true, 4, 32772u, 26u, 0, 4, 2, 2);
          const char* cst = (!c.ok) ? "HANG/TO" : (c.badFrags==0 && c.okFrags>0 ? "PASS" : "*** FAIL");
          printf("    control A TWM2 TWN4 FM4 FN2 (8w per-wave)  okFrags=%lu badFrags=%lu  %s  => %s\n",
                 (unsigned long)c.okFrags, (unsigned long)c.badFrags, cst,
                 (c.ok && c.badFrags==0 && c.okFrags>0) ? "FM4xFN2 per-wave math OK -> bug is in the TWM=4 path"
                                                        : "FM4xFN2 per-wave math BROKEN -> bug is the per-wave/frag combo");
        }
        // STEP 0b -- M-DOUBLE ISOLATOR: TWM=4 TWN=2 FM=4 FN=4 (wave_m 0-3, TM=256, TROW_SH=8) but 8 waves + PROVEN 2-band fill.
        { WgcResult c = run_wggemm_compute(node, "occ_wggemm2_44_tw4x2_kwin4_bpf_sp_st1.bin", Mo,No,Ko, 64u, true, 4, 32772u, 26u, 0, 2, 4, 4);
          const char* cst = (!c.ok) ? "HANG/TO" : (c.badFrags==0 && c.okFrags>0 ? "PASS" : "*** FAIL");
          printf("    control B TWM4 TWN2 FM4 FN4 (8w M-double)  okFrags=%lu badFrags=%lu  %s  => %s\n",
                 (unsigned long)c.okFrags, (unsigned long)c.badFrags, cst,
                 (c.ok && c.badFrags==0 && c.okFrags>0) ? "wave_m 0-3 / TM=256 / TROW_SH=8 OK -> bug is the 16-wave 512-thread fill"
                                                        : "wave_m 0-3 / TM=256 BROKEN -> bug is the TWM=4 M-doubling path");
        }
        // STEP 0d -- KWIN=0 BASE-PATH ISOLATOR: the lean 16-wave geometry on the simplest base path (LDS = 8196, no KWIN ring).
        //   GUARDED behind TW4LEAN_KWIN0=1: this is a 16-wave dispatch whose co-residency is UNPROVEN; if it can't co-reside it
        //   DEADLOCKS THE BARRIER AND BRICKS THE GPU (happened at vf=26). Only opt in after warning the user.
        if (getenv("TW4LEAN_KWIN0")) { WgcResult c = run_wggemm_compute(node, "occ_wggemm2_42_tw4x4_kwin0_st1.bin", Mo,No,Ko, 64u, true, 4, 8196u, 22u, 0, 4, 2, 4);
          const char* cst = (!c.ok) ? "HANG/TO" : (c.badFrags==0 && c.okFrags>0 ? "PASS" : "*** FAIL");
          printf("    control C TWM4 TWN4 FM4 FN2 KWIN=0 (16w base)  okFrags=%lu badFrags=%lu  %s  => %s\n",
                 (unsigned long)c.okFrags, (unsigned long)c.badFrags, cst,
                 (c.ok && c.badFrags==0 && c.okFrags>0) ? "16-wave base path OK -> half-contraction bug is in the KWIN windowed feed"
                                                        : "16-wave base path BROKEN -> bug is in the base A-fill/WMMA, not KWIN");
        }
        // STEP 0c -- K-SCALING PROBE on the lean bin (WGC_DBG prints ratio=|maxD|/|maxExp|). Constant ratio across K -> a
        //   proportional feed/data factor; ratio halving as K doubles -> kernel processes a FIXED slice count (K-window loop bug).
        if (getenv("WGC_DBG")) {
            printf("  --- STEP 0c: K-scaling probe on lean bin (vf=22) -- see WGC_DBG ratio lines on stderr ---\n");
            for (int Kp : {256, 512, 1024})
                run_wggemm_compute(node, obin, 512,512,Kp, 64u, true, 4, 32772u, 22u, 0, 4, 2, 4);
        }
        uint32_t fields[] = { 22u };   // ONLY vf=22 (176): proven to admit 16 waves this session w/o bricking. vf>=26 deadlock-bricks; vf<22 under-provisions.
        bool oraclePass[3] = { false, false, false };
        printf("  --- STEP 1: tiny 512^3 oracle per vgprField (THE safety gate -- big perf runs only where this PASSes) ---\n");
        printf("    %-7s %6s  %-8s %9s %9s\n", "vgprFld","VGPR","oracle","okFrags","badFrags");
        for (int i = 0; i < 3; ++i) {
            uint32_t vf = fields[i];
            WgcResult o = run_wggemm_compute(node, obin, Mo,No,Ko, 64u, true, 4, 32772u, vf, 0, 4, 2, 4);
            bool pass = o.ok && o.badFrags==0 && o.okFrags>0;
            oraclePass[i] = pass;
            const char* ost = (!o.ok) ? "HANG/TO" : (pass ? "PASS" : "*** FAIL");
            printf("    %-7u %6u  %-8s %9lu %9lu\n", vf, vf*8u, ost, (unsigned long)o.okFrags, (unsigned long)o.badFrags);
            if (o.ok && !pass) rc=3;   // admitted-but-wrong is a real bug; HANG/TO is expected co-residency at high vf
        }
        printf("  --- STEP 2: 65536^2 x K16384 perf -- ONLY vgprFields that PASSED the oracle above ---\n");
        printf("    %-7s %6s | %8s %7s %8s %8s %s\n", "vgprFld","VGPR","nWG","TF","%307","residWv","perf-correct");
        for (int i = 0; i < 3; ++i) {
            if (!oraclePass[i]) { printf("    %-7u %6u | skipped (oracle not PASS -> not run on silicon)\n", fields[i], fields[i]*8u); continue; }
            WgpResult r = run_wggemm_perf(node, pbin, M,N,Ks, 512u, freq_hz, 4, 32772u, fields[i], 4, 0, 2, 4);
            if (!r.ok) { printf("    %-7u %6u | %8u INCOMPLETE (oracle admitted @nWG=64 but not perf @nWG=512 -> co-residency)\n", fields[i], fields[i]*8u, 512u); continue; }
            printf("    %-7u %6u | %8u %7.1f %7.1f%% %8u %s\n",
                   fields[i], fields[i]*8u, 512u, r.tf, 100.0*r.tf/307.0, r.maxlive*16u, r.badSamp?"*** acc00 BAD":"acc00 OK");
            if (r.badSamp) rc=3;
        }
        printf("  --- reference: 8x2 TWM=2 TWN=4 winner (8 fat waves, 256x128) @ vgprField=26 (GPU sanity) ---\n");
        WgpResult w = run_wggemm_perf(node, "occ_wggemm2_82_tw4_kwin4_bpf_sp.bin", M,N,Ks, 512u, freq_hz, 8, 32772u, 26u, 4, 0, 2, 2);
        if (w.ok) printf("    %-7u %6u | %8u %7.1f %7.1f%% %8u %s\n",
                         26u, 208u, 512u, w.tf, 100.0*w.tf/307.0, w.maxlive*8u, w.badSamp?"*** BAD":"acc00 OK");
        else      printf("    winner reference INCOMPLETE\n");
        printf("    [oracle PASS at any vf = the page-fault C-store bug is FIXED (no fault). perf TF at a passing vf vs 165.7 =\n");
        printf("     whether 16 lean waves beat 8 fat. perf INCOMPLETE while oracle PASSed = co-residency only fits at nWG<512.]\n");
    } else if (mode == WALL82) {
        // ===== MAD-305 8x2 WALL ATTRIBUTION: after 8x2 broke 4x4's wall (162 vs 149), what is 8x2's NEW wall?
        //   FED / FEEDONLY / NOFEED at saturated 65536^2 x K16384, occupancy-matched (vgprField 26), FED full-oracle.
        //   FED==FEEDONLY -> still feed-bound (keep growing FM / cut B). FED<FEEDONLY or NOFEED~FED -> compute/sched
        //   emerged. NOFEED drops badly -> 8x2 acc shape hurts issue density. (4x4 row = the control.) =====
        printf("\n=== MAD-305 8x2 WALL ATTRIBUTION @65536^2 x K16384 (TWN=4, vgprField=26, nWG=512) ===\n");
        // ---- FED full-fragment oracle gate first (small size) ----
        printf("  --- FED oracle gate @512x512x512 (all FM*FN frags, all waves) ---\n");
        { struct O { const char* name; const char* bin; int fm,fn; uint32_t lds; };
          O ors[] = { { "4x4", "occ_wggemm2_tw4_kwin4_st1.bin", 4,4, 16388u },
                      { "8x2", "occ_wggemm2_82_tw4_kwin4_st1.bin", 8,2, 32772u } };
          for (auto& o : ors) {
            WgcResult r = run_wggemm_compute(node, o.bin, 512,512,512, 64u, true, o.fm, o.lds, 26u, 0, 4, o.fn);
            bool pass = r.ok && r.badFrags==0 && r.okFrags>0;
            printf("    oracle %-4s okFrags=%u badFrags=%u  %s\n", o.name, r.okFrags, r.badFrags, pass?"PASS":"*** FAIL");
            if (!pass) rc=3;
          } }
        printf("    %-10s %8s %8s %9s %7s %8s %8s %9s %s\n","geom","NOFEED","FED","FEEDonly","fill%","FED/FO","maxlvWG","residWv","FEDacc00");
        const int M=65536, N=65536, Ks=16384; const uint32_t nWG=512u;
        struct W { const char* name; const char* nf; const char* fed; const char* fo; int fm, fn; uint32_t fedlds, nflds; };
        W rows[] = {
            { "4x4 TWN4", "occ_wggemm2_tw4_nofeed.bin",    "occ_wggemm2_tw4_kwin4_pw4.bin",    "occ_wggemm2_tw4_kwin4_pw4_feedonly.bin",    4,4, 16388u, 8196u },
            { "8x2 TWN4", "occ_wggemm2_82_tw4_nofeed.bin", "occ_wggemm2_82_tw4_kwin4_pw4.bin", "occ_wggemm2_82_tw4_kwin4_pw4_feedonly.bin", 8,2, 32772u, 8708u },
        };
        for (auto& w : rows) {
            // FED + FEEDONLY first (the wall question); NOFEED last + non-fatal (8x2 NOFEED can hang -> don't lose fed/fo).
            WgpResult fl = run_wggemm_perf(node, w.fed, M,N,Ks, nWG, freq_hz, w.fm, w.fedlds, 26u, 4, 0, w.fn);
            WgpResult fo = run_wggemm_perf(node, w.fo,  M,N,Ks, nWG, freq_hz, w.fm, w.fedlds, 26u, 4, 0, w.fn);
            if (!fl.ok||!fo.ok) { printf("    %-10s INCOMPLETE (fed=%d fo=%d)\n", w.name, fl.ok,fo.ok); rc=3; continue; }
            double fedfo = fo.tf>0 ? fl.tf/fo.tf : 0.0;
            WgpResult nf = run_wggemm_perf(node, w.nf,  M,N,Ks, nWG, freq_hz, w.fm, w.nflds,  26u, 4, 0, w.fn);
            char nfs[24], fills[16];
            if (nf.ok) { snprintf(nfs,24,"%.1f",nf.tf); snprintf(fills,16,"%.1f%%",nf.tf>0?100.0*fl.tf/nf.tf:0.0); }
            else       { snprintf(nfs,24,"HANG/TO"); snprintf(fills,16,"n/a"); }
            printf("    %-10s %8s %8.1f %9.1f %7s %8.3f %8u %9u %s\n",
                   w.name, nfs, fl.tf, fo.tf, fills, fedfo, fl.maxlive, fl.maxlive*8u,
                   fl.badSamp?"*** BAD":"OK");
            if (fl.badSamp) rc=3;
        }
        printf("    [FED/FO ~1.00 -> still feed-bound (FED==FEEDONLY: WMMA free). FED/FO <1 or NOFEED~FED -> compute/sched emerged.\n");
        printf("     8x2 FEEDONLY isolates whether the A-LDS-read doubling (A-rd/MAC 0.25->0.50) now dominates the feed.]\n");
    } else if (mode == WAVESWEEP) {
        // ===== MAD-305 WAVE-SIZE SWEEP (WG wave count, biggest->smallest) -- feed-bound check at each size.
        //   For each geometry: NOFEED ceiling, FED full, FED feed-only. FED==FEEDonly => WMMA free, feed is wall. =====
        printf("\n=== MAD-305 WAVE-SIZE SWEEP @65536^2 x K16384 (biggest->smallest WG; TWM=2 fixed) ===\n");
        printf("    feed-bound test: FED == FEED-only -> WMMA is hidden; the gap NOFEED->FED is the feed cost.\n");
        printf("    %-18s %5s %8s %8s %9s %7s %7s %8s %s\n","geom","waves","NOFEED","FED","FEEDonly","%307","fill%","span_ms","correct");
        struct W { const char* name; const char* full; const char* fo; const char* nf; int twn; };
        W rows[] = {
            { "128x256 TWN=4", "occ_wggemm2_tw4_kwin4_pw4.bin", "occ_wggemm2_tw4_kwin4_pw4_feedonly.bin", "occ_wggemm2_tw4_nofeed.bin", 4 },
            { "128x128 TWN=2", "occ_wggemm2_kwin4_pw4.bin",     "occ_wggemm2_kwin4_pw4_feedonly.bin",     "occ_wggemm2_nofeed4_perf.bin", 2 },
            { "128x64  TWN=1", "occ_wggemm2_tw1_kwin4_pw4.bin", "occ_wggemm2_tw1_kwin4_pw4_feedonly.bin", "occ_wggemm2_tw1_nofeed.bin", 1 },
        };
        const int M=65536, N=65536, Ks=16384;
        for (auto& w : rows) {
            WgpResult nf = run_wggemm_perf(node, w.nf,   M,N,Ks,256u,freq_hz,4,8196u, 26u,w.twn);
            WgpResult fl = run_wggemm_perf(node, w.full, M,N,Ks,256u,freq_hz,4,16388u,26u,w.twn);
            WgpResult fo = run_wggemm_perf(node, w.fo,   M,N,Ks,256u,freq_hz,4,16388u,26u,w.twn);
            if (!nf.ok||!fl.ok||!fo.ok) { printf("    %-18s INCOMPLETE\n", w.name); rc=3; continue; }
            double fill = nf.tf>0 ? 100.0*fl.tf/nf.tf : 0.0;
            printf("    %-18s %5d %8.1f %8.1f %9.1f %6.1f%% %6.1f%% %8.1f  %s\n",
                   w.name, 2*w.twn, nf.tf, fl.tf, fo.tf, 100.0*fl.tf/307.0, fill,
                   (double)fl.wall/freq_hz*1e3, fl.badSamp?"*** acc00 BAD":"acc00 OK");
            if (fl.badSamp) rc=3;
        }
        printf("    [biggest WG != faster (per-WG-tile correction): at saturation all converge to the per-wave feed wall.]\n");
    } else if (mode == FEEDPROF) {
        // ===== MAD-305 Step A phase timers: where is the real FEEDONLY K-loop parked? (GPT: timers first) =====
        printf("\n=== MAD-305 Step A phase timers: real DEFAULT FED K-loop (DBUF==1 A-ping-pong, 4096^2 x K16384), 1 profiler wave ===\n");
        printf("    100 MHz realtime (10 ns/tick). Per-phase = avg over the profiler wave's K-tiles. This is the 1.4-TF FED path.\n");
        ProfResult r = run_feedprof(node, "occ_wggemm2_prof.bin", 4096, 4096, 16384, 256u, freq_hz);
        if (!r.ok) { printf("  INCOMPLETE\n"); rc = 3; }
        else {
            const char* ph[7] = {"1 B load wait (8tr)", "2 A-frag ds_load   ", "3 32 WMMA (+A pf)  ",
                                 "4 A-pf land+ds_stor", "5 tail barrier     ", "6 bookkeeping      ", "7 (unused)         "};
            double tot = 0; for (int i=0;i<7;i++) tot += r.sum[i];
            double perKt_ns = (r.ktiles>0) ? (tot / r.ktiles * 10.0) : 0.0;
            printf("  profiler K-tiles=%u  maxlive=%u  whole-wall TF=%.2f (FED-equivalent ~1.4 TF = 2.7 GB/s)\n",
                   r.ktiles, r.maxlive, r.tf);
            printf("  *** acc00 OK=%u/%u %s -- if BAD, the 96 TF was a bogus wall from incomplete/garbage work ***\n",
                   r.okSamp, r.okSamp + r.badSamp, r.badSamp ? "<<< OUTPUT WRONG" : "(output correct)");
            printf("  --- per-K-tile phase breakdown (ns, %% of K-tile) ---\n");
            for (int i=0;i<7;i++) {
                double ns = (r.ktiles>0) ? ((double)r.sum[i]/r.ktiles*10.0) : 0.0;
                double pct = (tot>0) ? (100.0*r.sum[i]/tot) : 0.0;
                printf("    %s  %10.1f ns  %5.1f%%\n", ph[i], ns, pct);
            }
            printf("    %-18s  %10.1f ns  100.0%%  (sum of phases per K-tile)\n", "TOTAL/K-tile", perKt_ns);
        }
    } else if (mode == FEEDSTAG) {
        // ===== MAD-305 rung 8: inert per-WG phase stagger on the REAL kernel (no PROFILE, no timers). Test the
        //   desync hypothesis directly: does decorrelating the inter-WG barrier lockstep recover 1.4 -> tens TF?
        //   baseline = the byte-identical perf bin; then delay = ((ti*13+wid*3)&MASK)<<SHIFT busy-loop iters.
        //   MASK=0 = delay==0 control (stagger code present, zero iters). Same dispatch/occupancy as --wggemm-perf. =====
        printf("\n=== MAD-305 rung 8: inert per-WG STAGGER @4096^2 x K16384 (decorrelate the barrier lockstep) ===\n");
        printf("    baseline = byte-identical perf bin (~1.4 TF). 1.4->tens = lockstep CONFIRMED; no move = PROFILE 70x was a side effect.\n");
        printf("    delay = ((ti*13 + wid*3) & MASK) << SHIFT busy-loop iters, once per claimed tile (per-WG via ti).\n");
        struct St { const char* name; const char* bin; };
        St rows[] = {
            { "baseline   (no stagger)   ", "occ_wggemm2_perf.bin"        },
            { "MASK=0  SHIFT=5 (delay==0)", "occ_wggemm2_stag_m0_s5.bin"  },
            { "MASK=3  SHIFT=5 (<=96 it) ", "occ_wggemm2_stag_m3_s5.bin"  },
            { "MASK=7  SHIFT=5 (<=224 it)", "occ_wggemm2_stag_m7_s5.bin"  },
            { "MASK=15 SHIFT=5 (<=480 it)", "occ_wggemm2_stag_m15_s5.bin" },
            { "MASK=31 SHIFT=5 (<=992 it)", "occ_wggemm2_stag_m31_s5.bin" },
            { "MASK=15 SHIFT=4 (<=240 it)", "occ_wggemm2_stag_m15_s4.bin" },
            { "MASK=15 SHIFT=6 (<=960 it)", "occ_wggemm2_stag_m15_s6.bin" },
        };
        for (auto& s : rows) {
            WgpResult r = run_wggemm_perf(node, s.bin, 4096, 4096, 16384, 256u, freq_hz);
            if (!r.ok) { printf("  %-26s INCOMPLETE (hang/timeout)\n", s.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0;
            printf("  %-26s %7.1f TF  %5.1f%%  maxlive=%u WGs  claims=%u  acc00 OK=%u/%u%s\n",
                   s.name, r.tf, pct, r.maxlive, r.total, r.okSamp, r.okSamp + r.badSamp,
                   r.badSamp ? "  *** ACC00 MISMATCH ***" : "");
        }
    } else if (mode == FEEDPB) {
        // ===== MAD-305 rung 9: bisect the PROFILE 70x. Each variant adds ONE PROFILE ingredient to the real
        //   DBUF==1 path (byte-close, non-PROFILE). Whichever reproduces ~96 TF IS the lever. Order per GPT:
        //   #1 per-K sendmsg (best first) -> #2 per-tile token atomic -> #3 per-K branch skeleton -> #4 per-K
        //   inert delay (control vs #1: special-sendmsg vs generic per-K perturbation). =====
        printf("\n=== MAD-305 rung 9: PROFILE 70x BISECTION @4096^2 x K16384 (which ingredient -> 96 TF?) ===\n");
        printf("    non-PROFILE=1.4 TF, PROFILE=96 TF. Find the lever. Whichever variant jumps to tens/90 TF IS the cause.\n");
        struct Pb { const char* name; const char* bin; const char* cadence; };
        Pb rows[] = {
            { "baseline (no ingredient)    ", "occ_wggemm2_perf.bin", "--"     },
            { "PB1 sendmsg+kmcnt (all-wave)", "occ_wggemm2_pb1.bin",  "per-K"  },
            { "PB2 leader token atomic     ", "occ_wggemm2_pb2.bin",  "per-tile" },
            { "PB3 cmp/branch skeleton     ", "occ_wggemm2_pb3.bin",  "per-K"  },
            { "PB4 inert busy-loop (ctrl)  ", "occ_wggemm2_pb4.bin",  "per-K"  },
        };
        for (auto& p : rows) {
            WgpResult r = run_wggemm_perf(node, p.bin, 4096, 4096, 16384, 256u, freq_hz);
            if (!r.ok) { printf("  %-28s INCOMPLETE (hang/timeout)\n", p.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0;
            printf("  %-28s [%-8s] %7.1f TF  %5.1f%%  maxlive=%u WGs  claims=%u  acc00 OK=%u/%u%s\n",
                   p.name, p.cadence, r.tf, pct, r.maxlive, r.total, r.okSamp, r.okSamp + r.badSamp,
                   r.badSamp ? "  *** ACC00 MISMATCH ***" : "");
        }
    } else if (mode == STACK) {
        // ===== MAD-305 STACK LADDER: rebuild the fast feed from a known-good core, +1 obligation/rung, with
        //   TF / GB/s / proof at every rung. Collapse toward the real kernel's 2.7 GB/s / 1.4 TF names the wall. =====
        printf("\n=== MAD-305 STACK LADDER -- BANDWIDTH HUNT: where is the 640 GB/s? (stride x working-set) ===\n");
        printf("    Residency sweep already showed: BW flat ~24 GB/s from 512..2048 waves, maxlive HARD-CAPS at 2048\n");
        printf("    -> NOT occupancy. Suspect: strided access (2KB read / 8KB stride) wrecks GDDR6 row-buffer efficiency.\n");
        printf("    %-30s %9s  %-20s %-8s %s\n", "config (nWG=2048)", "GB/s", "proof", "maxlive", "%peak640");
        const uint32_t CEILsw = 2097152u;
        struct Cfg { const char* name; uint32_t stride; uint64_t window; };
        Cfg cfgs[] = {
            { "16MiB win  STRIDED(8K/2K)", 8192u,  16ull*1024*1024 },
            { "16MiB win  CONTIG (2K/2K)", 2048u,  16ull*1024*1024 },
            { "256MiB win STRIDED(8K/2K)", 8192u, 256ull*1024*1024 },
            { "256MiB win CONTIG (2K/2K)", 2048u, 256ull*1024*1024 },
        };
        for (auto& c : cfgs) {
            StkResult r = run_stack(node, "occ_stack_r1.bin", freq_hz, 8, 256, c.stride, c.window, CEILsw, 2048u);
            if (!r.ok) { printf("  %-30s INCOMPLETE\n", c.name); rc = 3; continue; }
            char proof[40];
            snprintf(proof, sizeof proof, "%s cnt=%s chk%s", r.proof ? "PASS" : "*FAIL",
                     (r.consumed==CEILsw) ? "ok" : "BAD", (r.got==r.exp && r.exp!=0) ? "=CPU" : "!=CPU");
            printf("  %-30s %8.1f  %-20s %-8u %.1f%%\n", c.name, r.gbps, proof, r.maxlive, 100.0*r.gbps/640.0);
            if (!r.proof) rc = 3;
        }
        printf("    [CONTIG >> STRIDED on 256MiB(HBM) -> the stride wrecks DRAM efficiency; 16MiB(cache) -> structure ceiling]\n");
    } else if (mode == KWINORACLE) {
        // ===== STRONG correctness gate for the single-reuse-barrier (KWINNOTAIL): full 16-frag oracle, REPEATED,
        //   to catch the rare cross-wave LDS race that dropping the tail barrier could introduce. =====
        printf("\n=== MAD-305 KWIN single-reuse-barrier RACE GATE (full 16-frag oracle, repeated) ===\n");
        printf("    dropping the tail barrier risks a fast wave overwriting an LDS slot a slow wave is still reading.\n");
        printf("    %-22s %-14s %8s  %12s  %s\n", "config", "shape", "repeats", "badFrags(tot)", "verdict");
        struct O { const char* name; const char* bin; };
        O os[] = { { "KWIN=4 TAIL  (safe) ", "occ_wggemm2_kwin4_st1.bin"   },
                   { "KWIN=4 NO-TAIL      ", "occ_wggemm2_kwin4nt_st1.bin" } };
        struct Sm { const char* name; int M, N, K; uint32_t nWG; };
        Sm sh[] = { { "256x256x256",   256, 256,  256, 256u },
                    { "512x512x2048",  512, 512, 2048, 256u } };
        const int REPEATS = 10;
        for (auto& o : os) {
            for (auto& s : sh) {
                uint64_t totBad = 0, totOk = 0; int incomplete = 0;
                for (int r = 0; r < REPEATS; ++r) {
                    WgcResult res = run_wggemm_compute(node, o.bin, s.M, s.N, s.K, s.nWG, true, 4, 16388u, 26u);
                    if (!res.ok) { incomplete++; continue; }
                    totBad += res.badFrags; totOk += res.okFrags;
                }
                bool pass = (totBad == 0 && incomplete == 0 && totOk > 0);
                printf("  %-22s %-14s x%-7d %12llu  %s\n", o.name, s.name, REPEATS,
                       (unsigned long long)totBad, pass ? "PASS" : "*** RACE/FAIL");
                if (!pass) rc = 3;
            }
        }
        printf("    [NO-TAIL must show ZERO bad frags across ALL repeats to be trusted. Any bad => the tail barrier is required.]\n");
    } else if (mode == KWIN) {
        // ===== MAD-305 A-LDS-ring K-WINDOW (GPT structural lever): amortize the 4-wave A-publish barrier over
        //   KWIN K-tiles. CORRECTNESS-GATED (acc00 must stay 64/64). Decision: U=2/4 moves 79->120+ => barrier
        //   frequency was the wall; U flat => not barrier freq (look at feed issue ordering). =====
        printf("\n=== MAD-305 A-LDS-ring K-WINDOW @4096^2 x K16384 (amortize publish barrier; correctness-gated) ===\n");
        printf("    FED baseline 79 TF (1 barrier/K-tile). KWIN puts 2 barriers / KWIN K-tiles. NOFEED ceiling ~287.\n");
        struct K { const char* name; const char* bin; uint32_t lds; };
        K rows[] = {
            { "FED baseline (DBUF==1)    ", "occ_wggemm2_perf.bin",  8196u  },
            { "KWIN=2  (8KB ring)        ", "occ_wggemm2_kwin2.bin",      8196u  },
            { "KWIN=2 + B-prefetch       ", "occ_wggemm2_kwin2_bpf.bin",  8196u  },
            { "KWIN=4  (16KB ring)       ", "occ_wggemm2_kwin4.bin",      16388u },
            { "KWIN=4 + B-prefetch       ", "occ_wggemm2_kwin4_bpf.bin",  16388u },
            { "KWIN=4 + 2-wide publish   ", "occ_wggemm2_kwin4_pub2.bin", 16388u },
            { "KWIN=4 single-barrier     ", "occ_wggemm2_kwin4_notail.bin", 16388u },
            { "KWIN=4 NO-BARRIER (probe) ", "occ_wggemm2_kwin4_nobar.bin",  16388u },
            { "KWIN=8  (32KB ring)       ", "occ_wggemm2_kwin8.bin",      32772u },
        };
        for (auto& k : rows) {
            WgpResult r = run_wggemm_perf(node, k.bin, 4096, 4096, 16384, 256u, freq_hz, 4, k.lds, 26u);
            if (!r.ok) { printf("  %-27s INCOMPLETE (hang/timeout)\n", k.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0;
            printf("  %-27s %6.1f TF  %5.1f%%  %.2f WMMA/cyc  maxlive=%u  acc00 %s\n",
                   k.name, r.tf, pct, 15.9*r.tf/307.0, r.maxlive, r.badSamp ? "*** BAD" : "OK");
            if (r.badSamp) rc = 3;
        }
        printf("    [correctness MUST hold (acc00 OK). U=2/4 up => barrier-freq wall; flat => feed issue ordering; LDS up may cut occupancy.]\n");
    } else if (mode == BASELINES) {
        // ===== MAD-305 VRAM RE-BASELINE (post-PCIe-fix): re-establish the canonical four at TARGET with all
        //   operands device-local. Everything measured before the VRAM fix is CONTAMINATED for bandwidth. =====
        printf("\n=== MAD-305 VRAM RE-BASELINE @4096^2 x K16384 (operands device-local; pre-fix perf was PCIe-contaminated) ===\n");
        printf("    ceiling 307 TF; HIP winner 161 TF (hard pass 153). FEEDONLY/NOFEED outputs are garbage by design -- TF is the metric.\n");
        struct B { const char* name; const char* bin; bool correct; };
        B rows[] = {
            { "PM4 FED      (DBUF==1 real)  ", "occ_wggemm2_perf.bin",          true  },
            { "PM4 FEEDONLY (DBUF==1, -WMMA)", "occ_wggemm2_feedonly_perf.bin", false },
            { "PM4 NOFEED   (compute ceiling)", "occ_wggemm2_nofeed4_perf.bin",  false },
        };
        for (auto& b : rows) {
            WgpResult r = run_wggemm_perf(node, b.bin, 4096, 4096, 16384, 256u, freq_hz);
            if (!r.ok) { printf("  %-31s INCOMPLETE (hang/timeout)\n", b.name); rc = 3; continue; }
            double pct = 100.0*r.tf/307.0;
            const char* corr = b.correct ? (r.badSamp ? "*** acc00 BAD" : "acc00 OK") : "(output N/A)";
            printf("  %-31s %7.1f TF  %5.1f%%  %.2f WMMA/cyc  maxlive=%u  %s\n",
                   b.name, r.tf, pct, 15.9*r.tf/307.0, r.maxlive, corr);
            if (b.correct && r.badSamp) rc = 3;
        }
        printf("    [FED<<NOFEED -> feed-bound (tune feed/MLP); FED~NOFEED -> compute/issue-bound. vs HIP FED 161.]\n");
    } else if (mode == SUSTAIN) {
        // Toy ~18ms kernels never make the card commit to max clock -> the clock catches the ramp at a
        // random point each run (the 95<->119 variance). FIX = REAL sustained work: BIG M,N (many tiles,
        // long runtime) with small K (operands stay small, fit the RAM cap). The card ramps and HOLDS;
        // the ramp amortizes over the long run -> the big-size TF is the TRUE saturated number. Sample
        // sclk concurrently (external) to prove the card pins itself under genuine load.
        // SATURATED FEED-LEVER SWEEP. Saturation already proven (FED flat 16384->65536; NOFEED->301 @2350).
        // Run at 32768^2 x K16384 (proven saturated, ~333ms, card pins 2350) so feed-lever deltas are real,
        // not clock noise. NOFEED first = the fill denominator. fill = TF/NOFEED. The lever that closes
        // 35%->higher is the win; vs HIP FED 161 / hipBLASLt 143.
        printf("\n=== SATURATED LARGER-TILE A/B @32768^2 x K16384 (card pinned 2350; deltas are real) ===\n");
        printf("    [thesis: bigger tile -> higher arithmetic intensity -> less feed/MAC -> FED closes toward NOFEED.\n");
        printf("     128x256 (TWN=4, 8 waves) halves A-feed (A-strip reused by 4 N-waves not 2) vs 128x128 (4 waves).]\n");
        printf("    %-24s %8s %7s %7s %8s %s\n", "config", "TF", "%307", "fill%", "span_ms", "mx/correct");
        struct FL { const char* name; const char* bin; uint32_t lds; int fmt; int twn; bool nofeed; bool correct; };
        FL rows[] = {  // per-tile NOFEED ceiling precedes its FED rows (fill% = TF/most-recent-NOFEED).
            { "128x128 NOFEED ceil",  "occ_wggemm2_nofeed4_perf.bin", 8196u, 4, 2, true,  false },  // ~297 (97%)
            { "128x128 FED pw4 (was BEST)","occ_wggemm2_kwin4_pw4.bin", 16388u,4, 2, false, true  },  // ~147 = 92% HIP 161
            { "128x128 FED pw2",        "occ_wggemm2_kwin4_pw2.bin",   16388u,4, 2, false, true  },  // ~145
            { "128x256 NOFEED ceil",  "occ_wggemm2_tw4_nofeed.bin",    8196u, 4, 4, true,  false },  // NEW: TWN=4 8-wave ceiling
            { "128x256 FED pw4",        "occ_wggemm2_tw4_kwin4_pw4.bin",16388u,4, 4, false, true  },  // NEW: the larger-tile lever
            { "128x256 FED pw2",        "occ_wggemm2_tw4_kwin4_pw2.bin",16388u,4, 4, false, true  },  // NEW
        };
        const int M=65536, N=65536, Ks=16384;   // 4x the work vs 32768^2 -> ~940ms/row so the card commits+HOLDS 2350
        double nofeedTf = 0;
        for (auto& f : rows) {
            WgpResult r = run_wggemm_perf(node, f.bin, M, N, Ks, 256u, freq_hz, f.fmt, f.lds, 26u, f.twn);
            if (!r.ok) { printf("    %-22s INCOMPLETE (hang/timeout)\n", f.name); rc = 3; continue; }
            if (f.nofeed) nofeedTf = r.tf;
            double fill = nofeedTf > 0 ? 100.0*r.tf/nofeedTf : 0.0;
            const char* corr = f.correct ? (r.badSamp ? "*** acc00 BAD" : "acc00 OK") : "(ceiling)";
            printf("    %-24s %8.1f %6.1f%% %6.1f%% %8.1f  mx=%u %s\n",
                   f.name, r.tf, 100.0*r.tf/307.0, fill, (double)r.wall/freq_hz*1e3, r.maxlive, corr);
            if (f.correct && r.badSamp) rc = 3;
        }
        printf("    [128x256 FED > 128x128 FED (correct) => larger tile is the lever past the feed wall; vs HIP 161.]\n");
    } else if (mode == TILEPROBE) {
        // ===== CONFOUND PROBE for the 128x256 (TWN=4) result: FED came out FLAT (~148) vs 128x128, BUT maxlive
        //   dropped 192->64 (8-wave WGs are 3x heavier admission units). Two readings: (a) A-feed isn't the wall,
        //   or (b) the lower occupancy under-hid feed latency and EXACTLY cancelled a real A-reuse win. RESOLVE by
        //   throttling 128x128 (4-wave) occupancy DOWN toward 64 -- its occupancy is LDS-bound (16388 B/WG -> 192),
        //   so OVER-RESERVE LDS (RSRC2 LDS_SIZE; kernel only touches its 16388 -> SAME kernel, fewer co-resident WGs).
        //   If 128x128 FED stays ~148 while maxlive falls 192->~64 => FED occupancy-INSENSITIVE there => 128x256
        //   result is CLEAN: A-feed is NOT the wall (look at per-wave B global_load_tr / LDS-reads / issue mix next).
        //   If 128x128 FED DROPS toward ~148 only AT low maxlive => occupancy matters => the A-reuse DID help. =====
        printf("\n=== TILE WAVE-SATURATION PROBE @65536^2 x K16384 (4x work -> clock committed): TF vs RESIDENT WAVES ===\n");
        printf("    [maxlive=WORKGROUPS; resident_waves = maxlive_wg * waves_per_wg (128x128=4/WG, 128x256=8/WG). Throttle by nWG.\n");
        printf("     DECISIVE clock-fair pair: 128x128 @256w vs 128x256 @256w (both low-occ -> same clock regime, ratio is clean).\n");
        printf("     ~equal => no iso-wave geometry win; 128x256 materially higher => real reuse; lower => 8-wave overhead.]\n");
        printf("    %-22s %8s %7s %9s %8s %10s %s\n",
               "config", "TF", "%307", "maxliveWG", "wavesPWG", "residWaves", "correct");
        const int M=65536, N=65536, Ks=16384;
        struct PR { const char* name; const char* bin; uint32_t lds; uint32_t nwg; int twn; };
        PR rows[] = {
            // throttle by LAUNCHED workgroup count (nWG) -> maxlive<=nWG (all launched WGs co-reside); 8-wave cap can't dodge it.
            { "128x128 nWG=256",  "occ_wggemm2_kwin4_pw4.bin",     16388u, 256u, 2 },  // 768 w (full saturated)
            { "128x128 nWG=128",  "occ_wggemm2_kwin4_pw4.bin",     16388u, 128u, 2 },  // 512 w
            { "128x128 nWG=64",   "occ_wggemm2_kwin4_pw4.bin",     16388u,  64u, 2 },  // 256 w  <- clock-fair ref
            { "128x256 nWG=256",  "occ_wggemm2_tw4_kwin4_pw4.bin", 16388u, 256u, 4 },  // 512 w (full, capped at mx64)
            { "128x256 nWG=32",   "occ_wggemm2_tw4_kwin4_pw4.bin", 16388u,  32u, 4 },  // 256 w  <- THE DECISIVE row
            { "128x256 nWG=16",   "occ_wggemm2_tw4_kwin4_pw4.bin", 16388u,  16u, 4 },  // 128 w
        };
        for (auto& p : rows) {
            WgpResult r = run_wggemm_perf(node, p.bin, M, N, Ks, p.nwg, freq_hz, 4, p.lds, 26u, p.twn);
            uint32_t wpw = 2u * (uint32_t)p.twn;   // TWM=2 -> waves_per_wg = 2*TWN
            if (!r.ok) { printf("    %-22s INCOMPLETE (hang/timeout)\n", p.name); rc = 3; continue; }
            const char* corr = r.badSamp ? "*** acc00 BAD" : "acc00 OK";
            printf("    %-22s %8.1f %6.1f%% %9u %8u %10u  %s\n",
                   p.name, r.tf, 100.0*r.tf/307.0, r.maxlive, wpw, r.maxlive*wpw, corr);
            if (r.badSamp) rc = 3;
        }
        printf("    [READ the clock-fair pair (both 256w): equal=>no iso-wave geometry win; 128x256 higher=>reuse; lower=>8-wave cost.]\n");
    } else if (mode == BLDSPROBE) {
        // ===== B-IN-LDS FEASIBILITY (GPT-directed): B is currently streamed per-wave via global_load_tr; for TWM=2
        //   both M-waves of an N-tile consume the SAME B. Stage it once: only wave_m==0 loads B from global
        //   (global_load_tr -> ds_store), both M-waves ds_load the byte-identical frag back -> HALVES global B traffic.
        //   Built in the DBUF=0 path so "DBUF=0 BLDS vs DBUF=0 non-BLDS" ISOLATES the B-dedup (identical barrier
        //   structure, only the B source differs). KWIN=4 = the absolute best-baseline reference. acc00 GATES correctness
        //   first -- the frag is round-tripped (not re-derived from a plain layout) so it MUST match if addressing is right. =====
        printf("\n=== B-IN-LDS FEASIBILITY PROBE @65536^2 x K16384 (saturated): does M-dedup B-staging beat per-wave B? ===\n");
        printf("    [DBUF=0 BLDS vs DBUF=0 non-BLDS isolates the B-dedup. KWIN=4 = absolute ref. acc00 must be OK first.]\n");
        printf("    %-26s %8s %7s %9s %10s %7s %s\n",
               "config", "TF", "%307", "maxliveWG", "residWaves", "ldsB", "correct");
        const int M=65536, N=65536, Ks=16384;
        struct BP { const char* name; const char* bin; uint32_t lds; };
        BP rows[] = {
            { "KWIN=4 pw4 (cur best)",  "occ_wggemm2_kwin4_pw4.bin",       16388u },  // ~145 (best baseline, 16KB)
            { "KWIN=2 non-BLDS pw1",    "occ_wggemm2_kwin2.bin",            8196u },  // B-global KWIN=2 base (8KB)
            { "KWIN=2 B-in-LDS pw1",    "occ_wggemm2_kwin2_blds_pw1.bin",  16388u },  // NEW: HEADLINE vs KWIN=4 (same 16KB)
            { "KWIN=2 B-in-LDS pw2",    "occ_wggemm2_kwin2_blds_pw2.bin",  16388u },  // NEW
            { "DBUF=0 B-in-LDS (iso)",  "occ_wggemm2_blds.bin",             8704u },  // feasibility ref (+6.3% vs DBUF=0 base)
        };
        for (auto& b : rows) {
            WgpResult r = run_wggemm_perf(node, b.bin, M, N, Ks, 256u, freq_hz, 4, b.lds, 26u, 2);
            if (!r.ok) { printf("    %-26s INCOMPLETE (hang/timeout)\n", b.name); rc = 3; continue; }
            const char* corr = r.badSamp ? "*** acc00 BAD" : "acc00 OK";
            printf("    %-26s %8.1f %6.1f%% %9u %10u %7u  %s\n",
                   b.name, r.tf, 100.0*r.tf/307.0, r.maxlive, r.maxlive*4u, b.lds, corr);
            if (r.badSamp) rc = 3;
        }
        printf("    [READ: BLDS acc00 OK gates correctness. BLDS vs non-BLDS = the B-dedup delta; vs KWIN=4 = absolute.]\n");
    } else if (mode == BW) {
        // ===== MAD-305 CLEAN PM4 BANDWIDTH GATE: prove the raw-PM4 vehicle moves data near the ~640 GB/s
        //   spec before any more GEMM work. Pure streaming, 1 atomic/wave, no LDS/barriers/per-elem drain. =====
        printf("\n=== MAD-305 CLEAN PM4 BANDWIDTH GATE: near-spec HBM BW (~640) BEFORE GEMM (stack rung-1 = 24 = FAIL) ===\n");
        printf("    pure streaming hot loop, 1 atomic/wave (dense worker id), no LDS/barrier/per-elem drain. PASS = hundreds GB/s.\n");
        // --- WGSIZE sweep @ b128 read, constant 4096 workers: occupancy (maxlive) is the suspected wall ---
        printf("  -- WGSIZE sweep (b128 read, 4096 workers, STEPS=512) -> does occupancy drive BW? --\n");
        printf("    %-24s %9s  %-8s %-12s %s\n", "config", "GB/s", "maxlive", "proof", "%peak640");
        struct Wg { const char* name; uint32_t wgsize, nwg; };
        Wg wgs[] = { {"WG=32 (1 wave)",32u,4096u}, {"WG=64 (2 wave)",64u,2048u},
                     {"WG=128 (4 wave)",128u,1024u}, {"WG=256 (8 wave)",256u,512u} };
        for (auto& w : wgs) {
            BwResult r = run_bw(node, "occ_bw_read_b128.bin", freq_hz, 0, 16, 256ull*1024*1024, 512u, w.wgsize, w.nwg);
            if (!r.ok) { printf("  %-24s INCOMPLETE\n", w.name); rc = 3; continue; }
            printf("  %-24s %8.1f  %-8u %-12s %.1f%%\n", w.name, r.gbps, r.maxlive,
                   r.proof?"PASS":"*FAIL", 100.0*r.gbps/640.0);
            if (!r.proof) rc = 3;
        }
        // --- read/copy/write x width, at WG=32 (1-wave, best occupancy) ---
        printf("  -- mode x width @ WG=32, 2048 workers, STEPS=2048 --\n");
        printf("    %-24s %9s  %-22s %-8s %s\n", "config", "GB/s", "proof", "maxlive", "%peak640");
        struct Bw { const char* name; const char* bin; int mode; uint32_t ldw; };
        Bw rows[] = {
            { "read  b32  (UNROLL32)", "occ_bw_read_b32.bin",   0,  4 },
            { "read  b64  (UNROLL16)", "occ_bw_read_b64.bin",   0,  8 },
            { "read  b128 (UNROLL8) ", "occ_bw_read_b128.bin",  0, 16 },
            { "copy  b128 (load+stor)", "occ_bw_copy_b128.bin", 1, 16 },
            { "write b128 (fill)     ", "occ_bw_write_b128.bin",2, 16 },
        };
        for (auto& b : rows) {
            BwResult r = run_bw(node, b.bin, freq_hz, b.mode, b.ldw, 256ull*1024*1024, 2048u, 32u, 2048u);
            if (!r.ok) { printf("  %-24s INCOMPLETE (hang/timeout)\n", b.name); rc = 3; continue; }
            char proof[40];
            snprintf(proof, sizeof proof, "%s wkrs=%u steps%s", r.proof ? "PASS" : "*** FAIL",
                     r.workers, ((uint64_t)r.steps==(uint64_t)r.workers*2048u) ? "=ok" : "BAD");
            printf("  %-24s %8.1f  %-22s %-8u %.1f%%\n", b.name, r.gbps, proof, r.maxlive, 100.0*r.gbps/640.0);
            if (!r.proof) rc = 3;
        }
        // --- MLP-depth scrutiny: is the VRAM ceiling MLP-limited or real? (b64 read, UNROLL 4/8/16, WG=32) ---
        printf("  -- MLP-depth scrutiny (b64 read @ WG=32, verified VRAM): does deeper UNROLL raise BW? --\n");
        struct Mlp { const char* name; const char* bin; };
        Mlp mlps[] = { {"b64 UNROLL=4 ", "occ_bw_read_b64_u4.bin"}, {"b64 UNROLL=8 ", "occ_bw_read_b64_u8.bin"},
                       {"b64 UNROLL=16", "occ_bw_read_b64.bin"} };
        for (auto& m : mlps) {
            BwResult r = run_bw(node, m.bin, freq_hz, 0, 8, 256ull*1024*1024, 2048u, 32u, 2048u);
            if (!r.ok) { printf("  %-24s INCOMPLETE\n", m.name); rc = 3; continue; }
            printf("  %-24s %8.1f  %-22s %-8u %.1f%%\n", m.name, r.gbps, r.proof?"PASS":"*FAIL", r.maxlive, 100.0*r.gbps/640.0);
            if (!r.proof) rc = 3;
        }
        printf("    [GATE: if best << ~600 GB/s the raw-PM4 vehicle is the wall. If UNROLL raises BW -> probe is MLP-limited, true ceiling higher]\n");
    } else if (mode == LDSBOUND) {
        // ===== Phase-2 prep: confirm the raw-PM4 RSRC2.LDS_SIZE granule at the real A-tile size. =====
        printf("\n=== LDS BOUNDARY smoke (write/read LDS[0] and LDS[last] at Phase-2 sizes) ===\n");
        bool a = run_lds_bound(node, "occ_ldsbound.bin", 4096u);   // A single-buffer 128x32 fp8
        bool b = run_lds_bound(node, "occ_ldsbound.bin", 4608u);   // +1 granule (512)
        bool c = run_lds_bound(node, "occ_ldsbound.bin", 8192u);   // A double-buffer headroom
        printf("  -> %s (all sizes round-trip first+last word on all 4 waves => 512 B granule confirmed)\n",
               (a&&b&&c) ? "PASS" : "FAIL");
        if (!(a&&b&&c)) rc = 3;
    } else if (mode == WGLDS) {
        // ===== MAD-305 Phase 1 (pivot): atomic-claim + LDS-broadcast wave-group smoke. Persistent
        // 4-wave workgroups; leader atomic-claims a tile, broadcasts via LDS+barrier; every wave marks.
        // Verifies each tile covered exactly once with 4 correct wave marks. No TGID. =====
        printf("\n=== MAD-305 wave-group Phase 1 SMOKE (atomic-claim + LDS-broadcast, no TGID, no compute) ===\n");
        const int TWM = 2, TWN = 2, WAVES = TWM * TWN;
        const uint32_t LDS = 16;   // one u32 ti broadcast slot (+headroom); units=1 safe for any granule
        struct Sm { const char* name; int M, N; uint32_t nWG; };
        Sm cases[] = {
            { "512x512   (16 tiles)   nWG=4  persistent",  512,  512,   4u },
            { "1024x1024 (64 tiles)   nWG=16 persistent", 1024, 1024,  16u },
            { "4096x4096 (1024 tiles) nWG=64 persistent", 4096, 4096,  64u },
        };
        for (auto& c : cases) {
            WgResult r = run_wglds_smoke(node, "occ_wglds.bin", c.M, c.N, c.nWG, TWM, TWN, LDS);
            uint32_t TOTAL = (uint32_t)(c.M / 128) * (uint32_t)(c.N / 128);
            uint32_t expect = TOTAL * (uint32_t)WAVES;
            if (!r.ok) { printf("  %-38s  INCOMPLETE (hang/timeout)\n", c.name); rc = 3; continue; }
            // mark verification IS the coverage proof; claim counter ends at TOTAL+nWG (one drain claim/WG).
            bool pass = (r.okMarks == expect && r.badMarks == 0 && r.missMarks == 0);
            printf("  %-38s  maxlive=%u WGs  claims=%u(=%u+%u)  marks OK=%u/%u bad=%u miss=%u  %s\n",
                   c.name, r.maxlive, r.total, TOTAL, c.nWG, r.okMarks, expect, r.badMarks, r.missMarks, pass ? "PASS" : "FAIL");
            if (!pass) rc = 3;
        }
    } else if (mode == WGGEMM) {
        // ===== MAD-305 wave-group Phase 1 SMOKE: prove the 4-wave (128-thread) workgroup forms, the
        // lane/wave mapping (wave_m=wid/TWN, wave_n=wid%TWN) is correct, and grid-stride TGID tile
        // decode covers each logical 128x128 tile exactly once. No compute, no LDS, no atomic claim. =====
        printf("\n=== MAD-305 wave-group GRID-STRIDE TGID smoke [DEPRECATED -- TGID dead on raw PM4;\n"
               "    use --wglds-smoke. Retained only as the diagnostic that proved it] ===\n");
        const int TWM = 2, TWN = 2, WAVES = TWM * TWN;
        struct Sm { const char* name; int M, N; uint32_t nWG; };
        Sm diag1[] = { { "DIAG 1024x1024 (64 tiles) nWG=64 -> one-each (no stride)", 1024, 1024, 64u } };
        Sm full[] = {
            { "512x512   (16 tiles)   nWG=8   -> stride x2",  512,  512,  8u },
            { "1024x1024 (64 tiles)   nWG=64  -> one-each",  1024, 1024, 64u },
            { "4096x4096 (1024 tiles) nWG=256 -> stride x4", 4096, 4096, 256u },
        };
        Sm* cases = getenv("WG_DIAG") ? diag1 : full;
        int nCases = getenv("WG_DIAG") ? 1 : 3;
        for (int ci = 0; ci < nCases; ++ci) {
            auto& c = cases[ci];
            WgResult r = run_wggemm_smoke(node, "occ_wggemm_smoke.bin", c.M, c.N, c.nWG, TWM, TWN);
            uint32_t TOTAL = (uint32_t)(c.M / 128) * (uint32_t)(c.N / 128);
            uint32_t expect = TOTAL * (uint32_t)WAVES;
            if (!r.ok) { printf("  %-38s  INCOMPLETE (hang/timeout)\n", c.name); rc = 3; continue; }
            bool pass = (r.okMarks == expect && r.badMarks == 0 && r.missMarks == 0);
            printf("  %-38s  maxlive=%u WGs  marks OK=%u/%u bad=%u miss=%u  %s\n",
                   c.name, r.maxlive, r.okMarks, expect, r.badMarks, r.missMarks, pass ? "PASS" : "FAIL");
            if (!pass) rc = 3;
        }
    } else if (mode == DSWS) {
        // ===== MAD-305 DSWS adaptive wave-role controller. Phase 1 = STATIC 3-role substrate.
        // T1.1 (this) = config + validation refuse-path ONLY; the actual --dsws dispatch + oracle gate
        // wire in at T1.3. Mirrors the WAVESPEC / MBML8COOP brick-guard discipline: a geometry/bin
        // mismatch under bit6-armed dyn is exactly what bricks gfx1201, so we REFUSE (rc=4, no dispatch)
        // on any invalid config or missing bin. =====
        DswsCfg c = parse_dsws_cfg();
        printf("\n=== MAD-305 DSWS 3-role substrate  (nComp=%u nAfeed=%u nBfeed=%u  N=%u  RINGD=%u  LOW=%u HIGH=%u  EPOCH_SHIFT=%u  dyn=%d) ===\n",
               c.nComp, c.nAfeed, c.nBfeed, c.N(), c.ringd, c.low, c.high, c.epochShift, c.dyn);
        // ---- validation refuse-path (spec floors: compute>=1, A-feed>=1, B-feed>=1; band sanity) ----
        if      (c.nComp  < 1) { printf("  *** REFUSE: nComp>=1 required (compute floor); got %u ***\n",  c.nComp);  rc = 4; }
        else if (c.nAfeed < 1) { printf("  *** REFUSE: nAfeed>=1 required (A-feed floor); got %u ***\n", c.nAfeed); rc = 4; }
        else if (c.nBfeed < 1) { printf("  *** REFUSE: nBfeed>=1 required (B-feed floor); got %u ***\n", c.nBfeed); rc = 4; }
        else if (c.ringd  < 1) { printf("  *** REFUSE: RINGD>=1 required; got %u ***\n", c.ringd); rc = 4; }
        else if (c.low > c.high)   { printf("  *** REFUSE: LOW(%u) > HIGH(%u) — invalid watermark band ***\n", c.low, c.high); rc = 4; }
        else if (c.high > c.ringd) { printf("  *** REFUSE: HIGH(%u) > RINGD(%u) — occ is clamped to [0,RINGD] ***\n", c.high, c.ringd); rc = 4; }
        // ---- bin-presence brick-guard (T1.2/T1.3 build it via ./build_dsws.sh). Absent now by design. ----
        char dswsBin[160];
        snprintf(dswsBin, sizeof dswsBin, "occ_dsws_%uc%ua%ub_r%u%s_gd.bin",
                 c.nComp, c.nAfeed, c.nBfeed, c.ringd, c.dyn ? "_dyn" : "");
        if (rc == 0) {
            FILE* fb = fopen(dswsBin, "rb");
            if (fb) fclose(fb);
            else { printf("  *** DSWS kernel bin '%s' NOT BUILT — REFUSING to dispatch "
                          "(build it via ./build_dsws.sh; T1.2/T1.3) ***\n", dswsBin);
                   rc = 4; }
        }
        if (rc == 0) {
            // ===== T1.3 static 3-role dispatch + oracle gate. Role counts are baked into the bin (defsyms),
            //   so the harness only sets WG threads = N*32 (totalWaves) and the bigger DSWS LDS; the C-store
            //   /oracle partition uses P=NCOMP (DSWS compute count). Small tile-multiple oracle shape first
            //   (sub-second, brick-safe); GENDIV (ml8 N are non-pow2). [SUPERVISED at T1.4.] =====
            const int FMc = 2, FNc = 4;                  // DSWS v1 fixed coop tile (baked into the kernel + bin name)
            const int Nwaves = (int)c.N();               // launch N = NCOMP+NAFEED+NBFEED waves/WG
            // Replicate the kernel's LDS_TOTAL_DSWS EXACTLY (RINGD_A defaults to RINGD):
            //   BRING + cons[NCOMP] + (prod,ti,epoch,initflag) + prod_b_hi[NBFEED-1] + A-ring + prod_a[NCOMP] + cons_a[NCOMP]
            uint32_t BRING   = (uint32_t)c.ringd * FNc * 256;
            uint32_t ldsBase = BRING + 4u*c.nComp + 16u;                       // = LDS_TOTAL (the DSWS=0 prefix)
            uint32_t aring   = (uint32_t)c.ringd * c.nComp * FMc * 256;         // RINGD_A * NCOMP * FM * 256
            uint32_t ldsDsws = ldsBase + 4u*(c.nBfeed - 1u) + aring + 8u*c.nComp;
            const uint32_t poolD = getenv("ML8_POOL") ? (uint32_t)atoi(getenv("ML8_POOL")) : 64u;
            const char* onlyShape = getenv("DSWS_ONLY");
            int oMTL = getenv("DSWS_ORACLE_MTL") ? atoi(getenv("DSWS_ORACLE_MTL")) : 4;
            int oNTL = getenv("DSWS_ORACLE_NTL") ? atoi(getenv("DSWS_ORACLE_NTL")) : 8;
            int TM = (int)c.nComp*FMc*16, TN = FNc*16;   // WG tile
            struct SH { const char* name; int M, K, N; };
            SH shapes[] = { {"down   ", 2048, 9216, 2560}, {"down_pf", 512, 9216, 2560} };
            printf("  [dsws] N=%d waves/WG  WGtile=%dx%d  LDS=%uB  bin=%s\n", Nwaves, TM, TN, ldsDsws, dswsBin);
            for (auto& s : shapes) {
                if (onlyShape && !strstr(s.name, onlyShape)) continue;
                int Mo = TM*oMTL, No = TN*oNTL, Ko = 512;          // small tile-multiple oracle shape
                printf("\n  #### DSWS %s  oracle %dx%dx%d  (STORE=1, GENDIV) ####\n", s.name, Mo, No, Ko);
                CoopResult o = run_mbcoop(node, dswsBin, c.dyn, poolD < 64u ? poolD : 64u, Mo, No, Ko,
                                          FMc, FNc, (int)c.nComp, (int)c.ringd, /*fullCheck*/true,
                                          /*useGenDiv*/true, /*reps*/1, /*targetSecs*/0.0,
                                          /*totalWaves*/Nwaves, /*ldsBytesOverride*/ldsDsws);
                if (!o.ok) { printf("  oracle INCOMPLETE (hang/timeout) -> protocol/grow bug; STOP\n"); rc = 3; break; }
                bool clean = (o.badFrags == 0 && o.okFrags > 0);
                printf("  oracle %s  ok=%llu bad=%llu  maxlive=%u\n",
                       clean ? "CLEAN" : "*** BAD (race/math) ***",
                       (unsigned long long)o.okFrags, (unsigned long long)o.badFrags, o.maxlive);
                if (!clean) { rc = 3; break; }
            }
        }
    } else if (mode == GRIND) {
        const int FMc=2, FNc=4;
        const int Mo = getenv("GRIND_M") ? atoi(getenv("GRIND_M")) : 576;
        const int No = getenv("GRIND_N") ? atoi(getenv("GRIND_N")) : 512;
        const int Ko = getenv("GRIND_K") ? atoi(getenv("GRIND_K")) : 2048;
        const float orel = getenv("GRIND_REL") ? atof(getenv("GRIND_REL")) : 5e-3f;   // full-K single write -> TIGHT tier
        const float oabs = getenv("GRIND_ABS") ? atof(getenv("GRIND_ABS")) : 1e-2f;
        const char* gbin = getenv("GRIND_BIN") ? getenv("GRIND_BIN") : "occ_kernel_grind.bin";
        printf("\n=== GRIND control (non-split-K, one-tile-per-WG, full-K, write-once C) ===\n");
        FILE* fb=fopen(gbin,"rb"); if (fb) fclose(fb); else { printf("  *** grind bin '%s' NOT BUILT -> refuse ***\n",gbin); rc=4; }
        if (rc==0) {
            GrindResult o = run_grind(node,gbin,FMc,FNc,Mo,No,Ko,orel,oabs,freq_hz);
            if (!o.ok && o.okFrags==0 && o.badFrags==0) { printf("  grind INCOMPLETE (hang/refuse) -> STOP\n"); rc=3; }
            else if (o.badFrags>0) { printf("  grind oracle *** BAD *** -> STOP\n"); rc=3; }
            else { printf("  grind oracle CLEAN\n"); rc=0; }
        }
    } else if (mode == DSWS2) {
        // ===== MAD-305 DSWS v2 substrate (PLAN_DSWS_SUBSTRATE_V2.md, Task A8). Computes/dry-prints the
        //   super-tile pool params for occ_kernel_dsws.s and, when DSWS2_DRYRUN is unset, launches it via
        //   run_dsws2 (the v2 PM4 launch + tiered-oracle path). DSWS2_DRYRUN=1 -> print params + return
        //   rc=0 WITHOUT touching the GPU (gate 2 of A8: must still dry-print, never dispatch). =====
        DswsCfg c = parse_dsws_cfg();
        const int FNc = 4;                                           // FN is fixed (the shared N-reuse operand)
        // FM is a LEVER (2026-07-14). FM=1 drops NFV 112->80 (Gmax 3->6) and changes the super-tile M
        //   extent from G*16*FM to G*16, so REAL shapes only need M % (G*16) instead of M % (G*32) --
        //   doubling how many rowblks/waves can compute. FMc is a COMPILE-TIME MATCH to the bin: it sets
        //   the resident-A stride, C tile bytes, and oracle addressing. A host FMc that disagrees with the
        //   built FM silently corrupts C (wrong strides -- the oracle catches it as BAD, same discipline as
        //   G/SEGK). ALWAYS rebuild the flow bin with a MATCHING -defsym FM before changing DSWS2_FM.
        const int FMc = getenv("DSWS2_FM") ? atoi(getenv("DSWS2_FM")) : 2;   // default 2 == legacy fixed coop tile
        const int Gv    = getenv("DSWS2_G")    ? atoi(getenv("DSWS2_G"))    : 6;   // M-extent (rowblks/super-tile) = NCOMP_MAX
        const int SEGKv = getenv("DSWS2_SEGK") ? atoi(getenv("DSWS2_SEGK")) : 64;  // split-K segment (K-elements)
        // super-tile geometry (tile-multiple oracle shape, mirroring the --dsws oracle defaults: oMTL/oNTL/Ko).
        const int TMsuper = Gv*16*FMc;                               // super-tile M rows = G*16*FM
        const int TN      = FNc*16;                                  // N-panel cols = FN*16
        const int oMTL = getenv("DSWS2_ORACLE_MTL") ? atoi(getenv("DSWS2_ORACLE_MTL")) : 4;
        const int oNTL = getenv("DSWS2_ORACLE_NTL") ? atoi(getenv("DSWS2_ORACLE_NTL")) : 8;
        // n_kseg/Ko: DSWS2_NKSEG (when set) is the PRIMARY lever -- it derives Ko = SEGKv*n_kseg so the
        //   pool always covers the FULL K range in exactly n_kseg segments (SEGK is a compile-time defsym
        //   baked into the .bin's KSEG_STEPS-unrolled WMMA loop; Ko/SEGKv must stay exact or the resident
        //   A/B staging silently undercounts K). This is how the A8 command forces n_kseg=1 (TIGHT tier)
        //   without needing a separate DSWS2_K=64: `DSWS2_NKSEG=1` -> Ko=SEGKv*1=64 automatically.
        //   Without DSWS2_NKSEG, Ko comes from DSWS2_K (default 512) and n_kseg = Ko/SEGKv as before.
        int n_kseg, Ko;
        if (getenv("DSWS2_NKSEG")) {
            n_kseg = atoi(getenv("DSWS2_NKSEG"));
            Ko = (n_kseg > 0) ? SEGKv * n_kseg : 0;
        } else {
            Ko = getenv("DSWS2_K") ? atoi(getenv("DSWS2_K")) : 512;
            n_kseg = (SEGKv > 0) ? (Ko / SEGKv) : 0;          // = KT/(SEGK/16) (same K-units as the --dsws oracle)
        }
        const int Mo = TMsuper*oMTL, No = TN*oNTL;                   // tile-multiple oracle shape
        const int KT = (SEGKv > 0) ? Ko/16 : 0;
        const int NTL = No / TN;
        const int MTLsuper = Mo / TMsuper;
        const long long TOTAL_super = (long long)MTLsuper * NTL * n_kseg;   // (M/(G*16*FM)) * NTL * n_kseg
        const uint64_t TOTAL64 = (uint64_t)MTLsuper * (uint64_t)NTL;        // coop-compat output-tile count (C sizing)
        uint32_t poolSlots_h = 1u;   // FIX 1: flow N-deep pool / ring D=2 / single-slot
        if (getenv("DSWS2_FLOW"))      poolSlots_h = getenv("FLOW_POOL_N") ? (uint32_t)atoi(getenv("FLOW_POOL_N")) : 3u;
        else if (getenv("DSWS2_RING")) poolSlots_h = 2u;
        // FIX 1 STAGGER: flow per-rowblk accumulator pool (ACC_N banks x FM*FN*1024B), matches kernel ACC_*.
        const uint32_t accN_h = getenv("DSWS2_FLOW") ? (getenv("DSWS2_ACC_N") ? (uint32_t)atoi(getenv("DSWS2_ACC_N")) : 1u) : 0u;
        const uint32_t ldsBytes = 256u + poolSlots_h * ((uint32_t)(FNc*16*SEGKv) + (uint32_t)(Gv*16*FMc*SEGKv))
                                  + accN_h * (uint32_t)(FMc*FNc*1024);
        // A2 tiered oracle thresholds: TIGHT (proven gate) for n_kseg==1, LOOSE (split-K reassoc) for n_kseg>1.
        //   The A8 compare calls oracle_compare(got, ref, n, orel, oabs).
        const float orel = (n_kseg == 1) ? 5e-3f : 3e-2f;
        const float oabs = (n_kseg == 1) ? 1e-2f : 2e-2f;
        const bool dry = getenv("DSWS2_DRYRUN") != nullptr;
        printf("\n=== MAD-305 DSWS v2 substrate (A8 launch path; PLAN_DSWS_SUBSTRATE_V2.md) ===\n");
        printf("  G=%d SEGK=%d FM=%d FN=%d  NCOMP=%u NAFEED=%u NBFEED=%u\n",
               Gv, SEGKv, FMc, FNc, c.nComp, c.nAfeed, c.nBfeed);
        printf("  oracle shape %dx%dx%d  (super-tile %dx%d, KT=%d, NTL=%d, MTLsuper=%d)\n",
               Mo, No, Ko, TMsuper, TN, KT, NTL, MTLsuper);
        printf("  n_kseg=%d  TOTAL_super=%lld  LDS=%uB\n", n_kseg, TOTAL_super, ldsBytes);
        printf("  oracle tier: %s (rel=%.0e abs=%.0e)\n", n_kseg == 1 ? "TIGHT" : "LOOSE", orel, oabs);
        // FIX 3(m): DSWS2 input validation refuse-paths (mirror the --dsws T1.1 refuse-path discipline).
        // Resolve the positional-mix-arg check to a single bool+message BEFORE the dry/refuse chain below,
        // so it can sit as one `else if` link in that chain (dry-run must keep bypassing ALL of these
        // checks -- including this one -- exactly like it already bypasses the degenerate-geometry and
        // Gv/SEGK checks; that's an existing, load-bearing contract: DSWS2_DRYRUN never touches the GPU
        // AND never refuses, it just prints whatever params were computed).
        bool posMixBad = false; char posMixMsg[256] = {0};
        if (g_posMixArg[0]) {
            uint32_t pc = 0, pa = 0, pb = 0;
            if (sscanf(g_posMixArg, "%uc%ua%ub", &pc, &pa, &pb) != 3) {
                posMixBad = true;
                snprintf(posMixMsg, sizeof posMixMsg,
                         "unrecognized positional arg '%s' (expected a role-mix like '4c2a2b', or no "
                         "positional arg at all -- role counts come from DSWS_NCOMP/DSWS_NAFEED/DSWS_NBFEED)",
                         g_posMixArg);
            } else if (pc != c.nComp || pa != c.nAfeed || pb != c.nBfeed) {
                // a positional role-mix token (e.g. "4c2a2b") was given on the command line. The only built
                // v2 bin is compile-time-fixed at NCOMP=4/NAFEED=2/NBFEED=2 (build_dsws.sh mk2); the bin
                // filename is actually picked from c.nComp/nAfeed/nBfeed (the DSWS_NCOMP/AFEED/BFEED env
                // config), NOT from this positional token -- previously the token was silently ignored, so
                // a user passing a mix that disagreed with the active env config got the WRONG config with
                // no warning.
                posMixBad = true;
                snprintf(posMixMsg, sizeof posMixMsg,
                         "positional role-mix arg '%s' (%uc%ua%ub) does not match the active DSWS_NCOMP/"
                         "DSWS_NAFEED/DSWS_NBFEED config (%uc%ua%ub) -- set the env vars to match or drop "
                         "the positional arg", g_posMixArg, pc, pa, pb, c.nComp, c.nAfeed, c.nBfeed);
            }
        }
        if (dry) { printf("  [DSWS2_DRYRUN] params only -- NO GPU dispatch.\n"); rc = 0; }
        else if (c.nComp < 1) { printf("  *** REFUSE: nComp>=1 required (compute floor); got %u ***\n",  c.nComp);  rc = 4; }
        else if (c.nAfeed < 1) { printf("  *** REFUSE: nAfeed>=1 required (A-feed floor); got %u ***\n", c.nAfeed); rc = 4; }
        else if (c.nBfeed < 1) { printf("  *** REFUSE: nBfeed>=1 required (B-feed floor); got %u ***\n", c.nBfeed); rc = 4; }
        else if (c.N() != c.nComp + c.nAfeed + c.nBfeed) {
            // role-floor/SUM check: N() (the launched wave count, WAVES_LAUNCH downstream) must equal the
            // sum of the role counts actually used to size/decode the dispatch -- guards against a future
            // refactor desyncing N() from its components (currently tautological by construction).
            printf("  *** REFUSE: role-count sum mismatch (N()=%u != nComp+nAfeed+nBfeed=%u) ***\n",
                   c.N(), c.nComp + c.nAfeed + c.nBfeed);
            rc = 4;
        } else if (posMixBad && !getenv("DSWS2_FLOW")) {
            printf("  *** REFUSE: %s ***\n", posMixMsg);
            rc = 4;
        } else if (Mo <= 0 || No <= 0 || Ko <= 0 || n_kseg <= 0 || TOTAL_super <= 0) {
            printf("  *** REFUSE: degenerate geometry (Mo=%d No=%d Ko=%d n_kseg=%d TOTAL_super=%lld) ***\n",
                   Mo, No, Ko, n_kseg, TOTAL_super);
            rc = 4;
        } else if (TOTAL64 > 0xFFFFFFFFull || (uint64_t)TOTAL_super > 0xFFFFFFFFull) {
            printf("  *** REFUSE: pool size overflows uint32_t (TOTAL=%llu TOTAL_super=%lld) -- occ[20]'s claim "
                   "counter and the kernel's sti are both 32-bit ***\n", (unsigned long long)TOTAL64, TOTAL_super);
            rc = 4;
        } else if ( getenv("DSWS2_FLOW")
                        ? !((Gv >= 2 && Gv <= 32) &&
                            (SEGKv == 16 || SEGKv == 32 || SEGKv == 64 || SEGKv == 128 || SEGKv == 256) &&
                            (FMc == 1 || FMc == 2))                    // flow: FM=1 (Gmax->6) or FM=2 (legacy) -- bin must match
                        : (Gv != 6 || SEGKv != 64 || FMc != 2) ) {     // non-flow: fixed coop tile is always FM=2
            // G RAISED TO 16 (2026-07-13 night). G == ACC_N == the number of rowblks in a super-tile ==
            //   THE NUMBER OF WAVES THAT CAN EVER COMPUTE CONCURRENTLY (SL_RBNEXT hands out 0..ACC_N-1).
            //   At G=4 only 4 of 24 waves per WG could ever run a WMMA -- 83% of the fleet was
            //   GEOMETRICALLY FORBIDDEN from computing, which is what the 93.6% coast-frac really was.
            //   It also meant 4*112 = 448 VGPRs of a ~1536 budget, so the VGPR budget could NEVER bind
            //   and grow-fail was structurally pinned to 0 -- which is why the stagger looked dead.
            //   G was capped at 4 by the LDS accumulator banks (ACC_N*8192 = 64KB at G=8). K-DEPTH J
            //   makes WOFLUSH cheap again (J-fold fewer atomics), the banks go away, and G is free.
            // TILE GEOMETRY IS A LEVER (2026-07-13). Measured: the C flush is 83-97% of runtime, and
            //   flush/WMMA = 128/SEGK -- FM*FN cancels, so SEGK (K-depth per super-tile) is the ONLY
            //   knob on the flush:compute ratio. SEGK is capped by the 32KB operand pool
            //   (OPSTRIDE = SEGK*16*(FN + G*FM)), so raising SEGK REQUIRES lowering G. Hence the flow
            //   path must accept G in [2,6] x SEGK in {32,64,128,256}, not just G=6.
            //   The guard below still stands for the non-flow bin (fixed G=6 SEGK=64 geometry).
            //   NOTE this only widens what the HOST will *accept*: the bin must still be BUILT with
            //   matching -defsym G/SEGK. Always rebuild immediately before running (build_flow.sh
            //   rm -f's its bin on failure, so a failed build can never leave a runnable stale artifact).
            // G/SEGK are compile-time defsyms baked into the kernel's instruction immediates
            // (KSEG_STEPS-unrolled WMMA loop, resident-LDS strides). A host geometry that disagrees
            // with the bin's compiled G/SEGK silently corrupts the resident-A/B staging/compute
            // addressing (wrong strides, not a bounds violation the gate below would catch). REFUSE
            // rather than guess; rebuild a matching bin before changing these envs.
            // FIX 1 STAGGER: the flow bin (build_flow.sh) can now be built with SEGK=32 (halves the
            //   operand footprint so the g=6 write-once accumulator banks fit LDS) -- so SEGK=32 is
            //   allowed ONLY on the DSWS2_FLOW path, where the run must pass a matching DSWS2_SEGK=32.
            printf("  *** REFUSE: DSWS2_G=%d DSWS2_SEGK=%d DSWS2_FM=%d mismatches the built bin's compile-time geometry "
                   "(non-flow: G=6 SEGK=64 FM=2; flow: G in [2,32] SEGK in {16,32,64,128,256} FM in {1,2}) "
                   "-- REFUSING geometry/bin mismatch ***\n", Gv, SEGKv, FMc);
            rc = 4;
        } else {
            char dswsBin[160];
            // EMERGENT economy (flow): no baked mix. Derive the launch pool from FLOW_WAVES (host-set),
            //   cap at 30 (coordinator mailbox squat), sanity-check against the lean-fit budget.
            uint32_t Wlaunch = getenv("FLOW_WAVES") ? (uint32_t)atoi(getenv("FLOW_WAVES")) : 8u;   // 8 = proven-safe
            if (Wlaunch < 4)  Wlaunch = 4;                 // floor(wid0/1/2) + >=1 compute
            // SAFETY: W>~8 at POOL_N=1 overcommits the SIMD dyn-VGPR pool at launch -> some waves' s_alloc_vgpr 32
            //   fails -> the ONLY in-kernel exit (s_endpgm on a failed-alloc wave) corrupts the pool -> OOB page
            //   fault -> MODE1 brick (2026-07-05). Until that launch-starvation is root-caused, keep W_launch <= 8.
            if (Wlaunch > 30) { printf("  [flow] FLOW_WAVES=%u > 30 (coord cap) -> clamping to 30\n", Wlaunch); Wlaunch = 30; }
            {
                const uint32_t VB = getenv("FLOW_VBUDGET") ? (uint32_t)atoi(getenv("FLOW_VBUDGET")) : 1536u;
                const uint32_t leanFit = (VB - (112u - 32u)) / 32u;   // (VBUDGET-(NFV-VLEAN))/VLEAN
                if (Wlaunch > leanFit)
                    printf("  [flow] WARNING FLOW_WAVES=%u exceeds lean-fit=%u for VBUDGET=%u (bin's .error would catch a real overflow)\n", Wlaunch, leanFit, VB);
            }
            // FIX 1: DSWS2_FLOW -> flow bin, DSWS2_RING -> ring bin, else single-slot bin.
            if (getenv("DSWS2_FLOW"))
                snprintf(dswsBin, sizeof dswsBin, "occ_dsws2_w%u_flow_gd.bin", Wlaunch);   // Wlaunch == built WAVES
            else if (getenv("DSWS2_RING"))
                snprintf(dswsBin, sizeof dswsBin, "occ_dsws2_%uc%ua%ub_ring_gd.bin", c.nComp, c.nAfeed, c.nBfeed);
            else
                snprintf(dswsBin, sizeof dswsBin, "occ_dsws2_%uc%ua%ub_gd.bin", c.nComp, c.nAfeed, c.nBfeed);
            FILE* fb = fopen(dswsBin, "rb");
            if (fb) fclose(fb);
            else {
                printf("  *** DSWS2 kernel bin '%s' NOT BUILT -- REFUSING to dispatch (build it via "
                       "./build_dsws.sh, mk2) ***\n", dswsBin);
                rc = 4;
            }
            if (rc == 0) {
                const bool isFlow = getenv("DSWS2_FLOW");   // flow: launch Wlaunch (emergent), mix args unused
                Dsws2Result o = run_dsws2(node, dswsBin,
                                           isFlow ? Wlaunch : c.nComp, isFlow ? 0u : c.nAfeed, isFlow ? 0u : c.nBfeed,
                                           Gv, SEGKv, FMc, FNc, Mo, No, Ko, orel, oabs, freq_hz);
                if (!o.ok && o.okFrags == 0 && o.badFrags == 0) {
                    printf("  dsws2 INCOMPLETE (hang/timeout/refused before oracle) -> protocol/geometry bug; STOP\n");
                    rc = 3;
                } else if (o.badFrags > 0) {
                    printf("  dsws2 oracle *** BAD (race/math) *** -> STOP\n");
                    rc = 3;
                } else {
                    printf("  dsws2 oracle CLEAN\n");
                    rc = 0;
                }
            }
        }
    } else {
        // Default: dyn-VGPR cap probe. Test dyn correctness at increasing s_alloc footprints:
        //   light NACC=8  -> s_alloc 96  (<=128, expected OK)
        //   heavy NACC=16 -> s_alloc 160 (>128,  suspected over-cap -> zeros)
        printf("=== dyn-VGPR correctness INVESTIGATION (light s_alloc 96; KDEPTH=1) ===\n");
        printf("  Is the occ-16 'corruption' deterministic silicon, or a flaky harness race?\n");
        printf("  Each grid run 6x; report OK/CORRUPT counts + whether all-zero or partial.\n\n");
        printf("  nWaves  maxlive  occ/SIMD   OK/6   note\n");
        const uint32_t grids[] = {1536, 1792, 1920, 2048, 2176, 2560, 4096};
        for (uint32_t g : grids) {
            int ok = 0, allzero = 0; uint32_t mx = 0;
            for (int r = 0; r < 6; ++r) {
                RunResult rr = run_variant(node, "occ_n8_d1.bin", true, fragIn, g, 4, 1);
                if (!rr.ok) continue;
                mx = rr.maxlive;
                if (wmma_ok(rr.D)) ok++;
                else { bool z = true; for (int i=0;i<256;i++) if (rr.D[i]!=0.0f) { z=false; break; } if (z) allzero++; }
            }
            printf("  %6u  %5u    %6.2f    %d/6   %s\n", g, mx, mx/128.0, ok,
                   ok==6 ? "all OK" : (allzero==(6-ok) ? "fails = all-zero" : "fails = partial/garbage"));
        }
        printf("\n  heavy dyn (s_alloc 160) @ 4 waves x6 (no oversubscription -> isolates per-wave cap):\n  ");
        { int ok=0; for(int r=0;r<6;r++){ RunResult rr=run_variant(node,"occ_n16_d1.bin",true,fragIn,4,4,1); if(rr.ok&&wmma_ok(rr.D)) ok++; }
          printf("OK %d/6  -> if 0/6, the per-wave dyn-VGPR cap is genuinely < 160\n", ok); }
        rc = 0;
    }

    hsaKmtCloseKFD();
    return rc;
}
