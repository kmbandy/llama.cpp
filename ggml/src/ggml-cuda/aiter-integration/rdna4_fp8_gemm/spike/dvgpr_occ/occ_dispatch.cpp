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
#include "fp8_oracle.h"
#include "frag_layout.h"

// ---- the dgpu flag the vendored PM4 encoder asks for (same as MAD-304) -----
static bool g_is_dgpu = true;
bool hsakmt_is_dgpu() { return g_is_dgpu; }

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
struct MbgResult { bool ok=false; uint32_t maxlive=0, total=0, okTiles=0, nChecked=0; uint64_t wall=0; double secs=0; uint32_t phase[6]={0,0,0,0,0,0}; };
// When set (DYNFAT1 single-shot): run_mbgemm builds EVERYTHING (operands, queue, all PM4
// packets) then BLOCKS on this gate file before RingSubmit, so the volatile umr cap-flip can
// be applied <1s before dispatch and cannot revert in an idle gap. Cleared by the caller.
// g_readyFile is created the instant prep is done, so an operator one-liner can WAIT for it,
// then flip the cap + touch g_gateFile atomically -> cap is fresh (<100ms) at dispatch.
static const char* g_gateFile  = nullptr;
static const char* g_readyFile = nullptr;

static MbgResult run_mbgemm(uint32_t node, const char* isaPath, bool dynvgpr, uint32_t nWaves,
                            int M, int N, int K, int FM, int FN, bool fullCheck) {
    MbgResult res;
    int TMr = 16 * FM, TNc = 16 * FN, MTL = M / TMr, NTL = N / TNc, KT = K / 16, NT = N / 16;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not power of two\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};   // 1, 2, .5, -1, -2, -.5
    std::vector<uint8_t> Ah((size_t)M * K), Bh((size_t)K * N), Bshufh((size_t)K * N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i * 7 + i / (size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i * 5 + (i / (size_t)N) * 3) % 6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
        (uint32_t)(NTL-1), (uint32_t)log2NTL, (uint32_t)(FN*256), 0 };  // s12 mask, s13 log2, s14 FNx256
    uint32_t dispInit = BuildDispatchInitiator();
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

    // ---- DYNFAT1 GATE: all prep done (operands/queue/ring staged), GPU idle, cap not yet needed.
    // Block until the operator flips the umr cap AND signals (one atomic command), so dispatch
    // fires <1s after the flip -- the volatile cap has no idle window to revert in. ----
    if (g_gateFile) {
        remove(g_gateFile);   // clear any premature touch: only a signal AFTER this point (fresh cap) counts
        if (g_readyFile) { FILE* rf = fopen(g_readyFile, "w"); if (rf) fclose(rf); }  // announce: prep done
        fprintf(stderr, "\n[DYNFAT1] PREP COMPLETE (operands+queue+ring staged). GPU idle; cap NOT needed yet.\n");
        fprintf(stderr, "[DYNFAT1] Operator one-liner (waits for prep, flips cap, signals -- run via ! anytime):\n");
        fprintf(stderr, "    while [ ! -e %s ]; do sleep 0.05; done && sudo /home/kmbandy/GitHub/umr/build/src/app/umr -i 1 -w '*.*.regSQ_DYN_VGPR' 0x1ff && touch %s\n",
                g_readyFile, g_gateFile);
        fprintf(stderr, "[DYNFAT1] waiting for gate (dispatch fires <100ms after the cap flip)...\n");
        for (;;) { if (access(g_gateFile, F_OK) == 0) break; struct timespec ts={0,20000000}; nanosleep(&ts,nullptr); }
        remove(g_gateFile); if (g_readyFile) remove(g_readyFile);
        fprintf(stderr, "[DYNFAT1] gate seen -> RingSubmit NOW.\n");
    }
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
        fprintf(stderr, "\n*** MBGEMM TIMEOUT (%s): live=%u maxlive=%u nextTile=%u (queue may be hung) ***\n",
                isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[4];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.secs = t1 - t0;

    // ---- per-tile oracle: reference D = sum_kt A_block . B_block (chained wmma_ref, D=A*B+C) ----
    const float* Cf = (const float*)C.ptr; uint32_t okc=0, checked=0;
    uint32_t stride = fullCheck ? 1u : (TOTAL > 256 ? TOTAL / 256u : 1u);
    for (uint32_t ti = 0; ti < TOTAL; ti += stride) {
        int tc = ti & (NTL - 1), tr = ti >> log2NTL;
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
                                    int useAtr = 0, int TWN = 2, int FNt = -1) {
    WgcResult res;
    const int FM = FMt, FN = (FNt < 0 ? FMt : FNt), TWM = 2, WAVES = TWM*TWN, TBK = 32;
    const int TM = TWM*FM*16, TN = TWN*FN*16;     // claimed tile = (TWM*FM*16)x(TWN*FN*16)
    int NTL = N / TN, MTL = M / TM, NT = N / 16, NTILES = K / TBK;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not pow2\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Bshufh((size_t)K*N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);
    std::vector<uint8_t> Ashufh;   // ANOLDSTR oracle: A-shuf feed (plain Ah kept for the chained wmma_ref reference)
    if (useAtr) { Ashufh.resize((size_t)M*K); mbg_preshuffle_A(Ah.data(), Ashufh.data(), M, K); }

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, (uint32_t)(NT*256), (uint32_t)(NTL-1), (uint32_t)log2NTL,
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
        int trow = ti >> log2NTL, tcol = ti & (NTL - 1);
        for (int wid = 0; wid < WAVES; ++wid) {
            int wm = wid / TWN, wn = wid % TWN;
            for (int mi = 0; mi < FM; ++mi) for (int ni = 0; ni < FN; ++ni) {
                int rowbase = trow*TM + wm*(FM*16) + mi*16;
                int colbase = tcol*TN + wn*(FN*16) + ni*16;
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
                size_t foff = (size_t)ti*(size_t)(WAVES*FM*FN*256) + (size_t)wid*(size_t)(FM*FN*256) + (size_t)frag*256;   // float index
                float D[256]; unpack_D(Cf + foff, D);
                bool good=true;
                for (int i=0;i<256;i++) if (std::fabs(D[i]-Cacc[i]) > 5e-3f*std::fabs(Cacc[i])+1e-2f) { good=false; break; }
                if (good) res.okFrags++; else res.badFrags++;
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
                                 int TWN = 2, int useAtr = 0, int FNt = -1) {
    WgpResult res;
    if (FNt < 0) FNt = FMt;                        // FNt<0 -> square per-wave tile (FN=FM); set explicitly for 8x2 etc.
    const int TWM = 2, WAVES = TWM*TWN;           // TWM fixed; TWN grows the N-wave count (8 waves @ TWN=4)
    const int TM = TWM*FMt*16, TN = TWN*FNt*16;   // claimed tile = (TWM*FM*16)x(TWN*FN*16)  (128x256 @ TWN=4)
    int NTL = N / TN, MTL = M / TM, NT = N / 16, NTILES = K / 32;
    uint32_t TOTAL = (uint32_t)MTL * NTL;
    if ((NTL & (NTL - 1)) != 0) { fprintf(stderr, "  NTL=%d not pow2\n", NTL); return res; }
    int log2NTL = 0; while ((1 << log2NTL) < NTL) ++log2NTL;

    static const uint8_t NICE[6] = {0x38,0x40,0x30,0xB8,0xC0,0xB0};
    std::vector<uint8_t> Ah((size_t)M*K), Bh((size_t)K*N), Bshufh((size_t)K*N);
    for (size_t i = 0; i < Ah.size(); ++i) Ah[i] = NICE[(i*7 + i/(size_t)K) % 6];
    for (size_t i = 0; i < Bh.size(); ++i) Bh[i] = NICE[(i*5 + (i/(size_t)N)*3) % 6];
    mbg_preshuffle_B(Bh.data(), Bshufh.data(), K, N);
    std::vector<uint8_t> Ashufh;   // ANOLDSTR: A-shuf feed (mbg_preshuffle_A); plain Ah kept for the acc00 reference
    if (useAtr) { Ashufh.resize((size_t)M*K); mbg_preshuffle_A(Ah.data(), Ashufh.data(), M, K); }

    size_t isaLen = 0; uint8_t* isaBytes = ReadFile(isaPath, &isaLen);
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    uint32_t reslim[1]={0}, tmpring[1]={0}, restart[4]={0,0,0,0};
    uint32_t userdata[16] = {
        (uint32_t)occVa,(uint32_t)(occVa>>32), (uint32_t)aVa,(uint32_t)(aVa>>32),
        (uint32_t)bVa,(uint32_t)(bVa>>32), (uint32_t)cVa,(uint32_t)(cVa>>32),
        (uint32_t)K, (uint32_t)(NT*256), (uint32_t)(NTL-1), (uint32_t)log2NTL,
        (uint32_t)NTILES, TOTAL, 0, 0 };
    if (useAtr) { uint64_t asVa=(uint64_t)AshufD.ptr; userdata[2]=(uint32_t)asVa; userdata[3]=(uint32_t)(asVa>>32); userdata[14]=(uint32_t)((uint64_t)M*16); } // ANOLDSTR: s2:3=Ashuf, s14=MT*256
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
        if (now - t0 > 40.0) break;
    }
    if (!done) {
        fprintf(stderr, "\n*** WGGEMM PERF TIMEOUT (%s): live=%u maxlive=%u claim=%u ***\n", isaPath, occW[0], occW[1], occW[5]);
        CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
        if (useAtr) FreeGpu(AshufD);
        FreeGpu(ring.buf); FreeGpu(fence); FreeGpu(C); FreeGpu(Bd); FreeGpu(Ad); FreeGpu(occ); FreeGpu(isa);
        return res;
    }
    res.ok = true; res.maxlive = occW[1]; res.total = occW[5];
    { uint32_t gs=occW[2], ge=occW[3]; res.wall = (ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    res.tf = (res.wall > 0) ? (2.0*(double)M*N*K * freq_hz / (double)res.wall / 1e12) : 0.0;

    // ---- sampled acc[0][0] correctness (cheap; ~16 tiles) ----
    const float* Cf = (const float*)C.ptr;
    uint32_t tstride = TOTAL > 16 ? TOTAL/16u : 1u;
    for (uint32_t ti = 0; ti < TOTAL; ti += tstride) {
        int trow = ti >> log2NTL, tcol = ti & (NTL - 1);
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
    GpuBuf isa=AllocGpu(node,(isaLen+0xFFF)&~0xFFFull,true,false);
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
    GpuBuf isa=AllocGpu(node,(isaLen+0xFFF)&~0xFFFull,true,false);
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
    GpuBuf isa=AllocGpu(node,(isaLen+0xFFF)&~0xFFFull,true,false);
    GpuBuf occ=AllocGpu(node,0x1000,false,true);
    GpuBuf Sd =AllocGpu(node,(BUFSZ+0xFFF)&~0xFFFull,false,true,/*deviceLocal*/true);   // VRAM stream src
    GpuBuf Dd =AllocGpu(node,(BUFSZ+0xFFF)&~0xFFFull,false,true,/*deviceLocal*/true);   // VRAM write/copy dst
    GpuBuf sink=AllocGpu(node,0x1000,false,true);
    GpuBuf fence=AllocGpu(node,0x1000,false,true);
    if (!(Sd.vram && Dd.vram)) {   // VRAM GUARD: the BW probe MUST measure device-local memory, not PCIe/GTT
        fprintf(stderr, "\n*** BW VRAM GUARD FAILED (%s): src/dst not device-local (S=%d D=%d) ***\n", isaPath, Sd.vram, Dd.vram);
        abort();
    }
    memcpy(isa.ptr,isaBytes,isaLen); free(isaBytes);
    memcpy(Sd.ptr,src.data(),BUFSZ);
    volatile uint32_t* occW=(volatile uint32_t*)occ.ptr;
    volatile uint32_t* fenceW=(volatile uint32_t*)fence.ptr;
    for (int i=0;i<20;i++) occW[i]=0; occW[2]=0xFFFFFFFFu; *fenceW=0;

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
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_START_X,dims,8));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_LO,pgm,6));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1,rsrc,2));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESOURCE_LIMITS,reslim,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_TMPRING_SIZE,tmpring,1));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_RESTART_X,restart,4));
    RingPlace(ring,PM4SetShaderRegPacket(mmCOMPUTE_USER_DATA_0,userdata,16));
    RingPlace(ring,PM4DispatchDirectPacket(nWG*WGSIZE,1,1,dispInit));
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
        FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(sink);FreeGpu(Dd);FreeGpu(Sd);FreeGpu(occ);FreeGpu(isa); return res; }
    res.ok=true; res.maxlive=occW[1]; res.workers=occW[5]; res.steps=occW[6]; res.chk=occW[7];
    { uint32_t gs=occW[2],ge=occW[3]; res.wall=(ge>=gs)?(uint64_t)(ge-gs):((uint64_t)ge+0x100000000ull-(uint64_t)gs); }
    double secs=(double)res.wall/freq_hz;
    uint64_t per = (uint64_t)res.workers * STEPS * STEP;       // bytes moved per direction
    res.bytes = (mode==1) ? 2u*per : per;                     // copy moves 2x (load+store)
    res.gbps = secs>0 ? (double)res.bytes/secs/1e9 : 0.0;
    // proof: every worker did exactly STEPS steps (occ[6]==workers*STEPS) AND (read/copy) checksum != 0
    bool cntOK = ((uint64_t)res.steps == (uint64_t)res.workers * STEPS) && res.workers>0;
    res.proof = cntOK && ((mode==2) ? true : (res.chk != 0));
    CHECK(hsaKmtDestroyQueue(ring.res.QueueId));
    FreeGpu(ring.buf);FreeGpu(fence);FreeGpu(sink);FreeGpu(Dd);FreeGpu(Sd);FreeGpu(occ);FreeGpu(isa);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa  = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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
    GpuBuf isa = AllocGpu(node, (isaLen + 0xFFF) & ~0xFFFull, true, false);
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

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered: if a raw-PM4 run hangs and gets SIGKILL'd, the log still shows WHERE it died
    enum { CORRECT, PRONG1, PRONG2, PRONG3, COMBINED, TIMERCHECK, PROBE, MICROBATCH, MBGEMM, MBSAT, DYNFAT1, MBPROF, MERGE, WGGEMM, SGPRPROBE, WGLDS, LDSBOUND, WGGEMM2, WGPERF, WG2X2, NFUNROLL, NFOCC, NFBF, BANDSWP, FEEDPIPE, FEEDLADDER, FEEDBTR, FEEDPROF, FEEDSTAG, FEEDPB, STACK, BW, BASELINES, SUSTAIN, KWIN, KWINORACLE, TILEPROBE, BLDSPROBE, BTR128, ANOLDS, ANOLDSTR, WAVESWEEP, OCCSWEEP, REUSE82, REUSE82TW2, REUSE82KW2, VGPR82, BLDS82, WALL82 } mode = CORRECT;
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
        else if (!strcmp(argv[i], "--wall82"))        mode = WALL82;
        else if (!strcmp(argv[i], "--fat"))       fat = true;
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
            {12, 128/8, "occ_n12fed_d0.bin", "occ_n12fed_d1.bin"},
            {16, 160/8, "occ_n16fed_d0.bin", "occ_n16fed_d1.bin"},
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
        { const uint32_t K=16384, TOTAL=4096, POOL=1536;   // occ 12 (1536x96=1152<file): wedge-safe, no exact-fill edge
          for (int dv=0; dv<2; ++dv) {
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
            {2,4,true, "2x4", 8192, 8192, 1152u, 1.33},   // saturated ref
            {2,4,true, "2x4",16384,16384,1152u, 1.33},   // plateau point
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
            if (c.dyn && !fat) continue;                                     // dyn tiles need umr flip -> --fat gate
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
