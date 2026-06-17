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
struct MbgResult { bool ok=false; uint32_t maxlive=0, total=0, okTiles=0, nChecked=0; uint64_t wall=0; double secs=0; uint32_t phase[6]={0,0,0,0,0,0}; };
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
    GpuBuf Ad  = AllocGpu(node, (Ah.size() + 0xFFF) & ~0xFFFull, false, true);
    GpuBuf Bd  = AllocGpu(node, (Bshufh.size() + 0xFFF) & ~0xFFFull, false, true);
    uint64_t cbytes = ((uint64_t)TOTAL * 1024 + 0xFFF) & ~0xFFFull;
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

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered: if a raw-PM4 run hangs and gets SIGKILL'd, the log still shows WHERE it died
    enum { CORRECT, PRONG1, PRONG2, PRONG3, COMBINED, TIMERCHECK, PROBE, MICROBATCH, MBGEMM, MBPROF, MERGE } mode = CORRECT;
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
        else if (!strcmp(argv[i], "--mbprof"))    mode = MBPROF;
        else if (!strcmp(argv[i], "--merge"))     mode = MERGE;
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
        printf("  tile  reuse  vgpr  mode    oracle    span_ms  TFLOPS  %%307  x143\n");
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
            printf("  %-5s %.2f  %4d  %-6s  %s%u/%u  %8.3f  %6.1f  %4.1f  %.2f\n",
                   t.tag, reuse, fatregs, t.dyn?"dyn":"static",
                   o.okTiles==o.nChecked ? "OK " : "BAD", o.okTiles, o.nChecked,
                   (double)r.wall/freq_hz*1e3, tf, 100*tf/307.0, tf/143.0);
        }
        // ISOLATION: same kernel, operands loaded ONCE & reused (no per-K feed) -> framework ceiling.
        if (!fat) {
            printf("\n  [NO-FEED 2x4 b32 -- operands reused, ZERO per-K load]    K     KT   TFLOPS  span_ms\n");
            const int nfKs[] = {2048, 8192, 32768};
            for (int K : nfKs) {
                MbgResult r = run_mbgemm(node, "occ_mbgemm_2x4_b32_nf.bin", true, 1152u, 2048,2048,K, 2,4, false);
                if (!r.ok) continue;
                uint32_t TOTAL = (uint32_t)(2048/32)*(2048/64); int KT = K/16;
                double work = (double)TOTAL * 8 * KT;
                double tf = work * (2.0*16*16*16) * freq_hz / (double)r.wall / 1e12;
                printf("                                                       %6d %5d  %6.1f  %.3f\n",
                       K, KT, tf, (double)r.wall/freq_hz*1e3);
            }
            printf("    CLIMBS with K => per-tile FRAMEWORK overhead is the wall (amortized by work/tile), NOT the\n");
            printf("    issue port. The fed K-sweep stayed flat only because every K-step re-pays the feed.\n");
        }
        printf("\n  TF climbs with reuse -> tile size is the lever; push bigger (umr->8x8, reuse 4.0).\n");
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
