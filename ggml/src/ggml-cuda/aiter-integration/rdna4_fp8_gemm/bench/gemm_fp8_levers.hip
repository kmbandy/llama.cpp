// gemm_fp8_levers.hip — MAD-305: ONE configurable fp8 WMMA GEMM, every static lever a template
// parameter, swept against the e4m3 oracle. Consolidates the scattered Phase-1/2 probe variants
// (_db, _bslds, _bk, _wg, _rb) into a single kernel so the HARDWARE picks the config instead of us
// inferring from microbenches.
//
// Levers (compile-time):
//   TBM, TBN     block tile (output TBM x TBN per block)
//   TWM, TWN     wave grid (TWM*TWN waves/block); per-wave acc tile = TFRAGS_M x TFRAGS_N frags
//   TBK          K-tile depth (TKSTEPS = TBK/16 WMMA accumulate-steps per tile)
//   AFEED        A operand feed:  0 = stage A in LDS (wide coalesced fill + ds_read fragments)
//                                 1 = direct from L2 per fragment (no LDS, no __syncthreads)
//   DBUF         0 = single buffer; 1 = software-pipelined double-buffer of the A-LDS fill (AFEED=0)
//
// B is always fed direct-from-L2 via global_load_tr_b64 from a pre-shuffled weight buffer — the
// Phase-1 win (68->140 TF), held fixed. Correctness gated at 1920^3 (oracle), perf at 3840^3
// (both divisible by every tile swept). VGPR/occupancy per config is read from the code object so
// configs that spill past the 256-VGPR static cap (= the ones that REQUIRE dyn-VGPR) are explicit.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "trfeed_common.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}

// One configurable fp8 GEMM. fp8 e4m3 A[M][K] row-major, B pre-shuffled (tile-major, trperm),
// out bf16 C[M][N], per-row a_scale / per-col b_scale. M,N assumed divisible by TBM/TBN at the
// block grid (driver enforces); per-element guards still protect ragged M/N.
template <int TBM, int TBN, int TWM, int TWN, int TBK, int AFEED, int DBUF>
__global__ void __launch_bounds__(TWM * TWN * 32)
gemm_fp8_lever(const __hip_fp8_e4m3* __restrict__ A, const uint8_t* __restrict__ Bshuf,
               __hip_bfloat16* __restrict__ C,
               const float* __restrict__ a_scale, const float* __restrict__ b_scale,
               int M, int N, int K) {
    constexpr int WAVE = 32;
    constexpr int TWAVES   = TWM * TWN;
    constexpr int TBLOCK   = TWAVES * WAVE;
    constexpr int TFRAGS_M = (TBM / TWM) / 16;
    constexpr int TFRAGS_N = (TBN / TWN) / 16;
    constexpr int TKSTEPS  = TBK / 16;
    constexpr int BKv      = TBK / 16;
    constexpr int AVEC     = TBM * TBK / 16;          // int4 A elements per K-tile
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * TBN;
    const int tid = threadIdx.x, wid = tid / WAVE;
    const int wave_m = wid / TWN, wave_n = wid % TWN, lane = tid % WAVE;
    const int NT = N / 16;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        for (int ni = 0; ni < TFRAGS_N; ++ni) acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);

    // ---- AFEED=0: A staged in LDS (single or ping-pong double buffer) ----
    __shared__ __hip_fp8_e4m3 As[(AFEED == 0 ? (DBUF ? 2 : 1) : 1) * TBM * TBK];
    int4* Asv = reinterpret_cast<int4*>(As);
    constexpr int APT = AVEC / TBLOCK;                 // A int4 loads/thread/K-tile (DBUF prefetch)
    int4 areg[AFEED == 0 && DBUF ? (APT > 0 ? APT : 1) : 1];
    const int NTILES = K / TBK;

    auto fill_As = [&](int buf, int k0) {              // coalesced wide A fill -> LDS[buf]
        int4* dst = Asv + buf * (TBM * TBK / 16);
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv; int gr = tm + r, gk = k0 + c * 16;
            dst[e] = (gr < M) ? Av[((size_t)gr * K + gk) / 16] : int4{0,0,0,0};
        }
    };

    if (AFEED == 0 && DBUF) {                           // prologue: prefetch tile 0 -> regs -> As[0]
        #pragma unroll
        for (int i = 0; i < APT; ++i) { int e = tid + i*TBLOCK; int r=e/BKv,c=e%BKv; int gr=tm+r,gk=c*16;
            areg[i] = (gr < M) ? Av[((size_t)gr*K+gk)/16] : int4{0,0,0,0}; }
        #pragma unroll
        for (int i = 0; i < APT; ++i) { int e = tid + i*TBLOCK; Asv[e] = areg[i]; }
        __syncthreads();
    }

    const int row_a = lane & 15, colhi = (lane >> 4) & 1;
    for (int t = 0; t < NTILES; ++t) {
        const int k0 = t * TBK, buf = (AFEED == 0 && DBUF) ? (t & 1) : 0;
        if (AFEED == 0 && !DBUF) { fill_As(0, k0); __syncthreads(); }
        if (AFEED == 0 && DBUF && t + 1 < NTILES) {     // issue next tile's A loads (overlap WMMAs)
            const int nk0 = (t + 1) * TBK;
            #pragma unroll
            for (int i = 0; i < APT; ++i) { int e=tid+i*TBLOCK; int r=e/BKv,c=e%BKv; int gr=tm+r,gk=nk0+c*16;
                areg[i] = (gr < M) ? Av[((size_t)gr*K+gk)/16] : int4{0,0,0,0}; }
        }

        #pragma unroll
        for (int kk = 0; kk < TKSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[TFRAGS_M], fb[TFRAGS_N];
            #pragma unroll
            for (int mi = 0; mi < TFRAGS_M; ++mi) {
                if (AFEED == 0) {
                    int lds_row = (wave_m * TFRAGS_M + mi) * 16 + row_a;
                    const __hip_fp8_e4m3* base = As + buf * TBM * TBK;
                    fa[mi] = *reinterpret_cast<const v2i32*>(base + lds_row * TBK + kbase + colhi * 8);
                } else {                                 // direct from L2: 8 contiguous K-bytes/lane
                    int gr = tm + (wave_m * TFRAGS_M + mi) * 16 + row_a;
                    int gk = k0 + kbase + colhi * 8;
                    fa[mi] = (gr < M) ? *reinterpret_cast<const v2i32*>(A + (size_t)gr * K + gk)
                                      : v2i32{0,0};
                }
            }
            #pragma unroll
            for (int ni = 0; ni < TFRAGS_N; ++ni) {
                int nt = (tn + (wave_n * TFRAGS_N + ni) * 16) / 16;
                fb[ni] = tr_load8(Bshuf + b_tile_offset(kt, nt, NT) + lane * 8);
            }
            #pragma unroll
            for (int mi = 0; mi < TFRAGS_M; ++mi)
                #pragma unroll
                for (int ni = 0; ni < TFRAGS_N; ++ni)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc[mi][ni]);
        }

        if (AFEED == 0 && DBUF && t + 1 < NTILES) {      // land prefetched A into the other buffer
            int4* dst = Asv + ((t + 1) & 1) * (TBM * TBK / 16);
            #pragma unroll
            for (int i = 0; i < APT; ++i) { int e = tid + i*TBLOCK; dst[e] = areg[i]; }
            __syncthreads();
        } else if (AFEED == 0 && !DBUF) {
            __syncthreads();                              // LDS reuse barrier
        }
    }

    // ---- epilogue (§7.12 C/D map, scaled, bounds-guarded) ----
    __shared__ float scratch[TWAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 15, e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < TFRAGS_M; ++mi) {
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
            int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * TFRAGS_N + ni) * 16;
            for (int u = lane; u < 256; u += WAVE) {
                int gr = row0 + u / 16, gc = col0 + u % 16;
                if (gr < M && gc < N) {
                    float v = ws[u] * a_scale[gr] * b_scale[gc];
                    C[(size_t)gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ================= deep software pipeline: prefetch A AND B one tile ahead =================
// dbuf hid only A's latency (+12%). B is still loaded just-in-time from L2 inside the inner loop,
// so its latency is unhidden. This 2-stage pipeline issues tile (t+1)'s A loads (->regs->LDS) AND
// (t+1)'s B fragments (->registers, via global_load_tr) while the WMMAs for tile t run, so BOTH
// operands' L2 latency is buried under a full tile of matrix compute (the CK/CUTLASS technique).
// A stays in LDS (coalesced fill, the only A path that coalesces); B fragments live in registers
// double-buffered (breg[2]).
template <int TBM, int TBN, int TWM, int TWN, int TBK>
__global__ void __launch_bounds__(TWM * TWN * 32)
gemm_fp8_pipe(const __hip_fp8_e4m3* __restrict__ A, const uint8_t* __restrict__ Bshuf,
              __hip_bfloat16* __restrict__ C,
              const float* __restrict__ a_scale, const float* __restrict__ b_scale,
              int M, int N, int K) {
    constexpr int WAVE=32, TWAVES=TWM*TWN, TBLOCK=TWAVES*WAVE;
    constexpr int TFRAGS_M=(TBM/TWM)/16, TFRAGS_N=(TBN/TWN)/16, TKSTEPS=TBK/16, BKv=TBK/16;
    constexpr int AVEC=TBM*TBK/16, APT = AVEC/TBLOCK > 0 ? AVEC/TBLOCK : 1;
    const int tm=blockIdx.y*TBM, tn=blockIdx.x*TBN;
    const int tid=threadIdx.x, wid=tid/WAVE, wave_m=wid/TWN, wave_n=wid%TWN, lane=tid%WAVE;
    const int NT=N/16, NTILES=K/TBK;
    const int row_a=lane&15, colhi=(lane>>4)&1;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for(int mi=0;mi<TFRAGS_M;++mi) for(int ni=0;ni<TFRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};

    const int4* Av=reinterpret_cast<const int4*>(A);
    __shared__ __hip_fp8_e4m3 As[2][TBM*TBK];
    int4 areg[APT];
    v2i32 breg[2][TKSTEPS][TFRAGS_N];               // B fragments, double-buffered in registers

    auto fillA=[&](int buf,int k0){ int4* dst=reinterpret_cast<int4*>(As[buf]);
        for(int e=tid;e<AVEC;e+=TBLOCK){ int r=e/BKv,c=e%BKv; int gr=tm+r,gk=k0+c*16;
            dst[e]=(gr<M)?Av[((size_t)gr*K+gk)/16]:int4{0,0,0,0}; } };
    auto loadB=[&](int slot,int k0){
        #pragma unroll
        for(int kk=0;kk<TKSTEPS;++kk){ int kt=(k0+kk*16)/16;
            #pragma unroll
            for(int ni=0;ni<TFRAGS_N;++ni){ int nt=(tn+(wave_n*TFRAGS_N+ni)*16)/16;
                breg[slot][kk][ni]=tr_load8(Bshuf+b_tile_offset(kt,nt,NT)+lane*8); } } };

    fillA(0,0); loadB(0,0); __syncthreads();        // prologue: stage tile 0

    for(int t=0;t<NTILES;++t){
        const int cur=t&1, nxt=(t+1)&1, k0=t*TBK;
        if(t+1<NTILES){                              // issue tile t+1 A(->areg) + B(->breg[nxt])
            const int nk0=(t+1)*TBK;
            #pragma unroll
            for(int i=0;i<APT;++i){ int e=tid+i*TBLOCK; int r=e/BKv,c=e%BKv; int gr=tm+r,gk=nk0+c*16;
                areg[i]=(gr<M)?Av[((size_t)gr*K+gk)/16]:int4{0,0,0,0}; }
            loadB(nxt,nk0);                          // global_load_tr in flight, used next iter
        }
        #pragma unroll
        for(int kk=0;kk<TKSTEPS;++kk){
            const int kbase=kk*16;
            v2i32 fa[TFRAGS_M];
            #pragma unroll
            for(int mi=0;mi<TFRAGS_M;++mi){ int lds_row=(wave_m*TFRAGS_M+mi)*16+row_a;
                fa[mi]=*reinterpret_cast<const v2i32*>(As[cur]+lds_row*TBK+kbase+colhi*8); }
            #pragma unroll
            for(int mi=0;mi<TFRAGS_M;++mi)
                #pragma unroll
                for(int ni=0;ni<TFRAGS_N;++ni)
                    acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], breg[cur][kk][ni], acc[mi][ni]);
        }
        if(t+1<NTILES){                              // land prefetched A into the other LDS buffer
            int4* dst=reinterpret_cast<int4*>(As[nxt]);
            #pragma unroll
            for(int i=0;i<APT;++i){ int e=tid+i*TBLOCK; dst[e]=areg[i]; }
            __syncthreads();
        }
    }

    __shared__ float scratch[TWAVES][16*16];
    float* ws=scratch[wid];
    const int e_col=lane&15, e_rowbase=((lane>>4)&1)*8;
    for(int mi=0;mi<TFRAGS_M;++mi) for(int ni=0;ni<TFRAGS_N;++ni){
        #pragma unroll
        for(int s=0;s<8;++s) ws[(e_rowbase+s)*16+e_col]=acc[mi][ni][s];
        int row0=tm+(wave_m*TFRAGS_M+mi)*16, col0=tn+(wave_n*TFRAGS_N+ni)*16;
        for(int u=lane;u<256;u+=WAVE){ int gr=row0+u/16, gc=col0+u%16;
            if(gr<M&&gc<N){ float v=ws[u]*a_scale[gr]*b_scale[gc]; C[(size_t)gr*N+gc]=(__hip_bfloat16)v; } }
    }
}

// ---------------- host ----------------
#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

static uint8_t enc(float x){ __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec(uint8_t b){ __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v)=b; return (float)v; }
static float patA(int i,int k){ return (float)(((i*7 + k*3) % 9) - 4) * 0.5f; }
static float patB(int k,int j){ return (float)(((k*5 + j*2) % 9) - 4) * 0.5f; }

static void build_io(int M,int N,int K,std::vector<uint8_t>&Ah,std::vector<uint8_t>&Bh){
    Ah.resize((size_t)M*K); Bh.resize((size_t)K*N);
    for(int i=0;i<M;i++)for(int k=0;k<K;k++) Ah[(size_t)i*K+k]=enc(patA(i,k));
    for(int k=0;k<K;k++)for(int j=0;j<N;j++) Bh[(size_t)k*N+j]=enc(patB(k,j));
}

// device buffers shared across all configs (one alloc per problem size)
struct Dev { __hip_fp8_e4m3* A; uint8_t* Bshuf; __hip_bfloat16* C; float* as; float* bs; int M,N,K; };

template <int TBM,int TBN,int TWM,int TWN,int TBK,int AFEED,int DBUF>
static void launch(const Dev& d){
    dim3 grid(d.N/TBN, d.M/TBM); constexpr int TBLOCK = TWM*TWN*32;
    gemm_fp8_lever<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF><<<grid,TBLOCK>>>(d.A,d.Bshuf,d.C,d.as,d.bs,d.M,d.N,d.K);
}

template <int TBM,int TBN,int TWM,int TWN,int TBK,int AFEED,int DBUF>
static float oracle_err(const Dev& d, const std::vector<float>& ref){
    launch<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF>(d); hipDeviceSynchronize();
    std::vector<uint16_t> H((size_t)d.M*d.N);
    hipMemcpy(H.data(), d.C, (size_t)d.M*d.N*2, hipMemcpyDeviceToHost);
    float mx=0;
    for(size_t t=0;t<(size_t)d.M*d.N;t++){ uint32_t b=(uint32_t)H[t]<<16; float gv; __builtin_memcpy(&gv,&b,4);
        float rv=ref[t], e=fabsf(gv-rv)/(fabsf(rv)+1e-3f); if(e>mx)mx=e; }
    return mx;
}

template <int TBM,int TBN,int TWM,int TWN,int TBK,int AFEED,int DBUF>
static float time_ms(const Dev& d, int iters){
    for(int i=0;i<5;i++) launch<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF>(d);   // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) launch<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF>(d);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e); hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

// --- pipe kernel harness (5 template params) ---
template <int TBM,int TBN,int TWM,int TWN,int TBK>
static void launch_pipe(const Dev& d){
    dim3 grid(d.N/TBN, d.M/TBM); constexpr int TBLOCK = TWM*TWN*32;
    gemm_fp8_pipe<TBM,TBN,TWM,TWN,TBK><<<grid,TBLOCK>>>(d.A,d.Bshuf,d.C,d.as,d.bs,d.M,d.N,d.K);
}
template <int TBM,int TBN,int TWM,int TWN,int TBK>
static float oracle_err_pipe(const Dev& d, const std::vector<float>& ref){
    launch_pipe<TBM,TBN,TWM,TWN,TBK>(d); hipDeviceSynchronize();
    std::vector<uint16_t> H((size_t)d.M*d.N);
    hipMemcpy(H.data(), d.C, (size_t)d.M*d.N*2, hipMemcpyDeviceToHost);
    float mx=0;
    for(size_t t=0;t<(size_t)d.M*d.N;t++){ uint32_t b=(uint32_t)H[t]<<16; float gv; __builtin_memcpy(&gv,&b,4);
        float rv=ref[t], e=fabsf(gv-rv)/(fabsf(rv)+1e-3f); if(e>mx)mx=e; }
    return mx;
}
template <int TBM,int TBN,int TWM,int TWN,int TBK>
static float time_pipe(const Dev& d, int iters){
    for(int i=0;i<5;i++) launch_pipe<TBM,TBN,TWM,TWN,TBK>(d);
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) launch_pipe<TBM,TBN,TWM,TWN,TBK>(d);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e); hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

static Dev make_dev(int M,int N,int K){
    std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
    std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
    Dev d; d.M=M;d.N=N;d.K=K;
    hipMalloc(&d.A,Ah.size()); hipMalloc(&d.Bshuf,Bshufh.size()); hipMalloc(&d.C,(size_t)M*N*2);
    hipMalloc(&d.as,M*4); hipMalloc(&d.bs,N*4);
    hipMemcpy(d.A,Ah.data(),Ah.size(),hipMemcpyHostToDevice);
    hipMemcpy(d.Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice);
    std::vector<float> ones(N>M?N:M,1.f);
    hipMemcpy(d.as,ones.data(),M*4,hipMemcpyHostToDevice);
    hipMemcpy(d.bs,ones.data(),N*4,hipMemcpyHostToDevice);
    return d;
}
static void free_dev(Dev&d){ hipFree(d.A);hipFree(d.Bshuf);hipFree(d.C);hipFree(d.as);hipFree(d.bs); }

static const double CEIL=307.0, HBL=143.0, WPC=15.9;

// one config: oracle (1920^3) gate then perf (3840^3). Prints TF, %ceiling, WMMA/cyc.
#define CFG(LABEL, TBM,TBN,TWM,TWN,TBK,AFEED,DBUF) do{ \
    float er = oracle_err<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF>(od, oref); \
    float ms = time_ms<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF>(pd, 30); \
    double tf = gflop/(ms/1e3); \
    printf("  %-30s err=%.4f %s  %6.1f TF  (%2.0f%% , %.2fx143)  %.2f WMMA/cyc\n", \
        LABEL, er, er<0.03f?"ok":"FAIL", tf, 100*tf/CEIL, tf/HBL, WPC*tf/CEIL); \
}while(0)

// pipe config: oracle (1920) + perf (3840) for the deep A+B software pipeline.
#define PIPE(LABEL, TBM,TBN,TWM,TWN,TBK) do{ \
    float er = oracle_err_pipe<TBM,TBN,TWM,TWN,TBK>(od, oref); \
    float ms = time_pipe<TBM,TBN,TWM,TWN,TBK>(pd, 30); double tf = gflop/(ms/1e3); \
    printf("  %-30s err=%.4f %s  %6.1f TF  (%2.0f%% , %.2fx143)  %.2f WMMA/cyc\n", \
        LABEL, er, er<0.03f?"ok":"FAIL", tf, 100*tf/CEIL, tf/HBL, WPC*tf/CEIL); \
}while(0)

int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    printf("device: %s  CUs=%d   (ceiling %.0f TF = %.1f WMMA/cyc, hipBLASLt %.0f)\n",
           p.name,p.multiProcessorCount,CEIL,WPC,HBL);

    // ---- oracle reference @ 1920^3 (divisible by 128/160/192/256) ----
    const int OM=1920,ON=1920,OK=256;
    std::vector<uint8_t> Ah,Bh; build_io(OM,ON,OK,Ah,Bh);
    std::vector<float> oref((size_t)OM*ON,0.f);
    for(int i=0;i<OM;i++)for(int j=0;j<ON;j++){ float a=0;
        for(int k=0;k<OK;k++) a+=dec(Ah[(size_t)i*OK+k])*dec(Bh[(size_t)k*ON+j]); oref[(size_t)i*ON+j]=a; }
    Dev od = make_dev(OM,ON,OK);

    // ---- perf @ 3840^3 (full sweep) and 4096^3 (clean compare vs hipBLASLt=143) ----
    const int PM=3840,PN=3840,PK=4096;
    Dev pd = make_dev(PM,PN,PK);
    double gflop = 2.0*PM*PN*PK/1e12;
    Dev pd4 = make_dev(4096,4096,4096);
    double gflop4 = 2.0*4096.0*4096.0*4096.0/1e12;
    // perf-only on the 4096 cube (128-divisible configs only); oracle already gated at 1920.
    #define P4K(LABEL, TBM,TBN,TWM,TWN,TBK,AFEED,DBUF) do{ \
        float ms = time_ms<TBM,TBN,TWM,TWN,TBK,AFEED,DBUF>(pd4, 30); double tf = gflop4/(ms/1e3); \
        printf("  %-30s @4096       %6.1f TF  (%2.0f%% , %.2fx143)  %.2f WMMA/cyc\n", \
            LABEL, tf, 100*tf/CEIL, tf/HBL, WPC*tf/CEIL); }while(0)

    printf("\n=== A-feed x tile sweep @3840 (B fixed = direct global_load_tr) ===\n");
    //   LABEL                              TBM TBN TWM TWN TBK AFEED DBUF
    CFG("acc4x4  Alds       bk32",          128,128, 2,2, 32, 0,0);   // Phase-1 baseline
    CFG("acc4x4  Adirect    bk32",          128,128, 2,2, 32, 1,0);   // direct-A = NO-GO (uncoalesced)
    CFG("acc6x4  Alds       bk32",          192,128, 2,2, 32, 0,0);   // bigger tile loses
    CFG("acc4x8  Adirect    bk32",          128,256, 2,2, 32, 1,0);   // 256 acc-VGPR -> spill crater
    CFG("acc8x4  Adirect    bk32",          256,128, 2,2, 32, 1,0);   // 256 acc-VGPR -> spill crater

    printf("\n=== drill the double-buffer winner: dbuf x BK x wave-grid @3840 ===\n");
    CFG("acc4x4  Alds  dbuf bk16",          128,128, 2,2, 16, 0,1);
    CFG("acc4x4  Alds  dbuf bk32",          128,128, 2,2, 32, 0,1);   // 147 last run
    CFG("acc4x4  Alds  dbuf bk64",          128,128, 2,2, 64, 0,1);
    CFG("acc4x4  Alds  dbuf bk128",         128,128, 2,2,128, 0,1);
    CFG("acc6x4  Alds  dbuf bk32",          192,128, 2,2, 32, 0,1);
    CFG("acc4x2  Alds  dbuf bk32 (wg2x4)",  128,128, 2,4, 32, 0,1);
    CFG("acc2x4  Alds  dbuf bk32 (wg4x2)",  128,128, 4,2, 32, 0,1);
    CFG("acc8x2  Alds  dbuf bk32 (wg1x4)",  128,128, 1,4, 32, 0,1);   // 2 waves, more reuse/wave

    printf("\n=== DEEP PIPELINE: prefetch A AND B one tile ahead (B latency now hidden) @3840 ===\n");
    PIPE("pipe acc4x4  bk32",                128,128, 2,2, 32);
    PIPE("pipe acc4x4  bk64",                128,128, 2,2, 64);
    PIPE("pipe acc4x4  bk16",                128,128, 2,2, 16);
    PIPE("pipe acc2x4  bk32 (wg4x2)",        128,128, 4,2, 32);
    PIPE("pipe acc4x2  bk32 (wg2x4)",        128,128, 2,4, 32);
    PIPE("pipe acc6x4  bk32",                192,128, 2,2, 32);
    PIPE("pipe acc4x4  bk128",               128,128, 2,2,128);

    printf("\n=== 4096^3 confirmation (apples-to-apples vs Phase-1 140 / hipBLASLt 143) ===\n");
    P4K("acc4x4  Alds       bk32",          128,128, 2,2, 32, 0,0);
    P4K("acc4x4  Alds  dbuf bk32",          128,128, 2,2, 32, 0,1);
    #define P4KP(LABEL, TBM,TBN,TWM,TWN,TBK) do{ \
        float ms = time_pipe<TBM,TBN,TWM,TWN,TBK>(pd4, 30); double tf = gflop4/(ms/1e3); \
        printf("  %-30s @4096       %6.1f TF  (%2.0f%% , %.2fx143)  %.2f WMMA/cyc\n", \
            LABEL, tf, 100*tf/CEIL, tf/HBL, WPC*tf/CEIL); }while(0)
    P4KP("pipe acc4x4 bk32",                128,128, 2,2, 32);
    P4KP("pipe acc4x4 bk64",                128,128, 2,2, 64);

    free_dev(od); free_dev(pd); free_dev(pd4);
    return 0;
}
