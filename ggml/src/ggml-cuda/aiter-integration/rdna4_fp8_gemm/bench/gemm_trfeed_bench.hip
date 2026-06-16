// gemm_trfeed_bench.hip — MAD-305 Phase 1 (unsupervised normal-HIP launch).
//
// Thesis: the real fp8 GEMM walls at ~69-90 TF (22-29% of the proven 307 TF WMMA ceiling)
// because the B operand fragment is byte-gathered out of LDS (8 strided ds_load_u8 ahead of
// every WMMA) while A reads wide. Phase 0 verified (bit-exact) that gfx1201's hardware
// transpose-load `global_load_tr_b64` delivers the §7.12 B fragment in ONE instruction when B
// is pre-shuffled (B is static weights -> repack once, free at runtime; see
// bench/global_load_tr_contract.md). This bench runs the SAME 128x128/BK=32/4-wave tiling two
// ways: the byte-gather baseline (B staged in LDS) vs `gemm_fp8_trfeed` (A still LDS wide-read,
// B fed directly from a pre-shuffled global buffer via global_load_tr_b64, NO LDS for B).
// Correctness is gated against a CPU e4m3 oracle at 256^3 (A != B^T so a layout bug can't hide);
// throughput + baseline/trfeed agreement is measured at a compute-bound 4096^3.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <rocwmma/rocwmma.hpp>
#include "trfeed_common.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
using namespace rocwmma;

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

// Same tile geometry as the verified kernel (gemm_wmma.hip): 128x128 tile, BK=32, 4 waves.
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WAVES_M = 2, WAVES_N = 2, WAVES = WAVES_M * WAVES_N;
constexpr int WAVE_SIZE = 32, BLOCK_THREADS = WAVES * WAVE_SIZE;
constexpr int FRAGS_M = (BM / WAVES_M) / 16;   // 2
constexpr int FRAGS_N = (BN / WAVES_N) / 16;   // 2
constexpr int KSTEPS  = BK / 16;               // 2

// gfx12 global transpose-load, 8-bit element / b64 form (one wave32 lane gets 8 bytes).
__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}

// ---------------- baseline: byte-gather B feed (B staged in LDS) ----------------
__global__ void __launch_bounds__(BLOCK_THREADS)
gemm_fp8_baseline(const float8_t* __restrict__ A, const float8_t* __restrict__ B,
                  __hip_bfloat16* __restrict__ C,
                  const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                  int M, int N, int K) {
    const int tm = blockIdx.y * BM;
    const int tn = blockIdx.x * BN;

    __shared__ float8_t As[BM * BK];   // [BM][BK], ld=BK (K-inner: A reads wide)
    __shared__ float8_t Bs[BK * BN];   // [BK][BN], ld=BN (N-inner: fragment read must gather)

    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;

    v8f32 acc[FRAGS_M][FRAGS_N];
    for (int mi = 0; mi < FRAGS_M; ++mi)
        for (int ni = 0; ni < FRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    const int4* Bv = reinterpret_cast<const int4*>(B);
    int4*  Asv = reinterpret_cast<int4*>(As);
    int4*  Bsv = reinterpret_cast<int4*>(Bs);
    uint8_t* Bsb = reinterpret_cast<uint8_t*>(Bs);

    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int AVEC = BM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += BLOCK_THREADS) {
            int r = e / BKv, c = e % BKv;
            int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        constexpr int BVEC = BK * BN / 16, BNv = BN / 16;
        for (int e = tid; e < BVEC; e += BLOCK_THREADS) {
            int r = e / BNv, c = e % BNv;
            int gk = k0 + r, gc = tn + c * 16;
            Bsv[e] = Bv[(gk * N + gc) / 16];
        }
        __syncthreads();

        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;   // A-map (§7.12)
        const int col_b = lane & 0xF, rowhi = (lane >> 4) & 1;   // B-map (§7.12)
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16;
            v2i32 fa[FRAGS_M], fb[FRAGS_N];
            for (int mi = 0; mi < FRAGS_M; ++mi) {
                int lds_row = (wave_m * FRAGS_M + mi) * 16 + row_a;
                fa[mi] = *reinterpret_cast<const v2i32*>(As + lds_row * BK + kbase + colhi * 8);
            }
            for (int ni = 0; ni < FRAGS_N; ++ni) {
                int lds_col = (wave_n * FRAGS_N + ni) * 16 + col_b;
                const uint8_t* bp = Bsb + (kbase + rowhi * 8) * BN + lds_col;
                uint32_t lo = 0, hi = 0;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    lo |= (uint32_t)bp[p * BN]       << (p * 8);
                    hi |= (uint32_t)bp[(p + 4) * BN] << (p * 8);
                }
                fb[ni] = v2i32{(int)lo, (int)hi};
            }
            for (int mi = 0; mi < FRAGS_M; ++mi)
                for (int ni = 0; ni < FRAGS_N; ++ni)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    // ---- epilogue (manual scratch, §7.12 C/D map) ----
    __shared__ float scratch[WAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < FRAGS_M; ++mi) {
        for (int ni = 0; ni < FRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
            int row0 = tm + (wave_m * FRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * FRAGS_N + ni) * 16;
            for (int t = lane; t < 256; t += WAVE_SIZE) {
                int gr = row0 + t / 16, gc = col0 + t % 16;
                if (gr < M && gc < N) {
                    float v = ws[t] * a_scale[gr] * b_scale[gc];
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed: B fed via global_load_tr_b64 from pre-shuffled global ------------
// Templated on the M-tiling so bm128 (TBM=128,TWAVES_M=2) and the larger-M amortization
// variant bm256 (TBM=256,TWAVES_M=4) share one body. N-tiling fixed: BN=128, WAVES_N=2.
// Larger TBM serves each B column tile across more M-rows -> fewer distinct B-tile DRAM
// fetches per output tile (amortizes B re-fetch).
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                __hip_bfloat16* __restrict__ C,
                const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                int M, int N, int K) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * BK];                // A still staged in LDS (wide read)
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int AVEC = TBM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv;
            int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        __syncthreads();

        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[TFRAGS_M], fb[TFRAGS_N];
            for (int mi = 0; mi < TFRAGS_M; ++mi) {
                int lds_row = (wave_m * TFRAGS_M + mi) * 16 + row_a;
                fa[mi] = *reinterpret_cast<const v2i32*>(As + lds_row * BK + kbase + colhi * 8);
            }
            for (int ni = 0; ni < TFRAGS_N; ++ni) {
                int nt = (tn + (wave_n * TFRAGS_N + ni) * 16) / 16;
                fb[ni] = tr_load8(Bshuf + b_tile_offset(kt, nt, NT) + lane * 8);   // one global_load_tr
            }
            for (int mi = 0; mi < TFRAGS_M; ++mi)
                for (int ni = 0; ni < TFRAGS_N; ++ni)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc[mi][ni]);
        }
        __syncthreads();                              // A LDS reuse barrier
    }

    // ---- epilogue (identical to baseline) ----
    __shared__ float scratch[TWAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < TFRAGS_M; ++mi) {
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
            int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * TFRAGS_N + ni) * 16;
            for (int t = lane; t < 256; t += WAVE_SIZE) {
                int gr = row0 + t / 16, gc = col0 + t % 16;
                if (gr < M && gc < N) {
                    float v = ws[t] * a_scale[gr] * b_scale[gc];
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- host ----------------
#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

static uint8_t enc(float x){ __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec(uint8_t b){ __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v)=b; return (float)v; }

// Distinct A/B patterns (A != B^T) so a transpose bug cannot hide. Values in [-2,2] step 0.5,
// all exact in e4m3.
static float patA(int i,int k){ return (float)(((i*7 + k*3) % 9) - 4) * 0.5f; }
static float patB(int k,int j){ return (float)(((k*5 + j*2) % 9) - 4) * 0.5f; }

static int build_io(int M,int N,int K,
                    std::vector<uint8_t>&Ah,std::vector<uint8_t>&Bh){
    Ah.resize((size_t)M*K); Bh.resize((size_t)K*N);
    for(int i=0;i<M;i++)for(int k=0;k<K;k++) Ah[(size_t)i*K+k]=enc(patA(i,k));
    for(int k=0;k<K;k++)for(int j=0;j<N;j++) Bh[(size_t)k*N+j]=enc(patB(k,j));
    return 0;
}

static float time_baseline(const float8_t*A,const float8_t*B,__hip_bfloat16*C,
                           const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
    for(int i=0;i<5;i++) gemm_fp8_baseline<<<grid,BLOCK_THREADS>>>(A,B,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_baseline<<<grid,BLOCK_THREADS>>>(A,B,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                         const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    printf("device: %s  CUs=%d\n", p.name, p.multiProcessorCount);
    const double CEIL = 307.0, HIPBLASLT = 143.0;

    // ---------- correctness @ 256^3 (the Phase-1 oracle gate) ----------
    {
        const int M=256,N=256,K=256;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        std::vector<float> oref((size_t)M*N,0.f);
        for(int i=0;i<M;i++)for(int j=0;j<N;j++){
            float acc=0; for(int k=0;k<K;k++) acc+=dec(Ah[(size_t)i*K+k])*dec(Bh[(size_t)k*N+j]);
            oref[(size_t)i*N+j]=acc;   // a_scale=b_scale=1
        }
        float8_t *A,*B; uint8_t* Bshuf; __hip_bfloat16*C; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&B,Bh.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&C,(size_t)M*N*2));
        HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(B,Bh.data(),Bh.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N>M?N:M,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
        auto check=[&](const char*name)->float{
            std::vector<uint16_t> Chost((size_t)M*N);
            HCK(hipMemcpy(Chost.data(),C,(size_t)M*N*2,hipMemcpyDeviceToHost));
            float mx=0;
            for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t bits=(uint32_t)Chost[t]<<16; float gv; __builtin_memcpy(&gv,&bits,4);
                float rv=oref[t]; float e=fabsf(gv-rv)/(fabsf(rv)+1e-3f);
                if(e>mx)mx=e;
            }
            printf("  %-9s max_rel_err=%.4f  %s\n",name,mx,mx<0.03f?"PASS":"*** FAIL ***");
            return mx;
        };
        gemm_fp8_baseline<<<grid,BLOCK_THREADS>>>(A,B,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float eb=check("baseline");
        gemm_fp8_trfeed<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float et=check("trfeed128");
        dim3 grid256((N+BN-1)/BN,(M+256-1)/256);
        gemm_fp8_trfeed<256,4><<<grid256,256>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float et2=check("trfeed256");
        if(eb>=0.03f||et>=0.03f||et2>=0.03f){ printf("CORRECTNESS GATE FAILED — perf numbers below are meaningless.\n"); }
        hipFree(A);hipFree(B);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
    }

    // ---------- perf @ 4096^3 ----------
    {
        const int M=4096,N=4096,K=4096;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        float8_t *A,*B; uint8_t* Bshuf; __hip_bfloat16*Cb,*Ct,*Ct2; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&B,Bh.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&Cb,(size_t)M*N*2)); HCK(hipMalloc(&Ct,(size_t)M*N*2)); HCK(hipMalloc(&Ct2,(size_t)M*N*2));
        HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(B,Bh.data(),Bh.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        double gflop = 2.0*M*N*K/1e12;
        float msb  = time_baseline(A,B,Cb,as,bs,M,N,K,30);
        float mst  = time_trfeed<128,2>(A,Bshuf,Ct ,as,bs,M,N,K,30);
        float mst2 = time_trfeed<256,4>(A,Bshuf,Ct2,as,bs,M,N,K,30);
        double tfb = gflop/(msb/1e3), tft = gflop/(mst/1e3), tft2 = gflop/(mst2/1e3);
        printf("\n=== perf @ %d^3 (ceiling = %.0f TF, hipBLASLt = %.0f TF) ===\n",M,CEIL,HIPBLASLT);
        printf("  baseline (byte-gather)    : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msb,tfb,100*tfb/CEIL,tfb/HIPBLASLT);
        printf("  trfeed bm128 (global_tr)  : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               mst,tft,100*tft/CEIL,tft/HIPBLASLT);
        printf("  trfeed bm256 (amortize B) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               mst2,tft2,100*tft2/CEIL,tft2/HIPBLASLT);
        printf("  speedup bm128/baseline    : %.2fx   bm256/baseline : %.2fx\n",tft/tfb,tft2/tfb);
        // real-shape agreement: both trfeed variants vs baseline must match (same math).
        std::vector<uint16_t> Hb((size_t)M*N),Ht((size_t)M*N),Ht2((size_t)M*N);
        HCK(hipMemcpy(Hb.data(),Cb,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Ht.data(),Ct,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Ht2.data(),Ct2,(size_t)M*N*2,hipMemcpyDeviceToHost));
        auto maxdiff=[&](std::vector<uint16_t>&H)->double{ double md=0;
            for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t a=(uint32_t)Hb[t]<<16,b=(uint32_t)H[t]<<16; float fa,fb;
                __builtin_memcpy(&fa,&a,4);__builtin_memcpy(&fb,&b,4);
                double d=fabs((double)fa-fb); if(d>md)md=d;
            } return md; };
        double md=maxdiff(Ht), md2=maxdiff(Ht2);
        printf("  bm128 vs baseline max|Δ|  = %.4g  %s\n",md ,md <1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  bm256 vs baseline max|Δ|  = %.4g  %s\n",md2,md2<1e-2?"(match)":"(*** MISMATCH ***)");
        hipFree(A);hipFree(B);hipFree(Bshuf);hipFree(Cb);hipFree(Ct);hipFree(Ct2);hipFree(as);hipFree(bs);
    }
    return 0;
}
