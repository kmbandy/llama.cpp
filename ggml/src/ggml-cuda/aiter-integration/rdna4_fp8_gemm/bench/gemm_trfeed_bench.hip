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
#include "trfeed_kernels.h"   // frozen kernel bodies (byte-identical to production)
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
using namespace rocwmma;

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

template <int TBM, int TWAVES_M>
static float time_trfeed_db(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_db<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_db<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed_bslds(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                               const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_bslds<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_bslds<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M, int TBK>
static float time_trfeed_bk(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_bk<TBM,TWAVES_M,TBK><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_bk<TBM,TWAVES_M,TBK><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TWM, int TWN>
static float time_trfeed_wg(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+128-1)/128,(M+128-1)/128);
    constexpr int TBLOCK = TWM*TWN*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_wg<TWM,TWN><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_wg<TWM,TWN><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed_rb(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_rb<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_rb<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
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
        gemm_fp8_trfeed_db<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ed=check("trfeed_db");
        gemm_fp8_trfeed_bslds<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float el=check("trfeed_bsl");
        gemm_fp8_trfeed_bk<128,2,64><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ek64=check("trfeed_bk64");
        gemm_fp8_trfeed_bk<128,2,128><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ek128=check("trfeed_bk128");
        gemm_fp8_trfeed_wg<4,4><<<grid,512>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ew44=check("trfeed_wg44");
        gemm_fp8_trfeed_wg<2,4><<<grid,256>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ew24=check("trfeed_wg24");
        gemm_fp8_trfeed_rb<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float erb=check("trfeed_rb");
        if(eb>=0.03f||et>=0.03f||et2>=0.03f||ed>=0.03f||el>=0.03f||ek64>=0.03f||ek128>=0.03f||ew44>=0.03f||ew24>=0.03f||erb>=0.03f){ printf("CORRECTNESS GATE FAILED — perf numbers below are meaningless.\n"); }
        hipFree(A);hipFree(B);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
    }

    // ---------- perf @ 4096^3 ----------
    {
        const int M=4096,N=4096,K=4096;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        float8_t *A,*B; uint8_t* Bshuf; __hip_bfloat16*Cb,*Ct,*Ct2,*Cd,*Cl; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&B,Bh.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&Cb,(size_t)M*N*2)); HCK(hipMalloc(&Ct,(size_t)M*N*2));
        HCK(hipMalloc(&Ct2,(size_t)M*N*2)); HCK(hipMalloc(&Cd,(size_t)M*N*2)); HCK(hipMalloc(&Cl,(size_t)M*N*2));
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
        float msd  = time_trfeed_db<128,2>(A,Bshuf,Cd,as,bs,M,N,K,30);
        float msl  = time_trfeed_bslds<128,2>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msk64 = time_trfeed_bk<128,2,64>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msk128= time_trfeed_bk<128,2,128>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msw44 = time_trfeed_wg<4,4>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msw24 = time_trfeed_wg<2,4>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msrb  = time_trfeed_rb<128,2>(A,Bshuf,Cl,as,bs,M,N,K,30);
        double tfb=gflop/(msb/1e3), tft=gflop/(mst/1e3), tft2=gflop/(mst2/1e3),
               tfd=gflop/(msd/1e3), tfl=gflop/(msl/1e3),
               tfk64=gflop/(msk64/1e3), tfk128=gflop/(msk128/1e3),
               tfw44=gflop/(msw44/1e3), tfw24=gflop/(msw24/1e3), tfrb=gflop/(msrb/1e3);
        printf("\n=== perf @ %d^3 (ceiling = %.0f TF, hipBLASLt = %.0f TF) ===\n",M,CEIL,HIPBLASLT);
        printf("  baseline (byte-gather)    : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msb,tfb,100*tfb/CEIL,tfb/HIPBLASLT);
        printf("  trfeed bm128 (global_tr)  : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               mst,tft,100*tft/CEIL,tft/HIPBLASLT);
        printf("  trfeed bm256 (amortize B) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               mst2,tft2,100*tft2/CEIL,tft2/HIPBLASLT);
        printf("  trfeed_db bm128 (dbl-buf) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msd,tfd,100*tfd/CEIL,tfd/HIPBLASLT);
        printf("  trfeed_bslds bm128 (B@LDS): %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msl,tfl,100*tfl/CEIL,tfl/HIPBLASLT);
        printf("  trfeed_bk64 bm128 (BK=64) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msk64,tfk64,100*tfk64/CEIL,tfk64/HIPBLASLT);
        printf("  trfeed_bk128 bm128(BK=128): %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msk128,tfk128,100*tfk128/CEIL,tfk128/HIPBLASLT);
        printf("  trfeed_wg 4x4 (16w,acc2x2): %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msw44,tfw44,100*tfw44/CEIL,tfw44/HIPBLASLT);
        printf("  trfeed_wg 2x4 (8w,acc4x2) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msw24,tfw24,100*tfw24/CEIL,tfw24/HIPBLASLT);
        printf("  trfeed_rb bm128 (reg-blk) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msrb,tfrb,100*tfrb/CEIL,tfrb/HIPBLASLT);
        printf("  speedup bk64 %.2fx bk128 %.2fx wg44 %.2fx wg24 %.2fx rb %.2fx (vs bm128)\n",
               tfk64/tft,tfk128/tft,tfw44/tft,tfw24/tft,tfrb/tft);
        // real-shape agreement: all trfeed variants vs baseline must match (same math).
        std::vector<uint16_t> Hb((size_t)M*N),Ht((size_t)M*N),Ht2((size_t)M*N),Hd((size_t)M*N),Hl((size_t)M*N);
        HCK(hipMemcpy(Hb.data(),Cb,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Ht.data(),Ct,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Ht2.data(),Ct2,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Hd.data(),Cd,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Hl.data(),Cl,(size_t)M*N*2,hipMemcpyDeviceToHost));
        auto maxdiff=[&](std::vector<uint16_t>&H)->double{ double md=0;
            for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t a=(uint32_t)Hb[t]<<16,b=(uint32_t)H[t]<<16; float fa,fb;
                __builtin_memcpy(&fa,&a,4);__builtin_memcpy(&fb,&b,4);
                double d=fabs((double)fa-fb); if(d>md)md=d;
            } return md; };
        double md=maxdiff(Ht), md2=maxdiff(Ht2), mdd=maxdiff(Hd), mdl=maxdiff(Hl);
        printf("  bm128 vs baseline max|Δ|  = %.4g  %s\n",md ,md <1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  bm256 vs baseline max|Δ|  = %.4g  %s\n",md2,md2<1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  db    vs baseline max|Δ|  = %.4g  %s\n",mdd,mdd<1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  bslds vs baseline max|Δ|  = %.4g  %s\n",mdl,mdl<1e-2?"(match)":"(*** MISMATCH ***)");
        hipFree(A);hipFree(B);hipFree(Bshuf);hipFree(Cb);hipFree(Ct);hipFree(Ct2);hipFree(Cd);hipFree(Cl);hipFree(as);hipFree(bs);
    }

    // ---------- Phase-3 FREE PREVIEW: acc-tile reuse sweep — does a BIGGER accumulator
    //   tile raise WMMA/cycle? acc[4][4] (TBM=128) is the 140-TF/7.35-WMMA-per-cyc winner.
    //   Raising FRAGS_M amortizes operand loads over more WMMAs (reuse 2.00->2.22->2.40
    //   WMMA/load) but costs VGPR -> lower static occupancy. All three fit the 256-VGPR
    //   static-wave cap (acc[8][8]=512 would not -> that one needs dyn-VGPR). Dims are
    //   divisible by lcm(128,160,192)=1920 so there is NO edge-tile waste: useful-TF ==
    //   WMMA issue rate, and WMMA/cyc = (TF/307)*15.9 exactly (clock pinned, same ceiling).
    //   NOTE: the trfeed A-fill is unguarded, so TBM MUST divide M/N (hence 1920/3840). ----
    {
        // -- oracle @ 1920x1920x256 (1920 divisible by 128/160/192 -> A-fill in-bounds) --
        const int M=1920,N=1920,K=256;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        std::vector<float> oref((size_t)M*N,0.f);
        for(int i=0;i<M;i++)for(int j=0;j<N;j++){
            float acc=0; for(int k=0;k<K;k++) acc+=dec(Ah[(size_t)i*K+k])*dec(Bh[(size_t)k*N+j]);
            oref[(size_t)i*N+j]=acc;   // a_scale=b_scale=1
        }
        float8_t *A; uint8_t* Bshuf; __hip_bfloat16*C; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&C,(size_t)M*N*2)); HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        auto chk=[&](const char*name)->float{
            std::vector<uint16_t> H((size_t)M*N);
            HCK(hipMemcpy(H.data(),C,(size_t)M*N*2,hipMemcpyDeviceToHost));
            float mx=0; for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t b=(uint32_t)H[t]<<16; float gv; __builtin_memcpy(&gv,&b,4);
                float rv=oref[t], e=fabsf(gv-rv)/(fabsf(rv)+1e-3f); if(e>mx)mx=e; }
            printf("  oracle %-9s max_rel_err=%.4f  %s\n",name,mx,mx<0.03f?"PASS":"*** FAIL ***");
            return mx; };
        printf("\n=== acc-tile reuse sweep — oracle @ %d^3 ===\n",M);
        { dim3 g((N+BN-1)/BN,(M+128-1)/128); gemm_fp8_trfeed<128,2><<<g,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize()); }
        float o4=chk("acc[4][4]");
        { dim3 g((N+BN-1)/BN,(M+160-1)/160); gemm_fp8_trfeed<160,2><<<g,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize()); }
        float o5=chk("acc[5][4]");
        { dim3 g((N+BN-1)/BN,(M+192-1)/192); gemm_fp8_trfeed<192,2><<<g,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize()); }
        float o6=chk("acc[6][4]");
        hipFree(A);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
        if(o4>=0.03f||o5>=0.03f||o6>=0.03f) printf("  acc-tile ORACLE FAILED — perf below is meaningless.\n");
    }
    {
        // -- perf @ 3840x3840x4096 (divisible by 128/160/192; compute-bound) --
        const int M=3840,N=3840,K=4096;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        float8_t *A; uint8_t* Bshuf; __hip_bfloat16*C; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&C,(size_t)M*N*2)); HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        double gflop=2.0*M*N*K/1e12;
        float m4=time_trfeed<128,2>(A,Bshuf,C,as,bs,M,N,K,30);
        float m5=time_trfeed<160,2>(A,Bshuf,C,as,bs,M,N,K,30);
        float m6=time_trfeed<192,2>(A,Bshuf,C,as,bs,M,N,K,30);
        double t4=gflop/(m4/1e3),t5=gflop/(m5/1e3),t6=gflop/(m6/1e3);
        const double WPC=15.9;                       // WMMA/cyc at the 307 TF ceiling
        printf("\n=== acc-tile reuse sweep — perf @ %d^3 (ceiling %.0f TF = %.1f WMMA/cyc) ===\n",M,CEIL,WPC);
        printf("  acc[4][4] TBM=128 (128 acc-VGPR, reuse 2.00): %7.3f ms  %6.1f TF  (%2.0f%% of 307)  %.2f WMMA/cyc\n",
               m4,t4,100*t4/CEIL,WPC*t4/CEIL);
        printf("  acc[5][4] TBM=160 (160 acc-VGPR, reuse 2.22): %7.3f ms  %6.1f TF  (%2.0f%% of 307)  %.2f WMMA/cyc\n",
               m5,t5,100*t5/CEIL,WPC*t5/CEIL);
        printf("  acc[6][4] TBM=192 (192 acc-VGPR, reuse 2.40): %7.3f ms  %6.1f TF  (%2.0f%% of 307)  %.2f WMMA/cyc\n",
               m6,t6,100*t6/CEIL,WPC*t6/CEIL);
        printf("  slope acc[4]->acc[6]: %+.1f TF  (%+.2f WMMA/cyc)\n", t6-t4, WPC*(t6-t4)/CEIL);
        printf("  VERDICT: %s\n", t6>t4*1.02 ? "RISES -> reuse lever bends the curve; extrapolate acc[8][8] (needs dyn-VGPR)"
                                 : (t6<t4*0.98 ? "FALLS -> static occupancy loss outweighs reuse; dyn-VGPR REQUIRED to realize big tiles"
                                              : "FLAT -> reuse gain ~cancels static occupancy loss; dyn-VGPR needed to break the tie"));
        hipFree(A);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
    }
    return 0;
}
