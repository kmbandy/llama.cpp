// MAD-137: unit tests for mt::BlockPool — the physical block allocator
// that backs the paged tier (GPU pool + CPU pool, refcounting, watermark
// admission control). Bare main()/assert style; no real GPU.

#include "../src/memory-tier/mt-block-pool.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

using mt::BlockPool;
using mt::kInvalidBlockId;

int main() {
    // ─── Basic alloc / free cycle ───
    {
        BlockPool p;
        p.init(4, 4, 0.0f);
        assert(p.n_free_gpu() == 4);
        assert(p.n_free_cpu() == 4);

        const uint32_t b0 = p.alloc_gpu();
        assert(b0 != kInvalidBlockId);
        assert(p.refcount(b0) == 1);
        assert(p.n_free_gpu() == 3);

        p.free_block(b0);
        assert(p.refcount(b0) == 0);
        assert(p.n_free_gpu() == 4);
        printf("test-mt-block-pool: basic alloc/free ok\n");
    }

    // ─── Allocation order — init pushes descending so pop returns 0 first ───
    {
        BlockPool p;
        p.init(4, 0, 0.0f);
        const uint32_t a = p.alloc_gpu();
        const uint32_t b = p.alloc_gpu();
        const uint32_t c = p.alloc_gpu();
        assert(a == 0 && b == 1 && c == 2);
        printf("test-mt-block-pool: deterministic alloc order ok\n");
    }

    // ─── GPU vs CPU id ranges + is_gpu() ───
    {
        BlockPool p;
        p.init(4, 4, 0.0f);
        const uint32_t g = p.alloc_gpu();
        const uint32_t c = p.alloc_cpu();
        assert(g < 4);                 // GPU range [0, 4)
        assert(c >= 4 && c < 8);       // CPU range [4, 8)
        assert(p.is_gpu(g));
        assert(!p.is_gpu(c));
        printf("test-mt-block-pool: gpu/cpu id ranges ok (gpu=%u cpu=%u)\n", g, c);
    }

    // ─── Exhaustion — drain GPU, next alloc returns kInvalidBlockId, CPU unaffected ───
    {
        BlockPool p;
        p.init(2, 2, 0.0f);
        const uint32_t a = p.alloc_gpu();
        const uint32_t b = p.alloc_gpu();
        assert(a != kInvalidBlockId && b != kInvalidBlockId);
        assert(p.n_free_gpu() == 0);

        const uint32_t exhausted = p.alloc_gpu();
        assert(exhausted == kInvalidBlockId);

        // CPU pool independent: still allocates fine.
        const uint32_t c = p.alloc_cpu();
        assert(c != kInvalidBlockId);
        assert(p.n_free_cpu() == 1);
        printf("test-mt-block-pool: exhaustion + multi-pool independence ok\n");
    }

    // ─── Refcount bump (CoW path) ───
    {
        BlockPool p;
        p.init(4, 0, 0.0f);
        const uint32_t b = p.alloc_gpu();
        assert(p.refcount(b) == 1);
        assert(p.n_free_gpu() == 3);

        p.bump_ref(b);
        assert(p.refcount(b) == 2);
        assert(p.n_free_gpu() == 3);  // no change — still allocated

        p.free_block(b);
        assert(p.refcount(b) == 1);
        assert(p.n_free_gpu() == 3);  // still NOT in free stack — second owner holds it

        p.free_block(b);
        assert(p.refcount(b) == 0);
        assert(p.n_free_gpu() == 4);  // last ref dropped → returned to free stack
        printf("test-mt-block-pool: bump_ref + multi-owner free ok\n");
    }

    // ─── Watermark admission — has_free_*_blocks() reserves a slice ───
    {
        BlockPool p;
        // 10 GPU blocks, 0.5 watermark → reserve = ceil(10 * 0.5) = 5.
        // (Using 0.5f because it's exactly representable in float; values
        // like 0.2f promote to double as 0.2000…0004, which trips ceil()
        // and silently bumps the reserve by 1 — conservative behavior,
        // but not what callers writing exact fractions might expect.)
        // n_free_gpu() reports 10 (raw), has_free_gpu_blocks(N) is true
        // only when (free - reserve) >= N → max admissible N = 5.
        p.init(10, 0, 0.5f);
        assert(p.n_free_gpu() == 10);
        assert(p.has_free_gpu_blocks(5));
        assert(!p.has_free_gpu_blocks(6));   // would dip into reserve

        // Drain to reserve floor — has_free should refuse any further admit.
        for (int i = 0; i < 5; ++i) {
            const uint32_t id = p.alloc_gpu();
            assert(id != kInvalidBlockId);
        }
        assert(p.n_free_gpu() == 5);
        assert(!p.has_free_gpu_blocks(1));   // exactly at watermark → false

        // alloc_*() ignores watermark — still allocates the reserved blocks.
        for (int i = 0; i < 5; ++i) {
            const uint32_t id = p.alloc_gpu();
            assert(id != kInvalidBlockId);
        }
        assert(p.n_free_gpu() == 0);
        printf("test-mt-block-pool: watermark gating + alloc bypass ok\n");
    }

    // ─── Watermark with 0% disables the reserve ───
    {
        BlockPool p;
        p.init(4, 0, 0.0f);
        assert(p.has_free_gpu_blocks(4));
        assert(!p.has_free_gpu_blocks(5));
        printf("test-mt-block-pool: 0%% watermark = no reserve ok\n");
    }

    // ─── Watermark applies independently to CPU pool ───
    {
        BlockPool p;
        p.init(0, 10, 0.5f);  // reserve = ceil(10 * 0.5) = 5
        assert(p.has_free_cpu_blocks(5));
        assert(!p.has_free_cpu_blocks(6));
        // GPU side has 0 blocks → has_free returns false for any N.
        assert(!p.has_free_gpu_blocks(1));
        printf("test-mt-block-pool: cpu-side watermark + empty-gpu side ok\n");
    }

    // ─── Double-free is idempotent (warn, no corruption) ───
    {
        BlockPool p;
        p.init(2, 0, 0.0f);
        const uint32_t b = p.alloc_gpu();
        p.free_block(b);
        assert(p.refcount(b) == 0);
        const size_t free_after_first = p.n_free_gpu();
        p.free_block(b);  // second free — should be a no-op
        assert(p.refcount(b) == 0);
        assert(p.n_free_gpu() == free_after_first);  // didn't push twice
        printf("test-mt-block-pool: double-free is idempotent ok\n");
    }

    // ─── kInvalidBlockId on every entry-point is a no-op ───
    {
        BlockPool p;
        p.init(2, 2, 0.0f);
        p.bump_ref(kInvalidBlockId);          // no crash
        p.free_block(kInvalidBlockId);        // no crash
        assert(p.refcount(kInvalidBlockId) == 0);
        // Pool state untouched.
        assert(p.n_free_gpu() == 2);
        assert(p.n_free_cpu() == 2);
        printf("test-mt-block-pool: kInvalidBlockId paths safe ok\n");
    }

    // ─── refcount() for out-of-range ids returns 0 (safe accessor) ───
    {
        BlockPool p;
        p.init(2, 2, 0.0f);
        assert(p.refcount(9999) == 0);
        printf("test-mt-block-pool: out-of-range refcount safe ok\n");
    }

    // ─── reset() — restores all free counts and zeros all refcounts ───
    {
        BlockPool p;
        p.init(4, 4, 0.0f);
        const uint32_t g = p.alloc_gpu();
        const uint32_t c = p.alloc_cpu();
        p.bump_ref(g);
        p.bump_ref(c);
        assert(p.n_free_gpu() == 3 && p.n_free_cpu() == 3);
        assert(p.refcount(g) == 2 && p.refcount(c) == 2);

        p.reset();
        assert(p.n_free_gpu() == 4 && p.n_free_cpu() == 4);
        assert(p.refcount(g) == 0 && p.refcount(c) == 0);

        // Post-reset alloc returns the same low IDs as a fresh init.
        const uint32_t g2 = p.alloc_gpu();
        const uint32_t c2 = p.alloc_cpu();
        assert(g2 == 0 && c2 == 4);
        printf("test-mt-block-pool: reset() restores all counts + refcounts ok\n");
    }

    // ─── total_*_blocks() reports init values, immutable across alloc ───
    {
        BlockPool p;
        p.init(7, 3, 0.1f);
        assert(p.total_gpu_blocks() == 7);
        assert(p.total_cpu_blocks() == 3);
        (void) p.alloc_gpu();
        (void) p.alloc_cpu();
        assert(p.total_gpu_blocks() == 7);   // doesn't shrink with alloc
        assert(p.total_cpu_blocks() == 3);
        printf("test-mt-block-pool: total_*_blocks() immutable ok\n");
    }

    printf("test-mt-block-pool: ALL PASS\n");
    return 0;
}
