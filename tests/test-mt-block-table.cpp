// MAD-137: unit tests for mt::BlockTable — the per-sequence
// logical→physical block mapping. Bare main()/assert style; no real GPU.

#include "../src/memory-tier/mt-block-pool.h"   // kInvalidBlockId
#include "../src/memory-tier/mt-block-table.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

using mt::BlockTable;
using mt::kInvalidBlockId;

int main() {
    // ─── init() — accessors mirror the constructor args ───
    {
        BlockTable t;
        t.init(/* max_seqs */ 4, /* block_size */ 16);
        assert(t.max_seqs() == 4);
        assert(t.block_size() == 16);
        // No appends yet — every seq has zero blocks.
        for (uint32_t s = 0; s < 4; ++s) {
            assert(t.num_blocks((int) s) == 0);
            assert(t.get_physical((int) s, 0) == kInvalidBlockId);
        }
        printf("test-mt-block-table: init + zero state ok\n");
    }

    // ─── append + get_physical ───
    {
        BlockTable t;
        t.init(2, 16);
        t.append_block(0, 100);
        t.append_block(0, 101);
        t.append_block(0, 102);
        assert(t.num_blocks(0) == 3);
        assert(t.get_physical(0, 0) == 100);
        assert(t.get_physical(0, 1) == 101);
        assert(t.get_physical(0, 2) == 102);
        // Out-of-range logical_idx is a sentinel, not UB.
        assert(t.get_physical(0, 3) == kInvalidBlockId);
        printf("test-mt-block-table: append + get_physical ok\n");
    }

    // ─── get_physical_for_pos — pos / block_size division ───
    {
        BlockTable t;
        t.init(1, 16);
        t.append_block(0, 200);
        t.append_block(0, 201);
        t.append_block(0, 202);

        // Block 0 covers pos [0, 16); block 1 covers [16, 32); etc.
        assert(t.get_physical_for_pos(0, 0)  == 200);
        assert(t.get_physical_for_pos(0, 15) == 200);
        assert(t.get_physical_for_pos(0, 16) == 201);
        assert(t.get_physical_for_pos(0, 31) == 201);
        assert(t.get_physical_for_pos(0, 32) == 202);
        assert(t.get_physical_for_pos(0, 47) == 202);
        // Past the live range → sentinel.
        assert(t.get_physical_for_pos(0, 48) == kInvalidBlockId);
        // Negative pos guard.
        assert(t.get_physical_for_pos(0, -1) == kInvalidBlockId);
        printf("test-mt-block-table: get_physical_for_pos arithmetic ok\n");
    }

    // ─── swap_block — returns old id, get_physical reflects new ───
    {
        BlockTable t;
        t.init(1, 16);
        t.append_block(0, 300);
        t.append_block(0, 301);
        t.append_block(0, 302);

        const uint32_t old = t.swap_block(0, /* logical_idx */ 1, /* new */ 999);
        assert(old == 301);
        assert(t.get_physical(0, 1) == 999);
        // Neighbors untouched.
        assert(t.get_physical(0, 0) == 300);
        assert(t.get_physical(0, 2) == 302);
        assert(t.num_blocks(0) == 3);  // swap doesn't change count
        printf("test-mt-block-table: swap_block ok\n");
    }

    // ─── Non-contiguous mapping — swap_block(..., kInvalidBlockId) creates a hole ───
    // MAD-128 partial seq_rm uses this pattern: a logical block is wiped
    // (its physical id replaced with kInvalidBlockId) but the surrounding
    // blocks remain. The paged-attn kernel handles kInvalidBlockId by
    // returning -INFINITY logits for the missing positions.
    {
        BlockTable t;
        t.init(1, 16);
        t.append_block(0, 400);
        t.append_block(0, 401);
        t.append_block(0, 402);
        t.append_block(0, 403);

        const uint32_t evicted = t.swap_block(0, 2, kInvalidBlockId);
        assert(evicted == 402);

        assert(t.get_physical(0, 0) == 400);
        assert(t.get_physical(0, 1) == 401);
        assert(t.get_physical(0, 2) == kInvalidBlockId);  // the hole
        assert(t.get_physical(0, 3) == 403);

        // num_blocks still counts the hole — the logical sequence length
        // is unchanged; only the underlying physical mapping was wiped.
        assert(t.num_blocks(0) == 4);
        printf("test-mt-block-table: non-contiguous (hole) mapping ok\n");
    }

    // ─── clear_seq — returns the freed ids in append order, then empties ───
    {
        BlockTable t;
        t.init(2, 16);
        t.append_block(0, 500);
        t.append_block(0, 501);
        t.append_block(0, 502);
        // Throw a hole in to make sure it propagates through clear_seq too.
        t.swap_block(0, 1, kInvalidBlockId);

        const auto freed = t.clear_seq(0);
        assert(freed.size() == 3);
        assert(freed[0] == 500);
        assert(freed[1] == kInvalidBlockId);  // the hole survives clearing
        assert(freed[2] == 502);

        assert(t.num_blocks(0) == 0);
        assert(t.get_physical(0, 0) == kInvalidBlockId);
        printf("test-mt-block-table: clear_seq returns freed list + empties ok\n");
    }

    // ─── clear_seq on an already-empty sequence is a no-op ───
    {
        BlockTable t;
        t.init(2, 16);
        const auto freed = t.clear_seq(0);
        assert(freed.empty());
        printf("test-mt-block-table: clear_seq on empty seq ok\n");
    }

    // ─── Per-seq isolation — appends on one seq don't leak to others ───
    {
        BlockTable t;
        t.init(3, 16);
        t.append_block(0, 600);
        t.append_block(2, 700);

        assert(t.num_blocks(0) == 1);
        assert(t.num_blocks(1) == 0);
        assert(t.num_blocks(2) == 1);
        assert(t.get_physical(0, 0) == 600);
        assert(t.get_physical(1, 0) == kInvalidBlockId);
        assert(t.get_physical(2, 0) == 700);

        // Clearing seq 0 doesn't touch seq 2.
        t.clear_seq(0);
        assert(t.num_blocks(0) == 0);
        assert(t.num_blocks(2) == 1);
        assert(t.get_physical(2, 0) == 700);
        printf("test-mt-block-table: per-seq isolation ok\n");
    }

    // ─── Out-of-range seq queries return safe sentinels ───
    {
        BlockTable t;
        t.init(2, 16);
        // Negative seq — every accessor must return safe values.
        assert(t.get_physical(-1, 0) == kInvalidBlockId);
        assert(t.get_physical_for_pos(-1, 0) == kInvalidBlockId);
        assert(t.num_blocks(-1) == 0);
        assert(t.clear_seq(-1).empty());

        // Seq >= max_seqs — same.
        assert(t.get_physical(99, 0) == kInvalidBlockId);
        assert(t.get_physical_for_pos(99, 0) == kInvalidBlockId);
        assert(t.num_blocks(99) == 0);
        assert(t.clear_seq(99).empty());
        printf("test-mt-block-table: out-of-range seq queries safe ok\n");
    }

    // ─── reset() — wipes every sequence; the table is reusable ───
    {
        BlockTable t;
        t.init(3, 16);
        t.append_block(0, 800);
        t.append_block(1, 801);
        t.append_block(2, 802);
        assert(t.num_blocks(0) == 1 && t.num_blocks(1) == 1 && t.num_blocks(2) == 1);

        t.reset();
        for (uint32_t s = 0; s < 3; ++s) {
            assert(t.num_blocks((int) s) == 0);
            assert(t.get_physical((int) s, 0) == kInvalidBlockId);
        }
        // Accessors still report init values.
        assert(t.max_seqs() == 3);
        assert(t.block_size() == 16);

        // Table is reusable post-reset.
        t.append_block(0, 900);
        assert(t.get_physical(0, 0) == 900);
        printf("test-mt-block-table: reset() + reusable ok\n");
    }

    // ─── Different block sizes resolve positions correctly ───
    {
        BlockTable t;
        t.init(1, 32);  // larger blocks — block 0 covers pos [0, 32)
        t.append_block(0, 700);
        t.append_block(0, 701);

        assert(t.get_physical_for_pos(0, 0)  == 700);
        assert(t.get_physical_for_pos(0, 31) == 700);
        assert(t.get_physical_for_pos(0, 32) == 701);
        assert(t.get_physical_for_pos(0, 63) == 701);
        assert(t.get_physical_for_pos(0, 64) == kInvalidBlockId);
        printf("test-mt-block-table: block_size=32 arithmetic ok\n");
    }

    printf("test-mt-block-table: ALL PASS\n");
    return 0;
}
