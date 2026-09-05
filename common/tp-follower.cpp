#include "tp-follower.h"

#include "common.h"
#include "log.h"
#include "llama.h"

bool common_tp_is_follower(const common_params & params) {
    return params.tp_world > 1 && params.tp_rank > 0;
}

int common_tp_follower_run(common_params & params) {
    if (!common_tp_is_follower(params)) {
        LOG_ERR("%s: not a tensor-parallel follower (--tp-world %d, --tp-rank %d)\n",
                __func__, params.tp_world, params.tp_rank);
        return 1;
    }
    if (params.tp_peer.empty()) {
        LOG_ERR("%s: a follower rank needs --tp-peer HOST:PORT (the address rank 0 is bound to)\n",
                __func__);
        return 1;
    }

    // A follower must not decode anything of its own. The warm-up run in common_init_from_params()
    // is exactly that: a two-token decode issued locally, which would run a graph - and therefore
    // a hundred and twenty-eight reduce exchanges - that the leader is not running. Rank 0 keeps
    // its warm-up; the follower receives it as an ordinary mirrored batch, in order, like any
    // other decode.
    if (params.warmup) {
        LOG_INF("%s: tensor-parallel follower: disabling the local warm-up run; rank 0's warm-up "
                "arrives as a mirrored batch instead\n", __func__);
        params.warmup = false;
    }

    // No sampling on a follower: it never reads a logit. (backend_sampling is also forced off in
    // common_init_from_params for BOTH ranks under --tp-world, because it changes the graph.)
    params.sampling.backend_sampling = false;

    // Speculation is deliberately NOT stripped here. Under spec E.3 Option B the follower never
    // runs a draft or MTP graph - the drafted tokens reach it as ordinary token ids inside the
    // next verify batch - but --spec-type still feeds llama_cparams::n_rs_seq, the recurrent-state
    // rollback depth, which is part of the memory shape and is compared in the HELLO handshake.
    // Clearing it here would make the two ranks allocate differently shaped recurrent state and
    // fail the handshake for a reason that has nothing to do with what the operator typed. The
    // follower simply never builds a draft context, because it never runs the server or CLI loop
    // that would create one.

    LOG_INF("%s: tensor-parallel follower: world %d devices, this rank owns the window starting at "
            "world device %d, peer socket %s %s\n",
            __func__, params.tp_world, params.tp_rank,
            llama_tp_should_listen(params.tp_peer.c_str(), params.tp_rank, params.tp_listen)
                ? "bound on" : "dialling",
            params.tp_peer.c_str());

    // Same construction path as rank 0. Everything that could differ between the ranks - context
    // size, batch sizes, KV types, the tensor split, the model itself - is compared field by field
    // in the HELLO handshake inside llama_init_from_model() and refused there.
    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * ctx   = llama_init ? llama_init->context() : nullptr;
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load the model / create the context\n", __func__);
        return 1;
    }
    if (!llama_tp_is_follower(ctx)) {
        LOG_ERR("%s: the context did not come up as a tensor-parallel follower - check that "
                "--tp-world, --tp-rank and --device agree (this rank's device count must be "
                "--tp-world minus rank 0's)\n", __func__);
        return 1;
    }

    LOG_INF("%s: tensor-parallel follower ready; waiting for the leader\n", __func__);

    int64_t n_ops = 0;
    for (;;) {
        const int32_t st = llama_tp_follower_step(ctx);
        if (st == LLAMA_TP_STEP_OK) {
            n_ops++;
            continue;
        }
        if (st == LLAMA_TP_STEP_SHUTDOWN) {
            LOG_INF("%s: tensor-parallel follower finished after %lld mirrored operations\n",
                    __func__, (long long) n_ops);
            return 0;
        }
        // llama_tp_follower_step() has already logged what it saw. Returning here closes the
        // socket, which is what makes the leader's next reduce fail its request instead of
        // blocking on a partial sum that will never arrive.
        LOG_ERR("%s: tensor-parallel follower stopping after %lld mirrored operations\n",
                __func__, (long long) n_ops);
        return 1;
    }
}
