#pragma once

// Cross-host tensor parallelism, milestone M3: the FOLLOWER entry point.
//
// A tensor-parallel world of two processes is asymmetric only in ROLE, never in shape. Rank 0 is
// an ordinary llama-server or llama-cli - it owns the LM head, it samples, it serves HTTP. Every
// other rank owns the rest of the world's device window and does exactly one thing: apply the
// batches and memory mutations rank 0 mirrors to it, so that both ranks call llama_decode() over
// identical inputs and their per-layer reduce exchanges line up.
//
// This lives in common/ rather than in a binary of its own on purpose (spec T7): the follower has
// to construct its model and context through the SAME common_init_from_params() the server and
// the CLI use, or the two ranks build different contexts and diverge in ways HELLO cannot always
// see. Both llama-cli and llama-server divert into it a few lines after argument parsing.

struct common_params;

// Runs a tensor-parallel follower to completion. Returns a process exit status: 0 when the leader
// closed the world cleanly, 1 on a divergence, a failed handshake or a lost connection (the reason
// is logged). Only call when common_tp_is_follower(params) is true.
int common_tp_follower_run(common_params & params);

// True when these parameters describe a follower rank: a tensor-parallel world wider than one
// device, and a rank that does not own world device 0.
bool common_tp_is_follower(const common_params & params);
