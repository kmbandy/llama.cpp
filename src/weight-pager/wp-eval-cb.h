#pragma once

// weight_pager_eval_cb — ggml backend scheduler eval-callback adapter for
// the weight pager.
//
// Signature must match ggml_backend_sched's eval_callback type:
//   bool (*)(struct ggml_tensor * t, bool ask, void * user_data);
//
// Pass &wp::weight_pager_eval_cb to ggml_backend_sched_set_eval_callback,
// with user_data set to a wp::WeightPagerSet*.
//
// The callback fires twice per node: once with ask=true before execution
// (this is when we patch tensor->data and tensor->buffer for paged
// weights) and once with ask=false after. We only act on ask=true.

struct ggml_tensor;

namespace wp {

class WeightPager;
class WeightPagerSet;

// Free function with the ggml callback signature. user_data must be a
// WeightPagerSet*; nullptr is treated as "pager not active" and the callback
// returns true (no-op).
bool weight_pager_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

// Drain eval-callback state associated with pager before the pager tears down.
void weight_pager_eval_cb_reset(WeightPager * pager);

bool wp_paged_batch_enabled();

// Diagnostic (WP_PROFILE_EVAL=1, default off): print the total host-side wall
// time spent inside weight_pager_eval_cb over the run, split into the Step-2
// ensure/patch phase vs page-resolution+other. Lets us see whether paged-decode
// slowdown is host-side callback overhead or GPU/scheduler. No-op when disabled.
void weight_pager_eval_cb_print_profile();

}  // namespace wp
