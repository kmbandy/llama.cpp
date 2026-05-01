#pragma once

// weight_pager_eval_cb — ggml backend scheduler eval-callback adapter for
// the weight pager.
//
// Signature must match ggml_backend_sched's eval_callback type:
//   bool (*)(struct ggml_tensor * t, bool ask, void * user_data);
//
// Pass &wp::weight_pager_eval_cb to ggml_backend_sched_set_eval_callback,
// with user_data set to a wp::WeightPager*.
//
// The callback fires twice per node: once with ask=true before execution
// (this is when we patch tensor->data and tensor->buffer for paged
// weights) and once with ask=false after. We only act on ask=true.

struct ggml_tensor;

namespace wp {

class WeightPager;

// Free function with the ggml callback signature. user_data must be a
// WeightPager*; nullptr is treated as "pager not active" and the callback
// returns true (no-op).
bool weight_pager_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

}  // namespace wp
