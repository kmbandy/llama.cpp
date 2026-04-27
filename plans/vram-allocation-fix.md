  Task: Fix weight paging to prevent initial VRAM allocation in llama.cpp
                                                                                                                                             
  Context:        
  This is a fork of llama.cpp with a custom weight paging system that pages model weights from NVMe directly to VRAM on demand via SAM/BAR1. 
  The pager is implemented across these files:                                                                                               
  - src/llama-weight-pager.h — struct definitions
  - src/llama-weight-pager.cpp — implementation                                                                                              
  - src/llama.cpp — init_weight_pager() called at model load
  - src/llama-context.cpp — ggml_backend_sched_set_eval_callback registered at line ~1206                                                    
  - src/llama-model.cpp — model tensor allocation                                                                                            
                                                                                                                                             
  The Problem:                                                                                                                               
  When --weight-paging is used with a model larger than VRAM, startup fails with:                                                            
  ggml_backend_cuda_buffer_type_alloc_buffer: allocating 79252.12 MiB on device 0: cudaMalloc failed: out of memory                          
  alloc_tensor_range: failed to allocate ROCm0 buffer of size 83101871104                                          
  The eval callback intercepts at inference time, but llama.cpp tries to allocate ALL model weights into VRAM at load time — before inference
   ever starts.                                                                                                                              
                                                                                                                                             
  The Fix (3 parts):                                                                                                                         
                                                                                                                                             
  Part 1 — Prevent initial VRAM allocation (src/llama-model.cpp)
                                                                                                                                             
  In llama_model::load_tensors(), find the section that handles the non-mmap allocation path (around line 7990). It already has this logic:  
  if (ml.no_alloc) {
      buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer                                                              
      for (ggml_tensor * t = ggml_get_first_tensor(ctx); ...) {                
          t->buffer = buf; // set dummy buffer                 
      }                                                                                                                                      
  } else {
      buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer                                                              
  }                                                                            
                                                                                                                                             
  When weight_paging_enabled is true AND the buffer type is a GPU/ROCm buffer AND we're NOT doing a dry-run (i.e., params.no_alloc is false),
   force ml.no_alloc = true for this specific context allocation. Weight tensors will get a zero-size dummy buffer — no actual VRAM          
  allocated.      
                                                                                                                                             
  The cleanest approach: before the context loop that allocates buffers, check:                                                              
  const bool weight_paging_no_alloc = params.weight_paging_enabled;
  Then in the allocation branch, use weight_paging_no_alloc instead of ml.no_alloc when deciding whether to do a dummy allocation for GPU    
  buffer types. Non-GPU buffers (CPU, host) should still allocate normally.                                                                  
                                                                                                                                             
  After this change, at if (ml.no_alloc) { return true; } (line ~8048), we must NOT return early when weight paging is enabled — we still    
  need to initialize the weight pager. So change that early return to:                                                                       
  if (ml.no_alloc && !params.weight_paging_enabled) {
      return true;                                                                                                                           
  }               
  And skip load_all_data when weight paging is enabled (we don't read data at load time):
  if (!params.weight_paging_enabled) {
      for (auto & [ctx, buf_map] : ctx_buf_maps) {                                                                                           
          if (!ml.load_all_data(...)) { return false; }
      }                                                                                                                                      
  }               
                                                                                                                                             
  Part 2 — Set tensor->data in the eval callback (src/llama-weight-pager.cpp)
                                                                                                                                             
  Currently weight_pager_eval_cb calls pager->ensure(name) to get a VRAM pointer but doesn't set tensor->data. After a no_alloc load,        
  tensor->data is nullptr. The kernel will segfault.                                                                                         
                                                                                                                                             
  Fix the ask=false branch of weight_pager_eval_cb to also set tensor->data:                                                                 
  // In the ask=false branch, after calling ensure():
  void * vram_ptr = pager->ensure(ggml_get_name(t));                                                                                         
  if (vram_ptr) {                                                                                                                            
      t->data = vram_ptr;                                                                                                                    
  }                                                                                                                                          
                                                                                                                                             
  This redirects the tensor's data pointer to the live VRAM slot just before the kernel executes.
                                                                                                                                             
  Part 3 — Handle the case where tensor->data must stay valid (src/llama-weight-pager.cpp)
                                                                                                                                             
  The eval callback fires per-node. After a slot is evicted (LRU), the tensor->data pointer for the evicted tensor becomes stale — it points 
  to a slot that now contains different data. This is actually fine because:
  - We set tensor->data fresh each time the tensor is about to execute                                                                       
  - The pager's ensure() either returns the existing valid slot (if still resident) or pages it back in                                      
                                                                                                       
  No extra work needed here — the existing ensure() logic handles this correctly as long as we always set tensor->data before kernel         
  execution.                                                                                                                                 
                                                                                                                                             
  What NOT to change:                                                                                                                        
  - fit.cpp already has use_direct_io = false — don't touch
  - llama-context.cpp eval callback registration is correct — don't touch                                                                    
  - init_weight_pager() in llama.cpp is correct — don't touch            
  - The llama_weight_pager::ensure() / page_in() / evict_lru() logic is correct — don't touch                                                
                                                                                                                                             
  Key invariant to preserve:                                                                                                                 
  The dummy buffer set on weight tensors (t->buffer = dummy_buf) must be a valid ROCm buffer type (zero-size, but correct type). This tells  
  the ggml scheduler to run weight-consuming ops on the GPU backend. Do NOT change weight tensors to CPU buffer type — that would cause GPU  
  kernels to be skipped.
                                                                                                                                             
  Success criteria:
  Server starts, logs show weight pager initializing with N pages, model loads without OOM, and during inference the eval callback fires and
  pages tensors in/out of the VRAM slot pool.                                                                                                
  
  ---                                                                                                                                        
  That gives Qwen the full picture — the exact files, the existing hook points, the three discrete changes, and the invariants to preserve.