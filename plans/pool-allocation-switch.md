  ---                                                                                                                                        
  Phase 1 — Switch pool allocation + add pinned staging buffer                                                                               
                                                                                                                                             
  In llama-weight-pager.h, add a void* pinned_staging member to llama_weight_pager. Update the destructor to also call hipFree(pool.base) and
   hipHostFree(pinned_staging).                                                                                                              
   
  In llama-weight-pager.cpp, init_pool: replace hipExtMallocWithFlags(&pool.base, size, hipDeviceMallocFinegrained) and its fallback with a  
  single hipMalloc(&pool.base, size). After successful pool allocation, allocate the pinned staging buffer: hipHostMalloc(&pinned_staging, 
  slot_size, hipHostMallocDefault). Return false if either allocation fails.                                                                 
                  
  Remove _mm_sfence() and the <immintrin.h> include — no longer needed.                                                                      
   
  ---                                                                                                                                        
  Phase 2 — Rewrite page_in to use pinned staging + hipMemcpy
                                                                                                                                             
  In page_in in llama-weight-pager.cpp, replace the pread-directly-to-VRAM path with:
  1. pread(fd, pinned_staging, size, offset) — read from NVMe into pinned host buffer                                                        
  2. Check return value as before                                                                                                            
  3. hipMemcpy(dst, pinned_staging, size, hipMemcpyHostToDevice) — DMA from pinned RAM to VRAM slot                                          
  4. Keep all existing diagnostic logging, just update the log message to note the two-step path                                             
                                                                                                                                             
  This phase should get end-to-end inference working. Rebuild and test before moving to Phase 3.                                             
                                                                                                                                             
  ---                                                                                                                                        
  Phase 3 — Async pipeline with N+1 prefetch
                                                                                                                                             
  Add a hipStream_t transfer_stream member to llama_weight_pager, created in init_pool with hipStreamCreate and destroyed in the destructor.
                                                                                                                                             
  Change hipMemcpy in page_in to hipMemcpyAsync(..., transfer_stream) followed by hipStreamSynchronize(transfer_stream) for the synchronous  
  case.                                                                                                                                      
                                                                                                                                             
  Then wire the existing io_uring N+1 prefetch (already in the eval callback's Phase 4 block) to also kick off the GPU transfer leg: when    
  complete_prefetch finishes the NVMe read into the pinned buffer, immediately issue hipMemcpyAsync to the pre-reserved next slot on
  transfer_stream. This way the GPU transfer overlaps with GPU compute on the current tensor, and by the time the eval callback needs N+1,   
  it's already in VRAM.