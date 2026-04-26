  ---                                                                                                                                        
  Phase 1 — Embed model infrastructure
                                                                                                                                             
  You are implementing a semantic KV cache index for llama-server on a ROCm/HIP build.
  SSH target: ssh kmbandy@mad-lab-main                                                                                                       
  Repo: ~/GitHub/llama.cpp, branch: feature-kv-cache-improvements                                                                            
  Build: cd ~/GitHub/llama.cpp && cmake --build build --target llama-server -j$(nproc)                                                       
                                                                                                                                             
  Goal: Add optional bge-small embedding model support to server_tiered_cache.                                                               
                  
  1. Add --kv-semantic-index <path> to common_params (follow how --model is added in                                                         
     src/common/common.h and common.cpp). Default empty string = disabled.
                                                                                                                                             
  2. In tools/server/server-tiered-cache.h, add an optional embed model context:                                                             
     - struct llama_model * sem_model = nullptr;                                                                                             
     - struct llama_context * sem_ctx = nullptr;                                                                                             
     - std::vector<float> embed(const std::string & text);
     - bool sem_enabled() const { return sem_model != nullptr; }                                                                             
                  
  3. In server_tiered_cache constructor, if params.kv_semantic_index is non-empty:                                                           
     - Load the model with llama_model_load() using n_gpu_layers=0 (CPU always)
     - Create context with n_ctx=512, embeddings=true                                                                                        
     - Log: "semantic KV index loaded: <path> (CPU)"                                                                                         
                                                                                                                                             
  4. Implement embed(): tokenize text, run llama_decode(), extract embeddings via                                                            
     llama_get_embeddings_seq(), L2-normalize the result vector, return it.                                                                  
                                                                                                                                             
  5. Add destructor cleanup for sem_model/sem_ctx.
                                                                                                                                             
  Build must pass clean. No functional behavior change yet — just the infrastructure.                                                        
  
  ---                                                                                                                                        
  Phase 2 — Fingerprint on eviction

  Continuing feature-kv-cache-improvements on mad-lab-main (ssh kmbandy@mad-lab-main).
  Repo: ~/GitHub/llama.cpp. Build: cmake --build build --target llama-server -j$(nproc)                                                      
                                                                                                                                             
  Phase 1 is complete: server_tiered_cache has sem_model/sem_ctx and an embed() method.                                                      
                                                                                                                                             
  Goal: When tokens evict to warm or cold, compute and store a semantic fingerprint.                                                         
                  
  1. Add to llama-kv-cache-tiered.h:                                                                                                         
     struct SemanticFingerprint {
         std::vector<llama_pos> positions;  // token positions this covers                                                                   
         std::vector<float>     embedding;  // 768-dim normalized vector                                                                     
         llama_cache_tier       tier;       // TIER_WARM or TIER_COLD
         uint64_t               turn;       // conversation turn when evicted                                                                
     };                                                                                                                                      
     Add: std::vector<SemanticFingerprint> fingerprints; to llama_kv_cache_tiered private.                                                   
     Add: void add_fingerprint(const std::vector<llama_pos>& positions,                                                                      
                                std::vector<float>&& embedding, llama_cache_tier tier);
                                                                                                                                             
  2. In server_tiered_cache::evict_from_slot(), after successful eviction:                                                                   
     - If sem_enabled(), detokenize the evicted positions using llama_token_to_piece()                                                       
       on the vocab (pass llama_model* to evict_from_slot or store it at init)                                                               
     - Call embed() on the resulting string                                                                                                  
     - Call tiered_cache->add_fingerprint(positions, embedding, target_tier)                                                                 
                                                                                                                                             
  3. add_fingerprint() stores to the fingerprints vector, capped at 1000 entries                                                             
     (evict oldest fingerprint when over cap).                                                                                               
                                                                                                                                             
  4. Persist fingerprints to ssd_path/fingerprints.bin on each write using simple                                                            
     binary format: [n_entries][per entry: n_pos, positions[], n_embd, floats[], tier, turn]                                                 
                                                                                                                                             
  Build clean, no behavior change to the hot path.                                                                                           
                                                                                                                                             
  ---                                                                                                                                        
  Phase 3 — Similarity scoring and prefetch hints
                                                 
  Continuing feature-kv-cache-improvements on mad-lab-main (ssh kmbandy@mad-lab-main).
  Repo: ~/GitHub/llama.cpp. Build: cmake --build build --target llama-server -j$(nproc)                                                      
                                                                                                                                             
  Phases 1+2 complete: embed model loads, fingerprints stored on eviction.                                                                   
                                                                                                                                             
  Goal: On each new user input, score fingerprints and return prefetch candidates.                                                           
                  
  1. Add to llama_kv_cache_tiered:                                                                                                           
     struct PrefetchHint {
         std::vector<llama_pos> positions;                                                                                                   
         float                  score;
         llama_cache_tier       current_tier;                                                                                                
     };
     std::vector<PrefetchHint> score_fingerprints(                                                                                           
         const std::vector<float>& query_emb, int top_k, float threshold) const;                                                             
                                                                                                                                             
  2. score_fingerprints(): cosine similarity between query_emb and each fingerprint,                                                         
     return top_k results above threshold, sorted by score desc.                                                                             
     Cosine similarity: dot(a,b) since both are L2-normalized = just inner product.                                                          
                                                                                                                                             
  3. Add to server_tiered_cache:                                                                                                             
     std::vector<llama_kv_cache_tiered::PrefetchHint>                                                                                        
     get_prefetch_hints(int slot_id, const std::string& input_text, int top_k = 5);                                                          
     — embeds input_text, calls score_fingerprints, returns hints.                                                                           
                                                                                                                                             
  4. In server-context.cpp, find where new user input is first processed (before                                                             
     llama_decode is called for the prompt). If tiered_cache->is_enabled() and                                                               
     sem_enabled(), call get_prefetch_hints() and log the top results:                                                                       
     "semantic prefetch: score=X.XX positions=[a..b] tier=warm"                                                                              
                                                                                                                                             
     Do NOT wire actual restoration yet — just log the hints this phase.                                                                     
                                                                                                                                             
  5. Add --kv-semantic-threshold (default 0.65) and --kv-semantic-topk (default 5)                                                           
     to common_params.
                                                                                                                                             
  Build clean. Validate by running llama-server with a bge-small GGUF and checking                                                           
  prefetch hint logs appear when revisiting a topic.
                                                                                                                                             
  ---             
  Phase 4 — Wire prefetch to restoration + semantic eviction weighting                                                                       
                  
  Continuing feature-kv-cache-improvements on mad-lab-main (ssh kmbandy@mad-lab-main).
  Repo: ~/GitHub/llama.cpp. Build: cmake --build build --target llama-server -j$(nproc)                                                      
  
  Phases 1-3 complete: hints logged. Now make them do real work.                                                                             
                  
  Goal: Act on prefetch hints and weight eviction by semantic similarity.                                                                    
                  
  1. PREFETCH: In server-context.cpp where hints are logged, after logging, for each                                                         
     hint with current_tier == TIER_WARM:
     - Call tiered_cache->migrate_in_slot(slot_id, hint.positions, TIER_WARM, TIER_HOT)                                                      
     - This uses the existing warm_copy_from_host path to restore to VRAM                                                                    
     For TIER_COLD hints, log "cold prefetch TODO" for now — cold restoration path                                                           
     not yet stable.                                                                                                                         
                                                                                                                                             
  2. EVICTION WEIGHTING: In llama_kv_cache_tiered::evict_tokens(), after getting                                                             
     eviction candidates from token_metadata.get_eviction_candidates():
     - If fingerprints is non-empty and a query embedding is available (add                                                                  
       set_current_query_embedding(std::vector<float> emb) public method, stored                                                             
       as current_query_emb member):                                                                                                         
     - For each candidate position, find its fingerprint if one exists                                                                       
     - Compute similarity to current_query_emb                                                                                               
     - Reorder candidates: positions with similarity < 0.3 evicted first,
       positions with similarity > 0.65 moved to back of eviction queue                                                                      
     - This keeps semantically relevant tokens hot longer                                                                                    
                                                                                                                                             
  3. Wire set_current_query_embedding() call into the same place in server-context.cpp                                                       
     where get_prefetch_hints() is called — pass the embedded query vector through.                                                          
                                                                                                                                             
  4. Add to llama_tier_stats:                                                                                                                
     uint32_t semantic_prefetch_hits = 0;                                                                                                    
     uint32_t semantic_eviction_saves = 0;                                                                                                   
     Log these in the existing stats output.                                                                                                 
  
  5. Build, run with Qwen3-27B at --kv-tiered-enabled, multi-turn conversation                                                               
     switching between 2 distinct topics. Confirm in logs: prefetch hints fire on
     topic return, eviction saves fire when relevant tokens would have been evicted.                                                         
                                                                                                                                             
  ---   