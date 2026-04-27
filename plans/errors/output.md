ulimit: Permission denied when changing resource of type 'Maximum size that may be locked into memory'
ggml_cuda_init: found 2 ROCm devices (Total VRAM: 48992 MiB):
  Device 0: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 32624 MiB
  Device 1: AMD Radeon RX 6900 XT, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 16368 MiB
build_info: b8942-ff060471c
system_info: n_threads = 12 (n_threads_batch = 12) / 24 | ROCm : NO_VMM = 1 | PEER_MAX_BATCH_SIZE = 128 | FA_ALL_QUANTS = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
Running without SSL
init: using 23 threads for HTTP server
start: binding port with default address family
main: loading model
srv    load_model: loading model '/home/kmbandy/models/MiniMax-M2.7-UD-IQ3_S-00001-of-00003.gguf'
srv    load_model: tiered KV: total ctx=196608, hot ctx=24576 (12%)
llama_model_load_from_file_impl: using device ROCm0 (AMD Radeon AI PRO R9700) (0000:42:00.0) - 32532 MiB free
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 50 key-value pairs and 809 tensors from /home/kmbandy/models/MiniMax-M2.7-UD-IQ3_S-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = minimax-m2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 40
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Minimax-M2.7
llama_model_loader: - kv   6:                           general.basename str              = Minimax-M2.7
llama_model_loader: - kv   7:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   8:                         general.size_label str              = 256x4.9B
llama_model_loader: - kv   9:                            general.license str              = other
llama_model_loader: - kv  10:                       general.license.name str              = modified-mit
llama_model_loader: - kv  11:                       general.license.link str              = https://github.com/MiniMax-AI/MiniMax...
llama_model_loader: - kv  12:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  13:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv  14:                     minimax-m2.block_count u32              = 62
llama_model_loader: - kv  15:                  minimax-m2.context_length u32              = 196608
llama_model_loader: - kv  16:                minimax-m2.embedding_length u32              = 3072
llama_model_loader: - kv  17:             minimax-m2.feed_forward_length u32              = 1536
llama_model_loader: - kv  18:            minimax-m2.attention.head_count u32              = 48
llama_model_loader: - kv  19:         minimax-m2.attention.head_count_kv u32              = 8
llama_model_loader: - kv  20:                  minimax-m2.rope.freq_base f32              = 5000000.000000
llama_model_loader: - kv  21: minimax-m2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  22:                    minimax-m2.expert_count u32              = 256
llama_model_loader: - kv  23:               minimax-m2.expert_used_count u32              = 8
llama_model_loader: - kv  24:              minimax-m2.expert_gating_func u32              = 2
llama_model_loader: - kv  25:            minimax-m2.attention.key_length u32              = 128
llama_model_loader: - kv  26:          minimax-m2.attention.value_length u32              = 128
llama_model_loader: - kv  27:      minimax-m2.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:            minimax-m2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  30:                         tokenizer.ggml.pre str              = minimax-m2
llama_model_loader: - kv  31:                      tokenizer.ggml.tokens arr[str,200064]  = ["Ā", "ā", "Ă", "ă", "Ą", "ą", ...
llama_model_loader: - kv  32:                  tokenizer.ggml.token_type arr[i32,200064]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  33:                      tokenizer.ggml.merges arr[str,199744]  = ["Ġ Ġ", "Ġ t", "Ġ a", "i n", "e r...
llama_model_loader: - kv  34:                tokenizer.ggml.bos_token_id u32              = 200034
llama_model_loader: - kv  35:                tokenizer.ggml.eos_token_id u32              = 200020
llama_model_loader: - kv  36:            tokenizer.ggml.unknown_token_id u32              = 200021
llama_model_loader: - kv  37:            tokenizer.ggml.padding_token_id u32              = 200004
llama_model_loader: - kv  38:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  39:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  40:                    tokenizer.chat_template str              = {# ----------‑‑‑ special token ...
llama_model_loader: - kv  41:               general.quantization_version u32              = 2
llama_model_loader: - kv  42:                          general.file_type u32              = 26
llama_model_loader: - kv  43:                      quantize.imatrix.file str              = MiniMax-M2.7-GGUF/imatrix_unsloth.gguf
llama_model_loader: - kv  44:                   quantize.imatrix.dataset str              = unsloth_calibration_MiniMax-M2.7.txt
llama_model_loader: - kv  45:             quantize.imatrix.entries_count u32              = 496
llama_model_loader: - kv  46:              quantize.imatrix.chunks_count u32              = 81
llama_model_loader: - kv  47:                                   split.no u16              = 0
llama_model_loader: - kv  48:                        split.tensors.count i32              = 809
llama_model_loader: - kv  49:                                split.count u16              = 3
llama_model_loader: - type  f32:  373 tensors
llama_model_loader: - type q6_K:  250 tensors
llama_model_loader: - type iq3_s:   62 tensors
llama_model_loader: - type iq2_s:  124 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = IQ3_S - 3.4375 bpw
print_info: file size   = 77.86 GiB (2.92 BPW) 
load: 0 unused tokens
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: printing all EOG tokens:
load:   - 200004 ('<fim_pad>')
load:   - 200005 ('<reponame>')
load:   - 200020 ('[e~[')
load: special tokens cache size = 54
load: token to piece cache size = 1.3355 MB
print_info: arch                  = minimax-m2
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 196608
print_info: n_embd                = 3072
print_info: n_embd_inp            = 3072
print_info: n_layer               = 62
print_info: n_head                = 48
print_info: n_head_kv             = 8
print_info: n_rot                 = 64
print_info: n_swa                 = 0
print_info: is_swa_any            = 0
print_info: n_embd_head_k         = 128
print_info: n_embd_head_v         = 128
print_info: n_gqa                 = 6
print_info: n_embd_k_gqa          = 1024
print_info: n_embd_v_gqa          = 1024
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 0.0e+00
print_info: n_ff                  = 1536
print_info: n_expert              = 256
print_info: n_expert_used         = 8
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 5000000.0
print_info: freq_scale_train      = 1
print_info: n_ctx_orig_yarn       = 196608
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = 230B.A10B
print_info: model params          = 228.69 B
print_info: general.name          = Minimax-M2.7
print_info: vocab type            = BPE
print_info: n_vocab               = 200064
print_info: n_merges              = 199744
print_info: BOS token             = 200034 ']~!b['
print_info: EOS token             = 200020 '[e~['
print_info: UNK token             = 200021 ']!d~['
print_info: PAD token             = 200004 '<fim_pad>'
print_info: LF token              = 10 'Ċ'
print_info: FIM PRE token         = 200001 '<fim_prefix>'
print_info: FIM SUF token         = 200003 '<fim_suffix>'
print_info: FIM MID token         = 200002 '<fim_middle>'
print_info: FIM PAD token         = 200004 '<fim_pad>'
print_info: FIM REP token         = 200005 '<reponame>'
print_info: EOG token             = 200004 '<fim_pad>'
print_info: EOG token             = 200005 '<reponame>'
print_info: EOG token             = 200020 '[e~['
print_info: max token length      = 256
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: offloading output layer to GPU
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloaded 63/63 layers to GPU
load_tensors:          CPU model buffer size =     0.00 MiB
load_tensors:        ROCm0 model buffer size =     0.00 MiB
init_weight_pager: initializing weight pager with 809 pages
init_weight_pager: VRAM: 32532 MiB free / 32624 MiB total
init_weight_pager: initializing VRAM pool with 8 slots (495 MiB each = 3960 MiB total)
init_weight_pager: calling init_pool with slot_size=519045120 n_slots=8
init_pool: allocating 8 slots of 519045120 bytes each
init_pool: allocated pool.base=0x7f056dc00000 (slot_size=519045120, n_slots=8)
init_weight_pager: init_pool succeeded
init_weight_pager: set placeholder data pointers on 809 weight tensors
init_weight_pager: io_uring initialized for async prefetch
init_weight_pager: weight pager initialized
common_init_result: added <fim_pad> logit bias = -inf
common_init_result: added <reponame> logit bias = -inf
common_init_result: added [e~[ logit bias = -inf
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 24576
llama_context: n_ctx_seq     = 24576
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = enabled
llama_context: kv_unified    = false
llama_context: freq_base     = 5000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_seq (24576) < n_ctx_train (196608) -- the full capacity of the model will not be utilized
llama_context:  ROCm_Host  output buffer size =     0.76 MiB
llama_kv_cache:      ROCm0 KV buffer size =  1581.13 MiB
llama_kv_cache: TurboQuant rotation matrices initialized (128x128)
llama_kv_cache: size = 1581.00 MiB ( 24576 cells,  62 layers,  1/1 seqs), K (turbo4):  790.50 MiB, V (turbo4):  790.50 MiB
llama_kv_cache: upstream attention rotation disabled (TurboQuant uses kernel-level WHT)
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 128
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 128
sched_reserve: reserving ...
sched_reserve: resolving fused Gated Delta Net support:
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
sched_reserve:      ROCm0 compute buffer size =   396.75 MiB
sched_reserve:  ROCm_Host compute buffer size =    60.01 MiB
sched_reserve: graph nodes  = 4099
sched_reserve: graph splits = 2
sched_reserve: reserve took 22.15 ms, sched copies = 1
llama_model_load_from_file_impl: using device ROCm0 (AMD Radeon AI PRO R9700) (0000:42:00.0) - 26444 MiB free
llama_model_load_from_file_impl: using device ROCm1 (AMD Radeon RX 6900 XT) (0000:0b:00.0) - 16332 MiB free
llama_model_loader: loaded meta data with 29 key-value pairs and 197 tensors from /home/kmbandy/models/bge-small-en-v1.5-q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bert
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Bge Small En v1.5
llama_model_loader: - kv   3:                            general.version str              = v1.5
llama_model_loader: - kv   4:                           general.finetune str              = en
llama_model_loader: - kv   5:                           general.basename str              = bge
llama_model_loader: - kv   6:                         general.size_label str              = small
llama_model_loader: - kv   7:                            general.license str              = mit
llama_model_loader: - kv   8:                               general.tags arr[str,5]       = ["sentence-transformers", "feature-ex...
llama_model_loader: - kv   9:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv  10:                           bert.block_count u32              = 12
llama_model_loader: - kv  11:                        bert.context_length u32              = 512
llama_model_loader: - kv  12:                      bert.embedding_length u32              = 384
llama_model_loader: - kv  13:                   bert.feed_forward_length u32              = 1536
llama_model_loader: - kv  14:                  bert.attention.head_count u32              = 12
llama_model_loader: - kv  15:          bert.attention.layer_norm_epsilon f32              = 0.000000
llama_model_loader: - kv  16:                      bert.attention.causal bool             = false
llama_model_loader: - kv  17:                          bert.pooling_type u32              = 2
llama_model_loader: - kv  18:            tokenizer.ggml.token_type_count u32              = 2
llama_model_loader: - kv  19:                       tokenizer.ggml.model str              = bert
llama_model_loader: - kv  20:                         tokenizer.ggml.pre str              = jina-v2-en
llama_model_loader: - kv  21:                      tokenizer.ggml.tokens arr[str,30522]   = ["[PAD]", "[unused0]", "[unused1]", "...
llama_model_loader: - kv  22:                  tokenizer.ggml.token_type arr[i32,30522]   = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  23:            tokenizer.ggml.unknown_token_id u32              = 100
llama_model_loader: - kv  24:          tokenizer.ggml.seperator_token_id u32              = 102
llama_model_loader: - kv  25:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  26:               tokenizer.ggml.mask_token_id u32              = 103
llama_model_loader: - kv  27:               general.quantization_version u32              = 2
llama_model_loader: - kv  28:                          general.file_type u32              = 7
llama_model_loader: - type  f32:  124 tensors
llama_model_loader: - type q8_0:   73 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 34.38 MiB (8.68 BPW) 
load: 0 unused tokens
load: printing all EOG tokens:
load:   - 0 ('[PAD]')
load: special tokens cache size = 5
load: token to piece cache size = 0.2032 MB
print_info: arch                  = bert
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 512
print_info: n_embd                = 384
print_info: n_embd_inp            = 384
print_info: n_layer               = 12
print_info: n_head                = 12
print_info: n_head_kv             = 12
print_info: n_rot                 = 32
print_info: n_swa                 = 0
print_info: is_swa_any            = 0
print_info: n_embd_head_k         = 32
print_info: n_embd_head_v         = 32
print_info: n_gqa                 = 1
print_info: n_embd_k_gqa          = 384
print_info: n_embd_v_gqa          = 384
print_info: f_norm_eps            = 1.0e-12
print_info: f_norm_rms_eps        = 0.0e+00
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 0.0e+00
print_info: n_ff                  = 1536
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 0
print_info: pooling type          = 2
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 10000.0
print_info: freq_scale_train      = 1
print_info: n_ctx_orig_yarn       = 512
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = 33M
print_info: model params          = 33.21 M
print_info: general.name          = Bge Small En v1.5
print_info: vocab type            = WPM
print_info: n_vocab               = 30522
print_info: n_merges              = 0
print_info: BOS token             = 101 '[CLS]'
print_info: UNK token             = 100 '[UNK]'
print_info: SEP token             = 102 '[SEP]'
print_info: PAD token             = 0 '[PAD]'
print_info: MASK token            = 103 '[MASK]'
print_info: LF token              = 0 '[PAD]'
print_info: FIM PAD token         = 0 '[PAD]'
print_info: EOG token             = 0 '[PAD]'
print_info: max token length      = 21
load_tensors: loading model tensors, this can take a while... (mmap = true, direct_io = false)
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/13 layers to GPU
load_tensors:   CPU_Mapped model buffer size =    34.38 MiB
................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 512
llama_context: n_ctx_seq     = 512
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 0
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 1
llama_context:        CPU  output buffer size =     0.12 MiB
sched_reserve: reserving ...
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
sched_reserve:      ROCm0 compute buffer size =     5.10 MiB
sched_reserve:  ROCm_Host compute buffer size =     3.76 MiB
sched_reserve: graph nodes  = 396
sched_reserve: graph splits = 208 (with bs=512), 1 (with bs=1)
sched_reserve: reserve took 27.62 ms, sched copies = 1
srv  server_tiere: semantic KV index loaded: /home/kmbandy/models/bge-small-en-v1.5-q8_0.gguf (CPU)
srv    load_model: tiered cache initialized, hot=12.500000%, warm=50.000000%, cold=37.500000%
srv    load_model: initializing slots, n_slots = 1
process_ubatch: calling ggml_backend_sched_alloc_graph
process_ubatch: ggml_backend_sched_alloc_graph succeeded
graph_compute: calling ggml_backend_sched_graph_compute_async
weight_pager_eval_cb: tensor=embd ask=false
weight_pager_eval_cb: pager=0x55f6a0b35a80 pool.base=0x7f056dc00000 pool.n_slots=8
ensure: tensor=token_embd.weight page_idx=0 slot_idx=-1 vram_ptr=(nil)
ensure: calling page_in for tensor token_embd.weight
page_in: loading tensor token_embd.weight (file_idx=1, offset=504203680, size=504161280)
page_in: pread fd=13 dst=0x7f056dc00000 size=504161280 offset=504203680
page_in: pread returned 504161280 (errno=21)
ensure: page_in returned, page.slot_idx=0 page.vram_ptr=0x7f056dc00000
weight_pager_eval_cb: tensor=norm-0 ask=false
weight_pager_eval_cb: pager=0x55f6a0b35a80 pool.base=0x7f056dc00000 pool.n_slots=8
weight_pager_eval_cb: tensor=attn_norm-0 ask=false
weight_pager_eval_cb: pager=0x55f6a0b35a80 pool.base=0x7f056dc00000 pool.n_slots=8
ensure: tensor=blk.0.attn_norm.weight page_idx=7 slot_idx=-1 vram_ptr=(nil)
ensure: calling page_in for tensor blk.0.attn_norm.weight
page_in: loading tensor blk.0.attn_norm.weight (file_idx=1, offset=1010949536, size=12288)
page_in: pread fd=13 dst=0x7f05aba00000 size=12288 offset=1010949536
page_in: pread returned 12288 (errno=21)
ensure: page_in returned, page.slot_idx=2 page.vram_ptr=0x7f05aba00000
fish: Job 1, '/home/kmbandy/GitHub/llama.cpp/…' terminated by signal SIGSEGV (Address boundary error)