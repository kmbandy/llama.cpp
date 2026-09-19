R4D kernel library -- vendored subset (attention entry points only)

Source:      /home/kmbandy/.cache/radiance-libr4d/b9e42ab-rx6/
Commit id:   b9e42ab-rx6
License:     none present upstream at the source path above (internal use only;
             do not redistribute outside the org without checking with the
             library owner first).

What was copied and why:

  r4d.h                                     full C ABI header (unmodified)
  r4d_common.h                              shared device-side machinery (unmodified)
  r4d_dt16.h                                16-bit operand traits (unmodified)
  r4d_attn_paged_h256_gqa6.hip               attention dispatch: r4d_attn_prefill_h256_gqa6_{fp8kv,bf16kv},
                                             r4d_attn_decode_h256_gqa6_{fp8kv,bf16kv},
                                             r4d_attn_decode_h256_gqa6_scratch_bytes, r4d_attn_dims
  r4d_attn_prefill_h256_gqa6.hip             prefill kernel template, #included by the dispatch TU above
  r4d_attn_decode_h256_gqa6.hip              decode kernel + split-KV combine template, #included by
                                             the dispatch TU above
  Makefile, build.sh                        upstream build reference only (not invoked by the ggml
                                             build; kept so the flags/units upstream compiles with
                                             stay visible and auditable)

  r4d_gdn_*.hip, r4d_gdn_wmma.h             gated-delta-net kernels, copied for future use but NOT
                                             wired into the ggml-hip CMake build yet (see
                                             ggml/src/ggml-cuda/CMakeLists.txt r4d section)

Explicitly EXCLUDED (not copied):
  r4d_module.hip           pybind11 module surface -- has a torch/pybind dependency this build does
                            not want.
  r4d_registry.hip          introspection table (r4d_kernel_count/r4d_kernel_at) -- not needed by any
                            entry point this integration calls; not part of the "minimal set for the
                            attention entry points" this vendoring targets.
  r4d_ar_*.hip, r4d_gemm_*.hip, r4d_attn_vit_h72_bf16.hip
                            all-reduce, GEMM and vision-attention kernels -- out of scope for this task.
  *.o, *.so, .git, .gitignore, README.md    build artifacts / upstream repo metadata, not sources.

ggml-side integration:
  ggml-r4d.h    (new, not from upstream) -- extern "C" trampoline exposing r4d.h's attention entry
                points plus ggml_cuda_r4d_available() to the rest of ggml-cuda.
