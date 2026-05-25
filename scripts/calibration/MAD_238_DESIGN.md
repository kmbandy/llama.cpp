# MAD-238 — pybind11 Weight Pager Binding for Calibration

**Status:** design doc, no code yet.
**Branch:** `wip/mad-230-paging-correctness` (or a new branch off this once we start).
**Scope estimate:** ~2 weeks focused work (this is the honest number after reading the C++ surface).

---

## Why

Calibration currently does `AutoModelForCausalLM.from_pretrained(...).to("cuda:0")`. Caps us at models that fit in VRAM:

| Model | f16 size | Fits R9700 (32 GB)? |
|---|---|---|
| Qwen3.5-4B | 8 GB | ✓ |
| Qwen3.6-35B-A3B | ~70 GB | ✗ |
| Qwen3.5-REAP-97B-A10B | ~194 GB | ✗ (production pager paged it at 0.34 t/s decode) |

To calibrate ml8 on the actual production target (35B-A3B) and stretch (97B-A10B), the Python calibration pipeline needs to use the **same weight pager** that production inference uses (MAD-230 closed Saturday). One paging story across calibration + inference.

## What we already have (C++ side)

`src/weight-pager/` — production code shipped MAD-230:
- `wp-pager.h/cpp` — `WeightPager` façade (173+418 lines)
- `wp-page-catalog.h/cpp` — tensor-name → file_idx/offset/size metadata (115+241 lines)
- `wp-pool.h/cpp` — LRU ring of VRAM slots in one ggml_backend_buffer (101+114 lines)
- `wp-file-io.h/cpp` — sync + async (io_uring) page reads (92+336 lines)
- `wp-gpu-transport.h/cpp` — host→device transfer + zero-pad (85+247 lines)
- `wp-prefetch.h/cpp` — N-deep speculative prefetch queue (134+358 lines)
- `wp-eval-cb.h/cpp` — ggml eval callback adapter that drives the pager from inference (27+555 lines)

Total ~3000 LOC of production C++. **All correct, all running on R9700 today.** We don't touch this code — we wrap it.

## What we need to add

### 1. `python_bindings/wp/` — pybind11 module

```cpp
// python_bindings/wp/wp_bindings.cpp
PYBIND11_MODULE(wp_native, m) {
    py::class_<wp::WeightPager>(m, "WeightPager")
        .def(py::init<>())
        .def("add_page",      &wp::WeightPager::add_page)
        .def("init",          &wp::WeightPager::init)
        .def("shutdown",      &wp::WeightPager::shutdown)
        .def("find_page",     &wp::WeightPager::find_page)
        .def("n_pages",       &wp::WeightPager::n_pages)
        .def("ensure",        &wp::WeightPager::ensure,
             py::return_value_policy::reference)
        .def("prefetch_page", &wp::WeightPager::prefetch_page)
        .def("tick",          &wp::WeightPager::tick)
        .def("page_meta",     &wp::WeightPager::page_meta,
             py::return_value_policy::reference_internal);

    py::class_<wp::WeightPager::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("n_slots", &wp::WeightPager::Config::n_slots)
        .def_readwrite("prefetch_depth", &wp::WeightPager::Config::prefetch_depth)
        .def_readwrite("prefer_async_io", &wp::WeightPager::Config::prefer_async_io);

    py::class_<wp::PageMeta>(m, "PageMeta")
        .def_readonly("tensor_name", &wp::PageMeta::tensor_name)
        .def_readonly("size", &wp::PageMeta::size)
        .def_readonly("block_idx", &wp::PageMeta::block_idx)
        .def_readonly("is_consolidated", &wp::PageMeta::is_consolidated);

    // Helper: HIP backend buffer type (needed by init()).
    m.def("hip_buffer_type", [](int device_idx) {
        return ggml_backend_cuda_buffer_type(device_idx);
    });

    // Helper: prepare fd per pager precondition (dup + clear O_DIRECT).
    m.def("open_gguf_for_paging", [](const std::string & path) {
        return wp::dup_clear_o_direct(path);
    });
}
```

### 2. `scripts/calibration/wp_wrapper.py` — Python convenience layer

```python
import torch
import gguf
from wp_native import WeightPager, Config, hip_buffer_type, open_gguf_for_paging

class PagedModel:
    """Python-side facade: GGUF → catalog populated → pager initialized →
    ergonomic .get_tensor(name) returning a torch GPU tensor."""

    def __init__(self, gguf_paths: list[str], device_idx: int = 0,
                 n_slots: int = 200, prefetch_depth: int = 4):
        self.pager = WeightPager()
        # 1. Build catalog from GGUF metadata
        readers = [gguf.GGUFReader(p) for p in gguf_paths]
        for file_idx, reader in enumerate(readers):
            for t in reader.tensors:
                self.pager.add_page(t.name, file_idx, t.data_offset, t.n_bytes)
        # 2. Open fds
        fds = [open_gguf_for_paging(p) for p in gguf_paths]
        # 3. Init pager
        cfg = Config()
        cfg.n_slots = n_slots
        cfg.prefetch_depth = prefetch_depth
        ok = self.pager.init(cfg, hip_buffer_type(device_idx), device_idx,
                             fds, [device_idx])
        if not ok:
            raise RuntimeError("WeightPager init failed")
        # 4. Cache tensor metadata for shape/dtype
        self._meta = {t.name: (t.shape, t.tensor_type)
                      for r in readers for t in r.tensors}

    def get_tensor(self, name: str) -> torch.Tensor:
        """Page-in the tensor and return a torch GPU view.

        IMPORTANT: returned tensor's lifetime is bounded by the slot. If the
        pager evicts this slot (LRU under pressure), the tensor's data becomes
        invalid. For safe long-lived use, .clone() before retaining.
        """
        page_idx = self.pager.find_page(name)
        if page_idx < 0:
            raise KeyError(name)
        ptr = self.pager.ensure(page_idx)
        if ptr is None:
            raise RuntimeError(f"ensure({name}) returned null")
        shape, dtype = self._meta[name]
        # Reverse for numpy: GGUF stores ne-natural (innermost-first), torch is opposite
        torch_shape = tuple(reversed(shape))
        # torch.from_blob over the VRAM pointer
        # NOTE: this is the lifetime-fragile path; we may switch to a copy-out
        # variant when MAD-231 slot-pin lands.
        return _wrap_vram_ptr(ptr, torch_shape, dtype, device_idx=0)

    def get_tensor_copy(self, name: str) -> torch.Tensor:
        """Safer: copy the page into a fresh torch tensor we own."""
        view = self.get_tensor(name)
        return view.clone()  # forces a CUDA→CUDA memcpy onto a fresh allocation
```

### 3. Calibration flow rewrite — layer-sequential via `PagedLinear`

Current `calibrate_ml8.py` runs full forward through the model for EVERY calibration sample (32×1024 tokens × 96 linears). Doesn't work with paging — we'd page-in/out the same weights on every sample.

**Better than re-implementing Qwen's forward path: subclass `nn.Linear`.**

```python
class PagedLinear(nn.Linear):
    """nn.Linear whose .weight is page-faulted from a WeightPager on access.

    HF transformers' existing forward code uses .weight transparently; our
    page-on-access property is invisible to it. After calibration, we overlay
    the dequantized ml8 weight by setting .weight_override, switching the
    property to return the override (so subsequent layers see post-quant output).
    """
    def __init__(self, pager, page_idx, shape, dtype, bias=False):
        super().__init__(shape[1], shape[0], bias=bias)
        del self._parameters['weight']   # drop HF-allocated weight
        self.pager = pager
        self.page_idx = page_idx
        self.weight_shape = shape
        self.weight_dtype = dtype
        self.weight_override = None      # post-quant overlay

    @property
    def weight(self):
        if self.weight_override is not None:
            return self.weight_override
        return ensure_as_torch(self.pager, self.page_idx,
                              self.weight_shape, self.weight_dtype)
```

The trick: **swap HF's `nn.Linear` modules with `PagedLinear` instances at model-load time.** HF transformers' code that uses `self.q_proj(x)` is unchanged — it just transparently page-faults each forward call.

This sidesteps re-implementing RoPE / attention / MLP / RMSNorm / SSM layers in Python — HF does all that, we only intercept the parameter access path.

**Calibration loop becomes:**
```python
# Forward + collect Hessian + AWQ stats (existing path, unchanged)
H, sum_abs = compute_hessian(layer, calib, model, dev, collect_awq=True)
# Quantize
export = gptq_quantize_linear(layer, H, q, ...)
# Overlay the dequantized weight (paged layer now serves quantized data)
layer.weight_override = reconstruct_inference_weight_from(export, ...)
```

Subsequent layers' forward passes see the quantized output naturally — no per-sample re-paging because `weight_override` short-circuits the pager. Once a layer is quantized, its pager slot is free for the next layer to claim.

**Memory budget for 35B-A3B:**
- Pool: 16-32 slots × 17 MB MoE slot size = 272-544 MB VRAM (vs 70 GB for full model)
- Per-layer Hessian: 9216² × 4 = 340 MB (down_proj is the biggest)
- Per-layer activation cache: 32 × 1024 × 2560 × 2 = 167 MB (intermediate hidden)
- Pre-quant cached activations (one layer worth): 167 MB on CPU RAM
- Total: ~1 GB VRAM + ~1 GB CPU at any time → fits easily

**Sequential flow:**
```python
for layer_idx in range(n_layers):
    # 1. Inputs to this layer = outputs of previous (cached on CPU between iterations)
    input_acts = load_cached_activations(layer_idx)
    # 2. Forward THIS LAYER ONLY: weight is page-faulted via PagedLinear
    output_acts = model.model.layers[layer_idx](input_acts, ...)
    # 3. The Hessian + sum_abs were collected via forward hooks during step 2
    # 4. Quantize each Linear inside this layer: gate_proj, up_proj, down_proj
    for lin_name in ("gate_proj", "up_proj", "down_proj"):
        lin = get_module(model, f"model.layers.{layer_idx}.mlp.{lin_name}")
        # GPTQ as before, but lin.weight is the paged tensor
        export = gptq_quantize_linear(lin, H[lin_name], q, ...)
        lin.weight_override = reconstruct_inference_weight_from(export)
        save_blob(export, lin_name)
    # 5. Cache layer's outputs for next iter
    save_cached_activations(layer_idx + 1, output_acts)
```

## Scope breakdown (honest, REVISED 2026-05-24 after bindings shipped)

| Piece | Status | Days | Risk |
|---|---|---|---|
| pybind11 module + setup.py | **DONE** (iter 1+2+3 today) | 0.5 done | low |
| Catalog API (add_page, find, meta, MoE sub-experts) | **DONE** + 7 TDD tests | done | low |
| Init/ensure/shutdown via init_for_device | **DONE** + GPU integration test gated | done | low |
| device_memcpy + ensure_as_torch wrapper | **DONE** + 2 CPU tests, GPU test gated | done | low |
| PagedLinear nn.Module subclass | TODO | 1 | medium (HF parameter override) |
| Module-swap at model-load: replace nn.Linear with PagedLinear | TODO | 1 | medium (HF model surgery) |
| Layer-sequential calibration loop + activation caching | TODO | 2-3 | medium (test on 4B vs current path for parity first) |
| Real 35B-A3B end-to-end test + debug | TODO | 2-3 | high (first real big run, hardware-safety budget) |
| **REMAINING** | | **6-7 days** | |

That's ~1.5-2 weeks of focused work, matching the earlier estimate.

## Open design questions

1. **Slot pinning during calibration**: when we `ensure()` a weight and start doing GPTQ on it, can the LRU evict it mid-computation? Production pager works because the eval callback finishes consuming each weight in one ggml step. For Python calibration, computations can span multiple Python statements with garbage collection in between. Need either explicit pin/unpin OR copy-out semantics (slower but safe).

2. **Activation cache size**: layer-sequential calibration caches per-layer activations between layers. For 35B-A3B at 32×1024 tokens, each layer's activation tensor is `32 * 1024 * 2560 * 2 = ~167 MB` (Qwen3.6-35B-A3B hidden=2560). 32 of these = 5.3 GB. Fits in RAM but tight. Could spool to disk if needed.

3. **MoE expert dispatch**: 35B-A3B has 64 experts per layer, gates choose ~8 per token. For calibration we need to quantize ALL 64 experts per layer. The pager has consolidated-expert sub-pages — we'd iterate through them.

4. **Calibration time on 35B**: at the 4B pace (~12s/linear), 35B has ~32 layers × (3 MLP + 64 experts × 3 expert linears) = 6,176 linears. At 12s each = 20 hours. Could parallelize across multiple experts within a layer (all 64 share the same H since they share the same input).

## What to commit TODAY

Just this design doc + reading. Implementation starts in a focused session.

When implementation starts, first cell to validate end-to-end on:
- 4B model (Qwen3.5-4B) via the paged path → should reproduce today's Cell D number
- THEN scale to 35B-A3B

Reproducing the 4B number through the new path is the gate for trusting the 35B result.
