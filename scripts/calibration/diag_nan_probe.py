#!/usr/bin/env python3
"""Pinpoint the determinism-induced NaN in the first full-attention layer of the 0.8B
dense-hybrid Qwen3.5 bed. Honors ML8_DETERMINISTIC=1 (replicates the driver's determinism
block EXACTLY, before torch import). Loads the model resident (0.8B fits), registers:
  (a) a forward hook on EVERY decoder layer  -> first layer whose output goes NaN
  (b) a forward hook on EVERY submodule of the first full_attention layer -> the exact op
Runs ONE forward over a small real-token batch and prints, in execution-completion order,
each module's  input-nan? / output-nan? / output-inf? / finite-absmax.

Usage:  ML8_DETERMINISTIC=1 /usr/bin/python3 diag_nan_probe.py      # hammer ON
        /usr/bin/python3 diag_nan_probe.py                          # hammer OFF (control)
"""
import os
import sys

# ── replicate the driver determinism block (env half MUST precede torch import) ──
_DET = os.environ.get("ML8_DETERMINISTIC") == "1"
if _DET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HIPBLASLT_DETERMINISTIC", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

if _DET:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for _attr, _val in (("allow_tf32", False),
                        ("allow_fp16_reduced_precision_reduction", False),
                        ("allow_bf16_reduced_precision_reduction", False)):
        try: setattr(torch.backends.cuda.matmul, _attr, _val)
        except AttributeError: pass
    try: torch.backends.cudnn.allow_tf32 = False
    except AttributeError: pass

print(f"[probe] ML8_DETERMINISTIC={'1 (HAMMER ON)' if _DET else '0 (control)'}", flush=True)

MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()

# match the driver: apply the RDNA fla fp32-scan shim (env ML8_FLA_SHIM=1)
if os.environ.get("ML8_FLA_SHIM") == "1":
    from fla_compat import apply_fla_arch_shim
    apply_fla_arch_shim(model, DEVICE)

# ── locate the decoder-layer list (robust to multimodal carrier nesting) ──
def find_layers(m):
    for name, mod in m.named_modules():
        if mod.__class__.__name__.endswith("DecoderLayer"):
            # parent ModuleList: strip trailing ".<idx>"
            return name.rsplit(".", 1)[0], mod
    raise RuntimeError("no *DecoderLayer found")

first_layer_name, _ = find_layers(model)
layers = dict(model.named_modules())[first_layer_name]
print(f"[probe] decoder layers at '{first_layer_name}'  n={len(layers)}", flush=True)

# layer_types from config to label which is the first full_attention
lt = getattr(model.config, "text_config", model.config)
layer_types = getattr(lt, "layer_types", None)
if layer_types is None:
    layer_types = ["?"] * len(layers)
first_full = next((i for i, t in enumerate(layer_types) if "full" in t), 3)
print(f"[probe] layer_types[:8]={layer_types[:8]}  first_full_attention=layer {first_full}", flush=True)

events = []  # (label, in_nan, out_nan, out_inf, finite_absmax)

def stat(t):
    if not torch.is_tensor(t):
        # tuple/list output: inspect the first tensor element
        if isinstance(t, (tuple, list)) and len(t) and torch.is_tensor(t[0]):
            t = t[0]
        else:
            return (None, None, None, None)
    tf = t.float()
    n = int(torch.isnan(tf).sum())
    inf = int(torch.isinf(tf).sum())
    finite = tf[torch.isfinite(tf)]
    amax = float(finite.abs().max()) if finite.numel() else float("nan")
    return (n, inf, amax, t.numel())

def mk_hook(label):
    def h(mod, inp, out):
        in_t = inp[0] if isinstance(inp, (tuple, list)) and len(inp) else inp
        in_nan = int(torch.isnan(in_t.float()).sum()) if torch.is_tensor(in_t) else -1
        n, inf, amax, numel = stat(out)
        events.append((label, in_nan, n, inf, amax, numel))
    return h

handles = []
# (a) every decoder layer
for i in range(len(layers)):
    handles.append(layers[i].register_forward_hook(mk_hook(f"LAYER[{i:02d}] ({layer_types[i]})")))
# (b) every submodule of the first full_attention layer
for sub_name, sub_mod in layers[first_full].named_modules():
    if sub_name == "":
        continue
    handles.append(sub_mod.register_forward_hook(
        mk_hook(f"  L{first_full}.{sub_name} <{sub_mod.__class__.__name__}>")))

# ── input: either a benign synthetic sentence, or the EXACT driver calib cache ──
samples = []
_calib_pt = os.environ.get("ML8_CALIB_PT")
if _calib_pt:
    obj = torch.load(_calib_pt, map_location="cpu", weights_only=False)
    raw = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    for s in raw:
        t = s if torch.is_tensor(s) else torch.as_tensor(s)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        samples.append(t.to(DEVICE))
    print(f"[probe] loaded {len(samples)} cached calib samples from {_calib_pt}", flush=True)
else:
    text = ("The history of scientific discovery is a long and winding road, "
            "full of unexpected turns and surprising connections between fields. ") * 8
    samples = [tok(text, return_tensors="pt", truncation=True,
                   max_length=512).input_ids.to(DEVICE)]

print(f"[probe] forward over {len(samples)} sample(s), shapes={[tuple(s.shape) for s in samples[:3]]}...",
      flush=True)

def first_bad_in(ev):
    for rec in ev:
        label, in_nan, n, inf, amax, numel = rec
        if (n and n > 0) or (inf and inf > 0):
            return label
    return None

bad_sample = None
bad_events = None
with torch.no_grad():
    for si, ids in enumerate(samples):
        events.clear()
        model(input_ids=ids)
        fb = first_bad_in(events)
        print(f"[probe] sample {si} shape={tuple(ids.shape)}  first_NaN={fb}", flush=True)
        if fb is not None and bad_sample is None:
            bad_sample, bad_events = si, list(events)
            break

for h in handles:
    h.remove()

# ── full trace of the FIRST sample that went bad (or the last clean one) ──
trace = bad_events if bad_events is not None else events
print(f"\n=== forward trace (sample {bad_sample if bad_sample is not None else 'last'}, completion order) ===")
print(f"{'module':<52}{'in_nan':>8}{'out_nan':>9}{'out_inf':>9}{'absmax':>14}")
first_bad = None
for label, in_nan, n, inf, amax, numel in trace:
    bad = (n and n > 0) or (inf and inf > 0)
    mark = "  <<< FIRST NaN/Inf" if bad and first_bad is None else ""
    if bad and first_bad is None:
        first_bad = label
    print(f"{label:<52}{str(in_nan):>8}{str(n):>9}{str(inf):>9}{amax:>14.4g}{mark}")

print(f"\n[probe] OVERALL first NaN/Inf: sample={bad_sample} module={first_bad}", flush=True)
