#!/usr/bin/env python3
"""Pre-flight ml8 coverage check — no GPU, ~1 second.

Reads an HF model's config + safetensors *headers* (not weights) and resolves every
parameter through the SAME role classifier the calibrator uses (role_targets, backed by
llama.cpp's authoritative TensorNameMap). Prints a tier coverage table and EXITS NON-ZERO
if any 2D (Linear) weight in the attention / SSM / MLP stack would be left NATIVE/bf16 —
i.e. the name map does not cover this checkpoint.

Run this BEFORE `calibrate_ml8_paged.py --dense-coverage full` to catch name drift on a
new model/variant without spending a minute of GPU time:

    python3 preflight_coverage.py --model /path/to/hf_dir [--arch qwen35]
"""
import argparse, json, struct, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from role_targets import classify_role, Tier, configure, _layer_idx, _MAIN_STACK_LINEAR_PARENTS

# HF architectures[] / model_type -> MODEL_ARCH name. Extend as new arches land, or
# pass --arch explicitly (matches the calibrator's --arch override).
_ARCH_FROM_HF = {
    "Qwen3_5ForConditionalGeneration": "qwen35",
    "Qwen3_5ForCausalLM":              "qwen35",
    "Qwen3_5MoeForConditionalGeneration": "qwen35moe",
    "Qwen3_5MoeForCausalLM":              "qwen35moe",
}


def _read_safetensors_header(path: Path) -> dict:
    """Parse just the JSON header (name -> {dtype, shape, ...}); no weight bytes."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model directory")
    ap.add_argument("--arch", default=None,
                    help="MODEL_ARCH name (e.g. qwen35); auto-detected from config if omitted")
    ap.add_argument("--require-mtp", action="store_true",
                    help="treat uncovered MTP/NextN draft-head GEMMs as a hard failure too")
    args = ap.parse_args()

    mdir = Path(args.model)
    cfg = json.loads((mdir / "config.json").read_text())
    tcfg = cfg.get("text_config", cfg)   # multimodal models nest the LM config here

    arch = args.arch
    if arch is None:
        for a in cfg.get("architectures", []):
            if a in _ARCH_FROM_HF:
                arch = _ARCH_FROM_HF[a]; break
    if arch is None:
        sys.exit(f"[preflight] could not auto-detect arch from {cfg.get('architectures')!r}; "
                 f"pass --arch (e.g. --arch qwen35)")

    n_blocks = int(tcfg["num_hidden_layers"])
    print(f"[preflight] model={mdir.name}  arch={arch}  n_blocks={n_blocks}")
    if tcfg.get("layer_types"):
        print(f"[preflight] layer_types: {dict(Counter(tcfg['layer_types']))}")
    configure(arch, n_blocks)

    shapes = {}
    for shard in sorted(mdir.glob("*.safetensors")):
        for name, meta in _read_safetensors_header(shard).items():
            shapes[name] = meta["shape"]
    if not shapes:
        sys.exit(f"[preflight] no .safetensors found under {mdir}")

    tier_counts, role_counts, trunk_unc, mtp_unc = Counter(), Counter(), [], []
    for name, shape in shapes.items():
        if not name.endswith(".weight"):
            continue
        mod = name[:-len(".weight")]
        _, role, tier = classify_role(mod)
        tier_counts[tier.value] += 1
        if tier is not Tier.NATIVE:
            role_counts[role] += 1
        # A 2D weight in the main LM attention/SSM/MLP stack left NATIVE = coverage gap.
        if (len(shape) == 2 and tier is Tier.NATIVE and "visual" not in name
                and _layer_idx(name) is not None
                and any(p in name for p in _MAIN_STACK_LINEAR_PARENTS)):
            # The MTP / NextN draft head is a separate block the calibrator does not
            # currently process (it loads num_hidden_layers trunk blocks). Report it,
            # but don't hard-fail the trunk check on it.
            (mtp_unc if ("mtp." in name or "nextn" in name) else trunk_unc).append((name, shape))

    print("\n[coverage] tier totals (over .weight params):")
    for t in ("ml8", "fp8", "native"):
        print(f"   {t:7s}: {tier_counts.get(t, 0)}")
    print("[coverage] covered roles:")
    for r, c in sorted(role_counts.items()):
        print(f"   {r:12s}: {c}")

    if mtp_unc:
        print(f"\n[preflight] note — {len(mtp_unc)} MTP/NextN draft-head GEMM(s): not "
              f"GPTQ-calibrated (the draft graph isn't in the calibration forward pass), "
              f"but ml8_to_gguf --mtp-fp8 (default on) casts them to scaled FP8 at convert "
              f"time. Pass --require-mtp here to treat them as a hard failure instead.")
        for n, sh in mtp_unc[:10]:
            print(f"     {n}  shape={sh}  -> ML8_FP8 (convert-time)")
        if getattr(args, "require_mtp", False):
            trunk_unc.extend(mtp_unc)

    if trunk_unc:
        print(f"\n[preflight] FAIL — {len(trunk_unc)} TRUNK 2D weight(s) UNCOVERED "
              f"(would ship as bf16); role map does not match this checkpoint:")
        for n, sh in trunk_unc[:20]:
            print(f"     {n}  shape={sh}")
        sys.exit(1)
    print("\n[preflight] OK — every main-trunk Linear weight is covered (ML8|FP8).")


if __name__ == "__main__":
    main()
