#!/usr/bin/env python3
"""Generate the ml8 gauntlet manifest (the list of calibration cells the MI300X
dispatcher runs). Tier-1 = the group_size grid on {27B dense, 35B MoE}.

Each cell carries its computed average bpv so the dispatcher / reviewer can see at
a glance that it honors the 4.25-bpv budget (the "win both axes" constraint). bpv
is reported, never assumed — change a knob, the number moves.

    python3 gen_gauntlet_manifest.py > gauntlet_tier1.json

The dispatcher (run_gauntlet.py) consumes the JSON. PPL is NOT run here — ml8
inference is gfx1201-only, so MI300X produces blobs + Y_SNR; PPL happens later on
the R9700. Y_SNR per cell (in each run's manifest.json) is the tier-1 ranking proxy.
"""
import json
import sys

# ── Models (paths are relative to --models-root on the instance) ──────────────
# arch must match what convert_hf_to_gguf wrote into the GGUF — verify on the box
# with `gguf-dump <file> | grep general.architecture` before trusting a run.
MODELS = {
    "qwen36-27b": {
        "strategy": "dense",
        "arch": "qwen35",                       # CONFIRM on box (dense Qwen3.6-27B)
        "gguf": "Qwen3.6-27B-bf16.gguf",
        "role": "dense scale control",
    },
    "qwen36-35b-a3b": {
        "strategy": "moe",
        "arch": "qwen35moe",                    # CONFIRM on box
        "gguf": "Qwen3.6-35B-A3B-bf16.gguf",
        "role": "MoE target",
    },
}

# ── The fixed baseline recipe every tier-1 cell shares (only group_size varies) ─
BASE = {
    "rotation": "kronecker",
    "snap_centroids": "e4m3",
    "fit_loss": "mse",
    "n_centroids": 16,
    "n_samples": 32,
    "seq_len": 1024,
    "act_order": True,
    "heavy_rounds": 4,
    "heavy_steps": 60,
    "heavy_dtype": "bf16",
}

# ── The group_size grid: (gate/up gs, down gs) ───────────────────────────────
GS_GRID = [
    (128, 16), (128, 32), (128, 64),
    (64, 16),  (64, 32),  (64, 64),
]

SCALE_BITS = 16  # fp16 per-group scale (the current ml8 convention)


def matrix_bpv(group_size: int, n_centroids: int = 16) -> float:
    """Stored index field width (4-bit for ≤16 centroids, 5-bit for 17–32 — fixed
    fields, no fractional packing) + amortized per-group scale bits. Centroid LUT
    storage is negligible (amortized over rows×group_size)."""
    field_bits = 4 if n_centroids <= 16 else 5
    return field_bits + SCALE_BITS / group_size


def avg_bpv(gu_gs: int, down_gs: int, n_centroids: int = 16) -> float:
    """Average over {gate, up, down} (equal param counts per matrix)."""
    return round((2 * matrix_bpv(gu_gs, n_centroids) + matrix_bpv(down_gs, n_centroids)) / 3, 3)


def main() -> int:
    jobs = []
    for model_key, m in MODELS.items():
        for gu_gs, down_gs in GS_GRID:
            jobs.append({
                "id": f"{model_key}_gu{gu_gs}_d{down_gs}",
                "tier": 1,
                "model_key": model_key,
                "strategy": m["strategy"],
                "arch": m["arch"],
                "gguf": m["gguf"],
                "role": m["role"],
                "group_size": gu_gs,          # gate/up
                "group_size_down": down_gs,    # down override
                "avg_bpv": avg_bpv(gu_gs, down_gs),
                **BASE,
            })
    manifest = {
        "name": "ml8-gauntlet-tier1-groupsize",
        "note": "group_size main-effect grid; PPL on R9700, Y_SNR ranks tiers here",
        "bpv_target": 4.25,
        "jobs": jobs,
    }
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
