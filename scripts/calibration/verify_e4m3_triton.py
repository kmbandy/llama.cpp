"""Compare a fp32->e4m3 cast (Triton, or any callable) against the ml8_e4m3_sim
oracle, bit-exactly via decoded values. Used to verify Triton's OCP e4m3 codes
before/after #10458 on gfx1201."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from ml8_e4m3_sim import e4m3_roundtrip


def sweep_inputs() -> torch.Tensor:
    # Dense small-magnitude sweep (subnormal + normal) plus saturation probes.
    lin = torch.linspace(-512.0, 512.0, steps=200001)
    sub = torch.linspace(-(2.0 ** -6), 2.0 ** -6, steps=50001)
    sat = torch.tensor([-1e4, -448.0, -447.9, 447.9, 448.0, 1e4])
    return torch.cat([lin, sub, sat]).contiguous()


def compare_codes(x: torch.Tensor, cast_fn) -> dict:
    ref = e4m3_roundtrip(x.float())          # oracle dequantized values
    got = cast_fn(x.float()).float()
    # Compare decoded values; NaN slots compare equal-as-NaN.
    both_nan = torch.isnan(ref) & torch.isnan(got)
    mism = (~both_nan) & (ref != got)
    idx = torch.nonzero(mism).flatten()[:20].tolist()
    return dict(n_total=int(x.numel()), n_mismatch=int(mism.sum()),
                sample_mismatch_inputs=[float(x[i]) for i in idx])


def _triton_cast(x: torch.Tensor) -> torch.Tensor:
    # Reference Triton OCP e4m3 cast for the on-device check. The exact tl dtype
    # name (float8e4nv = OCP e4m3) is confirmed against the installed Triton.
    import triton, triton.language as tl
    x = x.to("cuda")
    out = torch.empty_like(x)

    @triton.jit
    def _k(xp, op, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        v = tl.load(xp + off, mask=m)
        tl.store(op + off, v.to(tl.float8e4nv).to(tl.float32), mask=m)

    n = x.numel(); BLOCK = 1024
    _k[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/phase0_e4m3.json"))
    args = ap.parse_args()
    x = sweep_inputs()
    res = compare_codes(x, cast_fn=_triton_cast)
    res.update(label=args.label, triton_version=__import__("triton").__version__)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
