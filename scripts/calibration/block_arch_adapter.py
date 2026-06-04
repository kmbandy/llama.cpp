"""Per-arch seam for block-sequential GPTQ. A default HF adapter for standard
decoder models (model.model.layers, block.forward), plus per-arch overrides ONLY
where the run_block equivalence gate demands one. The driver supplies is_ml8(name)
(tier knowledge); the adapter only declares dependency GROUPING + runs blocks."""
from dataclasses import dataclass


@dataclass
class SubGroup:
    """One intra-block dependency group: quantize these together, then re-forward."""
    names: list   # dotted leaf names within the block, e.g. ["mlp.gate_proj", ...]


class DefaultBlockAdapter:
    def iter_blocks(self, model):
        return list(model.model.layers)

    def run_block(self, block, args, kwargs):
        out = block(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        return hidden, kwargs   # default: kwargs unchanged across blocks

    def ml8_targets(self, block, block_idx, is_ml8):
        """Default: one sub-group per ML8 Linear (no intra-block ordering knowledge)."""
        import torch
        groups = []
        for n, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear) and is_ml8(n):
                groups.append(SubGroup(names=[n]))
        return groups


class Qwen35BlockAdapter(DefaultBlockAdapter):
    # Dependency sub-groups, leaf names grounded by the Step-1 probe.
    # Probe output (Qwen3.5-0.8B-hf, all delta-net blocks):
    #   LINEARS ['linear_attn.out_proj', 'linear_attn.in_proj_qkv', 'linear_attn.in_proj_z',
    #            'linear_attn.in_proj_b', 'linear_attn.in_proj_a',
    #            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    # FP8-tier (not ML8 targets, excluded from groups):
    #   linear_attn.out_proj, linear_attn.in_proj_a, linear_attn.in_proj_b, mlp.down_proj
    # ML8-tier targets (confirmed by probe):
    #   linear_attn.in_proj_qkv, linear_attn.in_proj_z  -- read block input together
    #   mlp.gate_proj, mlp.up_proj                        -- read FFN input together
    _SSM_GROUPS = [
        ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
        ["mlp.gate_proj", "mlp.up_proj"],
    ]
    # Full-attention block kind (best-effort; 0.8B is all delta-net, validated
    # later by the run_block equivalence gate):
    _ATTN_GROUPS = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
    ]

    def ml8_targets(self, block, block_idx, is_ml8):
        present = {n for n, _ in block.named_modules()}
        is_ssm = any(n.startswith("linear_attn") for n in present)
        raw = self._SSM_GROUPS if is_ssm else self._ATTN_GROUPS
        groups = []
        for grp in raw:
            kept = [n for n in grp if n in present and is_ml8(n)]
            if kept:
                groups.append(SubGroup(names=kept))
        return groups


def get_adapter(arch):
    return Qwen35BlockAdapter() if str(arch).startswith("qwen35") else DefaultBlockAdapter()
