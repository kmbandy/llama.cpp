"""Act-replay KL trainer — student dequant-STE wrapper + faithful acts (Task 5).

The student is the ORIGINAL model with selected ml8 linears wrapped so that:
  * The linear's effective weight is the differentiable ml8 dequant
        W[r, c] = snap_e4m3(centroids)[gidx[c], indices[r, c]] * scales[r, gidx[c]]
    where centroids/scales are fp32 LEAF params (the only trainables) and the
    e4m3 snap is straight-through (snap in forward, identity grad in backward).
  * When the target carries a Kronecker rotation, the forward runs the
    deployment-faithful W4A8 activation path before the matmul:
        x_eff = quantize_act_per_row(x @ Q)   (Q = rotation, orthogonal)
    NO inverse rotation: the weight is rehydrated from GGUF in the ROTATED
    basis (W_rot), and the deployed kernel computes y = e4m3(x@Q) @ W_rot.T.

Indices stay FROZEN (argmin is non-differentiable). The original nn.Linear.weight
is left untouched — we monkey-patch module.forward so nothing about the host
module's parameter set changes; only the AttachedTarget's centroids/scales train.

Reused: snap_to_e4m3 (centroid_quantizer), KroneckerRotation (kronecker_rotation),
quantize_act_per_row (ml8_e4m3_sim). Dequant formula matches gguf_state.dequant_ml8_state
and the QK_ML8=64 grouping, but the group width is derived from the state shapes
so stub states with arbitrary G work too.
"""
from __future__ import annotations

import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F

from centroid_quantizer import snap_to_e4m3
from fp8_qat import Ml8Fp8Fn
from kronecker_rotation import KroneckerRotation
from ml8_e4m3_sim import quantize_act_per_row


def _gidx_for(K: int, n_groups: int, device=None) -> torch.Tensor:
    """Column -> group index [K]. Group width = K // n_groups (== QK_ML8 for
    real ml8 tensors; derived here so stub states with any G also work)."""
    if K % n_groups != 0:
        raise ValueError(f"K={K} not divisible by n_groups={n_groups}")
    width = K // n_groups
    return torch.arange(K, device=device) // width


class AttachedTarget(nn.Module):
    """Trainable ml8 codebook attached to one linear.

    Leaf params (trainable): centroids [G, 16], scales [N, G].
    Buffers (frozen):         indices [N, K] uint8, gidx [K] long.
    weight() returns the STE dequant — snapped e4m3 centroids in the forward,
    identity gradient into the raw fp32 centroids in the backward.

    Indices are stored uint8 (values 0..15; 8x smaller than int64 over all 136
    ml8 tensors) and promoted to long transiently inside weight() for gather().
    """

    def __init__(self, target: dict):
        super().__init__()
        indices = target["indices"].to(torch.uint8)
        centroids = target["centroids"].to(torch.float32).clone()
        scales = target["scales"].to(torch.float32).clone()

        N, K = indices.shape
        G = centroids.shape[0]
        if scales.shape != (N, G):
            raise ValueError(
                f"scales shape {tuple(scales.shape)} != (N={N}, G={G})")

        self.centroids = nn.Parameter(centroids)   # [G, 16] fp32, trainable
        self.scales = nn.Parameter(scales)         # [N, G]  fp32, trainable
        self.register_buffer("indices", indices)   # [N, K] long, frozen
        self.register_buffer("gidx", _gidx_for(K, G, indices.device))  # [K] long

        # Frozen anchor: initial bf16 ml8 dequant for the mse E-step index-reassign.
        self.register_buffer("W_orig", self.weight().detach().clone())  # [N, K]

        # Optional deployment-faithful rotation (W4A8 activation path).
        rot = target.get("rotation")
        if rot is not None:
            self.rotation = KroneckerRotation(
                h_a=rot["h_a"].to(torch.float32), b_dim=int(rot["b_dim"]))
        else:
            self.rotation = None

        # Optional input-column (V-head) reorder, carried as the hf->tiled (=GGUF)
        # ACTIVATION permutation. The ml8 weight/scales/centroids stay in pristine
        # GGUF order; apply_acts reorders the incoming HF-order activation into GGUF
        # order BEFORE the rotation. This is exact (per-row act-quant commutes with a
        # column permutation) and never conjugates the Kronecker rotation — unlike
        # permuting the weight columns, which would break the Kronecker structure the
        # deployed kernel requires. See _attach_one for where col_perm = argsort(index).
        col_perm = target.get("col_perm")
        if col_perm is not None:
            self.register_buffer(
                "col_perm", torch.as_tensor(col_perm, dtype=torch.long))
        else:
            self.col_perm = None

        # Transient Hessian collection state — plain Python attrs, NOT registered
        # buffers (must not appear in state_dict or pollute checkpoints).
        self._collecting_h = False
        self._h_acc = None        # [K, K] fp32 accumulator, None until first batch
        self._h_tokens = 0        # total token count accumulated so far

    def start_hessian_collection(self) -> None:
        """Enable per-target rotated-space Hessian accumulation.

        After calling this, every subsequent apply_acts call accumulates
        xrot^T xrot into self._h_acc (where xrot = x @ Q is the ROTATED,
        pre-quant activation). Call finalize_hessian() to retrieve the result.
        """
        self._collecting_h = True
        self._h_acc = None
        self._h_tokens = 0

    def finalize_hessian(self) -> torch.Tensor:
        """Stop accumulation and return H = (1/N) sum Xrot^T Xrot as fp32 [K, K].

        Raises RuntimeError if no data was accumulated (either the target has
        no rotation — apply_acts is a no-op — or start_hessian_collection was
        never called before any apply_acts invocation).
        """
        self._collecting_h = False
        if self._h_acc is None:
            raise RuntimeError(
                "collect_target_hessians: no rotated activations were accumulated "
                "for this target. This target either has no rotation (apply_acts is "
                "a no-op and never accumulates), or start_hessian_collection() was "
                "not called before the forward passes. Hessian collection requires a "
                "rotated target whose apply_acts ran at least once.")
        H = self._h_acc / max(1, self._h_tokens)
        self._h_acc = None
        self._h_tokens = 0
        return H

    def weight(self) -> torch.Tensor:
        """Differentiable ml8 dequant [N, K] with e4m3 straight-through snap."""
        c = self.centroids
        cent_ste = c + (snap_to_e4m3(c) - c).detach()   # snap fwd, identity grad
        # cent_ste[gidx, indices]: gather centroid per (row, col) by its group LUT.
        cent_per_col = cent_ste[self.gidx]              # [K, 16]
        idx_long = self.indices.long()                  # uint8 -> long for gather
        gathered = cent_per_col.unsqueeze(0).expand(
            self.indices.shape[0], -1, -1).gather(
            2, idx_long.unsqueeze(-1)).squeeze(-1)       # [N, K]
        return gathered * self.scales[:, self.gidx]     # × per-col scale

    def apply_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Faithful W4A8 activation transform x_eff = e4m3(x @ Q) — NO inverse.

        The AttachedTarget weight is rehydrated from GGUF in the ROTATED basis
        (the deployed kernel layout W_rot), so the deployed math is
            y = e4m3(x @ Q) @ W_rot.T
        with no inverse rotation. We therefore leave the quantized activation in
        the rotated basis and matmul directly against W_rot.

        (Calibration's FaithfulActHook applied the inverse Q.T because it had
        OVERRIDDEN the weight with the UNROTATED W_unrot = W_rot @ Q.T, making
        e4m3(x@Q) @ Q.T @ W_unrot.T == e4m3(x@Q) @ W_rot.T. With rotated weights
        rehydrated from GGUF, the inverse must go — keeping it leaves W
        un-derotated and produces garbage, e.g. holdout KL 12.15 nats.)

        The activation e4m3 quant is applied through a STRAIGHT-THROUGH estimator
        (snap in the forward value, identity gradient in the backward), mirroring
        the weight-quant STE in weight(). quantize_act_per_row is @no_grad and
        non-differentiable; without the STE the activation path detaches x_eff
        from x, so in a stack the gradient to an upstream layer's centroids/scales
        is severed (a single-linear test cannot see this — there is no trainable
        param upstream of the detach). The forward VALUE is unchanged: x_rot +
        (q - x_rot).detach() == q exactly; only the backward gains the identity
        path through the (piecewise-constant) act quant, back through the linear
        rotation to x.

        When an input-column reorder is attached (col_perm), the incoming HF-order
        activation is permuted into GGUF order FIRST, so it lines up with the
        GGUF-order weight columns; this happens whether or not a rotation follows.

        Returns x unchanged when neither a col_perm nor a rotation is attached."""
        if self.col_perm is not None:
            x = x.index_select(-1, self.col_perm)
        if self.rotation is None:
            return x
        orig_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1]).float()
        x_rot = self.rotation.forward(flat)
        # Accumulate rotated-space Hessian when collection is active.
        # Uses .detach() so this never perturbs gradients; runs only when
        # _collecting_h is True (a cheap attr check, off the training hot path).
        if self._collecting_h:
            xr = x_rot.detach().float()                  # [tokens, K] ROTATED, pre-quant
            xtx = xr.t() @ xr                            # [K, K]
            self._h_acc = xtx if self._h_acc is None else self._h_acc + xtx
            self._h_tokens += xr.shape[0]
        # STE: forward value = quantized acts; backward = identity into x_rot.
        a_q = x_rot + (quantize_act_per_row(x_rot) - x_rot).detach()
        x_eff = a_q.reshape(x.shape).to(orig_dtype)
        return x_eff


def attach_to_linear(lin: nn.Linear, target: dict,
                     faithful_acts: bool | None = None,
                     fp8: bool = False) -> AttachedTarget:
    """Wrap `lin` so its forward uses the ml8 dequant-STE weight (and, when a
    rotation is present and faithful_acts is on, the W4A8 activation path).

    The original nn.Linear.weight is NOT modified or trained. The AttachedTarget
    is stashed on the module as `_act_replay_target` so the optimizer can find its
    centroids/scales; module.forward is monkey-patched (the plan's approach).

    faithful_acts: None (default) -> auto: on iff the target carries a rotation.
                   True/False     -> explicit override.
    """
    at = AttachedTarget(target)

    if at.rotation is not None:
        rot = target["rotation"]
        a_dim = int(rot["a_dim"])
        b_dim = int(rot["b_dim"])
        if a_dim * b_dim != lin.in_features:
            raise ValueError(
                f"rotation dim mismatch: a_dim={a_dim} * b_dim={b_dim} = "
                f"{a_dim * b_dim} != lin.in_features={lin.in_features}")

    if faithful_acts is None:
        use_acts = at.rotation is not None or at.col_perm is not None
    else:
        use_acts = bool(faithful_acts)
        if use_acts and at.rotation is None:
            raise ValueError(
                "faithful_acts=True but target has no rotation to apply")

    # Move codebook params/buffers to the host linear's device/dtype context.
    at = at.to(lin.weight.device)

    lin._act_replay_target = at  # type: ignore[attr-defined]
    bias = lin.bias

    def _forward(x):
        x_eff = at.apply_acts(x) if use_acts else x
        if fp8:
            y = Ml8Fp8Fn.apply(x_eff, at.centroids, at.scales, at.indices, at.gidx)
            if bias is not None:
                y = y + bias.to(y.dtype)
            return y
        # The codebook params are fp32 leaves but the host model may run in bf16
        # (or fp16). Cast the dequant weight (and bias) to the activation dtype so
        # F.linear doesn't crash on a dtype mismatch. The cast happens AFTER the
        # STE dequant, so gradients still flow back to the fp32 centroids/scales.
        w = at.weight().to(x_eff.dtype)
        b = bias.to(x_eff.dtype) if bias is not None else None
        return F.linear(x_eff, w, b)

    lin.forward = _forward  # type: ignore[assignment]
    return at


def select_targets(names, train, skip=None):
    """Pick which GGUF tensor names to train.

    train: "ml8" keyword (all names) OR comma-separated fnmatch globs; a name is
           kept if it matches ANY train pattern.
    skip:  None or comma-separated fnmatch globs; a name is dropped if it matches
           ANY skip pattern. Skip wins over train.
    Returns the kept names in input order.
    """
    def _patterns(spec):
        return [p.strip() for p in spec.split(",") if p.strip()]

    train_all = (train is not None and train.strip() == "ml8")
    train_pats = [] if train_all else _patterns(train or "")
    skip_pats = _patterns(skip) if skip else []

    out = []
    for name in names:
        keep = train_all or any(fnmatch.fnmatch(name, p) for p in train_pats)
        if not keep:
            continue
        if any(fnmatch.fnmatch(name, p) for p in skip_pats):
            continue
        out.append(name)
    return out


__all__ = ["AttachedTarget", "attach_to_linear", "select_targets"]
