"""KroneckerRotation — Q = H_a ⊗ H_b for ml8 weight quantization.

Production-quality factored rotation for QuaRot-style weight quantization.
H_b is a Sylvester-construction Hadamard matrix (power-of-2 dim, deterministic).
H_a is a small random orthogonal matrix for the non-power-of-2 factor.

For input dim d = a * b, storing the rotation as the (a, b) factored form costs
~a^2 floats instead of d^2; applying it as a Kronecker product costs O(d*(a+log b))
instead of O(d^2). Both savings matter at production inference time.

Schema (stored in ml8 blob's "rotation" field):
    {
        "kind": "kronecker_orth_sylvester",
        "h_a": torch.Tensor(a, a),
        "a_dim": int a,
        "b_dim": int b,
        "in_features": int d = a*b,
        "seed": int,
    }

The forward(x) applies Q to the last dim of x (rotates along input axis of a Linear).
The inverse(x) applies Q^T (equal to Q^-1 since Q is orthogonal).
"""

import math
from typing import Optional

import torch


def fwht_raw(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized fast Walsh-Hadamard transform along the last dim (size = power
    of 2). Computes ``x @ H_raw`` where H_raw is the +/-1 Sylvester matrix, in
    O(n log n) butterfly stages instead of the O(n^2) dense matmul.

    ``fwht_raw(x) / sqrt(n) == x @ sylvester(n)`` to float precision (sylvester is
    normalized by 1/sqrt(n)). This mirrors the deployed ml8 kernel's fused FWHT
    H_b leg (ml8.cu / turbo_fp8_hadamard.cuh). Autograd-friendly (cat/slice/add)."""
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"fwht_raw: last dim must be a power of 2, got {n}")
    orig = x.shape
    h = 1
    y = x
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2 * h)
        a = y[..., :h]
        b = y[..., h:2 * h]
        y = torch.cat([a + b, a - b], dim=-1).reshape(orig)
        h *= 2
    return y


class KroneckerRotation:
    """Factored orthogonal rotation Q = H_a ⊗ H_b applied to last dim of inputs.

    For input dim d = a * b, reshape x to (..., a, b) and apply
        Y = H_a @ X @ H_b.T
    which equals (H_a ⊗ H_b) · vec(X) under row-major flatten.

    Inverse uses Q^T = H_a^T ⊗ H_b^T, so:
        X = H_a.T @ Y @ H_b

    Storage: only h_a (a×a) needs to be persisted — h_b is the deterministic
    Sylvester Hadamard of size b_dim, regenerated on load.
    """

    def __init__(self, h_a: torch.Tensor, b_dim: int):
        if h_a.dim() != 2 or h_a.shape[0] != h_a.shape[1]:
            raise ValueError(f"h_a must be square 2D, got shape {tuple(h_a.shape)}")
        if b_dim < 1 or (b_dim & (b_dim - 1)) != 0:
            raise ValueError(f"b_dim must be a positive power of 2, got {b_dim}")
        self.h_a = h_a
        self.a_dim = h_a.shape[0]
        self.b_dim = b_dim
        self.d = self.a_dim * b_dim
        self.h_b = sylvester(b_dim).to(dtype=h_a.dtype)
        self._inv_sqrt_b = 1.0 / math.sqrt(b_dim)   # FWHT normalization (== sylvester)

    def _factors_on(self, x: torch.Tensor):
        h_a = self.h_a.to(device=x.device, dtype=x.dtype)
        h_b = self.h_b.to(device=x.device, dtype=x.dtype)
        return h_a, h_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Right-multiply last dim of x by Q = H_a ⊗ H_b (PyTorch row-vector convention).

        Computes x @ Q. Derivation: with X = x.reshape(..., a, b),
            (x @ Q)[k*b + l] = sum_{i,j} X[i,j] H_a[i,k] H_b[j,l]
                              = (H_a.T @ X @ H_b)[k, l].
        """
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim {x.shape[-1]} != d={self.d}")
        h_a, _ = self._factors_on(x)
        X = x.reshape(*x.shape[:-1], self.a_dim, self.b_dim)
        # H_b leg via fast Walsh-Hadamard (O(b log b), == X @ sylvester(b) to float
        # precision; mirrors the deployed ml8 fused FWHT prologue). H_a leg stays a
        # small a×a matmul. Same fp32 math as the dense `h_a.T @ X @ h_b`.
        Xw = fwht_raw(X) * self._inv_sqrt_b
        if self.a_dim == 1:
            # a_dim=1 degenerates the "a×a matmul" to a scalar multiply by
            # h_a[0,0] (h_a is 1x1). Special-cased to avoid an actual
            # torch.matmul/bmm call here: on this GPU (ROCm, RX 9070 XT),
            # a batched matmul with contraction dim 1 reproducibly crashes
            # the composable-kernel `_bmm_outer_product_kernel` with an
            # HSA_STATUS_ERROR_MEMORY_FAULT (confirmed 2026-09-22, reproduces
            # on plain random 128x512 input with a_dim=1/b_dim=512 -- the
            # exact shape of blk.N.indexer.attn_k.weight [512,128], which is
            # in the ml8_4 rotate_kronecker role list -- while the identical
            # op on CPU is fine, so this is a device-specific kernel bug, not
            # a numerical issue). Elementwise multiply computes the exact
            # same math (h_a.T @ X == h_a[0,0] * X when a_dim==1) without
            # ever invoking the buggy matmul kernel.
            Y = h_a.reshape(()).to(Xw.dtype) * Xw
        else:
            Y = h_a.T @ Xw
        return Y.reshape(x.shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Right-multiply last dim of x by Q.T (= Q^{-1} since Q is orthogonal).

        Computes x @ Q.T = H_a @ X @ H_b.T.
        """
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim {x.shape[-1]} != d={self.d}")
        h_a, _ = self._factors_on(x)
        X = x.reshape(*x.shape[:-1], self.a_dim, self.b_dim)
        # Sylvester H_b is symmetric (H_b.T == H_b), so the inverse b-leg is the
        # same FWHT/sqrt(b); the a-leg uses h_a (not transposed).
        Xw = fwht_raw(X) * self._inv_sqrt_b
        if self.a_dim == 1:
            # See forward()'s a_dim==1 special case: same GPU matmul-kernel
            # crash avoidance, h_a @ X == h_a[0,0] * X here (h_a is 1x1, and
            # h_a == h_a.T trivially at that size).
            Y = h_a.reshape(()).to(Xw.dtype) * Xw
        else:
            Y = h_a @ Xw
        return Y.reshape(x.shape)

    def to_dict(self) -> dict:
        """Serialize for storage in ml8 blob's 'rotation' field.

        Only h_a needs to be stored — h_b is the deterministic Sylvester of size b_dim,
        regenerated by from_dict. h_a kept on CPU so the blob is device-agnostic.
        """
        return {
            "kind": "kronecker_orth_sylvester",
            "h_a": self.h_a.detach().cpu(),
            "a_dim": self.a_dim,
            "b_dim": self.b_dim,
            "in_features": self.d,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "KroneckerRotation":
        if blob.get("kind") != "kronecker_orth_sylvester":
            raise ValueError(f"unsupported rotation kind: {blob.get('kind')!r}")
        return cls(h_a=blob["h_a"], b_dim=int(blob["b_dim"]))


# rotation_meta kind_id — matches the [a_dim, b_dim, in_features, kind_id] I32[4]
# sidecar contract documented in ml8_to_gguf.py::_rotation_meta_bytes and
# ggml/src/ggml-cuda/aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md.
# 1 = kronecker_orth_sylvester (existing). 2 = block_hadamard — PLACEHOLDER: the
# C++ side (rotation op + registry agent) had not landed a kind_id constant for
# block_hadamard as of this writing; grep ggml/ and src/ for the real constant
# once it lands and update this value + the design doc table to match.
KRONECKER_ORTH_SYLVESTER_KIND_ID = 1
BLOCK_HADAMARD_KIND_ID = 2


class BlockHadamardRotation:
    """Q = I_a ⊗ H_b applied to the last dim: an independent normalized Hadamard
    on each contiguous b-sized block along K, no cross-block mixing (no h_a leg).

    Unlike KroneckerRotation, this has nothing to persist besides (a_dim, b_dim,
    in_features) — there is no h_a matrix, so only a rotation_meta sidecar is
    written (no rotation_h_a). By construction, forward()/inverse() here are
    identical to KroneckerRotation(h_a=eye(a_dim), b_dim=b_dim).forward()/
    inverse() — same H_b (Sylvester) normalization, just without the (trivial)
    identity a-leg matmul. See test_block_hadamard_matches_kronecker_identity.
    """

    def __init__(self, in_features: int, b_dim: int = 128):
        if b_dim < 1 or (b_dim & (b_dim - 1)) != 0:
            raise ValueError(f"b_dim must be a positive power of 2, got {b_dim}")
        if in_features % b_dim != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by b_dim={b_dim}"
            )
        self.b_dim = b_dim
        self.a_dim = in_features // b_dim
        self.d = in_features
        self._inv_sqrt_b = 1.0 / math.sqrt(b_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Right-multiply last dim of x by Q = I_a ⊗ H_b (row-vector convention).

        Equivalent to applying the normalized Hadamard independently to each
        contiguous b_dim-sized block of the last axis.
        """
        if x.shape[-1] != self.d:
            raise ValueError(f"last dim {x.shape[-1]} != d={self.d}")
        X = x.reshape(*x.shape[:-1], self.a_dim, self.b_dim)
        Y = fwht_raw(X) * self._inv_sqrt_b
        return Y.reshape(x.shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Q.T == Q since H_b (Sylvester) is symmetric and I_a is symmetric."""
        return self.forward(x)

    def to_dict(self) -> dict:
        return {
            "kind": "block_hadamard",
            "a_dim": self.a_dim,
            "b_dim": self.b_dim,
            "in_features": self.d,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "BlockHadamardRotation":
        if blob.get("kind") != "block_hadamard":
            raise ValueError(f"unsupported rotation kind: {blob.get('kind')!r}")
        return cls(in_features=int(blob["in_features"]), b_dim=int(blob["b_dim"]))


def factor_for_dim(d: int, max_b: int = 1024) -> tuple:
    """Pick (a, b) such that a*b == d, b is the largest power of 2 ≤ max_b that divides d.

    max_b defaults to 1024 to match the existing turbo_fp8_hadamard.cuh FWHT kernel's
    upper bound — keeps the factorization compatible with a future production runtime
    that reuses the KV-side kernel for the H_b leg.

    Examples (max_b=1024):
        2560 → (5, 512)
        9216 → (9, 1024)
        4096 → (4, 1024)   # capped at max_b; pure-Sylvester (1, 4096) would exceed kernel
        256  → (1, 256)
        7    → (7, 1)      # no power-of-2 factor, degenerates to pure random orthogonal
    """
    if d < 1:
        raise ValueError(f"d must be positive, got {d}")
    if max_b < 1 or (max_b & (max_b - 1)) != 0:
        raise ValueError(f"max_b must be a positive power of 2, got {max_b}")
    b = 1
    while (b * 2) <= max_b and d % (b * 2) == 0:
        b *= 2
    return d // b, b


def rotate_hessian(H: torch.Tensor, rotation: "KroneckerRotation") -> torch.Tensor:
    """Compute Q.T @ H @ Q using the rotation's Kronecker structure.

    For our row-vector convention:
      rotation.forward(M) = M @ Q   (right-multiply each row by Q)
      M.T @ Q             = rotation.forward(M.T)
      Q.T @ M             = (M.T @ Q).T = rotation.forward(M.T).T

    So Q.T @ H @ Q = rotation.forward( rotation.forward(H).T ).T

    H is (d, d); each rotation.forward call costs O(d * (a + log b)) instead of the
    O(d^3) of a dense Q matmul. For d=2560, a=5, b=512: ~36K ops × d rows ≈ 10^8 vs
    ~10^10 for dense. ~100× speedup, matters when Hessians are big.
    """
    return rotation.forward(rotation.forward(H).T).T


def random_orthogonal(a: int, seed: int) -> torch.Tensor:
    """Random orthogonal matrix of size (a, a), uniformly distributed on O(a).

    Implementation: QR decomposition of a Gaussian random matrix. The sign-fix
    on diag(R) ensures uniform distribution per Mezzadri 2007 (`How to generate
    random matrices from the classical compact groups`); without it the
    distribution is biased.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    M = torch.randn(a, a, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    # Mezzadri sign fix: multiply each column of Q by sign of corresponding diag(R)
    d = torch.sign(torch.diagonal(R))
    return Q * d.unsqueeze(0)


def sylvester(n: int) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix of size (n, n), normalized so H @ H.T = I.

    n must be a power of 2. H_1 = [[1]]; H_{2k} = [[H_k, H_k], [H_k, -H_k]] / sqrt(2)
    keeps orthonormality at each doubling.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"sylvester(n): n must be a positive power of 2, got {n}")
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0) / math.sqrt(2.0)
    return H
