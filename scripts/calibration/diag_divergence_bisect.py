"""DIAGNOSTIC (CPU, no GPU): localize the act-replay training-step DIVERGENCE.

The real train() loop is known to DESCEND on a single-layer, rotation-free,
no-checkpoint stub (test_train_step_loss_down passes). The overnight 4B run
DIVERGED (holdout KL 0.221 -> 6.33, monotonic). This bisects by adding the
4B-only features to the known-good baseline ONE AT A TIME and reporting whether
KL still descends. The first feature that flips DESCEND -> DIVERGE is the culprit.

Setup is faithful to act-replay: teacher = the SAME stack with quantization
DISABLED (exact gather + no e4m3 acts). Because the Kronecker rotation is
orthogonal/exact, that equals the bf16 unrotated reference. The only gap the
student closes is the e4m3 quantization — correct behavior is descent, not a 30x
blow-up. Uses the REAL train(), kl_topk, attach_to_linear, LiveTeacher.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from act_replay import train
from act_replay_student import attach_to_linear
from gguf_state import dequant_ml8_state
from kronecker_rotation import KroneckerRotation, random_orthogonal
from teacher_source import LiveTeacher
from kl_loss import kl_topk

D, VOCAB, NHID = 8, 32, 8          # embed dim D=a*b (2*4), vocab, logits hidden
A_DIM, B_DIM = 2, 4


def mk_state(seed, rotation):
    g = torch.Generator().manual_seed(seed)
    cent = torch.randn(2, 16, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    s = {"indices": torch.randint(0, 16, (NHID, D), generator=g),
         "scales": torch.rand(NHID, 2, generator=g) + 0.1,
         "centroids": cent, "rotation": None}
    if rotation:
        s["rotation"] = {"h_a": random_orthogonal(A_DIM, seed=seed),
                         "a_dim": A_DIM, "b_dim": B_DIM, "in_features": D}
    return s


class Student(nn.Module):
    """embed -> [residual ml8 layers] -> head. Layers carry attached ml8 targets."""
    def __init__(self, n_layers, rotation, residual, ckpt, seed=0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.head = nn.Linear(NHID, VOCAB, bias=False)
        self.residual, self.ckpt = residual, ckpt
        self.lins = nn.ModuleList()
        self.ats = []
        for i in range(n_layers):
            st = mk_state(seed + i, rotation)
            self.states_for_teacher = getattr(self, "states_for_teacher", [])
            self.states_for_teacher.append(st)
            # perturb the student's centroids so step-0 KL is nonzero (a gap to close)
            pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
            if st["rotation"] is not None:
                pert["rotation"] = dict(st["rotation"])
            pert["centroids"] = pert["centroids"] + 0.15
            lin = nn.Linear(D, NHID, bias=False)
            at = attach_to_linear(lin, pert, faithful_acts=(st["rotation"] is not None))
            self.lins.append(lin); self.ats.append(at)
        self.embed.requires_grad_(False)
        self.head.requires_grad_(False)

    def _layer(self, lin, h):
        y = lin(h)                       # [B,T,NHID]; monkey-patched ml8 forward
        return h + y if self.residual else y

    def forward(self, ids):
        h = self.embed(ids)
        for lin in self.lins:
            if self.ckpt:
                h = torch.utils.checkpoint.checkpoint(
                    self._layer, lin, h, use_reentrant=False)
            else:
                h = self._layer(lin, h)
        return self.head(h)


class Teacher(nn.Module):
    """Same arch, quantization DISABLED: exact dequant weight + exact (x@Q) acts."""
    def __init__(self, student: Student):
        super().__init__()
        self.embed = student.embed
        self.head = student.head
        self.residual = student.residual
        self.W, self.rot = [], []
        for st in student.states_for_teacher:
            self.W.append(dequant_ml8_state(st))                # [NHID,D] exact
            self.rot.append(None if st["rotation"] is None
                            else KroneckerRotation(st["rotation"]["h_a"],
                                                   st["rotation"]["b_dim"]))

    def forward(self, ids):
        h = self.embed(ids)
        for W, rot in zip(self.W, self.rot):
            x = rot.forward(h) if rot is not None else h        # exact rotation, no e4m3
            y = F.linear(x, W)
            h = h + y if self.residual else y
        return self.head(h)


def run(name, *, n_layers, rotation, residual, ckpt, steps=40):
    torch.manual_seed(0)
    student = Student(n_layers, rotation, residual, ckpt)
    teacher = Teacher(student).eval()
    src = LiveTeacher(teacher, 8)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, VOCAB, (1, 16), generator=g) for _ in range(4)]
    train_idx, hold_idx = torch.arange(3), torch.tensor([3])

    def kl_now():
        tot = 0.0
        with torch.no_grad():
            for i, ids in enumerate(batches):
                idx, vals, tail = src.get(i, ids)
                lg = student(ids)
                tot += kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail).item()
        return tot / len(batches)

    kl0 = kl_now()
    params = [p for at in student.ats for p in (at.centroids, at.scales)]
    opt = torch.optim.Adam(params, lr=1e-2)
    train(student, src, batches, train_idx, hold_idx, opt,
          grad_accum=1, epochs=steps, eval_interval=0)
    kl1 = kl_now()
    verdict = "DESCEND" if kl1 < kl0 else f"DIVERGE (x{kl1/max(kl0,1e-9):.1f})"
    print(f"  {name:<46} KL {kl0:.4f} -> {kl1:.4f}   {verdict}")
    return kl1 < kl0


def run_windowed(name, *, n_layers, steps=40, train_seq_len=8):
    """Config matching main()'s windowed CachedTeacher wiring: train on WINDOWS,
    teacher cache content-addressed over (holdout-full + train-windows). Verifies
    the cached per-window target equals a fresh teacher forward on that window."""
    import tempfile
    from act_replay import split_batches_seq
    from teacher_source import make_teacher, CachedTeacher
    torch.manual_seed(0)
    student = Student(n_layers, rotation=True, residual=True, ckpt=True)
    teacher = Teacher(student).eval()
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, VOCAB, (1, 16), generator=g) for _ in range(4)]
    train_idx, hold_idx = torch.arange(3), torch.tensor([3])
    train_batches, train_idx_w = split_batches_seq(batches, train_idx, train_seq_len)

    with tempfile.TemporaryDirectory() as td:
        teacher_batches = ([batches[i] for i in hold_idx.tolist()]
                           + [train_batches[i] for i in train_idx_w.tolist()])
        src = make_teacher("cache", model_loader=lambda: teacher, K=8,
                           cache_dir=td, batches=teacher_batches, cache_key="stub")
        # CORRECTNESS: cached per-window target == fresh teacher forward on it.
        live = LiveTeacher(teacher, 8)
        max_mismatch = 0.0
        for i in train_idx_w.tolist():
            ids = train_batches[i]
            ci, cv, ct = src.get(i, ids)
            li, lv, lt = live.get(i, ids)
            max_mismatch = max(max_mismatch, float((cv - lv).abs().max()),
                               float((ci != li).sum()))

        def kl_now():
            tot = 0.0
            with torch.no_grad():
                for i, ids in enumerate(batches):
                    idx, vals, tail = live.get(i, ids)
                    lg = student(ids)
                    tot += kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail).item()
            return tot / len(batches)

        kl0 = kl_now()
        params = [p for at in student.ats for p in (at.centroids, at.scales)]
        opt = torch.optim.Adam(params, lr=1e-2)
        train(student, src, train_batches, train_idx_w, hold_idx, opt,
              grad_accum=1, epochs=steps, eval_interval=0, eval_batches=batches)
        kl1 = kl_now()
    verdict = "DESCEND" if kl1 < kl0 else f"DIVERGE (x{kl1/max(kl0,1e-9):.1f})"
    print(f"  {name:<46} KL {kl0:.4f} -> {kl1:.4f}   {verdict}  "
          f"[cache==live target mismatch={max_mismatch:.2e}]")
    return kl1 < kl0


if __name__ == "__main__":
    print("Bisection: add 4B-only features to the known-good descending baseline.\n")
    configs = [
        ("1) baseline: 1 layer, no rot, no resid, no ckpt",
         dict(n_layers=1, rotation=False, residual=False, ckpt=False)),
        ("2) + rotation + faithful-acts (1 layer)",
         dict(n_layers=1, rotation=True, residual=False, ckpt=False)),
        ("3) + stacking(3) + residual, rotation on",
         dict(n_layers=3, rotation=True, residual=True, ckpt=False)),
        ("4) + gradient checkpointing (use_reentrant=False)",
         dict(n_layers=3, rotation=True, residual=True, ckpt=True)),
    ]
    results = []
    for name, kw in configs:
        try:
            results.append((name, run(name, **kw)))
        except Exception as e:
            print(f"  {name:<46} ERROR: {type(e).__name__}: {e}")
            results.append((name, None))
    try:
        results.append(("5) + windowed CachedTeacher (split + content-addr)",
                        run_windowed("5) + windowed CachedTeacher (split + content-addr)",
                                     n_layers=3)))
    except Exception as e:
        print(f"  5) windowed CachedTeacher ERROR: {type(e).__name__}: {e}")
        results.append(("5", None))
    bad = [n for n, ok in results if ok is False]
    print("\nFirst feature that breaks descent:",
          bad[0] if bad else "NONE — divergence is not reproduced by these features")
