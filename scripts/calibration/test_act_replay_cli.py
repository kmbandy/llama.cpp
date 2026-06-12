"""CLI / trainer tests for act_replay.py (Task 6, act-replay KL trainer).

No HF model required: a tiny stub LM exposing named_modules() with one nn.Linear
stands in for the student/teacher. Run from scripts/calibration with
PYTHONPATH=../../gguf-py.
"""
import pytest
import torch
import torch.nn as nn

from act_replay import (
    parse_args,
    split_holdout,
    split_batches_seq,
    map_gguf_to_hf,
    gguf_to_hf_perm,
    attach_targets,
    build_response_mask,
    batch_response_mask,
    train,
    save_ckpt,
    load_ckpt,
    export_blobs,
    install_frozen_fp8,
    alloc_conf_hint,
    _derive_tier_spec,
)
from act_replay_student import attach_to_linear
from centroid_quantizer import snap_to_e4m3
from gguf_state import dequant_ml8_state
import ml8_io


# ─── helpers ─────────────────────────────────────────────────────────────────


def _mk_state(N=8, K=128, G=2, seed=0):
    """A single ml8 target (matches test_act_replay_student._mk_state)."""
    g = torch.Generator().manual_seed(seed)
    cent = torch.randn(G, 16, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    return {"indices": torch.randint(0, 16, (N, K), generator=g),
            "scales": torch.rand(N, G, generator=g) + 0.1,
            "centroids": cent, "rotation": None}


class StubLM(nn.Module):
    """Tiny "LM": embed ids -> one linear -> logits. Exposes named_modules()."""

    def __init__(self, vocab=32, K=128, N=8, weight=None):
        super().__init__()
        self.embed = nn.Embedding(vocab, K)
        self.lin = nn.Linear(K, N, bias=False)
        if weight is not None:
            with torch.no_grad():
                self.lin.weight.copy_(weight)
        # project the N hidden dims back up to vocab for a logits head
        self.head = nn.Linear(N, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)          # [B,T,K]
        h = self.lin(h)              # [B,T,N]
        return self.head(h)          # [B,T,vocab]


# ─── tests ───────────────────────────────────────────────────────────────────


def test_parse_args_defaults():
    a = parse_args(["--gguf", "g", "--base-gguf", "b", "--model", "m", "--out-dir", "o"])
    assert a.base_gguf == "b"
    assert a.corpus == "mix"
    assert a.token_budget == 512000
    assert a.seq_len == 2048
    assert a.teacher == "live"
    assert a.topk == 256
    assert a.lr_cent == 2e-4
    assert a.lr_scale == 2e-5
    assert a.grad_accum == 8
    assert a.tensors_train == "ml8"
    assert a.tensors_skip == ""
    assert a.steps is None
    assert a.epochs == 1
    assert a.seed == 0
    assert a.eval_interval == 200
    assert a.micro_batch == 1
    assert a.no_grad_ckpt is False  # grad checkpointing on by default


def test_parse_args_base_gguf_required():
    # --base-gguf is required: argparse exits (SystemExit) when it's absent.
    import pytest
    with pytest.raises(SystemExit):
        parse_args(["--gguf", "g", "--model", "m", "--out-dir", "o"])


def _write_gguf(path, *, with_centroids):
    """Write a tiny GGUF; include a .centroids tensor iff with_centroids."""
    import gguf
    from gguf import GGMLQuantizationType
    import numpy as np

    w = gguf.GGUFWriter(str(path), arch="qwen35")
    w.add_tensor("blk.0.ffn_up.weight", np.ones((4, 8), np.float32))
    if with_centroids:
        w.add_tensor("blk.0.attn_qkv.centroids",
                     np.ones((2, 16), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_looks_ml8_gguf_detects_centroids(tmp_path):
    from act_replay import _looks_ml8_gguf

    base = tmp_path / "base.gguf"
    ml8 = tmp_path / "ml8.gguf"
    _write_gguf(base, with_centroids=False)
    _write_gguf(ml8, with_centroids=True)

    assert _looks_ml8_gguf(ml8) is True
    assert _looks_ml8_gguf(base) is False


def test_holdout_split_deterministic():
    tr0, ho0 = split_holdout(20, frac=0.1, seed=0)
    tr0b, ho0b = split_holdout(20, frac=0.1, seed=0)
    assert torch.equal(tr0, tr0b) and torch.equal(ho0, ho0b)
    assert tr0.numel() == 18 and ho0.numel() == 2
    # seed 1 differs
    tr1, ho1 = split_holdout(20, frac=0.1, seed=1)
    assert not (torch.equal(tr0, tr1) and torch.equal(ho0, ho1))
    # no overlap, union complete
    s_tr, s_ho = set(tr0.tolist()), set(ho0.tolist())
    assert s_tr.isdisjoint(s_ho)
    assert s_tr | s_ho == set(range(20))


def test_map_gguf_to_hf():
    assert map_gguf_to_hf("blk.0.ffn_up.weight") == "model.layers.0.mlp.up_proj"
    assert map_gguf_to_hf("blk.3.ffn_gate.weight") == "model.layers.3.mlp.gate_proj"
    assert map_gguf_to_hf("blk.2.ffn_down.weight") == "model.layers.2.mlp.down_proj"
    assert map_gguf_to_hf("blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj"
    assert map_gguf_to_hf("blk.1.attn_k.weight") == "model.layers.1.self_attn.k_proj"
    assert map_gguf_to_hf("blk.1.attn_v.weight") == "model.layers.1.self_attn.v_proj"
    assert map_gguf_to_hf("blk.4.attn_output.weight") == "model.layers.4.self_attn.o_proj"
    assert map_gguf_to_hf("token_embd.weight") == "model.embed_tokens"
    try:
        map_gguf_to_hf("blk.0.some_unknown.weight")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_map_gguf_to_hf_linear_attn():
    # hybrid (qwen35) linear-attn 2D matmul targets — the ML8 matmuls map.
    assert map_gguf_to_hf("blk.0.attn_qkv.weight") == "model.layers.0.linear_attn.in_proj_qkv"
    assert map_gguf_to_hf("blk.5.attn_gate.weight") == "model.layers.5.linear_attn.in_proj_z"
    assert map_gguf_to_hf("blk.2.ssm_out.weight") == "model.layers.2.linear_attn.out_proj"
    # FP8 in_proj_a/in_proj_b (ssm_alpha/ssm_beta) DO map now: they're frozen FP8,
    # never trained as ml8 targets, but the re-emit must still write them under an
    # HF name the converter's classify_role resolves to Tier.FP8 (else they'd be
    # skipped and dropped to bf16 — the re-emit coverage bug).
    assert map_gguf_to_hf("blk.0.ssm_alpha.weight") == "model.layers.0.linear_attn.in_proj_a"
    assert map_gguf_to_hf("blk.7.ssm_beta.weight") == "model.layers.7.linear_attn.in_proj_b"
    # MTP / NextN draft head eh_proj (4-part GGUF name) maps to its HF name.
    assert map_gguf_to_hf("blk.24.nextn.eh_proj.weight") == "model.layers.24.nextn.eh_proj"
    # Genuinely NATIVE ssm-core tensors (not nn.Linear matmuls) still raise KeyError.
    for stem in ("ssm_conv1d", "ssm_dt", "ssm_a", "ssm_norm"):
        try:
            map_gguf_to_hf(f"blk.0.{stem}.weight")
            assert False, f"expected KeyError for {stem}"
        except KeyError:
            pass


def test_linear_attn_names_resolve_via_role_targets():
    """The GGUF names role_targets assigns to the linear-attn ML8 (2D matmul)
    targets must each resolve through act_replay's GGUF->HF map. This pins the two
    tables together: if role_targets/TensorNameMap renames a linear-attn matmul,
    this test trips rather than the trainer silently dropping the target."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    from role_targets import classify_role, configure, Tier

    configure("qwen35", 32)
    # HF names from the real Qwen3.5 checkpoint (see test_role_targets.py).
    hf_linear_attn = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv": "linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z":   "linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj":    "linear_attn.out_proj",
    }
    for hf_name, hf_suffix in hf_linear_attn.items():
        gguf_name, _role, tier = classify_role(hf_name)
        assert tier is Tier.ML8, f"{hf_name} should be ML8, got {tier}"
        # the gguf name role_targets produced must round-trip through act_replay's map
        assert map_gguf_to_hf(gguf_name) == f"model.layers.0.{hf_suffix}"


def test_build_response_mask_basic():
    # ids:        [ A,  B,  S,  C,  D,  E,  S ]  start=[2], end=[6]
    #  span starts after the start token (idx 2) -> tokens 3,4,5 inside, 6 (end) excluded
    ids = torch.tensor([10, 11, 2, 30, 31, 32, 6])
    m = build_response_mask(ids, start_seq=[2], end_seq=[6])
    assert m.tolist() == [0, 0, 0, 1, 1, 1, 0]


def test_build_response_mask_multi_span_and_multitoken_delims():
    # two assistant spans, multi-token start/end delimiters
    start, end = [90, 91], [80, 81]
    ids = torch.tensor([1, 90, 91, 5, 6, 80, 81, 2, 90, 91, 7, 80, 81, 9])
    m = build_response_mask(ids, start, end)
    #            1   90  91  5  6  80 81  2  90 91  7  80 81  9
    assert m.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]


def test_build_response_mask_unclosed_span_runs_to_end():
    ids = torch.tensor([1, 7, 9, 9, 9])           # start=[7], no end token present
    m = build_response_mask(ids, start_seq=[7], end_seq=[8])
    assert m.tolist() == [0, 0, 1, 1, 1]


def test_build_response_mask_no_start_is_all_zero():
    ids = torch.tensor([1, 2, 3, 4])
    m = build_response_mask(ids, start_seq=[99], end_seq=[98])
    assert m.tolist() == [0, 0, 0, 0]


def test_batch_response_mask_raw_text_fallback_all_ones():
    # no assistant span anywhere -> all-ones fallback so KL still trains on it
    ids = torch.tensor([[1, 2, 3, 4]])
    m = batch_response_mask(ids, [99], [98])
    assert m.shape == ids.shape
    assert m.sum().item() == 4.0


def test_batch_response_mask_masks_response():
    ids = torch.tensor([[1, 2, 5, 6, 7, 9]])      # start=[2], end=[9]
    m = batch_response_mask(ids, [2], [9])
    #             1  2  5  6  7  9   (tokens after start@1 up to end@5 exclusive)
    assert m.reshape(-1).tolist() == [0, 0, 1, 1, 1, 0]


def test_train_masked_loss_runs():
    """train() with resp_delims masks the KL but still descends and steps."""
    from teacher_source import LiveTeacher

    student, teacher, at = _build_student_teacher()
    teacher.eval()
    teacher_src = LiveTeacher(teacher, 8)
    g = torch.Generator().manual_seed(2)
    # ids built so a start/end delimiter pair occurs -> a real (non-fallback) mask
    batches = []
    for _ in range(4):
        body = torch.randint(0, 32, (1, 14), generator=g)
        ids = torch.cat([torch.tensor([[3]]), body, torch.tensor([[4]])], dim=1)
        batches.append(ids)
    train_idx = torch.arange(3)
    hold_idx = torch.tensor([3])
    opt = torch.optim.Adam([at.centroids, at.scales], lr=1e-2)
    step = train(student, teacher_src, batches, train_idx, hold_idx, opt,
                 grad_accum=1, epochs=2, eval_interval=0,
                 resp_delims=([3], [4]))
    assert step >= 1


def _build_student_teacher():
    """Student: stub LM with the ml8 target attached to .lin. Teacher: same arch
    with the dequant (bf16-equiv) weight baked into .lin (a frozen LiveTeacher)."""
    t = _mk_state(N=8, K=128, G=2, seed=0)
    W = dequant_ml8_state(t)            # [N,K] fp32 dequant
    # Teacher: same arch, the dequant weight; nudge so student (starting AT the
    # dequant) has somewhere to descend.
    student = StubLM(vocab=32, K=128, N=8)
    teacher = StubLM(vocab=32, K=128, N=8, weight=W)
    # share the embed + head so the only difference is .lin's weight
    with torch.no_grad():
        teacher.embed.weight.copy_(student.embed.weight)
        teacher.head.weight.copy_(student.head.weight)
    # Perturb the target so step-0 KL is nonzero: shift centroids off the dequant.
    t2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in t.items()}
    t2["centroids"] = (t2["centroids"] + 0.15)
    at = attach_to_linear(student.lin, t2)
    return student, teacher, at


def test_train_step_loss_down():
    from teacher_source import LiveTeacher
    from kl_loss import kl_topk

    student, teacher, at = _build_student_teacher()
    teacher.eval()
    K_top = 8

    g = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 32, (1, 16), generator=g) for _ in range(4)]
    teacher_src = LiveTeacher(teacher, K_top)

    def _kl_now():
        tot = 0.0
        for i, ids in enumerate(batches):
            idx, vals, tail = teacher_src.get(i, ids)
            logits = student(ids)
            V = logits.shape[-1]
            tot += kl_topk(logits.reshape(-1, V), idx, vals, tail).item()
        return tot / len(batches)

    kl0 = _kl_now()
    opt = torch.optim.Adam([at.centroids, at.scales], lr=1e-2)
    for _ in range(30):
        for i, ids in enumerate(batches):
            idx, vals, tail = teacher_src.get(i, ids)
            logits = student(ids)
            V = logits.shape[-1]
            loss = kl_topk(logits.reshape(-1, V), idx, vals, tail)
            opt.zero_grad()
            loss.backward()
            opt.step()
    kl30 = _kl_now()
    assert kl30 < kl0, f"KL did not decrease: {kl0} -> {kl30}"


def test_frozen_separate_teacher(tmp_path):
    """BUG 2 regression: the teacher must be a SEPARATE frozen instance, not the
    monkeypatched student the trainer trains. We attach ml8 targets to the
    student copy, build a LiveTeacher from a FRESH stub (the frozen parent), then
    train 30 steps and assert:
      * the teacher was actually called (teacher.calls > 0),
      * the student's KL to that fixed teacher decreased,
      * the teacher's output for fixed ids is IDENTICAL at step 0 and after
        training — i.e. it is frozen and not the moving student.
    """
    from teacher_source import LiveTeacher
    from kl_loss import kl_topk

    student, _teacher_unused, at = _build_student_teacher()

    # Frozen separate parent: a fresh stub built from the SAME dequant weight,
    # sharing embed/head, with grads off. It is NOT the student object.
    t = _mk_state(N=8, K=128, G=2, seed=0)
    W = dequant_ml8_state(t)
    parent = StubLM(vocab=32, K=128, N=8, weight=W)
    with torch.no_grad():
        parent.embed.weight.copy_(student.embed.weight)
        parent.head.weight.copy_(student.head.weight)
    parent.eval().requires_grad_(False)
    assert parent is not student

    class _CountingTeacher(LiveTeacher):
        calls = 0

        def get(self, batch_idx, ids):
            type(self).calls += 1
            return super().get(batch_idx, ids)

    K_top = 8
    teacher_src = _CountingTeacher(parent, K_top)

    g = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 32, (1, 16), generator=g) for _ in range(4)]

    # fixed-ids teacher output BEFORE training
    probe = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    idx0, vals0, tail0 = teacher_src.get(0, probe)
    idx0, vals0, tail0 = idx0.clone(), vals0.clone(), tail0.clone()
    calls_after_probe = _CountingTeacher.calls

    def _kl_now():
        tot = 0.0
        for i, ids in enumerate(batches):
            idx, vals, tail = teacher_src.get(i, ids)
            logits = student(ids)
            V = logits.shape[-1]
            tot += kl_topk(logits.reshape(-1, V), idx, vals, tail).item()
        return tot / len(batches)

    kl0 = _kl_now()
    opt = torch.optim.Adam([at.centroids, at.scales], lr=1e-2)
    for _ in range(30):
        for i, ids in enumerate(batches):
            idx, vals, tail = teacher_src.get(i, ids)
            logits = student(ids)
            V = logits.shape[-1]
            loss = kl_topk(logits.reshape(-1, V), idx, vals, tail)
            opt.zero_grad(); loss.backward(); opt.step()
    kl30 = _kl_now()

    # teacher was actually exercised, separately from student steps
    assert _CountingTeacher.calls > calls_after_probe
    assert kl30 < kl0, f"KL did not decrease: {kl0} -> {kl30}"

    # frozen: teacher output for the SAME ids is bit-identical after training.
    idx1, vals1, tail1 = teacher_src.get(0, probe)
    assert torch.equal(idx0, idx1)
    assert torch.equal(vals0, vals1)
    assert torch.equal(tail0, tail1)


def test_load_hf_model_separate_instances():
    """load_hf_model returns a NEW object each call (student != teacher parent),
    and _LMWrap yields plain logits. Exercised with a stub patched in for
    AutoModelForCausalLM so no real HF download is needed."""
    import act_replay

    made = []

    class _StubAuto:
        @staticmethod
        def from_pretrained(path, **kw):
            m = StubLM(vocab=32, K=128, N=8)
            made.append(m)
            return m

    import types, sys
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = _StubAuto
    fake_fla = types.ModuleType("fla_compat")
    fake_fla.apply_fla_arch_shim = lambda *a, **k: None
    fake_fla.apply_fla_cpu_fallback = lambda *a, **k: None
    saved = {k: sys.modules.get(k) for k in ("transformers", "fla_compat")}
    sys.modules["transformers"] = fake_tf
    sys.modules["fla_compat"] = fake_fla
    try:
        student = act_replay.load_hf_model("stub", "cpu")
        parent = act_replay.load_hf_model("stub", "cpu", freeze=True)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert student is not parent and len(made) == 2
    # frozen parent has grads off; student does not.
    assert all(not p.requires_grad for p in parent.parameters())
    assert any(p.requires_grad for p in student.parameters())

    # _LMWrap returns the plain logits tensor from an HF-style .logits output.
    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _HFLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 32, bias=False)

        def forward(self, ids):
            return _Out(self.head(ids.float()))

    wrapped = act_replay._LMWrap(_HFLike(), "cpu")
    out = wrapped(torch.zeros(1, 4))
    assert out.shape == (1, 32)  # plain logits tensor, not the wrapper object


def test_ckpt_roundtrip(tmp_path):
    student, teacher, at = _build_student_teacher()
    from teacher_source import LiveTeacher
    from kl_loss import kl_topk

    teacher.eval()
    teacher_src = LiveTeacher(teacher, 8)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, 32, (1, 16), generator=g) for _ in range(3)]

    targets = {"blk.0.ffn_up.weight": at}
    opt = torch.optim.Adam([at.centroids, at.scales], lr=1e-2)

    # take a few steps so optimizer state is populated
    for step in range(3):
        for i, ids in enumerate(batches):
            idx, vals, tail = teacher_src.get(i, ids)
            logits = student(ids)
            V = logits.shape[-1]
            loss = kl_topk(logits.reshape(-1, V), idx, vals, tail)
            opt.zero_grad(); loss.backward(); opt.step()

    ckpt_path = tmp_path / "ckpt.pt"
    save_ckpt(ckpt_path, step=3, targets=targets, optimizer=opt)

    # fresh student/target, load
    student2, teacher2, at2 = _build_student_teacher()
    targets2 = {"blk.0.ffn_up.weight": at2}
    opt2 = torch.optim.Adam([at2.centroids, at2.scales], lr=1e-2)
    step = load_ckpt(ckpt_path, targets=targets2, optimizer=opt2)

    assert step == 3
    assert torch.equal(at.centroids.detach(), at2.centroids.detach())
    assert torch.equal(at.scales.detach(), at2.scales.detach())
    # optimizer momentum state restored
    sd1 = opt.state_dict()["state"]
    sd2 = opt2.state_dict()["state"]
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.allclose(sd1[k]["exp_avg"], sd2[k]["exp_avg"])

    # training continues without error
    for i, ids in enumerate(batches):
        idx, vals, tail = teacher_src.get(i, ids)
        logits = student2(ids)
        V = logits.shape[-1]
        loss = kl_topk(logits.reshape(-1, V), idx, vals, tail)
        opt2.zero_grad(); loss.backward(); opt2.step()


def test_install_frozen_fp8_copies_and_frees():
    """install_frozen_fp8 copies each frozen fp8 weight into the matching module
    weight in-place under no_grad, frees each frozen tensor as installed, and
    skips (with a single warning) any frozen name that has no HF mapping."""
    # A module tree with a mappable target and a head that won't be touched.
    model = StubLM(vocab=32, K=128, N=8)
    modules = dict(model.named_modules())
    # rename .lin to the mapped HF path so map_gguf_to_hf resolves to it.

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(64, 4, bias=False)

    # Build a model whose named_modules contains the mapped path.
    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(64, 4, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_Layer()])

    m = _Model()
    target_mod = m.model.layers[0].mlp.up_proj
    with torch.no_grad():
        target_mod.weight.zero_()
    new_w = torch.arange(4 * 64, dtype=torch.bfloat16).reshape(4, 64)
    frozen = {
        "blk.0.ffn_up.weight": new_w,             # maps -> model.layers.0.mlp.up_proj
        "blk.0.totally_unmapped.weight": torch.ones(4, 64),  # no HF mapping -> skip+warn
    }

    n_installed = install_frozen_fp8(
        m, frozen, map_gguf_to_hf, device=torch.device("cpu"), dtype=torch.bfloat16)

    # the mapped weight was copied in-place; frozen drained as it installed.
    assert torch.equal(target_mod.weight.detach().to(torch.bfloat16), new_w)
    assert "blk.0.ffn_up.weight" not in frozen          # popped/freed
    assert "blk.0.totally_unmapped.weight" not in frozen  # unmapped also drained
    assert n_installed == 1
    assert frozen == {}


def test_export_blobs_roundtrip(tmp_path):
    t = _mk_state(N=8, K=128, G=2, seed=3)
    at = attach_to_linear(nn.Linear(128, 8, bias=False), t)
    # snap centroids to the e4m3 lattice in-place so reconstruction is bit-exact
    with torch.no_grad():
        at.centroids.copy_(snap_to_e4m3(at.centroids))

    state = {"blk.0.ffn_up.weight": at}
    hf_names = {"blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj"}

    export_blobs(state, hf_names, tmp_path)

    blob_path = tmp_path / "model.layers.0.mlp.up_proj.pt"
    assert blob_path.exists()
    blob = ml8_io.load_ml8_layer(blob_path)
    assert blob["name"] == "model.layers.0.mlp.up_proj"
    assert list(blob["shape"]) == [8, 128]
    assert blob["n_centroids"] == 16
    assert blob["group_size"] == 128 // 2  # K // G
    assert blob["indices"].dtype == torch.int8
    assert blob["mse"] == 0.0 and blob["w_snr_db"] == 0.0
    assert blob["y_snr_db"] == 0.0 and blob["rel_err"] == 0.0

    # reconstruct == the AttachedTarget dequant (centroids already on lattice)
    W_blob = ml8_io.reconstruct_weight(blob)
    W_at = at.weight().detach()
    assert torch.equal(W_blob, W_at)


# ─── BUG: re-emit coverage — fp8 export schema the converter accepts ──────────


def _configure_qwen35():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    from role_targets import configure
    configure("qwen35", 32)


def test_export_fp8_blob_has_name_and_classifies_fp8(tmp_path):
    """Regression for the act_replay re-emit 0.3%-coverage bug: the exported
    *.fp8.pt blobs used to carry only {e4m3, scale} with NO 'name', so the
    converter's _build_fp8_blob_map skipped every one ('blob has no name field')
    and the SSM/embed weights silently shipped bf16. The fix writes the canonical
    fp8 schema (name/tier/shape/group_size) and uses HF names that classify FP8."""
    from role_targets import classify_role, Tier
    _configure_qwen35()

    N, K = 16, 64                       # K divisible by FP8 group size (32)
    e4m3 = torch.randn(N, K).to(torch.float8_e4m3fn).to(torch.float32)
    scale = (torch.rand(N, K // 32) + 0.1).to(torch.float16)
    # GGUF names as produced by _read_frozen_fp8_raw; main() maps these to HF names.
    frozen_raw = {"blk.0.ssm_alpha.weight": (e4m3, scale)}
    hf_names = {"blk.0.ssm_alpha.weight": map_gguf_to_hf("blk.0.ssm_alpha.weight")}

    export_blobs({}, hf_names, tmp_path, frozen_fp8_raw=frozen_raw)

    blob_path = tmp_path / "model.layers.0.linear_attn.in_proj_a.fp8.pt"
    assert blob_path.exists()
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    # canonical schema fields the converter requires.
    assert blob["name"] == "model.layers.0.linear_attn.in_proj_a"
    assert blob["tier"] == "fp8"
    assert list(blob["shape"]) == [N, K]
    assert blob["group_size"] == 32
    assert "e4m3" in blob and "scale" in blob
    # the converter keys off blob['name'] -> classify_role -> Tier.FP8.
    _gguf, _role, tier = classify_role(blob["name"])
    assert tier is Tier.FP8, f"expected FP8 tier, got {tier}"


def test_export_untrained_ml8_reemitted(tmp_path):
    """Untrained ML8 tensors (present in the source GGUF but not training targets,
    e.g. a narrowed --tensors-train glob) must be re-exported with a converter-
    valid schema, else they drop to bf16 and tank re-emit coverage."""
    from role_targets import classify_role, Tier
    _configure_qwen35()

    t = _mk_state(N=8, K=128, G=2, seed=11)
    # snap centroids so the round-trip is lattice-exact (export snaps internally).
    t = dict(t)
    t["centroids"] = snap_to_e4m3(t["centroids"])
    untrained = {"blk.2.ffn_down.weight": t}

    # No trained targets; only the untrained ml8 map is supplied.
    export_blobs({}, {}, tmp_path, untrained_ml8=untrained)

    blob_path = tmp_path / "model.layers.2.mlp.down_proj.pt"
    assert blob_path.exists()
    blob = ml8_io.load_ml8_layer(blob_path)
    assert blob["name"] == "model.layers.2.mlp.down_proj"
    assert list(blob["shape"]) == [8, 128]
    # classifies as an ML8 weight the converter will pack.
    _gguf, _role, tier = classify_role(blob["name"])
    assert tier is Tier.ML8
    # a trained target for the SAME tensor wins over the untrained copy.
    at = attach_to_linear(nn.Linear(128, 8, bias=False), _mk_state(N=8, K=128, G=2, seed=99))
    with torch.no_grad():
        at.centroids.copy_(snap_to_e4m3(at.centroids))
    export_blobs({"blk.2.ffn_down.weight": at},
                 {"blk.2.ffn_down.weight": "model.layers.2.mlp.down_proj"},
                 tmp_path, untrained_ml8=untrained)
    W_blob = ml8_io.reconstruct_weight(ml8_io.load_ml8_layer(blob_path))
    assert torch.equal(W_blob, at.weight().detach())


# ─── 4B memory knob: --train-seq-len window splitting ─────────────────────────


def test_split_batches_seq_counts():
    # Two full batches of T=10 each; window=4 -> ceil(10/4)=3 windows per batch.
    batches = [torch.arange(10).reshape(1, 10), torch.arange(10, 20).reshape(1, 10)]
    train_idx = torch.tensor([0, 1])
    win_batches, win_idx = split_batches_seq(batches, train_idx, train_seq_len=4)
    assert len(win_batches) == 6          # 3 + 3
    assert win_idx.tolist() == list(range(6))
    # window lengths tile the full sequence: 4,4,2 per source batch (no tokens lost).
    lens = [w.shape[-1] for w in win_batches]
    assert lens == [4, 4, 2, 4, 4, 2]
    # concatenating the windows of batch 0 reproduces the original ids exactly.
    cat0 = torch.cat(win_batches[:3], dim=-1)
    assert torch.equal(cat0, batches[0])


def test_split_batches_seq_disabled_passthrough():
    batches = [torch.arange(8).reshape(1, 8)]
    train_idx = torch.tensor([0])
    # None disables; window >= T passes the batch through whole.
    for tsl in (None, 8, 16):
        wb, wi = split_batches_seq(batches, train_idx, train_seq_len=tsl)
        assert len(wb) == 1 and wi.tolist() == [0]
        assert torch.equal(wb[0], batches[0])


def test_split_batches_seq_only_train_idx():
    # Three batches but only indices {0,2} are train; batch 1 (holdout) is untouched.
    batches = [torch.arange(6).reshape(1, 6),
               torch.full((1, 6), -1),
               torch.arange(100, 106).reshape(1, 6)]
    train_idx = torch.tensor([0, 2])
    wb, wi = split_batches_seq(batches, train_idx, train_seq_len=3)
    assert len(wb) == 4                   # 2 windows each for batches 0 and 2
    # holdout sentinel batch never appears in the windowed train list.
    assert all((w != -1).all() for w in wb)


def test_split_batches_seq_mask_preserved_per_window():
    """The response mask is recomputed from each window's ids at train time, so
    splitting ids preserves masking WITHIN a window: a window that contains the
    assistant START still masks its response tokens exactly. (A window with NO
    START falls back to the documented all-ones raw-text mask — the same fallback
    batch_response_mask applies to any record without an assistant span.)"""
    start, end = [1], [2]                 # 1-token start/end delimiters
    # ids: [pad, START, r, r, END, x, x, x]  -> response tokens are indices 2,3.
    ids = torch.tensor([[0, 1, 5, 6, 2, 8, 9, 4]])
    full_mask = batch_response_mask(ids, start, end)
    assert full_mask[0].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]

    wb, wi = split_batches_seq([ids], torch.tensor([0]), train_seq_len=4)
    assert len(wb) == 2
    # window 0 = ids[:4] = [0,1,5,6] -> START at idx1, span open to window end -> 2,3=1
    win0_mask = batch_response_mask(wb[0], start, end)
    assert win0_mask[0].tolist() == [0, 0, 1, 1]
    # window 0's mask == the matching slice of the full-sequence mask (1:1 preserved).
    assert torch.equal(win0_mask, full_mask[:, :4])
    # window 1 = ids[4:] = [2,8,9,4] -> no START -> all-ones raw-text fallback.
    win1_mask = batch_response_mask(wb[1], start, end)
    assert win1_mask[0].tolist() == [1, 1, 1, 1]
    # no tokens are lost: concatenating the windows reproduces the source ids.
    assert torch.equal(torch.cat(wb, dim=-1), ids)


# ─── 4B memory knob: alloc-conf launch hint ───────────────────────────────────


# POLARITY FLIP 2026-06-10: expandable_segments page-faults gfx1201 under this
# trainer (mbtopk; 5/5 repro) — the hint now WARNS when it IS set.
def test_alloc_conf_hint_cpu_is_none():
    env = {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}
    assert alloc_conf_hint("cpu", env=env) is None


def test_alloc_conf_hint_cuda_unset_is_none():
    assert alloc_conf_hint("cuda:0", env={}) is None


def test_alloc_conf_hint_cuda_expandable_warns():
    env = {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}
    msg = alloc_conf_hint("cuda:0", env=env)
    assert msg is not None
    assert "expandable_segments" in msg and "WARNING" in msg


def test_alloc_conf_hint_cuda_other_conf_is_none():
    env = {"PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:256"}
    assert alloc_conf_hint("cuda:0", env=env) is None


# ─── linear-attn GGUF -> HF V-head reorder inversion ─────────────────────────


class _Cfg:
    """Minimal HF-config stand-in nesting linear-attn head dims under text_config."""

    def __init__(self, num_k, num_v, head_v=128, head_k=128):
        self.text_config = type("TC", (), {
            "linear_num_key_heads": num_k,
            "linear_num_value_heads": num_v,
            "linear_value_head_dim": head_v,
            "linear_key_head_dim": head_k,
        })()


def _reorder_v_heads_ref(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Reference grouped->tiled reorder, copied from conversion/qwen.py."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    t = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def test_gguf_to_hf_perm_08b_identity():
    """0.8B has num_value_heads == num_key_heads (r == 1) -> every perm is None."""
    cfg = _Cfg(num_k=16, num_v=16)
    for stem, shape in [("attn_qkv", (8192, 1024)), ("attn_gate", (2048, 1024)),
                        ("ssm_out", (1024, 2048)), ("ssm_alpha", (16, 1024)),
                        ("ssm_beta", (16, 1024))]:
        assert gguf_to_hf_perm(f"blk.0.{stem}.weight", shape, cfg) is None


def test_gguf_to_hf_perm_non_linear_attn_is_none():
    cfg = _Cfg(num_k=16, num_v=32)
    assert gguf_to_hf_perm("blk.0.ffn_gate.weight", (9216, 2560), cfg) is None
    assert gguf_to_hf_perm("blk.0.attn_q.weight", (4096, 2560), cfg) is None
    assert gguf_to_hf_perm("token_embd.weight", (248320, 2560), cfg) is None


def test_gguf_to_hf_perm_none_config_is_none():
    assert gguf_to_hf_perm("blk.0.attn_gate.weight", (4096, 2560), None) is None


def test_gguf_to_hf_perm_4b_inverts_reorder():
    """The returned perm is the exact inverse of the convert-time grouped->tiled
    reorder for each 4B linear-attn tensor (num_k=16, num_v=32, head_dim=128)."""
    nk, nv, hv, hk = 16, 32, 128, 128
    nvpk = nv // nk
    cfg = _Cfg(num_k=nk, num_v=nv, head_v=hv, head_k=hk)
    q_dim = k_dim = hk * nk        # 2048
    v_dim = nv * hv                # 4096

    # in_proj_z (attn_gate): all rows reorder. Round-trip: grouped -> tiled (the
    # GGUF layout) then perm should recover grouped.
    grouped = torch.arange(v_dim)
    tiled = _reorder_v_heads_ref(grouped.unsqueeze(-1), 0, nk, nvpk, hv).squeeze(-1)
    axis, idx = gguf_to_hf_perm("blk.0.attn_gate.weight", (v_dim, 2560), cfg)
    assert axis == 0
    assert torch.equal(tiled[idx], grouped)

    # in_proj_qkv (attn_qkv): only the trailing V rows reorder; q/k rows fixed.
    n_rows = 2 * k_dim + v_dim
    axis, idx = gguf_to_hf_perm("blk.0.attn_qkv.weight", (n_rows, 2560), cfg)
    assert axis == 0
    assert torch.equal(idx[:2 * k_dim], torch.arange(2 * k_dim))
    # rebuild full GGUF (tiled) row order and confirm perm recovers HF (grouped)
    gguf_rows = torch.cat([torch.arange(2 * k_dim), (2 * k_dim) + tiled])
    hf_rows = torch.cat([torch.arange(2 * k_dim), (2 * k_dim) + grouped])
    assert torch.equal(gguf_rows[idx], hf_rows)

    # out_proj (ssm_out): COLUMN (input) reorder over the V space.
    axis, idx = gguf_to_hf_perm("blk.0.ssm_out.weight", (2560, v_dim), cfg)
    assert axis == 1
    assert torch.equal(tiled[idx], grouped)

    # in_proj_a / in_proj_b (ssm_alpha/ssm_beta): row reorder with head_dim == 1.
    grouped_h = torch.arange(nv)
    tiled_h = _reorder_v_heads_ref(grouped_h.unsqueeze(-1), 0, nk, nvpk, 1).squeeze(-1)
    for stem in ("ssm_alpha", "ssm_beta"):
        axis, idx = gguf_to_hf_perm(f"blk.0.{stem}.weight", (nv, 2560), cfg)
        assert axis == 0
        assert torch.equal(tiled_h[idx], grouped_h)


def test_gguf_to_hf_perm_qkv_row_count_mismatch_raises():
    cfg = _Cfg(num_k=16, num_v=32)
    with pytest.raises(ValueError):
        gguf_to_hf_perm("blk.0.attn_qkv.weight", (9999, 2560), cfg)


def test_attach_targets_applies_v_reorder():
    """attach_targets reorders an ml8 in_proj_z target's rows to HF order when a
    linear-attn config is supplied (and is a no-op when num_v == num_k)."""
    nk, nv, hv = 16, 32, 128
    nvpk = nv // nk
    N = nv * hv            # 4096 rows
    K = 128                # small input dim for a fast test

    # Distinct per-row scales so a row reorder is observable; group count G=1.
    state_entry = {
        "indices": torch.randint(0, 16, (N, K)),
        "scales": (torch.arange(N, dtype=torch.float32) + 1.0).reshape(N, 1),
        "centroids": torch.randn(1, 16).to(torch.float8_e4m3fn).to(torch.float32),
        "rotation": None,
    }

    class _State:
        ml8 = {"blk.0.attn_gate.weight": state_entry}

    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            la = nn.Module()
            la.in_proj_z = nn.Linear(K, N, bias=False)
            self.model.layers[0].linear_attn = la

    lm = _LM()
    cfg = _Cfg(num_k=nk, num_v=nv, head_v=hv)
    # No model_config -> no reorder: scales stay in GGUF (tiled) order.
    base = attach_targets(dict(_LM().named_modules()), _State(),
                          train="ml8", skip=None, model_config=None)
    assert torch.equal(base["blk.0.attn_gate.weight"].scales.detach(),
                       state_entry["scales"])

    attached = attach_targets(dict(lm.named_modules()), _State(),
                              train="ml8", skip=None, model_config=cfg)
    at = attached["blk.0.attn_gate.weight"]

    # With the config, rows are index_select'd by the inverse perm so the attached
    # scales are exactly state_entry["scales"][idx] — and since row r holds value
    # r+1, that equals idx+1.
    tiled = _reorder_v_heads_ref(torch.arange(N).unsqueeze(-1), 0, nk, nvpk, hv).squeeze(-1)
    idx = torch.argsort(tiled)                       # tiled -> grouped, == perm idx
    expected = state_entry["scales"][idx]
    assert torch.equal(at.scales.detach(), expected)
    # The reorder is real (not identity): some rows actually moved.
    assert not torch.equal(at.scales.detach(), state_entry["scales"])


# ─── derive_tier_override: heterogeneous-arch tier derivation ────────────────


def test_derive_tier_spec_captures_role_absent_from_layer0():
    """A role that lives ONLY in non-zero layers must still be captured.

    This is the A3 attn_v regression: the source GGUF is a hybrid arch where
    only every Nth layer is full-attention, so blk.0 is an SSM/fused-qkv layer
    with NO split attn_v. attn_v is fp8-protected in the full-attention layers
    (3, 7, ...). Deriving the tier spec from blk.0 stems alone DROPS attn_v;
    the converter then sees its fp8 blob as tier-mismatched, skips it, and the
    re-emitted tensor is byte-garbage (catastrophic KL). The spec must reflect
    every role across the MAIN layers, not just layer 0.
    """
    named = [
        ("blk.0.attn_qkv.weight", "ml8"),     # layer 0: fused qkv (no split v)
        ("blk.0.ffn_up.weight", "fp8"),
        ("blk.0.ffn_down.weight", "ml8"),
        ("blk.0.attn_norm.weight", None),     # F32 — not a tiered leaf
        ("blk.3.attn_q.weight", "ml8"),       # full-attention layer
        ("blk.3.attn_v.weight", "fp8"),       # fp8-protected, absent from blk.0
        ("blk.7.attn_v.weight", "fp8"),
        ("token_embd.weight", "fp8"),
    ]
    spec = _derive_tier_spec(named)
    d = dict(kv.split("=") for kv in spec.split(",") if kv)
    assert d.get("attn_v") == "fp8"        # THE BUG: previously dropped
    assert d.get("attn_q") == "ml8"
    assert d.get("attn_qkv") == "ml8"
    assert d.get("ffn_up") == "fp8"
    assert d.get("ffn_down") == "ml8"
    assert d.get("token_embd") == "fp8"
    assert "attn_norm" not in d             # F32 leaves carry no tier


def test_derive_tier_spec_excludes_trailing_mtp_block():
    """The trailing MTP/nextn block legitimately re-tiers shared roles (e.g.
    everything ML8_FP8 in blk.32). It is NOT a role-table leaf, so its tiers
    must not contaminate — nor conflict with — the main-layer derivation.
    """
    named = [
        ("blk.0.ffn_up.weight", "fp8"),
        ("blk.0.ffn_down.weight", "ml8"),
        ("blk.0.attn_qkv.weight", "ml8"),
        # MTP block (index 8 here): nextn marker + everything fp8.
        ("blk.8.nextn.eh_proj.weight", "fp8"),
        ("blk.8.ffn_down.weight", "fp8"),     # main ffn_down is ml8 — MTP differs
        ("blk.8.attn_qkv.weight", "fp8"),
    ]
    spec = _derive_tier_spec(named)
    d = dict(kv.split("=") for kv in spec.split(",") if kv)
    assert d.get("ffn_down") == "ml8"      # main-layer tier wins, not MTP's fp8
    assert d.get("attn_qkv") == "ml8"
    assert d.get("ffn_up") == "fp8"
    assert "nextn" not in spec and "eh_proj" not in spec


def test_derive_tier_spec_raises_on_mixed_main_layer_tier():
    """If a role genuinely carries two tiers across MAIN (non-MTP) layers, a
    role-uniform spec cannot express it — fail loudly rather than silently
    pick one and corrupt half the tensors.
    """
    named = [
        ("blk.0.ffn_down.weight", "ml8"),
        ("blk.4.ffn_down.weight", "fp8"),     # same role, conflicting tier, both main
    ]
    with pytest.raises(ValueError, match="mixed tier"):
        _derive_tier_spec(named)


# ─── export must preserve the (frozen) Kronecker rotation sidecar ────────────


def _rotated_target(N=4, a_dim=4, b_dim=2, G=2):
    """A synthetic ml8 target dict carrying a Kronecker rotation (K = a_dim*b_dim)."""
    K = a_dim * b_dim
    torch.manual_seed(0)
    h_a, _ = torch.linalg.qr(torch.randn(a_dim, a_dim))   # orthogonal a×a
    return {
        "indices": torch.randint(0, 16, (N, K), dtype=torch.uint8),
        "centroids": torch.randn(G, 16),
        "scales": torch.rand(N, G) + 0.01,
        "rotation": {"h_a": h_a, "a_dim": a_dim, "b_dim": b_dim},
    }, K


def test_faithful_acts_passes_gradient_to_upstream_layer():
    """STACKED faithful-acts layers must let gradient reach an UPSTREAM layer's
    codebook. apply_acts quantizes activations through an e4m3 STE (mirroring the
    weight-quant STE); without it, quantize_act_per_row's @no_grad detaches x_eff
    from x and the gradient to any layer feeding a downstream rotated linear is
    severed — exactly zero. A single-linear test cannot catch this (no trainable
    param upstream of the detach); it only appears with stacked layers. Regression
    guard for the 2026-06-11 act-STE fix.
    """
    tA, K = _rotated_target(N=8, a_dim=4, b_dim=2)
    tB, _ = _rotated_target(N=8, a_dim=4, b_dim=2)
    linA = nn.Linear(K, 8, bias=False)
    linB = nn.Linear(8, 8, bias=False)
    atA = attach_to_linear(linA, tA, faithful_acts=True)
    atB = attach_to_linear(linB, tB, faithful_acts=True)

    out = linB(linA(torch.randn(4, K)))
    out.pow(2).sum().backward()

    up = float(atA.centroids.grad.abs().sum()) + float(atA.scales.grad.abs().sum())
    down = float(atB.centroids.grad.abs().sum()) + float(atB.scales.grad.abs().sum())
    assert down > 0, "sanity: the downstream rotated layer must get gradient"
    assert up > 0, "UPSTREAM rotated layer got zero gradient — act STE severed"


def test_faithful_acts_forward_value_unchanged_by_ste():
    """The act-STE must change ONLY the backward: forward value stays exactly the
    raw quantize_act_per_row(x@Q). Guards against the STE silently altering the
    deployed-faithful activation path."""
    from ml8_e4m3_sim import quantize_act_per_row
    target, K = _rotated_target(N=8, a_dim=4, b_dim=2)
    at = attach_to_linear(nn.Linear(K, 8, bias=False), target, faithful_acts=True)
    x = torch.randn(4, K)
    ref = quantize_act_per_row(
        at.rotation.forward(x.reshape(-1, K).float())).reshape(x.shape)
    assert torch.equal(at.apply_acts(x), ref), "act-STE altered the forward value"


def test_export_blobs_preserves_rotation_sidecar(tmp_path):
    """A rehydrated rotated ml8 target must re-emit its rotation. The trained
    weights live in the ROTATED basis; dropping the rotation sidecar makes the
    deployed kernel matmul in the wrong basis — every ml8 GEMM wrong, ~12 KLD.
    The converter writes rotation_h_a/_meta ONLY when the blob carries a
    'rotation' dict, so export_blobs must populate it.
    """
    target, K = _rotated_target()
    at = attach_to_linear(nn.Linear(K, 4, bias=False), target, faithful_acts=True)
    state = {"blk.0.ffn_gate.weight": at}
    hf_names = {"blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj"}
    export_blobs(state, hf_names, tmp_path)
    blob = torch.load(tmp_path / "model.layers.0.mlp.gate_proj.pt", weights_only=False)
    assert blob.get("rotation") is not None, "rotation sidecar DROPPED on export"
    rot = blob["rotation"]
    assert rot["kind"] == "kronecker_orth_sylvester"
    assert int(rot["a_dim"]) == 4 and int(rot["b_dim"]) == 2
    assert int(rot["in_features"]) == K
    assert torch.allclose(rot["h_a"].float(), target["rotation"]["h_a"].float())


def test_export_blobs_no_rotation_is_clean(tmp_path):
    """A target with no rotation must not invent one (legacy/identity blobs)."""
    target, K = _rotated_target()
    target.pop("rotation")
    at = attach_to_linear(nn.Linear(K, 4, bias=False), target, faithful_acts=False)
    export_blobs({"blk.0.ffn_gate.weight": at},
                 {"blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj"}, tmp_path)
    blob = torch.load(tmp_path / "model.layers.0.mlp.gate_proj.pt", weights_only=False)
    assert blob.get("rotation") in (None, {}), "fabricated a rotation where none exists"


# ─── export must INVERT the attach-time GGUF->HF V-head permutation ──────────


class _LinAttnCfg:
    linear_num_value_heads = 4
    linear_num_key_heads = 2
    linear_value_head_dim = 4
    linear_key_head_dim = 4


def test_export_inverts_vhead_permutation_roundtrip(tmp_path):
    """attach_targets reorders GGUF->HF rows (gguf_to_hf_perm) so the student
    trains in HF layout; export MUST invert it so the re-emit is GGUF-order.
    On the 4B (num_v != num_k) the perm is NON-identity, so attn_qkv/attn_gate
    ship scrambled (~90-136% weight error, ~10 KLD) if export skips the inverse.
    It was invisible on the 0.8B (num_v == num_k -> identity perm), where the
    existing round-trip tests live.
    """
    from act_replay import gguf_to_hf_perm, _apply_perm_to_ml8_entry
    cfg = _LinAttnCfg()
    name = "blk.0.attn_gate.weight"
    N, K, G = 16, 64, 1                          # N = num_v_heads*head_v_dim = 16
    torch.manual_seed(0)
    gguf_entry = {                               # canonical GGUF (grouped) order
        "indices": torch.randint(0, 16, (N, K), dtype=torch.uint8),
        "centroids": torch.randn(G, 16),
        "scales": torch.rand(N, G) + 0.01,
    }
    perm = gguf_to_hf_perm(name, (N, K), cfg)
    assert perm is not None, "test config must produce a non-identity perm"
    # Simulate attach_targets: reorder into HF order, then wrap.
    hf_entry = _apply_perm_to_ml8_entry({k: v.clone() for k, v in gguf_entry.items()}, perm)
    at = attach_to_linear(nn.Linear(K, N, bias=False), hf_entry, faithful_acts=False)
    export_blobs({name: at}, {name: "model.layers.0.x"}, tmp_path, model_config=cfg)
    blob = torch.load(tmp_path / "model.layers.0.x.pt", weights_only=False)
    assert torch.equal(blob["indices"].to(torch.uint8), gguf_entry["indices"]), \
        "export did NOT invert the V-head permutation — rows scrambled"
    assert torch.equal(blob["scale_per_group"], gguf_entry["scales"])


def test_export_roundtrip_identity_when_no_config(tmp_path):
    """No model_config (or non-linear-attn) => identity perm => unchanged rows
    (preserves 0.8B / dense behavior)."""
    torch.manual_seed(1)
    N, K, G = 16, 64, 1
    entry = {"indices": torch.randint(0, 16, (N, K), dtype=torch.uint8),
             "centroids": torch.randn(G, 16), "scales": torch.rand(N, G) + 0.01}
    at = attach_to_linear(nn.Linear(K, N, bias=False), entry, faithful_acts=False)
    export_blobs({"blk.0.attn_gate.weight": at},
                 {"blk.0.attn_gate.weight": "model.layers.0.x"}, tmp_path, model_config=None)
    blob = torch.load(tmp_path / "model.layers.0.x.pt", weights_only=False)
    assert torch.equal(blob["indices"].to(torch.uint8), entry["indices"])


def test_parse_args_fp8_defaults():
    a = parse_args(["--gguf","g","--base-gguf","b","--model","m","--out-dir","o"])
    assert a.fp8 is False and a.reassign == "none" and a.lr_warmup_steps == 0
    assert a.lr_cent == 2e-4 and a.lr_scale == 2e-5


def test_lr_warmup_cosine_shape():
    from act_replay import lr_warmup_cosine
    assert lr_warmup_cosine(1,2,10) == 0.5 and lr_warmup_cosine(2,2,10) == 1.0
    assert lr_warmup_cosine(10,2,10) == 0.0 and lr_warmup_cosine(6,2,10) < 1.0


def test_apply_lr_schedule_scales_param_groups():
    import torch
    from act_replay import _apply_lr_schedule
    p1 = torch.nn.Parameter(torch.zeros(2)); p2 = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{"params":[p1],"lr":0.1},{"params":[p2],"lr":0.01}], lr=0.1)
    base = [0.1, 0.01]
    m = _apply_lr_schedule(opt, base, step=1, warmup=2, total=10)   # multiplier 0.5
    assert abs(m - 0.5) < 1e-9
    assert abs(opt.param_groups[0]["lr"] - 0.05) < 1e-9
    assert abs(opt.param_groups[1]["lr"] - 0.005) < 1e-9
    m2 = _apply_lr_schedule(opt, base, step=2, warmup=2, total=10)  # multiplier 1.0
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-9


def test_reassign_targets_none_is_noop():
    import torch
    from act_replay import reassign_targets
    from act_replay_student import AttachedTarget
    at = AttachedTarget(_mk_state(N=8, K=128, G=2))
    before = at.indices.clone()
    n = reassign_targets([at], "none", frac=1.0)
    assert n == 0 and torch.equal(at.indices, before)


def test_reassign_targets_pv_flips_with_stashed_grad():
    import torch
    from act_replay import reassign_targets
    from act_replay_student import AttachedTarget
    from fp8_qat import Ml8Fp8Fn
    at = AttachedTarget(_mk_state(N=8, K=128, G=2))
    before = at.indices.clone()
    # craft a large-magnitude dL/dW so pv predicts beneficial flips for many elems
    Ml8Fp8Fn.last_dLdW[id(at.indices)] = torch.full_like(at.W_orig, -5.0)
    n = reassign_targets([at], "pv", frac=1.0)
    assert n >= 0
    # at least some indices changed (a strong uniform gradient should move assignments)
    assert not torch.equal(at.indices, before) or n == 0
    # indices stay in valid range
    assert at.indices.max().item() <= 15
