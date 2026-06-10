"""CLI / trainer tests for act_replay.py (Task 6, act-replay KL trainer).

No HF model required: a tiny stub LM exposing named_modules() with one nn.Linear
stands in for the student/teacher. Run from scripts/calibration with
PYTHONPATH=../../gguf-py.
"""
import torch
import torch.nn as nn

from act_replay import (
    parse_args,
    split_holdout,
    map_gguf_to_hf,
    build_response_mask,
    batch_response_mask,
    train,
    save_ckpt,
    load_ckpt,
    export_blobs,
    install_frozen_fp8,
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
    assert a.lr_cent == 1e-3
    assert a.lr_scale == 1e-4
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
    # hybrid (qwen35) linear-attn 2D matmul targets — only the ML8 matmuls map.
    assert map_gguf_to_hf("blk.0.attn_qkv.weight") == "model.layers.0.linear_attn.in_proj_qkv"
    assert map_gguf_to_hf("blk.5.attn_gate.weight") == "model.layers.5.linear_attn.in_proj_z"
    assert map_gguf_to_hf("blk.2.ssm_out.weight") == "model.layers.2.linear_attn.out_proj"
    # FP8 / NATIVE ssm-core tensors are intentionally NOT mapped (raise KeyError).
    for stem in ("ssm_alpha", "ssm_beta", "ssm_conv1d", "ssm_dt", "ssm_a", "ssm_norm"):
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
