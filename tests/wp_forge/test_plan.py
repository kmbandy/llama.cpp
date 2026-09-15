"""CPU-only checks for conversion/wp_forge/plan.py: YAML shape, resolve() rules, naming."""
from __future__ import annotations

from pathlib import Path

import pytest

from conversion.wp_forge.machines import Machine
from conversion.wp_forge.plan import (
    LayerRange, Plan, PlanError, Placement, StageSpec,
    load_plan, parse_widths, resolve,
)

# Mirrors the real dsv41 config.json (flattened). The spec-head expert-count key
# is dspark_n_routed_experts (NOT nextn_n_routed_experts).
HP = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "num_hidden_layers": 40, "num_nextn_predict_layers": 3,
    "n_routed_experts": 384, "dspark_n_routed_experts": 128,
    "moe_intermediate_size": 2304, "hidden_size": 5120,
}
MS = {
    "main": Machine("main", None, "/home/u/models"),
    "b2026": Machine("b2026", "u@b2026", "/mnt/nvme"),
}


def plan(**over) -> Plan:
    base = dict(
        name="dsv41", source="hf:deepseek-ai/DeepSeek-V4.1-Flash",
        quant={"experts": "mxfp4", "dense": "q8_0", "engram": "q4_0"},
        machines_path=None,
        spine=Placement("main", None), sidecars=Placement("main", None),
        experts=[StageSpec(LayerRange(0, 25), "main", None, [1408, 896]),
                 StageSpec(LayerRange(26, 39), "b2026", None, None)],
        spec_head=StageSpec(LayerRange(40, 42), "b2026", None, None),
    )
    base.update(over)
    return Plan(**base)


# --- load_plan: YAML shape ---

def test_load_plan_yaml(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("""
name: dsv41
source: hf:deepseek-ai/DeepSeek-V4.1-Flash
quant: {experts: mxfp4, dense: q8_0}
spine: main
experts:
  - {layers: 0-25, machine: main, widths: [1408, 896]}
  - {layers: 26-39, machine: b2026, path: /mnt/nvme}
spec_head: {layers: 40-42, machine: b2026}
""")
    pl = load_plan(p)
    assert pl.experts[0].layers == LayerRange(0, 25) and pl.experts[0].widths == [1408, 896]
    assert pl.experts[1].path == "/mnt/nvme" and pl.spec_head.layers == LayerRange(40, 42)
    assert pl.sidecars == Placement("main", None)  # defaults to spine placement
    assert pl.allow_partial is False


def test_load_plan_yaml_shape_errors(tmp_path: Path) -> None:
    def fails(text: str, match: str) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(text)
        with pytest.raises(PlanError, match=match):
            load_plan(p)

    fails("", "mapping")
    fails("source: hf:x\nspine: main", "name")
    fails("name: dsv41\nsource: tarball:x\nspine: main", "hf:")
    fails("name: dsv41\nsource: hf:x\nspine: main\nquant: {experts: mxfp4}\nexperts: [{layers: 0-5}]", "machine")
    fails("name: dsv41\nsource: hf:x\nspine: main\nquant: {experts: mxfp4}\nexperts: [{layers: 5-0, machine: main}]", "layers")


# --- resolve: ids, dirs, naming, order, est_bytes ---

def test_resolve_ids_paths_and_order() -> None:
    r = resolve(plan(), HP, MS, source_is_gguf=False)
    ids = [s.id for s in r.sets]
    assert ids == ["L0-25-w0", "L0-25-w1", "L26-39", "L40-42"]
    assert r.sets[0].dir == "/home/u/models/dsv41/L0-25-w0"
    assert r.sets[0].output_base == "dsv41-L0-25-w0"
    assert r.sets[0].slice_index == 0 and r.sets[1].slice_index == 1
    assert r.sets[2].dir == "/mnt/nvme/dsv41/L26-39" and r.sets[2].machine == "b2026"
    assert r.sets[2].slice_index is None and r.sets[2].widths is None
    assert r.sets[3].role == "spec_head" and r.sets[3].quant == "mxfp4"
    assert r.spine_path == "/home/u/models/dsv41/dsv41-spine.gguf"
    assert r.bundle_dir == "/home/u/models/dsv41"
    assert r.sidecar_paths == {"engram": "/home/u/models/dsv41/dsv41-engram.gguf"}
    assert r.dispatch_order() == ids
    # est_bytes from GGML_QUANT_SIZES: positive, and a width-sliced set's
    # slice ests sum to what an unsliced set over the same layers would cost
    assert all(s.est_bytes > 0 for s in r.sets)
    full = resolve(plan(experts=[StageSpec(LayerRange(0, 25), "main", None, None),
                                StageSpec(LayerRange(26, 39), "b2026", None, None)]),
                   HP, MS, source_is_gguf=False)
    assert r.sets[0].est_bytes + r.sets[1].est_bytes == full.sets[0].est_bytes


def test_parse_widths() -> None:
    assert parse_widths([1408, 896], 2304, 32) == [1408, 896]
    assert parse_widths("2:1", 2304, 32) == [1536, 768]
    assert parse_widths("1:1:1", 2304, 32) == [768, 768, 768]
    with pytest.raises(PlanError, match="multiple of 32"):
        parse_widths([1400, 904], 2304, 32)
    with pytest.raises(PlanError, match="ratio"):
        parse_widths("1:0", 2304, 32)


# --- resolve: the rule table, one PlanError per rule ---

def test_widths_must_tile_n_ff_exp() -> None:
    # [1408, 640] sums to 2048, not 2304
    bad = plan(experts=[StageSpec(LayerRange(0, 25), "main", None, [1408, 640]),
                        StageSpec(LayerRange(26, 39), "b2026", None, None)])
    with pytest.raises(PlanError, match="widths"):
        resolve(bad, HP, MS, source_is_gguf=False)


def test_tiling_gap() -> None:
    bad = plan(experts=[StageSpec(LayerRange(0, 20), "main", None, None),
                        StageSpec(LayerRange(26, 39), "b2026", None, None)])
    with pytest.raises(PlanError, match="gap"):
        resolve(bad, HP, MS, source_is_gguf=False)
    # allow_partial: true means the gap is intentional
    ok = plan(allow_partial=True, experts=bad.experts)
    assert [s.id for s in resolve(ok, HP, MS, source_is_gguf=False).sets] == ["L0-20", "L26-39", "L40-42"]


def test_tiling_overlap() -> None:
    bad = plan(experts=[StageSpec(LayerRange(0, 26), "main", None, None),
                        StageSpec(LayerRange(26, 39), "b2026", None, None)])
    with pytest.raises(PlanError, match="overlap"):
        resolve(bad, HP, MS, source_is_gguf=False)


def test_layers_outside_model() -> None:
    bad = plan(experts=[StageSpec(LayerRange(0, 41), "main", None, None)])
    with pytest.raises(PlanError, match="not expert"):
        resolve(bad, HP, MS, source_is_gguf=False)


def test_spec_head_rules() -> None:
    with pytest.raises(PlanError, match="spec_head"):
        resolve(plan(spec_head=None), HP, MS, source_is_gguf=False)
    r = resolve(plan(spec_head="none"), HP, MS, source_is_gguf=False)
    assert all(s.role == "experts" for s in r.sets)
    with pytest.raises(PlanError, match="spec_head"):  # partial spec head is a gap
        resolve(plan(spec_head=StageSpec(LayerRange(40, 41), "b2026", None, None)),
                HP, MS, source_is_gguf=False)
    with pytest.raises(PlanError, match="no spec-head"):  # model without nextn layers
        hp = {k: v for k, v in HP.items() if k != "num_nextn_predict_layers"}
        resolve(plan(spec_head=StageSpec(LayerRange(40, 42), "b2026", None, None)),
                hp, MS, source_is_gguf=False)


def test_quant_gguf_source_forbidden() -> None:
    with pytest.raises(PlanError, match="gguf"):
        resolve(plan(source="gguf:/x.gguf"), HP, MS, source_is_gguf=True)
    # gguf source + no quant block: expert sets are byte-verbatim, est 0
    r = resolve(plan(source="gguf:/x.gguf", quant={}), HP, MS, source_is_gguf=True)
    assert all(s.est_bytes == 0 and s.quant == "src" for s in r.sets)


def test_quant_experts_required_for_hf() -> None:
    with pytest.raises(PlanError, match="experts"):
        resolve(plan(quant={"dense": "q8_0"}), HP, MS, source_is_gguf=False)


def test_quant_unknown_class() -> None:
    with pytest.raises(PlanError, match=r"unknown class 'bogus' for deepseek41 \(classes: dense, experts, spec_head\.dense, spec_head\.experts, engram\)"):
        resolve(plan(quant={"experts": "mxfp4", "bogus": "q4_0"}), HP, MS, source_is_gguf=False)


def test_quant_type_must_be_producible() -> None:
    with pytest.raises(PlanError, match="not producible"):
        resolve(plan(quant={"experts": "q9_9"}), HP, MS, source_is_gguf=False)


def test_spec_head_expert_quant_override() -> None:
    r = resolve(plan(quant={"experts": "mxfp4", "spec_head.experts": "q8_0", "engram": "q4_0"}),
                HP, MS, source_is_gguf=False)
    assert r.sets[3].quant == "q8_0"


def test_machine_rule() -> None:
    with pytest.raises(PlanError, match="machine 'nope'"):
        resolve(plan(spine=Placement("nope", None)), HP, MS, source_is_gguf=False)
    with pytest.raises(PlanError, match="machine 'nope'"):
        resolve(plan(experts=[StageSpec(LayerRange(0, 25), "nope", None, None),
                              StageSpec(LayerRange(26, 39), "b2026", None, None)]),
                HP, MS, source_is_gguf=False)
