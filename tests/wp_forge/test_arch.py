"""CPU-only checks for conversion/wp_forge/arch.py. No GPU, no HF weights."""
from __future__ import annotations

from conversion.wp_forge import arch as A


# Mirrors the real /home/kmbandy/models/dsv41-hf-spine/config.json (text_config
# flattened). The plan's "nextn_n_routed_experts" guess is wrong: the actual key
# in config.json is "dspark_n_routed_experts".
HP = {"architectures": ["DeepseekV41ForCausalLM"], "num_hidden_layers": 40,
      "num_nextn_predict_layers": 3, "n_routed_experts": 384,
      "dspark_n_routed_experts": 128, "moe_intermediate_size": 2304,
      "hidden_size": 5120}


def test_arch_for_hparams_and_registry() -> None:
    a = A.arch_for_hparams(HP)
    assert a.name == "deepseek41"
    assert set(A.ARCHS) == {"deepseek41", "deepseek4", "qwen4exp"}
    assert A.CLASSES == ("dense", "experts", "spec_head.dense", "spec_head.experts")
    assert "engram" in a.classes  # sidecar classes extend the base four
    try:
        A.arch_for_hparams({"architectures": ["NopeForCausalLM"]})
        raise AssertionError("unknown arch accepted")
    except KeyError:
        pass


def test_classify_deepseek41_precedence() -> None:
    a = A.arch_for_hparams(HP)
    assert a.experts.scale_suffix == ".scale"  # fp8/mxfp4 source ships .scale
    # sidecar wins over everything
    assert A.classify(a, "layers.1.engram.embed.weight") == "engram"
    assert A.classify(a, "layers.14.engram.k_weight") == "engram"
    # spec-head experts beat the spec-head prefix
    assert A.classify(a, "mtp.0.ffn.experts.7.w1.weight") == "spec_head.experts"
    assert A.classify(a, "mtp.2.ffn.experts.127.w3.scale") == "spec_head.experts"
    # spec-head prefix -> spec_head.dense
    assert A.classify(a, "mtp.0.attn.wkv.weight") == "spec_head.dense"
    assert A.classify(a, "mtp.0.ffn.shared_experts.w1.weight") == "spec_head.dense"
    assert A.classify(a, "mtp.2.norm.weight") == "spec_head.dense"
    # main experts
    assert A.classify(a, "layers.5.ffn.experts.3.w2.weight") == "experts"
    assert A.classify(a, "layers.39.ffn.experts.383.w3.scale") == "experts"
    # everything else is dense
    assert A.classify(a, "embed.weight") == "dense"
    assert A.classify(a, "layers.5.ffn.shared_experts.w1.weight") == "dense"
    assert A.classify(a, "head.weight") == "dense"


def test_expert_tensor_names() -> None:
    a = A.arch_for_hparams(HP)
    d = A.expert_tensor_names(a, 7, 3)
    assert set(d) == {"gate", "up", "down"}
    assert d["gate"] == ("layers.7.ffn.experts.3.w1.weight", "layers.7.ffn.experts.3.w1.scale")
    assert d["up"] == ("layers.7.ffn.experts.3.w3.weight", "layers.7.ffn.experts.3.w3.scale")
    assert d["down"] == ("layers.7.ffn.experts.3.w2.weight", "layers.7.ffn.experts.3.w2.scale")
    assert A.expert_tensor_names_stage(a, 1, 0) == {
        "gate": ("mtp.1.ffn.experts.0.w1.weight", "mtp.1.ffn.experts.0.w1.scale"),
        "up": ("mtp.1.ffn.experts.0.w3.weight", "mtp.1.ffn.experts.0.w3.scale"),
        "down": ("mtp.1.ffn.experts.0.w2.weight", "mtp.1.ffn.experts.0.w2.scale"),
    }


def test_layer_ranges() -> None:
    a = A.arch_for_hparams(HP)
    assert A.expert_layers(a, HP) == list(range(40))
    assert A.spec_head_layers(a, HP) == [40, 41, 42]  # n_layer..n_layer+nextn-1
    assert A.spec_head_layers(a, {**HP, "num_nextn_predict_layers": 0}) == []
    assert A.spec_head_layers(a, {**HP, "num_nextn_predict_layers": 1}) == [40]


def test_other_arches() -> None:
    d4 = A.arch_for_hparams({"architectures": ["DeepseekV4ForCausalLM"]})
    assert d4.experts.name(3, 1, "gate") == (
        "layers.3.ffn.experts.1.w1.weight", "layers.3.ffn.experts.1.w1.scale")
    assert A.classify(d4, "layers.3.ffn.experts.1.w2.weight") == "experts"
    assert A.classify(d4, "layers.3.ffn.experts.1.w1.scale") == "experts"
    assert A.classify(d4, "mtp.0.ffn.experts.5.w3.weight") == "spec_head.experts"
    assert A.classify(d4, "mtp.0.attn.wkv.weight") == "spec_head.dense"
    assert A.classify(d4, "layers.3.ffn.gate.weight") == "dense"

    q = A.arch_for_hparams({"architectures": ["Qwen4ExpForCausalLM"]})
    assert A.classify(q, "model.layers.2.mlp.experts.9.gate_proj.weight") == "experts"
    assert A.classify(q, "model.layers.2.mlp.experts.9.down_proj.weight") == "experts"
    assert A.classify(q, "model.layers.2.mlp.experts.9.up_proj.weight") == "experts"
    # MTP block's own MoE (qwen4exp.py: mtp.layers.0.mlp.experts)
    assert A.classify(q, "mtp.layers.0.mlp.experts.5.gate_up_proj.weight") == "spec_head.experts"
    assert A.classify(q, "mtp.fc_hidden.weight") == "spec_head.dense"
    # PLE n-gram shard sidecar wins over dense
    assert A.classify(q, "model.layers.0.ngram_embedding.shard_1.weight") == "ple"
    assert A.spec_head_layers(q, {"num_hidden_layers": 48, "mtp_num_hidden_layers": 1}) == [48]
    assert q.converter_tolerates_missing_experts
