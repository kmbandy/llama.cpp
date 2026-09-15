"""Per-arch tensor classification table for wp-forge.

One class per HF tensor: "dense", "experts", "spec_head.dense", "spec_head.experts",
plus named sidecars (e.g. "engram"). Precedence: sidecar regex wins first, then
spec-head experts, then any spec-head prefix, then main experts, else dense.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExpertNaming:
    """How HF names routed experts: {prefix}.{layer}.{infix}.{eid}.{proj}.weight.

    layer is the absolute HF layer index, or the MTP stage for spec-head experts.
    proj maps ggml role -> HF projection name.
    """

    prefix: str
    infix: str
    proj: dict[str, str]
    scale_suffix: str | None = None

    @property
    def fmt(self) -> str:
        return f"{self.prefix}.{{layer}}.{self.infix}.{{eid}}.{{proj}}.weight"

    def name(self, layer: int, eid: int, role: str) -> tuple[str, str | None]:
        """(weight_name, scale_name) for one expert projection."""
        weight = self.fmt.format(layer=layer, eid=eid, proj=self.proj[role])
        scale = weight[:-len(".weight")] + self.scale_suffix if self.scale_suffix else None
        return weight, scale

    def _re(self) -> re.Pattern[str]:
        # any projection name (arches ship fused gate_up_proj under mtp blocks);
        # scale partner only for pre-quantized sources.
        suffix = r"(weight|scale)" if self.scale_suffix is not None else r"weight"
        return re.compile(
            rf"^{re.escape(self.prefix)}\.(\d+)\.{re.escape(self.infix)}\.\d+\.\w+\.{suffix}$"
        )


@dataclass(frozen=True)
class ArchSpec:
    name: str  # gguf arch string: "deepseek41"
    hf_architectures: tuple[str, ...]
    experts: ExpertNaming
    spec_head_experts: ExpertNaming | None = None  # MTP/nextn experts, or None
    sidecars: dict[str, str] = field(default_factory=dict)  # class -> regex on HF tensor name
    n_layer_key: str = "num_hidden_layers"
    nextn_key: str | None = "num_nextn_predict_layers"
    n_expert_key: str = "n_routed_experts"
    n_ff_exp_key: str = "moe_intermediate_size"
    hidden_key: str = "hidden_size"
    spec_head_n_expert_key: str | None = None
    converter_tolerates_missing_experts: bool = False
    # HF name prefix that marks a spec-head (MTP/nextn) tensor. Defaults to the
    # spec-head experts prefix; overridden when non-expert MTP tensors live under
    # a broader prefix (qwen4exp: experts under "mtp.layers", dense under "mtp").
    spec_head_prefix: str | None = None

    @property
    def classes(self) -> tuple[str, ...]:
        return (*CLASSES, *self.sidecars)

    def n_layer(self, hp: dict) -> int:
        return int(hp[self.n_layer_key])

    def nextn_count(self, hp: dict) -> int:
        if self.nextn_key is None or self.nextn_key not in hp:
            return 0
        return int(hp[self.nextn_key])


_CLS_BASE = ("dense", "experts", "spec_head.dense", "spec_head.experts")

_DSV41 = ArchSpec(
    name="deepseek41",
    hf_architectures=("DeepseekV41ForCausalLM",),
    experts=ExpertNaming(prefix="layers", infix="ffn.experts",
                         proj={"gate": "w1", "down": "w2", "up": "w3"}, scale_suffix=".scale"),
    spec_head_experts=ExpertNaming(prefix="mtp", infix="ffn.experts",
                                   proj={"gate": "w1", "down": "w2", "up": "w3"}, scale_suffix=".scale"),
    sidecars={"engram": r"^layers\.\d+\.engram\."},
    spec_head_n_expert_key="dspark_n_routed_experts",
)

_DSV4 = ArchSpec(
    name="deepseek4",
    hf_architectures=("DeepseekV4ForCausalLM",),
    # HF names are unprefixed like V4.1 (deepseek.py _write_mxfp4_expert_tensor:
    # layers.{bid}.ffn.experts.{eid}.{w1|w2|w3}.weight/.scale). The V4 converter
    # rekeys mtp.{s}.* -> layers.{num_hidden+s}.*, so spec-head experts live at
    # mtp.{s}.ffn.experts.{E}.w{1,2,3}.
    experts=ExpertNaming(prefix="layers", infix="ffn.experts",
                         proj={"gate": "w1", "down": "w2", "up": "w3"}, scale_suffix=".scale"),
    spec_head_experts=ExpertNaming(prefix="mtp", infix="ffn.experts",
                                   proj={"gate": "w1", "down": "w2", "up": "w3"}, scale_suffix=".scale"),
    spec_head_n_expert_key="dspark_n_routed_experts",
)

_QWEN4EXP = ArchSpec(
    name="qwen4exp",
    hf_architectures=("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"),
    experts=ExpertNaming(prefix="model.layers", infix="mlp.experts",
                         proj={"gate": "gate_proj", "down": "down_proj", "up": "up_proj"}),
    # mtp.layers.{i}.* remap to model.layers.{num_hidden+i}.* in the qwen converter;
    # the MTP block carries its own MoE (qwen4exp.py: "mtp.layers.0.mlp.experts").
    spec_head_experts=ExpertNaming(prefix="mtp.layers", infix="mlp.experts",
                                   proj={"gate": "gate_proj", "down": "down_proj", "up": "up_proj"}),
    n_expert_key="num_experts",
    nextn_key="mtp_num_hidden_layers",
    sidecars={"ple": r"\.ngram_embedding\.shard_\d+\."},
    spec_head_prefix="mtp",  # dense MTP tensors (fc_hidden, ...) sit directly under mtp.
    converter_tolerates_missing_experts=True,
)

ARCHS: dict[str, ArchSpec] = {a.name: a for a in (_DSV41, _DSV4, _QWEN4EXP)}
CLASSES: tuple[str, ...] = _CLS_BASE  # base classes; sidecar names extend per spec


def arch_for_hparams(hp: dict) -> ArchSpec:
    """Resolve the arch spec from hp["architectures"][0]."""
    arch = hp.get("architectures", [None])[0]
    for spec in ARCHS.values():
        if arch in spec.hf_architectures:
            return spec
    raise KeyError(f"unknown HF architecture {arch!r}")


def classify(a: ArchSpec, tensor_name: str) -> str:
    """Exactly one class per HF tensor name.

    Precedence: sidecar regex -> spec-head experts -> spec-head prefix ->
    main experts -> dense.
    """
    for cls, pat in a.sidecars.items():
        if re.search(pat, tensor_name):
            return cls
    if a.spec_head_experts is not None and a.spec_head_experts._re().match(tensor_name):
        return "spec_head.experts"
    sh_prefix = a.spec_head_prefix or (a.spec_head_experts.prefix if a.spec_head_experts else None)
    if sh_prefix is not None and tensor_name.startswith(sh_prefix + "."):
        return "spec_head.dense"
    if a.experts._re().match(tensor_name):
        return "experts"
    return "dense"


def expert_layers(a: ArchSpec, hp: dict) -> list[int]:
    """Main-stack layers that carry routed experts (all of them for current entries)."""
    return list(range(a.n_layer(hp)))


def spec_head_layers(a: ArchSpec, hp: dict) -> list[int]:
    """GGUF absolute layers for the spec head: [n_layer .. n_layer+nextn-1], or []."""
    n = a.nextn_count(hp)
    return list(range(a.n_layer(hp), a.n_layer(hp) + n)) if n else []


def expert_tensor_names(a: ArchSpec, layer: int, eid: int) -> dict[str, tuple[str, str | None]]:
    """Main-stack routed experts, indexed by absolute layer.

    {role: (weight_name, scale_name|None)} for roles gate/up/down.
    """
    return {role: a.experts.name(layer, eid, role) for role in ("gate", "up", "down")}


def expert_tensor_names_stage(a: ArchSpec, stage: int, eid: int) -> dict[str, tuple[str, str | None]]:
    """Spec-head experts. HF indexes them by STAGE ("mtp.{stage}"), GGUF by absolute
    layer n_layer+stage."""
    assert a.spec_head_experts is not None
    return {role: a.spec_head_experts.name(stage, eid, role) for role in ("gate", "up", "down")}
