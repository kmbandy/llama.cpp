"""wp-forge plan: schema, YAML loading, and resolution against arch + machines.

A Plan is what a human writes; a ResolvedPlan is what stages and the dispatch
contract consume. resolve() is what the CLI --dry-run prints and what a
console pane will later validate against, so every failure is one PlanError,
one line, naming the offending field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .arch import ArchSpec, arch_for_hparams, expert_layers, spec_head_layers
from .machines import expand_remote_home, Machine

# ggml types gguf-py can produce AND llama-wp-repack can slice; the value is
# the block size (elements per quant block) used to align expert widths.
QUANT_BLOCK = {
    "f16": 1, "bf16": 1, "q8_0": 32, "q4_0": 32, "q4_1": 32, "q5_0": 32, "q5_1": 32,
    "q2_k": 256, "q3_k": 256, "q4_k": 256, "q5_k": 256, "q6_k": 256,
    "mxfp4": 32, "iq4_nl": 32, "iq4_xs": 256, "iq3_xxs": 256, "iq2_xs": 256,
}
SLICE_ALIGNMENT = 32  # wp-repack's slice_alignment floor

# ml8 quant types: data-free ROTATED ML8_FP8 / ML8_4 (conversion/wp_forge/ml8.py).
# Allowed ONLY for the "experts" / "spec_head.experts" classes (dense/sidecar
# classes keep QUANT_BLOCK's whitelist unchanged) -- see resolve()'s per-class
# allowed-set selection. Block sizes match GGML_QUANT_SIZES: ML8_FP8 (32, 34),
# ML8_4 (64, 36).
ML8_QUANT_BLOCK = {"ml8_fp8": 32, "ml8_4": 64}
EXPERTS_CLASSES = ("experts", "spec_head.experts")
ML8_ROTATIONS = ("kronecker", "block_hadamard", "none")


@dataclass(frozen=True)
class Ml8Settings:
    """Resolved conversion/wp_forge/plan.yaml `ml8:` block (defaults match the
    contract: kronecker rotation, seed 0, max_b 1024, fit_rows 65536)."""
    rotation: str = "kronecker"
    rotation_seed: int = 0
    max_b: int = 1024
    fit_rows: int = 65536


class PlanError(ValueError):
    """One-line plan failure; the reason names the offending field."""


@dataclass(frozen=True)
class LayerRange:
    first: int  # inclusive
    last: int   # inclusive

    @staticmethod
    def parse(s: object) -> "LayerRange":
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(s))
        if not m or int(m[1]) > int(m[2]):
            raise PlanError(f"layers: bad range {s!r} (want 'a-b' with a<=b)")
        return LayerRange(int(m[1]), int(m[2]))

    def __str__(self) -> str:
        return f"{self.first}-{self.last}"

    def layers(self) -> list[int]:
        return list(range(self.first, self.last + 1))


@dataclass
class Placement:
    machine: str
    path: str | None  # dir override on the machine; None -> machine's models_dir


@dataclass
class StageSpec:
    layers: LayerRange
    machine: str
    path: str | None
    widths: list[int] | str | None  # explicit element widths, or "a:b" ratios
    # one machine per width slice; None keeps every slice on `machine`
    slice_machines: list[str] | None = None
    # whole-expert id ranges, "first-last", one per output set. Not an FFN cut.
    expert_ranges: list[tuple[int, int]] | None = None


@dataclass
class Plan:
    name: str
    source: str  # "hf:<repo>" or "gguf:<path>"
    quant: dict[str, str]
    machines_path: Path | None
    spine: Placement
    sidecars: Placement
    experts: list[StageSpec]
    spec_head: StageSpec | str | None  # StageSpec, "none", or None (unspecified)
    allow_partial: bool = False
    experts_only: bool = False
    # existing spine gguf to point the bundle at when experts_only skips the rebuild
    spine_file: str | None = None
    # raw `ml8:` block from the yaml (validated + defaulted into Ml8Settings in resolve())
    ml8: dict | None = None


@dataclass(frozen=True)
class SetSpec:
    id: str  # "L26-39" | "L0-25-w0"
    role: str  # "experts" | "spec_head"
    layers: LayerRange
    slice_index: int | None
    widths: list[int] | None  # resolved element widths, None if unsliced
    machine: str
    dir: str  # absolute dir on the machine: <models_dir>/<name>/<id>
    output_base: str  # blob prefix: "<name>-<id>"
    quant: str  # ggml type name for the experts of this set ("src" = verbatim)
    est_bytes: int
    expert_first: int | None = None
    expert_last: int | None = None
    ml8: "Ml8Settings | None" = None  # resolved ml8: block, only when quant is ml8_fp8/ml8_4


@dataclass
class ResolvedPlan:
    plan: Plan
    arch: ArchSpec
    hparams: dict
    machines: dict[str, Machine]
    spine_path: str
    sidecar_paths: dict[str, str]
    sets: list[SetSpec]
    bundle_dir: str  # <models_dir>/<name> on the spine machine

    def dispatch_order(self) -> list[str]:
        return [s.id for s in self.sets]


# --- YAML loading: shape errors only; semantic checks live in resolve() ---

def _placement(v: object, default: Placement | None, what: str) -> Placement:
    if v is None:
        if default is None:
            raise PlanError(f"{what}: required")
        return Placement(default.machine, default.path)
    if isinstance(v, str):
        return Placement(v, None)
    if isinstance(v, dict) and "machine" in v:
        return Placement(v["machine"], v.get("path"))
    raise PlanError(f"{what}: want a machine name or {{machine, path}}")


def _stage(v: dict, what: str) -> StageSpec:
    if not isinstance(v, dict) or "layers" not in v or "machine" not in v:
        raise PlanError(f"{what}: stage needs 'layers' and 'machine'")
    raw_slices = v.get("slice_machines")
    if raw_slices is not None and (not isinstance(raw_slices, list) or not all(isinstance(x, str) for x in raw_slices)):
        raise PlanError(f"{what}: slice_machines must be a list of machine names")
    raw_ranges = v.get("expert_ranges")
    ranges = None
    if raw_ranges is not None:
        if not isinstance(raw_ranges, list):
            raise PlanError(f"{what}: expert_ranges must be a list of 'first-last'")
        ranges = []
        for item in raw_ranges:
            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(item))
            if not m or int(m[1]) > int(m[2]):
                raise PlanError(f"{what}: bad expert range {item!r}")
            ranges.append((int(m[1]), int(m[2])))
    return StageSpec(
        LayerRange.parse(v["layers"]), v["machine"], v.get("path"), v.get("widths"), raw_slices, ranges
    )


def load_plan(path: Path) -> Plan:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PlanError("plan yaml: top level must be a mapping")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(raw.get("name", ""))):
        raise PlanError("name: required, [A-Za-z0-9._-]+")
    src = str(raw.get("source", ""))
    if not (src.startswith("hf:") or src.startswith("gguf:")):
        raise PlanError("source: must be hf:<repo> or gguf:<path>")
    spine = _placement(raw.get("spine"), None, "spine")
    sh = raw.get("spec_head")
    raw_ml8 = raw.get("ml8")
    if raw_ml8 is not None and not isinstance(raw_ml8, dict):
        raise PlanError("ml8: must be a mapping")
    return Plan(
        name=raw["name"],
        source=src,
        quant={str(k): str(v).lower() for k, v in (raw.get("quant") or {}).items()},
        machines_path=Path(raw["machines"]).expanduser() if raw.get("machines") else None,
        spine=spine,
        sidecars=_placement(raw.get("sidecars"), spine, "sidecars"),
        experts=[_stage(s, f"experts[{i}]") for i, s in enumerate(raw.get("experts") or [])],
        spec_head="none" if sh == "none" else (_stage(sh, "spec_head") if sh else None),
        allow_partial=bool(raw.get("allow_partial", False)),
        experts_only=bool(raw.get("experts_only", False)),
        spine_file=str(raw["spine_file"]) if raw.get("spine_file") else None,
        ml8=raw_ml8,
    )


# --- resolution ---

def parse_widths(spec: list[int] | str, n_ff_exp: int, block: int) -> list[int]:
    """Explicit element widths or "a:b[:c]" ratios, solved against n_ff_exp.

    Result must tile n_ff_exp exactly and be block-aligned (block = the
    quant's block size, floored at SLICE_ALIGNMENT).
    """
    if isinstance(spec, str):
        try:
            ratios = [int(x) for x in spec.split(":")]
        except ValueError:
            raise PlanError(f"widths: ratio {spec!r} not ints (want 'a:b[:c...]')")
        if len(ratios) < 2 or any(r <= 0 for r in ratios):
            raise PlanError(f"widths: ratio {spec!r} (want 'a:b[:c...]' positive ints)")
        units = n_ff_exp // block
        if units == 0:
            raise PlanError(f"widths: n_ff_exp {n_ff_exp} not a multiple of block {block}")
        alloc = [units * r // sum(ratios) for r in ratios]
        alloc[0] += units - sum(alloc)  # hand the remainder to slice 0, stays aligned
        widths = [a * block for a in alloc]
    else:
        widths = [int(w) for w in spec]
    if any(w <= 0 for w in widths):
        raise PlanError(f"widths {widths}: every entry must be positive")
    if sum(widths) != n_ff_exp:
        raise PlanError(f"widths {widths} sum {sum(widths)} != n_ff_exp {n_ff_exp}")
    if any(w % block for w in widths):
        raise PlanError(f"widths {widths}: every entry must be a multiple of {block} (quant block size)")
    return widths


def _check_tiling(stages: list[StageSpec], required: list[int], allow_partial: bool, what: str) -> None:
    seen: dict[int, LayerRange] = {}
    for st in stages:
        for l in st.layers.layers():
            if l in seen:
                raise PlanError(f"{what}: layer {l} overlap between {seen[l]} and {st.layers}")
            seen[l] = st.layers
    extra = sorted(set(seen) - set(required))
    if extra:
        raise PlanError(f"{what}: layers {extra} are not {what} layers of this model")
    if not allow_partial:
        missing = sorted(set(required) - set(seen))
        if missing:
            raise PlanError(f"{what}: gap, layers {missing} not covered (set allow_partial: true to mean it)")


def _dest_dir(m: Machine, override: str | None, name: str, set_id: str | None) -> str:
    # absolute on the target machine: these paths end up in manifests and
    # per-blob indexes as model_files, which the descriptor tool and the worker
    # canonicalize verbatim (a literal "~" fails there)
    root = expand_remote_home(m, (override or m.models_dir).rstrip("/"))
    return f"{root}/{name}/{set_id}" if set_id else f"{root}/{name}"


def _expert_bytes(quant: str, n_expert: int, n_ff: int, n_embd: int, n_layers: int) -> int:
    """Routed-expert bytes for one quant: gate+up (n_embd x n_ff) + down (n_ff x n_embd)."""
    import gguf  # in-tree gguf-py, on sys.path via tests/wp_forge/conftest.py

    bs, ts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[quant.upper()]]
    row_up = (n_embd // bs) * ts
    row_down = (n_ff // bs) * ts
    return (2 * row_up * n_ff + row_down * n_embd) * n_expert * n_layers


def resolve(plan: Plan, hparams: dict, machines: dict[str, Machine], *, source_is_gguf: bool) -> ResolvedPlan:
    arch = arch_for_hparams(hparams)

    if source_is_gguf and plan.quant:
        raise PlanError("quant: gguf: sources are byte-verbatim, a quant: block is forbidden (llama-quantize first)")
    if not source_is_gguf and "experts" not in plan.quant:
        raise PlanError("quant: 'experts' is required for hf: sources (ggml type for the expert tensors)")
    for cls in plan.quant:
        if cls not in arch.classes:
            raise PlanError(f"quant: unknown class '{cls}' for {arch.name} (classes: {', '.join(arch.classes)})")
    for cls, t in plan.quant.items():
        allowed = {**QUANT_BLOCK, **ML8_QUANT_BLOCK} if cls in EXPERTS_CLASSES else QUANT_BLOCK
        if t not in allowed:
            raise PlanError(f"quant: type '{t}' for {cls} is not producible+sliceable (allowed: {', '.join(sorted(allowed))})")

    machine_names = {plan.spine.machine, plan.sidecars.machine}
    for st in plan.experts:
        machine_names.add(st.machine)
        machine_names.update(st.slice_machines or [])
    if isinstance(plan.spec_head, StageSpec):
        machine_names.add(plan.spec_head.machine)
        machine_names.update(plan.spec_head.slice_machines or [])
    for name in machine_names:
        if name not in machines:
            raise PlanError(f"machine '{name}' not in machines.json (have: {', '.join(sorted(machines))})")

    _check_tiling(plan.experts, expert_layers(arch, hparams), plan.allow_partial, "experts")
    sh_layers = spec_head_layers(arch, hparams)
    if sh_layers:
        if plan.spec_head is None:
            raise PlanError(f"spec_head: model has spec-head layers {sh_layers[0]}-{sh_layers[-1]}; place them or say spec_head: none")
        if plan.spec_head != "none":
            _check_tiling([plan.spec_head], sh_layers, False, "spec_head")
    elif isinstance(plan.spec_head, StageSpec):
        raise PlanError("spec_head: model has no spec-head layers")

    n_ff = int(hparams[arch.n_ff_exp_key])
    n_embd = int(hparams[arch.hidden_key])
    n_exp = int(hparams[arch.n_expert_key])
    q_exp = plan.quant.get("experts", "src")
    q_sh = plan.quant.get("spec_head.experts", q_exp)

    uses_ml8 = q_exp in ML8_QUANT_BLOCK or q_sh in ML8_QUANT_BLOCK
    if plan.ml8 is not None and not uses_ml8:
        raise PlanError("ml8: block present but no quant.experts/quant.spec_head.experts uses ml8_fp8/ml8_4")
    ml8_settings: Ml8Settings | None = None
    if uses_ml8:
        rd = plan.ml8 or {}
        rotation = str(rd.get("rotation", "kronecker")).lower()
        if rotation not in ML8_ROTATIONS:
            raise PlanError(f"ml8.rotation: {rotation!r} not one of {ML8_ROTATIONS}")
        try:
            rotation_seed = int(rd.get("rotation_seed", 0))
            max_b = int(rd.get("max_b", 1024))
            fit_rows = int(rd.get("fit_rows", 65536))
        except (TypeError, ValueError):
            raise PlanError("ml8: rotation_seed/max_b/fit_rows must be ints")
        if max_b < 1 or (max_b & (max_b - 1)) != 0:
            raise PlanError(f"ml8.max_b: must be a positive power of 2, got {max_b}")
        if fit_rows < 1:
            raise PlanError(f"ml8.fit_rows: must be positive, got {fit_rows}")
        ml8_settings = Ml8Settings(rotation, rotation_seed, max_b, fit_rows)

    sets: list[SetSpec] = []

    def emit(st: StageSpec, role: str, quant: str, n_expert: int) -> None:
        n_layers = len(st.layers.layers())
        block = max(SLICE_ALIGNMENT, {**QUANT_BLOCK, **ML8_QUANT_BLOCK}.get(quant, SLICE_ALIGNMENT))
        set_ml8 = ml8_settings if quant in ML8_QUANT_BLOCK else None
        if st.expert_ranges and st.widths:
            raise PlanError("expert_ranges and widths cannot both be set")
        if st.expert_ranges:
            covered = [False] * n_expert
            for first, last in st.expert_ranges:
                if first < 0 or last >= n_expert:
                    raise PlanError(f"expert range {first}-{last} outside 0-{n_expert - 1}")
                for eid in range(first, last + 1):
                    if covered[eid]:
                        raise PlanError(f"expert {eid} is in more than one expert_range")
                    covered[eid] = True
            if not all(covered):
                raise PlanError("expert_ranges do not cover every expert")
            if st.slice_machines is not None and len(st.slice_machines) != len(st.expert_ranges):
                raise PlanError(
                    f"slice_machines has {len(st.slice_machines)} entries, want {len(st.expert_ranges)}"
                )
            full = 0 if quant == "src" else _expert_bytes(quant, n_expert, n_ff, n_embd, n_layers)
            for i, (first, last) in enumerate(st.expert_ranges):
                machine_name = st.slice_machines[i] if st.slice_machines else st.machine
                m = machines[machine_name]
                sid = f"L{st.layers}-e{i}"
                count = last - first + 1
                est = 0 if quant == "src" else full * count // n_expert
                sets.append(SetSpec(
                    sid, role, st.layers, None, None,
                    machine_name, _dest_dir(m, st.path, plan.name, sid),
                    f"{plan.name}-{sid}", quant, est, first, last, set_ml8))
            return
        widths = parse_widths(st.widths, n_ff, block) if st.widths else None
        if widths is not None and set_ml8 is not None and set_ml8.rotation != "none":
            raise PlanError(
                f"quant '{quant}' with ml8.rotation '{set_ml8.rotation}' cannot be used on "
                f"FFN width-sliced sets (widths {st.widths}) -- the down rotation would span slices"
            )
        n_slices = len(widths) if widths else 1
        if st.slice_machines is not None and len(st.slice_machines) != n_slices:
            raise PlanError(
                f"slice_machines {st.slice_machines} has {len(st.slice_machines)} entries, want {n_slices}"
            )
        for i, w in enumerate(widths or [n_ff]):
            machine_name = st.slice_machines[i] if st.slice_machines else st.machine
            m = machines[machine_name]
            sid = f"L{st.layers}-w{i}" if widths else f"L{st.layers}"
            est = 0 if quant == "src" else _expert_bytes(quant, n_expert, w, n_embd, n_layers)
            sets.append(SetSpec(
                sid, role, st.layers, i if widths else None, widths,
                machine_name, _dest_dir(m, st.path, plan.name, sid),
                f"{plan.name}-{sid}", quant, est, None, None, set_ml8))

    for st in plan.experts:
        emit(st, "experts", q_exp, n_exp)
    if isinstance(plan.spec_head, StageSpec):
        n_sh = int(hparams.get(arch.spec_head_n_expert_key or "", n_exp)) if arch.spec_head_n_expert_key else n_exp
        emit(plan.spec_head, "spec_head", q_sh, n_sh)

    sm = machines[plan.spine.machine]
    bundle_dir = _dest_dir(sm, plan.spine.path, plan.name, None)
    spine_path = plan.spine_file or f"{bundle_dir}/{plan.name}-spine.gguf"
    scm = machines[plan.sidecars.machine]
    sc_dir = _dest_dir(scm, plan.sidecars.path, plan.name, None)
    sidecar_paths = {cls: f"{sc_dir}/{plan.name}-{cls}.gguf" for cls in arch.sidecars}
    return ResolvedPlan(plan, arch, hparams, machines, spine_path, sidecar_paths, sets, bundle_dir)


# --- convert mode: any model -> one GGUF (convert.py) ---
#
# A plan with no `experts:` block is a convert plan:
#   name: clef-ml8-4
#   source: hf:Cloudflare/clef        # or dir:<path> / gguf:<path>
#   output: {machine: mad-lab-main, path: /mnt/nvme/models}   # file: <path>/<name>.gguf
#   quant: {default: ml8_4, rules: [{match: ..., type: ...}], imatrix: ...}
#   ml8: {rotation_seed: 0, max_b: 1024, fit_rows: 65536, local_b: 128, rotation: kronecker}
#   converter: {mtp: auto, fuse_qkv: false, fuse_gate_up_exps: false, fp8_as_q8: false}

@dataclass
class ConvertPlan:
    name: str
    source: str
    machines_path: Path | None
    output: Placement
    quant: dict
    ml8: dict
    converter: dict


@dataclass
class ResolvedConvert:
    plan: ConvertPlan
    machine: Machine
    out_path: str  # absolute on the output machine


def is_convert_plan(path: Path) -> bool:
    raw = yaml.safe_load(Path(path).read_text())
    return isinstance(raw, dict) and "experts" not in raw


def load_convert_plan(path: Path) -> ConvertPlan:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PlanError("plan yaml: top level must be a mapping")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(raw.get("name", ""))):
        raise PlanError("name: required, [A-Za-z0-9._-]+")
    src = str(raw.get("source", ""))
    if not src.startswith(("hf:", "dir:", "gguf:")):
        raise PlanError("source: must be hf:<repo>, dir:<path> or gguf:<path>")
    unknown = set(raw) - {"name", "source", "machines", "output", "quant", "ml8", "converter"}
    if unknown:
        raise PlanError(f"convert plan: unknown key(s) {sorted(unknown)}")
    ml8 = raw.get("ml8") or {}
    if not isinstance(ml8, dict):
        raise PlanError("ml8: must be a mapping")
    return ConvertPlan(
        name=raw["name"], source=src,
        machines_path=Path(raw["machines"]).expanduser() if raw.get("machines") else None,
        output=_placement(raw.get("output"), None, "output"),
        quant=raw.get("quant") or {}, ml8=ml8, converter=_converter_opts(raw.get("converter")),
    )


def _converter_opts(v: object) -> dict:
    from .convert import CONVERTER_OPTS
    if v is None:
        return {}
    if not isinstance(v, dict) or set(v) - set(CONVERTER_OPTS):
        raise PlanError(f"converter: want a mapping with keys from {CONVERTER_OPTS}")
    if "mtp" in v and str(v["mtp"]).lower() not in ("auto", "true", "false"):
        raise PlanError("converter.mtp: auto, true or false")
    return dict(v)


def resolve_convert(plan: ConvertPlan, machines: dict[str, Machine]):
    """-> (ResolvedConvert, QuantRules, Ml8Opts)."""
    from .convert import Ml8Opts
    from .rules import QuantRules, RulesError

    if plan.output.machine not in machines:
        raise PlanError(f"machine '{plan.output.machine}' not in machines.json (have: {', '.join(sorted(machines))})")
    try:
        rules = QuantRules.parse(plan.quant)
    except RulesError as e:
        raise PlanError(str(e)) from e
    rd = plan.ml8
    unknown = set(rd) - {"rotation", "rotation_seed", "max_b", "fit_rows", "local_b"}
    if unknown:
        raise PlanError(f"ml8: unknown key(s) {sorted(unknown)}")
    rotation = str(rd.get("rotation", "kronecker")).lower()
    if rotation not in ML8_ROTATIONS:
        raise PlanError(f"ml8.rotation: {rotation!r} not one of {ML8_ROTATIONS}")
    try:
        opts = Ml8Opts(rotation, int(rd.get("rotation_seed", 0)), int(rd.get("max_b", 1024)),
                       int(rd.get("fit_rows", 65536)), int(rd.get("local_b", 128)))
    except (TypeError, ValueError):
        raise PlanError("ml8: rotation_seed/max_b/fit_rows/local_b must be ints")
    if opts.max_b < 1 or opts.max_b & (opts.max_b - 1):
        raise PlanError(f"ml8.max_b: must be a positive power of 2, got {opts.max_b}")
    m = machines[plan.output.machine]
    root = expand_remote_home(m, (plan.output.path or m.models_dir).rstrip("/"))
    return ResolvedConvert(plan, m, f"{root}/{plan.name}.gguf"), rules, opts
