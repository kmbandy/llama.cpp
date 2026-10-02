"""Per-tensor quant policy for wp-forge's convert path.

A plan's ``quant:`` block in convert mode is

    quant:
      default: ml8_4            # every tensor the converter would quantize
      rules:                    # first match wins; regex searched in the GGUF name
        - {match: '^token_embd\\.weight$', type: q8_0}
        - {match: '^output\\.weight$',     type: drop}
      imatrix: /path/imatrix.gguf   # optional, for types that need or use one

Types:
  keep            the converter's own choice (F32 for norms, BF16/F16 otherwise)
  drop            the tensor is not written at all
  <ggml type>     any type ggml_quantize_chunk produces (ggml_quant.QUANTIZABLE):
                  f16, bf16, q8_0, q4_k, iq4_xs, ...
  ml8_4 / ml8_3   rotated codebook 4-bit; ml8_3 fits an 8-entry codebook and
                  stores it as ML8_4 (ml8.py)
  ml8_fp8         rotated per-32 scaled e4m3

``default`` applies only to tensors the converter itself would quantize: 2+
dims and not forced to F32 (norms, gate_inp, ssm_conv1d, pos embeddings, ...
-- see conversion/base.py prepare_tensors). A rule applies to whatever it
matches, so a rule can still force one of those.

When a type cannot be stored for a tensor (row length not a multiple of the
type's block, an ml8 role the runtime has no rotation for), the tensor falls
back, in order, to q8_0 and then keep; every fallback is reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import ggml_quant

ML8_TYPES = {"ml8_4": 16, "ml8_3": 8, "ml8_fp8": None}
ML8_BLOCK = {"ml8_4": 64, "ml8_3": 64, "ml8_fp8": 32}
SPECIAL = ("keep", "drop")


class RulesError(ValueError):
    """One-line quant-rules failure naming the field."""


def valid_type(t: str) -> bool:
    return t in SPECIAL or t in ML8_TYPES or ggml_quant.is_quantizable(t)


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    type: str

    @property
    def match(self) -> str:
        return self.pattern.pattern


@dataclass
class QuantRules:
    default: str = "keep"
    rules: list[Rule] = field(default_factory=list)
    imatrix: str | None = None

    @staticmethod
    def parse(raw: object) -> "QuantRules":
        if raw is None:
            return QuantRules()
        if not isinstance(raw, dict):
            raise RulesError("quant: want a mapping {default, rules, imatrix}")
        unknown = set(raw) - {"default", "rules", "imatrix"}
        if unknown:
            raise RulesError(f"quant: unknown key(s) {sorted(unknown)} (convert mode takes default, rules, imatrix)")
        default = str(raw.get("default", "keep")).lower()
        if not valid_type(default):
            raise RulesError(f"quant.default: unknown type {default!r}")
        if default == "drop":
            raise RulesError("quant.default: 'drop' would drop every weight")
        rules: list[Rule] = []
        for i, r in enumerate(raw.get("rules") or []):
            if not isinstance(r, dict) or "match" not in r or "type" not in r:
                raise RulesError(f"quant.rules[{i}]: want {{match: <regex>, type: <type>}}")
            t = str(r["type"]).lower()
            if not valid_type(t):
                raise RulesError(f"quant.rules[{i}]: unknown type {t!r}")
            try:
                pat = re.compile(str(r["match"]))
            except re.error as e:
                raise RulesError(f"quant.rules[{i}]: bad regex {r['match']!r}: {e}") from e
            rules.append(Rule(pat, t))
        im = raw.get("imatrix")
        return QuantRules(default, rules, str(im) if im else None)

    def types(self) -> set[str]:
        return {self.default, *(r.type for r in self.rules)}

    def uses_ml8(self) -> bool:
        return any(t in ML8_TYPES for t in self.types())

    def decide(self, name: str, n_dims: int, converter_is_f32: bool) -> tuple[str, str | None]:
        """(type, rule_match) for one GGUF tensor; rule_match is None when the
        default (or the keep-for-unquantized fallthrough) decided."""
        for r in self.rules:
            if r.pattern.search(name):
                return r.type, r.match
        if n_dims < 2 or converter_is_f32:
            return "keep", None
        return self.default, None


def storable(t: str, n_per_row: int) -> bool:
    """Can type t hold a row of n_per_row elements?"""
    if t in SPECIAL:
        return True
    if t in ML8_TYPES:
        return n_per_row % ML8_BLOCK[t] == 0
    return n_per_row % ggml_quant.block_size(t) == 0
