# A7. Auto-default `--kv-tier-paged-blocks` for hybrid models

**Status**: Accepted (2026-05-10)
**Decided in**: [MAD-126](https://mad-lab-ai.atlassian.net/browse/MAD-126)
**Implemented in**: MAD-134 (config ergonomics)

## Context

A hybrid model running with `--kv-tiered` and without
`--kv-tier-paged-blocks` falls into the legacy non-paged tiered
codepath. After [A1](A1-hybrid-paged-primary.md), that path is
neither maintained nor recommended for hybrid models. Most operators
who set `--kv-tiered` on a hybrid model **mean** to get paged + tiered
— forgetting `--kv-tier-paged-blocks` produces a worse cache
silently.

Three options for the default behavior:

1. **Keep current default off**. Operators who want paged must
   remember the flag. Bad ergonomics; quiet performance regressions.

2. **Default on for everyone (hybrid AND pure-attention)**. Risks
   regressing pure-attention configurations that worked fine on the
   non-paged path.

3. **Default on for hybrid, leave pure-attention as-is**. Targets
   the actual user — hybrid is what people actually run with
   tiering.

## Decision

Option 3. When the model is hybrid AND `--kv-tiered` is set AND
`--kv-tier-paged-blocks` was NOT explicitly disabled, default-on.
Operators can opt out via `--no-kv-tier-paged-blocks` for
backwards-compat.

The implementation uses a tristate-via-explicit-bool pattern in
`common_params`: a `bool kv_tier_paged_blocks_explicit` records
whether the user actually mentioned the flag. The default-on logic
runs in `common_context_params_to_llama` and only fires when the user
hasn't expressed an opinion either way.

## Consequences

**Positive**:
- Operator running the army-goal config doesn't need to remember the
  paged flag. `--kv-tiered 25,75,0` on a hybrid model just works.
- Pure-attention deployments are untouched — no regression risk for
  existing configurations that don't expect paged.
- The opt-out path (`--no-kv-tier-paged-blocks`) is still there for
  edge cases or A/B comparisons.

**Negative**:
- Adds one more piece of "automatic" behavior that operators have to
  read about to fully understand. Mitigated by the boot-time log
  line that prints the resolved configuration (whether paged is on
  and why).

**Neutral**:
- Anyone who explicitly set `--no-kv-tier-paged-blocks` keeps that
  intent across all hybrid configurations. The default-on only
  applies when the user hasn't expressed an opinion.
