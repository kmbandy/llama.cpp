"""Render calibration corpora through the target model's OWN chat template.

Two layers, so this generalizes to any model with a chat template baked into its
tokenizer (Qwen `<|im_start|>`, Gemma `<start_of_turn>`, Llama headers, …) with
ZERO per-model code here:

  1. parse_to_messages(text, source) — corpus record → canonical
     [{"role","content"}, …]. The only source-specific logic. Knows nothing
     about which model will render it.
  2. render_ids(messages, tokenizer, seq_len) — messages → token ids via
     tokenizer.apply_chat_template. The tokenizer owns the per-model template,
     so swapping --model swaps the rendering for free.

Background: the cleaned SE/tool corpora carry custom `[HUMAN]/[GPT]/[SYSTEM]`
markers, NOT the model's real chat template, and the old loader tokenized that
raw — so calibration never saw the model's structural tokens (`<|im_start|>`,
Qwen's `<think>` block, …) that the deployed chat model actually operates in.
This module closes that regime gap. (Decision 2026-06-05: document sources are
wrapped as a single user-turn "context" message — RAG/agentic-faithful.)
"""
import re

# Marker grammar of the cleaned corpus (see calib_corpus._SRC). HUMAN/GPT/SYSTEM
# are the only markers present across stackoverflow/math_se/softwareeng_se/rpg_se
# (HUMAN→GPT pairs) and tool_calls (SYSTEM-only).
_MARKER_RE = re.compile(r"\[(HUMAN|GPT|SYSTEM)\]:\s?")
_ROLE = {"HUMAN": "user", "GPT": "assistant", "SYSTEM": "system"}

# Sources whose records are raw document bodies (no conversational structure).
# Wrapped as one user-turn context message rather than parsed for markers.
DOC_SOURCES = frozenset({"wikipedia", "arxiv", "fineweb"})


def parse_to_messages(text, source=None):
    """Raw corpus record → canonical chat messages [{"role","content"}, …].

    - DOC_SOURCES → one user turn carrying the whole document.
    - marker text → split on [HUMAN]/[GPT]/[SYSTEM] into role turns.
    - markerless / unknown → one user turn (safe fallback).
    Empty/whitespace → [].
    """
    text = (text or "").strip()
    if not text:
        return []
    if source in DOC_SOURCES:
        return [{"role": "user", "content": text}]

    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]

    msgs = []
    for i, m in enumerate(matches):
        role = _ROLE[m.group(1)]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


def render_ids(messages, tokenizer, seq_len):
    """messages → list[int] token ids via the model's chat template.

    Render to the template STRING first (it carries all of the model's special
    tokens as text), then encode with add_special_tokens=False so the tokenizer
    does NOT prepend an extra BOS on top of the template's own. Stable across
    transformers versions (tokenize=True can return an Encoding object).
    Tail-truncated to seq_len to match the old loader's truncation semantics.
    """
    if not messages:
        return []
    text = render_text(messages, tokenizer)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids[:seq_len])


def render_text(messages, tokenizer):
    """messages → string render via the model's chat template.

    Some templates reject role patterns that lack a user turn (Qwen raises
    "No user query found" on a system-only record, e.g. the tool_calls source).
    Fall back to folding all content into a single user turn so the record still
    lands in-template — robust across templates with stricter role rules.
    """
    if not messages:
        return ""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        merged = "\n\n".join(m["content"] for m in messages if m.get("content"))
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            tokenize=False, add_generation_prompt=False,
        )
