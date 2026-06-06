"""Render an eval corpus through the model's chat template for in-regime PPL (Set C).

Three modes, mirroring how calibration treats the same content:
  --mode chat     : a [HUMAN]/[GPT] eval .txt → split into records → real role turns.
  --mode doc      : a raw prose .txt (e.g. wiki.test.raw) → ~chunk-char user-turn docs.
  --mode messages : a messages-jsonl (one {"messages":[…]} per line, tool turns and
                    all) → each record rendered through the template verbatim. This is
                    the agentic held-out path: evaluate the deployment regime in the
                    same envelope the agentic calibration draws were rendered in.

Uses the SAME source content as the raw (Set A/B) evals, so the only difference vs
those baselines is the chat-template envelope — keeping gap-to-UD apples-to-apples
within a regime. Output is a plain .txt for `llama-perplexity -f`.
"""
import argparse
import json
import re

from transformers import AutoTokenizer

import chat_format as cf

_HUMAN_SPLIT = re.compile(r"(?=^\[HUMAN\]:)", re.MULTILINE)


def render_chat(in_path, tokenizer, source):
    """Split a [HUMAN]/[GPT] eval file into records, render each through the template."""
    text = open(in_path, encoding="utf-8", errors="ignore").read()
    records = [r for r in _HUMAN_SPLIT.split(text) if r.strip()]
    out = []
    for rec in records:
        msgs = cf.parse_to_messages(rec, source=source)
        if msgs:
            out.append(cf.render_text(msgs, tokenizer))
    return out


def render_doc(in_path, tokenizer, chunk_chars):
    """Chunk a raw prose file into ~chunk_chars windows, each a user-turn document."""
    text = open(in_path, encoding="utf-8", errors="ignore").read()
    out, i, n = [], 0, len(text)
    while i < n:
        chunk = text[i:i + chunk_chars].strip()
        i += chunk_chars
        if not chunk:
            continue
        msgs = cf.parse_to_messages(chunk, source="wikipedia")  # doc → user turn
        out.append(cf.render_text(msgs, tokenizer))
    return out


def render_messages(in_path, tokenizer):
    """Render a messages-jsonl (one {"messages":[…]} per line) through the template.

    Each record's pre-built messages — tool_calls, role:"tool" results and all — go
    straight to chat_format.render_text, the exact path the agentic calibration draws
    take, so eval and calibration share one rendering. Records with no messages skip.
    """
    out = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("messages") or []
            if msgs:
                out.append(cf.render_text(msgs, tokenizer))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["chat", "doc", "messages"], required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--source", default="math_se",
                    help="conversational source name for chat mode (role mapping is identical)")
    ap.add_argument("--chunk-chars", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=1_500_000,
                    help="cap total OUTPUT chars to keep PPL runtime comparable")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    if a.mode == "chat":
        pieces = render_chat(a.in_path, tok, a.source)
    elif a.mode == "messages":
        pieces = render_messages(a.in_path, tok)
    else:
        pieces = render_doc(a.in_path, tok, a.chunk_chars)

    written = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pieces:
            f.write(p)
            if not p.endswith("\n"):
                f.write("\n")
            written += len(p)
            if written >= a.max_chars:
                break
    print(f"[template-eval] {a.mode}: {a.in_path} → {a.out} "
          f"({len(pieces)} pieces, {written} chars)")


if __name__ == "__main__":
    main()
