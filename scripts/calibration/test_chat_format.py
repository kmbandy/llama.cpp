"""Tests for chat_format — corpus → canonical messages → model chat template.

Parser tests are tokenizer-free (pure logic). One integration test exercises the
real Qwen3.5 tokenizer to prove the render path produces the in-template token
stream with no double special tokens.
"""
import os
import pytest

import chat_format as cf

QWEN_TOK_DIR = "/home/kmbandy/models/Qwen3.5-0.8B-hf"


# ---------------- parse_to_messages (pure) ----------------

def test_human_gpt_pair():
    text = "[HUMAN]: What is 2+2?\n\n[GPT]: It is 4."
    msgs = cf.parse_to_messages(text, source="math_se")
    assert msgs == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
    ]


def test_system_only_tool_calls():
    text = "[SYSTEM]: You are an assistant.\n\nAvailable tools:\nfoo()"
    msgs = cf.parse_to_messages(text, source="tool_calls")
    assert msgs == [
        {"role": "system", "content": "You are an assistant.\n\nAvailable tools:\nfoo()"},
    ]


def test_multi_turn_alternation():
    text = "[HUMAN]: a\n[GPT]: b\n[HUMAN]: c\n[GPT]: d"
    msgs = cf.parse_to_messages(text, source="rpg_se")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in msgs] == ["a", "b", "c", "d"]


def test_document_source_wrapped_as_user():
    # Even if a doc accidentally contains a bracket marker, doc sources are one user turn.
    text = "The mitochondria [HUMAN]: is the powerhouse of the cell."
    msgs = cf.parse_to_messages(text, source="wikipedia")
    assert msgs == [{"role": "user", "content": text.strip()}]


def test_markerless_nonchat_falls_back_to_user():
    msgs = cf.parse_to_messages("just some raw text", source="unknown_src")
    assert msgs == [{"role": "user", "content": "just some raw text"}]


def test_empty_returns_empty():
    assert cf.parse_to_messages("", source="math_se") == []
    assert cf.parse_to_messages("   ", source="wikipedia") == []


def test_leading_preamble_before_first_marker_is_dropped():
    # SE records start at [HUMAN]:, but guard the rare preamble case: no orphan turn.
    text = "garbage preamble\n[HUMAN]: real question\n[GPT]: real answer"
    msgs = cf.parse_to_messages(text, source="stackoverflow")
    assert msgs[0] == {"role": "user", "content": "real question"}
    assert len(msgs) == 2


# ---------------- render (needs the real tokenizer) ----------------

@pytest.fixture(scope="module")
def qwen_tok():
    if not os.path.isdir(QWEN_TOK_DIR):
        pytest.skip(f"tokenizer dir missing: {QWEN_TOK_DIR}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN_TOK_DIR, trust_remote_code=True)


def test_render_ids_is_in_template_no_double_special(qwen_tok):
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    ids = cf.render_ids(msgs, qwen_tok, seq_len=2048)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    decoded = qwen_tok.decode(ids)
    assert decoded.startswith("<|im_start|>")          # in-template envelope
    assert decoded.count("<|im_start|>") == 2          # one per turn, not doubled
    # no double-BOS: im_start should appear exactly where the template puts it
    im_start = qwen_tok.convert_tokens_to_ids("<|im_start|>")
    assert ids[0] == im_start


def test_render_ids_truncates_to_seq_len(qwen_tok):
    msgs = [{"role": "user", "content": "word " * 5000}]
    ids = cf.render_ids(msgs, qwen_tok, seq_len=512)
    assert len(ids) <= 512


def test_render_ids_empty_messages_returns_empty(qwen_tok):
    assert cf.render_ids([], qwen_tok, seq_len=512) == []


def test_system_only_record_renders_via_fallback(qwen_tok):
    # tool_calls records are [SYSTEM]-only; Qwen's template rejects "no user query".
    # render_text must fall back (fold into a user turn) rather than raise.
    msgs = cf.parse_to_messages("[SYSTEM]: You are an assistant. Tools: foo()", source="tool_calls")
    assert msgs == [{"role": "system", "content": "You are an assistant. Tools: foo()"}]
    ids = cf.render_ids(msgs, qwen_tok, seq_len=512)
    assert ids and ids[0] == qwen_tok.convert_tokens_to_ids("<|im_start|>")
