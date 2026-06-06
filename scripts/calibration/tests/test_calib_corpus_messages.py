"""calib_corpus message-source path: a {"messages":[…]} jsonl must route its pre-built
messages straight to chat_format.render_ids (NOT re-parse them from a text field), and
honour the min_chars floor by total message content length."""
import json
import random
import sys
from pathlib import Path

CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))

import calib_corpus as cc  # noqa: E402


def _msg_record(user_chars=300, asst_chars=300):
    return {"messages": [
        {"role": "user", "content": "u" * user_chars},
        {"role": "assistant", "content": "a" * asst_chars},
    ]}


def test_message_source_passes_messages_through(tmp_path, monkeypatch):
    rec = _msg_record()
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(rec) + "\n")

    captured = {}

    def fake_render_ids(messages, tokenizer, seq_len):
        captured["messages"] = messages
        return list(range(seq_len))  # seq_len ids back

    monkeypatch.setattr(cc.chat_format, "render_ids", fake_render_ids)

    out = cc._sample_jsonl(str(p), n=1, text_field="messages", seq_len=8,
                           tokenizer=object(), rng=random.Random(0),
                           min_chars=100, source="claude_traces", chat_fmt=True)
    assert len(out) == 1
    assert out[0].shape[1] == 8
    # the pre-built messages were handed through verbatim — no text re-parse
    assert captured["messages"] == rec["messages"]


def test_message_source_min_chars_skips_thin_records(tmp_path, monkeypatch):
    rec = _msg_record(user_chars=5, asst_chars=5)  # 10 content chars < min_chars
    p = tmp_path / "thin.jsonl"
    p.write_text(json.dumps(rec) + "\n")

    monkeypatch.setattr(cc.chat_format, "render_ids",
                        lambda m, t, s: list(range(s)))

    out = cc._sample_jsonl(str(p), n=1, text_field="messages", seq_len=8,
                           tokenizer=object(), rng=random.Random(0),
                           min_chars=100, source="claude_traces", chat_fmt=True,
                           token_target=None)
    # nothing met the floor; sampler exhausts its attempt budget and returns empty
    assert out == []


def test_claude_traces_registered_as_message_source():
    assert "claude_traces" in cc.MESSAGE_SOURCES
    assert "claude_traces" in cc._SRC
    # both agentic compositions carry the real-trace slice and drop the synthetic one
    for comp in ("agentic15", "agentic35"):
        assert "claude_traces" in cc.COMPOSITIONS[comp]
        assert "tool_calls" not in cc.COMPOSITIONS[comp]
    # dose-response: agentic35 weights the trace slice heavier than agentic15
    assert cc.COMPOSITIONS["agentic35"]["claude_traces"] > cc.COMPOSITIONS["agentic15"]["claude_traces"]
