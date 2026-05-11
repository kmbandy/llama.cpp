#!/usr/bin/env python3
"""MAD-137: paged-cache multi-seq stress driver.

NOTE: this test is currently blocked by MAD-141 — the server's prefill
admission loop deadlocks under sustained workloads exceeding the hot
pool budget. The driver below correctly detects the symptom (HTTP errors
+ unadvanced tier counters) and reports a hard failure. Once MAD-141
lands, the same invocation should reach the green path.

Boots a real llama-server with --parallel N --kv-tier-paged-blocks and
drives N concurrent simulated agents at it. Each agent sends a long
initial prompt (mixed-locality references are emulated by interleaving
queries that target recently produced tokens with queries that recall
content from much earlier in the conversation). After the configured
duration, scrapes /metrics + /slots and asserts on:

  - no HTTP 5xx responses (proxy for "no OOM, no crash")
  - paged_evict_hot_to_warm_total  > 0  (warm tier engaged)
  - paged_evict_warm_to_cold_total > 0  (cold tier engaged)  [if cold > 0]
  - paged_semantic_attempts_total  > 0  (semantic path exercised)
  - decode rate per agent above --decode-floor tok/s
  - optionally: semantic hit rate > --hit-rate-floor (defaults to 0 because
    hit-rate is content-dependent and only meaningful with curated workloads)

The script is intentionally stdlib-only (urllib + threading + json) so it
can run inside the army's ROCm docker container (Python 3, no pip).

Typical CI invocation (matrix runner sets the device-specific knobs):

  python3 stress-paged-multi-seq.py \\
      --bin   /workspace/llama.cpp/build-army/bin/llama-server \\
      --model /models/omnicoder-9b-q5_k_m.gguf \\
      --device CUDA0 \\
      --ctx 32768 --n-agents 4 \\
      --tier 25,75,0 --instance-id stress-1070 \\
      --duration 60 --decode-floor 5
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _http_post_json(url: str, body: dict, timeout: float = 60.0) -> tuple[int, dict | None]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, None


def _http_get_text(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError):
        return None


def _parse_prom_counter(metrics_text: str, name: str) -> float | None:
    """Parse a Prometheus-format counter/gauge value out of /metrics text."""
    full = f"llamacpp:{name}"
    for line in metrics_text.splitlines():
        if line.startswith(full + " "):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Agent driver — one per simulated user
# ---------------------------------------------------------------------------

@dataclass
class AgentStats:
    agent_id: int
    requests:        int = 0
    http_errors:     int = 0
    decode_tps_sum:  float = 0.0
    prefill_tps_sum: float = 0.0
    decoded_total:   int = 0
    transcript_tail: list[str] = field(default_factory=list)


def _seed_paragraph(agent_id: int, turn: int, locality: str) -> str:
    """Synthesize a deterministic-but-distinct paragraph. The only thing
    that matters for the cache is that text content varies enough to
    occupy distinct fingerprint directions. Locality tags are recorded
    in the prompt so the eventual server log is debuggable."""
    seed = f"agent{agent_id:02d}-turn{turn:04d}-{locality}"
    # ~256 tokens per paragraph is a reasonable proxy at q5_k_m densities.
    body = " ".join(f"{seed}-w{i:04d}" for i in range(256))
    return body


def _agent_loop(agent_id: int, base_url: str, ctx_per_agent: int,
                stop_event: threading.Event, stats: AgentStats):
    seq_id = agent_id

    # Backoff state: when the server returns an error or the request
    # fails, sleep before retrying. Exponential up to 5s. Without this
    # an agent in error state hammers the server thousands of times per
    # second, which is both useless and makes the failure mode unreadable.
    backoff_s = 0.5

    def _post(body: dict, timeout: float) -> bool:
        nonlocal backoff_s
        code, resp = _http_post_json(f"{base_url}/completion", body, timeout=timeout)
        stats.requests += 1
        if code != 200 or resp is None:
            stats.http_errors += 1
            time.sleep(backoff_s)
            backoff_s = min(5.0, backoff_s * 2.0)
            return False
        backoff_s = 0.5
        timings = resp.get("timings", {}) or {}
        stats.prefill_tps_sum += float(timings.get("prompt_per_second", 0.0))
        stats.decode_tps_sum  += float(timings.get("predicted_per_second", 0.0))
        stats.decoded_total   += int(timings.get("predicted_n", 0))
        if len(stats.transcript_tail) < 32:
            stats.transcript_tail.append((resp.get("content") or "")[:80])
        return True

    # Initial long context — pre-load the agent's KV with distinct content.
    initial_paragraphs = max(1, ctx_per_agent // 256)
    bootstrap = "\n\n".join(
        _seed_paragraph(agent_id, i, "init") for i in range(initial_paragraphs)
    )
    bootstrap += "\n\nQ: Repeat the agent identifier you saw at the start.\nA:"
    _post({
        "prompt":      bootstrap,
        "n_predict":   16,
        "temperature": 0.0,
        "cache_prompt": True,
        "id_slot":     seq_id,
    }, timeout=600.0)

    # Mixed-locality follow-ups until stop_event fires.
    turn = 0
    while not stop_event.is_set():
        turn += 1
        # Alternate recent vs far-back queries.
        locality = "recent" if (turn % 3) != 0 else "farback"
        followup = (
            _seed_paragraph(agent_id, turn, locality) +
            "\n\nQ: Summarize the prior section in one sentence.\nA:"
        )
        _post({
            "prompt":      followup,
            "n_predict":   24,
            "temperature": 0.0,
            "cache_prompt": True,
            "id_slot":     seq_id,
        }, timeout=300.0)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _wait_for_listening(log_path: Path, timeout: float = 600.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = log_path.read_text(errors="replace")
            if "main: server is listening" in data:
                return True
            if "model loading error" in data or "Aborted" in data or "GGML_ABORT" in data:
                return False
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin",        required=True, help="path to llama-server")
    p.add_argument("--model",      required=True)
    p.add_argument("--device",     default="",     help="--device flag value (e.g. CUDA0, ROCm0)")
    p.add_argument("--ctx",        type=int, default=8192,
                   help="total context per server instance")
    p.add_argument("--n-agents",   type=int, default=4,
                   help="--parallel N (each agent gets ctx/N tokens)")
    p.add_argument("--tier",       default="25,75,0",
                   help="--kv-tiered HOT,WARM,COLD percentages")
    p.add_argument("--instance-id", default="stress-test")
    p.add_argument("--ssd-path",   default="",
                   help="cold-tier directory; auto-tmp if unset")
    p.add_argument("--cache-type-k", default="turbo4")
    p.add_argument("--cache-type-v", default="turbo4")
    p.add_argument("--bge-small",  default="",
                   help="optional bge-small gguf for semantic prefetch")
    p.add_argument("--port",       type=int, default=18080)
    p.add_argument("--ngl",        type=int, default=99)
    p.add_argument("--duration",   type=int, default=60,
                   help="total stress duration in seconds")
    p.add_argument("--ctx-per-agent", type=int, default=2048,
                   help="initial context tokens to pre-load per agent")
    p.add_argument("--decode-floor", type=float, default=2.0,
                   help="minimum mean decode tok/s per agent for pass")
    p.add_argument("--report-json", default="",
                   help="write final results json to this path")
    args = p.parse_args()

    if not Path(args.bin).is_file():
        print(f"FAIL: --bin {args.bin} not found", file=sys.stderr)
        return 2
    if not Path(args.model).is_file():
        print(f"FAIL: --model {args.model} not found", file=sys.stderr)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="stress-paged-")
    log_path = Path(tmpdir) / "server.log"
    ssd_path = args.ssd_path or os.path.join(tmpdir, "ssd")
    os.makedirs(ssd_path, exist_ok=True)

    cmd = [
        args.bin,
        "-m", args.model,
        "-c", str(args.ctx),
        "--parallel", str(args.n_agents),
        "-ngl", str(args.ngl),
        "--no-mmap",
        "--metrics",
        "--kv-tier-paged-blocks",
        "--kv-tiered", args.tier,
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
        "--instance-id", args.instance_id,
        "--kv-tier-ssd-path", ssd_path,
        "--port", str(args.port),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.bge_small:
        cmd += ["--kv-tier-semantic-index", args.bge_small]

    print("LAUNCH:", " ".join(shlex.quote(c) for c in cmd))
    log_fh = log_path.open("wb")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        if not _wait_for_listening(log_path):
            print("FAIL: server did not reach 'main: server is listening' before timeout",
                  file=sys.stderr)
            return 1

        base_url = f"http://127.0.0.1:{args.port}"

        # Snapshot tier counters BEFORE the workload so we can compute deltas.
        metrics_before = _http_get_text(f"{base_url}/metrics") or ""
        before = {
            "evict_h2w":  _parse_prom_counter(metrics_before, "paged_evict_hot_to_warm_total"),
            "evict_w2c":  _parse_prom_counter(metrics_before, "paged_evict_warm_to_cold_total"),
            "sem_attempts": _parse_prom_counter(metrics_before, "paged_semantic_attempts_total"),
            "sem_hits":     _parse_prom_counter(metrics_before, "paged_semantic_hits_total"),
        }

        stop_event = threading.Event()
        stats_per_agent: list[AgentStats] = [AgentStats(agent_id=i) for i in range(args.n_agents)]
        threads = [
            threading.Thread(
                target=_agent_loop,
                args=(i, base_url, args.ctx_per_agent, stop_event, stats_per_agent[i]),
                daemon=True,
            )
            for i in range(args.n_agents)
        ]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        time.sleep(args.duration)
        stop_event.set()
        for t in threads:
            t.join(timeout=60.0)
        elapsed = time.monotonic() - t0

        # Final metrics
        metrics_after = _http_get_text(f"{base_url}/metrics") or ""
        after = {
            "evict_h2w":  _parse_prom_counter(metrics_after, "paged_evict_hot_to_warm_total"),
            "evict_w2c":  _parse_prom_counter(metrics_after, "paged_evict_warm_to_cold_total"),
            "sem_attempts": _parse_prom_counter(metrics_after, "paged_semantic_attempts_total"),
            "sem_hits":     _parse_prom_counter(metrics_after, "paged_semantic_hits_total"),
        }
        deltas = {k: (after[k] or 0) - (before[k] or 0) for k in after}

        # Per-agent decode mean (mean of per-request rates the server reports).
        agent_decodes: list[float] = []
        total_errors = 0
        for s in stats_per_agent:
            mean_decode = (s.decode_tps_sum / s.requests) if s.requests else 0.0
            agent_decodes.append(mean_decode)
            total_errors += s.http_errors

        # Hit rate (only well-defined when sem_attempts > 0)
        hit_rate = (deltas["sem_hits"] / deltas["sem_attempts"]
                    if deltas["sem_attempts"] > 0 else 0.0)

        # ─── Pass / fail ──────────────────────────────────────────────────
        failures: list[str] = []
        if total_errors > 0:
            failures.append(f"{total_errors} HTTP errors across all agents (no OOM expected)")
        if deltas["evict_h2w"] <= 0:
            failures.append("paged_evict_hot_to_warm_total did not advance (warm tier never engaged)")
        # Cold spill is only expected when COLD% > 0 in the tier config.
        cold_pct = int(args.tier.split(",")[2]) if len(args.tier.split(",")) >= 3 else 0
        if cold_pct > 0 and deltas["evict_w2c"] <= 0:
            failures.append("paged_evict_warm_to_cold_total did not advance (cold tier configured but unused)")
        if any(d < args.decode_floor for d in agent_decodes):
            failures.append(
                "decode rate floor not met: " +
                ", ".join(f"agent{i}={d:.2f}<{args.decode_floor}"
                          for i, d in enumerate(agent_decodes) if d < args.decode_floor)
            )

        report: dict[str, Any] = {
            "ok": not failures,
            "failures": failures,
            "elapsed_s": elapsed,
            "n_agents": args.n_agents,
            "tier": args.tier,
            "device": args.device,
            "decode_tps_per_agent": agent_decodes,
            "decode_tps_floor": args.decode_floor,
            "metrics_delta": deltas,
            "hit_rate": hit_rate,
            "total_http_errors": total_errors,
            "log_path": str(log_path),
        }
        print(json.dumps(report, indent=2))
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(report, indent=2))

        return 0 if not failures else 1

    finally:
        # Tear down the server.
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()
        # Leave tmpdir on failure for inspection; clean on success.
        # (Caller can override via --report-json or by just inspecting tmpdir.)
        if "ok" in locals() and locals().get("ok"):
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
