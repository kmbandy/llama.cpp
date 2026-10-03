#!/usr/bin/env python3
# Stand-in for llama-wp-expert-worker in the router group-lifecycle tests. No GPU, no model.
#
#   fake-worker.py --listen HOST:PORT | --port N
#
# FAKE_WORKER_MODE:
#   ok          (default) listen; on SIGTERM print the stop-snapshot "written" line, exit 0
#   snap-fail   listen; on SIGTERM print the stop-snapshot "FAILED" line, exit 0
#   hip         print a "Hip error" line and never listen (the router must fail the start)
#   ignore-term listen and ignore SIGTERM (the router must SIGKILL it)
#   die         listen, then exit 3 after FAKE_WORKER_DIE_AFTER_MS (a worker dying mid-serve)
# FAKE_WORKER_LISTEN_DELAY_MS: wait this long before listening (startup ordering tests)
#
# At start it prints the router-injected env, so tests can check what the worker received.

import os
import signal
import socket
import sys
import time


def say(msg):
    print(msg, flush=True)


def parse_endpoint(argv):
    host, port = "127.0.0.1", None
    for i, a in enumerate(argv):
        if a == "--listen" and i + 1 < len(argv):
            h, _, p = argv[i + 1].rpartition(":")
            host, port = (h or "127.0.0.1"), int(p)
        elif a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
    if host in ("0.0.0.0", "", "::"):
        host = "127.0.0.1"
    return host, port


def main():
    mode = os.environ.get("FAKE_WORKER_MODE", "ok")
    host, port = parse_endpoint(sys.argv[1:])
    park = os.environ.get("WP_EXPERT_PARK_FILE", "")
    say("fake worker: env WP_EXPERT_PARK_FILE=%s WP_EXPERT_SEED_FROM_PARK=%s LLAMA_ROUTER_GEN=%s" % (
        park, os.environ.get("WP_EXPERT_SEED_FROM_PARK", ""), os.environ.get("LLAMA_ROUTER_GEN", "")))

    if mode == "hip":
        say("ggml_cuda_init: Hip error: out of memory (fake)")
        while True:
            time.sleep(1)

    def on_term(signum, frame):
        if mode == "snap-fail":
            say("wp expert worker: stop snapshot FAILED: fake failure")
        else:
            say("wp expert worker: stop snapshot written: %s (42 rows)" % (park or "/dev/null"))
        sys.exit(0)

    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, on_term)

    delay = int(os.environ.get("FAKE_WORKER_LISTEN_DELAY_MS", "0"))
    if delay > 0:
        time.sleep(delay / 1000.0)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(16)
    srv.settimeout(0.1)
    say("fake worker: expert worker listening on %s:%d" % (host, port))

    die_at = None
    if mode == "die":
        die_at = time.time() + int(os.environ.get("FAKE_WORKER_DIE_AFTER_MS", "500")) / 1000.0

    while True:
        if die_at is not None and time.time() >= die_at:
            say("fake worker: dying on purpose")
            os._exit(3)
        try:
            conn, _ = srv.accept()
            conn.close()
        except socket.timeout:
            pass
        except InterruptedError:
            pass


if __name__ == "__main__":
    main()
