import threading
import pytest
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_router_props():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["role"] == "router"
    assert res.body["max_instances"] == 2
    assert res.body["models_autoload"] is False
    assert res.body["build_info"].startswith("b")


@pytest.mark.parametrize(
    "model,success",
    [
        ("ggml-org/tinygemma3-GGUF:Q8_0", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(model: str, success: bool):
    global server
    server.start()
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_ids(is_reload: bool, headers: dict | None = None) -> set[str]:
    res = server.make_request(
        "GET", "/models" + ("?reload=1" if is_reload else ""), headers=headers
    )
    assert res.status_code == 200
    return {item["id"] for item in res.body.get("data", [])}


def _get_model_status(model_id: str, headers: dict | None = None) -> str:
    res = server.make_request("GET", "/models", headers=headers)
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60, headers: dict | None = None) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id, headers=headers)
        if last_status in desired:
            return last_status
        time.sleep(0.01)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    # 200 when already up, else 202 {state: loading | queued}: the load goes on in the background
    assert load_res.status_code in (200, 202)
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout, headers=headers)


def test_router_unload_model():
    global server
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K:F32",
        "ggml-org/test-model-stories260K-infill:F32",
    ]

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


# server_lru_sched tests (relying on LLAMA_SERVER_DEBUG_FAKE_TIMING)

MODEL_A = "ggml-org/tinygemma3-GGUF:Q8_0"
MODEL_B = "ggml-org/test-model-stories260K:F32"
MODEL_C = "ggml-org/test-model-stories260K-infill:F32"


def _tokenize(model_id: str, timeout: float | None = DEFAULT_REQUEST_TIMEOUT) -> ServerResponse:
    return server.make_request(
        "POST", "/tokenize", data={"model": model_id, "content": "hello world"}, timeout=timeout
    )


class _Bg:
    """runs one request in a thread, keeps its result, error and finish time"""

    def __init__(self, fn):
        self.result = None
        self.error: Exception | None = None
        self.done_at: float = 0.0
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)

    def _run(self, fn):
        try:
            self.result = fn()
        except Exception as e:
            self.error = e
        self.done_at = time.time()

    def start(self):
        self._thread.start()
        return self

    def join(self, timeout: int = 180):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "background request did not finish in time"
        return self

    def assert_ok(self, what: str):
        assert self.error is None, f"{what} raised {self.error!r}"
        assert self.result is not None and self.result.status_code == 200, \
            f"{what} failed: {self.result.status_code if self.result else None} {self.result.body if self.result else None}"


def test_router_queue_does_not_evict_busy_model():
    """a request that finds no free slot waits, and the model serving a request survives it"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)  # let the request reach the child and take the only slot

    # no slot free and MODEL_A is busy, so this queues instead of evicting mid-request
    queued = _Bg(lambda: _tokenize(MODEL_B)).start()

    busy.join()
    queued.join()

    # had MODEL_A been evicted while serving, its own request would have died
    busy.assert_ok("request against the busy model")
    queued.assert_ok("queued request")

    _wait_for_model_status(MODEL_B, {"loaded"}, timeout=120)
    assert _get_model_status(MODEL_A) == "unloaded"


def test_router_queue_coalesces_requests_for_same_model():
    """many requests for one missing model share a slot, so only one model is given up"""
    global server
    server.models_max = 2
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)
    _load_model_and_wait(MODEL_B, timeout=120)

    # keep MODEL_A busy so MODEL_B is the only model that can be given up
    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    waiters = [_Bg(lambda: _tokenize(MODEL_C)).start() for _ in range(3)]

    busy.join()
    for w in waiters:
        w.join()

    busy.assert_ok("request against the busy model")
    for i, w in enumerate(waiters):
        w.assert_ok(f"queued request {i}")

    _wait_for_model_status(MODEL_C, {"loaded"}, timeout=120)
    # one entry for 3 requests means one eviction: MODEL_B goes, MODEL_A is left alone.
    # without coalescing the leftover entries still ask for a slot,
    # and MODEL_A is taken too as soon as it goes idle
    assert _get_model_status(MODEL_A) == "loaded"
    assert _get_model_status(MODEL_B) == "unloaded"


def test_router_queue_client_disconnect_keeps_model():
    """a client that leaves while queued must not cost a running model its slot"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    # queues behind MODEL_A, then gives up long before MODEL_A goes idle
    with pytest.raises(requests.exceptions.RequestException):
        _tokenize(MODEL_B, timeout=1)

    busy.join()
    busy.assert_ok("request against the busy model")

    # nobody is waiting anymore, so MODEL_A keeps its slot
    time.sleep(3)
    assert _get_model_status(MODEL_A) == "loaded"
    assert _get_model_status(MODEL_B) == "unloaded"


def test_router_queue_is_fifo():
    """the queue is served in arrival order"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    first = _Bg(lambda: _tokenize(MODEL_B)).start()
    time.sleep(1)  # keep the arrival order unambiguous
    second = _Bg(lambda: _tokenize(MODEL_C)).start()

    busy.join()
    first.join()
    second.join()

    busy.assert_ok("request against the busy model")
    first.assert_ok("first queued request")
    second.assert_ok("second queued request")

    assert first.done_at < second.done_at, "queue was not served in arrival order"


def test_router_queue_two_waiters_share_one_eviction():
    """two requests that both find the same idle model must both be served in the end"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    # both arrive while MODEL_A is idle, so both want its slot; only one eviction can happen
    first = _Bg(lambda: _tokenize(MODEL_B)).start()
    second = _Bg(lambda: _tokenize(MODEL_C)).start()

    first.join(90)
    second.join(90)

    first.assert_ok("first queued request")
    second.assert_ok("second queued request")
    assert _get_model_status(MODEL_A) == "unloaded"


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_reload_models():
    """POST /models/reload re-reads the INI preset and updates the model list."""
    global server

    preset_path = os.path.join(TMP_DIR, "test_reload.ini")

    # Initial preset: two models
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-a]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "\n"
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
        )

    server.models_preset = preset_path
    server.start()

    ids = _get_model_ids(is_reload=False)
    assert "model-reload-a" in ids
    assert "model-reload-b" in ids

    # Updated preset: remove a, keep b unchanged, add c
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
            "\n"
            "[model-reload-c]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
        )

    try:
        ids = _get_model_ids(is_reload=True)
        assert "model-reload-a" not in ids, "removed model should no longer appear"
        assert "model-reload-b" in ids, "unchanged model should still appear"
        assert "model-reload-c" in ids, "newly added model should appear"
    finally:
        os.remove(preset_path)


def test_router_dedup_cache_models():
    """dedup-cache-models hides the cache entry backing a preset from GET /models"""
    global server

    preset_path = os.path.join(TMP_DIR, "test_dedup.ini")
    main_cache_id = "ggml-org/test-model-stories260K:F32"
    draft_cache_id = "ggml-org/test-model-stories260K-infill:F32"

    with open(preset_path, "w") as f:
        f.write(
            "[model-dedup]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "spec-draft-hf = ggml-org/test-model-stories260K-infill\n"
            "dedup-cache-models = 1\n"
        )

    server.models_preset = preset_path
    server.start()

    try:
        ids = _get_model_ids(is_reload=False)
        assert "model-dedup" in ids
        assert main_cache_id not in ids, "main cache model should be hidden by dedup"
        assert draft_cache_id not in ids, "draft cache model should be hidden by dedup"
        # other cache models are unaffected
        assert "ggml-org/tinygemma3-GGUF:Q8_0" in ids

        # the hidden model is only hidden from the listing, it can still be used
        res = server.make_request("POST", "/tokenize", data={"model": main_cache_id, "content": "hello"})
        assert res.status_code == 200

        # disabling the flag brings the cache entry back on reload
        with open(preset_path, "w") as f:
            f.write(
                "[model-dedup]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
                "spec-draft-hf = ggml-org/test-model-stories260K-infill\n"
            )
        ids = _get_model_ids(is_reload=True)
        assert main_cache_id in ids
        assert draft_cache_id in ids

        # the flag also works from the global section
        with open(preset_path, "w") as f:
            f.write(
                "[*]\n"
                "dedup-cache-models = 1\n"
                "\n"
                "[model-dedup]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
                "spec-draft-hf = ggml-org/test-model-stories260K-infill\n"
            )
        ids = _get_model_ids(is_reload=True)
        assert "model-dedup" in ids
        assert main_cache_id not in ids, "main cache model should be hidden by global dedup"
        assert draft_cache_id not in ids, "draft cache model should be hidden by global dedup"
    finally:
        os.remove(preset_path)


def test_router_remote_preset():
    global server
    server.model_hf_repo = "ggml-org/test-preset-ci"
    server.model_hf_file = None
    server.offline = False
    server.start()

    # Should see preset models in GET /models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    ids = {item["id"] for item in res.body.get("data", [])}
    assert "tinygemma3-preset" in ids
    assert "stories260K-test" in ids

    # Should be able to load a preset model
    model_id = "tinygemma3-preset"
    _load_model_and_wait(model_id)


MODEL_DOWNLOAD_ID = "ggml-org/test-model-router-download:F16"
MODEL_DOWNLOAD_TIMEOUT = 30


def _listen_sse(
    server: ServerProcess, collected: list, stop: threading.Event, ready: threading.Event | None = None
):
    """Collect /models/sse events into `collected` until `stop` is set.

    When `ready` is provided, it is set once the streaming response is open,
    i.e. the server has accepted the connection and registered us as a
    subscriber. Callers that trigger one-shot events (e.g. download_finished)
    must wait on `ready` before acting, otherwise the event can be broadcast
    before this client is subscribed and be lost.
    """
    url = f"http://{server.server_host}:{server.server_port}/models/sse"
    try:
        with requests.get(url, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT) as resp:
            if ready is not None:
                ready.set()
            for line_bytes in resp.iter_lines():
                if stop.is_set():
                    break
                line = line_bytes.decode("utf-8")
                if line.startswith("data: "):
                    collected.append(json.loads(line[6:]))
    except Exception:
        pass


def _wait_for_sse_event(collected: list, event_type: str, model: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e.get("event") == event_type and e.get("model") == model for e in collected):
            return True
        time.sleep(0.01)
    return False


def test_router_download_model():
    """Case 1: download a model at the model limit, verify SSE events and GET /models."""
    global server
    server.models_max = 1
    server.start()

    # Ensure the model is not present before we start
    server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")

    # A download worker must not consume or evict a model slot
    _load_model_and_wait(MODEL_B, timeout=120)

    sse_events: list = []
    stop = threading.Event()
    sse_ready = threading.Event()
    sse_thread = threading.Thread(
        target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
    )
    sse_thread.start()

    # wait for the SSE client to be subscribed before triggering the download,
    # otherwise the one-shot download_finished event can be broadcast before
    # this client is registered and be lost
    assert sse_ready.wait(10), "SSE client failed to connect"

    # Trigger the download
    res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
    assert res.status_code == 200
    assert res.body.get("success") is True

    # Wait for download_finished SSE event
    finished = _wait_for_sse_event(
        sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
    )
    stop.set()

    assert finished, "Never received download_finished SSE event"
    assert any(
        e.get("event") == "download_progress" and e.get("model") == MODEL_DOWNLOAD_ID
        for e in sse_events
    ), "No download_progress events received"

    # Model should now appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID in ids, f"{MODEL_DOWNLOAD_ID} not found in /models after download"
    assert _get_model_status(MODEL_B) == "loaded"


def test_router_delete_model():
    """Case 2: delete the downloaded model, verify it disappears from GET /models."""
    global server
    server.start()

    # Ensure the model exists (download it if needed)
    if MODEL_DOWNLOAD_ID not in _get_model_ids(is_reload=False):
        sse_events: list = []
        stop = threading.Event()
        sse_ready = threading.Event()
        threading.Thread(
            target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
        ).start()
        # subscribe before triggering the download so the one-shot
        # download_finished event is not lost (see test_router_download_model)
        assert sse_ready.wait(10), "SSE client failed to connect"
        res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
        assert res.status_code == 200
        finished = _wait_for_sse_event(
            sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
        )
        stop.set()
        assert finished, "Model did not finish downloading before delete test"

    # Delete the model
    del_res = server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")
    assert del_res.status_code == 200
    assert del_res.body.get("success") is True

    # Model should no longer appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID not in ids, f"{MODEL_DOWNLOAD_ID} still present after deletion"


# model groups: one spine (a real model) + kind=external workers (the fake worker fixture).
# Runs a model process: the controller runs this, not the implementer.

FAKE_WORKER = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tests", "router-fixtures", "fake-worker.py"))
GROUP_SPINE = "group-spine"
GROUP_WORKERS = ["group-w1", "group-w2"]


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pid_gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            text = f.read()
    except OSError:
        return True
    return text[text.rfind(")") + 2] == "Z"


def _group_entry() -> dict:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == GROUP_SPINE:
            return item
    raise AssertionError(f"{GROUP_SPINE} not in /models")


def _wait_group(predicate, timeout: float, what: str) -> dict:
    deadline = time.time() + timeout
    entry = None
    while time.time() < deadline:
        entry = _group_entry()
        if predicate(entry):
            return entry
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}; last entry: {entry}")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="model groups are POSIX/Linux only")
@pytest.mark.skipif(not os.path.exists(FAKE_WORKER), reason="fake worker fixture missing")
@pytest.mark.skipif(not sys.executable or not os.path.isabs(sys.executable), reason="no absolute python3 for the fake worker")
def test_router_model_group_lifecycle():
    import shlex
    import signal
    global server

    preset_path = os.path.join(TMP_DIR, "test_group.ini")
    park_dir = os.path.join(TMP_DIR, "group-park")
    os.makedirs(park_dir, exist_ok=True)
    listen_delay_ms = 1500
    ports = {w: _free_port() for w in GROUP_WORKERS}
    lines = [
        f"[{GROUP_SPINE}]",
        "hf-repo = ggml-org/test-model-stories260K",
        f"depends = {','.join(GROUP_WORKERS)}",
        "startup-timeout = 60",
        "",
    ]
    for w in GROUP_WORKERS:
        launch = " ".join(shlex.quote(a) for a in [sys.executable, FAKE_WORKER, "--listen", f"127.0.0.1:{ports[w]}"])
        lines += [
            f"[{w}]",
            "kind = external",
            f"launch = {launch}",
            f"park-file = {os.path.join(park_dir, w + '.park')}",
            f"env = FAKE_WORKER_LISTEN_DELAY_MS={listen_delay_ms}",
            "",
        ]
    with open(preset_path, "w") as f:
        f.write("\n".join(lines))

    server.models_preset = preset_path
    server.start()
    try:
        # workers are not requestable models: not in the OAI listing
        res = server.make_request("GET", "/v1/models")
        oai_ids = {m["id"] for m in res.body.get("data", [])}
        assert GROUP_SPINE in oai_ids and not (set(GROUP_WORKERS) & oai_ids)

        # load: watch from a second thread that the spine is never loaded before every worker is ready
        seen_bad_order = []
        seen_starting = []
        stop_watch = threading.Event()

        def watch():
            while not stop_watch.is_set():
                try:
                    e = _group_entry()
                except Exception:
                    continue
                g = e["status"].get("group") or {}
                states = [w.get("state") for w in g.get("workers", [])]
                if "starting" in states:
                    seen_starting.append(e["status"]["value"])
                if e["status"]["value"] == "loaded" and any(s != "ready" for s in states):
                    seen_bad_order.append(states)
                time.sleep(0.05)

        watcher = threading.Thread(target=watch)
        watcher.start()
        t0 = time.time()
        load_res = server.make_request("POST", "/models/load", data={"model": GROUP_SPINE}, timeout=180)
        # /models/load answers 202 while the load goes on (workers first, then the spine)
        assert load_res.status_code in (200, 202), load_res.body
        entry = _wait_group(lambda e: e["status"]["value"] == "loaded", 180, "spine loaded")
        stop_watch.set()
        watcher.join()
        assert not seen_bad_order, f"spine loaded while a worker was not ready: {seen_bad_order}"
        assert seen_starting and all(v == "loading" for v in seen_starting), seen_starting

        group = entry["status"]["group"]
        assert group["status"] == "ready"
        assert [w["name"] for w in group["workers"]] == GROUP_WORKERS
        for w in group["workers"]:
            assert w["state"] == "ready" and w["pid"] > 0 and w["port"] == ports[w["name"]]
            assert w["stop_snapshot"] == "none"
        first_pids = [w["pid"] for w in group["workers"]]

        # unload: spine, then every worker (TERM -> stop snapshot written); no process left
        unload_res = server.make_request("POST", "/models/unload", data={"model": GROUP_SPINE})
        assert unload_res.status_code == 200
        entry = _wait_group(lambda e: e["status"]["value"] == "unloaded", 120, "group unloaded")
        group = entry["status"]["group"]
        assert group["status"] == "unloaded"
        for w in group["workers"]:
            assert w["state"] == "exited" and w["exit_code"] == 0 and not w["killed"]
            assert w["stop_snapshot"] == "written"
            assert w["stop_snapshot_detail"].startswith(os.path.join(park_dir, w["name"] + ".park"))
        for pid in first_pids:
            assert _pid_gone(pid), f"worker pid {pid} still running after unload"

        # reload, then kill a worker while ready: spine stopped, group failed
        _load_model_and_wait(GROUP_SPINE, timeout=180)
        group = _group_entry()["status"]["group"]
        assert group["status"] == "ready"
        victim = group["workers"][0]
        os.kill(victim["pid"], signal.SIGKILL)
        entry = _wait_group(lambda e: e["status"]["value"] == "unloaded", 120, "spine stopped after worker death")
        assert entry["status"].get("failed") is True
        group = entry["status"]["group"]
        assert group["status"] == "failed"
        assert victim["name"] in group.get("reason", "")
        for w in group["workers"]:
            assert w["state"] == "exited"
            assert _pid_gone(w["pid"])

        # the next request reloads the whole group
        res = server.make_request("POST", "/tokenize", data={"model": GROUP_SPINE, "content": "hello"}, timeout=180)
        assert res.status_code == 200, res.body
        group = _group_entry()["status"]["group"]
        assert group["status"] == "ready"
        new_pids = [w["pid"] for w in group["workers"]]
        assert all(p > 0 for p in new_pids) and victim["pid"] not in new_pids
    finally:
        try:
            server.make_request("POST", "/models/unload", data={"model": GROUP_SPINE})
            _wait_group(lambda e: e["status"]["value"] == "unloaded", 120, "final unload")
        except Exception:
            pass
        os.remove(preset_path)
