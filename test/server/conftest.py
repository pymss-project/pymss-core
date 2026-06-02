from __future__ import annotations

import asyncio
import contextlib
import json as json_module
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest


class FakeEntry:
    def __init__(self, name, model_type="fake", architecture="fake", category="test", target_stem="vocals"):
        self.name = name
        self.model_type = model_type
        self.architecture = architecture
        self.category_path = category
        self.primary_category = category
        self.target_stem = target_stem
        self.supported = True


class FakeSeparator:
    def __init__(
        self,
        name,
        instruments=("vocals", "instrument"),
        sample_rate=44100,
        model_type="fake",
        audio_params=None,
    ):
        self.name = name
        self.model_type = model_type
        self.device = "cpu"
        self.audio_params = audio_params or {}
        self.closed = False
        self.config = SimpleNamespace(
            training=SimpleNamespace(instruments=list(instruments), target_instrument=None),
            audio={"sample_rate": sample_rate, "chunk_size": 1024},
            inference={"batch_size": 1, "normalize": False},
        )

    def separate(self, mix, pbar=False, stems=None):
        requested = stems or self.config.training.instruments
        mix_array = np.asarray(mix, dtype=np.float32)
        output = mix_array.T if mix_array.ndim == 2 else mix_array
        return {stem: output for stem in requested}

    def close(self):
        self.closed = True


class ASGIResponse:
    def __init__(self, status_code, headers, body):
        self.status_code = status_code
        self.headers = headers
        self.content = body

    def json(self):
        return json_module.loads(self.content.decode("utf-8"))


class ASGIClient:
    def __init__(self, app):
        self.app = app
        self.loop = asyncio.new_event_loop()

    def get(self, path, headers=None):
        return self._run(self._request("GET", path, headers=headers))

    def post(self, path, json=None, content=None, headers=None):
        headers = dict(headers or {})
        if json is not None:
            content = json_module.dumps(json).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        return self._run(self._request("POST", path, body=content or b"", headers=headers))

    def close(self):
        self._shutdown_executor()
        self.loop.close()

    def _run(self, coroutine):
        response = self.loop.run_until_complete(_with_heartbeat(coroutine))
        self._shutdown_executor()
        return response

    def _shutdown_executor(self):
        executor = getattr(self.loop, "_default_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self.loop._default_executor = None

    async def _request(self, method, path, body=b"", headers=None):
        raw_headers = [
            (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
            for key, value in (headers or {}).items()
        ]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": raw_headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        messages = []
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await self.app(scope, receive, send)
        status_code = 500
        response_headers = {}
        chunks = []
        for message in messages:
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
        return ASGIResponse(status_code, response_headers, b"".join(chunks))


async def _with_heartbeat(coroutine):
    async def heartbeat():
        while True:
            await asyncio.sleep(0.01)

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        return await coroutine
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task


class LiveHTTPClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get(self, path, headers=None):
        return self._request("GET", path, headers=headers)

    def post(self, path, json_body=None, content=None, headers=None):
        headers = dict(headers or {})
        if json_body is not None:
            content = json_module.dumps(json_body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        return self._request("POST", path, body=content or b"", headers=headers)

    def _request(self, method, path, body=None, headers=None):
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers or {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return ASGIResponse(response.status, dict(response.headers.items()), response.read())
        except urllib.error.HTTPError as exc:
            return ASGIResponse(exc.code, dict(exc.headers.items()), exc.read())


def _free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def _live_client(app):
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_tcp_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            lifespan="off",
            log_level="error",
        )
    )
    errors = []

    def run_server():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def serve():
                await _with_heartbeat(server.serve())

            try:
                loop.run_until_complete(serve())
            finally:
                executor = getattr(loop, "_default_executor", None)
                if executor is not None:
                    executor.shutdown(wait=False)
                loop.close()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    client = LiveHTTPClient(base_url)
    deadline = time.monotonic() + 5
    last_error = None
    try:
        while time.monotonic() < deadline:
            try:
                if client.get("/health").status_code == 200:
                    break
            except OSError as exc:
                last_error = exc
            if errors:
                raise RuntimeError(f"uvicorn test server failed: {errors[0]}")
            time.sleep(0.05)
        else:
            raise RuntimeError(f"uvicorn test server did not start: {last_error}")
        yield client
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if errors:
            raise RuntimeError(f"uvicorn test server failed: {errors[0]}")


def _install_fake_loader(monkeypatch, loaded, *, fail_model=None):
    def fake_resolve_model(model, model_dir=None, require_supported=True, require_exists=True):
        if model == fail_model:
            raise KeyError(f"Unknown pymss model: {model}")
        canonical = {
            "alias-a": "model-a.ckpt",
            "model-a": "model-a.ckpt",
            "model-b": "model-b.ckpt",
        }.get(model, f"{model}.ckpt" if not str(model).endswith(".ckpt") else str(model))
        return {
            "entry": FakeEntry(canonical),
            "model_type": "fake",
            "model_path": f"/models/{canonical}",
            "config_path": "/configs/fake.yaml",
        }

    def fake_preload_config(_resolved):
        return {
            "training": {"instruments": ["vocals", "instrument"]},
            "audio": {"chunk_size": 1024},
            "inference": {"batch_size": 1},
        }

    def fake_create_separator(model, **_kwargs):
        canonical = fake_resolve_model(model)["entry"].name
        separator = FakeSeparator(canonical)
        loaded.append(separator)
        return separator

    monkeypatch.setattr("pymss.server.state.resolve_model", fake_resolve_model)
    monkeypatch.setattr("pymss.server.state._preload_config", fake_preload_config)
    monkeypatch.setattr("pymss.server.state.create_separator", fake_create_separator)


@pytest.fixture
def asgi_client_factory():
    clients = []

    def factory(app):
        client = ASGIClient(app)
        clients.append(client)
        return client

    yield factory

    for client in clients:
        client.close()


@pytest.fixture
def fake_loader(monkeypatch):
    loaded = []

    def install(*, fail_model=None):
        _install_fake_loader(monkeypatch, loaded, fail_model=fail_model)
        return loaded

    return install


@pytest.fixture
def live_client_factory():
    return _live_client
