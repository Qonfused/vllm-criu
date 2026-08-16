## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from vllm_criu.vllm_api import Response, VllmApi


def test_level2_wake_reloads_weights_before_kv_cache(monkeypatch) -> None:
    api = VllmApi()
    calls: list[tuple[str, str, object | None]] = []

    def request(path: str, method: str = "GET", timeout: float = 5.0, payload=None) -> Response:
        calls.append((path, method, payload))
        return Response(200, b"{}")

    monkeypatch.setattr(api, "request", request)
    api.wake_level2(30)

    assert calls == [
        ("/wake_up?tags=weights", "POST", None),
        ("/collective_rpc", "POST", {"method": "reload_weights", "timeout": 30}),
        ("/wake_up?tags=kv_cache", "POST", None),
    ]


def test_reconnect_zmq(monkeypatch) -> None:
    api = VllmApi()
    calls: list[tuple[str, str, object | None]] = []

    def request(path: str, method: str = "GET", timeout: float = 5.0, payload=None) -> Response:
        calls.append((path, method, payload))
        return Response(200, b"{}")

    monkeypatch.setattr(api, "request", request)
    api.reconnect_zmq(12)

    assert calls == [("/lifecycle/reconnect_zmq", "POST", None)]


def test_wait_for_enginecore_ready(monkeypatch) -> None:
    api = VllmApi()
    calls: list[tuple[str, str, object | None]] = []

    def request(path: str, method: str = "GET", timeout: float = 5.0, payload=None) -> Response:
        calls.append((path, method, payload))
        return Response(200, b"{}")

    monkeypatch.setattr(api, "request", request)
    api.wait_for_enginecore_ready(12)

    assert calls == [("/lifecycle/wait_for_enginecore_ready", "POST", None)]


def test_checkpoint_communicator_hooks(monkeypatch) -> None:
    api = VllmApi()
    calls: list[tuple[str, str, object | None]] = []

    def request(path: str, method: str = "GET", timeout: float = 5.0, payload=None) -> Response:
        calls.append((path, method, payload))
        return Response(200, b"{}")

    monkeypatch.setattr(api, "request", request)
    api.checkpoint_prepare(30)
    api.checkpoint_restore(30)

    assert calls == [
        (
            "/collective_rpc",
            "POST",
            {"method": "checkpoint_prepare", "timeout": 30},
        ),
        (
            "/collective_rpc",
            "POST",
            {"method": "checkpoint_restore", "timeout": 30},
        ),
    ]


def test_reinitialize_after_restore(monkeypatch) -> None:
    api = VllmApi()
    calls: list[tuple[str, str, object | None]] = []

    def request(path: str, method: str = "GET", timeout: float = 5.0, payload=None) -> Response:
        calls.append((path, method, payload))
        return Response(200, b"{}")

    monkeypatch.setattr(api, "request", request)
    api.reinitialize_after_restore(30)

    assert calls == [("/lifecycle/reinitialize_after_restore", "POST", None)]
