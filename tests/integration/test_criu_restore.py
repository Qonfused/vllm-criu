## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""Opt-in validation of the graph-preserving CRIU resume path.

Run inside the vLLM container with:

    VLLM_CRIU_INTEGRATION=1 uv run --project /opt/vllm-project pytest \
      -m criu_integration -s /opt/vllm-project/tests/integration/test_criu_restore.py

These tests deliberately exercise the real suspend/resume endpoints. They are
not part of the normal unit-test run and do not add any validation work to the
production resume path. Multi-GPU/NCCL graph behavior is outside this TP=1
test suite.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest


pytestmark = pytest.mark.criu_integration


CONTROL_URL = os.environ.get("VLLM_CRIU_CONTROL_URL", "http://127.0.0.1:9000")
VLLM_URL = os.environ.get("VLLM_CRIU_VLLM_URL", "http://127.0.0.1:8001")
GUARD_URL = os.environ.get("VLLM_CRIU_GUARD_URL", "http://vllm-guard:8002")
MODEL = os.environ.get("MODEL_NAME", "Qwen3.8-27B-NVFP4")
TIMEOUT = float(os.environ.get("VLLM_CRIU_TEST_TIMEOUT", "180"))


def _request(
    base_url: str,
    path: str,
    method: str = "GET",
    payload: object | None = None,
    timeout: float = TIMEOUT,
) -> tuple[int, dict[str, object]]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        base_url + path,
        method=method,
        data=data,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body or b"{}")
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            parsed = json.loads(body or b"{}")
        except json.JSONDecodeError:
            parsed = {"error": body.decode(errors="replace")}
        return error.code, parsed


def _control_health() -> dict[str, object]:
    status, body = _request(CONTROL_URL, "/launcher/healthz", timeout=5)
    assert status in {200, 503}, body
    return body


def _wait_until_running() -> None:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        health = _control_health()
        if health.get("state") == "running" and health.get("ok"):
            return
        time.sleep(0.5)
    pytest.fail(f"vLLM did not become healthy: {_control_health()}")


def _ensure_running() -> None:
    health = _control_health()
    if health.get("state") == "checkpointed":
        status, body = _request(CONTROL_URL, "/launcher/resume", method="POST")
        assert status == 200, body
    _wait_until_running()


def _rpc(method: str) -> dict[str, object]:
    status, body = _request(
        VLLM_URL,
        "/collective_rpc",
        method="POST",
        payload={"method": method, "timeout": int(TIMEOUT)},
    )
    assert status == 200, body
    results = body.get("results")
    assert isinstance(results, list) and results, body
    result = results[0]
    assert isinstance(result, dict), body
    return result


def _chat(max_tokens: int) -> float:
    started = time.monotonic()
    status, body = _request(
        GUARD_URL,
        "/v1/chat/completions",
        method="POST",
        payload={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    elapsed = time.monotonic() - started
    assert status == 200, body
    return elapsed


@pytest.mark.skipif(
    os.environ.get("VLLM_CRIU_INTEGRATION", "0").lower()
    not in {"1", "true", "yes", "on"},
    reason="set VLLM_CRIU_INTEGRATION=1 to run stateful CRIU validation",
)
def test_graphs_survive_criu_restore_and_request_matrix() -> None:
    """Check graph inventory and representative graph-backed requests.

    The request matrix is intentionally a validation workload, not a restore
    hook. Comparing the graph inventory before/after the requests also catches
    an unexpected recapture or loss of restored entries.
    """

    _ensure_running()
    before = _rpc("vllm_criu_preserve_cudagraphs")
    assert before["cudagraphs_preserved"] is True, before
    assert before["live_captures"] == before["graph_entries"], before

    status, body = _request(
        CONTROL_URL,
        "/launcher/suspend",
        method="POST",
    )
    assert status == 200, body
    assert body.get("state") == "checkpointed", body

    status, body = _request(
        CONTROL_URL,
        "/launcher/resume",
        method="POST",
    )
    assert status == 200, body
    _wait_until_running()

    after_restore = _rpc("vllm_criu_preserve_cudagraphs")
    assert after_restore["cudagraphs_preserved"] is True, after_restore
    assert after_restore["live_captures"] == before["live_captures"], (
        before,
        after_restore,
    )

    timings = [_chat(max_tokens) for max_tokens in (1, 4, 8)]
    after_requests = _rpc("vllm_criu_preserve_cudagraphs")
    assert after_requests["live_captures"] == after_restore["live_captures"], (
        after_restore,
        after_requests,
    )
    print(
        "CRIU graph validation:",
        {"before": before, "after_restore": after_restore, "after_requests": after_requests, "request_seconds": timings},
    )
