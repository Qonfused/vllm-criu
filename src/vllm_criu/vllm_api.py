## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Response:
  status: int
  body: bytes


class VllmApi:
  def __init__(self, host: str = "127.0.0.1", port: int = 8001) -> None:
    self.base_url = f"http://{host}:{port}"

  def request(
    self,
    path: str,
    method: str = "GET",
    timeout: float = 5.0,
    payload: object | None = None,
  ) -> Response:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
      data = json.dumps(payload).encode()
      headers["content-type"] = "application/json"
    request = urllib.request.Request(
      self.base_url + path,
      method=method,
      data=data,
      headers=headers,
    )
    try:
      with urllib.request.urlopen(request, timeout=timeout) as response:
        return Response(response.status, response.read())
    except urllib.error.HTTPError as error:
      return Response(error.code, error.read())

  def ready(self) -> bool:
    try:
      return 200 <= self.request("/v1/models", timeout=2).status < 300
    except (OSError, urllib.error.URLError):
      return False

  def sleep_level2(self, timeout: float) -> None:
    self.sleep_level(2, timeout)

  def sleep_level(self, level: int, timeout: float) -> None:
    response = self.request(f"/sleep?level={int(level)}", "POST", timeout)
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM level-2 sleep", response))

  def wake_weights(self, timeout: float) -> None:
    response = self.request("/wake_up?tags=weights", "POST", timeout)
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM weights wake", response))

  def reload_weights(self, timeout: float) -> None:
    response = self.request(
      "/collective_rpc",
      "POST",
      timeout,
      {"method": "reload_weights", "timeout": int(timeout)},
    )
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM weight reload", response))

  def wake_kv_cache(self, timeout: float) -> None:
    response = self.request("/wake_up?tags=kv_cache", "POST", timeout)
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM KV-cache wake", response))

  def wake_scheduler(self, timeout: float) -> None:
    response = self.request("/wake_up?tags=scheduling", "POST", timeout)
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM scheduler wake", response))

  def wake_level2(self, timeout: float) -> None:
    self.wake_weights(timeout)
    self.reload_weights(timeout)
    self.wake_kv_cache(timeout)

  def checkpoint_prepare(self, timeout: float) -> None:
    response = self.request(
      "/collective_rpc",
      "POST",
      timeout,
      {"method": "checkpoint_prepare", "timeout": int(timeout)},
    )
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM checkpoint prepare", response))

  def checkpoint_restore(self, timeout: float) -> None:
    response = self.request(
      "/collective_rpc",
      "POST",
      timeout,
      {"method": "checkpoint_restore", "timeout": int(timeout)},
    )
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM checkpoint restore", response))

  def reinitialize_after_restore(self, timeout: float) -> None:
    response = self.request(
      "/lifecycle/reinitialize_after_restore", "POST", timeout
    )
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM allocator reinitialization", response))

  def reconnect_zmq(self, timeout: float) -> None:
    response = self.request("/lifecycle/reconnect_zmq", "POST", timeout)
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM ZMQ reconnect", response))

  def wait_for_enginecore_ready(self, timeout: float) -> None:
    response = self.request(
      "/lifecycle/wait_for_enginecore_ready", "POST", timeout
    )
    if response.status >= 300:
      raise RuntimeError(self._error("vLLM EngineCore READY handshake", response))

  @staticmethod
  def _error(operation: str, response: Response) -> str:
    detail = response.body.decode(errors="replace").strip()
    return f"{operation} returned HTTP {response.status}: {detail}"
