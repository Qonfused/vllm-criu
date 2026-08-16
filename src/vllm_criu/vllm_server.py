## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""vLLM entrypoint with an optional restore-friendly ZMQ transport patch."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

from .enginecore import install_enginecore_restore_patch


LOG = logging.getLogger("vllm-criu.server")


def _restore_marker_path() -> Path:
  return (
    Path(os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current"))
    .parent
    / ".vllm-criu-restore-request"
  )


def _write_restore_diag(message: str) -> None:
  try:
    path = _restore_marker_path().parent / ".vllm-criu-api-diag.log"
    with path.open("a") as stream:
      stream.write(f"{time.time():.6f} {message}\n")
  except OSError:
    pass


def _repair_restored_standard_streams() -> None:
  """Replace API stdio descriptors that CRIU restored as dead pipes.

  The API process is part of the CRIU image, so its stdout/stderr pipe can
  refer to the pre-dump launcher connection.  Python may later terminate
  with status 120 when interpreter shutdown tries to flush that pipe.  The
  EngineCore has its own equivalent repair in its local restore sequence;
  the API process needs the repair independently.
  """

  log_path = _restore_marker_path().parent / ".vllm-criu-api-stdio.log"
  try:
    output = open(log_path, "a", buffering=1)
    error = open(log_path, "a", buffering=1)
  except OSError:
    output = open(os.devnull, "w", buffering=1)
    error = open(os.devnull, "w", buffering=1)

  for stream, replacement in ((sys.stdout, output), (sys.stderr, error)):
    try:
      stream.flush()
    except (BrokenPipeError, OSError, ValueError):
      pass
  sys.stdout = output
  sys.stderr = error
  LOG.warning("repaired restored API stdout/stderr streams")


def _install_restore_stdio_watcher() -> None:
  if getattr(_install_restore_stdio_watcher, "_installed", False):
    return
  _install_restore_stdio_watcher._installed = True  # type: ignore[attr-defined]
  marker = _restore_marker_path()
  pending = marker.parent / ".vllm-criu-restore-pending"

  def watch() -> None:
    while True:
      if marker.exists() or pending.exists():
        _repair_restored_standard_streams()
        return
      time.sleep(0.1)

  threading.Thread(
    target=watch,
    name="VLLMCRIUApiStdioRepair",
    daemon=True,
  ).start()


def install_tcp_transport() -> None:
  """Use loopback TCP for API↔EngineCore ZMQ sockets.

  vLLM normally uses filesystem IPC sockets for a single local EngineCore.
  libzmq's in-memory connection state is not reliably reconstructible after
  CRIU restores both endpoints. TCP lets the endpoints reconnect after
  CRIU's ``--tcp-close`` handling. This is opt-in because it changes the
  normal local transport.
  """
  import vllm.v1.engine.utils as engine_utils

  original = engine_utils.get_engine_zmq_addresses

  def get_engine_zmq_addresses(*args, **kwargs):
    addresses = original(*args, **kwargs)
    addresses.inputs = ["tcp://127.0.0.1:0" for _ in addresses.inputs]
    addresses.outputs = ["tcp://127.0.0.1:0" for _ in addresses.outputs]
    return addresses

  engine_utils.get_engine_zmq_addresses = get_engine_zmq_addresses

  # core_client imports this function into its own module namespace.
  import vllm.v1.engine.core_client as core_client

  core_client.get_engine_zmq_addresses = get_engine_zmq_addresses


def install_restore_reconnect() -> None:
  """Expose a small, opt-in API to repair restored ZMQ channels.

  CRIU restores the Python and libzmq objects, but ``--tcp-close`` drops the
  live TCP connections.  Rebinding the API-side sockets gives the restored
  EngineCore DEALER/PUSH sockets a fresh connection target while preserving
  the endpoint ports captured in the checkpoint.
  """
  import asyncio
  import threading
  import weakref

  import zmq
  import zmq.asyncio
  import msgspec.msgpack
  from vllm.entrypoints.serve.dev.rpc import api_router
  from vllm.utils.network_utils import make_zmq_socket
  from vllm.v1.engine.core_client import AsyncMPClient
  from vllm.v1.engine.core_client import _process_utility_output
  from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    FT_STATUS_CALL_ID,
    EngineCoreReadyResponse,
  )
  from vllm.v1.engine.exceptions import EngineDeadError
  from vllm.v1.engine.core_client import AsyncMPClient, BackgroundResources

  _install_restore_stdio_watcher()

  if not getattr(BackgroundResources, "_vllm_criu_diag_patch", False):
    original_validate_alive = BackgroundResources.validate_alive

    def traced_validate_alive(self, frames):
      try:
        return original_validate_alive(self, frames)
      except Exception as error:
        _write_restore_diag(
          "validate_alive raised "
          f"{type(error).__name__}: engine_dead={self.engine_dead} "
          f"frames={len(frames)}"
        )
        raise

    BackgroundResources.validate_alive = traced_validate_alive
    BackgroundResources._vllm_criu_diag_patch = True

  if not getattr(AsyncMPClient, "_vllm_criu_diag_patch", False):
    original_get_output_async = AsyncMPClient.get_output_async

    async def traced_get_output_async(self):
      try:
        return await original_get_output_async(self)
      except Exception as error:
        _write_restore_diag(
          "get_output_async raised "
          f"{type(error).__name__}: engine_dead={self.resources.engine_dead}"
        )
        raise

    AsyncMPClient.get_output_async = traced_get_output_async
    AsyncMPClient._vllm_criu_diag_patch = True

  if not getattr(AsyncMPClient, "_vllm_criu_send_trace", False):
    original_send_input_message = AsyncMPClient._send_input_message

    def traced_send_input_message(self, message, engine):
      LOG.warning(
        "sending EngineCore utility/input message: engine=%r dead=%s "
        "input_socket=%r",
        engine,
        self.resources.engine_dead,
        self.input_socket,
      )
      try:
        return original_send_input_message(self, message, engine)
      except Exception:
        LOG.exception("EngineCore utility/input ZMQ send failed")
        raise

    AsyncMPClient._send_input_message = traced_send_input_message
    AsyncMPClient._vllm_criu_send_trace = True

  async def apply_ready_response_async(
    self: AsyncMPClient, payload: bytes
  ) -> None:
    """Apply READY metadata without blocking the asyncio event loop."""

    if not payload:
      return
    response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
    vllm_config = self.vllm_config
    vllm_config.model_config.max_model_len = min(
      vllm_config.model_config.max_model_len, response.max_model_len
    )
    num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks + response.num_gpu_blocks
    cache_config = vllm_config.cache_config
    cache_config.block_size = response.block_size
    cache_config.kv_cache_size_tokens = (
      getattr(cache_config, "kv_cache_size_tokens", None)
      if getattr(cache_config, "kv_cache_size_tokens", None) is not None
      else response.kv_cache_size_tokens
    )
    cache_config.kv_cache_max_concurrency = (
      getattr(cache_config, "kv_cache_max_concurrency", None)
      if getattr(cache_config, "kv_cache_max_concurrency", None) is not None
      else response.kv_cache_max_concurrency
    )
    if response.dp_stats_address is not None:
      if self.stats_update_address is None:
        self.stats_update_address = response.dp_stats_address
      else:
        assert response.dp_stats_address == self.stats_update_address
    # Do not issue get_weight_version here. The READY barrier is also the
    # transport-repair probe; defer utility RPCs until the channel has
    # been independently verified after restore.

  async def replace_api_zmq_context(self: AsyncMPClient) -> None:
    """Replace restored pyzmq state before rebuilding API sockets.

    Recreating only the sockets can leave pyzmq's asyncio FD watchers and
    libzmq IO thread attached to state captured by CRIU.  The API client
    owns the local context for these sockets, so replace that context and
    let the normal output-task setup register fresh asyncio watchers.
    """

    resources = self.resources
    tasks = (resources.output_queue_task, resources.stats_update_task)
    for task in tasks:
      if task is not None and not task.done():
        task.cancel()
    await asyncio.gather(
      *(task for task in tasks if task is not None),
      return_exceptions=True,
    )

    sockets = (
      self.input_socket,
      resources.output_socket,
      resources.first_req_send_socket,
      resources.first_req_rcv_socket,
      resources.stats_update_socket,
    )
    for socket in sockets:
      if socket is not None:
        socket.close(linger=0)

    old_context = resources.ctx
    # Keep the client and its resource holder on the same asyncio-aware
    # context.  Wrapping a separate synchronous context leaves pyzmq's
    # asyncio FD watchers attached to the pre-CRIU context and can make
    # the output task stop consuming after the first restored response.
    new_async_context = zmq.asyncio.Context(io_threads=2)

    # Keep the resource finalizer pointed at the new context and clear
    # sockets/tasks that belonged to the restored context. vLLM recreates
    # these lazily when needed.
    self.ctx = new_async_context
    resources.ctx = new_async_context
    resources.output_socket = None
    resources.output_queue_task = None
    resources.first_req_send_socket = None
    resources.first_req_rcv_socket = None
    resources.stats_update_socket = None
    resources.stats_update_task = None

    try:
      old_context.destroy(linger=0)
    except zmq.ZMQError:
      # A restored auxiliary socket may still be owned by another vLLM
      # component. The new context is still safe to use for reconnect.
      pass

  async def start_restore_output_reader(
    self: AsyncMPClient, output_address: str
  ) -> None:
    """Install a fresh synchronous output reader after CRIU restore.

    The pre-restore AsyncMPClient output task depends on pyzmq's asyncio
    FD watcher state.  That state can remain alive enough to accept the
    first restored message while no longer draining the PULL socket.  A
    dedicated synchronous reader has its own context, socket, and OS
    thread, then hands decoded results back to the API event loop.
    """

    resources = self.resources
    loop = asyncio.get_running_loop()
    decoder = self.decoder
    utility_results = self.utility_results
    outputs_queue = self.outputs_queue
    output_handler = getattr(self.__class__, "process_engine_outputs", None)
    notification_handler = getattr(
      self.__class__, "eep_process_engine_core_notification", None
    )
    self_ref = weakref.ref(self)
    reader_context = zmq.Context(io_threads=2)
    output_socket = make_zmq_socket(
      reader_context,
      output_address,
      zmq.PULL,
      bind=True,
    )
    transport_debug = os.environ.get(
      "VLLM_LIFECYCLE_DEBUG_TRANSPORT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if transport_debug:
      LOG.warning("installed restored API output reader at %s", output_address)
    stop_event = threading.Event()

    def call_on_loop(callback, *args):
      if not loop.is_closed():
        loop.call_soon_threadsafe(callback, *args)

    def read_outputs() -> None:
      try:
        while not stop_event.is_set():
          frames = output_socket.recv_multipart(copy=False)
          resources.validate_alive(frames)
          outputs = decoder.decode(frames)
          if transport_debug:
            LOG.warning(
              "API restored output reader received utility=%s outputs=%s stats=%s",
              outputs.utility_output is not None,
              len(outputs.outputs),
              outputs.scheduler_stats is not None,
            )
          utility_output = outputs.utility_output
          if utility_output is not None:
            if (
              utility_output.call_id == EEP_NOTIFICATION_CALL_ID
              and notification_handler is not None
            ):
              client = self_ref()
              if client is None:
                return
              if utility_output.result is None:
                continue
              notification_data = utility_output.result.result
              call_on_loop(
                lambda: asyncio.create_task(
                  notification_handler(client, notification_data)
                )
              )
            elif utility_output.call_id == FT_STATUS_CALL_ID:
              client = self_ref()
              if client is None:
                return
              if utility_output.result is not None:
                call_on_loop(
                  client._engine_status.__setitem__,
                  outputs.engine_index,
                  utility_output.result.result,
                )
            else:
              call_on_loop(
                _process_utility_output,
                utility_output,
                utility_results,
              )
            continue

          client = self_ref()
          if client is None:
            return
          if output_handler is not None:
            future = asyncio.run_coroutine_threadsafe(
              output_handler(client, outputs), loop
            )
            future.result()
          if outputs.outputs or outputs.scheduler_stats:
            call_on_loop(outputs_queue.put_nowait, outputs)
      except Exception as error:
        if not stop_event.is_set():
          LOG.exception("restored API output reader failed")
          call_on_loop(outputs_queue.put_nowait, error)
      finally:
        output_socket.close(linger=0)
        reader_context.destroy(linger=0)

    reader_thread = threading.Thread(
      target=read_outputs,
      name="VLLMCRIUOutputReader",
      daemon=True,
    )
    reader_thread.start()

    async def reader_lifetime():
      try:
        await asyncio.Event().wait()
      finally:
        stop_event.set()
        output_socket.close(linger=0)
        reader_thread.join(timeout=2)

    resources.output_queue_task = asyncio.create_task(
      reader_lifetime(), name="VLLMCRIUOutputReaderLifetime"
    )
    # Keep the reader resources reachable until the task is cancelled.
    resources._vllm_criu_output_reader = (  # type: ignore[attr-defined]
      reader_context,
      output_socket,
      stop_event,
      reader_thread,
    )

  async def reconnect_zmq(self: AsyncMPClient) -> dict[str, str]:
    input_socket = self.input_socket
    output_socket = self.resources.output_socket
    if input_socket is None or output_socket is None:
      raise RuntimeError("vLLM EngineCore sockets are not initialized")

    input_address = input_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
    output_address = output_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
    LOG.warning(
      "rewiring API ZMQ sockets: input=%s output=%s context_rehook=%s",
      input_address,
      output_address,
      os.environ.get("VLLM_LIFECYCLE_API_CONTEXT_REHOOK", "0"),
    )

    context_rehook = os.environ.get(
      "VLLM_LIFECYCLE_API_CONTEXT_REHOOK", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if context_rehook:
      await replace_api_zmq_context(self)
    else:
      output_task = self.resources.output_queue_task
      if output_task is not None and not output_task.done():
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)
      input_socket.close(linger=0)
      output_socket.close(linger=0)
      self.resources.output_queue_task = None

    # CRIU can leave the restored ROUTER's peer identity registered even
    # after the API-side socket is recreated.  Allow the freshly restored
    # EngineCore DEALER to take that identity over; this is required even
    # for a single engine, where vLLM normally disables handover.
    router_handover = True
    self.input_socket = self.resources.input_socket = make_zmq_socket(
      self.ctx,
      input_address,
      zmq.ROUTER,
      bind=True,
      router_handover=router_handover,
    )
    if context_rehook:
      await start_restore_output_reader(self, output_address)
    else:
      self.resources.output_socket = make_zmq_socket(
        self.ctx,
        output_address,
        zmq.PULL,
        bind=True,
      )
      self._ensure_output_queue_task()
    return {"input": input_address, "output": output_address}

  async def wait_for_enginecore_ready(self: AsyncMPClient, timeout: float) -> int:
    """Consume READY payloads after a restored EngineCore reconnect."""
    expected = set(self.core_engines)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while expected:
      remaining = max(0.1, deadline - loop.time())
      frames = await asyncio.wait_for(
        self.input_socket.recv_multipart(copy=False), timeout=remaining
      )
      identity_frame, payload = frames
      # copy=False returns zmq.Frame objects. Normalize the routing
      # identity before comparing it with MPClient's bytes identities.
      identity = bytes(identity_frame)
      if identity not in expected:
        if len(expected) != 1:
          continue
        # A restored EngineCore uses a fresh DEALER identity so the
        # rebound ROUTER cannot select a stale pre-CRIU route.
        self.core_engines = [identity]
        self.core_engine = identity
        expected = {identity}
      await apply_ready_response_async(self, bytes(payload))
      expected.remove(identity)
    return len(self.core_engines)

  AsyncMPClient.reconnect_zmq = reconnect_zmq  # type: ignore[attr-defined]
  AsyncMPClient.wait_for_enginecore_ready = wait_for_enginecore_ready  # type: ignore[attr-defined]

  @api_router.router.post("/lifecycle/reconnect_zmq")
  async def reconnect_zmq_endpoint(raw_request: Request):
    client = raw_request.app.state.engine_client
    engine_core = getattr(client, "engine_core", client)
    # AsyncMPClient's output-reader cancellation path publishes a
    # synthetic EngineDeadError into outputs_queue. Stop AsyncLLM's
    # consumer first so it cannot mistake that transport teardown marker
    # for a real EngineCore failure.
    output_handler = getattr(client, "output_handler", None)
    if output_handler is not None and not output_handler.done():
      output_handler.cancel()
      await asyncio.gather(output_handler, return_exceptions=True)
      client.output_handler = None
    try:
      addresses = await engine_core.reconnect_zmq()
    except Exception:
      LOG.exception("API-side ZMQ reconnect failed")
      raise
    # CRIU stops the EngineCore during dump.  vLLM's output task observes
    # that disconnect and marks the AsyncLLM client dead; clear that
    # transient state and recreate the reader before issuing wake RPCs.
    engine_core.resources.engine_dead = False
    pending_outputs = []
    while True:
      try:
        output = engine_core.outputs_queue.get_nowait()
      except asyncio.QueueEmpty:
        break
      if not isinstance(output, EngineDeadError):
        pending_outputs.append(output)
    for output in pending_outputs:
      engine_core.outputs_queue.put_nowait(output)
    client._run_output_handler()
    return JSONResponse(content={"ok": True, "addresses": addresses})

  @api_router.router.post("/lifecycle/wait_for_enginecore_ready")
  async def wait_for_enginecore_ready_endpoint(raw_request: Request):
    client = raw_request.app.state.engine_client
    engine_core = getattr(client, "engine_core", client)
    engines = await engine_core.wait_for_enginecore_ready(120.0)
    return JSONResponse(content={"ok": True, "engines": engines})

  @api_router.router.post("/lifecycle/reinitialize_after_restore")
  async def reinitialize_after_restore_endpoint(raw_request: Request):
    client = raw_request.app.state.engine_client
    engine_core = getattr(client, "engine_core", client)
    try:
      result = await engine_core.collective_rpc_async(
        "vllm_criu_reinitialize_after_restore"
      )
    except Exception:
      LOG.exception("restored EngineCore allocator RPC failed")
      raise
    return JSONResponse(content={"ok": True, "result": result})


def main() -> None:
  if os.environ.get("VLLM_LIFECYCLE_ZMQ_TRANSPORT", "ipc") == "tcp":
    install_tcp_transport()
  install_enginecore_restore_patch()
  install_restore_reconnect()

  from vllm.entrypoints.cli.main import main as vllm_main

  vllm_main()


if __name__ == "__main__":
  main()
