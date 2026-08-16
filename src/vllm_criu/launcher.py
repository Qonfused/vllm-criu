## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from __future__ import annotations

import json
import logging
import os
import pathlib
import socket
import socketserver
import signal
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .checkpoint import CheckpointError, CommandRunner, Criu, CudaCheckpoint, Tools
from .config import Settings
from .process_tree import (
  enable_child_subreaper,
  find_engine,
  reap_children,
  terminate_tree,
  terminate_resource_trackers,
)
from .runtime import child_runtime
from .vllm_api import VllmApi

LOG = logging.getLogger("vllm-launcher")

settings = Settings.from_env()
api = VllmApi(settings.vllm_host, settings.vllm_port)
runner = CommandRunner()
state_lock = threading.RLock()
operation_lock = threading.Lock()
state = "starting"
vllm_process: subprocess.Popen[bytes] | _RestoredProcess | None = None


def set_state(value: str) -> None:
  global state
  with state_lock:
    state = value
  LOG.info("state=%s", value)


def get_state() -> str:
  with state_lock:
    return state


def capabilities() -> dict[str, object]:
  criu = os.environ.get("CRIU_PATH") or shutil.which("criu")
  cuda = os.environ.get("CUDA_CHECKPOINT_PATH") or shutil.which("cuda-checkpoint")
  return {
    "criu": criu,
    "cudaCheckpoint": cuda,
    "checkpointReady": bool(criu and cuda),
    "checkpointDir": str(settings.checkpoint_dir),
    "resourceTrackerMode": settings.resource_tracker_mode,
  }


def vllm_command() -> list[str]:
  return [
    "/usr/bin/python3",
    "-m",
    "vllm_criu.vllm_server",
    "serve",
    settings.model_id,
    "--served-model-name",
    settings.model_name,
    "--port",
    str(settings.vllm_port),
    "--host",
    "0.0.0.0",
    "--uds",
    str(settings.vllm_uds_path),
    "--trust-remote-code",
    "--max-model-len",
    settings.max_model_len,
    "--gpu-memory-utilization",
    settings.gpu_memory_utilization,
    "--max-num-seqs",
    settings.max_num_seqs,
    "--chat-template-content-format",
    "openai",
    "--kv-cache-dtype",
    settings.kv_cache_dtype,
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--enable-sleep-mode",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "qwen3",
    "--speculative-config",
    json.dumps({"method": "mtp", "num_speculative_tokens": 3}),
    "--compilation-config",
    json.dumps({"cudagraph_capture_sizes": [4, 8], "cudagraph_num_of_warmups": 1}),
    "--default-chat-template-kwargs",
    json.dumps({"enable_thinking": True, "preserve_thinking": True}),
    "--override-generation-config",
    json.dumps(
      {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_tokens": int(settings.max_tokens),
        "max_new_tokens": int(settings.max_tokens),
      }
    ),
    *json.loads(os.environ.get("VLLM_EXTRA_ARGS_JSON", "[]")),
  ]


def start_vllm() -> subprocess.Popen[bytes]:
  global vllm_process
  settings.vllm_uds_path.unlink(missing_ok=True)
  command = vllm_command()
  LOG.info("starting vLLM: %s", " ".join(command))
  runtime = child_runtime(settings.resource_tracker_mode)
  process = subprocess.Popen(command, env=runtime.environment, pass_fds=runtime.pass_fds)
  with state_lock:
    vllm_process = process
  set_state("running")
  return process


class _ApiProxyHandler(socketserver.BaseRequestHandler):
  def handle(self) -> None:
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      upstream.connect(str(settings.vllm_uds_path))
    except OSError:
      upstream.close()
      return

    client = self.request

    # Do not leave long-lived proxy↔vLLM Unix streams in the CRIU tree.
    # vLLM's API honors this request header and closes each upstream
    # connection after the response (including after a streaming EOF).
    request = b""
    while b"\r\n\r\n" not in request and len(request) < 131072:
      chunk = client.recv(65536)
      if not chunk:
        upstream.close()
        return
      request += chunk
    header, separator, remainder = request.partition(b"\r\n\r\n")
    header_lines = [
      line for line in header.split(b"\r\n")
      if not line.lower().startswith(b"connection:")
    ]
    upstream.sendall(b"\r\n".join(header_lines) + b"\r\nConnection: close\r\n\r\n" + remainder)

    def relay(source: socket.socket, target: socket.socket) -> None:
      try:
        while data := source.recv(65536):
          target.sendall(data)
      except OSError:
        pass
      finally:
        try:
          target.shutdown(socket.SHUT_WR)
        except OSError:
          pass

    threads = [
      threading.Thread(target=relay, args=(client, upstream), daemon=True),
      threading.Thread(target=relay, args=(upstream, client), daemon=True),
    ]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    upstream.close()


class _ApiProxyServer(socketserver.ThreadingTCPServer):
  allow_reuse_address = True
  daemon_threads = True


def start_api_proxy() -> _ApiProxyServer:
  server = _ApiProxyServer(("0.0.0.0", settings.vllm_port), _ApiProxyHandler)
  threading.Thread(target=server.serve_forever, name="vllm-api-proxy", daemon=True).start()
  LOG.info("stable API proxy listening on 0.0.0.0:%s -> %s", settings.vllm_port, settings.vllm_uds_path)
  return server


def checkpoint_tools() -> tuple[Criu, CudaCheckpoint]:
  discovered = Tools.discover()
  return (
    Criu(
      discovered.criu,
      runner,
      settings.checkpoint_timeout,
      tcp_close=settings.criu_tcp_close,
    ),
    CudaCheckpoint(discovered.cuda_checkpoint, runner, settings.checkpoint_timeout),
  )


def _restore_rpc_with_process_watch(
  operation, restored_pid: int, timeout: float
) -> None:
  """Run a restore RPC while detecting a dead restored EngineCore."""

  result: list[BaseException | None] = [None]

  def invoke() -> None:
    try:
      operation()
    except BaseException as error:
      result[0] = error

  thread = threading.Thread(target=invoke, name="vllm-restore-rpc", daemon=True)
  thread.start()
  deadline = time.monotonic() + timeout
  while thread.is_alive():
    if not pathlib.Path(f"/proc/{restored_pid}").exists():
      raise CheckpointError(
        "restored EngineCore exited while reinitializing CUDA state"
      )
    if time.monotonic() >= deadline:
      raise CheckpointError("timed out reinitializing restored CUDA state")
    thread.join(0.25)
  if result[0] is not None:
    raise result[0]


def _wait_for_enginecore_local_restore(
  restore_status: pathlib.Path, restored_pid: int, timeout: float
) -> None:
  """Wait for the process-local V1 EngineCore restore sequence."""

  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if restore_status.exists():
      detail = restore_status.read_text(errors="replace")
      if detail.startswith("done"):
        return
      if detail.startswith("failed"):
        raise CheckpointError(
          "V1 EngineCore-local restore failed:\n" + detail
        )
    if not pathlib.Path(f"/proc/{restored_pid}").exists():
      raise CheckpointError(
        "restored EngineCore exited during local restore"
      )
    time.sleep(0.25)
  raise CheckpointError("timed out waiting for V1 EngineCore-local restore")


def _archive_failed_checkpoint() -> pathlib.Path | None:
  """Move a failed image out of the active slot before a fresh restart."""

  checkpoint_dir = settings.checkpoint_dir
  if not checkpoint_dir.exists():
    return None
  if not any(checkpoint_dir.iterdir()):
    checkpoint_dir.rmdir()
    return None

  stamp = int(time.time())
  archive = checkpoint_dir.parent / f"{checkpoint_dir.name}.stale-failed-{stamp}"
  suffix = 1
  while archive.exists():
    archive = checkpoint_dir.parent / (
      f"{checkpoint_dir.name}.stale-failed-{stamp}-{suffix}"
    )
    suffix += 1
  checkpoint_dir.replace(archive)
  LOG.warning("archived failed CRIU checkpoint at %s", archive)
  return archive


def checkpoint_vllm() -> dict[str, object]:
  global vllm_process
  if not operation_lock.acquire(blocking=False):
    raise CheckpointError("another launcher operation is already in progress")
  cuda_pid: int | None = None
  cuda_checkpointed = False
  staging_dir: pathlib.Path | None = None
  try:
    if not vllm_process or not hasattr(vllm_process, "poll") or vllm_process.poll() is not None:
      raise CheckpointError("vLLM is not running")
    if settings.checkpoint_dir.exists() and any(settings.checkpoint_dir.iterdir()):
      raise CheckpointError(f"checkpoint directory is not empty: {settings.checkpoint_dir}")
    settings.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = settings.checkpoint_dir.parent / (
      f".{settings.checkpoint_dir.name}.partial-{os.getpid()}-{int(time.time())}"
    )
    staging_dir.mkdir()

    root_pid = vllm_process.pid
    cuda_pid = find_engine(root_pid)
    if not cuda_pid:
      raise CheckpointError(f"could not find VLLM::EngineCore below pid {root_pid}")

    if settings.checkpoint_sleep_level:
      api.sleep_level(settings.checkpoint_sleep_level, settings.checkpoint_timeout)
    else:
      LOG.info("checkpointing with vLLM GPU allocations resident")
    # vLLM exposes an explicit communicator lifecycle for process
    # checkpointing.  Prepare it after level-2 sleep and before CUDA/CRIU
    # freeze the EngineCore, so restored NCCL/device communicators do not
    # retain stale driver-side state.
    LOG.info("preparing vLLM device communicators for checkpoint")
    api.checkpoint_prepare(settings.checkpoint_timeout)
    if settings.resource_tracker_mode == "terminate":
      terminate_resource_trackers(root_pid)

    criu, cuda = checkpoint_tools()
    cuda.lock(cuda_pid)
    cuda.checkpoint(cuda_pid)
    cuda_checkpointed = True
    set_state("checkpointing")
    criu.dump(root_pid, staging_dir)
    if settings.checkpoint_dir.exists():
      settings.checkpoint_dir.rmdir()
    staging_dir.replace(settings.checkpoint_dir)
    staging_dir = None
    reap_children()
    set_state("checkpointed")
    return {"ok": True, "state": "checkpointed", "rootPid": root_pid, "cudaPid": cuda_pid}
  except Exception:
    if staging_dir is not None and staging_dir.exists():
      shutil.rmtree(staging_dir)
    if cuda_checkpointed and cuda_pid:
      try:
        _, cuda = checkpoint_tools()
        cuda.rollback(cuda_pid)
        api.wake_level2(settings.checkpoint_timeout)
      except Exception:
        LOG.exception("could not restore level-2 state after failed checkpoint")
    set_state("running")
    raise
  finally:
    operation_lock.release()


def restore_vllm() -> dict[str, object]:
  global vllm_process
  if not operation_lock.acquire(blocking=False):
    raise CheckpointError("another launcher operation is already in progress")
  restored_pid: int | None = None
  restore_marker = settings.checkpoint_dir.parent / ".vllm-criu-restore-request"
  restore_pending = settings.checkpoint_dir.parent / ".vllm-criu-restore-pending"
  restore_status = settings.checkpoint_dir.parent / ".vllm-criu-restore-status"
  try:
    if not settings.checkpoint_dir.exists() or not any(settings.checkpoint_dir.iterdir()):
      raise CheckpointError(f"no checkpoint found at {settings.checkpoint_dir}")
    restore_marker.unlink(missing_ok=True)
    restore_pending.unlink(missing_ok=True)
    restore_status.unlink(missing_ok=True)
    set_state("restoring")
    criu, cuda = checkpoint_tools()
    # Create this outside the image before CRIU restores the process tree.
    # The restored V1 liveness monitor can then distinguish the expected
    # pre-restore EngineCore exit from a genuine post-restore failure.
    restore_pending.touch()
    restored_pid = criu.restore(settings.checkpoint_dir)
    deadline = time.monotonic() + settings.checkpoint_timeout
    while not pathlib.Path(f"/proc/{restored_pid}").exists() and time.monotonic() < deadline:
      time.sleep(0.1)
    cuda_pid = find_engine(restored_pid)
    if not cuda_pid:
      raise CheckpointError(f"could not find restored EngineCore below pid {restored_pid}")
    cuda.restore_if_needed(cuda_pid)
    # This marker is outside the checkpoint image. The restored EngineCore
    # uses it both to rehook sockets and to run the V1-local worker restore
    # sequence after CUDA state has been restored.
    restore_marker.touch()
    deadline = time.monotonic() + settings.checkpoint_timeout
    ready = api.ready()
    while not ready and time.monotonic() < deadline:
      time.sleep(0.25)
      ready = api.ready()
    if not ready:
      raise CheckpointError("restored vLLM API did not become ready")
    enginecore_patch_enabled = os.environ.get(
      "VLLM_LIFECYCLE_ENGINECORE_PATCH", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if enginecore_patch_enabled or os.environ.get(
      "VLLM_LIFECYCLE_ZMQ_TRANSPORT", "ipc"
    ) == "tcp":
      LOG.info("repairing restored API↔EngineCore ZMQ channels")
      api.reconnect_zmq(settings.checkpoint_timeout)
    if enginecore_patch_enabled:
      LOG.info("requesting restored EngineCore socket reconnect")
      os.kill(cuda_pid, signal.SIGUSR1)
      LOG.info("waiting for restored EngineCore READY handshake")
      api.wait_for_enginecore_ready(settings.checkpoint_timeout)
    if settings.checkpoint_sleep_level:
      LOG.info("waiting for V1 EngineCore-local level-2 restore")
      _wait_for_enginecore_local_restore(
        restore_status, cuda_pid, settings.checkpoint_timeout
      )
    else:
      LOG.info("GPU allocations were resident during checkpoint")
    with state_lock:
      vllm_process = _RestoredProcess(restored_pid)
    # A restored image is single-use: the live process now owns the
    # runtime state, and retaining the consumed image would prevent the
    # next suspend from staging a new checkpoint. Keep the active slot
    # available for the next cycle instead of accumulating multi-GB
    # superseded images.
    shutil.rmtree(settings.checkpoint_dir)
    LOG.info("removed consumed CRIU checkpoint %s", settings.checkpoint_dir)
    LOG.info("restored V1 process tree is alive and EngineCore-local recovery completed")
    set_state("running")
    threading.Thread(target=reap_vllm, daemon=True).start()
    return {"ok": True, "state": "running", "rootPid": restored_pid, "cudaPid": cuda_pid}
  except Exception:
    if restored_pid is not None and pathlib.Path(f"/proc/{restored_pid}").exists():
      terminate_tree(restored_pid)
    reap_children()
    if settings.restore_fallback_fresh:
      # The marker is deliberately outside the CRIU image so the
      # restored EngineCore can see it.  Remove it before starting a
      # cold process as fallback; otherwise that new process sees the
      # stale marker during its normal startup and incorrectly runs the
      # restore-only worker rebootstrap while its model is already
      # resident.
      restore_marker.unlink(missing_ok=True)
      restore_status.unlink(missing_ok=True)
      LOG.warning(
        "CRIU restore could not recover CUDA state; starting a fresh vLLM "
        "process as the configured fallback",
        exc_info=True,
      )
      try:
        _archive_failed_checkpoint()
        process = start_vllm()
        deadline = time.monotonic() + settings.checkpoint_timeout
        while not api.ready() and time.monotonic() < deadline:
          time.sleep(0.25)
        if api.ready():
          set_state("running")
          return {
            "ok": True,
            "state": "running",
            "fallback": "fresh_vllm_process",
            "vllmPid": process.pid,
          }
        process.terminate()
      except Exception:
        LOG.exception("fresh vLLM fallback failed")
    set_state("checkpointed")
    raise
  finally:
    restore_marker.unlink(missing_ok=True)
    restore_pending.unlink(missing_ok=True)
    operation_lock.release()


class _RestoredProcess:
  def __init__(self, pid: int) -> None:
    self.pid = pid

  def poll(self) -> int | None:
    return None if pathlib.Path(f"/proc/{self.pid}").exists() else 1

  def wait(self, timeout: float | None = None) -> int:
    deadline = None if timeout is None else time.monotonic() + timeout
    while pathlib.Path(f"/proc/{self.pid}").exists():
      if deadline is not None and time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired("restored-vllm", timeout)
      time.sleep(0.1)
    return 0

  def terminate(self) -> None:
    try:
      os.kill(self.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass


class Handler(BaseHTTPRequestHandler):
  server_version = "vllm-launcher/0.2"

  def log_message(self, format: str, *args: object) -> None:
    LOG.info("control: " + format, *args)

  def send_json(self, status_code: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode()
    self.send_response(status_code)
    self.send_header("content-type", "application/json")
    self.send_header("content-length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # noqa: N802
    if self.path == "/launcher/healthz":
      process = vllm_process
      current_state = get_state()
      healthy = current_state in {"checkpointed", "restoring"} or (
        current_state == "running" and api.ready()
      )
      self.send_json(
        200 if healthy else 503,
        {
          "ok": healthy,
          "state": current_state,
          "vllmPid": process.pid if process else None,
          "capabilities": capabilities(),
        },
      )
      return
    if self.path == "/launcher/capabilities":
      self.send_json(200, {"ok": True, "state": get_state(), "capabilities": capabilities()})
      return
    self.send_json(404, {"ok": False, "error": "not found"})

  def do_POST(self) -> None:  # noqa: N802
    try:
      if self.path == "/launcher/suspend":
        self.send_json(200, checkpoint_vllm())
        return
      if self.path == "/launcher/resume":
        self.send_json(200, restore_vllm())
        return
      self.send_json(404, {"ok": False, "error": "not found"})
    except Exception as error:
      LOG.exception("control operation failed")
      self.send_json(500, {"ok": False, "error": str(error), "state": get_state()})


def reap_vllm() -> None:
  process = vllm_process
  if process is None or not hasattr(process, "wait"):
    return
  return_code = process.wait()
  if get_state() in {"checkpointing", "checkpointed", "restoring"}:
    LOG.info("checkpointed vLLM tree exited with code %s", return_code)
    return
  set_state("failed" if return_code else "stopped")
  LOG.error("vLLM exited with code %s", return_code)


def forward_signal(signum: int, _frame: object) -> None:
  process = vllm_process
  if process and hasattr(process, "send_signal") and process.poll() is None:
    process.send_signal(signum)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  signal.signal(signal.SIGTERM, forward_signal)
  signal.signal(signal.SIGINT, forward_signal)
  enable_child_subreaper()
  LOG.info("checkpoint capabilities: %s", capabilities())
  api_proxy = start_api_proxy()
  # A forced container stop can interrupt restore_vllm before its finally
  # block removes the out-of-image marker.  Never let that stale marker
  # make a fresh vLLM process re-enter the restore socket path on startup.
  restore_marker = settings.checkpoint_dir.parent / ".vllm-criu-restore-request"
  restore_status = settings.checkpoint_dir.parent / ".vllm-criu-restore-status"
  restore_marker.unlink(missing_ok=True)
  restore_status.unlink(missing_ok=True)

  if settings.resume_on_start and settings.checkpoint_dir.exists() and any(settings.checkpoint_dir.iterdir()):
    set_state("checkpointed")
    try:
      restore_vllm()
    except Exception:
      LOG.exception("could not resume checkpoint; starting a fresh vLLM process")
      start_vllm()
  else:
    start_vllm()

  threading.Thread(target=reap_vllm, daemon=True).start()
  server = ThreadingHTTPServer((settings.control_host, settings.control_port), Handler)
  LOG.info("control API listening on %s:%s", settings.control_host, settings.control_port)
  try:
    server.serve_forever()
  finally:
    server.server_close()
    api_proxy.shutdown()
    api_proxy.server_close()
    if vllm_process and hasattr(vllm_process, "poll") and vllm_process.poll() is None:
      vllm_process.terminate()
      vllm_process.wait(timeout=10)


if __name__ == "__main__":
  main()
