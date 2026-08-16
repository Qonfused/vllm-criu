## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""V1 EngineCore lifecycle orchestration for CRIU restore."""

from __future__ import annotations

import gc
import logging
import os
import signal
import threading
import time
from functools import wraps
from pathlib import Path

from .diagnostics import (
  _append_restore_diagnostic,
  _install_enginecore_crash_diagnostic,
  _restore_active_paths,
)
from .memory import (
  _discard_stale_engine_dead_outputs,
  _install_allocator_restore_patch,
  _rebootstrap_worker_after_restore,
  _reinitialize_worker_allocations,
  _wake_worker_allocations_after_restore,
)
from .graphs import (
  graph_preservation_enabled,
  mark_restored_graphs,
)
from .sockets import _input_sockets, _output_sockets

LOG = logging.getLogger("vllm-criu.enginecore-patch")

_INSTALLED = False


def install_enginecore_restore_patch(*, force: bool = False) -> bool:
  """Install the EngineCore socket patch when explicitly enabled."""

  global _INSTALLED
  if os.environ.get("VLLM_LIFECYCLE_ENGINECORE_PATCH", "0").lower() not in {
    "1",
    "true",
    "yes",
    "on",
  }:
    return False
  if _INSTALLED and not force:
    return True

  _install_allocator_restore_patch()

  from vllm.v1.engine.core import EngineCore, EngineCoreProc
  from vllm.v1.engine import EngineCoreRequestType

  _install_enginecore_crash_diagnostic()

  # multiprocessing.Process sentinels are not durable across a CRIU dump:
  # the sentinel can remain readable because the pre-dump EngineCore exited,
  # even though CRIU has restored a live process with the same PID.  V1's
  # stock monitor treats that stale readiness as a real engine failure. Use
  # the restored PID's proc entry while the lifecycle markers are active and
  # retain the same shutdown behavior for genuine exits.
  from vllm.v1.engine.utils import CoreEngineProcManager

  if not getattr(CoreEngineProcManager, "_vllm_criu_proc_monitor", False):
    original_monitor_engine_liveness = (
      CoreEngineProcManager.monitor_engine_liveness
    )

    def monitor_engine_liveness(self):
      pending_marker, restore_marker = _restore_active_paths()
      processes = list(self.processes)
      while not self.manager_stopped.is_set():
        if pending_marker.exists() or restore_marker.exists():
          time.sleep(0.25)
          continue
        dead = [
          process
          for process in processes
          if not Path(f"/proc/{process.pid}").exists()
        ]
        if dead:
          self.failed_proc_name = dead[0].name
          break
        time.sleep(1.0)
      if not processes or self.manager_stopped.is_set():
        return
      self.shutdown()

    CoreEngineProcManager.monitor_engine_liveness = monitor_engine_liveness
    CoreEngineProcManager._vllm_criu_proc_monitor = True

  if not getattr(EngineCoreProc, "_vllm_criu_failure_diagnostic", False):
    original_send_engine_dead = EngineCoreProc._send_engine_dead
    original_process_engine_step = EngineCoreProc._process_engine_step

    @wraps(original_send_engine_dead)
    def send_engine_dead_with_diagnostic(self, *args, **kwargs):
      import traceback

      _append_restore_diagnostic(
        ".vllm-criu-enginecore-failure.log",
        "\n=== _send_engine_dead ===\n"
        f"pid={os.getpid()} generation="
        f"{getattr(self, '_vllm_criu_reconnect_generation', -1)} "
        f"shutdown_state={getattr(self, 'shutdown_state', None)}\n"
        + "".join(traceback.format_stack()),
      )
      return original_send_engine_dead(self, *args, **kwargs)

    @wraps(original_process_engine_step)
    def process_engine_step_with_diagnostic(self, *args, **kwargs):
      import traceback

      try:
        return original_process_engine_step(self, *args, **kwargs)
      except BaseException:
        _append_restore_diagnostic(
          ".vllm-criu-enginecore-failure.log",
          "\n=== _process_engine_step exception ===\n"
          f"pid={os.getpid()} generation="
          f"{getattr(self, '_vllm_criu_reconnect_generation', -1)}\n"
          + traceback.format_exc(),
        )
        raise

    EngineCoreProc._send_engine_dead = send_engine_dead_with_diagnostic
    EngineCoreProc._process_engine_step = process_engine_step_with_diagnostic
    EngineCoreProc._vllm_criu_failure_diagnostic = True

  original_init = EngineCoreProc.__init__

  def patched_init(self, *args, **kwargs):
    self._vllm_criu_reconnect_generation = 0
    self._vllm_criu_supervisor_started = False
    self._vllm_criu_restarted_generation = 0
    self._vllm_criu_local_restore_complete = False
    self._vllm_criu_local_restore_in_progress = False
    self._vllm_criu_restore_thread_started = False

    def request_reconnect(_signum, _frame):
      self._vllm_criu_reconnect_generation += 1
      for socket in (
        getattr(self, "_vllm_criu_input_sockets", [])
        + getattr(self, "_vllm_criu_output_sockets", [])
      ):
        try:
          socket.close(linger=0)
        except Exception:
          pass
      # Wake the EngineCore main loop directly. Waiting for the
      # restored input thread to enqueue this item is exactly the race
      # this rehook is intended to avoid.
      try:
        self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
      except Exception:
        LOG.exception("failed to wake EngineCore after restore signal")

    try:
      signal.signal(signal.SIGUSR1, request_reconnect)
    except ValueError:
      # EngineCore is normally initialized on the child main thread,
      # but leave startup usable if a future vLLM version constructs it
      # from a worker thread.
      LOG.warning("could not install SIGUSR1 EngineCore reconnect handler")
    else:
      LOG.warning("installed CRIU EngineCore SIGUSR1 reconnect handler")
    original_init(self, *args, **kwargs)

    restore_marker = (
      Path(os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current"))
      .parent
      / ".vllm-criu-restore-request"
    )
    restore_status = restore_marker.with_name(".vllm-criu-restore-status")

    def run_local_restore() -> None:
      """Recover the V1 worker without using restored utility ZMQ."""

      if (
        self._vllm_criu_local_restore_complete
        or self._vllm_criu_local_restore_in_progress
        or not restore_marker.exists()
      ):
        return

      self._vllm_criu_local_restore_in_progress = True
      restore_status.unlink(missing_ok=True)
      restore_fault_log = restore_status.with_name(".vllm-criu-restore-fault.log")
      restore_fault_log.unlink(missing_ok=True)
      LOG.warning("starting V1 EngineCore-local CRIU restore sequence")
      try:
        # CRIU can restore EngineCore with the launcher's stdout pipe
        # closed. vLLM's distributed setup flushes sys.stdout inside
        # suppress_stdout(), so leave stdout on a valid sink for the
        # remainder of this restored process. The restored stderr
        # descriptor can be stale as well (tqdm flushes it while
        # loading weights), so redirect both streams.
        import sys

        try:
          sys.stdout.flush()
        except (BrokenPipeError, OSError):
          pass
        # A flush can succeed while the descriptor itself is still
        # unusable after CRIU. Replace it unconditionally so
        # suppress_stdout() never operates on the stale pipe.
        sys.stdout = open(os.devnull, "w")
        try:
          sys.stderr.flush()
        except (BrokenPipeError, OSError):
          pass
        sys.stderr = open(os.devnull, "w")

        from vllm.config import set_current_vllm_config
        import faulthandler

        fault_file = restore_fault_log.open("w")
        self._vllm_criu_restore_fault_file = fault_file
        faulthandler.enable(file=fault_file, all_threads=True)

        # Direct EngineCore-local calls bypass the normal vLLM
        # utility-RPC wrapper that establishes this context.
        with set_current_vllm_config(self.vllm_config):
          restore_status.write_text("stage checkpoint_restore\n")
          # checkpoint_prepare cleaned the distributed environment
          # before CRIU. The patched worker hook recreates it here,
          # without using the restored API-side ZMQ client.
          self.collective_rpc("checkpoint_restore")
          restore_status.write_text("stage reinitialize\n")
          self.collective_rpc("vllm_criu_reinitialize_after_restore")
          restore_status.write_text("stage reload_weights\n")
          # Level 2 discarded weights and KV cache. Reload the
          # weights through the restored worker while retaining the
          # initialized V1 process/runtime and compiler artifacts.
          self.collective_rpc("reload_weights")
          if os.environ.get(
            "VLLM_LIFECYCLE_RESTORE_NATIVE_WAKE", "0"
          ).lower() in {"1", "true", "yes", "on"}:
            restore_status.write_text("stage kv_cache_wake\n")
            self.collective_rpc("vllm_criu_restore_kv_cache")
          restore_status.write_text("stage cudagraph_capture\n")
          restore_graphs = os.environ.get(
            "VLLM_LIFECYCLE_RESTORE_CUDAGRAPHS", "0"
          ).lower() in {"1", "true", "yes", "on"}
          if restore_graphs and graph_preservation_enabled():
            # Level-2 remaps the original allocation handles before
            # reload_weights and KV wake. Retain the CUDA graph
            # executables when their driver-side state survived;
            # the worker reports the restored graph inventory so
            # the experimental mode is observable rather than
            # silently pretending that graphs were recovered.
            result = self.collective_rpc(
              "vllm_criu_preserve_cudagraphs"
            )
            LOG.warning(
              "retained restored CUDA graphs instead of recapturing: %s",
              result,
            )
          elif restore_graphs:
            # The default path clears graph objects before the
            # checkpoint. Re-run vLLM's normal capture routine after
            # weights and KV cache are ready.
            self.collective_rpc("vllm_criu_restore_cudagraphs")
          else:
            # The current MTP/mamba runner can retain CPU-side
            # capture staging state across level-2 sleep. Eager
            # execution is the stable default while graph capture
            # recovery remains an opt-in experiment.
            self.collective_rpc("vllm_criu_disable_cudagraphs")

          # reload_weights() and CUDA-graph capture operate on the
          # worker, but the sleep flag and scheduler pause state
          # belong to EngineCore.  Some restore generations leave
          # the executor's sleeping_tags populated again after the
          # final worker RPC.  In that state utility RPCs succeed
          # while normal generation requests remain queued forever.
          # Make the lifecycle transition explicit only after every
          # level-2 recovery operation has completed.
          self.model_executor.sleeping_tags.clear()
          self.model_executor.is_sleeping = False
          self.resume_scheduler()
          LOG.warning(
            "marked restored V1 EngineCore awake after level-2 recovery"
          )
          _discard_stale_engine_dead_outputs(self)
      except Exception:
        import traceback

        detail = traceback.format_exc()
        restore_status.write_text("failed\n" + detail)
        LOG.error("V1 EngineCore-local CRIU restore failed\n%s", detail)
        raise
      else:
        restore_status.write_text("done\n")
        self._vllm_criu_local_restore_complete = True
        LOG.warning("completed V1 EngineCore-local CRIU restore sequence")
      finally:
        fault_file = getattr(self, "_vllm_criu_restore_fault_file", None)
        if fault_file is not None:
          try:
            import faulthandler

            faulthandler.disable()
            fault_file.close()
          except Exception:
            pass
        self._vllm_criu_local_restore_in_progress = False

    def start_local_restore_thread() -> None:
      """Run restore independently of the restored queue waiter.

      A CRIU-restored EngineCore can remain blocked in its pre-dump
      ``Queue.get``/CUDA wait state even though its Python process and
      socket threads are alive.  Waiting for ``_process_input_queue`` to
      call ``run_local_restore`` consequently makes the launcher time
      out.  The restore marker is created only after CRIU and
      cuda-checkpoint restoration are complete, so a dedicated thread
      is safe to use as the primary trigger.  The main-loop hook remains
      idempotent and acts as a fallback for older restore timing.
      """

      if (
        not restore_marker.exists()
        or self._vllm_criu_local_restore_complete
        or self._vllm_criu_local_restore_in_progress
        or self._vllm_criu_restore_thread_started
      ):
        return
      self._vllm_criu_restore_thread_started = True
      LOG.warning("starting independent V1 EngineCore-local restore thread")
      threading.Thread(
        target=run_local_restore,
        name="VLLMCRIULocalRestore",
        daemon=True,
      ).start()

    def start_replacement_threads(generation: int) -> None:
      restore_marker = (
        Path(os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current"))
        .parent
        / ".vllm-criu-restore-request"
      )
      if generation <= 0 and restore_marker.exists():
        # CRIU can restore the Python process with SIGUSR1 pending or
        # blocked behind a C-level wait. The launcher marker is the
        # deterministic fallback, observed by our nonblocking main
        # loop after restore.
        generation = 1
      if generation <= 0 or generation == self._vllm_criu_restarted_generation:
        return
      self._vllm_criu_restarted_generation = generation
      os.environ["VLLM_CRIU_FORCE_NONBLOCKING"] = "0"
      self.process_input_queue_block = True
      input_args = getattr(self, "_vllm_criu_input_args", None)
      output_args = getattr(self, "_vllm_criu_output_args", None)
      if input_args is not None:
        threading.Thread(
          target=self.process_input_sockets,
          args=input_args,
          name="VLLMCRIUInputSockets",
          daemon=True,
        ).start()
      if output_args is not None:
        threading.Thread(
          target=self.process_output_sockets,
          args=output_args,
          name="VLLMCRIUOutputSockets",
          daemon=True,
        ).start()
      LOG.warning(
        "started replacement EngineCore socket threads for restore generation %s",
        generation,
      )

    self._vllm_criu_start_replacement_threads = start_replacement_threads
    self._vllm_criu_run_local_restore = run_local_restore

    def watch_restore_marker() -> None:
      """Wake the restored process from a kernel directory event.

      CRIU can restore Python threads in a blocked futex or libzmq poll
      state.  The launcher creates the marker after restore, so an
      inotify watch provides a kernel-level wakeup that does not depend
      on either of those restored waiters running Python bytecode.
      """

      import ctypes
      import ctypes.util
      import select

      marker_parent = restore_marker.parent
      libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
      init1 = libc.inotify_init1
      init1.argtypes = [ctypes.c_int]
      init1.restype = ctypes.c_int
      add_watch = libc.inotify_add_watch
      add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
      add_watch.restype = ctypes.c_int
      in_nonblock = getattr(os, "O_NONBLOCK", 0x800)
      in_cloexec = getattr(os, "O_CLOEXEC", 0x80000)
      fd = init1(in_nonblock | in_cloexec)
      if fd < 0:
        LOG.exception("could not create CRIU restore marker inotify fd")
        return
      mask = 0x00000100 | 0x00000080 | 0x00000008  # CREATE|MOVED_TO|CLOSE_WRITE
      watch = add_watch(fd, os.fsencode(str(marker_parent)), mask)
      if watch < 0:
        LOG.exception("could not watch CRIU restore marker directory")
        os.close(fd)
        return
      LOG.warning("installed CRIU restore marker inotify watchdog")
      try:
        while True:
          readable, _, _ = select.select([fd], [], [], 0.25)
          if not readable:
            continue
          os.read(fd, 4096)
          if restore_marker.exists() and self._vllm_criu_reconnect_generation <= 0:
            self._vllm_criu_reconnect_generation = 1
            start_local_restore_thread()
            try:
              self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            except Exception:
              LOG.exception("failed to wake EngineCore from restore marker watchdog")
            return
      except Exception:
        LOG.exception("CRIU restore marker watchdog failed")
      finally:
        os.close(fd)

    threading.Thread(
      target=watch_restore_marker,
      name="VLLMCRIURestoreMarkerWatchdog",
      daemon=True,
    ).start()

    def supervise_reconnect() -> None:
      """Recreate socket threads if CRIU left the originals blocked.

      Closing a pyzmq socket from the signal handler normally causes the
      patched socket loop to leave its poll cycle.  CRIU can restore the
      loop in a libzmq wait state where that close is not observed. A
      tiny polling supervisor is deliberately independent of libzmq and
      starts replacement loops once the generation changes. The old
      loops will exit when their 250ms poll returns, so this does not
      create a second long-lived socket owner.
      """

      generation_seen = self._vllm_criu_reconnect_generation
      restore_marker = (
        Path(os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current"))
        .parent
        / ".vllm-criu-restore-request"
      )
      while True:
        generation = self._vllm_criu_reconnect_generation
        if restore_marker.exists():
          start_local_restore_thread()
        # Do not depend on SIGUSR1 or on the restored EngineCore main
        # loop reaching _process_input_queue.  Both can be suspended
        # behind a restored futex/C-level wait.  The launcher creates
        # this marker only after CRIU has restored the process, so the
        # independent supervisor can deterministically trigger the
        # socket rehook from a fresh Python thread.
        if generation <= 0 and restore_marker.exists():
          self._vllm_criu_reconnect_generation = generation = 1
          start_local_restore_thread()
        if generation != generation_seen:
          generation_seen = generation
          # Let the API-side ROUTER finish replacing its sockets
          # before the replacement DEALER/PUSH connections attach.
          time.sleep(0.05)
          input_args = getattr(self, "_vllm_criu_input_args", None)
          output_args = getattr(self, "_vllm_criu_output_args", None)
          del input_args, output_args
          self._vllm_criu_start_replacement_threads(generation)
        time.sleep(0.05)

    if not self._vllm_criu_supervisor_started:
      self._vllm_criu_supervisor_started = True
      threading.Thread(
        target=supervise_reconnect,
        name="VLLMCRIUSocketSupervisor",
        daemon=True,
      ).start()

  EngineCoreProc.__init__ = patched_init
  EngineCoreProc.process_input_sockets = _input_sockets
  EngineCoreProc.process_output_sockets = _output_sockets

  if not getattr(EngineCoreProc, "_vllm_criu_main_loop_rehook", False):
    original_process_input_queue = EngineCoreProc._process_input_queue

    @wraps(original_process_input_queue)
    def process_input_queue_with_rehook(self, *args, **kwargs):
      start_threads = getattr(
        self, "_vllm_criu_start_replacement_threads", None
      )
      if start_threads is not None:
        start_threads(self._vllm_criu_reconnect_generation)

      # The marker is created only after CRIU has restored CUDA state.
      # Run recovery in EngineCore's own thread, where the model
      # executor is directly available, instead of depending on the
      # restored API↔EngineCore utility ZMQ path.
      run_local_restore = getattr(self, "_vllm_criu_run_local_restore", None)
      if run_local_restore is not None:
        run_local_restore()

      # The stock implementation uses Queue.get(block=True) with no
      # timeout. After CRIU restores a sleeping EngineCore, that futex
      # can remain asleep indefinitely and Python will not run the
      # SIGUSR1 handler that advances the reconnect generation. Keep the
      # vLLM request semantics, but bound the idle wait so the active
      # main loop can observe restore state and rehook sockets.
      import queue

      while not self.has_work() and self.is_running():
        self._notify_idle_state_callbacks()
        if self.input_queue.empty():
          with self.aborts_queue.mutex:
            self.aborts_queue.queue.clear()
        block = self.process_input_queue_block
        force_nonblocking = os.environ.get(
          "VLLM_CRIU_FORCE_NONBLOCKING", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if force_nonblocking:
          block = False
        try:
          request = self.input_queue.get(
            block=block,
            timeout=0.25 if block else 0,
          )
          self._handle_client_request(*request)
        except queue.Empty:
          if force_nonblocking:
            time.sleep(0.01)
          break

      while not self.input_queue.empty():
        request = self.input_queue.get_nowait()
        self._handle_client_request(*request)

    EngineCoreProc._process_input_queue = process_input_queue_with_rehook
    EngineCoreProc._vllm_criu_main_loop_rehook = True

  from vllm.v1.worker.gpu_worker import Worker

  if not getattr(Worker, "_vllm_criu_allocator_restore_patch", False):

    original_checkpoint_prepare = Worker.checkpoint_prepare
    original_checkpoint_restore = Worker.checkpoint_restore

    def checkpoint_prepare_for_criu(self):
      if graph_preservation_enabled():
        # Level-2 has already unmapped the model/KV allocations. Keep
        # vLLM's graph pool in the process so NVIDIA's CUDA checkpoint
        # layer can release and restore it. The level-2 allocator
        # retains the allocation handles and remaps those virtual
        # addresses before weights/KV are restored.
        import torch

        torch.cuda.synchronize()
        LOG.warning(
          "retaining CUDA graph captures for hybrid level-2 CRIU "
          "checkpoint"
        )
      else:
        # Default safe path: graph pools are not tracked by
        # CuMemAllocator and can otherwise remain as hidden GPU
        # allocations or retain stale addresses after level-2 sleep.
        try:
          from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphWrapper,
          )
          from vllm.compilation.cuda_graph import CUDAGraphWrapper

          CUDAGraphWrapper.clear_all_graphs()
          BreakableCUDAGraphWrapper.clear_all_graphs()
          if getattr(
            self.model_runner, "encoder_cudagraph_manager", None
          ) is not None:
            self.model_runner.encoder_cudagraph_manager = None
          gc.collect()
          import torch

          torch.cuda.empty_cache()
          LOG.warning("cleared CUDA graph captures before CRIU checkpoint")
        except Exception:
          LOG.exception("could not clear CUDA graph captures before checkpoint")

      if os.environ.get("VLLM_LIFECYCLE_CUDA_DIST_CLEANUP", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
      }:
        import torch

        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        # CUDA checkpoint releases every GPU resource, including
        # process-group/NCCL resources. Destroy their Python wrappers
        # before CRIU captures the CPU-side object graph.
        torch.cuda.synchronize()
        cleanup_dist_env_and_memory()
        self._vllm_criu_distributed_cleaned = True
        # The EngineCore main loop is in this same process. Keep it
        # out of an unbounded Queue.get() while CRIU snapshots the
        # process so restore resumes into a pollable Python frame.
        os.environ["VLLM_CRIU_FORCE_NONBLOCKING"] = "1"
        LOG.warning("cleaned vLLM distributed state before CUDA checkpoint")

      # Keep vLLM's communicator-specific preparation for versions that
      # need it, but run it after cleanup. With the current single-GPU
      # executor this is a no-op after the process groups are gone; with
      # a future communicator it remains the version-owned hook.
      result = original_checkpoint_prepare(self)
      LOG.warning("completed vLLM checkpoint_prepare for CRIU")
      return result

    def checkpoint_restore_for_criu(self):
      distributed_cleanup_enabled = os.environ.get(
        "VLLM_LIFECYCLE_CUDA_DIST_CLEANUP", "0"
      ).lower() in {"1", "true", "yes", "on"}
      if distributed_cleanup_enabled or getattr(
        self, "_vllm_criu_distributed_cleaned", False
      ):
        # The full worker rebootstrap initializes distributed state as
        # part of Worker.init_device(). The allocation-only path needs
        # to recreate it here before touching model/KV tensors.
        if os.environ.get("VLLM_LIFECYCLE_CUDA_REBOOTSTRAP", "1").lower() not in {
          "1",
          "true",
          "yes",
          "on",
        }:
          from vllm.platforms import current_platform
          from vllm.v1.worker.gpu_worker import (
            init_worker_distributed_environment,
          )

          init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
          )
          LOG.warning("reinitialized vLLM distributed state after CUDA restore")
        self._vllm_criu_distributed_cleaned = False
        LOG.warning("bypassed stale vLLM checkpoint_restore after CRIU")
        return None
      result = original_checkpoint_restore(self)
      LOG.warning("completed vLLM checkpoint_restore after CRIU")
      return result

    Worker.checkpoint_prepare = checkpoint_prepare_for_criu
    Worker.checkpoint_restore = checkpoint_restore_for_criu

    def reinitialize_after_restore(self):
      if os.environ.get("VLLM_LIFECYCLE_RESTORE_NATIVE_WAKE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
      }:
        return _wake_worker_allocations_after_restore(self)
      if os.environ.get("VLLM_LIFECYCLE_CUDA_REBOOTSTRAP", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
      }:
        return _rebootstrap_worker_after_restore(self)
      return _reinitialize_worker_allocations(self)

    Worker.vllm_criu_reinitialize_after_restore = reinitialize_after_restore

    def disable_cudagraphs(self):
      """Make the restored V1 runner execute eagerly for this generation."""

      from vllm.config import CUDAGraphMode
      from vllm.compilation.monitor import set_cudagraph_capturing_enabled

      set_cudagraph_capturing_enabled(False)
      try:
        from vllm.compilation.breakable_cudagraph import (
          BreakableCUDAGraphWrapper,
        )
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()
      except Exception:
        LOG.exception("could not clear partial CRIU CUDA graph state")

      runners = [self.model_runner, getattr(self.model_runner, "drafter", None)]
      for runner in runners:
        if runner is None:
          continue
        compilation_config = getattr(runner, "compilation_config", None)
        if compilation_config is not None:
          compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        dispatcher = getattr(runner, "cudagraph_dispatcher", None)
        if dispatcher is not None:
          for keys in dispatcher.cudagraph_keys.values():
            keys.clear()
          dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)
      LOG.warning("disabled CUDA graphs for restored V1 generation")
      return {"cudagraphs_disabled": True, "eager_execution": True}

    Worker.vllm_criu_disable_cudagraphs = disable_cudagraphs

    def restore_cudagraphs(self):
      """Recreate vLLM CUDA graphs after CRIU invalidates old graphs."""

      try:
        size = self.model_runner.capture_model()
      except Exception:
        # Some hybrid/speculative runners retain a CPU-side staging
        # tensor after level-2 sleep. If vLLM's normal post-restore
        # capture warmup encounters it, keep the restored process
        # usable by switching this generation to eager execution.
        # This avoids a cold-process fallback; graph capture remains
        # an optional optimization for a later restore iteration.
        LOG.exception(
          "could not recreate vLLM CUDA graphs after CRIU restore; "
          "falling back to eager V1 execution"
        )
        disable_cudagraphs(self)
        return {"cudagraphs_recreated": False, "eager_fallback": True}
      LOG.warning(
        "recreated vLLM CUDA graphs after CRIU restore: %.2f GiB",
        size / (1 << 30),
      )
      return {"cudagraphs_recreated": True, "bytes": size}

    def preserve_cudagraphs_after_restore(self):
      return mark_restored_graphs()

    Worker.vllm_criu_restore_cudagraphs = restore_cudagraphs
    Worker.vllm_criu_preserve_cudagraphs = preserve_cudagraphs_after_restore

    def restore_kv_cache_after_restore(self):
      self.wake_up(tags=["kv_cache"])
      return {"kv_cache_woken": True}

    Worker.vllm_criu_restore_kv_cache = restore_kv_cache_after_restore
    Worker._vllm_criu_allocator_restore_patch = True

  if not getattr(EngineCore, "_vllm_criu_reinitialize_rpc", False):
    original_collective_rpc = EngineCore.collective_rpc

    @wraps(original_collective_rpc)
    def collective_rpc_with_restore_reinit(self, method, *args, **kwargs):
      result = original_collective_rpc(self, method, *args, **kwargs)
      if method == "vllm_criu_reinitialize_after_restore":
        # The worker has now replaced every level-2 allocation that
        # was reachable from the model/KV state.  Mark both executor
        # tags resident without entering CuMemAllocator.wake_up(),
        # which would remap the pre-CRIU handles.
        self.model_executor.sleeping_tags.clear()
        self.model_executor.is_sleeping = False
        self.resume_scheduler()
      return result

    EngineCore.collective_rpc = collective_rpc_with_restore_reinit
    EngineCore._vllm_criu_reinitialize_rpc = True

  # Keep opt-in tracebacks at the EngineCore and worker boundaries.  The
  # development RPC endpoint deliberately serializes only ``str(error)``,
  # which hides whether a restored failure came from the RPC transport or
  # from the worker method itself.  The worker boundary is the useful one
  # for this experiment because UniProcExecutor invokes the worker directly
  # in the EngineCore process.
  if (
    os.environ.get("VLLM_LIFECYCLE_DEBUG_RPC", "0").lower()
    in {"1", "true", "yes", "on"}
  ):
    if not getattr(EngineCore, "_vllm_criu_rpc_trace", False):
      original_collective_rpc = EngineCore.collective_rpc

      @wraps(original_collective_rpc)
      def traced_collective_rpc(self, *args, **kwargs):
        try:
          return original_collective_rpc(self, *args, **kwargs)
        except Exception:
          method = args[0] if args else kwargs.get("method")
          LOG.exception("EngineCore collective_rpc failed for %r", method)
          raise

      EngineCore.collective_rpc = traced_collective_rpc
      EngineCore._vllm_criu_rpc_trace = True

    if not getattr(Worker, "_vllm_criu_reload_trace", False):
      original_reload_weights = Worker.reload_weights

      def traced_reload_weights(self, *args, **kwargs):
        LOG.info("starting restored GPUWorker.reload_weights")
        try:
          result = original_reload_weights(self, *args, **kwargs)
        except Exception:
          LOG.exception("GPUWorker.reload_weights failed after restore")
          raise
        LOG.info("completed restored GPUWorker.reload_weights")
        return result

      Worker.reload_weights = traced_reload_weights
      Worker._vllm_criu_reload_trace = True

  _INSTALLED = True
  return True


__all__ = [
  "install_enginecore_restore_patch",
]
