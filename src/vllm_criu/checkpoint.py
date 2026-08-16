## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from __future__ import annotations

import logging
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


class CheckpointError(RuntimeError):
  pass


@dataclass(frozen=True)
class Tools:
  criu: str
  cuda_checkpoint: str

  @classmethod
  def discover(cls) -> "Tools":
    criu = shutil.which("criu")
    cuda = shutil.which("cuda-checkpoint")
    if not criu or not cuda:
      raise CheckpointError(f"checkpoint tooling unavailable: criu={criu}, cuda-checkpoint={cuda}")
    return cls(criu, cuda)


class CommandRunner:
  def run(self, command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    LOG.info("running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
      LOG.info("stdout: %s", result.stdout.strip())
    if result.stderr.strip():
      LOG.info("stderr: %s", result.stderr.strip())
    if result.returncode:
      raise CheckpointError(f"command exited {result.returncode}: {' '.join(command)}")
    return result


class CudaCheckpoint:
  def __init__(self, path: str, runner: CommandRunner, timeout: int) -> None:
    self.path = path
    self.runner = runner
    self.timeout = timeout

  def state(self, pid: int) -> str:
    result = self.runner.run([self.path, "--get-state", "--pid", str(pid)], self.timeout)
    return result.stdout.strip()

  def lock(self, pid: int) -> None:
    self.runner.run(
      [self.path, "--action", "lock", "--pid", str(pid), "--timeout", str(self.timeout * 1000)],
      self.timeout,
    )

  def checkpoint(self, pid: int) -> None:
    self.runner.run([self.path, "--action", "checkpoint", "--pid", str(pid)], self.timeout)

  def restore_if_needed(self, pid: int) -> None:
    # CRIU can restore the process while cuda-checkpoint reports the
    # process as ``running``.  That state only describes the checkpoint
    # utility's lock state; it does not prove that the CUDA driver's
    # allocations and execution state have been reattached.  In
    # particular, vLLM's CuMemAllocator.wake_up() can otherwise block on
    # its first post-restore cuMemMap call.
    #
    # Always give cuda-checkpoint the opportunity to complete its restore
    # handshake.  Older driver/tool combinations may reject the action
    # because CRIU already restored the CUDA state, so retain the previous
    # behavior as a narrowly scoped fallback for that case.
    was_running = self.state(pid) == "running"
    try:
      self.runner.run(
        [self.path, "--action", "restore", "--pid", str(pid)],
        self.timeout,
      )
      self.runner.run(
        [self.path, "--action", "unlock", "--pid", str(pid)],
        self.timeout,
      )
      LOG.info("CUDA state restored and unlocked for pid %s", pid)
    except CheckpointError:
      if not was_running:
        raise
      LOG.warning(
        "cuda-checkpoint restore was rejected for already-running pid %s; "
        "continuing with CRIU-restored CUDA state",
        pid,
        exc_info=True,
      )

  def rollback(self, pid: int) -> None:
    if not pathlib.Path(f"/proc/{pid}").exists():
      return
    try:
      self.runner.run([self.path, "--action", "restore", "--pid", str(pid)], self.timeout)
      self.runner.run([self.path, "--action", "unlock", "--pid", str(pid)], self.timeout)
    except Exception:
      LOG.exception("could not roll back CUDA checkpoint for pid %s", pid)


class Criu:
  def __init__(
    self, path: str, runner: CommandRunner, timeout: int, *, tcp_close: bool = False
  ) -> None:
    self.path = path
    self.runner = runner
    self.timeout = timeout
    self.tcp_close = tcp_close

  @property
  def tcp_args(self) -> list[str]:
    return ["--tcp-close" if self.tcp_close else "--tcp-established"]

  def dump(self, root_pid: int, image_dir: pathlib.Path) -> None:
    self.runner.run(
      [
        self.path,
        "dump",
        "--shell-job",
        *self.tcp_args,
        "--ext-unix-sk",
        "--link-remap",
        "--images-dir",
        str(image_dir),
        "--tree",
        str(root_pid),
        "--pidfile",
        str(image_dir / "root.pid"),
      ],
      self.timeout,
    )

  def restore(self, image_dir: pathlib.Path) -> int:
    pidfile = image_dir / "restored.pid"
    pidfile.unlink(missing_ok=True)
    self.runner.run(
      [
        self.path,
        "restore",
        "--shell-job",
        "--restore-detached",
        *self.tcp_args,
        "--link-remap",
        "--file-validation",
        "filesize",
        "--images-dir",
        str(image_dir),
        "--pidfile",
        str(pidfile),
      ],
      self.timeout,
    )
    return int(pidfile.read_text().strip())
