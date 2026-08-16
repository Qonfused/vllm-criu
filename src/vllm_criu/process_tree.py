## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import signal
import time

LOG = logging.getLogger(__name__)


def command_line(pid: int) -> str:
  try:
    return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
  except (FileNotFoundError, PermissionError):
    return ""


def comm(pid: int) -> str:
  try:
    return pathlib.Path(f"/proc/{pid}/comm").read_text().strip()
  except (FileNotFoundError, PermissionError):
    return ""


def children(pid: int) -> list[int]:
  try:
    values = pathlib.Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
  except (FileNotFoundError, PermissionError):
    return []
  return [int(value) for value in values]


def descendants(root_pid: int) -> list[int]:
  found: list[int] = []
  queue = [root_pid]
  while queue:
    current = queue.pop(0)
    current_children = children(current)
    found.extend(current_children)
    queue.extend(current_children)
  return found


def find_engine(root_pid: int) -> int | None:
  for pid in [root_pid, *descendants(root_pid)]:
    if "EngineCore" in comm(pid) or "EngineCore" in command_line(pid):
      return pid
  return None


def find_resource_trackers(root_pid: int) -> list[int]:
  return [
    pid
    for pid in descendants(root_pid)
    if "multiprocessing.resource_tracker" in command_line(pid)
  ]


def enable_child_subreaper() -> None:
  """Adopt orphaned CRIU descendants so their PIDs can be restored."""
  try:
    result = ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
  except OSError:
    LOG.exception("could not enable child-subreaper mode")
    return
  if result:
    LOG.warning("PR_SET_CHILD_SUBREAPER returned %s", result)


def reap_children() -> list[int]:
  reaped: list[int] = []
  while True:
    try:
      pid, status = os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
      break
    if pid == 0:
      break
    reaped.append(pid)
    LOG.info("reaped child pid=%s status=%s", pid, status)
  return reaped


def terminate_resource_trackers(root_pid: int, timeout: float = 2.0) -> list[int]:
  """Stop Python resource trackers before CRIU sees the process tree.

  The tracker is recreated lazily by multiprocessing when later shared-memory
  registration needs it. Keeping it out of the checkpoint is experimental but
  avoids asking the CUDA CRIU plugin to restore a non-CUDA helper process.
  """
  pids = find_resource_trackers(root_pid)
  for pid in pids:
    try:
      os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
      continue
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline and any(pathlib.Path(f"/proc/{pid}").exists() for pid in pids):
    time.sleep(0.05)
  for pid in pids:
    if pathlib.Path(f"/proc/{pid}").exists():
      try:
        os.kill(pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
  reap_children()
  LOG.info("terminated resource trackers before checkpoint: %s", pids)
  return pids


def terminate_tree(root_pid: int) -> None:
  """Terminate a restored process tree after a failed post-restore step."""
  pids = [*reversed(descendants(root_pid)), root_pid]
  for pid in pids:
    try:
      os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  reap_children()
