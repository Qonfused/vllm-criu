## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_EXTERNAL_TRACKER = """
\"\"\"Connect this child to the launcher's resource tracker.\"\"\"

import os
from multiprocessing import resource_tracker


fd = os.environ.get("VLLM_LIFECYCLE_RESOURCE_TRACKER_FD")
pid = os.environ.get("VLLM_LIFECYCLE_RESOURCE_TRACKER_PID")
if fd and pid:
  resource_tracker._resource_tracker._fd = int(fd)
  resource_tracker._resource_tracker._pid = int(pid)
""".lstrip()


_ENGINECORE_PATCH = """
\"\"\"Install the CRIU EngineCore socket patch in spawned vLLM children.\"\"\"

from vllm_criu.enginecore import install_enginecore_restore_patch

install_enginecore_restore_patch()
""".lstrip()


@dataclass(frozen=True)
class ChildRuntime:
  environment: dict[str, str]
  pass_fds: tuple[int, ...] = ()


def _external_tracker() -> tuple[int, int] | None:
  from multiprocessing import resource_tracker

  resource_tracker.ensure_running()
  tracker = resource_tracker._resource_tracker
  if tracker._fd is None or tracker._pid is None:
    return None
  return tracker._fd, tracker._pid


def child_runtime(resource_tracker_mode: str) -> ChildRuntime:
  """Return the vLLM child environment for the selected tracker strategy.

  ``externalize`` starts the helper under the launcher and passes its pipe
  into vLLM, so the helper is outside the CRIU process tree while remaining
  available to multiprocessing spawn. This preserves Python's normal
  resource cleanup semantics.
  """
  environment = os.environ.copy()
  package_root = str(Path(__file__).resolve().parent.parent)
  pythonpath = environment.get("PYTHONPATH")
  environment["PYTHONPATH"] = package_root + (os.pathsep + pythonpath if pythonpath else "")
  needs_enginecore_patch = os.environ.get(
    "VLLM_LIFECYCLE_ENGINECORE_PATCH", "0"
  ).lower() in {"1", "true", "yes", "on"}
  if resource_tracker_mode == "externalize" or needs_enginecore_patch:
    runtime_dir = Path("/tmp/vllm-criu-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = ""
    pass_fds: tuple[int, ...] = ()
    if resource_tracker_mode == "externalize":
      tracker = _external_tracker()
      if tracker is None:
        raise RuntimeError("could not start launcher's resource tracker")
      fd, pid = tracker
      sitecustomize += _EXTERNAL_TRACKER
      environment["VLLM_LIFECYCLE_RESOURCE_TRACKER_FD"] = str(fd)
      environment["VLLM_LIFECYCLE_RESOURCE_TRACKER_PID"] = str(pid)
      pass_fds = (fd,)

    # Configure the inherited resource tracker before importing vLLM in
    # the spawned child. This keeps any multiprocessing setup performed by
    # the EngineCore patch attached to the launcher's tracker.
    if needs_enginecore_patch:
      sitecustomize += _ENGINECORE_PATCH

    (runtime_dir / "sitecustomize.py").write_text(sitecustomize)
    environment["PYTHONPATH"] = str(runtime_dir) + os.pathsep + environment["PYTHONPATH"]
    if resource_tracker_mode == "externalize":
      return ChildRuntime(environment, pass_fds)
    return ChildRuntime(environment)

  return ChildRuntime(environment)
