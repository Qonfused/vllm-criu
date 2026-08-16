## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""Diagnostics and restore markers for EngineCore CRIU recovery."""

from __future__ import annotations

import logging
import os
from functools import wraps
from pathlib import Path

LOG = logging.getLogger("vllm-criu.enginecore-patch")


def _restore_diagnostic_path(name: str) -> Path:
  parent = Path(
    os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current")
  ).parent
  return parent / name


def _append_restore_diagnostic(name: str, text: str) -> None:
  try:
    with _restore_diagnostic_path(name).open("a", encoding="utf-8") as stream:
      stream.write(text)
      if not text.endswith("\n"):
        stream.write("\n")
      stream.flush()
  except OSError:
    # Diagnostics must never change the EngineCore lifecycle.
    pass


def _install_enginecore_crash_diagnostic() -> None:
  """Persist the exception that causes V1 EngineCore to publish DEAD.

  During CRIU recovery the EngineCore stdio descriptors can refer to the
  pre-dump launcher pipe.  vLLM's normal ``logger.exception`` therefore
  becomes invisible after we repair the restored process.  Keep the stock
  behavior, but copy the traceback to a file before it is re-raised.
  """

  from vllm.v1.engine.core import EngineCoreProc

  if getattr(EngineCoreProc, "_vllm_criu_crash_diagnostic", False):
    return

  original_run_engine_core = EngineCoreProc.run_engine_core

  @wraps(original_run_engine_core)
  def run_engine_core_with_diagnostic(*args, **kwargs):
    import traceback

    try:
      return original_run_engine_core(*args, **kwargs)
    except BaseException:
      _append_restore_diagnostic(
        ".vllm-criu-enginecore-crash.log",
        "\n=== EngineCore exception ===\n" + traceback.format_exc(),
      )
      raise

  EngineCoreProc.run_engine_core = staticmethod(run_engine_core_with_diagnostic)
  EngineCoreProc._vllm_criu_crash_diagnostic = True


def _transport_debug_enabled() -> bool:
  return os.environ.get("VLLM_LIFECYCLE_DEBUG_TRANSPORT", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
  }


def _restore_active_paths() -> tuple[Path, Path]:
  parent = Path(
    os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current")
  ).parent
  return (
    parent / ".vllm-criu-restore-pending",
    parent / ".vllm-criu-restore-request",
  )



__all__ = [
  "_append_restore_diagnostic",
  "_install_enginecore_crash_diagnostic",
  "_restore_active_paths",
  "_restore_diagnostic_path",
  "_transport_debug_enabled",
]
