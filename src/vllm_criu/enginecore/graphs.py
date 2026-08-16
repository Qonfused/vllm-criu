## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""CUDA-graph preservation helpers for the level-2 CRIU path."""

from __future__ import annotations

import logging
import os
from typing import Any

LOG = logging.getLogger("vllm-criu.enginecore-graphs")


def graph_preservation_enabled() -> bool:
  """Return whether the experimental graph-preserving path is enabled."""

  return os.environ.get("VLLM_LIFECYCLE_PRESERVE_CUDAGRAPHS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
  }


def _graph_entry_count(wrapper: Any) -> tuple[int, int]:
  """Return (known entries, live executable captures) for a graph wrapper."""

  entries = getattr(wrapper, "concrete_cudagraph_entries", None)
  if entries is None:
    entries = getattr(wrapper, "entries", None)
  if not isinstance(entries, dict):
    return 0, 0

  live = 0
  for entry in entries.values():
    if getattr(entry, "cudagraph", None) is not None:
      live += 1
    elif getattr(entry, "capture", None) is not None:
      live += 1
  return len(entries), live


def mark_restored_graphs() -> dict[str, object]:
  """Mark restored graph wrappers and return a diagnostic summary.

  This does not execute a graph. The first post-restore replay is the
  meaningful validation because only CUDA can determine whether all driver
  objects and backing virtual addresses were restored consistently.
  """

  from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
  from vllm.compilation.cuda_graph import CUDAGraphWrapper

  wrappers = [
    *list(getattr(CUDAGraphWrapper, "_all_instances", ())),
    *list(getattr(BreakableCUDAGraphWrapper, "_all_instances", ())),
  ]
  known = 0
  live = 0
  for wrapper in wrappers:
    entries, captures = _graph_entry_count(wrapper)
    known += entries
    live += captures
    wrapper._vllm_criu_restored_graphs = True

  summary = {
    "cudagraphs_preserved": live > 0,
    "wrapper_count": len(wrappers),
    "graph_entries": known,
    "live_captures": live,
  }
  LOG.warning("restored CUDA graph state summary: %s", summary)
  return summary


__all__ = [
  "graph_preservation_enabled",
  "mark_restored_graphs",
]
