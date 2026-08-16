## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
  model_id: str
  model_name: str
  max_model_len: str
  max_tokens: str
  gpu_memory_utilization: str
  max_num_seqs: str
  kv_cache_dtype: str
  vllm_host: str = "127.0.0.1"
  vllm_port: int = 8001
  vllm_uds_path: Path = Path("/tmp/vllm-api.sock")
  control_host: str = "0.0.0.0"
  control_port: int = 9000
  checkpoint_dir: Path = Path("/checkpoints/current")
  checkpoint_timeout: int = 120
  checkpoint_sleep_level: int = 2
  allocator_reinit: bool = True
  restore_fallback_fresh: bool = True
  criu_tcp_close: bool = False
  resume_on_start: bool = False
  resource_tracker_mode: str = "terminate"

  @classmethod
  def from_env(cls) -> "Settings":
    mode = os.environ.get("LAUNCHER_RESOURCE_TRACKER_MODE", "terminate")
    if mode not in {"keep", "terminate", "externalize"}:
      raise ValueError(
        "LAUNCHER_RESOURCE_TRACKER_MODE must be keep, terminate, or externalize"
      )
    return cls(
      model_id=os.environ["MODEL_ID"],
      model_name=os.environ["MODEL_NAME"],
      max_model_len=os.environ["MAX_MODEL_LEN"],
      max_tokens=os.environ["MAX_TOKENS"],
      gpu_memory_utilization=os.environ["GPU_MEMORY_UTILIZATION"],
      max_num_seqs=os.environ["MAX_NUM_SEQS"],
      kv_cache_dtype=os.environ["KV_CACHE_DTYPE"],
      vllm_host=os.environ.get("VLLM_HOST", "127.0.0.1"),
      vllm_port=int(os.environ.get("VLLM_PORT", "8001")),
      vllm_uds_path=Path(os.environ.get("VLLM_API_UDS_PATH", "/tmp/vllm-api.sock")),
      control_host=os.environ.get("LAUNCHER_CONTROL_HOST", "0.0.0.0"),
      control_port=int(os.environ.get("LAUNCHER_CONTROL_PORT", "9000")),
      checkpoint_dir=Path(
        os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current")
      ),
      checkpoint_timeout=int(os.environ.get("LAUNCHER_CHECKPOINT_TIMEOUT", "120")),
      checkpoint_sleep_level=int(os.environ.get("LAUNCHER_CHECKPOINT_SLEEP_LEVEL", "2")),
      allocator_reinit=_bool_env("VLLM_LIFECYCLE_ALLOCATOR_REINIT", True),
      restore_fallback_fresh=_bool_env("LAUNCHER_RESTORE_FALLBACK_FRESH", True),
      criu_tcp_close=_bool_env("LAUNCHER_CRIU_TCP_CLOSE"),
      resume_on_start=_bool_env("LAUNCHER_RESUME_ON_START"),
      resource_tracker_mode=mode,
    )
