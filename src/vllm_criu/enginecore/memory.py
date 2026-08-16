## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""CUDA allocator and worker recovery helpers for CRIU restore."""

from __future__ import annotations

import gc
import logging
import os
from functools import wraps
from pathlib import Path

LOG = logging.getLogger("vllm-criu.enginecore-patch")


def _install_allocator_restore_patch() -> None:
  """Add a restore-only replacement path for level-2 CUDA allocations.

  vLLM's level-2 sleep intentionally releases the physical backing for each
  managed allocation but retains the allocation handle in ``pointer_to_data``.
  That handle is valid for an in-process wake, but it is not a durable CUDA
  allocation identity across CRIU.  After a restore, calling
  ``CuMemAllocator.wake_up`` therefore reaches the first stale ``cuMemMap``
  call and can block forever.

  The restore path below allocates fresh managed tensors and rebinds the
  restored model/KV references to them.  Weight data is subsequently loaded
  by vLLM's normal ``reload_weights`` method.  This is deliberately opt-in
  and only installed alongside the CRIU EngineCore patch.
  """

  from vllm.device_allocator.cumem import CuMemAllocator

  if getattr(CuMemAllocator, "_vllm_criu_restore_patch", False):
    return

  original_free = CuMemAllocator._python_free_callback

  @wraps(original_free)
  def free_restored_sleep_allocation(self, ptr):
    data = self.pointer_to_data.get(ptr)
    if data is not None and data.is_asleep:
      # level-2 sleep already unmapped/released the physical chunks.
      # Returning an empty chunk list lets the pluggable allocator drop
      # the stale virtual allocation without attempting a second CUDA
      # unmap/release while the old model tensor is being rebound.
      self.pointer_to_data.pop(ptr, None)
      data.cpu_backup_tensor = None
      device, size, d_mem, _chunks = data.handle
      return (device, size, d_mem, [])
    return original_free(self, ptr)

  CuMemAllocator._python_free_callback = free_restored_sleep_allocation
  CuMemAllocator._vllm_criu_restore_patch = True


def _reinitialize_worker_allocations(worker) -> dict[str, int]:
  """Replace CRIU-invalid level-2 allocations in one vLLM worker."""

  from contextlib import contextmanager

  import torch

  from vllm.device_allocator.cumem import (
    CuMemAllocator,
    get_pluggable_allocator,
  )

  allocator = CuMemAllocator.get_instance()
  managed = allocator.pointer_to_data
  replacement_by_pointer: dict[int, torch.Tensor] = {}
  replaced = {"weights": 0, "kv_cache": 0}
  old_asleep = sum(1 for data in managed.values() if data.is_asleep)

  # CUDA graph captures retain private allocator pools and references to the
  # pre-sleep model/KV addresses.  The level-2 sleep accounting excludes
  # those pools, so they can make a seemingly empty GPU appear full when the
  # first replacement tensors are allocated after CRIU restore.  Drop the
  # captures before rebinding; they can be recreated lazily by inference.
  try:
    from vllm.compilation.breakable_cudagraph import (
      BreakableCUDAGraphWrapper,
    )
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
  except Exception:
    LOG.exception("could not clear restored CUDA graph captures")
  if getattr(worker.model_runner, "encoder_cudagraph_manager", None) is not None:
    worker.model_runner.encoder_cudagraph_manager = None
  gc.collect()
  torch.cuda.empty_cache()

  # CRIU can bring back the CUDA driver's mappings for allocations that were
  # marked asleep before the dump.  vLLM's Python bookkeeping still marks
  # those handles asleep, so wake_up() cannot distinguish them from valid
  # allocations and a fresh replacement would double the GPU footprint.
  # Release only those stale asleep handles; the small non-sleeping runtime
  # allocations remain untouched.
  from vllm.device_allocator.cumem import unmap_and_release

  stale_handles_released = 0
  for data in list(managed.values()):
    if not data.is_asleep:
      continue
    try:
      unmap_and_release(data.handle)
      stale_handles_released += 1
    except Exception:
      LOG.exception("could not release restored asleep cuMem handle")
  if stale_handles_released:
    LOG.warning(
      "released %s restored asleep cuMem handles before rebinding",
      stale_handles_released,
    )
  torch.cuda.empty_cache()

  @contextmanager
  def restore_memory_pool(tag: str):
    """Allocate a fresh cuMem pool without PyTorch's restore-unsafe snapshot.

    CuMemAllocator.use_memory_pool() calls ``MemPool.snapshot()`` when its
    context exits so it can manually release unused allocations.  After a
    CRIU restore, that snapshot path can touch a stale allocator/event and
    abort in native PyTorch before the new tensor is usable.  The restored
    process is deliberately keeping these fresh tensors alive, so retain
    the pool and allocator and omit only that optional cleanup pass.
    """

    new_alloc = get_pluggable_allocator(
      allocator.python_malloc_callback,
      allocator.python_free_callback,
    )
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    allocator.allocator_and_pools[tag] = (mem_pool, new_alloc)
    old_tag = allocator.current_tag
    allocator.current_tag = tag
    try:
      with torch.cuda.memory.use_mem_pool(mem_pool):
        yield
    finally:
      allocator.current_tag = old_tag

  def is_managed(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda or tensor.numel() == 0:
      return False
    pointer = tensor.data_ptr()
    if pointer in managed:
      return True
    try:
      return tensor.untyped_storage().data_ptr() in managed
    except RuntimeError:
      return False

  def replacement(tensor: torch.Tensor, tag: str) -> torch.Tensor:
    pointer = tensor.data_ptr()
    existing = replacement_by_pointer.get(pointer)
    if existing is not None:
      return existing
    # Use a restore-specific pool.  The original sleep-mode pool contains
    # virtual allocations whose driver handles were checkpointed, and its
    # normal context-exit snapshot is unsafe after CRIU.
    restore_tag = f"criu_restore_{tag}"
    with restore_memory_pool(restore_tag):
      fresh = torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
      )
    replacement_by_pointer[pointer] = fresh
    replaced[tag] += 1
    return fresh

  def replace_tensor(tensor: torch.Tensor, tag: str):
    if not is_managed(tensor):
      return tensor
    return replacement(tensor, tag)

  def replace_nested(value, tag: str):
    if isinstance(value, torch.Tensor):
      return replace_tensor(value, tag)
    if isinstance(value, list):
      for index, item in enumerate(value):
        value[index] = replace_nested(item, tag)
    elif isinstance(value, tuple):
      return tuple(replace_nested(item, tag) for item in value)
    return value

  def replace_module(module, tag: str) -> None:
    for name, parameter in list(module._parameters.items()):
      if parameter is not None:
        module._parameters[name].data = replace_tensor(parameter, tag)
    for name, buffer in list(module._buffers.items()):
      if buffer is not None:
        module._buffers[name] = replace_tensor(buffer, tag)

  def replace_model(model) -> None:
    if model is None:
      return
    for module in model.modules():
      replace_module(module, "weights")

    # Some attention implementations keep the active KV tensor on the
    # layer as a regular attribute rather than as a registered buffer.
    for module in model.modules():
      if hasattr(module, "kv_cache"):
        module.kv_cache = replace_nested(module.kv_cache, "kv_cache")

  model_runner = worker.model_runner
  replace_model(model_runner.model)
  draft = worker.get_draft_model()
  replace_model(draft)

  model_runner.kv_caches = replace_nested(model_runner.kv_caches, "kv_cache")
  if model_runner.cross_layers_kv_cache is not None:
    model_runner.cross_layers_kv_cache = replace_nested(
      model_runner.cross_layers_kv_cache, "kv_cache"
    )

  # Level 2 stores non-parameter buffers on CPU before discarding their
  # device allocation.  Put them into the fresh managed buffers now.
  for model, saved in (
    (model_runner.model, worker._sleep_saved_buffers),
    (draft, worker._sleep_saved_draft_buffers),
  ):
    if model is None:
      continue
    for name, buffer in model.named_buffers():
      saved_buffer = saved.get(name)
      if saved_buffer is not None:
        buffer.copy_(saved_buffer)
    saved.clear()

  # Reinitialize the small amount of KV metadata that is normally refreshed
  # by Worker.wake_up(tags=["kv_cache"]).  The cache tensors themselves were
  # freshly allocated above, so invoking the normal wake path would try to
  # map the restored handles a second time.
  model_runner.post_kv_cache_wake_up()
  gc.collect()

  remaining_asleep = sum(1 for data in managed.values() if data.is_asleep)
  LOG.info(
    "reinitialized CRIU-invalid allocations: old_asleep=%s, "
    "new_weight_tensors=%s, new_kv_tensors=%s, remaining_asleep=%s",
    old_asleep,
    replaced["weights"],
    replaced["kv_cache"],
    remaining_asleep,
  )
  return {
    "old_asleep": old_asleep,
    "new_weight_tensors": replaced["weights"],
    "new_kv_tensors": replaced["kv_cache"],
    "remaining_asleep": remaining_asleep,
  }


def _wake_worker_allocations_after_restore(worker) -> dict[str, object]:
  """Try vLLM's native level-2 wake path on the restored CUDA context."""

  # Keep KV physically discarded while reload_weights uses temporary
  # tensors. Waking both tags first leaves too little headroom for the
  # layerwise weight loader on this model.
  worker.wake_up(tags=["weights"])
  gc.collect()
  return {"native_wake": True}


def _discard_stale_engine_dead_outputs(engine_core) -> int:
  """Remove pre-CRIU dead sentinels from the restored output queue."""

  import queue

  retained = []
  discarded = 0
  while True:
    try:
      output = engine_core.output_queue.get_nowait()
    except queue.Empty:
      break
    if output == engine_core.ENGINE_CORE_DEAD:
      discarded += 1
    else:
      retained.append(output)
  for output in retained:
    engine_core.output_queue.put_nowait(output)
  if discarded:
    LOG.warning(
      "discarded %s stale ENGINE_CORE_DEAD output(s) before CRIU restore completion",
      discarded,
    )
  return discarded


def _rebootstrap_worker_after_restore(worker) -> dict[str, object]:
  """Recreate vLLM's CUDA worker after CRIU restored invalid driver state.

  CRIU can restore Python and IPC state while NVIDIA driver objects owned by
  the old CUDA context (streams, graphs, NCCL channels) no longer exist.
  Rebinding tensors in that context is not enough.  This follows vLLM's
  normal worker startup sequence after tearing down the restored worker,
  including a fresh CUDA context, distributed group, model runner, KV cache,
  and warmup artifacts.
  """

  import copy
  import ctypes
  import ctypes.util

  import torch

  restore_status = (
    Path(os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current"))
    .parent
    / ".vllm-criu-restore-status"
  )

  def stage(name: str) -> None:
    try:
      restore_status.write_text(f"stage {name}\n")
    except OSError:
      pass

  old_runner = worker.model_runner
  kv_cache_config = copy.deepcopy(getattr(old_runner, "kv_cache_config", None))
  old_local_rank = worker.local_rank

  LOG.info("starting full CUDA/vLLM worker rebootstrap after CRIU restore")
  stage("rebootstrap_start")
  try:
    old_runner.shutdown()
  except Exception:
    LOG.exception("old model runner cleanup failed; continuing with rebootstrap")
  worker.model_runner = None
  stage("old_runner_shutdown")
  worker._sleep_saved_buffers.clear()
  worker._sleep_saved_draft_buffers.clear()
  if worker.weight_transfer_engine is not None:
    try:
      worker.weight_transfer_engine.shutdown()
    except Exception:
      LOG.exception("old weight-transfer cleanup failed")
    worker.weight_transfer_engine = None

  from vllm.device_allocator.cumem import CuMemAllocator

  if CuMemAllocator.instance is not None:
    try:
      CuMemAllocator.instance.release_pools()
    except Exception:
      LOG.exception("old CuMemAllocator pool cleanup failed")
    CuMemAllocator.instance = None
  stage("allocator_released")

  try:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
  except Exception:
    LOG.exception("old distributed state cleanup failed; continuing")
  gc.collect()
  stage("distributed_cleanup")

  # cudaDeviceReset destroys the driver objects left behind by the restored
  # context.  PyTorch will lazily create a new primary context on init_device.
  cudart_path = ctypes.util.find_library("cudart") or "libcudart.so"
  reset_status = ctypes.CDLL(cudart_path).cudaDeviceReset()
  if reset_status:
    raise RuntimeError(f"cudaDeviceReset failed with status {reset_status}")
  stage("cuda_reset")

  worker.local_rank = old_local_rank
  worker._sleep_mode_backend = None
  torch.cuda.init()
  # The restored process can retain a small amount of driver memory even
  # after cudaDeviceReset(). vLLM's startup guard compares the configured
  # utilization against the instantaneous free memory and otherwise rejects
  # a restore that is only a few hundred MiB below the cold-start budget.
  # Keep the user's target when it fits; otherwise use a conservative
  # post-reset budget for this rebootstrap only.
  try:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    configured_utilization = worker.cache_config.gpu_memory_utilization
    available_utilization = (free_bytes / total_bytes) - 0.01
    restore_utilization = min(configured_utilization, available_utilization)
    if restore_utilization < configured_utilization:
      worker.cache_config.gpu_memory_utilization = max(
        0.80, restore_utilization
      )
      LOG.warning(
        "adjusted restore GPU memory utilization from %.4f to %.4f "
        "after CUDA reset (%s/%s bytes free)",
        configured_utilization,
        worker.cache_config.gpu_memory_utilization,
        free_bytes,
        total_bytes,
      )
  except Exception:
    LOG.exception("could not calculate restore GPU memory budget")
  stage("device_init_start")
  worker.init_device()
  stage("device_init_done")
  worker.load_model()
  stage("model_load_done")
  if kv_cache_config is not None:
    worker.initialize_from_config(kv_cache_config)
    stage("kv_init_done")
    if os.environ.get("VLLM_LIFECYCLE_RESTORE_WARMUP", "0").lower() in {
      "1",
      "true",
      "yes",
      "on",
    }:
      worker.compile_or_warm_up_model()
    else:
      LOG.warning(
        "skipping restore-time Triton/CUDA warmup; using restored "
        "compile artifacts and lazy kernel initialization"
      )

  LOG.info("completed full CUDA/vLLM worker rebootstrap after CRIU restore")
  return {
    "rebootstrapped": True,
    "kv_cache_recreated": kv_cache_config is not None,
  }



__all__ = [
  "_discard_stale_engine_dead_outputs",
  "_install_allocator_restore_patch",
  "_rebootstrap_worker_after_restore",
  "_reinitialize_worker_allocations",
  "_wake_worker_allocations_after_restore",
]
