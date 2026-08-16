## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""EngineCore input/output socket loops for CRIU reconnects."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .diagnostics import (
  _append_restore_diagnostic,
  _restore_active_paths,
  _transport_debug_enabled,
)

LOG = logging.getLogger("vllm-criu.enginecore-patch")


def _input_sockets(self, input_addresses, coord_input_address, identity, ready_event):
  import msgspec
  import zmq
  from contextlib import ExitStack

  from vllm.multimodal.cache import MultiModalCacheMissError
  from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
  from vllm.utils.network_utils import make_zmq_socket
  from vllm.v1.engine.core import FT_UTILITY_METHOD
  from vllm.v1.serial_utils import MsgpackDecoder

  self._vllm_criu_input_args = (
    input_addresses,
    coord_input_address,
    identity,
    ready_event,
  )
  restore_marker = (
    Path(os.environ.get("LAUNCHER_CHECKPOINT_DIR", "/checkpoints/current"))
    .parent
    / ".vllm-criu-restore-request"
  )
  add_request_decoder = MsgpackDecoder(
    EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
  )
  generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)
  ready_sent = False
  generation_seen = -1

  while True:
    generation_seen = self._vllm_criu_reconnect_generation
    if generation_seen:
      LOG.info("rebuilding EngineCore input sockets after restore generation %s", generation_seen)
      # A restored ROUTER can retain the old DEALER route in the
      # opposite direction. Use a fresh identity so the API can adopt
      # the new connection instead of routing utility requests to the
      # stale restored peer.
      identity = f"vllm-criu-{self.engine_index}-{generation_seen}".encode()
    with ExitStack() as stack, zmq.Context() as ctx:
      input_sockets = [
        stack.enter_context(
          make_zmq_socket(ctx, address, zmq.DEALER, identity=identity, bind=False)
        )
        for address in input_addresses
      ]
      coord_socket = None
      if coord_input_address is not None:
        coord_socket = stack.enter_context(
          make_zmq_socket(
            ctx, coord_input_address, zmq.XSUB, identity=identity, bind=False
          )
        )
        coord_socket.send(b"\x01")

      self._vllm_criu_input_sockets = [
        *input_sockets,
        *([coord_socket] if coord_socket is not None else []),
      ]

      poller = zmq.Poller()
      ready_payload = msgspec.msgpack.encode(self._make_ready_response())
      for input_socket in input_sockets:
        input_socket.send(ready_payload)
        poller.register(input_socket, zmq.POLLIN)

      if coord_socket is not None:
        while generation_seen == self._vllm_criu_reconnect_generation:
          if coord_socket.poll(250):
            assert coord_socket.recv() == b"READY"
            poller.register(coord_socket, zmq.POLLIN)
            break
        if generation_seen != self._vllm_criu_reconnect_generation:
          continue

      if not ready_sent:
        ready_event.set()
        ready_sent = True

      # CRIU can restore a queue waiter with a stale condition-variable
      # state.  Once the replacement socket is ready, explicitly wake
      # the EngineCore loop so it re-checks input_queue.
      try:
        with self.input_queue.not_empty:
          self.input_queue.not_empty.notify_all()
      except Exception:
        LOG.exception("failed to notify restored EngineCore input queue")
      try:
        # A restored EngineCore can retain an idle main thread inside
        # Queue.get() even after the replacement ZMQ thread has
        # reconnected.  The condition notification above is subject
        # to a restore-time waiter race; a real queue item gives the
        # main loop an unambiguous wake-up and is harmless because
        # WAKEUP is intentionally a no-op in _handle_client_request.
        self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
        LOG.warning("woke restored EngineCore dispatch loop")
      except Exception:
        LOG.exception("failed to wake restored EngineCore dispatch loop")

      while generation_seen == self._vllm_criu_reconnect_generation:
        # The restored supervisor may itself be asleep in a CRIU-
        # restored futex.  This socket poll is independently active,
        # so use the launcher's external marker here as a second,
        # deterministic rehook trigger.
        if generation_seen <= 0 and restore_marker.exists():
          self._vllm_criu_reconnect_generation = 1
          break
        try:
          events = poller.poll(250)
        except zmq.ZMQError:
          break
        for input_socket, _ in events:
          type_frame, *data_frames = input_socket.recv_multipart(copy=False)
          if type_frame.buffer == b"READY":
            assert input_socket == coord_socket
            continue
          request_type = EngineCoreRequestType(bytes(type_frame.buffer))
          if _transport_debug_enabled():
            LOG.warning(
              "received restored EngineCore request type=%s",
              request_type,
            )
          if request_type == EngineCoreRequestType.ADD:
            request = add_request_decoder.decode(data_frames)
            try:
              request = self.preprocess_add_request(request)
            except MultiModalCacheMissError as error:
              self._handle_mm_cache_miss(request, error)
              continue
            except Exception:
              self._handle_request_preproc_error(request)
              continue
          elif request_type == EngineCoreRequestType.UTILITY:
            request = generic_decoder.decode(data_frames)
            client_idx, call_id, method, args = request
            if method in {
              "checkpoint_restore",
              "vllm_criu_reinitialize_after_restore",
              "reload_weights",
            }:
              LOG.warning("dispatching restored EngineCore utility %s", method)
            if method == FT_UTILITY_METHOD:
              self.ft_sentinel.handle_command(client_idx, call_id, args[0])
              continue
          else:
            request = generic_decoder.decode(data_frames)
            if request_type == EngineCoreRequestType.ABORT:
              self.aborts_queue.put_nowait(request)

          self.input_queue.put_nowait((request_type, request))


def _output_sockets(self, output_paths, coord_output_path, engine_index):
  import queue
  import zmq
  from collections import deque
  from contextlib import ExitStack

  from vllm.utils.network_utils import make_zmq_socket
  from vllm.v1.serial_utils import MsgpackEncoder

  self._vllm_criu_output_args = (output_paths, coord_output_path, engine_index)
  encoder = MsgpackEncoder()
  generation_seen = -1

  while True:
    generation_seen = self._vllm_criu_reconnect_generation
    if generation_seen:
      LOG.info("rebuilding EngineCore output sockets after restore generation %s", generation_seen)
    reuse_buffers = []
    pending = deque()
    with ExitStack() as stack, zmq.Context() as ctx:
      sockets = [
        stack.enter_context(make_zmq_socket(ctx, path, zmq.PUSH, linger=4000))
        for path in output_paths
      ]
      coord_socket = (
        stack.enter_context(
          make_zmq_socket(ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000)
        )
        if coord_output_path is not None
        else None
      )
      self._vllm_criu_output_sockets = [
        *sockets,
        *([coord_socket] if coord_socket is not None else []),
      ]
      max_reuse_buffers = len(sockets) + 1

      while generation_seen == self._vllm_criu_reconnect_generation:
        try:
          output = self.output_queue.get(timeout=0.25)
        except queue.Empty:
          continue
        if output == self.ENGINE_CORE_DEAD:
          pending_marker, restore_marker = _restore_active_paths()
          _append_restore_diagnostic(
            ".vllm-criu-enginecore-dead.log",
            "generation=%s pending=%s request=%s local_restore_complete=%s "
            "local_restore_in_progress=%s output_thread=%s\n"
            % (
              generation_seen,
              pending_marker.exists(),
              restore_marker.exists(),
              getattr(self, "_vllm_criu_local_restore_complete", False),
              getattr(self, "_vllm_criu_local_restore_in_progress", False),
              threading.current_thread().name,
            ),
          )
          if pending_marker.exists() or restore_marker.exists():
            # The pre-dump EngineCore shutdown path can leave its
            # dead sentinel in output_queue. CRIU restores that
            # Python queue together with a live EngineCore, so
            # forwarding the stale sentinel would make the API
            # client permanently mark the healthy restored engine
            # dead before its first request.
            LOG.warning(
              "discarding stale ENGINE_CORE_DEAD during CRIU restore"
            )
            continue
          LOG.error("forwarding ENGINE_CORE_DEAD after CRIU restore")
          for socket in sockets:
            socket.send(output)
          return

        client_index, outputs = output
        outputs.engine_index = engine_index
        if _transport_debug_enabled():
          LOG.warning(
            "sending EngineCore output client=%s outputs=%s stats=%s",
            client_index,
            len(outputs.outputs),
            outputs.scheduler_stats is not None,
          )
        if client_index == -1:
          assert coord_socket is not None
          coord_socket.send_multipart(encoder.encode(outputs))
          continue

        while pending and pending[-1][0].done:
          reclaimed = pending.pop()[1]
          if len(reuse_buffers) < max_reuse_buffers:
            reuse_buffers.append(reclaimed)
        buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
        buffers = encoder.encode_into(outputs, buffer)
        tracker = self._send_msg_tracking_payload(sockets[client_index], buffers)
        if not tracker.done:
          pending.appendleft((tracker, buffer))
        elif len(reuse_buffers) < max_reuse_buffers:
          reuse_buffers.append(buffer)



__all__ = [
  "_input_sockets",
  "_output_sockets",
]
