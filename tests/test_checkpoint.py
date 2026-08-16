## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from pathlib import Path
from subprocess import CompletedProcess

from vllm_criu.checkpoint import Criu, CudaCheckpoint


class FakeRunner:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = iter(outputs or [])
        self.calls: list[list[str]] = []

    def run(self, command: list[str], timeout: float) -> CompletedProcess[str]:
        self.calls.append(command)
        return CompletedProcess(command, 0, next(self.outputs, ""), "")


def test_restore_attempts_manual_cuda_restore_when_criu_reports_running() -> None:
    runner = FakeRunner(["running\n"])
    checkpoint = CudaCheckpoint("cuda-checkpoint", runner, 10)

    checkpoint.restore_if_needed(419)

    assert runner.calls == [
        ["cuda-checkpoint", "--get-state", "--pid", "419"],
        ["cuda-checkpoint", "--action", "restore", "--pid", "419"],
        ["cuda-checkpoint", "--action", "unlock", "--pid", "419"],
    ]


def test_restore_restores_and_unlocks_when_cuda_is_not_running() -> None:
    runner = FakeRunner(["locked\n", "", ""])
    checkpoint = CudaCheckpoint("cuda-checkpoint", runner, 10)

    checkpoint.restore_if_needed(419)

    assert runner.calls == [
        ["cuda-checkpoint", "--get-state", "--pid", "419"],
        ["cuda-checkpoint", "--action", "restore", "--pid", "419"],
        ["cuda-checkpoint", "--action", "unlock", "--pid", "419"],
    ]


def test_criu_dump_uses_options_required_by_vllm_process_tree(tmp_path: Path) -> None:
    runner = FakeRunner()
    criu = Criu("criu", runner, 10)

    criu.dump(22, tmp_path)

    command = runner.calls[0]
    assert "--tcp-established" in command
    assert "--link-remap" in command
    assert command[command.index("--tree") + 1] == "22"
