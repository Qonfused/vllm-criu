## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from pathlib import Path

from vllm_criu.runtime import child_runtime


def test_enginecore_patch_is_injected_for_spawned_children(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_LIFECYCLE_ENGINECORE_PATCH", "1")
    monkeypatch.setenv("PYTHONPATH", "/existing")

    runtime = child_runtime("terminate")
    sitecustomize = Path("/tmp/vllm-criu-runtime/sitecustomize.py").read_text()

    assert "install_enginecore_restore_patch" in sitecustomize
    assert "from vllm_criu.enginecore import" in sitecustomize
    assert runtime.pass_fds == ()
    package_root = str(Path(__file__).resolve().parent.parent / "src")
    assert runtime.environment["PYTHONPATH"].endswith(f":{package_root}:/existing")
