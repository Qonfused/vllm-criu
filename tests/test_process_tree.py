## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from vllm_criu import process_tree


def test_find_resource_trackers_only_returns_descendants(monkeypatch) -> None:
    monkeypatch.setattr(process_tree, "descendants", lambda _root: [101, 102, 103])
    command_lines = {
        101: "python -m multiprocessing.resource_tracker",
        102: "VLLM::EngineCore",
        103: "python worker.py",
    }
    monkeypatch.setattr(process_tree, "command_line", command_lines.get)

    assert process_tree.find_resource_trackers(22) == [101]


def test_find_resource_trackers_does_not_match_unrelated_tracker_text(monkeypatch) -> None:
    monkeypatch.setattr(process_tree, "descendants", lambda _root: [101])
    monkeypatch.setattr(process_tree, "command_line", lambda _pid: "resource_tracker_helper")

    assert process_tree.find_resource_trackers(22) == []
