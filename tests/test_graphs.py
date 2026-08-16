## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##

from types import SimpleNamespace

from vllm_criu.enginecore.graphs import _graph_entry_count, graph_preservation_enabled


def test_graph_preservation_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_LIFECYCLE_PRESERVE_CUDAGRAPHS", raising=False)

    assert not graph_preservation_enabled()


def test_graph_preservation_accepts_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("VLLM_LIFECYCLE_PRESERVE_CUDAGRAPHS", value)
        assert graph_preservation_enabled()


def test_graph_entry_count_counts_known_and_live_captures() -> None:
    wrapper = SimpleNamespace(
        concrete_cudagraph_entries={
            "full-4": SimpleNamespace(cudagraph=object()),
            "full-8": SimpleNamespace(cudagraph=object()),
            "piecewise-4": SimpleNamespace(capture=object()),
            "empty": SimpleNamespace(cudagraph=None, capture=None),
        }
    )

    assert _graph_entry_count(wrapper) == (4, 3)


def test_graph_entry_count_supports_legacy_entries_attribute() -> None:
    wrapper = SimpleNamespace(
        entries={
            "full-4": SimpleNamespace(cudagraph=object()),
            "full-8": SimpleNamespace(cudagraph=None, capture=object()),
        }
    )

    assert _graph_entry_count(wrapper) == (2, 2)
