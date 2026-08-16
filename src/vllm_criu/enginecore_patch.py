## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""Backward-compatible import for the EngineCore CRIU patch."""

from .enginecore import install_enginecore_restore_patch

__all__ = [
  "install_enginecore_restore_patch",
]
