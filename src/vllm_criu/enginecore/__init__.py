## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""CRIU recovery support for vLLM V1 EngineCore."""

from .patch import install_enginecore_restore_patch

__all__ = [
  "install_enginecore_restore_patch",
]
