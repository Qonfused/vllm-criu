#!/usr/bin/env python3
## @file
# Copyright (c) 2026, Cory Bennett. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
##
"""Compatibility shim for the uv-managed lifecycle package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).with_name("src")))

from vllm_criu.launcher import main


if __name__ == "__main__":
  main()
