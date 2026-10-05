"""Process-wide settings that must be applied before numerical libraries load."""

from __future__ import annotations

import os
import sys


def configure_runtime() -> None:
    """Apply platform workarounds; idempotent, explicitly set variables win.

    On macOS arm64 PyTorch and lightgbm/xgboost each ship an OpenMP runtime.
    Loading both into one process segfaults unless duplicates are allowed, and
    a single OpenMP thread keeps the shared runtime stable. Linux (Docker, CI)
    is not affected and keeps the library defaults.
    """
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
