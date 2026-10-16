"""leaklab.model - backwards-compatibility shim.

The data layer now lives in :mod:`leaklab.data`.  This module re-exports its
public surface so existing imports keep working.
"""
from __future__ import annotations

from .data import (  # noqa: F401
    ADVERSE_ROLES,
    TERMINAL_ROLES,
    JoinHealth,
    Model,
    build_feature_table,
    build_model,
    classify_status,
)

__all__ = [
    "ADVERSE_ROLES",
    "TERMINAL_ROLES",
    "JoinHealth",
    "Model",
    "build_feature_table",
    "build_model",
    "classify_status",
]
