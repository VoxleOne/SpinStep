# array_backend.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""NumPy / CuPy array backend selection."""

from __future__ import annotations

__all__ = ["get_array_module"]

from types import ModuleType


def get_array_module(use_cuda: bool = False) -> ModuleType:
    """Return the appropriate array module (NumPy or CuPy).

    Parameters
    ----------
    use_cuda:
        When *True*, attempt to import and return :mod:`cupy`.  Falls back to
        :mod:`numpy` if CuPy is not installed.

    Returns
    -------
    ModuleType
        Either :mod:`cupy` or :mod:`numpy`.
    """
    if use_cuda:
        try:
            import cupy as cp

            return cp
        except ImportError:
            pass
    import numpy as np

    return np
