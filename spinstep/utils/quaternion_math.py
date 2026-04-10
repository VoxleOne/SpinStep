# quaternion_math.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Backward-compatible re-export of batch quaternion angle computation.

The implementation has moved to :mod:`spinstep.math.analysis`.

.. deprecated::
    Import from :mod:`spinstep.math` instead.
"""

from __future__ import annotations

__all__ = ["batch_quaternion_angle"]

# Re-export from canonical location
from spinstep.math.analysis import batch_quaternion_angle
