# quaternion_math.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

from __future__ import annotations

from typing import Any


def batch_quaternion_angle(qs1: Any, qs2: Any, xp: Any) -> Any:
    """Compute pairwise angular distances between two sets of quaternions.

    Parameters
    ----------
    qs1:
        Array of shape ``(N, 4)`` — first set of quaternions.
    qs2:
        Array of shape ``(M, 4)`` — second set of quaternions.
    xp:
        Array module (:mod:`numpy` or :mod:`cupy`).

    Returns
    -------
    array
        ``(N, M)`` array of angular distances in radians.
    """
    dots = xp.abs(xp.dot(qs1, qs2.T))
    dots = xp.clip(dots, -1.0, 1.0)
    angles = 2 * xp.arccos(dots)
    return angles
