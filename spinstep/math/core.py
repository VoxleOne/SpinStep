# math/core.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Core quaternion operations: multiply, conjugate, normalize, inverse."""

from __future__ import annotations

__all__ = [
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_normalize",
    "quaternion_inverse",
]

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> npt.NDArray[np.floating[Any]]:
    """Hamilton product of two quaternions in ``[x, y, z, w]`` order.

    Args:
        q1: First quaternion ``[x, y, z, w]``.
        q2: Second quaternion ``[x, y, z, w]``.

    Returns:
        Product quaternion ``[x, y, z, w]`` as a NumPy array of shape ``(4,)``.
    """
    a1 = np.asarray(q1, dtype=float)
    a2 = np.asarray(q2, dtype=float)
    x1, y1, z1, w1 = a1
    x2, y2, z2, w2 = a2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quaternion_conjugate(q: ArrayLike) -> npt.NDArray[np.floating[Any]]:
    """Return the conjugate of quaternion *q* ``[x, y, z, w]``.

    The conjugate negates the vector part while keeping the scalar part.

    Args:
        q: Quaternion ``[x, y, z, w]``.

    Returns:
        Conjugate quaternion ``[-x, -y, -z, w]``.
    """
    a = np.asarray(q, dtype=float)
    return np.array([-a[0], -a[1], -a[2], a[3]])


def quaternion_normalize(q: ArrayLike) -> npt.NDArray[np.floating[Any]]:
    """Normalize a quaternion to unit length.

    Args:
        q: Quaternion ``[x, y, z, w]``.

    Returns:
        Unit quaternion. Returns ``[0, 0, 0, 1]`` if the input has
        near-zero norm.
    """
    a = np.asarray(q, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return cast(npt.NDArray[np.floating[Any]], a / n)


def quaternion_inverse(q: ArrayLike) -> npt.NDArray[np.floating[Any]]:
    """Return the inverse of a unit quaternion.

    For unit quaternions the inverse equals the conjugate.

    Args:
        q: Unit quaternion ``[x, y, z, w]``.

    Returns:
        Inverse quaternion ``[x, y, z, w]``.
    """
    a = np.asarray(q, dtype=float)
    norm_sq = np.dot(a, a)
    if norm_sq < 1e-16:
        return np.array([0.0, 0.0, 0.0, 1.0])
    conj = np.array([-a[0], -a[1], -a[2], a[3]])
    return cast(npt.NDArray[np.floating[Any]], conj / norm_sq)
