# math/constraints.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion constraints: clamping, limiting rotation magnitude."""

from __future__ import annotations

__all__ = [
    "clamp_rotation_angle",
]

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R


def clamp_rotation_angle(q: ArrayLike, max_angle: float) -> npt.NDArray[np.floating[Any]]:
    """Clamp a rotation quaternion so its angle does not exceed *max_angle*.

    If the rotation represented by *q* has an angle larger than *max_angle*,
    the quaternion is scaled to represent exactly *max_angle* around the
    same axis.

    Args:
        q: Rotation quaternion ``[x, y, z, w]``.
        max_angle: Maximum allowed rotation angle in radians.  Must be
            non-negative.

    Returns:
        Clamped unit quaternion ``[x, y, z, w]``.

    Raises:
        ValueError: If *max_angle* is negative.
    """
    if max_angle < 0:
        raise ValueError(f"max_angle must be non-negative, got {max_angle}")

    rot = R.from_quat(q)
    angle = rot.magnitude()

    if angle <= max_angle:
        return np.asarray(q, dtype=float)

    # Scale the rotation vector to max_angle
    rotvec = rot.as_rotvec()
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])

    axis = rotvec / angle
    clamped_rotvec = axis * max_angle
    return cast(npt.NDArray[np.floating[Any]], R.from_rotvec(clamped_rotvec).as_quat())
