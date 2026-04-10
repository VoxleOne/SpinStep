# math/conversions.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion conversions: Euler, rotation matrix, rotation vector."""

from __future__ import annotations

__all__ = [
    "quaternion_from_euler",
    "rotation_matrix_to_quaternion",
    "quaternion_from_rotvec",
    "quaternion_to_rotvec",
]

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R


def quaternion_from_euler(
    angles: Sequence[float],
    order: str = "zyx",
    degrees: bool = True,
) -> np.ndarray:
    """Convert Euler angles to a quaternion ``[x, y, z, w]``.

    Args:
        angles: Euler angles as a 3-element sequence.
        order: Rotation order string (e.g. ``"zyx"``).
        degrees: If ``True``, angles are in degrees; otherwise radians.

    Returns:
        Unit quaternion ``[x, y, z, w]``.
    """
    return R.from_euler(order, angles, degrees=degrees).as_quat()


def rotation_matrix_to_quaternion(m: ArrayLike) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion ``[x, y, z, w]``.

    Args:
        m: A 3×3 rotation matrix.

    Returns:
        Unit quaternion ``[x, y, z, w]``.
    """
    mat = np.asarray(m, dtype=float)
    t = np.trace(mat)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        qw = 0.25 * s
        qx = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 2] - mat[2, 0]) / s
        qz = (mat[1, 0] - mat[0, 1]) / s
    elif (mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2]):
        s = np.sqrt(1 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
        qx = 0.25 * s
        qw = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 1] + mat[1, 0]) / s
        qz = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
        qy = 0.25 * s
        qw = (mat[0, 2] - mat[2, 0]) / s
        qx = (mat[0, 1] + mat[1, 0]) / s
        qz = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
        qz = 0.25 * s
        qw = (mat[1, 0] - mat[0, 1]) / s
        qx = (mat[0, 2] + mat[2, 0]) / s
        qy = (mat[1, 2] + mat[2, 1]) / s
    q = np.array([qx, qy, qz, qw])
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def quaternion_from_rotvec(rotvec: ArrayLike) -> np.ndarray:
    """Convert a rotation vector (axis × angle) to a quaternion.

    Args:
        rotvec: Rotation vector of shape ``(3,)``.

    Returns:
        Unit quaternion ``[x, y, z, w]``.
    """
    return R.from_rotvec(np.asarray(rotvec, dtype=float)).as_quat()


def quaternion_to_rotvec(q: ArrayLike) -> np.ndarray:
    """Convert a quaternion to a rotation vector (axis × angle).

    Args:
        q: Quaternion ``[x, y, z, w]``.

    Returns:
        Rotation vector of shape ``(3,)``.
    """
    return R.from_quat(np.asarray(q, dtype=float)).as_rotvec()
