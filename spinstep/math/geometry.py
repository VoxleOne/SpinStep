# math/geometry.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion geometry: distance, angle, direction conversions."""

from __future__ import annotations

__all__ = [
    "quaternion_distance",
    "is_within_angle_threshold",
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "angle_between_directions",
    "rotate_quaternion",
]

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R


def quaternion_distance(q1: ArrayLike, q2: ArrayLike) -> float:
    """Return the angular distance (radians) between two quaternions.

    Args:
        q1: First quaternion ``[x, y, z, w]``.
        q2: Second quaternion ``[x, y, z, w]``.

    Returns:
        Angular distance in radians.
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return float((r1.inv() * r2).magnitude())


def is_within_angle_threshold(
    q_current: ArrayLike,
    q_target: ArrayLike,
    threshold_rad: float,
) -> bool:
    """Check whether two quaternions are within *threshold_rad* of each other.

    Args:
        q_current: Current quaternion ``[x, y, z, w]``.
        q_target: Target quaternion ``[x, y, z, w]``.
        threshold_rad: Maximum angular distance in radians.

    Returns:
        ``True`` if the angular distance is less than *threshold_rad*.
    """
    return quaternion_distance(q_current, q_target) < threshold_rad


def rotate_quaternion(q: ArrayLike, rotation_step: ArrayLike) -> np.ndarray:
    """Apply *rotation_step* to quaternion *q* and return the result.

    Args:
        q: Base quaternion ``[x, y, z, w]``.
        rotation_step: Rotation to apply, as quaternion ``[x, y, z, w]``.

    Returns:
        Composed quaternion ``[x, y, z, w]``.
    """
    r1 = R.from_quat(q)
    step = R.from_quat(rotation_step)
    return (r1 * step).as_quat()


def forward_vector_from_quaternion(q: ArrayLike) -> np.ndarray:
    """Extract the forward (look) direction from a quaternion.

    The forward direction is defined as ``[0, 0, -1]`` rotated by the
    quaternion, following the convention where negative-Z is "forward".

    Args:
        q: Quaternion ``[x, y, z, w]``.

    Returns:
        Unit direction vector ``(3,)`` pointing forward.
    """
    return R.from_quat(q).apply([0, 0, -1])


def direction_to_quaternion(direction: ArrayLike) -> np.ndarray:
    """Convert a 3D direction vector to an orientation quaternion.

    The returned quaternion represents the rotation that aligns the
    default forward axis ``[0, 0, -1]`` with the given *direction*.

    Args:
        direction: Target direction vector (does not need to be normalised).

    Returns:
        Unit quaternion ``[x, y, z, w]``.
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    d = d / norm
    rot, _ = R.align_vectors([d], [[0, 0, -1]])
    return rot.as_quat()


def angle_between_directions(d1: ArrayLike, d2: ArrayLike) -> float:
    """Compute the angular distance (radians) between two direction vectors.

    Args:
        d1: First direction vector.
        d2: Second direction vector.

    Returns:
        Angle in radians in the range ``[0, π]``.
    """
    v1 = np.asarray(d1, dtype=float)
    v2 = np.asarray(d2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.dot(v1 / n1, v2 / n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))
