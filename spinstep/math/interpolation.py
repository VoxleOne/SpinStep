# math/interpolation.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion interpolation: SLERP and SQUAD."""

from __future__ import annotations

__all__ = [
    "slerp",
    "squad",
]

import numpy as np
from numpy.typing import ArrayLike


def slerp(q0: ArrayLike, q1: ArrayLike, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Interpolates along the shortest arc on the unit quaternion hypersphere.

    Args:
        q0: Start quaternion ``[x, y, z, w]``.
        q1: End quaternion ``[x, y, z, w]``.
        t: Interpolation parameter in ``[0, 1]``.

    Returns:
        Interpolated unit quaternion ``[x, y, z, w]``.
    """
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)

    # Normalize inputs
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Ensure shortest path
    dot = np.dot(a, b)
    if dot < 0.0:
        b = -b
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * a + s1 * b
    return result / np.linalg.norm(result)


def squad(
    q0: ArrayLike,
    q1: ArrayLike,
    q2: ArrayLike,
    q3: ArrayLike,
    t: float,
) -> np.ndarray:
    """Spherical cubic interpolation (SQUAD) between quaternion waypoints.

    Produces a smooth C¹-continuous curve through a sequence of orientations.
    *q0* and *q3* are the neighboring control points; the interpolation is
    between *q1* (at *t* = 0) and *q2* (at *t* = 1).

    Args:
        q0: Control quaternion before *q1*.
        q1: Start quaternion for this segment.
        q2: End quaternion for this segment.
        q3: Control quaternion after *q2*.
        t: Interpolation parameter in ``[0, 1]``.

    Returns:
        Interpolated unit quaternion ``[x, y, z, w]``.
    """
    a1 = np.asarray(q1, dtype=float)
    a2 = np.asarray(q2, dtype=float)

    s1 = _squad_intermediate(np.asarray(q0, dtype=float), a1, a2)
    s2 = _squad_intermediate(a1, a2, np.asarray(q3, dtype=float))

    slerp_q1_q2 = slerp(a1, a2, t)
    slerp_s1_s2 = slerp(s1, s2, t)
    return slerp(slerp_q1_q2, slerp_s1_s2, 2.0 * t * (1.0 - t))


def _squad_intermediate(
    q_prev: np.ndarray, q_curr: np.ndarray, q_next: np.ndarray
) -> np.ndarray:
    """Compute the SQUAD intermediate control point for *q_curr*."""
    from .core import quaternion_conjugate, quaternion_multiply

    q_curr = q_curr / np.linalg.norm(q_curr)
    inv_curr = quaternion_conjugate(q_curr)

    log_prev = _quat_log(quaternion_multiply(inv_curr, q_prev / np.linalg.norm(q_prev)))
    log_next = _quat_log(quaternion_multiply(inv_curr, q_next / np.linalg.norm(q_next)))

    avg = -(log_prev + log_next) / 4.0
    result = quaternion_multiply(q_curr, _quat_exp(avg))
    return result / np.linalg.norm(result)


def _quat_log(q: np.ndarray) -> np.ndarray:
    """Quaternion logarithm (returns pure-quaternion vector part)."""
    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 0.0])
    theta = np.arctan2(vec_norm, w)
    return np.array([*(vec / vec_norm * theta), 0.0])


def _quat_exp(v: np.ndarray) -> np.ndarray:
    """Quaternion exponential (from pure-quaternion vector)."""
    vec = v[:3]
    theta = np.linalg.norm(vec)
    if theta < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / theta
    return np.array([*(axis * np.sin(theta)), np.cos(theta)])
