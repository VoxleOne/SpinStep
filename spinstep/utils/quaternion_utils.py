# quaternion_utils.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion conversion, distance, and manipulation utilities."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R


def quaternion_from_euler(
    angles: Sequence[float],
    order: str = "zyx",
    degrees: bool = True,
) -> np.ndarray:
    """Convert Euler angles to a quaternion ``[x, y, z, w]``."""
    return R.from_euler(order, angles, degrees=degrees).as_quat()


def quaternion_distance(q1: ArrayLike, q2: ArrayLike) -> float:
    """Return the angular distance (radians) between two quaternions."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return float((r1.inv() * r2).magnitude())


def rotate_quaternion(q: ArrayLike, rotation_step: ArrayLike) -> np.ndarray:
    """Apply *rotation_step* to quaternion *q* and return the result."""
    r1 = R.from_quat(q)
    step = R.from_quat(rotation_step)
    return (r1 * step).as_quat()


def is_within_angle_threshold(
    q_current: ArrayLike,
    q_target: ArrayLike,
    threshold_rad: float,
) -> bool:
    """Check whether two quaternions are within *threshold_rad* of each other."""
    return quaternion_distance(q_current, q_target) < threshold_rad


def quaternion_conjugate(q: ArrayLike) -> np.ndarray:
    """Return the conjugate of quaternion *q* ``[x, y, z, w]``."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    """Hamilton product of two quaternions in ``[x, y, z, w]`` order."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def rotation_matrix_to_quaternion(m: ArrayLike) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion ``[x, y, z, w]``."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qx = 0.25 * s
        qw = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qy = 0.25 * s
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qz = 0.25 * s
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
    q = np.array([qx, qy, qz, qw])
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def get_relative_spin(nf: object, nt: object) -> np.ndarray:
    """Return the relative quaternion rotation from node *nf* to node *nt*.

    Both nodes must have an ``.orientation`` attribute storing a quaternion
    ``[x, y, z, w]``.
    """
    qfc = quaternion_conjugate(nf.orientation)  # type: ignore[union-attr]
    qr = quaternion_multiply(qfc, nt.orientation)  # type: ignore[union-attr]
    n = np.linalg.norm(qr)
    return qr / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def get_unique_relative_spins(
    nodes: Sequence[object],
    nside: int,
    nest: bool,
    threshold: float = 1e-3,
) -> List[np.ndarray]:
    """Compute unique relative rotations between HEALPix neighbours.

    Requires the ``healpy`` package.

    Parameters
    ----------
    nodes:
        Sequence of node objects with ``.orientation`` attributes.
    nside:
        HEALPix *nside* parameter.
    nest:
        Whether to use the NESTED pixel ordering.
    threshold:
        Angular threshold (radians) for considering two rotations identical.
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError(
            "healpy is required for get_unique_relative_spins(). "
            "Install it with: pip install healpy"
        )
    spins: List[np.ndarray] = []
    NPIX = hp.nside2npix(nside)
    for i in range(NPIX):
        nf = nodes[i]
        nidx = hp.get_all_neighbours(nside, i, nest=nest)
        for idx in nidx:
            if idx != -1:
                q = get_relative_spin(nf, nodes[idx])
                if q[3] < 0:
                    q = -q  # Canonical form (w >= 0)
                is_uniq = True
                for s_q in spins:
                    dot = np.abs(np.dot(q, s_q))
                    dot = np.clip(dot, -1, 1)
                    angle = 2 * np.arccos(dot)
                    if angle < threshold:
                        is_uniq = False
                        break
                if is_uniq:
                    spins.append(q)
    return spins
