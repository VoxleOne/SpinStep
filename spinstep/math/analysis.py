# math/analysis.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion analysis: batch distances, angular velocity, relative spins."""

from __future__ import annotations

__all__ = [
    "batch_quaternion_angle",
    "angular_velocity_from_quaternions",
    "get_relative_spin",
    "get_unique_relative_spins",
    "SpatialNodeProtocol",
]

from types import ModuleType
from typing import Any, List, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from .core import quaternion_conjugate, quaternion_multiply


@runtime_checkable
class NodeProtocol(Protocol):
    """Structural type for objects accepted by :func:`get_relative_spin`.

    Any object with an ``orientation`` attribute holding a quaternion
    ``[x, y, z, w]`` satisfies this protocol.
    """

    orientation: npt.NDArray[np.floating[Any]]


@runtime_checkable
class SpatialNodeProtocol(Protocol):
    """Structural type for spatial nodes with orientation, distance, and name.

    Any object with ``orientation``, ``distance``, and ``name`` attributes
    satisfies this protocol.  Used by :func:`~spinstep.control.state.compute_relative_state`.
    """

    orientation: npt.NDArray[np.floating[Any]]
    distance: float
    name: str


def batch_quaternion_angle(qs1: Any, qs2: Any, xp: ModuleType) -> Any:
    """Compute pairwise angular distances between two sets of quaternions.

    Args:
        qs1: Array of shape ``(N, 4)`` — first set of quaternions.
        qs2: Array of shape ``(M, 4)`` — second set of quaternions.
        xp: Array module (:mod:`numpy` or :mod:`cupy`).

    Returns:
        ``(N, M)`` array of angular distances in radians.
    """
    dots = xp.abs(xp.dot(qs1, qs2.T))
    dots = xp.clip(dots, -1.0, 1.0)
    angles = 2 * xp.arccos(dots)
    return angles


def angular_velocity_from_quaternions(
    q1: ArrayLike, q2: ArrayLike, dt: float
) -> npt.NDArray[np.floating[Any]]:
    """Estimate angular velocity from two quaternions separated by *dt* seconds.

    Computes the rotation from *q1* to *q2*, converts to a rotation vector
    (axis × angle), and divides by *dt* to obtain angular velocity in rad/s.

    Args:
        q1: Start quaternion ``[x, y, z, w]``.
        q2: End quaternion ``[x, y, z, w]``.
        dt: Time step in seconds.  Must be positive.

    Returns:
        Angular velocity vector ``(3,)`` in radians per second.

    Raises:
        ValueError: If *dt* is not positive.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    from scipy.spatial.transform import Rotation as R

    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    delta = r1.inv() * r2
    rotvec = delta.as_rotvec()
    return rotvec / dt


def get_relative_spin(nf: NodeProtocol, nt: NodeProtocol) -> npt.NDArray[np.floating[Any]]:
    """Return the relative quaternion rotation from node *nf* to node *nt*.

    Both nodes must have an ``.orientation`` attribute storing a quaternion
    ``[x, y, z, w]``.

    Args:
        nf: Source node with ``.orientation`` attribute.
        nt: Target node with ``.orientation`` attribute.

    Returns:
        Unit quaternion representing the relative rotation.
    """
    qfc = quaternion_conjugate(nf.orientation)
    qr = quaternion_multiply(qfc, nt.orientation)
    n = np.linalg.norm(qr)
    return qr / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def get_unique_relative_spins(
    nodes: Sequence[NodeProtocol],
    nside: int,
    nest: bool,
    threshold: float = 1e-3,
) -> List[npt.NDArray[np.floating[Any]]]:
    """Compute unique relative rotations between HEALPix neighbours.

    Requires the ``healpy`` package.

    Args:
        nodes: Sequence of node objects with ``.orientation`` attributes.
        nside: HEALPix *nside* parameter.
        nest: Whether to use the NESTED pixel ordering.
        threshold: Angular threshold (radians) for considering two
            rotations identical.

    Returns:
        List of unique unit quaternions representing relative rotations.

    Raises:
        ImportError: If ``healpy`` is not installed.
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError(
            "healpy is required for get_unique_relative_spins(). "
            "Install it with: pip install healpy"
        )
    spins: List[npt.NDArray[np.floating[Any]]] = []
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
