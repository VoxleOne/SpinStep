# control/state.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Orientation state model: dataclass, integration, and error computation."""

from __future__ import annotations

__all__ = [
    "OrientationState",
    "integrate_orientation",
    "compute_orientation_error",
]

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from ..math.core import quaternion_multiply, quaternion_normalize


@dataclass
class OrientationState:
    """Immutable orientation state: pose, angular velocity, and timestamp.

    All quaternions use ``[x, y, z, w]`` convention.

    Args:
        quaternion: Unit quaternion ``[x, y, z, w]`` representing the
            current orientation.
        angular_velocity: Angular velocity vector ``[ωx, ωy, ωz]`` in
            radians per second.  Defaults to zero.
        timestamp: Time in seconds.  Defaults to ``0.0``.

    Attributes:
        quaternion: Normalised quaternion as a NumPy array of shape ``(4,)``.
        angular_velocity: Angular velocity as a NumPy array of shape ``(3,)``.
        timestamp: Timestamp in seconds.

    Example::

        from spinstep.control import OrientationState

        state = OrientationState([0, 0, 0, 1])
        print(state.quaternion)  # [0. 0. 0. 1.]
    """

    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0

    def __init__(
        self,
        quaternion: ArrayLike = (0.0, 0.0, 0.0, 1.0),
        angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
        timestamp: float = 0.0,
    ) -> None:
        q = np.asarray(quaternion, dtype=float)
        if q.shape != (4,):
            raise ValueError(
                f"quaternion must have shape (4,), got {q.shape}"
            )
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            raise ValueError("quaternion must be non-zero")
        self.quaternion = q / norm

        omega = np.asarray(angular_velocity, dtype=float)
        if omega.shape != (3,):
            raise ValueError(
                f"angular_velocity must have shape (3,), got {omega.shape}"
            )
        self.angular_velocity = omega
        self.timestamp = float(timestamp)

    def __repr__(self) -> str:
        return (
            f"OrientationState("
            f"q={self.quaternion.tolist()}, "
            f"ω={self.angular_velocity.tolist()}, "
            f"t={self.timestamp})"
        )


def integrate_orientation(state: OrientationState, dt: float) -> OrientationState:
    """Integrate orientation forward by *dt* seconds using current angular velocity.

    Uses the exponential map: ``q(t+dt) = q(t) * exp(ω · dt / 2)``,
    which is the standard first-order quaternion integration.

    Args:
        state: Current orientation state.
        dt: Time step in seconds.  Must be positive.

    Returns:
        New :class:`OrientationState` with updated quaternion and timestamp.
        Angular velocity is carried forward unchanged.

    Raises:
        ValueError: If *dt* is not positive.

    Example::

        from spinstep.control import OrientationState, integrate_orientation

        state = OrientationState([0, 0, 0, 1], [0, 0, 1.0])
        new_state = integrate_orientation(state, dt=0.01)
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    omega = state.angular_velocity
    angle = np.linalg.norm(omega)

    if angle < 1e-10:
        # No rotation — return state with updated timestamp
        return OrientationState(
            quaternion=state.quaternion.copy(),
            angular_velocity=state.angular_velocity.copy(),
            timestamp=state.timestamp + dt,
        )

    # Compute the incremental rotation quaternion: exp(ω·dt/2)
    half_angle = angle * dt / 2.0
    axis = omega / angle
    delta_q = np.array([
        *(axis * np.sin(half_angle)),
        np.cos(half_angle),
    ])

    new_q = quaternion_multiply(state.quaternion, delta_q)
    new_q = quaternion_normalize(new_q)

    return OrientationState(
        quaternion=new_q,
        angular_velocity=state.angular_velocity.copy(),
        timestamp=state.timestamp + dt,
    )


def compute_orientation_error(
    current_q: ArrayLike, target_q: ArrayLike
) -> np.ndarray:
    """Compute the orientation error as an axis-angle vector from current to target.

    The error is expressed in the body frame of *current_q*.  Its direction is the
    rotation axis and its magnitude is the rotation angle in radians.

    Args:
        current_q: Current orientation quaternion ``[x, y, z, w]``.
        target_q: Target orientation quaternion ``[x, y, z, w]``.

    Returns:
        Error rotation vector ``(3,)`` in radians.  Zero vector when
        the orientations are identical.

    Example::

        from spinstep.control import compute_orientation_error

        error = compute_orientation_error([0, 0, 0, 1], [0, 0, 0.383, 0.924])
        print(error)  # approximately [0, 0, 0.785]
    """
    r_current = R.from_quat(current_q)
    r_target = R.from_quat(target_q)
    # Error rotation in the body frame of current
    r_error = r_current.inv() * r_target
    return r_error.as_rotvec()
