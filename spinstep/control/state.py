# control/state.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Observer-centered spherical state model.

In the SpinStep control model the observer sits at the origin.
Every guided vehicle (node) is located by:

- **orientation** — a unit quaternion giving the direction from the observer
- **distance** — the radial distance (layer) from the observer

Velocities follow the same decomposition: angular velocity (rad/s) for
the tangential component and radial velocity (units/s) for the range
component.
"""

from __future__ import annotations

__all__ = [
    "OrientationState",
    "ControlCommand",
    "integrate_state",
    "compute_orientation_error",
]

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from ..math.core import quaternion_multiply, quaternion_normalize


@dataclass
class OrientationState:
    """Observer-centered state: direction, distance, velocities, timestamp.

    The state describes a guided vehicle (node) in the observer's spherical
    frame.  The quaternion gives the direction from the observer; the
    distance gives the radial layer.

    All quaternions use ``[x, y, z, w]`` convention.

    Args:
        quaternion: Unit quaternion ``[x, y, z, w]`` — direction from observer.
        distance: Radial distance from observer.  Defaults to ``0.0``.
        angular_velocity: Angular velocity ``[ωx, ωy, ωz]`` in rad/s.
        radial_velocity: Radial velocity (units/s).  Positive = moving away
            from observer.
        timestamp: Time in seconds.

    Attributes:
        quaternion: Normalised quaternion ``(4,)``.
        distance: Radial distance (≥ 0).
        angular_velocity: Angular velocity ``(3,)``.
        radial_velocity: Radial velocity scalar.
        timestamp: Timestamp in seconds.

    Example::

        from spinstep.control import OrientationState

        # A vehicle at distance 5.0 looking along +Z
        state = OrientationState([0, 0, 0, 1], distance=5.0)
    """

    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )
    distance: float = 0.0
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radial_velocity: float = 0.0
    timestamp: float = 0.0

    def __init__(
        self,
        quaternion: ArrayLike = (0.0, 0.0, 0.0, 1.0),
        distance: float = 0.0,
        angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
        radial_velocity: float = 0.0,
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

        if distance < 0:
            raise ValueError(f"distance must be non-negative, got {distance}")
        self.distance = float(distance)

        omega = np.asarray(angular_velocity, dtype=float)
        if omega.shape != (3,):
            raise ValueError(
                f"angular_velocity must have shape (3,), got {omega.shape}"
            )
        self.angular_velocity = omega
        self.radial_velocity = float(radial_velocity)
        self.timestamp = float(timestamp)

    def __repr__(self) -> str:
        return (
            f"OrientationState("
            f"q={self.quaternion.tolist()}, "
            f"d={self.distance}, "
            f"ω={self.angular_velocity.tolist()}, "
            f"ṙ={self.radial_velocity}, "
            f"t={self.timestamp})"
        )


@dataclass
class ControlCommand:
    """Command output from a controller: angular + radial velocity.

    Separates the tangential (angular) and radial components of the
    velocity command so they can be applied independently to actuators.

    Args:
        angular_velocity: Desired angular velocity ``[ωx, ωy, ωz]`` in
            rad/s.
        radial_velocity: Desired radial velocity in units/s.  Positive
            means moving away from the observer.

    Example::

        from spinstep.control.state import ControlCommand

        cmd = ControlCommand(angular_velocity=[0, 0, 1.0], radial_velocity=0.5)
    """

    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radial_velocity: float = 0.0

    def __init__(
        self,
        angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
        radial_velocity: float = 0.0,
    ) -> None:
        omega = np.asarray(angular_velocity, dtype=float)
        if omega.shape != (3,):
            raise ValueError(
                f"angular_velocity must have shape (3,), got {omega.shape}"
            )
        self.angular_velocity = omega
        self.radial_velocity = float(radial_velocity)

    def __repr__(self) -> str:
        return (
            f"ControlCommand("
            f"ω={self.angular_velocity.tolist()}, "
            f"ṙ={self.radial_velocity})"
        )


def integrate_state(state: OrientationState, dt: float) -> OrientationState:
    """Integrate the full spherical state forward by *dt* seconds.

    Orientation is integrated via the exponential map:
    ``q(t+dt) = q(t) * exp(ω · dt / 2)``.
    Distance is integrated linearly:
    ``d(t+dt) = max(0, d(t) + ṙ · dt)``.

    Args:
        state: Current state.
        dt: Time step in seconds.  Must be positive.

    Returns:
        New :class:`OrientationState` with updated quaternion, distance,
        and timestamp.  Velocities are carried forward unchanged.

    Raises:
        ValueError: If *dt* is not positive.

    Example::

        from spinstep.control import OrientationState, integrate_state

        state = OrientationState([0, 0, 0, 1], distance=5.0,
                                 angular_velocity=[0, 0, 1.0],
                                 radial_velocity=0.5)
        new = integrate_state(state, dt=0.01)
        # new.distance ≈ 5.005
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # --- angular integration ---
    omega = state.angular_velocity
    angle = np.linalg.norm(omega)

    if angle < 1e-10:
        new_q = state.quaternion.copy()
    else:
        half_angle = angle * dt / 2.0
        axis = omega / angle
        delta_q = np.array([
            *(axis * np.sin(half_angle)),
            np.cos(half_angle),
        ])
        new_q = quaternion_multiply(state.quaternion, delta_q)
        new_q = quaternion_normalize(new_q)

    # --- radial integration ---
    new_distance = max(0.0, state.distance + state.radial_velocity * dt)

    return OrientationState(
        quaternion=new_q,
        distance=new_distance,
        angular_velocity=state.angular_velocity.copy(),
        radial_velocity=state.radial_velocity,
        timestamp=state.timestamp + dt,
    )


def compute_orientation_error(
    current_q: ArrayLike,
    target_q: ArrayLike,
    current_distance: float = 0.0,
    target_distance: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute the full spherical error: angular + radial.

    The angular error is expressed in the body frame of *current_q* as a
    rotation vector (axis × angle, in radians).

    The radial error is ``target_distance − current_distance`` (positive
    means the target is farther from the observer).

    Args:
        current_q: Current orientation quaternion ``[x, y, z, w]``.
        target_q: Target orientation quaternion ``[x, y, z, w]``.
        current_distance: Current radial distance.
        target_distance: Target radial distance.

    Returns:
        A tuple ``(angular_error, radial_error)`` where *angular_error*
        is a rotation vector ``(3,)`` and *radial_error* is a float.

    Example::

        from spinstep.control import compute_orientation_error

        ang_err, rad_err = compute_orientation_error(
            [0, 0, 0, 1], [0, 0, 0.383, 0.924],
            current_distance=3.0, target_distance=5.0,
        )
    """
    r_current = R.from_quat(current_q)
    r_target = R.from_quat(target_q)
    r_error = r_current.inv() * r_target
    angular_error: np.ndarray = r_error.as_rotvec()
    radial_error = float(target_distance - current_distance)
    return angular_error, radial_error
