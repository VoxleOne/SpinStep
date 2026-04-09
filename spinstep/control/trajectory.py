# control/trajectory.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Orientation trajectories: waypoints, interpolation, and tracking."""

from __future__ import annotations

__all__ = [
    "OrientationTrajectory",
    "TrajectoryInterpolator",
    "TrajectoryController",
]

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from ..math.interpolation import slerp
from .controllers import OrientationController


class OrientationTrajectory:
    """A sequence of quaternion waypoints with associated timestamps.

    Waypoints must be in ascending time order.

    Args:
        waypoints: Sequence of ``(quaternion, time)`` pairs where each
            quaternion is ``[x, y, z, w]`` and time is in seconds.

    Raises:
        ValueError: If fewer than two waypoints are provided or times
            are not strictly increasing.

    Attributes:
        quaternions: Array of shape ``(N, 4)`` — waypoint quaternions.
        times: Array of shape ``(N,)`` — waypoint times in seconds.

    Example::

        from spinstep.control import OrientationTrajectory

        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 0.0),
            ([0, 0, 0.383, 0.924], 1.0),
            ([0, 0, 0.707, 0.707], 2.0),
        ])
    """

    quaternions: np.ndarray
    times: np.ndarray

    def __init__(
        self,
        waypoints: Sequence[Tuple[ArrayLike, float]],
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError(
                f"At least 2 waypoints are required, got {len(waypoints)}"
            )

        quats: List[np.ndarray] = []
        times: List[float] = []
        for q, t in waypoints:
            arr = np.asarray(q, dtype=float)
            if arr.shape != (4,):
                raise ValueError(
                    f"Each waypoint quaternion must have shape (4,), got {arr.shape}"
                )
            norm = np.linalg.norm(arr)
            if norm < 1e-8:
                raise ValueError("Waypoint quaternion must be non-zero")
            quats.append(arr / norm)
            times.append(float(t))

        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                raise ValueError(
                    f"Waypoint times must be strictly increasing: "
                    f"t[{i-1}]={times[i-1]}, t[{i}]={times[i]}"
                )

        self.quaternions = np.array(quats)
        self.times = np.array(times)

    @property
    def duration(self) -> float:
        """Total duration of the trajectory in seconds."""
        return float(self.times[-1] - self.times[0])

    @property
    def start_time(self) -> float:
        """Start time of the trajectory."""
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        """End time of the trajectory."""
        return float(self.times[-1])

    def __len__(self) -> int:
        return len(self.times)

    def __repr__(self) -> str:
        return (
            f"OrientationTrajectory({len(self)} waypoints, "
            f"t=[{self.start_time}, {self.end_time}])"
        )


class TrajectoryInterpolator:
    """SLERP-based interpolator for an :class:`OrientationTrajectory`.

    Evaluates the orientation at any time within the trajectory's time span
    using spherical linear interpolation between adjacent waypoints.

    Args:
        trajectory: The trajectory to interpolate.

    Example::

        from spinstep.control import OrientationTrajectory, TrajectoryInterpolator

        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 0.0),
            ([0, 0, 0.383, 0.924], 1.0),
        ])
        interp = TrajectoryInterpolator(traj)
        q = interp.evaluate(0.5)
    """

    def __init__(self, trajectory: OrientationTrajectory) -> None:
        self.trajectory = trajectory

    def evaluate(self, t: float) -> np.ndarray:
        """Return the interpolated quaternion at time *t*.

        Times before the first waypoint return the first quaternion.
        Times after the last waypoint return the last quaternion.

        Args:
            t: Query time in seconds.

        Returns:
            Interpolated unit quaternion ``[x, y, z, w]``.
        """
        traj = self.trajectory

        if t <= traj.times[0]:
            return traj.quaternions[0].copy()
        if t >= traj.times[-1]:
            return traj.quaternions[-1].copy()

        # Find the segment
        idx = int(np.searchsorted(traj.times, t, side="right") - 1)
        idx = min(idx, len(traj.times) - 2)

        t0 = traj.times[idx]
        t1 = traj.times[idx + 1]
        alpha = (t - t0) / (t1 - t0)

        return slerp(traj.quaternions[idx], traj.quaternions[idx + 1], alpha)

    @property
    def duration(self) -> float:
        """Total duration of the underlying trajectory."""
        return self.trajectory.duration


class TrajectoryController:
    """Controller that tracks an orientation trajectory over time.

    Wraps a base :class:`OrientationController` and a
    :class:`TrajectoryInterpolator`.  At each time step the controller
    queries the interpolator for the desired orientation and computes the
    angular velocity command to drive the system towards it.

    Args:
        controller: An :class:`OrientationController` instance (e.g.
            :class:`ProportionalOrientationController` or
            :class:`PIDOrientationController`).
        trajectory: The trajectory to follow.

    Attributes:
        interpolator: The :class:`TrajectoryInterpolator` used internally.
        controller: The wrapped base controller.
        is_complete: Whether the trajectory end time has been reached.

    Example::

        from spinstep.control import (
            OrientationTrajectory,
            ProportionalOrientationController,
            TrajectoryController,
        )

        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 0.0),
            ([0, 0, 0.383, 0.924], 1.0),
        ])
        ctrl = ProportionalOrientationController(kp=2.0)
        traj_ctrl = TrajectoryController(ctrl, traj)
        cmd = traj_ctrl.update([0, 0, 0, 1], t=0.5, dt=0.01)
    """

    def __init__(
        self,
        controller: OrientationController,
        trajectory: OrientationTrajectory,
    ) -> None:
        self.controller = controller
        self.interpolator = TrajectoryInterpolator(trajectory)
        self.is_complete: bool = False

    def update(
        self,
        current_q: ArrayLike,
        t: float,
        dt: float,
    ) -> np.ndarray:
        """Compute angular velocity command to track the trajectory at time *t*.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            t: Current time in seconds.
            dt: Time step in seconds.

        Returns:
            Angular velocity command ``(3,)`` in rad/s.
        """
        target_q = self.interpolator.evaluate(t)
        self.is_complete = t >= self.interpolator.trajectory.end_time
        return self.controller.update(current_q, target_q, dt)

    def reset(self) -> None:
        """Reset the controller state."""
        self.controller.reset()
        self.is_complete = False
