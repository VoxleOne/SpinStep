# control/trajectory.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Orientation trajectories: waypoints, interpolation, and tracking.

Trajectories in SpinStep are sequences of ``(quaternion, distance, time)``
waypoints in the observer-centered spherical frame.  The quaternion gives
the direction from the observer and the distance gives the radial layer.
"""

from __future__ import annotations

__all__ = [
    "OrientationTrajectory",
    "TrajectoryInterpolator",
    "TrajectoryController",
]

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from ..math.interpolation import slerp
from .controllers import OrientationController
from .state import ControlCommand


class OrientationTrajectory:
    """A sequence of spherical waypoints with timestamps.

    Each waypoint is ``(quaternion, distance, time)`` or, for backward
    compatibility, ``(quaternion, time)`` (distance defaults to ``0.0``).

    Waypoints must be in ascending time order.

    Args:
        waypoints: Sequence of waypoint tuples.  Accepted forms:
            - ``(quaternion, distance, time)``
            - ``(quaternion, time)`` — distance defaults to ``0.0``

    Raises:
        ValueError: If fewer than two waypoints are provided, times are
            not strictly increasing, or quaternions are invalid.

    Attributes:
        quaternions: Array of shape ``(N, 4)`` — waypoint quaternions.
        distances: Array of shape ``(N,)`` — waypoint distances.
        times: Array of shape ``(N,)`` — waypoint times in seconds.

    Example::

        from spinstep.control import OrientationTrajectory

        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0.383, 0.924], 7.5, 1.0),
            ([0, 0, 0.707, 0.707], 10.0, 2.0),
        ])
    """

    quaternions: npt.NDArray[np.floating[Any]]
    distances: npt.NDArray[np.floating[Any]]
    times: npt.NDArray[np.floating[Any]]

    def __init__(
        self,
        waypoints: Sequence[Union[Tuple[ArrayLike, float, float], Tuple[ArrayLike, float]]],
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError(
                f"At least 2 waypoints are required, got {len(waypoints)}"
            )

        quats: List[npt.NDArray[np.floating[Any]]] = []
        dists: List[float] = []
        times: List[float] = []

        for wp in waypoints:
            if len(wp) == 3:
                q_raw, dist, t = wp
            elif len(wp) == 2:
                q_raw, t = wp
                dist = 0.0
            else:
                raise ValueError(
                    f"Each waypoint must be (quaternion, distance, time) "
                    f"or (quaternion, time), got tuple of length {len(wp)}"
                )

            arr = np.asarray(q_raw, dtype=float)
            if arr.shape != (4,):
                raise ValueError(
                    f"Each waypoint quaternion must have shape (4,), got {arr.shape}"
                )
            norm = np.linalg.norm(arr)
            if norm < 1e-8:
                raise ValueError("Waypoint quaternion must be non-zero")
            quats.append(arr / norm)
            dists.append(float(dist))
            times.append(float(t))

        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                raise ValueError(
                    f"Waypoint times must be strictly increasing: "
                    f"t[{i-1}]={times[i-1]}, t[{i}]={times[i]}"
                )

        self.quaternions = np.array(quats)
        self.distances = np.array(dists)
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
    """SLERP + linear interpolator for an :class:`OrientationTrajectory`.

    Orientation is interpolated via SLERP; distance is linearly
    interpolated between adjacent waypoints.

    Args:
        trajectory: The trajectory to interpolate.

    Example::

        from spinstep.control import OrientationTrajectory, TrajectoryInterpolator

        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0.383, 0.924], 10.0, 1.0),
        ])
        interp = TrajectoryInterpolator(traj)
        q, d = interp.evaluate(0.5)
        # q ≈ slerp midpoint, d ≈ 7.5
    """

    def __init__(self, trajectory: OrientationTrajectory) -> None:
        self.trajectory = trajectory

    def evaluate(self, t: float) -> Tuple[npt.NDArray[np.floating[Any]], float]:
        """Return the interpolated quaternion and distance at time *t*.

        Times before the first waypoint return the first pose; times
        after the last return the last pose.

        Args:
            t: Query time in seconds.

        Returns:
            A tuple ``(quaternion, distance)``.
        """
        traj = self.trajectory

        if t <= traj.times[0]:
            return traj.quaternions[0].copy(), float(traj.distances[0])
        if t >= traj.times[-1]:
            return traj.quaternions[-1].copy(), float(traj.distances[-1])

        # Find the segment
        idx = int(np.searchsorted(traj.times, t, side="right") - 1)
        idx = min(idx, len(traj.times) - 2)

        t0 = traj.times[idx]
        t1 = traj.times[idx + 1]
        alpha = (t - t0) / (t1 - t0)

        q = slerp(traj.quaternions[idx], traj.quaternions[idx + 1], alpha)
        d = float(traj.distances[idx] + alpha * (traj.distances[idx + 1] - traj.distances[idx]))
        return q, d

    @property
    def duration(self) -> float:
        """Total duration of the underlying trajectory."""
        return self.trajectory.duration


class TrajectoryController:
    """Controller that tracks a spherical trajectory over time.

    Wraps a base :class:`OrientationController` and a
    :class:`TrajectoryInterpolator`.  At each step the controller queries
    the interpolator for the desired pose and computes the
    :class:`~.state.ControlCommand` to drive the vehicle towards it.

    Args:
        controller: Base controller instance.
        trajectory: The trajectory to follow.

    Attributes:
        interpolator: Internal :class:`TrajectoryInterpolator`.
        controller: The wrapped base controller.
        is_complete: Whether the trajectory end time has been reached.

    Example::

        from spinstep.control import (
            OrientationTrajectory,
            ProportionalOrientationController,
            TrajectoryController,
        )

        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            ([0, 0, 0.383, 0.924], 10.0, 1.0),
        ])
        ctrl = ProportionalOrientationController(kp=2.0, kp_radial=1.0)
        traj_ctrl = TrajectoryController(ctrl, traj)
        cmd = traj_ctrl.update([0, 0, 0, 1], current_distance=5.0, t=0.5, dt=0.01)
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
        current_distance: float = 0.0,
    ) -> ControlCommand:
        """Compute the control command to track the trajectory at time *t*.

        Args:
            current_q: Current orientation ``[x, y, z, w]``.
            t: Current time in seconds.
            dt: Time step in seconds.
            current_distance: Current radial distance from observer.

        Returns:
            :class:`~.state.ControlCommand` with angular and radial components.
        """
        target_q, target_distance = self.interpolator.evaluate(t)
        self.is_complete = t >= self.interpolator.trajectory.end_time
        return self.controller.update(
            current_q, target_q, dt, current_distance, target_distance
        )

    def reset(self) -> None:
        """Reset the controller state."""
        self.controller.reset()
        self.is_complete = False
