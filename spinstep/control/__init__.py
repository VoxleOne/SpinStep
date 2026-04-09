# control/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Orientation control: state, controllers, and trajectory tracking.

Sub-modules:

- :mod:`~.state` — :class:`OrientationState`, integration, error computation
- :mod:`~.controllers` — proportional and PID orientation controllers
- :mod:`~.trajectory` — waypoint trajectories and trajectory tracking
"""

__all__ = [
    # state
    "OrientationState",
    "integrate_orientation",
    "compute_orientation_error",
    # controllers
    "OrientationController",
    "ProportionalOrientationController",
    "PIDOrientationController",
    # trajectory
    "OrientationTrajectory",
    "TrajectoryInterpolator",
    "TrajectoryController",
]

from .state import (
    OrientationState,
    compute_orientation_error,
    integrate_orientation,
)
from .controllers import (
    OrientationController,
    PIDOrientationController,
    ProportionalOrientationController,
)
from .trajectory import (
    OrientationTrajectory,
    TrajectoryController,
    TrajectoryInterpolator,
)
