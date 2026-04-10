# control/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Observer-centered orientation control: state, controllers, trajectories,
frames, agents, and events.

The SpinStep control model places the observer at the origin.  Every
guided vehicle (node) is located by a quaternion (direction from observer)
and a radial distance (layer).  Controllers produce commands with both
angular and radial velocity components.

Sub-modules:

- :mod:`~.state` — :class:`OrientationState`, :class:`ControlCommand`,
  integration, error computation, relative state
- :mod:`~.controllers` — proportional and PID orientation controllers
- :mod:`~.trajectory` — waypoint trajectories and trajectory tracking
- :mod:`~.frames` — reference frames and coordinate transforms
- :mod:`~.agent` — controlled spatial entities with own reference frames
- :mod:`~.agent_manager` — multi-agent coordination
- :mod:`~.events` — callback-based event system
"""

__all__ = [
    # state
    "OrientationState",
    "ControlCommand",
    "integrate_state",
    "compute_orientation_error",
    "compute_relative_state",
    # controllers
    "OrientationController",
    "ProportionalOrientationController",
    "PIDOrientationController",
    # trajectory
    "OrientationTrajectory",
    "TrajectoryInterpolator",
    "TrajectoryController",
    # frames
    "ReferenceFrame",
    "rebase_state",
    # agents
    "Agent",
    "AgentManager",
    # events
    "EventEmitter",
]

from .state import (
    ControlCommand,
    OrientationState,
    compute_orientation_error,
    compute_relative_state,
    integrate_state,
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
from .frames import (
    ReferenceFrame,
    rebase_state,
)
from .agent import Agent
from .agent_manager import AgentManager
from .events import EventEmitter
