# __init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""SpinStep: Quaternion-based orientation control and traversal.

SpinStep uses an observer-centered spherical model where every guided
vehicle (node) is located by a quaternion (direction from the observer)
and a radial distance (layer).

Control layer (primary API)::

    from spinstep import (
        OrientationState, ControlCommand,
        ProportionalOrientationController, PIDOrientationController,
        OrientationTrajectory, TrajectoryController,
        integrate_state, compute_orientation_error,
        slerp,
    )

Traversal layer (original tree-walking API)::

    from spinstep.traversal import (
        Node, QuaternionDepthIterator,
        DiscreteOrientationSet, DiscreteQuaternionIterator,
    )

Math layer::

    from spinstep.math import quaternion_multiply, quaternion_distance, slerp
"""

__version__ = "0.6.0a0"

# --- control layer (primary API) ---
from .control.state import (
    ControlCommand,
    OrientationState,
    compute_orientation_error,
    compute_relative_state,
    integrate_state,
)
from .control.controllers import (
    OrientationController,
    PIDOrientationController,
    ProportionalOrientationController,
)
from .control.trajectory import (
    OrientationTrajectory,
    TrajectoryController,
    TrajectoryInterpolator,
)
from .control.frames import (
    ReferenceFrame,
    rebase_state,
)

# --- key math utilities at top level ---
from .math.interpolation import slerp

# --- backward-compatible traversal re-exports ---
from .traversal.node import Node
from .traversal.spatial_node import SpatialNode
from .traversal.continuous import QuaternionDepthIterator
from .traversal.discrete import DiscreteOrientationSet
from .traversal.discrete_iterator import DiscreteQuaternionIterator
from .traversal.scene_graph import SceneGraph
from .traversal.graph_iterators import BreadthFirstIterator, GraphQuaternionIterator

__all__ = [
    # control
    "OrientationState",
    "ControlCommand",
    "integrate_state",
    "compute_orientation_error",
    "compute_relative_state",
    "ReferenceFrame",
    "rebase_state",
    "OrientationController",
    "ProportionalOrientationController",
    "PIDOrientationController",
    "OrientationTrajectory",
    "TrajectoryInterpolator",
    "TrajectoryController",
    # math
    "slerp",
    # traversal (backward compat)
    "Node",
    "SpatialNode",
    "QuaternionDepthIterator",
    "DiscreteOrientationSet",
    "DiscreteQuaternionIterator",
    "SceneGraph",
    "BreadthFirstIterator",
    "GraphQuaternionIterator",
]
