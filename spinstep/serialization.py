# serialization.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""JSON-compatible serialization for SpinStep spatial types.

Provides ``to_dict()`` and ``from_dict()`` functions for
:class:`OrientationState`, :class:`SpatialNode`, :class:`SceneGraph`,
and :class:`OrientationTrajectory`.

All serialized representations use plain Python types (lists, dicts,
floats) and are compatible with :func:`json.dumps` / :func:`json.loads`.
"""

from __future__ import annotations

__all__ = [
    "state_to_dict",
    "state_from_dict",
    "node_to_dict",
    "node_from_dict",
    "graph_to_dict",
    "graph_from_dict",
    "trajectory_to_dict",
    "trajectory_from_dict",
]

from typing import Any, Dict, List

from .control.state import OrientationState
from .control.trajectory import OrientationTrajectory
from .traversal.scene_graph import SceneGraph
from .traversal.spatial_node import SpatialNode


# ------------------------------------------------------------------
# OrientationState
# ------------------------------------------------------------------


def state_to_dict(state: OrientationState) -> Dict[str, Any]:
    """Serialize an :class:`OrientationState` to a JSON-compatible dict.

    Args:
        state: The state to serialize.

    Returns:
        Dictionary with ``quaternion``, ``distance``, ``angular_velocity``,
        ``radial_velocity``, ``timestamp`` keys.
    """
    return {
        "quaternion": state.quaternion.tolist(),
        "distance": state.distance,
        "angular_velocity": state.angular_velocity.tolist(),
        "radial_velocity": state.radial_velocity,
        "timestamp": state.timestamp,
    }


def state_from_dict(data: Dict[str, Any]) -> OrientationState:
    """Deserialize an :class:`OrientationState` from a dict.

    Args:
        data: Dictionary produced by :func:`state_to_dict`.

    Returns:
        Reconstructed :class:`OrientationState`.
    """
    return OrientationState(
        quaternion=data["quaternion"],
        distance=data.get("distance", 0.0),
        angular_velocity=data.get("angular_velocity", [0.0, 0.0, 0.0]),
        radial_velocity=data.get("radial_velocity", 0.0),
        timestamp=data.get("timestamp", 0.0),
    )


# ------------------------------------------------------------------
# SpatialNode
# ------------------------------------------------------------------


def node_to_dict(node: SpatialNode) -> Dict[str, Any]:
    """Serialize a :class:`SpatialNode` to a JSON-compatible dict.

    Args:
        node: The node to serialize.

    Returns:
        Dictionary with ``name``, ``orientation``, ``distance``,
        ``angular_velocity``, ``radial_velocity``, ``timestamp`` keys.
    """
    return {
        "name": node.name,
        "orientation": node.orientation.tolist(),
        "distance": node.distance,
        "angular_velocity": node.angular_velocity.tolist(),
        "radial_velocity": node.radial_velocity,
        "timestamp": node.timestamp,
    }


def node_from_dict(data: Dict[str, Any]) -> SpatialNode:
    """Deserialize a :class:`SpatialNode` from a dict.

    Args:
        data: Dictionary produced by :func:`node_to_dict`.

    Returns:
        Reconstructed :class:`SpatialNode`.
    """
    return SpatialNode(
        name=data["name"],
        orientation=data["orientation"],
        distance=data.get("distance", 0.0),
        angular_velocity=data.get("angular_velocity", [0.0, 0.0, 0.0]),
        radial_velocity=data.get("radial_velocity", 0.0),
        timestamp=data.get("timestamp", 0.0),
    )


# ------------------------------------------------------------------
# SceneGraph
# ------------------------------------------------------------------


def graph_to_dict(graph: SceneGraph) -> Dict[str, Any]:
    """Serialize a :class:`SceneGraph` to a JSON-compatible dict.

    Args:
        graph: The graph to serialize.

    Returns:
        Dictionary with ``nodes`` (list of node dicts) and ``edges``
        (list of ``[from, to]`` pairs) keys.
    """
    return {
        "nodes": [node_to_dict(n) for n in graph.nodes()],
        "edges": [list(e) for e in graph.edges()],
    }


def graph_from_dict(data: Dict[str, Any]) -> SceneGraph:
    """Deserialize a :class:`SceneGraph` from a dict.

    Args:
        data: Dictionary produced by :func:`graph_to_dict`.

    Returns:
        Reconstructed :class:`SceneGraph`.
    """
    graph = SceneGraph()
    for nd in data["nodes"]:
        graph.add_node(node_from_dict(nd))
    for edge in data["edges"]:
        from_name, to_name = edge
        graph.add_edge(from_name, to_name, bidirectional=False)
    return graph


# ------------------------------------------------------------------
# OrientationTrajectory
# ------------------------------------------------------------------


def trajectory_to_dict(traj: OrientationTrajectory) -> Dict[str, Any]:
    """Serialize an :class:`OrientationTrajectory` to a JSON-compatible dict.

    Args:
        traj: The trajectory to serialize.

    Returns:
        Dictionary with ``waypoints`` key containing a list of
        ``{"quaternion": [...], "distance": ..., "time": ...}`` dicts.
    """
    waypoints: List[Dict[str, Any]] = []
    for i in range(len(traj)):
        waypoints.append({
            "quaternion": traj.quaternions[i].tolist(),
            "distance": float(traj.distances[i]),
            "time": float(traj.times[i]),
        })
    return {"waypoints": waypoints}


def trajectory_from_dict(data: Dict[str, Any]) -> OrientationTrajectory:
    """Deserialize an :class:`OrientationTrajectory` from a dict.

    Args:
        data: Dictionary produced by :func:`trajectory_to_dict`.

    Returns:
        Reconstructed :class:`OrientationTrajectory`.
    """
    wps = [
        (wp["quaternion"], wp.get("distance", 0.0), wp["time"])
        for wp in data["waypoints"]
    ]
    return OrientationTrajectory(wps)
