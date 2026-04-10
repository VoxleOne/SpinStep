# test_serialization.py — SpinStep Test Suite — MIT License
# Phase 4 tests: Serialization round-trips

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.control.state import OrientationState
from spinstep.control.trajectory import OrientationTrajectory
from spinstep.traversal.scene_graph import SceneGraph
from spinstep.traversal.spatial_node import SpatialNode
from spinstep.serialization import (
    state_to_dict,
    state_from_dict,
    node_to_dict,
    node_from_dict,
    graph_to_dict,
    graph_from_dict,
    trajectory_to_dict,
    trajectory_from_dict,
)


class TestStateSerialisation:
    def test_roundtrip(self):
        state = OrientationState(
            [0, 0, 0.383, 0.924],
            distance=5.0,
            angular_velocity=[1, 2, 3],
            radial_velocity=0.5,
            timestamp=2.0,
        )
        d = state_to_dict(state)
        restored = state_from_dict(d)
        assert np.allclose(restored.quaternion, state.quaternion, atol=1e-3)
        assert restored.distance == pytest.approx(state.distance)
        assert np.allclose(restored.angular_velocity, state.angular_velocity)
        assert restored.radial_velocity == pytest.approx(state.radial_velocity)
        assert restored.timestamp == pytest.approx(state.timestamp)

    def test_json_compatible(self):
        state = OrientationState([0, 0, 0, 1], distance=3.0)
        d = state_to_dict(state)
        json_str = json.dumps(d)
        restored = state_from_dict(json.loads(json_str))
        assert np.allclose(restored.quaternion, state.quaternion, atol=1e-6)

    def test_defaults(self):
        d = {"quaternion": [0, 0, 0, 1]}
        state = state_from_dict(d)
        assert state.distance == 0.0
        assert state.radial_velocity == 0.0


class TestNodeSerialisation:
    def test_roundtrip(self):
        node = SpatialNode(
            "robot",
            [0, 0, 0.383, 0.924],
            distance=5.0,
            angular_velocity=[1, 0, 0],
            radial_velocity=0.3,
            timestamp=1.5,
        )
        d = node_to_dict(node)
        restored = node_from_dict(d)
        assert restored.name == node.name
        assert np.allclose(restored.orientation, node.orientation, atol=1e-3)
        assert restored.distance == pytest.approx(node.distance)
        assert np.allclose(restored.angular_velocity, node.angular_velocity)
        assert restored.radial_velocity == pytest.approx(node.radial_velocity)
        assert restored.timestamp == pytest.approx(node.timestamp)

    def test_json_compatible(self):
        node = SpatialNode("test", [0, 0, 0, 1], distance=2.0)
        d = node_to_dict(node)
        json_str = json.dumps(d)
        restored = node_from_dict(json.loads(json_str))
        assert restored.name == "test"


class TestGraphSerialisation:
    def test_roundtrip(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1], distance=5.0))
        q = R.from_euler("z", 45, degrees=True).as_quat()
        g.add_node(SpatialNode("b", q, distance=7.0))
        g.add_edge("a", "b")

        d = graph_to_dict(g)
        restored = graph_from_dict(d)

        assert len(restored) == 2
        assert "a" in restored
        assert "b" in restored
        # Check edge preserved (as directed)
        edges = restored.edges()
        assert ("a", "b") in edges
        assert ("b", "a") in edges  # bidirectional stored as 2 directed

    def test_json_compatible(self):
        g = SceneGraph()
        g.add_node(SpatialNode("x", [0, 0, 0, 1]))
        d = graph_to_dict(g)
        json_str = json.dumps(d)
        restored = graph_from_dict(json.loads(json_str))
        assert "x" in restored

    def test_empty_graph(self):
        g = SceneGraph()
        d = graph_to_dict(g)
        restored = graph_from_dict(d)
        assert len(restored) == 0


class TestTrajectorySerialisation:
    def test_roundtrip(self):
        q_end = R.from_euler("z", 45, degrees=True).as_quat()
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            (q_end, 10.0, 1.0),
        ])
        d = trajectory_to_dict(traj)
        restored = trajectory_from_dict(d)
        assert len(restored) == 2
        assert np.allclose(restored.quaternions[0], traj.quaternions[0], atol=1e-3)
        assert np.allclose(restored.distances, traj.distances, atol=1e-3)
        assert np.allclose(restored.times, traj.times, atol=1e-3)

    def test_json_compatible(self):
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 0.0, 0.0),
            ([0, 0, 0.383, 0.924], 5.0, 1.0),
        ])
        d = trajectory_to_dict(traj)
        json_str = json.dumps(d)
        restored = trajectory_from_dict(json.loads(json_str))
        assert len(restored) == 2
