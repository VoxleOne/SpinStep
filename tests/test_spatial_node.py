# test_spatial_node.py — SpinStep Test Suite — MIT License
# Phase 1 tests: SpatialNode, as_state/as_node, compute_relative_state

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.traversal.node import Node
from spinstep.traversal.spatial_node import SpatialNode
from spinstep.control.state import OrientationState, compute_relative_state
from spinstep.math.analysis import NodeProtocol, SpatialNodeProtocol


# ===== SpatialNode construction =====


class TestSpatialNodeConstruction:
    def test_defaults(self):
        sn = SpatialNode("test", [0, 0, 0, 1])
        assert sn.name == "test"
        assert np.allclose(sn.orientation, [0, 0, 0, 1])
        assert sn.distance == 0.0
        assert np.allclose(sn.angular_velocity, [0, 0, 0])
        assert sn.radial_velocity == 0.0
        assert sn.timestamp == 0.0

    def test_all_fields(self):
        sn = SpatialNode(
            "robot",
            [0, 0, 0.383, 0.924],
            distance=5.0,
            angular_velocity=[0, 0, 1.0],
            radial_velocity=0.5,
            timestamp=2.0,
        )
        assert sn.distance == 5.0
        assert np.allclose(sn.angular_velocity, [0, 0, 1.0])
        assert sn.radial_velocity == 0.5
        assert sn.timestamp == 2.0

    def test_isinstance_node(self):
        sn = SpatialNode("test", [0, 0, 0, 1])
        assert isinstance(sn, Node)

    def test_normalises_quaternion(self):
        sn = SpatialNode("test", [0, 0, 0, 2])
        assert np.allclose(sn.orientation, [0, 0, 0, 1])

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SpatialNode("test", [0, 0, 0, 1], distance=-1.0)

    def test_invalid_angular_velocity_shape(self):
        with pytest.raises(ValueError, match="shape"):
            SpatialNode("test", [0, 0, 0, 1], angular_velocity=[1, 2])

    def test_children_supported(self):
        child = SpatialNode("child", [1, 0, 0, 0])
        parent = SpatialNode("parent", [0, 0, 0, 1], [child])
        assert len(parent.children) == 1
        assert parent.children[0].name == "child"

    def test_add_child_works(self):
        parent = SpatialNode("p", [0, 0, 0, 1])
        child = SpatialNode("c", [1, 0, 0, 0])
        result = parent.add_child(child)
        assert result is child
        assert child in parent.children

    def test_repr_includes_distance(self):
        sn = SpatialNode("test", [0, 0, 0, 1], distance=5.0)
        r = repr(sn)
        assert "SpatialNode" in r
        assert "d=5.0" in r


# ===== Protocol satisfaction =====


class TestProtocols:
    def test_spatial_node_satisfies_node_protocol(self):
        sn = SpatialNode("test", [0, 0, 0, 1])
        assert isinstance(sn, NodeProtocol)

    def test_spatial_node_satisfies_spatial_protocol(self):
        sn = SpatialNode("test", [0, 0, 0, 1], distance=5.0)
        assert isinstance(sn, SpatialNodeProtocol)

    def test_plain_node_satisfies_node_protocol(self):
        node = Node("test", [0, 0, 0, 1])
        assert isinstance(node, NodeProtocol)

    def test_custom_class_satisfies_spatial_protocol(self):
        class MyNode:
            def __init__(self):
                self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
                self.distance = 5.0
                self.name = "custom"

        assert isinstance(MyNode(), SpatialNodeProtocol)


# ===== as_state / as_node =====


class TestAsState:
    def test_node_as_state_defaults(self):
        node = Node("test", [0, 0, 0, 1])
        state = node.as_state()
        assert isinstance(state, OrientationState)
        assert np.allclose(state.quaternion, [0, 0, 0, 1])
        assert state.distance == 0.0

    def test_spatial_node_as_state_full(self):
        sn = SpatialNode(
            "robot",
            [0, 0, 0.383, 0.924],
            distance=5.0,
            angular_velocity=[0, 0, 1.0],
            radial_velocity=0.5,
            timestamp=2.0,
        )
        state = sn.as_state()
        assert state.distance == 5.0
        assert np.allclose(state.angular_velocity, [0, 0, 1.0])
        assert state.radial_velocity == 0.5
        assert state.timestamp == 2.0

    def test_orientation_state_as_node(self):
        state = OrientationState(
            [0, 0, 0.383, 0.924],
            distance=5.0,
            angular_velocity=[0, 0, 1.0],
            radial_velocity=0.5,
            timestamp=2.0,
        )
        node = state.as_node("robot")
        assert isinstance(node, SpatialNode)
        assert node.name == "robot"
        assert node.distance == 5.0
        assert np.allclose(node.angular_velocity, [0, 0, 1.0])
        assert node.radial_velocity == 0.5
        assert node.timestamp == 2.0

    def test_roundtrip_spatial_node(self):
        """SpatialNode → OrientationState → SpatialNode preserves data."""
        sn = SpatialNode(
            "test", [0, 0, 0.383, 0.924],
            distance=7.0, angular_velocity=[1, 0, 0],
            radial_velocity=0.3, timestamp=1.5,
        )
        state = sn.as_state()
        restored = state.as_node(sn.name)
        assert restored.name == sn.name
        assert np.allclose(restored.orientation, sn.orientation, atol=1e-6)
        assert restored.distance == pytest.approx(sn.distance)
        assert np.allclose(restored.angular_velocity, sn.angular_velocity)
        assert restored.radial_velocity == pytest.approx(sn.radial_velocity)
        assert restored.timestamp == pytest.approx(sn.timestamp)


# ===== compute_relative_state =====


class TestComputeRelativeState:
    def test_identity(self):
        """Same node → relative quaternion is identity, distance 0."""
        obs = SpatialNode("obs", [0, 0, 0, 1], distance=5.0)
        rel = compute_relative_state(obs, obs)
        assert np.allclose(rel.quaternion, [0, 0, 0, 1], atol=1e-6)
        assert rel.distance == pytest.approx(0.0, abs=1e-6)

    def test_known_rotation(self):
        """90° rotation about Z axis."""
        q_90z = R.from_euler("z", 90, degrees=True).as_quat()
        obs = SpatialNode("obs", [0, 0, 0, 1])
        tgt = SpatialNode("tgt", q_90z)
        rel = compute_relative_state(obs, tgt)
        # Relative quaternion should match the 90° rotation
        angle = R.from_quat(rel.quaternion).magnitude()
        assert angle == pytest.approx(np.pi / 2, abs=0.01)

    def test_with_distance(self):
        """Nodes at different distances."""
        obs = SpatialNode("obs", [0, 0, 0, 1], distance=5.0)
        tgt = SpatialNode("tgt", [0, 0, 0, 1], distance=10.0)
        rel = compute_relative_state(obs, tgt)
        # Same orientation, different distance → rel distance = |10-5| = 5
        assert rel.distance == pytest.approx(5.0, abs=0.1)

    def test_different_orientation_and_distance(self):
        """Nodes at different orientations and distances."""
        q_180x = R.from_euler("x", 180, degrees=True).as_quat()
        obs = SpatialNode("obs", [0, 0, 0, 1], distance=5.0)
        tgt = SpatialNode("tgt", q_180x, distance=5.0)
        rel = compute_relative_state(obs, tgt)
        # Relative distance: both at d=5 but opposite forward directions → ~10
        assert rel.distance == pytest.approx(10.0, abs=0.1)

    def test_inverse_symmetry(self):
        """Relative rotation from A→B is inverse of B→A."""
        q = R.from_euler("z", 45, degrees=True).as_quat()
        a = SpatialNode("a", [0, 0, 0, 1], distance=3.0)
        b = SpatialNode("b", q, distance=7.0)
        rel_ab = compute_relative_state(a, b)
        rel_ba = compute_relative_state(b, a)
        # Distances should be the same
        assert rel_ab.distance == pytest.approx(rel_ba.distance, abs=0.1)
