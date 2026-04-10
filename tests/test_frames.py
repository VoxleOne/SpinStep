# test_frames.py — SpinStep Test Suite — MIT License
# Phase 2 tests: ReferenceFrame, rebase_state

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.control.frames import ReferenceFrame, rebase_state
from spinstep.control.state import OrientationState
from spinstep.traversal.spatial_node import SpatialNode


class TestReferenceFrameWorld:
    def test_world_is_identity(self):
        frame = ReferenceFrame.world()
        assert np.allclose(frame.origin_quaternion, [0, 0, 0, 1])
        assert frame.origin_distance == 0.0

    def test_world_to_local_identity(self):
        """World frame to_local should not change quaternion."""
        frame = ReferenceFrame.world()
        state = OrientationState([0, 0, 0, 1], distance=5.0)
        local = frame.to_local(state)
        assert np.allclose(local.quaternion, state.quaternion, atol=1e-6)

    def test_world_to_world_identity(self):
        """World frame to_world should not change quaternion."""
        frame = ReferenceFrame.world()
        state = OrientationState([0, 0, 0, 1], distance=5.0)
        world = frame.to_world(state)
        assert np.allclose(world.quaternion, state.quaternion, atol=1e-6)


class TestReferenceFrameFromNode:
    def test_extracts_correct_origin(self):
        q_90z = R.from_euler("z", 90, degrees=True).as_quat()
        node = SpatialNode("obs", q_90z, distance=5.0)
        frame = ReferenceFrame.from_node(node)
        assert np.allclose(frame.origin_quaternion, q_90z, atol=1e-6)
        assert frame.origin_distance == 5.0


class TestToLocalToWorld:
    def test_roundtrip(self):
        """to_local then to_world should approximately preserve quaternion."""
        q_45z = R.from_euler("z", 45, degrees=True).as_quat()
        frame = ReferenceFrame(q_45z, 3.0)
        state = OrientationState(
            [0, 0, 0.383, 0.924],
            distance=5.0,
            angular_velocity=[0, 0, 1.0],
            radial_velocity=0.5,
            timestamp=1.0,
        )
        local = frame.to_local(state)
        restored = frame.to_world(local)
        # Quaternion should roundtrip
        assert np.allclose(
            restored.quaternion, state.quaternion, atol=1e-3
        ) or np.allclose(restored.quaternion, -state.quaternion, atol=1e-3)
        # Velocities and timestamp preserved
        assert np.allclose(restored.angular_velocity, state.angular_velocity)
        assert restored.radial_velocity == pytest.approx(state.radial_velocity)
        assert restored.timestamp == pytest.approx(state.timestamp)

    def test_known_90z_rotation(self):
        """Frame at 90° about Z, state at identity → local should be -90° about Z."""
        q_90z = R.from_euler("z", 90, degrees=True).as_quat()
        frame = ReferenceFrame(q_90z, 0.0)
        state = OrientationState([0, 0, 0, 1])
        local = frame.to_local(state)
        angle = R.from_quat(local.quaternion).magnitude()
        assert angle == pytest.approx(np.pi / 2, abs=0.01)

    def test_preserves_velocities(self):
        q_45z = R.from_euler("z", 45, degrees=True).as_quat()
        frame = ReferenceFrame(q_45z)
        state = OrientationState(
            [0, 0, 0, 1],
            angular_velocity=[1, 2, 3],
            radial_velocity=0.5,
            timestamp=99.0,
        )
        local = frame.to_local(state)
        assert np.allclose(local.angular_velocity, [1, 2, 3])
        assert local.radial_velocity == 0.5
        assert local.timestamp == 99.0


class TestRebaseState:
    def test_same_frame_noop(self):
        """Rebase from frame A to frame A should preserve state."""
        q = R.from_euler("z", 30, degrees=True).as_quat()
        frame = ReferenceFrame(q, 5.0)
        state = OrientationState([0, 0, 0, 1], distance=3.0)
        rebased = rebase_state(state, frame, frame)
        assert np.allclose(
            rebased.quaternion, state.quaternion, atol=1e-3
        ) or np.allclose(rebased.quaternion, -state.quaternion, atol=1e-3)

    def test_world_to_frame_matches_to_local(self):
        """Rebase from world to frame should equal frame.to_local."""
        q = R.from_euler("z", 45, degrees=True).as_quat()
        world = ReferenceFrame.world()
        frame = ReferenceFrame(q, 5.0)
        state = OrientationState([0, 0, 0.383, 0.924], distance=3.0)
        rebased = rebase_state(state, world, frame)
        direct = frame.to_local(state)
        assert np.allclose(rebased.quaternion, direct.quaternion, atol=1e-6)

    def test_transitivity(self):
        """Rebase A→B→C should approximately equal rebase A→C."""
        q_a = R.from_euler("z", 10, degrees=True).as_quat()
        q_b = R.from_euler("z", 30, degrees=True).as_quat()
        q_c = R.from_euler("z", 60, degrees=True).as_quat()
        frame_a = ReferenceFrame(q_a, 1.0)
        frame_b = ReferenceFrame(q_b, 2.0)
        frame_c = ReferenceFrame(q_c, 3.0)
        state = OrientationState([0, 0, 0, 1])
        via_b = rebase_state(state, frame_a, frame_b)
        via_bc = rebase_state(via_b, frame_b, frame_c)
        direct = rebase_state(state, frame_a, frame_c)
        assert np.allclose(
            via_bc.quaternion, direct.quaternion, atol=1e-3
        ) or np.allclose(via_bc.quaternion, -direct.quaternion, atol=1e-3)


class TestReferenceFrameEquality:
    def test_equal(self):
        q = R.from_euler("z", 45, degrees=True).as_quat()
        assert ReferenceFrame(q, 5.0) == ReferenceFrame(q, 5.0)

    def test_not_equal(self):
        q1 = R.from_euler("z", 45, degrees=True).as_quat()
        q2 = R.from_euler("z", 90, degrees=True).as_quat()
        assert ReferenceFrame(q1, 5.0) != ReferenceFrame(q2, 5.0)

    def test_not_equal_to_non_frame(self):
        q = R.from_euler("z", 45, degrees=True).as_quat()
        assert ReferenceFrame(q, 5.0) != "not a frame"


class TestNumericalStability:
    def test_near_identity_quaternion(self):
        """Very small rotation should not cause issues."""
        q = R.from_euler("z", 0.001, degrees=True).as_quat()
        frame = ReferenceFrame(q)
        state = OrientationState([0, 0, 0, 1])
        local = frame.to_local(state)
        assert np.all(np.isfinite(local.quaternion))

    def test_near_antipodal(self):
        """Nearly opposite quaternions should be handled."""
        q = R.from_euler("z", 179.9, degrees=True).as_quat()
        frame = ReferenceFrame(q)
        state = OrientationState([0, 0, 0, 1])
        local = frame.to_local(state)
        assert np.all(np.isfinite(local.quaternion))
