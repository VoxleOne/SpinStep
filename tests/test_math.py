# test_math.py — SpinStep Test Suite — MIT License
# Tests for spinstep.math subpackage

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.math.core import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
)
from spinstep.math.interpolation import slerp, squad
from spinstep.math.geometry import quaternion_distance
from spinstep.math.conversions import (
    quaternion_from_rotvec,
    quaternion_to_rotvec,
)
from spinstep.math.analysis import angular_velocity_from_quaternions
from spinstep.math.constraints import clamp_rotation_angle


# ===== core =====


class TestQuaternionNormalize:
    def test_unit_quaternion(self):
        q = quaternion_normalize([0.5, 0.5, 0.5, 0.5])
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-10)

    def test_zero_quaternion(self):
        q = quaternion_normalize([0, 0, 0, 0])
        assert np.allclose(q, [0, 0, 0, 1])

    def test_non_unit(self):
        q = quaternion_normalize([0, 0, 0, 2])
        assert np.allclose(q, [0, 0, 0, 1])


class TestQuaternionInverse:
    def test_unit_quaternion_inverse(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        inv = quaternion_inverse(q)
        product = quaternion_multiply(q, inv)
        # Should be close to identity [0, 0, 0, 1]
        assert np.allclose(product, [0, 0, 0, 1], atol=1e-10) or np.allclose(
            product, [0, 0, 0, -1], atol=1e-10
        )

    def test_zero_quaternion(self):
        inv = quaternion_inverse([0, 0, 0, 0])
        assert np.allclose(inv, [0, 0, 0, 1])


# ===== interpolation =====


class TestSlerp:
    def test_endpoints(self):
        q0 = np.array([0, 0, 0, 1.0])
        q1 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        assert np.allclose(slerp(q0, q1, 0.0), q0, atol=1e-6)
        assert np.allclose(slerp(q0, q1, 1.0), q1 / np.linalg.norm(q1), atol=1e-6)

    def test_midpoint_unit(self):
        q0 = np.array([0, 0, 0, 1.0])
        q1 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        mid = slerp(q0, q1, 0.5)
        assert np.linalg.norm(mid) == pytest.approx(1.0, abs=1e-10)

    def test_same_quaternion(self):
        q = np.array([0, 0, 0, 1.0])
        result = slerp(q, q, 0.5)
        assert np.allclose(result, q, atol=1e-6)

    def test_shortest_path(self):
        """SLERP takes the shortest arc even when quaternions are antipodal representatives."""
        q0 = np.array([0, 0, 0, 1.0])
        q1 = np.array([0, 0, 0, -1.0])  # same rotation, opposite sign
        mid = slerp(q0, q1, 0.5)
        assert np.linalg.norm(mid) == pytest.approx(1.0, abs=1e-6)

    def test_interpolation_angle(self):
        """Midpoint of 0° and 90° should be ~45°."""
        q0 = np.array([0, 0, 0, 1.0])
        q1 = R.from_euler("z", 90, degrees=True).as_quat()
        mid = slerp(q0, q1, 0.5)
        angle = R.from_quat(mid).magnitude()
        assert angle == pytest.approx(np.deg2rad(45), abs=0.01)


class TestSquad:
    def test_returns_unit_quaternion(self):
        q0 = R.from_euler("z", 0, degrees=True).as_quat()
        q1 = R.from_euler("z", 30, degrees=True).as_quat()
        q2 = R.from_euler("z", 60, degrees=True).as_quat()
        q3 = R.from_euler("z", 90, degrees=True).as_quat()
        result = squad(q0, q1, q2, q3, 0.5)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_endpoints(self):
        q0 = R.from_euler("z", 0, degrees=True).as_quat()
        q1 = R.from_euler("z", 30, degrees=True).as_quat()
        q2 = R.from_euler("z", 60, degrees=True).as_quat()
        q3 = R.from_euler("z", 90, degrees=True).as_quat()
        start = squad(q0, q1, q2, q3, 0.0)
        end = squad(q0, q1, q2, q3, 1.0)
        # At t=0 should be close to q1, at t=1 close to q2
        angle_start = quaternion_distance(start, q1)
        angle_end = quaternion_distance(end, q2)
        assert angle_start < 0.1
        assert angle_end < 0.1


# ===== analysis =====


class TestAngularVelocityFromQuaternions:
    def test_no_rotation(self):
        q = [0, 0, 0, 1]
        omega = angular_velocity_from_quaternions(q, q, dt=0.1)
        assert np.allclose(omega, [0, 0, 0], atol=1e-6)

    def test_known_rotation(self):
        """90° about Z in 1 second → ω ≈ [0, 0, π/2]."""
        q1 = [0, 0, 0, 1]
        q2 = R.from_euler("z", 90, degrees=True).as_quat()
        omega = angular_velocity_from_quaternions(q1, q2, dt=1.0)
        assert np.allclose(omega, [0, 0, np.pi / 2], atol=1e-6)

    def test_invalid_dt(self):
        with pytest.raises(ValueError):
            angular_velocity_from_quaternions([0, 0, 0, 1], [0, 0, 0, 1], dt=0)


# ===== constraints =====


class TestClampRotationAngle:
    def test_within_limit(self):
        """Small rotation should not be changed."""
        q = R.from_euler("z", 10, degrees=True).as_quat()
        clamped = clamp_rotation_angle(q, max_angle=np.pi)
        assert np.allclose(clamped, q, atol=1e-6)

    def test_clamped(self):
        """Large rotation should be clamped to max_angle."""
        q = R.from_euler("z", 90, degrees=True).as_quat()
        max_angle = np.deg2rad(45)
        clamped = clamp_rotation_angle(q, max_angle=max_angle)
        angle = R.from_quat(clamped).magnitude()
        assert angle == pytest.approx(max_angle, abs=1e-6)

    def test_preserves_axis(self):
        """Clamping preserves the rotation axis."""
        q = R.from_euler("z", 90, degrees=True).as_quat()
        max_angle = np.deg2rad(45)
        clamped = clamp_rotation_angle(q, max_angle=max_angle)
        original_axis = R.from_quat(q).as_rotvec()
        clamped_axis = R.from_quat(clamped).as_rotvec()
        # Axes should be parallel
        original_dir = original_axis / np.linalg.norm(original_axis)
        clamped_dir = clamped_axis / np.linalg.norm(clamped_axis)
        assert np.allclose(original_dir, clamped_dir, atol=1e-6)

    def test_negative_max_angle_raises(self):
        with pytest.raises(ValueError):
            clamp_rotation_angle([0, 0, 0, 1], max_angle=-1.0)


# ===== conversions =====


class TestQuaternionFromRotvec:
    def test_identity(self):
        q = quaternion_from_rotvec([0, 0, 0])
        assert np.allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_90_about_z(self):
        rotvec = [0, 0, np.pi / 2]
        q = quaternion_from_rotvec(rotvec)
        expected = R.from_rotvec(rotvec).as_quat()
        assert np.allclose(q, expected, atol=1e-6)


class TestQuaternionToRotvec:
    def test_roundtrip(self):
        rotvec = [0.3, -0.5, 0.7]
        q = quaternion_from_rotvec(rotvec)
        result = quaternion_to_rotvec(q)
        assert np.allclose(result, rotvec, atol=1e-6)
