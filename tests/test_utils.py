# test_utils.py — SpinStep Test Suite — MIT License
# Author: SpinStep Contributors — Created: 2026-03-24
# See LICENSE.txt for full terms.

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.traversal.node import Node
from spinstep.utils.array_backend import get_array_module
from spinstep.utils.quaternion_math import batch_quaternion_angle
from spinstep.utils.quaternion_utils import (
    is_within_angle_threshold,
    quaternion_conjugate,
    quaternion_distance,
    quaternion_from_euler,
    quaternion_multiply,
    rotate_quaternion,
    rotation_matrix_to_quaternion,
    get_relative_spin,
    forward_vector_from_quaternion,
    direction_to_quaternion,
    angle_between_directions,
)


# ===== array_backend tests =====


class TestArrayBackend:
    def test_get_array_module_numpy(self):
        """Default (use_cuda=False) returns numpy."""
        xp = get_array_module(use_cuda=False)
        assert xp is np

    def test_get_array_module_cuda_fallback(self):
        """When CuPy is unavailable, use_cuda=True falls back to numpy."""
        xp = get_array_module(use_cuda=True)
        # Either cupy (if installed) or numpy
        assert hasattr(xp, "array")


# ===== quaternion_math tests =====


class TestBatchQuaternionAngle:
    def test_identity_vs_identity(self):
        """Angle between identical quaternions is zero."""
        q = np.array([[0, 0, 0, 1]])
        angles = batch_quaternion_angle(q, q, np)
        assert angles.shape == (1, 1)
        assert np.allclose(angles, 0.0, atol=1e-6)

    def test_identity_vs_180(self):
        """Angle between identity and 180° rotation is π."""
        q1 = np.array([[0, 0, 0, 1]])
        q2 = np.array([[1, 0, 0, 0]])  # 180° about X
        angles = batch_quaternion_angle(q1, q2, np)
        assert angles.shape == (1, 1)
        assert np.allclose(angles, np.pi, atol=1e-6)

    def test_batch_shape(self):
        """Output shape is (N, M) for N×M inputs."""
        q1 = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])
        q2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
        angles = batch_quaternion_angle(q1, q2, np)
        assert angles.shape == (2, 3)

    def test_symmetry(self):
        """dist(q1, q2) == dist(q2, q1)."""
        q1 = np.array([[0, 0, 0, 1]])
        q2 = np.array([[0, 0, np.sin(np.pi / 6), np.cos(np.pi / 6)]])
        assert np.allclose(
            batch_quaternion_angle(q1, q2, np),
            batch_quaternion_angle(q2, q1, np),
        )


# ===== quaternion_utils tests =====


class TestQuaternionFromEuler:
    def test_identity(self):
        """Zero Euler angles give identity quaternion."""
        q = quaternion_from_euler([0, 0, 0])
        expected = R.from_euler("zyx", [0, 0, 0], degrees=True).as_quat()
        assert np.allclose(q, expected)

    def test_90_degree_yaw(self):
        """90° yaw produces the expected quaternion."""
        q = quaternion_from_euler([90, 0, 0], order="zyx", degrees=True)
        expected = R.from_euler("zyx", [90, 0, 0], degrees=True).as_quat()
        assert np.allclose(q, expected, atol=1e-6)

    def test_radians(self):
        """Works in radians when degrees=False."""
        q = quaternion_from_euler([np.pi / 2, 0, 0], order="zyx", degrees=False)
        expected = R.from_euler("zyx", [np.pi / 2, 0, 0], degrees=False).as_quat()
        assert np.allclose(q, expected, atol=1e-6)


class TestQuaternionDistance:
    def test_same_quaternion(self):
        """Distance between identical quaternions is zero."""
        q = [0, 0, 0, 1]
        assert quaternion_distance(q, q) == pytest.approx(0.0, abs=1e-7)

    def test_opposite_quaternion(self):
        """Distance between q and 180° rotated is π."""
        q1 = [0, 0, 0, 1]
        q2 = [1, 0, 0, 0]
        assert quaternion_distance(q1, q2) == pytest.approx(np.pi, abs=1e-6)

    def test_symmetry(self):
        """Distance is symmetric."""
        q1 = [0, 0, 0, 1]
        q2 = [0, 0, np.sin(np.pi / 6), np.cos(np.pi / 6)]
        assert quaternion_distance(q1, q2) == pytest.approx(
            quaternion_distance(q2, q1), abs=1e-7
        )


class TestRotateQuaternion:
    def test_identity_rotation(self):
        """Rotating by identity gives back the same quaternion."""
        q = [0, 0, np.sin(np.pi / 6), np.cos(np.pi / 6)]
        result = rotate_quaternion(q, [0, 0, 0, 1])
        assert np.allclose(result, q, atol=1e-6)

    def test_compose_rotations(self):
        """Composing two rotations matches scipy."""
        q = [0, 0, np.sin(np.pi / 6), np.cos(np.pi / 6)]
        step = [0, np.sin(np.pi / 8), 0, np.cos(np.pi / 8)]
        result = rotate_quaternion(q, step)
        expected = (R.from_quat(q) * R.from_quat(step)).as_quat()
        assert np.allclose(result, expected, atol=1e-6)


class TestIsWithinAngleThreshold:
    def test_within(self):
        """Quaternions within threshold returns True."""
        q1 = [0, 0, 0, 1]
        q2 = [0, 0, np.sin(0.05), np.cos(0.05)]
        assert is_within_angle_threshold(q1, q2, threshold_rad=0.2)

    def test_outside(self):
        """Quaternions outside threshold returns False."""
        q1 = [0, 0, 0, 1]
        q2 = [1, 0, 0, 0]  # 180° away
        assert not is_within_angle_threshold(q1, q2, threshold_rad=0.1)


class TestQuaternionConjugate:
    def test_conjugate(self):
        """Conjugate negates the vector part."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        conj = quaternion_conjugate(q)
        assert np.allclose(conj, [-0.1, -0.2, -0.3, 0.9])


class TestQuaternionMultiply:
    def test_identity_multiply(self):
        """Multiplying by identity gives the same quaternion."""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        result = quaternion_multiply(q, identity)
        assert np.allclose(result, q, atol=1e-10)

    def test_matches_scipy(self):
        """Hamilton product matches scipy Rotation composition."""
        q1 = R.random().as_quat()
        q2 = R.random().as_quat()
        result = quaternion_multiply(q1, q2)
        expected = (R.from_quat(q1) * R.from_quat(q2)).as_quat()
        assert np.allclose(result, expected, atol=1e-10) or np.allclose(
            result, -expected, atol=1e-10
        )


class TestRotationMatrixToQuaternion:
    def test_identity_matrix(self):
        """Identity matrix maps to identity quaternion."""
        q = rotation_matrix_to_quaternion(np.eye(3))
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(q, expected, atol=1e-6) or np.allclose(
            q, -expected, atol=1e-6
        )

    def test_roundtrip_with_scipy(self):
        """Conversion matches scipy's matrix→quaternion roundtrip."""
        original = R.random()
        mat = original.as_matrix()
        q = rotation_matrix_to_quaternion(mat)
        # Check that the resulting rotation is equivalent
        reconstructed = R.from_quat(q)
        angle = (original.inv() * reconstructed).magnitude()
        assert angle < 1e-6 or abs(angle - 2 * np.pi) < 1e-6


class TestGetRelativeSpin:
    def test_same_orientation(self):
        """Relative spin between identical orientations is identity."""
        n = Node("n", [0, 0, 0, 1])
        q = get_relative_spin(n, n)
        assert np.allclose(q, [0, 0, 0, 1], atol=1e-6) or np.allclose(
            q, [0, 0, 0, -1], atol=1e-6
        )

    def test_relative_spin_nonzero(self):
        """Relative spin between different orientations is non-trivial."""
        n1 = Node("a", [0, 0, 0, 1])
        n2 = Node("b", [1, 0, 0, 0])
        q = get_relative_spin(n1, n2)
        # Should be a unit quaternion
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)


# ===== get_unique_relative_spins tests (requires healpy) =====


class TestGetUniqueRelativeSpins:
    def test_returns_list_of_unit_quaternions(self):
        """Unique spins should be a list of unit quaternions."""
        hp = pytest.importorskip("healpy", reason="healpy not installed")
        from spinstep.utils.quaternion_utils import get_unique_relative_spins

        nside = 1
        npix = hp.nside2npix(nside)
        # Build nodes with orientations derived from HEALPix pixel directions
        nodes = []
        for i in range(npix):
            theta, phi = hp.pix2ang(nside, i, nest=True)
            q = R.from_euler("yz", [theta, phi], degrees=False).as_quat()
            nodes.append(Node(f"n{i}", q))

        spins = get_unique_relative_spins(nodes, nside=nside, nest=True)
        assert isinstance(spins, list)
        assert len(spins) > 0
        for q in spins:
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)
            assert q[3] >= 0  # canonical form (w >= 0)


# ===== forward_vector_from_quaternion tests =====


class TestForwardVectorFromQuaternion:
    def test_identity_forward(self):
        """Identity quaternion forward is [0, 0, -1]."""
        fwd = forward_vector_from_quaternion([0, 0, 0, 1])
        assert np.allclose(fwd, [0, 0, -1], atol=1e-6)

    def test_180_yaw(self):
        """180° yaw flips forward to [0, 0, 1]."""
        q = R.from_euler("y", 180, degrees=True).as_quat()
        fwd = forward_vector_from_quaternion(q)
        assert np.allclose(fwd, [0, 0, 1], atol=1e-6)

    def test_unit_length(self):
        """Forward vector is always unit length."""
        q = R.random().as_quat()
        fwd = forward_vector_from_quaternion(q)
        assert np.linalg.norm(fwd) == pytest.approx(1.0, abs=1e-6)


# ===== direction_to_quaternion tests =====


class TestDirectionToQuaternion:
    def test_forward_direction(self):
        """Direction [0, 0, -1] gives identity-like quaternion."""
        q = direction_to_quaternion([0, 0, -1])
        fwd = R.from_quat(q).apply([0, 0, -1])
        assert np.allclose(fwd, [0, 0, -1], atol=1e-6)

    def test_roundtrip(self):
        """direction_to_quaternion → forward_vector_from_quaternion roundtrip."""
        direction = np.array([1.0, 2.0, -3.0])
        direction = direction / np.linalg.norm(direction)
        q = direction_to_quaternion(direction)
        fwd = forward_vector_from_quaternion(q)
        assert np.allclose(fwd, direction, atol=1e-6)

    def test_unit_quaternion(self):
        """Returned quaternion is a unit quaternion."""
        q = direction_to_quaternion([1, 0, 0])
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector(self):
        """Zero vector returns identity quaternion."""
        q = direction_to_quaternion([0, 0, 0])
        assert np.allclose(q, [0, 0, 0, 1], atol=1e-6)


# ===== angle_between_directions tests =====


class TestAngleBetweenDirections:
    def test_same_direction(self):
        """Angle between identical directions is zero."""
        d = [1, 0, 0]
        assert angle_between_directions(d, d) == pytest.approx(0.0, abs=1e-7)

    def test_opposite_directions(self):
        """Angle between opposite directions is π."""
        assert angle_between_directions(
            [0, 0, 1], [0, 0, -1]
        ) == pytest.approx(np.pi, abs=1e-6)

    def test_perpendicular_directions(self):
        """Angle between perpendicular directions is π/2."""
        assert angle_between_directions(
            [1, 0, 0], [0, 1, 0]
        ) == pytest.approx(np.pi / 2, abs=1e-6)

    def test_unnormalized_inputs(self):
        """Works with non-unit direction vectors."""
        assert angle_between_directions(
            [3, 0, 0], [0, 4, 0]
        ) == pytest.approx(np.pi / 2, abs=1e-6)

    def test_zero_vector(self):
        """Zero vector returns 0.0."""
        assert angle_between_directions([0, 0, 0], [1, 0, 0]) == pytest.approx(
            0.0, abs=1e-7
        )

