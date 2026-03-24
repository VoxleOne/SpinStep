# test_traversal.py — SpinStep Test Suite — MIT License
# Author: SpinStep Contributors — Created: 2026-03-24
# See LICENSE.txt for full terms.

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from spinstep.node import Node
from spinstep.traversal import QuaternionDepthIterator


class TestQuaternionDepthIterator:
    """Tests for the continuous quaternion-driven depth-first traversal."""

    def test_single_node(self):
        """Iterating a single root node yields that node."""
        root = Node("root", [0, 0, 0, 1])
        names = [n.name for n in QuaternionDepthIterator(root, [0, 0, 0, 1])]
        assert names == ["root"]

    def test_root_with_close_child(self):
        """A child within the angular threshold is visited."""
        step_quat = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]  # ~30° about Z
        child_quat = step_quat  # same orientation as the step
        root = Node("root", [0, 0, 0, 1], [Node("child", child_quat)])

        names = [n.name for n in QuaternionDepthIterator(root, step_quat)]
        assert "root" in names
        assert "child" in names

    def test_root_with_far_child(self):
        """A child far from the rotated state is not visited."""
        step_quat = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]  # ~30° about Z
        far_quat = [0, np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]  # 180° about Y
        root = Node("root", [0, 0, 0, 1], [Node("far", far_quat)])

        names = [n.name for n in QuaternionDepthIterator(root, step_quat)]
        assert "root" in names
        assert "far" not in names

    def test_explicit_angle_threshold(self):
        """Providing an explicit threshold overrides the dynamic default."""
        step_quat = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
        child_quat = [0, 0, np.sin(np.pi / 6), np.cos(np.pi / 6)]  # 60° about Z
        root = Node("root", [0, 0, 0, 1], [Node("child", child_quat)])

        # Very wide threshold — child should be visited
        names_wide = [
            n.name
            for n in QuaternionDepthIterator(root, step_quat, angle_threshold=np.pi)
        ]
        assert "child" in names_wide

        # Very narrow threshold — child should not be visited
        names_narrow = [
            n.name
            for n in QuaternionDepthIterator(root, step_quat, angle_threshold=0.001)
        ]
        assert "child" not in names_narrow

    def test_dynamic_threshold_identity_step(self):
        """When the step is identity (zero angle) the dynamic threshold clips to 1°."""
        root = Node("root", [0, 0, 0, 1])
        it = QuaternionDepthIterator(root, [0, 0, 0, 1])
        assert it.angle_threshold == pytest.approx(np.deg2rad(1.0))

    def test_dynamic_threshold_nonzero_step(self):
        """Dynamic threshold is 30% of the step angle for non-trivial steps."""
        step_angle = np.pi / 4  # 45°
        step_quat = [0, 0, np.sin(step_angle / 2), np.cos(step_angle / 2)]
        root = Node("root", [0, 0, 0, 1])
        it = QuaternionDepthIterator(root, step_quat)
        expected = step_angle * 0.3
        assert it.angle_threshold == pytest.approx(expected, abs=1e-6)

    def test_depth_traversal_order(self):
        """Depth-first traversal visits deeper nodes before siblings."""
        # Build a small tree:
        #       root
        #      /
        #     A
        #    /
        #   B
        step_quat = [0, 0, np.sin(np.pi / 24), np.cos(np.pi / 24)]  # ~15° about Z
        b = Node("B", step_quat)
        a = Node("A", step_quat, [b])
        root = Node("root", [0, 0, 0, 1], [a])

        names = [
            n.name
            for n in QuaternionDepthIterator(root, step_quat, angle_threshold=np.pi)
        ]
        assert names[0] == "root"
        # A and B should both be visited
        assert "A" in names
        assert "B" in names

    def test_multiple_children_selective(self):
        """Only children within threshold are visited from multiple candidates."""
        step_quat = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]  # ~30° about Z
        close_child = Node("close", step_quat)
        far_child = Node("far", [0, np.sin(np.pi / 2), 0, np.cos(np.pi / 2)])
        root = Node("root", [0, 0, 0, 1], [close_child, far_child])

        names = [n.name for n in QuaternionDepthIterator(root, step_quat)]
        assert "root" in names
        assert "close" in names
        assert "far" not in names

    def test_iterator_protocol(self):
        """QuaternionDepthIterator supports the iterator protocol."""
        root = Node("root", [0, 0, 0, 1])
        it = QuaternionDepthIterator(root, [0, 0, 0, 1])
        assert iter(it) is it
        first = next(it)
        assert first.name == "root"
        with pytest.raises(StopIteration):
            next(it)

    def test_zero_orientation_child_skipped(self):
        """Children with zero quaternion orientation are skipped gracefully."""
        step_quat = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
        # Build a child with near-zero orientation manually
        zero_child = Node.__new__(Node)
        zero_child.name = "zero"
        zero_child.orientation = np.array([0.0, 0.0, 0.0, 0.0])
        zero_child.children = []
        root = Node("root", [0, 0, 0, 1], [zero_child])

        # Should not raise; the zero-orientation child is skipped
        names = [
            n.name
            for n in QuaternionDepthIterator(root, step_quat, angle_threshold=np.pi)
        ]
        assert "root" in names
        assert "zero" not in names
