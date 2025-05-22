# discrete_iterator.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R

class DiscreteQuaternionIterator:
    """
    Traverses a tree/graph using a discrete set of orientation steps.
    Each "step" is a quaternion from the provided orientation set.
    """

    def __init__(self, start_node, orientation_set, angle_threshold=np.pi/8, max_depth=100):
        self.orientation_set = orientation_set
        self.angle_threshold = angle_threshold
        self.max_depth = max_depth
        # Validate node
        if not hasattr(start_node, "orientation") or not hasattr(start_node, "children"):
            raise AttributeError("Node must have .orientation and .children")
        self.stack = [(start_node, R.from_quat(start_node.orientation), 0)]

    def __iter__(self):
        return self

    def __next__(self):
        while self.stack:
            node, state, depth = self.stack.pop()
            if depth >= self.max_depth:
                continue
            if not hasattr(node, "children") or not hasattr(node, "orientation"):
                raise AttributeError("Node missing required attributes during traversal")
            for step_quat in self.orientation_set.orientations:
                rotated_state = state * R.from_quat(step_quat)
                for child in getattr(node, "children", []):
                    target_orientation = R.from_quat(child.orientation)
                    angle = (rotated_state.inv() * target_orientation).magnitude()
                    if angle < self.angle_threshold:
                        self.stack.append((child, target_orientation, depth + 1))
            return node
        raise StopIteration
