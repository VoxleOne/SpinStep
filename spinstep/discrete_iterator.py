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
        if not hasattr(start_node, "orientation") or not hasattr(start_node, "children"):
            raise AttributeError("Node must have .orientation and .children")
        self.stack = [(start_node, R.from_quat(start_node.orientation), 0)]
        self._visited = set()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.stack:
            raise StopIteration

        while self.stack:
            node, current_node_world_orientation, depth = self.stack.pop()

            if depth > self.max_depth:
                continue

            node_id = id(node)
            if node_id in self._visited:
                continue
            self._visited.add(node_id)

            child_potential_depth = depth + 1
            if child_potential_depth <= self.max_depth:
                if hasattr(node, "children"):
                    for step_quat_from_set in self.orientation_set.orientations:
                        potential_next_world_orientation = current_node_world_orientation * R.from_quat(step_quat_from_set)

                        for child_node_obj in getattr(node, "children", []):
                            if id(child_node_obj) not in self._visited:
                                child_actual_world_orientation = R.from_quat(child_node_obj.orientation)

                                angle_rad = 2 * (potential_next_world_orientation.inv() * child_actual_world_orientation).magnitude()

                                if angle_rad < self.angle_threshold:
                                    self.stack.append((child_node_obj, child_actual_world_orientation, child_potential_depth))

            return node

        raise StopIteration
