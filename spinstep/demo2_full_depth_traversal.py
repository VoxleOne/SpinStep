# demo2_full_depth_traversal.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R

# This version performs a depth-first traversal, rotating the current 
# quaternion state at each level, and only visits branches that are 
# within a defined angular threshold of the rotated orientation.

class Node:
    def __init__(self, name, orientation, children=None):
        self.name = name
        self.orientation = orientation  # Quaternion: [x, y, z, w]
        self.children = children if children else []

class QuaternionDepthIterator:
    def __init__(self, start_node, rotation_step_quat, angle_threshold=np.pi / 4):
        self.rotation_step = R.from_quat(rotation_step_quat)
        self.angle_threshold = angle_threshold
        self.stack = [(start_node, R.from_quat(start_node.orientation))]  # Stack holds (node, orientation)

    def __iter__(self):
        return self

    def __next__(self):
        while self.stack:
            node, state = self.stack.pop()
            rotated_state = state * self.rotation_step

            # Add children that are within angular threshold
            for child in node.children:
                target_orientation = R.from_quat(child.orientation)
                angle = (rotated_state.inv() * target_orientation).magnitude()
                if angle < self.angle_threshold:
                    self.stack.append((child, target_orientation))

            return node

        raise StopIteration

# Build a test tree
root = Node("root", [0, 0, 0, 1], [
    Node("child_1", R.from_euler('z', 30, degrees=True).as_quat(), [
        Node("grandchild_1", R.from_euler('z', 60, degrees=True).as_quat())
    ]),
    Node("child_2", R.from_euler('z', 90, degrees=True).as_quat(), [
        Node("grandchild_2", R.from_euler('z', 120, degrees=True).as_quat())
    ]),
    Node("child_3", R.from_euler('z', 150, degrees=True).as_quat())
])

# Use rotation around Z by 30 degrees per step
rotation_step = R.from_euler('z', 30, degrees=True).as_quat()
iterator = QuaternionDepthIterator(root, rotation_step)

# Run the traversal and print results
visited_nodes = [node.name for node in iterator]
print("Visited nodes:", visited_nodes)
