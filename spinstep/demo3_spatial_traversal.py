# demo3_spatial_traversal.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R

# This version combines yaw, pitch, and roll as the step, 
# and filters branches based on their angular alignment with 
# the rotated state — effectively simulating 
# "rotating into branches" in full 3D.

class Node:
    def __init__(self, name, orientation, children=None):
        self.name = name
        self.orientation = orientation  # Quaternion: [x, y, z, w]
        self.children = children if children else []

class QuaternionDepthIterator:
    def __init__(self, start_node, rotation_step_quat, angle_threshold=np.pi / 4):
        self.rotation_step = R.from_quat(rotation_step_quat)
        self.angle_threshold = angle_threshold
        self.stack = [(start_node, R.from_quat(start_node.orientation))]

    def __iter__(self):
        return self

    def __next__(self):
        while self.stack:
            node, state = self.stack.pop()
            rotated_state = state * self.rotation_step

            for child in node.children:
                target_orientation = R.from_quat(child.orientation)
                angle = (rotated_state.inv() * target_orientation).magnitude()
                if angle < self.angle_threshold:
                    self.stack.append((child, target_orientation))

            return node

        raise StopIteration

# Composite 3D rotation step: yaw (Z), pitch (Y), roll (X)
rotation_step_3d = R.from_euler('zyx', [30, 15, 10], degrees=True).as_quat()

# Define a 3D orientation tree
root_3d = Node("root", [0, 0, 0, 1], [
    Node("child_yaw", R.from_euler('z', 30, degrees=True).as_quat(), [
        Node("grandchild_yaw", R.from_euler('z', 60, degrees=True).as_quat())
    ]),
    Node("child_pitch", R.from_euler('y', 15, degrees=True).as_quat(), [
        Node("grandchild_pitch", R.from_euler('y', 30, degrees=True).as_quat())
    ]),
    Node("child_roll", R.from_euler('x', 10, degrees=True).as_quat(), [
        Node("grandchild_roll", R.from_euler('x', 20, degrees=True).as_quat())
    ]),
    Node("unrelated", R.from_euler('x', 90, degrees=True).as_quat())  # Should be skipped
])

# Create the iterator
iterator_3d = QuaternionDepthIterator(root_3d, rotation_step_3d)

# Run the traversal and print visited node names
visited_nodes = [node.name for node in iterator_3d]
print("Visited nodes:", visited_nodes)
