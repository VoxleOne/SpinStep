# demo.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

from spinstep.node import Node
from spinstep.traversal import QuaternionDepthIterator
from spinstep.utils.quaternion_utils import quaternion_from_euler

#What This Demo Does
# .Defines a 3D orientation tree with yaw/pitch/roll variations.
# .Applies a quaternion-based depth-first traversal.
# .Only visits nodes that lie within a given angular threshold (like aiming a "cone" of rotation).

def build_demo_tree():
    """Build a small tree with varied 3D orientations (yaw, pitch, roll)."""
    root = Node("root", [0, 0, 0, 1], [
        Node("yaw_branch", quaternion_from_euler([30, 0, 0]), [  # Yaw 30°
            Node("yaw_leaf", quaternion_from_euler([60, 0, 0]))   # Yaw 60°
        ]),
        Node("pitch_branch", quaternion_from_euler([0, 15, 0]), [  # Pitch 15°
            Node("pitch_leaf", quaternion_from_euler([0, 30, 0]))  # Pitch 30°
        ]),
        Node("roll_branch", quaternion_from_euler([0, 0, 10]), [   # Roll 10°
            Node("roll_leaf", quaternion_from_euler([0, 0, 20]))   # Roll 20°
        ]),
        Node("irrelevant", quaternion_from_euler([0, 90, 90]))      # Large offset
    ])
    return root

def main():
    print("🔄 Starting quaternion-based traversal...\n")

    rotation_step = quaternion_from_euler([30, 15, 10])  # Yaw + Pitch + Roll
    threshold_rad = 0.8  # ~45 degrees

    tree = build_demo_tree()
    iterator = QuaternionDepthIterator(tree, rotation_step, threshold_rad)

    for node in iterator:
        print(f"✅ Visited: {node.name}")

if __name__ == "__main__":
    main()
