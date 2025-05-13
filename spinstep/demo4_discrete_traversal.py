import numpy as np
from scipy.spatial.transform import Rotation as R
from spinstep.node import Node
from spinstep.discrete import DiscreteOrientationSet
from spinstep.discrete_iterator import DiscreteQuaternionIterator

# Build a test tree with orientations based on 90-degree cube steps
root = Node("root", [0, 0, 0, 1], [
    Node("child_x90", R.from_euler('x', 90, degrees=True).as_quat(), [
        Node("grandchild_x180", R.from_euler('x', 180, degrees=True).as_quat())
    ]),
    Node("child_y90", R.from_euler('y', 90, degrees=True).as_quat(), [
        Node("grandchild_y180", R.from_euler('y', 180, degrees=True).as_quat())
    ]),
    Node("child_z90", R.from_euler('z', 90, degrees=True).as_quat(), [
        Node("grandchild_z180", R.from_euler('z', 180, degrees=True).as_quat())
    ]),
])

# Use the cube symmetry group as allowed discrete steps
cube_set = DiscreteOrientationSet.from_cube()

iterator = DiscreteQuaternionIterator(root, cube_set, angle_threshold=np.pi/6)  # 30 degrees

visited_nodes = []
for node in iterator:
    visited_nodes.append(node.name)
    print("Visited:", node.name)