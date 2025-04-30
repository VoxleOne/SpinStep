from the root, stepping 30° at a time.import numpy as np
from scipy.spatial.transform import Rotation as R

#This will output the names of child nodes visited by “rotating” toward them

class Node:
    def __init__(self, name, orientation, children=None):
        self.name = name
        self.orientation = orientation  # Quaternion: [x, y, z, w]
        self.children = children if children else []

def quaternion_angle_distance(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1.inv() * r2).magnitude()

class QuaternionBranchIterator:
    def __init__(self, start_node, rotation_step_quat, angle_threshold=np.pi / 4):
        self.current = start_node
        self.rotation_step = R.from_quat(rotation_step_quat)
        self.state = R.from_quat(start_node.orientation)
        self.angle_threshold = angle_threshold

    def __iter__(self):
        return self

    def __next__(self):
        if not self.current.children:
            raise StopIteration

        for child in self.current.children:
            target_orientation = R.from_quat(child.orientation)
            rotated_state = self.state * self.rotation_step
            angle = (rotated_state.inv() * target_orientation).magnitude()

            if angle < self.angle_threshold:
                self.current = child
                self.state = target_orientation
                return child

        raise StopIteration

# Example Tree
root = Node("root", [0, 0, 0, 1], [
    Node("child_1", R.from_euler('z', 30, degrees=True).as_quat()),
    Node("child_2", R.from_euler('z', 90, degrees=True).as_quat()),
    Node("child_3", R.from_euler('z', 150, degrees=True).as_quat())
])

# Create the iterator
rotation_step = R.from_euler('z', 30, degrees=True).as_quat()
iterator = QuaternionBranchIterator(root, rotation_step)

# Print visited nodes
visited_nodes = [node.name for node in iterator]
print("Visited nodes:", visited_nodes)
