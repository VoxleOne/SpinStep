from scipy.spatial.transform import Rotation as R

class QuaternionDepthIterator:
    def __init__(self, start_node, rotation_step_quat, angle_threshold):
        self.rotation_step = R.from_quat(rotation_step_quat)
        self.angle_threshold = angle_threshold
        self.stack = [(start_node, R.from_quat(start_node.orientation))]

    def __iter__(self): return self

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
