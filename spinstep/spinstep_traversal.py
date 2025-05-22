from scipy.spatial.transform import Rotation as R
import numpy as np # Added for np.pi and small number comparison

class QuaternionDepthIterator:
    DEFAULT_DYNAMIC_THRESHOLD_FACTOR = 0.3 # Example: threshold is 30% of the step angle

    def __init__(self, start_node, rotation_step_quat, angle_threshold=None):
        self.rotation_step = R.from_quat(rotation_step_quat)

        if angle_threshold is None:
            # Dynamically calculate the threshold
            step_angle_rad = self.rotation_step.magnitude()

            # Handle the case where the rotation step is very small (or zero)
            if step_angle_rad < 1e-7: # Threshold for considering it a "zero" rotation
                # For a near-identity rotation step, a percentage-based threshold is not ideal.
                # Use a small, fixed absolute threshold, e.g., 1 degree (in radians).
                # You might want to adjust this default for such cases.
                self.angle_threshold = np.deg2rad(1.0)
                # Alternatively, you could raise an error or require an explicit threshold
                # if the step angle is too small for dynamic calculation.
            else:
                self.angle_threshold = step_angle_rad * self.DEFAULT_DYNAMIC_THRESHOLD_FACTOR
        else:
            # Use the explicitly provided angle_threshold
            self.angle_threshold = angle_threshold
            
        self.stack = [(start_node, R.from_quat(start_node.orientation))]

    def __iter__(self): return self

    def __next__(self):
        while self.stack:
            node, state = self.stack.pop()
            rotated_state = state * self.rotation_step

            for child in node.children:
                target_orientation = R.from_quat(child.orientation)
                # Calculate the angle between the expected (rotated_state) and actual child orientation
                # (A.inv() * B).magnitude() gives the angle between rotations A and B
                try:
                    # Ensure target_orientation is not a zero quaternion if not already normalized in Node
                    if np.allclose(child.orientation, [0,0,0,0]): # Or however you check for invalid orientations
                        # Skip this child or handle appropriately
                        continue
                    
                    # The angle calculation should be robust.
                    # If state or target_orientation could be invalid, ensure they are handled.
                    # R.from_quat should handle normalization if quaternions are valid.
                    angle_difference_rotation = rotated_state.inv() * target_orientation
                    angle = angle_difference_rotation.magnitude() # This is in radians
                except ValueError as e:
                    # This might happen if a quaternion is invalid (e.g., zero norm)
                    # before being passed to R.from_quat, though Node class should prevent this.
                    print(f"Warning: Could not calculate angle for child {child.name}. Error: {e}")
                    continue


                if angle < self.angle_threshold:
                    self.stack.append((child, target_orientation)) # Use target_orientation as the new state for this path

            return node
        raise StopIteration
