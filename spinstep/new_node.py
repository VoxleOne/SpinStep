# spinstep/node.py
# MIT License - Author: Eraldo Marques <eraldo.bernardo@gmail.com> (for original parts if any directly adapted)
# SpinStep contributors (for adaptations and new functions).
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np

class Node:
    """
    Represents a single discrete point (node) in the SpinStep graph.
    Each node has a defined position, orientation, and connections to neighbors.
    """
    def __init__(self, layer_index, node_index_on_layer, resolution_tier, orientation_q, radius):
        """
        Initializes a Node.

        Args:
            layer_index (int): The index of the layer this node belongs to.
            node_index_on_layer (int): The index of this node within its layer.
            resolution_tier (int): The resolution tier of the layer this node belongs to.
                                   Used for informational purposes or if different node types exist.
            orientation_q (np.ndarray): A unit quaternion [x, y, z, w] representing the
                                        node's intrinsic orientation in the global frame.
                                        The local Z-axis of the node points radially outward.
            radius (float): The radius of the spherical layer this node resides on.
        """
        if not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("layer_index must be a non-negative integer.")
        if not isinstance(node_index_on_layer, int) or node_index_on_layer < 0:
            raise ValueError("node_index_on_layer must be a non-negative integer.")
        if not isinstance(orientation_q, np.ndarray) or orientation_q.shape != (4,):
            raise ValueError("orientation_q must be a numpy array of shape (4,).")
        if not (np.isclose(np.linalg.norm(orientation_q), 1.0)):
            # Normalize if close, error if far (or just normalize always)
            norm = np.linalg.norm(orientation_q)
            if norm > 1e-6 : # Avoid division by zero for zero vector
                 orientation_q = orientation_q / norm
            else: # Should not happen for valid orientations
                 raise ValueError("orientation_q must be a unit quaternion (or normalizable).")

        if not (isinstance(radius, (int, float)) and radius > 0):
            raise ValueError("radius must be a positive number.")

        self.layer_index = layer_index
        self.node_index_on_layer = node_index_on_layer
        self.id = (layer_index, node_index_on_layer) # Unique identifier for the node

        self.resolution_tier = resolution_tier
        self.orientation = orientation_q # Intrinsic orientation [x,y,z,w]
        self.radius = radius

        # Position is derived from orientation (local Z pointing outward) and radius
        # Assuming orientation quaternion rotates [0,0,1] (local Z) to the world frame.
        # To get the world vector for local Z: q * [0,0,0,1] * q_conj (if [0,0,0,1] is pure quat for [0,0,1] vector)
        # Or, more simply, the third column of the rotation matrix from the quaternion.
        # If q = [qx,qy,qz,qw], R_31 = 2(qxqz + qyqw), R_32 = 2(qyqz - qxqw), R_33 = 1 - 2(qx^2 + qy^2)
        # This vector is the direction of the node from the origin.
        qx, qy, qz, qw = orientation_q[0], orientation_q[1], orientation_q[2], orientation_q[3]
        # Position vector = radius * (direction vector of node's Z-axis in world frame)
        # The direction vector (local Z) from orientation_q:
        # This is the third column of the rotation matrix represented by orientation_q
        # R = [[1-2(qy^2+qz^2),   2(qxqy-qzqw),   2(qxqz+qyqw)],
        #      [2(qxqy+qzqw), 1-2(qx^2+qz^2),   2(qyqz-qxqw)],
        #      [2(qxqz-qyqw),   2(qyqz+qxqw), 1-2(qx^2+qy^2)]]
        # So, local_z_direction = [2(qxqz+qyqw), 2(qyqz-qxqw), 1-2(qx^2+qy^2)]
        # No, this is if Z is [0,0,1]. Our Fibonacci points used Y as polar axis.
        # The _calculate_node_orientation_from_vector_and_angles in quaternion_utils
        # sets vec_er_local_z as the local Z-axis. This vec_er_local_z *is* the direction.
        # The orientation quaternion itself, when applied to a reference "pointing" vector (e.g. [0,0,1]),
        # gives the direction. The vec_er_local_z from Fibonacci IS this direction.
        # So, the orientation_q is constructed such that its local Z *is* vec_er_local_z.
        # We need to extract this effective "pointing vector" (local Z) from the orientation_q.
        # This is equivalent to rotating a basis vector (e.g., [0,0,1]) by orientation_q.
        # Let v = [0,0,1]. Rotated v' = qvq*.
        # Simpler: the third column of the rotation matrix from q.
        # Or, if orientation_q was built from vec_er (as in our orientations.py), then vec_er was:
        # fx, fy, fz. This vec_er *is* the direction.
        # The `get_discrete_node_orientation` returns a quaternion. The position should be
        # determined by what that quaternion considers its "forward" or "up" vector,
        # scaled by radius.
        # In our `_calculate_node_orientation_from_vector_and_angles`, `vec_er_local_z`
        # becomes the local Z axis. So, this vector *is* the direction.
        # We need to re-derive it or ensure Node gets it.
        # For now, let's assume orientation_q correctly orients a canonical frame,
        # and its Z-axis points along the Fibonacci vector.
        # The Z-axis of the local frame defined by `orientation_q` is:
        # [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx*qx + qy*qy)]
        # This is the direction vector if the canonical "up" was [0,0,1] before rotation.
        # Our `_calculate_node_orientation_from_vector_and_angles` makes `vec_er_local_z` the 3rd column of R.
        # So, R[:, 2] is the direction.
        # R_matrix = ScipyRotation.from_quat(orientation_q).as_matrix()
        # direction_vec = R_matrix[:, 2]
        # self.position = direction_vec * radius
        # Let's recalculate direction_vec for simplicity, assuming local Z is [0,0,1] rotated by orientation_q
        # (This is standard for qvq*)
        # v = np.array([0,0,1]) ; v_pure_quat = np.concatenate(([v, [0]]))
        # q_conj = np.array([-qx,-qy,-qz,qw])
        # temp = quaternion_multiply(orientation_q, v_pure_quat)
        # rotated_v_pure_quat = quaternion_multiply(temp, q_conj)
        # direction_vec = rotated_v_pure_quat[:3]
        # This is complex. Simpler: the Z-axis of the rotation matrix from orientation_q.
        # Z-axis components from quaternion:
        dir_x = 2 * (qx * qz + qw * qy)
        dir_y = 2 * (qy * qz - qw * qx)
        dir_z = 1 - 2 * (qx * qx + qy * qy)
        direction_vector = np.array([dir_x, dir_y, dir_z])
        self.position = direction_vector * radius


        # Connections to other nodes
        self.tangential_neighbors = []  # List of tuples: (Node_object, relative_spin_quaternion)
        self.radial_outward_neighbor = None # Node_object or None
        self.radial_inward_neighbor = None  # Node_object or None
        # Relative spins for radial neighbors are calculated on-the-fly by SpinStepGraph if needed.

    def add_tangential_neighbor(self, neighbor_node, relative_spin_q):
        """
        Adds a tangential neighbor and the relative spin to reach it.

        Args:
            neighbor_node (Node): The neighboring Node object.
            relative_spin_q (np.ndarray): Quaternion for relative spin from this node's
                                          orientation to the neighbor's orientation.
        """
        if not isinstance(neighbor_node, Node):
            raise ValueError("neighbor_node must be a Node object.")
        if not (isinstance(relative_spin_q, np.ndarray) and relative_spin_q.shape == (4,)):
            raise ValueError("relative_spin_q must be a numpy array of shape (4,).")
        
        self.tangential_neighbors.append((neighbor_node, relative_spin_q))

    def set_radial_outward_neighbor(self, neighbor_node):
        """Sets the radially outward neighbor."""
        if neighbor_node is not None and not isinstance(neighbor_node, Node):
            raise ValueError("neighbor_node must be a Node object or None.")
        self.radial_outward_neighbor = neighbor_node

    def set_radial_inward_neighbor(self, neighbor_node):
        """Sets the radially inward neighbor."""
        if neighbor_node is not None and not isinstance(neighbor_node, Node):
            raise ValueError("neighbor_node must be a Node object or None.")
        self.radial_inward_neighbor = neighbor_node

    def __repr__(self):
        return (f"Node(id={self.id}, tier={self.resolution_tier}, r={self.radius:.2f}, "
                f"pos=({self.position[0]:.2f},{self.position[1]:.2f},{self.position[2]:.2f}), "
                f"ori_q=({self.orientation[0]:.2f},{self.orientation[1]:.2f},"
                f"{self.orientation[2]:.2f},{self.orientation[3]:.2f}), "
                f"T_neigh={len(self.tangential_neighbors)}, "
                f"R_out={'Yes' if self.radial_outward_neighbor else 'No'}, "
                f"R_in={'Yes' if self.radial_inward_neighbor else 'No'})")

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

# Helper for quaternion multiplication if needed directly here (though usually in utils)
# def quaternion_multiply(q1, q2):
#     x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
#     return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
#                      w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])