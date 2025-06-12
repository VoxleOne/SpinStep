# spinstep/utils/quaternion_utils.py
# Consolidated quaternion utilities for the SpinStep project.
# MIT License - Author: Eraldo Marques <eraldo.bernardo@gmail.com> (for original parts)
# SpinStep contributors (for adaptations and new functions).
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

# --- Functions leveraging scipy.spatial.transform.Rotation ---

def quaternion_from_euler(angles, order='zyx', degrees=True):
    """Convert Euler angles to a quaternion [x, y, z, w]."""
    return ScipyRotation.from_euler(order, angles, degrees=degrees).as_quat()

def quaternion_angular_distance_scipy(q1_xyzw, q2_xyzw):
    """
    Calculate angular distance between two quaternions (in radians) [x, y, z, w].
    Uses scipy.spatial.transform.Rotation.
    """
    # Ensure inputs are normalized for robust calculation with scipy
    norm1 = np.linalg.norm(q1_xyzw)
    norm2 = np.linalg.norm(q2_xyzw)
    if norm1 < 1e-8 or norm2 < 1e-8: return np.pi # Max distance if one is zero quat
    q1_normalized = q1_xyzw / norm1
    q2_normalized = q2_xyzw / norm2
    
    r1 = ScipyRotation.from_quat(q1_normalized)
    r2 = ScipyRotation.from_quat(q2_normalized)
    # The magnitude of the relative rotation r1_inv * r2 is the angle.
    return (r1.inv() * r2).magnitude()

def rotate_quaternion_scipy(q_xyzw, rotation_step_xyzw):
    """
    Apply a rotation_step_xyzw to quaternion q_xyzw. Returns new quaternion [x, y, z, w].
    Uses scipy.spatial.transform.Rotation.
    """
    r1 = ScipyRotation.from_quat(q_xyzw)
    step = ScipyRotation.from_quat(rotation_step_xyzw)
    return (r1 * step).as_quat()

def is_within_angle_threshold_scipy(q_current_xyzw, q_target_xyzw, threshold_rad):
    """
    Check if two quaternions are within a given angular distance.
    Uses scipy.spatial.transform.Rotation.
    """
    return quaternion_angular_distance_scipy(q_current_xyzw, q_target_xyzw) <= threshold_rad

# --- Manual NumPy-based Quaternion Operations ---

def quaternion_conjugate(q_xyzw):
    """Return the conjugate of a quaternion [x,y,z,w]."""
    return np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]])

def quaternion_multiply(q1_xyzw, q2_xyzw):
    """Multiply two quaternions [x,y,z,w]. q_product = q1 * q2."""
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    res = np.array([x, y, z, w])
    # Renormalize after multiplication to counteract potential floating point drift
    norm = np.linalg.norm(res)
    return res / norm if norm > 1e-8 else np.array([0.,0.,0.,1.])


def get_relative_spin_quaternion(q_from_xyzw, q_to_xyzw):
    """
    Calculates the relative spin quaternion to rotate from orientation q_from to q_to.
    q_spin * q_from = q_to  => q_spin = q_to * conjugate(q_from)
    Assumes q_from and q_to are unit quaternions [x,y,z,w].
    """
    q_from_conj = quaternion_conjugate(q_from_xyzw)
    q_spin = quaternion_multiply(q_to_xyzw, q_from_conj) # Order: q_to * q_from_conj
    
    norm = np.linalg.norm(q_spin) # Should be close to 1 if inputs are normalized
    return q_spin / norm if norm > 1e-8 else np.array([0.,0.,0.,1.])

def rotation_matrix_to_quaternion(m):
    """
    Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].
    """
    t = np.trace(m)
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0 # Default for safety
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qx = 0.25 * s
        qw = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qy = 0.25 * s
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qz = 0.25 * s
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
    
    q = np.array([qx, qy, qz, qw])
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([0., 0., 0., 1.])

# --- Core function for SpinStep's Fibonacci-based orientation generation ---

def _calculate_node_orientation_from_vector_and_angles(theta_from_y_pole, phi_in_xz_plane, vec_er_local_z):
    """
    Given spherical coordinates (theta_from_y_pole, phi_in_xz_plane) for a Y-up system, 
    and a radial unit vector vec_er_local_z (which becomes the local Z-axis),
    return a quaternion [x,y,z,w] representing the local frame.
    The local X-axis is defined based on phi_in_xz_plane, and Y-axis is derived.
    """
    e_r_local = vec_er_local_z # This is the local Z-axis

    if np.isclose(theta_from_y_pole, 0.0):
        e_x_local = np.array([np.cos(phi_in_xz_plane), 0.0, np.sin(phi_in_xz_plane)])
        e_y_local = np.cross(e_r_local, e_x_local)
    elif np.isclose(theta_from_y_pole, np.pi):
        e_x_local = np.array([np.cos(phi_in_xz_plane), 0.0, np.sin(phi_in_xz_plane)])
        e_y_local = np.cross(e_r_local, e_x_local)
    else:
        global_y_axis = np.array([0.0, 1.0, 0.0])
        e_x_local = np.cross(global_y_axis, e_r_local)
        norm_ex = np.linalg.norm(e_x_local)
        if norm_ex < 1e-6: 
             e_x_local = np.array([1.0, 0.0, 0.0]) 
        else:
            e_x_local /= norm_ex
        e_y_local = np.cross(e_r_local, e_x_local)
    
    R_matrix = np.stack([e_x_local, e_y_local, e_r_local], axis=1)
    return rotation_matrix_to_quaternion(R_matrix)
