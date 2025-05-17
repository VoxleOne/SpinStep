# quaternion_utils.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_from_euler(angles, order='zyx', degrees=True):
    """Convert Euler angles to a quaternion."""
    return R.from_euler(order, angles, degrees=degrees).as_quat()

def quaternion_distance(q1, q2):
    """Calculate angular distance between two quaternions (in radians)."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1.inv() * r2).magnitude()

def rotate_quaternion(q, rotation_step):
    """Apply a rotation step to quaternion q."""
    r1 = R.from_quat(q)
    step = R.from_quat(rotation_step)
    return (r1 * step).as_quat()

def is_within_angle_threshold(q_current, q_target, threshold_rad):
    """Check if two quaternions are within a given angular distance."""
    return quaternion_distance(q_current, q_target) < threshold_rad
