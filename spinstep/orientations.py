# spinstep/orientations.py
# MIT License - Author: Eraldo Marques <eraldo.bernardo@gmail.com> (for original parts if any directly adapted)
# SpinStep contributors (for adaptations and new functions).
# See LICENSE.txt for full terms. This header must be retained in redistributions.
#
# This module provides functions for generating discrete node orientations on a sphere.
# The approach for distributing points (Fibonacci lattice) and defining orientations
# was developed independently. However, the initial conceptualization of using
# regular, indexed points on a sphere for node generation was partly inspired by
# the structured nature of systems like HEALPix, though HEALPix itself is not
# used as a dependency or component in this module's implementation.

import numpy as np
from spinstep.utils.quaternion_utils import _calculate_node_orientation_from_vector_and_angles

_TIER_TO_N_POINTS_MAP = {
    0: 12, 1: 48, 2: 192, 3: 768, 4: 3072,
}

def _generate_fibonacci_sphere_point(index, num_points):
    if not (0 <= index < num_points):
        raise ValueError(f"Index {index} is out of bounds for {num_points} points.")
    phi_angle_increment = np.pi * (3.0 - np.sqrt(5.0))
    y_coord = 1.0 - (2.0 * (index + 0.5)) / num_points
    radius_at_y = np.sqrt(np.clip(1.0 - y_coord * y_coord, 0.0, 1.0))
    current_phi = phi_angle_increment * index
    x_coord = np.cos(current_phi) * radius_at_y
    z_coord = np.sin(current_phi) * radius_at_y
    return np.array([x_coord, y_coord, z_coord])

def get_number_of_nodes_at_tier(resolution_tier):
    if resolution_tier not in _TIER_TO_N_POINTS_MAP:
        raise ValueError(f"Resolution tier {resolution_tier} is not defined in _TIER_TO_N_POINTS_MAP.")
    return _TIER_TO_N_POINTS_MAP[resolution_tier]

def get_discrete_node_orientation(resolution_tier, node_index_at_tier):
    num_points_for_tier = get_number_of_nodes_at_tier(resolution_tier)
    if not (0 <= node_index_at_tier < num_points_for_tier):
        raise ValueError(
            f"node_index_at_tier {node_index_at_tier} is out of bounds for "
            f"resolution_tier {resolution_tier}, which has {num_points_for_tier} nodes "
            f"(indices 0 to {num_points_for_tier-1})."
        )
    vec_er_local_z = _generate_fibonacci_sphere_point(node_index_at_tier, num_points_for_tier)
    fx, fy, fz = vec_er_local_z[0], vec_er_local_z[1], vec_er_local_z[2]
    theta_from_y_pole = np.arccos(np.clip(fy, -1.0, 1.0))
    phi_in_xz_plane = np.arctan2(fz, fx)
    if phi_in_xz_plane < 0:
        phi_in_xz_plane += 2 * np.pi
    orientation_quaternion = _calculate_node_orientation_from_vector_and_angles(
        theta_from_y_pole, 
        phi_in_xz_plane, 
        vec_er_local_z
    )
    return orientation_quaternion
