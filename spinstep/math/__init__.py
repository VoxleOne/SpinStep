# math/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion mathematics library.

Sub-modules:

- :mod:`~.core` — multiply, conjugate, normalize, inverse
- :mod:`~.interpolation` — slerp, squad
- :mod:`~.geometry` — distance, angle, direction conversions
- :mod:`~.conversions` — Euler ↔ quaternion, matrix ↔ quaternion
- :mod:`~.analysis` — batch distances, angular velocity, relative spins
- :mod:`~.constraints` — rotation angle clamping
"""

__all__ = [
    # core
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_normalize",
    "quaternion_inverse",
    # interpolation
    "slerp",
    "squad",
    # geometry
    "quaternion_distance",
    "is_within_angle_threshold",
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "angle_between_directions",
    "rotate_quaternion",
    # conversions
    "quaternion_from_euler",
    "rotation_matrix_to_quaternion",
    "quaternion_from_rotvec",
    "quaternion_to_rotvec",
    # analysis
    "batch_quaternion_angle",
    "angular_velocity_from_quaternions",
    "get_relative_spin",
    "get_unique_relative_spins",
    # constraints
    "clamp_rotation_angle",
    # protocols
    "NodeProtocol",
    "SpatialNodeProtocol",
]

from .core import (
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
)
from .interpolation import slerp, squad
from .geometry import (
    angle_between_directions,
    direction_to_quaternion,
    forward_vector_from_quaternion,
    is_within_angle_threshold,
    quaternion_distance,
    rotate_quaternion,
)
from .conversions import (
    quaternion_from_euler,
    quaternion_from_rotvec,
    quaternion_to_rotvec,
    rotation_matrix_to_quaternion,
)
from .analysis import (
    NodeProtocol,
    SpatialNodeProtocol,
    angular_velocity_from_quaternions,
    batch_quaternion_angle,
    get_relative_spin,
    get_unique_relative_spins,
)
from .constraints import clamp_rotation_angle
