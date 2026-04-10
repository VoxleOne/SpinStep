# utils/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Utilities: array backend selection and backward-compatible quaternion re-exports.

The ``get_array_module`` function is the primary utility provided here.
All quaternion math functions have moved to :mod:`spinstep.math` and are
re-exported here only for backward compatibility.

.. deprecated::
    For quaternion operations, import from :mod:`spinstep.math` instead.

Example (preferred)::

    from spinstep.math import quaternion_multiply, quaternion_distance
"""

__all__ = [
    "get_array_module",
    "batch_quaternion_angle",
    "quaternion_from_euler",
    "quaternion_distance",
    "rotate_quaternion",
    "is_within_angle_threshold",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotation_matrix_to_quaternion",
    "get_relative_spin",
    "get_unique_relative_spins",
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "angle_between_directions",
]

from .array_backend import get_array_module
from .quaternion_math import batch_quaternion_angle
from .quaternion_utils import (
    angle_between_directions,
    direction_to_quaternion,
    forward_vector_from_quaternion,
    get_relative_spin,
    get_unique_relative_spins,
    is_within_angle_threshold,
    quaternion_conjugate,
    quaternion_distance,
    quaternion_from_euler,
    quaternion_multiply,
    rotate_quaternion,
    rotation_matrix_to_quaternion,
)
