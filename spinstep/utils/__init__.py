# utils/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Utilities for quaternion math and array backend selection.

This sub-package provides:

- :func:`~.array_backend.get_array_module` — NumPy / CuPy backend selection.
- :func:`~.quaternion_math.batch_quaternion_angle` — batch angular distances.
- Quaternion helpers in :mod:`~.quaternion_utils` (conversion, distance,
  multiplication, etc.).
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
]

from .array_backend import get_array_module
from .quaternion_math import batch_quaternion_angle
from .quaternion_utils import (
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
