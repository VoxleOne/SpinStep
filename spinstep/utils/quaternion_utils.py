# quaternion_utils.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Backward-compatible re-exports of quaternion utilities.

All quaternion functions have moved to :mod:`spinstep.math`.  This module
re-exports them so that existing ``from spinstep.utils.quaternion_utils import …``
statements continue to work.

.. deprecated::
    Import from :mod:`spinstep.math` instead.
"""

from __future__ import annotations

__all__ = [
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

# Re-export from canonical locations in spinstep.math
from spinstep.math.core import quaternion_conjugate, quaternion_multiply
from spinstep.math.geometry import (
    angle_between_directions,
    direction_to_quaternion,
    forward_vector_from_quaternion,
    is_within_angle_threshold,
    quaternion_distance,
    rotate_quaternion,
)
from spinstep.math.conversions import (
    quaternion_from_euler,
    rotation_matrix_to_quaternion,
)
from spinstep.math.analysis import (
    get_relative_spin,
    get_unique_relative_spins,
)
