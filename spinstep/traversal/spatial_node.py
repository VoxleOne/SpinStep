# traversal/spatial_node.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Spatial node: a tree node enriched with distance, velocity, and timestamp."""

from __future__ import annotations

__all__ = ["SpatialNode"]

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from .node import Node

if TYPE_CHECKING:
    from ..control.state import OrientationState


class SpatialNode(Node):
    """A tree node with full spatial state: orientation, distance, velocities.

    Extends :class:`Node` with the fields needed for the SpinStep control
    model so that a single object can participate in both traversal and
    control workflows.

    All new fields are optional with backward-compatible defaults, so
    ``SpatialNode("name", [0, 0, 0, 1])`` works identically to
    ``Node("name", [0, 0, 0, 1])``.

    Args:
        name: Human-readable identifier for this node.
        orientation: Quaternion as ``[x, y, z, w]``.  Must have non-zero norm.
        children: Optional initial child nodes.
        distance: Radial distance from world origin.  Defaults to ``0.0``.
        angular_velocity: Angular velocity ``[ωx, ωy, ωz]`` in rad/s.
        radial_velocity: Radial velocity in units/s.
        timestamp: Time in seconds.

    Raises:
        ValueError: If *orientation* is not a 4-element vector, has
            near-zero norm, or *distance* is negative.

    Example::

        from spinstep import SpatialNode

        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0,
                           angular_velocity=[0, 0, 1.0])
    """

    distance: float
    angular_velocity: npt.NDArray[np.floating[Any]]
    radial_velocity: float
    timestamp: float

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        children: Optional[Sequence[Node]] = None,
        *,
        distance: float = 0.0,
        angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
        radial_velocity: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        super().__init__(name, orientation, children)

        if distance < 0:
            raise ValueError(f"distance must be non-negative, got {distance}")
        self.distance = float(distance)

        omega = np.asarray(angular_velocity, dtype=float)
        if omega.shape != (3,):
            raise ValueError(
                f"angular_velocity must have shape (3,), got {omega.shape}"
            )
        self.angular_velocity = omega
        self.radial_velocity = float(radial_velocity)
        self.timestamp = float(timestamp)

    def as_state(self) -> "OrientationState":
        """Convert this spatial node to an :class:`OrientationState`.

        Returns:
            An :class:`OrientationState` with quaternion, distance,
            angular velocity, radial velocity, and timestamp matching
            this node's fields.
        """
        from ..control.state import OrientationState

        return OrientationState(
            quaternion=self.orientation.copy(),
            distance=self.distance,
            angular_velocity=self.angular_velocity.copy(),
            radial_velocity=self.radial_velocity,
            timestamp=self.timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"SpatialNode({self.name!r}, "
            f"orientation={self.orientation.tolist()}, "
            f"d={self.distance})"
        )
