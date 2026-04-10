# control/frames.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Reference frames for observer-relative coordinate transforms.

A :class:`ReferenceFrame` encapsulates an observer position (quaternion +
distance from world origin) and provides methods to convert states between
the world frame and the observer's local frame.
"""

from __future__ import annotations

__all__ = [
    "ReferenceFrame",
    "rebase_state",
]

from typing import TYPE_CHECKING

import numpy as np

from ..math.core import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_normalize,
)

if TYPE_CHECKING:
    from ..math.analysis import SpatialNodeProtocol

from .state import OrientationState


class ReferenceFrame:
    """A reference frame defined by an origin orientation and distance.

    The frame's origin sits at a position on the observer-centered
    sphere determined by ``origin_quaternion`` (direction) and
    ``origin_distance`` (radial layer).

    Transform operations convert :class:`OrientationState` instances
    between this frame's local coordinates and world coordinates.

    Args:
        origin_quaternion: Orientation of the frame's origin in world
            coordinates ``[x, y, z, w]``.
        origin_distance: Radial distance of the frame's origin from the
            world origin.

    Example::

        from spinstep.control.frames import ReferenceFrame

        frame = ReferenceFrame.world()
        local = frame.to_local(state)
    """

    def __init__(
        self,
        origin_quaternion: np.ndarray,
        origin_distance: float = 0.0,
    ) -> None:
        q = np.asarray(origin_quaternion, dtype=float)
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            raise ValueError("origin_quaternion must be non-zero")
        self.origin_quaternion: np.ndarray = q / norm
        self.origin_distance: float = float(origin_distance)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def world(cls) -> "ReferenceFrame":
        """Return the identity world frame (origin at [0,0,0,1], distance 0)."""
        return cls(np.array([0.0, 0.0, 0.0, 1.0]), 0.0)

    @classmethod
    def from_node(cls, node: "SpatialNodeProtocol") -> "ReferenceFrame":
        """Create a reference frame from a spatial node.

        Args:
            node: Node satisfying
                :class:`~spinstep.math.analysis.SpatialNodeProtocol`.

        Returns:
            A :class:`ReferenceFrame` centred on the node.
        """
        return cls(node.orientation.copy(), node.distance)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def to_local(self, state: OrientationState) -> OrientationState:
        """Transform a world-frame state into this frame's local coordinates.

        The local quaternion is ``conjugate(q_frame) * q_world``.  The
        local distance is the Euclidean distance between the frame origin
        and the state position in Cartesian space.

        Args:
            state: State expressed in world coordinates.

        Returns:
            The same state expressed in this frame's local coordinates.
        """
        q_local = quaternion_normalize(
            quaternion_multiply(
                quaternion_conjugate(self.origin_quaternion),
                state.quaternion,
            )
        )

        # Cartesian distance
        from ..math.geometry import forward_vector_from_quaternion

        frame_pos = (
            forward_vector_from_quaternion(self.origin_quaternion)
            * self.origin_distance
        )
        state_pos = (
            forward_vector_from_quaternion(state.quaternion) * state.distance
        )
        local_distance = float(np.linalg.norm(state_pos - frame_pos))

        return OrientationState(
            quaternion=q_local,
            distance=local_distance,
            angular_velocity=state.angular_velocity.copy(),
            radial_velocity=state.radial_velocity,
            timestamp=state.timestamp,
        )

    def to_world(self, state: OrientationState) -> OrientationState:
        """Transform a local-frame state back to world coordinates.

        The world quaternion is ``q_frame * q_local``.

        Args:
            state: State expressed in this frame's local coordinates.

        Returns:
            The same state expressed in world coordinates.
        """
        q_world = quaternion_normalize(
            quaternion_multiply(self.origin_quaternion, state.quaternion)
        )

        # Reconstruct world distance from local offset
        from ..math.geometry import forward_vector_from_quaternion

        frame_pos = (
            forward_vector_from_quaternion(self.origin_quaternion)
            * self.origin_distance
        )
        local_dir = forward_vector_from_quaternion(q_world)
        # Place the state at world distance = |frame_pos + local_dir * state.distance|
        # approximated as the radial distance of the reconstructed position
        world_pos = frame_pos + local_dir * state.distance
        world_distance = float(np.linalg.norm(world_pos))

        return OrientationState(
            quaternion=q_world,
            distance=world_distance,
            angular_velocity=state.angular_velocity.copy(),
            radial_velocity=state.radial_velocity,
            timestamp=state.timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"ReferenceFrame(q={self.origin_quaternion.tolist()}, "
            f"d={self.origin_distance})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReferenceFrame):
            return NotImplemented
        return (
            np.allclose(self.origin_quaternion, other.origin_quaternion)
            and abs(self.origin_distance - other.origin_distance) < 1e-8
        )


def rebase_state(
    state: OrientationState,
    from_frame: ReferenceFrame,
    to_frame: ReferenceFrame,
) -> OrientationState:
    """Transform a state from one frame to another.

    Equivalent to ``to_frame.to_local(from_frame.to_world(state))``.

    Args:
        state: The state in *from_frame*'s local coordinates.
        from_frame: The source reference frame.
        to_frame: The destination reference frame.

    Returns:
        The state expressed in *to_frame*'s local coordinates.

    Example::

        from spinstep.control.frames import ReferenceFrame, rebase_state

        frame_a = ReferenceFrame(q_a, d_a)
        frame_b = ReferenceFrame(q_b, d_b)
        state_in_b = rebase_state(state_in_a, frame_a, frame_b)
    """
    world_state = from_frame.to_world(state)
    return to_frame.to_local(world_state)
