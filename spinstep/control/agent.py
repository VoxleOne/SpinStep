# control/agent.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Agent: a spatial node paired with a controller and optional trajectory."""

from __future__ import annotations

__all__ = ["Agent"]

from typing import Optional

from ..traversal.spatial_node import SpatialNode
from .controllers import OrientationController
from .events import EventEmitter
from .frames import ReferenceFrame
from .state import ControlCommand, OrientationState, compute_relative_state, integrate_state
from .trajectory import TrajectoryController


class Agent:
    """A controlled spatial entity with its own reference frame.

    Wraps a :class:`~spinstep.traversal.SpatialNode`,
    an :class:`OrientationController`, and an optional
    :class:`TrajectoryController`.  Each agent sees the world from its
    own perspective.

    Args:
        node: The agent's spatial node.
        controller: The orientation controller.
        trajectory: Optional trajectory controller.
        target_state: Optional target :class:`OrientationState` for the
            controller.  Required if *trajectory* is not provided.

    Example::

        from spinstep import SpatialNode, ProportionalOrientationController
        from spinstep.control import Agent

        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
    """

    def __init__(
        self,
        node: SpatialNode,
        controller: OrientationController,
        trajectory: Optional[TrajectoryController] = None,
        target_state: Optional[OrientationState] = None,
    ) -> None:
        self.node = node
        self.controller = controller
        self.trajectory = trajectory
        self.target_state = target_state
        self.events = EventEmitter()

    @property
    def frame(self) -> ReferenceFrame:
        """The agent's current reference frame (derived from its node)."""
        return ReferenceFrame.from_node(self.node)

    @property
    def state(self) -> OrientationState:
        """The agent's current :class:`OrientationState`."""
        return self.node.as_state()

    def observe(self, target: SpatialNode) -> OrientationState:
        """Compute *target*'s state as seen from this agent.

        Args:
            target: The target spatial node.

        Returns:
            The target's :class:`OrientationState` in this agent's frame.
        """
        return compute_relative_state(self.node, target)

    def update(
        self,
        dt: float,
        t: float = 0.0,
    ) -> ControlCommand:
        """Step the agent's controller and integrate the node state.

        If a trajectory controller is set it takes precedence.
        Otherwise the agent drives toward :attr:`target_state`.

        Args:
            dt: Time step in seconds.
            t: Current simulation time (used by trajectory controller).

        Returns:
            The :class:`ControlCommand` produced this step.
        """
        if self.trajectory is not None:
            cmd = self.trajectory.update(
                self.node.orientation,
                t=t,
                dt=dt,
                current_distance=self.node.distance,
            )
            if self.trajectory.is_complete:
                self.events.emit(
                    "trajectory_complete", agent=self, time=t
                )
        elif self.target_state is not None:
            cmd = self.controller.update(
                self.node.orientation,
                self.target_state.quaternion,
                dt,
                current_distance=self.node.distance,
                target_distance=self.target_state.distance,
            )
        else:
            cmd = ControlCommand()

        # Integrate node state
        old_state = self.node.as_state()
        new_state = integrate_state(
            OrientationState(
                quaternion=self.node.orientation,
                distance=self.node.distance,
                angular_velocity=cmd.angular_velocity,
                radial_velocity=cmd.radial_velocity,
                timestamp=self.node.timestamp,
            ),
            dt,
        )
        self.node.orientation = new_state.quaternion
        self.node.distance = new_state.distance
        self.node.timestamp = new_state.timestamp
        self.node.angular_velocity = cmd.angular_velocity.copy()
        self.node.radial_velocity = cmd.radial_velocity

        self.events.emit(
            "state_change", agent=self, old_state=old_state, new_state=new_state
        )
        return cmd

    def reset(self) -> None:
        """Reset the controller and trajectory state."""
        self.controller.reset()
        if self.trajectory is not None:
            self.trajectory.reset()
        self.events.clear()

    def __repr__(self) -> str:
        return f"Agent({self.node.name!r}, d={self.node.distance})"
