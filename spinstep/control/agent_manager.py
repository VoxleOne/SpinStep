# control/agent_manager.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Multi-agent coordinator for shared scene control."""

from __future__ import annotations

__all__ = ["AgentManager"]

from typing import Dict, List, Tuple

from ..math.geometry import quaternion_distance
from ..traversal.scene_graph import SceneGraph
from .agent import Agent
from .events import EventEmitter
from .state import ControlCommand


class AgentManager:
    """Coordinates multiple :class:`Agent` instances in a shared scene.

    Agents are updated in deterministic (sorted-by-name) order.  The
    manager provides proximity queries and a central event bus.

    Args:
        scene: The shared :class:`~spinstep.traversal.SceneGraph`.

    Example::

        from spinstep.control import AgentManager

        mgr = AgentManager(scene)
        mgr.add_agent("a", agent_a)
        commands = mgr.step(dt=0.01)
    """

    def __init__(self, scene: SceneGraph) -> None:
        self.scene = scene
        self.agents: Dict[str, Agent] = {}
        self.events = EventEmitter()

    def add_agent(self, name: str, agent: Agent) -> None:
        """Register an agent.

        Args:
            name: Unique agent identifier.
            agent: The :class:`Agent` instance.

        Raises:
            ValueError: If an agent with the same name exists.
        """
        if name in self.agents:
            raise ValueError(f"Agent {name!r} already registered")
        self.agents[name] = agent

    def remove_agent(self, name: str) -> None:
        """Remove a registered agent.

        Args:
            name: Agent identifier.

        Raises:
            KeyError: If the agent does not exist.
        """
        if name not in self.agents:
            raise KeyError(f"Agent {name!r} not registered")
        del self.agents[name]

    def step(
        self, dt: float, t: float = 0.0
    ) -> Dict[str, ControlCommand]:
        """Update all agents in deterministic (sorted) order.

        Args:
            dt: Time step in seconds.
            t: Current simulation time.

        Returns:
            Dictionary mapping agent names to their
            :class:`~spinstep.control.state.ControlCommand`.
        """
        commands: Dict[str, ControlCommand] = {}
        for name in sorted(self.agents):
            commands[name] = self.agents[name].update(dt, t)
        return commands

    def query_proximity(
        self, agent_name: str, radius: float
    ) -> List[str]:
        """Return names of agents within *radius* angular distance.

        Args:
            agent_name: The reference agent.
            radius: Maximum angular distance in radians.

        Returns:
            List of agent names within the radius (excluding the
            reference agent).

        Raises:
            KeyError: If the agent does not exist.
        """
        if agent_name not in self.agents:
            raise KeyError(f"Agent {agent_name!r} not registered")
        ref = self.agents[agent_name].node
        result: List[str] = []
        for name in sorted(self.agents):
            if name != agent_name:
                d = quaternion_distance(
                    ref.orientation, self.agents[name].node.orientation
                )
                if d < radius:
                    result.append(name)
        return result

    def pairwise_distances(self) -> Dict[Tuple[str, str], float]:
        """Compute angular distances between all agent pairs.

        Returns:
            Dictionary mapping ``(name_a, name_b)`` pairs to angular
            distances in radians.  Only includes pairs where
            ``name_a < name_b`` lexicographically.
        """
        names = sorted(self.agents)
        result: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = quaternion_distance(
                    self.agents[a].node.orientation,
                    self.agents[b].node.orientation,
                )
                result[(a, b)] = d
        return result

    def __repr__(self) -> str:
        return f"AgentManager({len(self.agents)} agents)"
