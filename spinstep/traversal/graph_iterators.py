# traversal/graph_iterators.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Graph traversal iterators: BFS and quaternion-driven graph traversal."""

from __future__ import annotations

__all__ = [
    "BreadthFirstIterator",
    "GraphQuaternionIterator",
]

from collections import deque
from typing import Iterator, Optional, Set

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from .scene_graph import SceneGraph
from .spatial_node import SpatialNode


class BreadthFirstIterator:
    """Breadth-first graph traversal yielding :class:`SpatialNode` instances.

    Handles cycles via visited-node tracking.

    Args:
        graph: The :class:`SceneGraph` to traverse.
        start: Name of the start node.

    Raises:
        KeyError: If *start* is not in the graph.

    Example::

        from spinstep.traversal import SceneGraph, BreadthFirstIterator

        graph = SceneGraph()
        # ... add nodes and edges ...
        for node in BreadthFirstIterator(graph, "root"):
            print(node.name)
    """

    def __init__(self, graph: SceneGraph, start: str) -> None:
        if start not in graph:
            raise KeyError(f"Node {start!r} not in graph")
        self._graph = graph
        self._queue: deque[str] = deque([start])
        self._visited: Set[str] = {start}

    def __iter__(self) -> Iterator[SpatialNode]:
        return self

    def __next__(self) -> SpatialNode:
        while self._queue:
            name = self._queue.popleft()
            node = self._graph.get_node(name)
            for neighbour in self._graph.neighbors(name):
                if neighbour.name not in self._visited:
                    self._visited.add(neighbour.name)
                    self._queue.append(neighbour.name)
            return node
        raise StopIteration


class GraphQuaternionIterator:
    """Quaternion-driven graph traversal, analogous to :class:`QuaternionDepthIterator`.

    At each visited node the iterator applies *rotation_step_quat* to the
    current orientation.  Neighbours whose orientation is within
    *angle_threshold* of the rotated state are enqueued.  Handles cycles
    via visited-node tracking.

    Args:
        graph: The :class:`SceneGraph` to traverse.
        start: Name of the start node.
        rotation_step_quat: Quaternion ``[x, y, z, w]`` applied at every step.
        angle_threshold: Maximum angular distance (radians) for a neighbour
            to be visited.  Defaults to π/8 (22.5°).

    Raises:
        KeyError: If *start* is not in the graph.

    Example::

        from spinstep.traversal import SceneGraph, GraphQuaternionIterator

        graph = SceneGraph()
        # ... add nodes and edges ...
        for node in GraphQuaternionIterator(graph, "root", [0.259, 0, 0, 0.966]):
            print(node.name)
    """

    def __init__(
        self,
        graph: SceneGraph,
        start: str,
        rotation_step_quat: ArrayLike,
        angle_threshold: Optional[float] = None,
    ) -> None:
        if start not in graph:
            raise KeyError(f"Node {start!r} not in graph")
        self._graph = graph
        self.rotation_step: R = R.from_quat(rotation_step_quat)

        if angle_threshold is None:
            step_angle_rad: float = self.rotation_step.magnitude()
            if step_angle_rad < 1e-7:
                self.angle_threshold: float = np.deg2rad(1.0)
            else:
                self.angle_threshold = step_angle_rad * 0.3
        else:
            self.angle_threshold = angle_threshold

        start_node = graph.get_node(start)
        self._stack = [(start_node, R.from_quat(start_node.orientation))]
        self._visited: Set[str] = set()

    def __iter__(self) -> Iterator[SpatialNode]:
        return self

    def __next__(self) -> SpatialNode:
        while self._stack:
            node, state = self._stack.pop()

            if node.name in self._visited:
                continue
            self._visited.add(node.name)

            rotated_state = state * self.rotation_step

            for neighbour in self._graph.neighbors(node.name):
                if neighbour.name not in self._visited:
                    try:
                        if np.allclose(neighbour.orientation, [0, 0, 0, 0]):
                            continue
                        target_orientation = R.from_quat(neighbour.orientation)
                        angle = (
                            rotated_state.inv() * target_orientation
                        ).magnitude()
                    except ValueError:
                        continue

                    if angle < self.angle_threshold:
                        self._stack.append((neighbour, target_orientation))

            return node
        raise StopIteration
