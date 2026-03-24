# discrete_iterator.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Discrete quaternion-driven depth-first tree traversal."""

from __future__ import annotations

__all__ = ["DiscreteQuaternionIterator"]

from typing import Iterator, List, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .discrete import DiscreteOrientationSet
from .node import Node


class DiscreteQuaternionIterator:
    """Depth-first tree iterator using a discrete set of orientation steps.

    For each visited node every quaternion in *orientation_set* is tried as a
    potential rotation step.  Children whose orientation is within
    *angle_threshold* of any resulting candidate orientation are pushed onto
    the traversal stack.

    Args:
        start_node: Root node of the tree.
        orientation_set: :class:`DiscreteOrientationSet` providing the
            candidate rotation steps.
        angle_threshold: Maximum angular distance (radians) for a child to
            be considered reachable.  Defaults to π/8 (22.5°).
        max_depth: Maximum traversal depth.  Defaults to 100.

    Raises:
        AttributeError: If *start_node* lacks ``.orientation`` or
            ``.children`` attributes.

    Example::

        import numpy as np
        from spinstep import Node, DiscreteOrientationSet, DiscreteQuaternionIterator

        root = Node("root", [0, 0, 0, 1], [
            Node("child", [0, 0, 0.3827, 0.9239])
        ])
        dos = DiscreteOrientationSet.from_cube()
        for node in DiscreteQuaternionIterator(root, dos, angle_threshold=np.pi / 4):
            print(node.name)
    """

    def __init__(
        self,
        start_node: Node,
        orientation_set: DiscreteOrientationSet,
        angle_threshold: float = np.pi / 8,
        max_depth: int = 100,
    ) -> None:
        self.orientation_set = orientation_set
        self.angle_threshold = angle_threshold
        self.max_depth = max_depth
        if not hasattr(start_node, "orientation") or not hasattr(start_node, "children"):
            raise AttributeError("Node must have .orientation and .children")
        self.stack: List[Tuple[Node, R, int]] = [
            (start_node, R.from_quat(start_node.orientation), 0)
        ]
        self._visited: Set[int] = set()

    def __iter__(self) -> Iterator[Node]:
        return self

    def __next__(self) -> Node:
        if not self.stack:
            raise StopIteration

        while self.stack:
            node, current_node_world_orientation, depth = self.stack.pop()

            if depth > self.max_depth:
                continue

            node_id = id(node)
            if node_id in self._visited:
                continue
            self._visited.add(node_id)

            child_potential_depth = depth + 1
            if child_potential_depth <= self.max_depth:
                if hasattr(node, "children"):
                    for step_quat_from_set in self.orientation_set.orientations:
                        potential_next_world_orientation = (
                            current_node_world_orientation * R.from_quat(step_quat_from_set)
                        )

                        for child_node_obj in getattr(node, "children", []):
                            if id(child_node_obj) not in self._visited:
                                child_actual_world_orientation = R.from_quat(
                                    child_node_obj.orientation
                                )

                                angle_rad = (
                                    potential_next_world_orientation.inv()
                                    * child_actual_world_orientation
                                ).magnitude()

                                if angle_rad < self.angle_threshold:
                                    self.stack.append(
                                        (
                                            child_node_obj,
                                            child_actual_world_orientation,
                                            child_potential_depth,
                                        )
                                    )

            return node

        raise StopIteration
