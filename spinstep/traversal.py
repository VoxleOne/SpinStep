# traversal.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Continuous quaternion-driven depth-first tree traversal."""

from __future__ import annotations

__all__ = ["QuaternionDepthIterator"]

from typing import Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from .node import Node


class QuaternionDepthIterator:
    """Depth-first tree iterator driven by a continuous quaternion rotation step.

    At each visited node the iterator applies *rotation_step_quat* to the
    current orientation state.  Children whose orientation is within
    *angle_threshold* of the rotated state are pushed onto the traversal
    stack.

    Parameters
    ----------
    start_node:
        Root node of the tree to traverse.
    rotation_step_quat:
        Quaternion ``[x, y, z, w]`` applied at every step.
    angle_threshold:
        Maximum angular distance (radians) between the rotated state and a
        child's orientation for the child to be visited.  When *None* the
        threshold is set to 30 % of the step angle (minimum 1°).
    """

    DEFAULT_DYNAMIC_THRESHOLD_FACTOR: float = 0.3

    def __init__(
        self,
        start_node: Node,
        rotation_step_quat: ArrayLike,
        angle_threshold: Optional[float] = None,
    ) -> None:
        self.rotation_step: R = R.from_quat(rotation_step_quat)

        if angle_threshold is None:
            step_angle_rad: float = self.rotation_step.magnitude()
            if step_angle_rad < 1e-7:
                self.angle_threshold: float = np.deg2rad(1.0)
            else:
                self.angle_threshold = step_angle_rad * self.DEFAULT_DYNAMIC_THRESHOLD_FACTOR
        else:
            self.angle_threshold = angle_threshold

        self.stack: List[Tuple[Node, R]] = [
            (start_node, R.from_quat(start_node.orientation))
        ]

    def __iter__(self) -> Iterator[Node]:
        return self

    def __next__(self) -> Node:
        while self.stack:
            node, state = self.stack.pop()
            rotated_state = state * self.rotation_step

            for child in node.children:
                try:
                    if np.allclose(child.orientation, [0, 0, 0, 0]):
                        continue

                    target_orientation = R.from_quat(child.orientation)
                    angle_difference_rotation = rotated_state.inv() * target_orientation
                    angle = angle_difference_rotation.magnitude()
                except ValueError:
                    continue

                if angle < self.angle_threshold:
                    self.stack.append((child, target_orientation))

            return node
        raise StopIteration
