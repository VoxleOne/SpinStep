# node.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Tree node with quaternion-based orientation."""

from __future__ import annotations

__all__ = ["Node"]

from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


class Node:
    """A tree node with a quaternion-based orientation.

    Each node stores a name, a unit quaternion orientation ``[x, y, z, w]``,
    and an optional list of child nodes.  The orientation is automatically
    normalised on construction.

    Args:
        name: Human-readable identifier for this node.
        orientation: Quaternion as ``[x, y, z, w]``.  Must have non-zero norm.
        children: Optional initial child nodes.

    Raises:
        ValueError: If *orientation* is not a 4-element vector or has
            near-zero norm.

    Attributes:
        name: Node identifier string.
        orientation: Normalised quaternion as a NumPy array of shape ``(4,)``.
        children: List of child :class:`Node` instances.

    Example::

        from spinstep import Node

        root = Node("root", [0, 0, 0, 1])
        child = Node("child", [0.2588, 0, 0, 0.9659])
        root.children.append(child)
    """

    name: str
    orientation: np.ndarray
    children: List["Node"]

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        children: Optional[Sequence["Node"]] = None,
    ) -> None:
        arr = np.array(orientation, dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"Orientation must be a quaternion [x,y,z,w], got shape {arr.shape}")
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Orientation quaternion must be non-zero")
        self.orientation = arr / norm
        self.name = name
        self.children = list(children) if children else []

    def __repr__(self) -> str:
        return f"Node({self.name!r}, orientation={self.orientation.tolist()})"
