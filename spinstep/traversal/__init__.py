# traversal/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Quaternion-based tree and graph traversal.

This sub-package contains the traversal classes:

- :class:`Node` — tree node with quaternion orientation
- :class:`SpatialNode` — enriched node with distance, velocity, and timestamp
- :class:`QuaternionDepthIterator` — continuous rotation-step depth-first traversal
- :class:`DiscreteOrientationSet` — queryable set of discrete orientations
- :class:`DiscreteQuaternionIterator` — discrete rotation-step depth-first traversal
- :class:`SceneGraph` — graph-based spatial scene with any-node observation
- :class:`BreadthFirstIterator` — BFS graph traversal
- :class:`GraphQuaternionIterator` — quaternion-driven graph traversal
"""

__all__ = [
    "Node",
    "SpatialNode",
    "QuaternionDepthIterator",
    "DiscreteOrientationSet",
    "DiscreteQuaternionIterator",
    "SceneGraph",
    "BreadthFirstIterator",
    "GraphQuaternionIterator",
]

from .node import Node
from .spatial_node import SpatialNode
from .continuous import QuaternionDepthIterator
from .discrete import DiscreteOrientationSet
from .discrete_iterator import DiscreteQuaternionIterator
from .scene_graph import SceneGraph
from .graph_iterators import BreadthFirstIterator, GraphQuaternionIterator
