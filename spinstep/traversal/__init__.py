# traversal/__init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Tree traversal using quaternion orientation.

This sub-package contains the traversal classes:

- :class:`Node` — tree node with quaternion orientation
- :class:`QuaternionDepthIterator` — continuous rotation-step depth-first traversal
- :class:`DiscreteOrientationSet` — queryable set of discrete orientations
- :class:`DiscreteQuaternionIterator` — discrete rotation-step depth-first traversal
"""

__all__ = [
    "Node",
    "QuaternionDepthIterator",
    "DiscreteOrientationSet",
    "DiscreteQuaternionIterator",
]

from .node import Node
from .continuous import QuaternionDepthIterator
from .discrete import DiscreteOrientationSet
from .discrete_iterator import DiscreteQuaternionIterator
