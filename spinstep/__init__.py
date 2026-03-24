# __init__.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""SpinStep: A quaternion-driven traversal framework.

Provides quaternion-based tree traversal for orientation-aware structures,
supporting both continuous and discrete rotation stepping.

Example usage::

    from spinstep import Node, QuaternionDepthIterator

    root = Node("root", [0, 0, 0, 1])
    child = Node("child", [0, 0, 0.1, 0.995])
    root.children.append(child)

    step = [0, 0, 0.05, 0.9987]  # small rotation about Z
    for node in QuaternionDepthIterator(root, step):
        print(node.name)
"""

__version__ = "0.1.0"

from .node import Node
from .traversal import QuaternionDepthIterator
from .discrete import DiscreteOrientationSet
from .discrete_iterator import DiscreteQuaternionIterator

__all__ = [
    "Node",
    "QuaternionDepthIterator",
    "DiscreteOrientationSet",
    "DiscreteQuaternionIterator",
]
