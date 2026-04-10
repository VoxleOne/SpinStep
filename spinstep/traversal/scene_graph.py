# traversal/scene_graph.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Graph-based spatial scene with any-node observation support."""

from __future__ import annotations

__all__ = ["SceneGraph"]

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

from .node import Node
from .spatial_node import SpatialNode


class SceneGraph:
    """A container of :class:`SpatialNode` instances with adjacency relationships.

    Supports both tree and general graph topologies.  Nodes are identified
    by their :attr:`~Node.name`.  Edges can be directed or bidirectional.

    The core multi-observer capability is :meth:`observe_from`, which
    computes the relative state of every other node as seen from a
    chosen observer node.

    Example::

        from spinstep import SpatialNode
        from spinstep.traversal import SceneGraph

        scene = SceneGraph()
        scene.add_node(SpatialNode("a", [0, 0, 0, 1], distance=5.0))
        scene.add_node(SpatialNode("b", [0, 0, 0.383, 0.924], distance=7.0))
        scene.add_edge("a", "b")

        view = scene.observe_from("a")
        # → {"b": OrientationState(...)}
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, SpatialNode] = {}
        self._adj: Dict[str, Set[str]] = {}
        self._kdtree: Optional[KDTree] = None
        self._kdtree_names: Optional[List[str]] = None
        self._dirty: bool = True

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node: SpatialNode) -> None:
        """Add a spatial node to the graph.

        Args:
            node: The node to add.

        Raises:
            ValueError: If a node with the same name already exists.
        """
        if node.name in self._nodes:
            raise ValueError(f"Node {node.name!r} already exists in the graph")
        self._nodes[node.name] = node
        self._adj.setdefault(node.name, set())
        self._dirty = True

    def remove_node(self, name: str) -> None:
        """Remove a node and all its edges from the graph.

        Args:
            name: Name of the node to remove.

        Raises:
            KeyError: If the node does not exist.
        """
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not in graph")
        del self._nodes[name]
        neighbours = self._adj.pop(name, set())
        for nb in neighbours:
            self._adj.get(nb, set()).discard(name)
        self._dirty = True

    def get_node(self, name: str) -> SpatialNode:
        """Return the node with the given name.

        Args:
            name: Node identifier.

        Raises:
            KeyError: If the node does not exist.
        """
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not in graph")
        return self._nodes[name]

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        from_name: str,
        to_name: str,
        bidirectional: bool = True,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            from_name: Source node name.
            to_name: Target node name.
            bidirectional: If ``True``, add the reverse edge too.

        Raises:
            KeyError: If either node does not exist.
        """
        if from_name not in self._nodes:
            raise KeyError(f"Node {from_name!r} not in graph")
        if to_name not in self._nodes:
            raise KeyError(f"Node {to_name!r} not in graph")
        self._adj[from_name].add(to_name)
        if bidirectional:
            self._adj[to_name].add(from_name)

    def remove_edge(
        self,
        from_name: str,
        to_name: str,
        bidirectional: bool = True,
    ) -> None:
        """Remove an edge between two nodes.

        Args:
            from_name: Source node name.
            to_name: Target node name.
            bidirectional: If ``True``, also remove the reverse edge.
        """
        self._adj.get(from_name, set()).discard(to_name)
        if bidirectional:
            self._adj.get(to_name, set()).discard(from_name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def neighbors(self, name: str) -> List[SpatialNode]:
        """Return the neighbours of the named node.

        Args:
            name: Node identifier.

        Raises:
            KeyError: If the node does not exist.
        """
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not in graph")
        return [self._nodes[n] for n in sorted(self._adj.get(name, set()))]

    def nodes(self) -> List[SpatialNode]:
        """Return all nodes in the graph (sorted by name)."""
        return [self._nodes[k] for k in sorted(self._nodes)]

    def edges(self) -> List[Tuple[str, str]]:
        """Return all directed edges as ``(from_name, to_name)`` pairs."""
        result: List[Tuple[str, str]] = []
        for src in sorted(self._adj):
            for dst in sorted(self._adj[src]):
                result.append((src, dst))
        return result

    # ------------------------------------------------------------------
    # Multi-observer
    # ------------------------------------------------------------------

    def observe_from(self, node_name: str) -> Dict[str, Any]:
        """Compute all other nodes' states as seen from the given observer.

        This is the core *any-node-as-observer* capability.

        Args:
            node_name: Name of the observer node.

        Returns:
            Dictionary mapping other node names to their
            :class:`~spinstep.control.state.OrientationState` as seen
            from the observer.

        Raises:
            KeyError: If the node does not exist.
        """
        from ..control.state import compute_relative_state

        if node_name not in self._nodes:
            raise KeyError(f"Node {node_name!r} not in graph")
        observer = self._nodes[node_name]
        result: Dict[str, Any] = {}
        for name, node in self._nodes.items():
            if name != node_name:
                result[name] = compute_relative_state(observer, node)
        return result

    # ------------------------------------------------------------------
    # Spatial indexing
    # ------------------------------------------------------------------

    def _rebuild_kdtree(self) -> None:
        """Rebuild the internal KDTree from node positions."""
        from ..math.geometry import forward_vector_from_quaternion

        if not self._nodes:
            self._kdtree = None
            self._kdtree_names = None
            self._dirty = False
            return

        names = sorted(self._nodes.keys())
        positions = np.array([
            forward_vector_from_quaternion(self._nodes[n].orientation)
            * self._nodes[n].distance
            for n in names
        ])
        self._kdtree = KDTree(positions)
        self._kdtree_names = names
        self._dirty = False

    def query_nearest(
        self, observer_name: str, k: int = 1
    ) -> List[SpatialNode]:
        """Return the *k* nearest nodes to the observer.

        Args:
            observer_name: Name of the observer node.
            k: Number of nearest neighbours to return.

        Returns:
            List of nearest :class:`SpatialNode` instances (excluding the
            observer itself).

        Raises:
            KeyError: If the node does not exist.
        """
        from ..math.geometry import forward_vector_from_quaternion

        if observer_name not in self._nodes:
            raise KeyError(f"Node {observer_name!r} not in graph")
        if self._dirty:
            self._rebuild_kdtree()
        if self._kdtree is None or self._kdtree_names is None:
            return []

        obs = self._nodes[observer_name]
        obs_pos = (
            forward_vector_from_quaternion(obs.orientation) * obs.distance
        )
        # Query k+1 because the observer itself is in the tree
        _, indices = self._kdtree.query(obs_pos, k=min(k + 1, len(self._kdtree_names)))
        indices = np.atleast_1d(indices)
        result: List[SpatialNode] = []
        for idx in indices:
            name = self._kdtree_names[idx]
            if name != observer_name:
                result.append(self._nodes[name])
            if len(result) >= k:
                break
        return result

    def query_within_angle(
        self, observer_name: str, angle: float
    ) -> List[SpatialNode]:
        """Return nodes whose angular distance from the observer is less than *angle*.

        Args:
            observer_name: Name of the observer node.
            angle: Maximum angular distance in radians.

        Returns:
            List of matching :class:`SpatialNode` instances.

        Raises:
            KeyError: If the node does not exist.
        """
        from ..math.geometry import quaternion_distance

        if observer_name not in self._nodes:
            raise KeyError(f"Node {observer_name!r} not in graph")
        obs = self._nodes[observer_name]
        result: List[SpatialNode] = []
        for name, node in self._nodes.items():
            if name != observer_name:
                d = quaternion_distance(obs.orientation, node.orientation)
                if d < angle:
                    result.append(node)
        return result

    # ------------------------------------------------------------------
    # Tree conversion
    # ------------------------------------------------------------------

    @classmethod
    def from_tree(cls, root: Node) -> "SceneGraph":
        """Build a :class:`SceneGraph` from an existing tree.

        Each tree node becomes a :class:`SpatialNode` (with default
        ``distance=0.0`` if the source is a plain :class:`Node`).
        Parent→child relationships become bidirectional edges.

        Args:
            root: Root of the tree.

        Returns:
            A new :class:`SceneGraph`.
        """
        graph = cls()
        _add_tree_node(graph, root, set())
        return graph

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, name: object) -> bool:
        return name in self._nodes

    def __repr__(self) -> str:
        return (
            f"SceneGraph({len(self._nodes)} nodes, "
            f"{len(self.edges())} directed edges)"
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _add_tree_node(
    graph: SceneGraph,
    node: Node,
    visited: Set[str],
) -> None:
    """Recursively add a tree node and its children to *graph*."""
    if node.name in visited:
        return
    visited.add(node.name)

    if isinstance(node, SpatialNode):
        sn = node
    else:
        sn = SpatialNode(node.name, node.orientation)
    if sn.name not in graph:
        graph.add_node(sn)

    for child in node.children:
        _add_tree_node(graph, child, visited)
        if child.name in graph:
            graph.add_edge(node.name, child.name, bidirectional=True)
