# test_scene_graph.py — SpinStep Test Suite — MIT License
# Phase 3 tests: SceneGraph, graph iterators, KDTree migration

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.traversal.node import Node
from spinstep.traversal.spatial_node import SpatialNode
from spinstep.traversal.scene_graph import SceneGraph
from spinstep.traversal.graph_iterators import (
    BreadthFirstIterator,
    GraphQuaternionIterator,
)
from spinstep.control.state import OrientationState


# ===== SceneGraph construction =====


class TestSceneGraphNodes:
    def test_add_node(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        assert "a" in g
        assert len(g) == 1

    def test_add_duplicate_raises(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(SpatialNode("a", [1, 0, 0, 0]))

    def test_remove_node(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.remove_node("a")
        assert "a" not in g
        assert len(g) == 0

    def test_remove_nonexistent_raises(self):
        g = SceneGraph()
        with pytest.raises(KeyError):
            g.remove_node("missing")

    def test_get_node(self):
        g = SceneGraph()
        n = SpatialNode("a", [0, 0, 0, 1], distance=5.0)
        g.add_node(n)
        assert g.get_node("a") is n

    def test_get_nonexistent_raises(self):
        g = SceneGraph()
        with pytest.raises(KeyError):
            g.get_node("missing")

    def test_nodes_sorted_by_name(self):
        g = SceneGraph()
        g.add_node(SpatialNode("c", [0, 0, 0, 1]))
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        names = [n.name for n in g.nodes()]
        assert names == ["a", "b", "c"]


class TestSceneGraphEdges:
    def test_add_bidirectional_edge(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        assert "b" in [n.name for n in g.neighbors("a")]
        assert "a" in [n.name for n in g.neighbors("b")]

    def test_add_directed_edge(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_edge("a", "b", bidirectional=False)
        assert "b" in [n.name for n in g.neighbors("a")]
        assert "a" not in [n.name for n in g.neighbors("b")]

    def test_edge_missing_node_raises(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        with pytest.raises(KeyError):
            g.add_edge("a", "missing")

    def test_remove_edge(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        g.remove_edge("a", "b")
        assert "b" not in [n.name for n in g.neighbors("a")]
        assert "a" not in [n.name for n in g.neighbors("b")]

    def test_remove_node_removes_edges(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        g.remove_node("b")
        assert len(g.neighbors("a")) == 0

    def test_edges_list(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        edges = g.edges()
        assert ("a", "b") in edges
        assert ("b", "a") in edges


# ===== from_tree =====


class TestFromTree:
    def test_preserves_structure(self):
        root = Node("root", [0, 0, 0, 1])
        child_a = root.add_child(Node("a", [0.383, 0, 0, 0.924]))
        child_b = root.add_child(Node("b", [0, 0.383, 0, 0.924]))
        g = SceneGraph.from_tree(root)
        assert len(g) == 3
        assert "root" in g and "a" in g and "b" in g
        # root should be neighbour of a and b
        root_neighbours = {n.name for n in g.neighbors("root")}
        assert "a" in root_neighbours
        assert "b" in root_neighbours

    def test_converts_plain_nodes(self):
        root = Node("root", [0, 0, 0, 1])
        g = SceneGraph.from_tree(root)
        node = g.get_node("root")
        assert isinstance(node, SpatialNode)


# ===== observe_from =====


class TestObserveFrom:
    def test_identity_observer(self):
        g = SceneGraph()
        g.add_node(SpatialNode("obs", [0, 0, 0, 1], distance=0.0))
        g.add_node(SpatialNode("tgt", [0, 0, 0, 1], distance=5.0))
        view = g.observe_from("obs")
        assert "tgt" in view
        assert "obs" not in view  # observer excluded
        assert isinstance(view["tgt"], OrientationState)

    def test_known_rotation(self):
        q_90z = R.from_euler("z", 90, degrees=True).as_quat()
        g = SceneGraph()
        g.add_node(SpatialNode("obs", [0, 0, 0, 1]))
        g.add_node(SpatialNode("tgt", q_90z))
        view = g.observe_from("obs")
        angle = R.from_quat(view["tgt"].quaternion).magnitude()
        assert angle == pytest.approx(np.pi / 2, abs=0.05)

    def test_symmetry(self):
        """A→B distance should equal B→A distance."""
        q = R.from_euler("z", 45, degrees=True).as_quat()
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1], distance=3.0))
        g.add_node(SpatialNode("b", q, distance=7.0))
        view_a = g.observe_from("a")
        view_b = g.observe_from("b")
        assert view_a["b"].distance == pytest.approx(view_b["a"].distance, abs=0.1)

    def test_nonexistent_observer_raises(self):
        g = SceneGraph()
        with pytest.raises(KeyError):
            g.observe_from("missing")


# ===== Spatial indexing =====


class TestSpatialIndexing:
    def test_query_nearest(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1], distance=1.0))
        g.add_node(SpatialNode("b", [0, 0, 0, 1], distance=2.0))
        g.add_node(SpatialNode("c", [0, 0, 0, 1], distance=100.0))
        nearest = g.query_nearest("a", k=1)
        assert len(nearest) == 1
        assert nearest[0].name == "b"

    def test_query_within_angle(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        q_small = R.from_euler("z", 5, degrees=True).as_quat()
        q_large = R.from_euler("z", 90, degrees=True).as_quat()
        g.add_node(SpatialNode("close", q_small))
        g.add_node(SpatialNode("far", q_large))
        result = g.query_within_angle("a", np.deg2rad(10))
        names = [n.name for n in result]
        assert "close" in names
        assert "far" not in names


# ===== BreadthFirstIterator =====


class TestBreadthFirstIterator:
    def test_visits_all_connected(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_node(SpatialNode("c", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        visited = [n.name for n in BreadthFirstIterator(g, "a")]
        assert set(visited) == {"a", "b", "c"}

    def test_disconnected_component(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        g.add_node(SpatialNode("c", [0, 0, 0, 1]))  # disconnected
        visited = [n.name for n in BreadthFirstIterator(g, "a")]
        assert "c" not in visited

    def test_handles_cycle(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", [0, 0, 0, 1]))
        g.add_node(SpatialNode("c", [0, 0, 0, 1]))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")  # cycle
        visited = [n.name for n in BreadthFirstIterator(g, "a")]
        assert len(visited) == 3  # no infinite loop

    def test_nonexistent_start_raises(self):
        g = SceneGraph()
        with pytest.raises(KeyError):
            BreadthFirstIterator(g, "missing")


# ===== GraphQuaternionIterator =====


class TestGraphQuaternionIterator:
    def test_visits_reachable(self):
        q_small = R.from_euler("z", 15, degrees=True).as_quat()
        g = SceneGraph()
        g.add_node(SpatialNode("root", [0, 0, 0, 1]))
        g.add_node(SpatialNode("close", q_small))
        g.add_edge("root", "close")
        visited = [
            n.name
            for n in GraphQuaternionIterator(
                g, "root", q_small, angle_threshold=np.pi / 4
            )
        ]
        assert "root" in visited

    def test_respects_angle_threshold(self):
        q_far = R.from_euler("z", 90, degrees=True).as_quat()
        q_step = R.from_euler("z", 5, degrees=True).as_quat()
        g = SceneGraph()
        g.add_node(SpatialNode("root", [0, 0, 0, 1]))
        g.add_node(SpatialNode("far", q_far))
        g.add_edge("root", "far")
        visited = [
            n.name
            for n in GraphQuaternionIterator(
                g, "root", q_step, angle_threshold=np.deg2rad(10)
            )
        ]
        # "far" should not be visited because it's 90° away
        assert "far" not in visited

    def test_handles_cycle(self):
        q = R.from_euler("z", 10, degrees=True).as_quat()
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        g.add_node(SpatialNode("b", q))
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        # Should not infinite loop
        visited = list(
            GraphQuaternionIterator(g, "a", q, angle_threshold=np.pi)
        )
        assert len(visited) <= 2

    def test_nonexistent_start_raises(self):
        g = SceneGraph()
        with pytest.raises(KeyError):
            GraphQuaternionIterator(g, "missing", [0, 0, 0, 1])


# ===== SceneGraph repr =====


class TestSceneGraphRepr:
    def test_repr(self):
        g = SceneGraph()
        g.add_node(SpatialNode("a", [0, 0, 0, 1]))
        r = repr(g)
        assert "SceneGraph" in r
        assert "1 nodes" in r


# ===== Large graph performance =====


class TestLargeGraph:
    def test_1k_nodes_bfs(self):
        """BFS over 1K-node linear chain completes quickly."""
        g = SceneGraph()
        for i in range(1000):
            q = R.from_euler("z", i * 0.1, degrees=True).as_quat()
            g.add_node(SpatialNode(f"n{i}", q))
        for i in range(999):
            g.add_edge(f"n{i}", f"n{i+1}")
        visited = list(BreadthFirstIterator(g, "n0"))
        assert len(visited) == 1000
