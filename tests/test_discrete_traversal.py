# test_discrete_traversal.py — SpinStep Test Suite — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-16
# See LICENSE.txt for full terms.

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Import the modules under test
from spinstep.orientations.discrete import DiscreteOrientationSet
from spinstep.discrete_iterator import DiscreteQuaternionIterator

# Simple node class for testing
class Node:
    def __init__(self, id, orientation, children=None):
        self.id = id
        self.orientation = orientation  # [x,y,z,w] quaternion
        self.children = children or []
    
    def add_child(self, child):
        self.children.append(child)
        return child
    
    def __repr__(self):
        return f"Node({self.id})"


# ===== DiscreteOrientationSet Tests =====

class TestDiscreteOrientationSet:
    def test_initialization(self):
        # Test with valid quaternions
        quats = [
            [0, 0, 0, 1],  # Identity
            [0, 1, 0, 0],  # 180° around Y
            [0, 0, 1, 0],  # 180° around Z
        ]
        orientation_set = DiscreteOrientationSet(quats)
        assert len(orientation_set) == 3
        
        # Test normalization
        unnormalized = [
            [0, 0, 0, 2],  # Non-unit quaternion
        ]
        orientation_set = DiscreteOrientationSet(unnormalized)
        assert np.allclose(orientation_set.orientations[0], [0, 0, 0, 1])
        
        # Test error cases
        with pytest.raises(ValueError):
            DiscreteOrientationSet([[0, 0, 0]])  # Wrong shape
        
        with pytest.raises(ValueError):
            DiscreteOrientationSet([[0, 0, 0, 0]])  # Zero quaternion
    
    def test_predefined_sets(self):
        # Test cube (octahedral) set - should have 24 orientations
        cube_set = DiscreteOrientationSet.from_cube()
        assert len(cube_set) == 24
        
        # Test icosahedral set - should have 60 orientations
        icosa_set = DiscreteOrientationSet.from_icosahedron()
        assert len(icosa_set) == 60
        
        # Test custom set
        custom_quats = [[0, 0, 0, 1], [1, 0, 0, 0]]
        custom_set = DiscreteOrientationSet.from_custom(custom_quats)
        assert len(custom_set) == 2
        
        # Test sphere grid with different point counts
        sphere_set_small = DiscreteOrientationSet.from_sphere_grid(n_points=10)
        assert len(sphere_set_small) == 10
        
        sphere_set_large = DiscreteOrientationSet.from_sphere_grid(n_points=100)
        assert len(sphere_set_large) == 100
    
    def test_query_within_angle(self):
        # Create a set with known angles
        quats = [
            [0, 0, 0, 1],  # Identity
            [0, np.sin(np.pi/8), 0, np.cos(np.pi/8)],  # 45° around Y
            [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)],  # 90° around Y
        ]
        orientation_set = DiscreteOrientationSet(quats)
        
        # Query within small angle - should only find identity
        results = orientation_set.query_within_angle([0, 0, 0, 1], np.pi/16)
        assert len(results) == 1
        
        # Query within medium angle - should find identity and 45°
        results = orientation_set.query_within_angle([0, 0, 0, 1], np.pi/6)
        assert len(results) == 2
        
        # Query within large angle - should find all three
        results = orientation_set.query_within_angle([0, 0, 0, 1], np.pi/3)
        assert len(results) == 3
    
    @pytest.mark.skipif(not np.array([True]), reason="CUDA not available")
    def test_cuda_support(self):
        try:
            cuda_set = DiscreteOrientationSet([[0, 0, 0, 1]], use_cuda=True)
            # If this succeeds, basic CUDA import worked
            assert cuda_set.use_cuda == True
            
            # Test as_numpy() method for GPU->CPU transfer
            cpu_array = cuda_set.as_numpy()
            assert isinstance(cpu_array, np.ndarray)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("CUDA/CuPy not available")


# ===== DiscreteQuaternionIterator Tests =====

class TestDiscreteQuaternionIterator:
    def setup_method(self):
        # Create a simple tree structure with specific orientations
        #       root (identity orientation)
        #      /    \
        #     A      B
        #    / \    / \
        #   C   D  E   F
        #
        # Where A is 45° rotation from root around Y
        # B is 90° rotation from root around Y
        # Others have further rotations
        
        self.root = Node("root", [0, 0, 0, 1])
        
        # 45° around Y
        self.node_a = self.root.add_child(
            Node("A", [0, np.sin(np.pi/8), 0, np.cos(np.pi/8)])
        )
        
        # 90° around Y
        self.node_b = self.root.add_child(
            Node("B", [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)])
        )
        
        # A's children - small variations
        self.node_c = self.node_a.add_child(
            Node("C", [0, np.sin(np.pi/8+0.1), 0, np.cos(np.pi/8+0.1)])
        )
        
        self.node_d = self.node_a.add_child(
            Node("D", [0, np.sin(np.pi/8-0.1), 0, np.cos(np.pi/8-0.1)])
        )
        
        # B's children - small variations
        self.node_e = self.node_b.add_child(
            Node("E", [0, np.sin(np.pi/4+0.1), 0, np.cos(np.pi/4+0.1)])
        )
        
        self.node_f = self.node_b.add_child(
            Node("F", [0, np.sin(np.pi/4-0.1), 0, np.cos(np.pi/4-0.1)])
        )
        
        # Create orientation set with basic steps
        steps = [
            [0, 0, 0, 1],  # Identity (no rotation)
            [0, np.sin(np.pi/8), 0, np.cos(np.pi/8)],  # 45° step around Y
            [0, np.sin(-np.pi/8), 0, np.cos(np.pi/8)],  # -45° step around Y
        ]
        self.orientation_set = DiscreteOrientationSet(steps)
    
    def test_iterator_creation(self):
        # Test basic creation
        iterator = DiscreteQuaternionIterator(
            self.root, 
            self.orientation_set,
            angle_threshold=np.pi/6
        )
        assert iterator is not None
        
        # Test error with invalid node
        class InvalidNode:
            pass
            
        with pytest.raises(AttributeError):
            DiscreteQuaternionIterator(
                InvalidNode(), 
                self.orientation_set
            )
    
    def test_traversal(self):
        # With large angle threshold, should visit all nodes
        iterator = DiscreteQuaternionIterator(
            self.root, 
            self.orientation_set,
            angle_threshold=np.pi/2,  # Very permissive
            max_depth=3
        )
        
        visited_nodes = list(iterator)
        visited_ids = [node.id for node in visited_nodes]
        
        # Should have visited all nodes in some order
        for node_id in ["root", "A", "B", "C", "D", "E", "F"]:
            assert node_id in visited_ids
        
        # With tiny threshold, should only visit root
        iterator = DiscreteQuaternionIterator(
            self.root, 
            self.orientation_set,
            angle_threshold=np.pi/64,  # Very restrictive
            max_depth=3
        )
        
        visited_nodes = list(iterator)
        assert len(visited_nodes) == 1
        assert visited_nodes[0].id == "root"
    
    def test_max_depth(self):
        # With max_depth=1, should only visit root and its direct children
        iterator = DiscreteQuaternionIterator(
            self.root, 
            self.orientation_set,
            angle_threshold=np.pi/2,  # Very permissive
            max_depth=1
        )
        
        visited_nodes = list(iterator)
        visited_ids = [node.id for node in visited_nodes]
        
        # Should have visited only root and direct children
        for node_id in ["root", "A", "B"]:
            assert node_id in visited_ids
        
        # Shouldn't have visited grandchildren
        for node_id in ["C", "D", "E", "F"]:
            assert node_id not in visited_ids


# ===== Integration Tests =====

def test_full_pipeline():
    """Test the complete pipeline from orientation set creation to traversal"""
    
    # Create a custom orientation set
    custom_steps = [
        [0, 0, 0, 1],  # Identity
        [1, 0, 0, 0],  # 180° around X
        [0, 1, 0, 0],  # 180° around Y
        [0, 0, 1, 0],  # 180° around Z
    ]
    
    orientation_set = DiscreteOrientationSet.from_custom(custom_steps)
    
    # Create a simple tree
    root = Node("root", [0, 0, 0, 1])
    node_a = root.add_child(Node("A", [1, 0, 0, 0]))
    node_b = root.add_child(Node("B", [0, 1, 0, 0]))
    
    # Create iterator and traverse
    iterator = DiscreteQuaternionIterator(
        root,
        orientation_set,
        angle_threshold=np.pi/4
    )
    
    # Check that we can iterate and get nodes
    visited = []
    for node in iterator:
        visited.append(node.id)
    
    # We expect to visit all nodes since our steps include direct rotations
    # to each child's orientation
    assert "root" in visited
    assert "A" in visited
    assert "B" in visited


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
