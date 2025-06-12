# tests/test_orientations.py
import numpy as np
import pytest

# Import functions to be tested
from spinstep.orientations import (
    _generate_fibonacci_sphere_point,
    get_number_of_nodes_at_tier,
    get_discrete_node_orientation,
    _TIER_TO_N_POINTS_MAP # For direct comparison in tests
)
from spinstep.utils.quaternion_utils import rotation_matrix_to_quaternion # If needed for checks

# Helper function to get local Z-axis from orientation quaternion
def get_local_z_from_quaternion(q_xyzw):
    qx, qy, qz, qw = q_xyzw
    # Z-axis of the rotation matrix from q: [2(qxqz+qyqw), 2(qyqz-qxqw), 1-2(qx^2+qy^2)]
    return np.array([
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx * qx + qy * qy)
    ])

# --- Tests for _generate_fibonacci_sphere_point ---

def test_fibonacci_point_is_normalized():
    """Check if generated Fibonacci points are on the unit sphere."""
    point = _generate_fibonacci_sphere_point(index=0, num_points=12)
    assert np.isclose(np.linalg.norm(point), 1.0), "Fibonacci point should be normalized."

@pytest.mark.parametrize("num_points", [1, 12, 48])
@pytest.mark.parametrize("index_offset", [0, 1]) # Test first and second point if num_points > 1
def test_fibonacci_point_valid_output(num_points, index_offset):
    """Test basic valid output for Fibonacci points."""
    if index_offset >= num_points and num_points > 0 : # Skip if index_offset makes index invalid
        pytest.skip(f"Index offset {index_offset} too large for {num_points} points.")
    
    index = index_offset
    if num_points == 0 and index == 0: # Special case if we allow num_points=0 (current code doesn't)
        # Current _generate_fibonacci_sphere_point raises ValueError for index 0, num_points 0
        # due to division by zero or out of bounds. Let's assume num_points > 0 for now.
        pytest.skip("Test case for num_points=0 needs clarification on expected behavior.")

    if num_points > 0:
        point = _generate_fibonacci_sphere_point(index=index, num_points=num_points)
        assert point.shape == (3,), "Point should be a 3D vector."
        assert np.isclose(np.linalg.norm(point), 1.0), "Point should be unit length."

def test_fibonacci_point_invalid_index_raises_error():
    """Test that an out-of-bounds index raises ValueError."""
    with pytest.raises(ValueError):
        _generate_fibonacci_sphere_point(index=12, num_points=12) # index must be < num_points
    with pytest.raises(ValueError):
        _generate_fibonacci_sphere_point(index=-1, num_points=12)

# --- Tests for get_number_of_nodes_at_tier ---

@pytest.mark.parametrize("tier, expected_nodes", _TIER_TO_N_POINTS_MAP.items())
def test_get_number_of_nodes_valid_tier(tier, expected_nodes):
    """Test getting node count for all defined valid tiers."""
    assert get_number_of_nodes_at_tier(tier) == expected_nodes

def test_get_number_of_nodes_invalid_tier_raises_error():
    """Test that an undefined tier raises ValueError."""
    invalid_tier = max(_TIER_TO_N_POINTS_MAP.keys()) + 1 # A tier not in the map
    with pytest.raises(ValueError):
        get_number_of_nodes_at_tier(invalid_tier)

# --- Tests for get_discrete_node_orientation ---

@pytest.mark.parametrize("tier", _TIER_TO_N_POINTS_MAP.keys())
def test_orientation_is_normalized(tier):
    """Check if generated orientation quaternions are normalized."""
    # Test for the first node (index 0) of each tier
    orientation_q = get_discrete_node_orientation(resolution_tier=tier, node_index_at_tier=0)
    assert orientation_q.shape == (4,), "Orientation should be a 4-vector quaternion."
    assert np.isclose(np.linalg.norm(orientation_q), 1.0), "Orientation quaternion should be unit length."

def test_orientation_local_z_matches_fibonacci_point():
    """
    Crucial test: The local Z-axis of the node's orientation should correspond
    to the Fibonacci sphere point used to generate it.
    """
    tier = 0 # Example tier
    node_idx = 5 # Example node index
    
    num_points_for_tier = get_number_of_nodes_at_tier(tier)
    expected_fibonacci_vec = _generate_fibonacci_sphere_point(node_idx, num_points_for_tier)
    
    orientation_q = get_discrete_node_orientation(tier, node_idx)
    actual_local_z_vec = get_local_z_from_quaternion(orientation_q)
    
    assert np.allclose(actual_local_z_vec, expected_fibonacci_vec), \
        "Local Z from orientation should match the generating Fibonacci vector."

@pytest.mark.parametrize("tier", _TIER_TO_N_POINTS_MAP.keys())
def test_orientation_invalid_node_index_raises_error(tier):
    """Test that an out-of-bounds node_index_at_tier raises ValueError."""
    num_nodes = get_number_of_nodes_at_tier(tier)
    with pytest.raises(ValueError):
        get_discrete_node_orientation(tier, num_nodes) # num_nodes is an invalid index (0 to num_nodes-1)
    with pytest.raises(ValueError):
        get_discrete_node_orientation(tier, -1)

def test_orientation_invalid_tier_raises_error():
    """Test that an invalid resolution_tier raises ValueError (via get_number_of_nodes_at_tier)."""
    invalid_tier = max(_TIER_TO_N_POINTS_MAP.keys()) + 1
    with pytest.raises(ValueError):
        # node_index_at_tier=0 is arbitrary here, error should come from invalid tier
        get_discrete_node_orientation(invalid_tier, 0) 

