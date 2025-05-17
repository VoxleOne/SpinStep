# test_spinstep.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from spinstep.discrete import DiscreteOrientationSet
# If you have continuous traversal classes, import them here
# from spinstep.continuous import QuaternionDepthIterator
from spinstep.node import Node

@pytest.fixture
def simple_tree():
    return Node("root", [0, 0, 0, 1], [
        Node("A", [0.707, 0, 0, 0.707]),
        Node("B", [0, 0.707, 0, 0.707])
    ])

def test_discrete_orientation_set_cpu():
    arr = np.eye(4)
    dset = DiscreteOrientationSet(arr)
    assert len(dset) == 4
    assert np.allclose(np.linalg.norm(dset.orientations, axis=1), 1)

def test_discrete_orientation_set_gpu():
    if not HAS_CUPY:
        pytest.skip("CuPy not installed")
    arr = np.eye(4)
    dset = DiscreteOrientationSet(arr, use_cuda=True)
    assert len(dset) == 4
    norms = dset.xp.linalg.norm(dset.orientations, axis=1)
    assert dset.xp.allclose(norms, 1)

def test_query_within_angle_cpu():
    arr = np.array([
        [0, 0, 0, 1],
        [0.707, 0, 0, 0.707],
        [0, 0.707, 0, 0.707]
    ])
    dset = DiscreteOrientationSet(arr)
    inds = dset.query_within_angle([0, 0, 0, 1], angle=1.0)
    assert 0 in inds

def test_query_within_angle_gpu():
    if not HAS_CUPY:
        pytest.skip("CuPy not installed")
    arr = np.array([
        [0, 0, 0, 1],
        [0.707, 0, 0, 0.707],
        [0, 0.707, 0, 0.707]
    ])
    dset = DiscreteOrientationSet(arr, use_cuda=True)
    inds = dset.query_within_angle([0, 0, 0, 1], angle=1.0)
    assert 0 in dset.xp.asnumpy(inds)

def test_from_cube_and_icosahedron():
    dset_cube = DiscreteOrientationSet.from_cube()
    dset_ico = DiscreteOrientationSet.from_icosahedron()
    assert len(dset_cube) == 24
    assert len(dset_ico) == 60

def test_from_custom():
    arr = np.eye(4)
    dset = DiscreteOrientationSet.from_custom(arr)
    assert len(dset) == 4

def test_from_sphere_grid():
    dset = DiscreteOrientationSet.from_sphere_grid(10)
    assert len(dset) == 10

def test_invalid_quaternion():
    arr = np.zeros((3, 4))
    with pytest.raises(ValueError):
        DiscreteOrientationSet(arr)

def test_as_numpy_gpu():
    if not HAS_CUPY:
        pytest.skip("CuPy not installed")
    arr = np.eye(4)
    dset = DiscreteOrientationSet(arr, use_cuda=True)
    arr2 = dset.as_numpy()
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == (4, 4)

# Example for continuous traversal (if implemented)
# def test_continuous_traversal(simple_tree):
#     # Replace with your actual traversal class and logic
#     it = QuaternionDepthIterator(simple_tree, angle_threshold=0.2)
#     names = [node.name for node in it]
#     assert "root" in names and "A" in names and "B" in names

