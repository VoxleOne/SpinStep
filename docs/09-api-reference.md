# SpinStep API Reference

This document details the main public API of the SpinStep library.  
For usage examples, see the [Examples & Tutorials](examples.md).

---

## Table of Contents

- [spinstep.node.Node](#spinstepnodenode)
- [spinstep.discrete.DiscreteOrientationSet](#spinstepdiscretediscreteorientationset)
- [spinstep.discrete_iterator.DiscreteQuaternionIterator](#spinstepdiscrete_iteratordiscretequaternioniterator)
- [spinstep.continuous.QuaternionDepthIterator](#spinstepcontinuousquaterniondepthiterator)
- [Exceptions](#exceptions)

---

## spinstep.node.Node

```python
class Node:
    def __init__(self, name: str, orientation: Sequence[float], children: Optional[Iterable["Node"]] = None)
```

- **name** (`str`): Node identifier (any string).
- **orientation** (`[x, y, z, w]`): Quaternion representing the orientation. Automatically normalized.
- **children** (`Iterable[Node]`, optional): List or iterable of child nodes.

#### Attributes

- **name**: Node name.
- **orientation**: Normalized quaternion (`numpy.ndarray`, shape `(4,)`).
- **children**: List of child nodes.

---

## spinstep.discrete.DiscreteOrientationSet

A container for a set of discrete orientations (quaternions).

```python
class DiscreteOrientationSet:
    def __init__(self, orientations: Sequence[Sequence[float]])
```

#### Constructor

- **orientations**: List or array of normalized quaternions (`[x, y, z, w]`).

#### Class Methods

- `from_cube()`: Returns a set of cube group orientations (24 elements).
- `from_icosahedron()`: Returns a set of icosahedral group orientations (60 elements).
- `from_custom(orientations)`: Create from user-specified list of quaternions.
- `from_sphere_grid(N: int)`: Approximate a uniform grid of `N` orientations on the sphere.

#### Methods

- `query_within_angle(quat: Sequence[float], angle: float) -> np.ndarray`:  
  Returns indices of orientations within `angle` (radians) of `quat`.

#### Attributes

- **orientations**: Array of normalized quaternions (shape `(n,4)`).

---

## spinstep.discrete_iterator.DiscreteQuaternionIterator

Depth-first traversal over a tree of Nodes using a discrete orientation set.

```python
class DiscreteQuaternionIterator:
    def __init__(
        self,
        start_node: Node,
        orientation_set: DiscreteOrientationSet,
        angle_threshold: float = np.pi/8,
        max_depth: int = 100
    )
```

- **start_node**: The root `Node` to start traversal from.
- **orientation_set**: Instance of `DiscreteOrientationSet`.
- **angle_threshold**: Maximum allowed angular distance (in radians) to consider two orientations "matching."
- **max_depth**: Maximum recursion depth.

#### Usage

```python
it = DiscreteQuaternionIterator(root_node, orientation_set, angle_threshold=0.2)
for node in it:
    print(node.name)
```

---

## spinstep.continuous.QuaternionDepthIterator

Depth-first traversal for continuous (non-discrete) orientation search.

```python
class QuaternionDepthIterator:
    def __init__(
        self,
        start_node: Node,
        angle_threshold: float = np.pi/8,
        max_depth: int = 100
    )
```

- **start_node**: The root `Node`.
- **angle_threshold**: Maximum allowed angular distance (in radians) for matching.
- **max_depth**: Maximum recursion depth.

#### Usage

```python
it = QuaternionDepthIterator(root_node, angle_threshold=0.1)
for node in it:
    print(node.name)
```

---

## Exceptions

- **ValueError**: Raised for invalid or non-normalized quaternions, or malformed orientation sets.
- **AttributeError**: Raised if a node lacks required `.orientation` or `.children` attributes.

---

## See Also

- [Orientation Sets](05_orientation_sets.md)
- [Discrete Traversal Guide](06_discrete_traversal.md)
- [Troubleshooting & FAQ](07_troubleshooting.md)

---
[⬅️ 08. Rationale](01-rationale.md) | [🏠 Home](index.md) 
