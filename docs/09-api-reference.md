# SpinStep API Reference

This document details the public API of the SpinStep library.

---

## Table of Contents

- [spinstep.node.Node](#node)
- [spinstep.traversal.QuaternionDepthIterator](#quaterniondepthiterator)
- [spinstep.discrete.DiscreteOrientationSet](#discreteorientationset)
- [spinstep.discrete_iterator.DiscreteQuaternionIterator](#discretequaternioniterator)
- [Exceptions](#exceptions)

---

## Node

```python
from spinstep import Node

class Node:
    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        children: Optional[Sequence[Node]] = None,
    ) -> None
```

A tree node with a quaternion-based orientation.

**Args:**

- **name** (`str`): Human-readable identifier.
- **orientation** (`[x, y, z, w]`): Quaternion. Automatically normalised. Must be non-zero.
- **children** (`Sequence[Node]`, optional): Initial child nodes.

**Attributes:**

- **name** (`str`): Node identifier.
- **orientation** (`numpy.ndarray`, shape `(4,)`): Normalised quaternion.
- **children** (`list[Node]`): Child nodes.

**Raises:**

- `ValueError`: If orientation is not a 4-element vector or has near-zero norm.

---

## QuaternionDepthIterator

```python
from spinstep import QuaternionDepthIterator

class QuaternionDepthIterator:
    def __init__(
        self,
        start_node: Node,
        rotation_step_quat: ArrayLike,
        angle_threshold: Optional[float] = None,
    ) -> None
```

Depth-first tree iterator driven by a continuous quaternion rotation step.

**Args:**

- **start_node** (`Node`): Root node of the tree.
- **rotation_step_quat** (`[x, y, z, w]`): Quaternion rotation applied at each step.
- **angle_threshold** (`float`, optional): Maximum angular distance in radians for a child to be visited. When `None`, defaults to 30% of the step angle (minimum 1°).

**Class Variables:**

- `DEFAULT_DYNAMIC_THRESHOLD_FACTOR` (`float`): `0.3`

**Iterator Protocol:**

Implements `__iter__` and `__next__`. Yields `Node` instances in depth-first order.

---

## DiscreteOrientationSet

```python
from spinstep import DiscreteOrientationSet

class DiscreteOrientationSet:
    def __init__(
        self,
        orientations: ArrayLike,
        use_cuda: bool = False,
    ) -> None
```

A set of discrete quaternion orientations with spatial querying.

**Args:**

- **orientations**: Array of shape `(N, 4)` — one quaternion `[x, y, z, w]` per row.
- **use_cuda** (`bool`): When `True`, store on GPU via CuPy.

**Attributes:**

- **orientations**: Normalised quaternion array of shape `(N, 4)`.
- **use_cuda** (`bool`): Whether GPU storage is active.
- **xp**: The array module in use (`numpy` or `cupy`).

### Methods

#### `query_within_angle(quat, angle) -> numpy.ndarray`

Return indices of orientations within `angle` radians of `quat`.

- **quat**: Query quaternion `[x, y, z, w]` or batch `(N, 4)`.
- **angle** (`float`): Maximum angular distance in radians.
- **Returns**: Integer index array.

#### `as_numpy() -> numpy.ndarray`

Convert orientations to a NumPy array (transfers from GPU if needed).

#### `__len__() -> int`

Return the number of orientations.

### Factory Class Methods

#### `from_cube() -> DiscreteOrientationSet`

24 orientations from the octahedral symmetry group.

#### `from_icosahedron() -> DiscreteOrientationSet`

60 orientations from the icosahedral symmetry group.

#### `from_custom(quat_list) -> DiscreteOrientationSet`

Create from a user-supplied array of quaternions `(N, 4)`.

#### `from_sphere_grid(n_points=100) -> DiscreteOrientationSet`

Fibonacci-sphere sampling of `n_points` orientations.

---

## DiscreteQuaternionIterator

```python
from spinstep import DiscreteQuaternionIterator

class DiscreteQuaternionIterator:
    def __init__(
        self,
        start_node: Node,
        orientation_set: DiscreteOrientationSet,
        angle_threshold: float = np.pi / 8,
        max_depth: int = 100,
    ) -> None
```

Depth-first tree iterator using a discrete set of orientation steps.

**Args:**

- **start_node** (`Node`): Root node of the tree.
- **orientation_set** (`DiscreteOrientationSet`): Candidate rotation steps.
- **angle_threshold** (`float`): Maximum angular distance in radians. Default: `π/8` (22.5°).
- **max_depth** (`int`): Maximum traversal depth. Default: `100`.

**Raises:**

- `AttributeError`: If `start_node` lacks `.orientation` or `.children`.

**Iterator Protocol:**

Implements `__iter__` and `__next__`. Yields `Node` instances in depth-first order. Tracks visited nodes to avoid cycles.

---

## Exceptions

- **ValueError**: Raised for invalid quaternions (zero-norm, wrong shape) or malformed orientation sets.
- **AttributeError**: Raised if a node lacks required `.orientation` or `.children` attributes.

---

[⬅️ Troubleshooting](08-troubleshooting.md) | [🏠 Home](index.md)
