# Continuous Traversal Guide

The `QuaternionDepthIterator` performs depth-first traversal of a quaternion-oriented tree using a single continuous rotation step.

## How It Works

At each visited node the iterator:

1. Applies the rotation step to the current orientation state.
2. Computes the angular distance between the rotated state and each child's orientation.
3. Pushes children within the `angle_threshold` onto the traversal stack.
4. Returns the current node to the caller.

## Basic Usage

```python
from spinstep import Node, QuaternionDepthIterator

root = Node("root", [0, 0, 0, 1], [
    Node("child_a", [0.2588, 0, 0, 0.9659]),  # ~30° around Z
    Node("child_b", [0.7071, 0, 0, 0.7071]),  # ~90° around Z
])

# Step with a 30° rotation — only child_a is close enough
for node in QuaternionDepthIterator(root, [0.2588, 0, 0, 0.9659]):
    print(node.name)
# Output:
# root
# child_a
```

## Angle Threshold

By default, the threshold is set dynamically to 30 % of the step angle (minimum 1°).
You can override it with an explicit value in radians:

```python
import numpy as np
from spinstep import Node, QuaternionDepthIterator

root = Node("root", [0, 0, 0, 1], [
    Node("child", [0.2588, 0, 0, 0.9659]),
])

# Explicit threshold of 45° (in radians)
it = QuaternionDepthIterator(root, [0.2588, 0, 0, 0.9659], angle_threshold=np.deg2rad(45))
visited = [n.name for n in it]
print(visited)
# Output: ['root', 'child']
```

### Dynamic Threshold Behaviour

When `angle_threshold` is not specified:

- The step angle is computed from `rotation_step_quat`.
- The threshold is set to `step_angle * 0.3`.
- If the step angle is near zero (identity rotation), the threshold defaults to 1° (`np.deg2rad(1.0)`).

## Deep Trees

The iterator naturally handles multi-level trees. Children are traversed depth-first:

```python
from spinstep import Node, QuaternionDepthIterator

grandchild = Node("grandchild", [0.5, 0, 0, 0.866])
child = Node("child", [0.2588, 0, 0, 0.9659], [grandchild])
root = Node("root", [0, 0, 0, 1], [child])

for node in QuaternionDepthIterator(root, [0.2588, 0, 0, 0.9659], angle_threshold=1.0):
    print(node.name)
# Output:
# root
# child
# grandchild
```

## Iterator Protocol

`QuaternionDepthIterator` implements Python's iterator protocol (`__iter__` and `__next__`), so you can use it in `for` loops, `list()` calls, or `next()`:

```python
from spinstep import Node, QuaternionDepthIterator

root = Node("root", [0, 0, 0, 1])
it = QuaternionDepthIterator(root, [0, 0, 0, 1])

first = next(it)
print(first.name)
# Output: root
```

## Edge Cases

- **Zero-norm child orientation:** Gracefully skipped (not visited, no error).
- **Identity rotation step:** Threshold defaults to 1°.
- **No matching children:** The iterator simply yields the root and stops.

---

[⬅️ Getting Started](getting-started.md) | [🏠 Home](index.md) | [Discrete Traversal ➡️](discrete-traversal.md)
