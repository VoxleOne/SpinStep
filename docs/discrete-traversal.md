# Discrete Traversal Guide

SpinStep supports traversal using discrete sets of quaternion rotations via `DiscreteOrientationSet` and `DiscreteQuaternionIterator`.

## When to Use Discrete Traversal

Use discrete traversal when:

- You want to explore orientations from a symmetry group (cube, icosahedron).
- You have a fixed set of allowed rotation steps (e.g. robot actuator positions).
- You want to search over a grid of orientations.

## DiscreteOrientationSet

A container for a set of normalised quaternion orientations with angular querying.

### Creating Orientation Sets

```python
from spinstep import DiscreteOrientationSet

# Predefined symmetry groups
cube_set = DiscreteOrientationSet.from_cube()           # 24 orientations (octahedral group)
icosa_set = DiscreteOrientationSet.from_icosahedron()    # 60 orientations (icosahedral group)

# Fibonacci-sphere sampling
grid_set = DiscreteOrientationSet.from_sphere_grid(200)  # 200 orientations

# Custom quaternions
import numpy as np
custom_quats = np.array([
    [0, 0, 0, 1],
    [0.7071, 0, 0, 0.7071],
    [0, 0.7071, 0, 0.7071],
])
custom_set = DiscreteOrientationSet.from_custom(custom_quats)
```

### Querying by Angle

Find all orientations within a given angular distance of a query quaternion:

```python
import numpy as np
from spinstep import DiscreteOrientationSet

dos = DiscreteOrientationSet.from_cube()
identity = [0, 0, 0, 1]
indices = dos.query_within_angle(identity, np.deg2rad(10))
print(f"Found {len(indices)} orientations within 10° of identity")
```

### GPU Support

Pass `use_cuda=True` to store orientations on GPU (requires [CuPy](https://cupy.dev/)):

```python
import numpy as np
from spinstep import DiscreteOrientationSet

orientations = np.random.randn(10000, 4)
gpu_set = DiscreteOrientationSet(orientations, use_cuda=True)
```

## DiscreteQuaternionIterator

Depth-first iterator that tries every orientation in the set as a rotation step at each node.

### Basic Usage

```python
import numpy as np
from spinstep import Node, DiscreteOrientationSet, DiscreteQuaternionIterator

root = Node("root", [0, 0, 0, 1], [
    Node("child1", [0, 0, 0.3827, 0.9239]),    # ~45° around Z
    Node("child2", [0, 0.7071, 0, 0.7071]),     # 90° around Y
])

orientation_set = DiscreteOrientationSet.from_cube()
it = DiscreteQuaternionIterator(root, orientation_set, angle_threshold=np.pi / 4)

for node in it:
    print(node.name)
# Output:
# root
# child1
# child2
```

### Parameters

- `angle_threshold` — maximum angular distance (radians) for a child to be reachable. Default: `π/8` (22.5°).
- `max_depth` — maximum traversal depth. Default: `100`.

### Depth Limiting

```python
import numpy as np
from spinstep import Node, DiscreteOrientationSet, DiscreteQuaternionIterator

deep_child = Node("deep", [0.5, 0, 0, 0.866])
child = Node("child", [0, 0, 0.3827, 0.9239], [deep_child])
root = Node("root", [0, 0, 0, 1], [child])

dos = DiscreteOrientationSet.from_cube()
it = DiscreteQuaternionIterator(root, dos, angle_threshold=1.0, max_depth=1)

visited = [n.name for n in it]
print(visited)
# 'deep' is not visited because max_depth=1 limits traversal
```

## Use Cases

- **Robot planning:** Restrict traversal to valid actuator orientations.
- **Crystal / molecular modelling:** Traverse only symmetry-equivalent orientations.
- **3D search / AI:** Explore all directions in a grid.
- **Procedural content:** Generate structures along discrete orientation paths.

## Performance Tips

- For large orientation sets (>1000), increase `angle_threshold` or decrease `max_depth` to limit the search space.
- The iterator tracks visited nodes by identity to avoid cycles.

---

[⬅️ Continuous Traversal](continuous-traversal.md) | [🏠 Home](index.md) | [FAQ ➡️](faq.md)
