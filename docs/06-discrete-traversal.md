# Discrete Quaternion Traversal in SpinStep

SpinStep now supports traversal using **arbitrary discrete sets of quaternion rotations**. This enables robust, grid-based, or symmetry-aware traversal for robotics, computer graphics, molecular modeling, and more.

## What is Discrete Traversal?

Instead of rotating by a fixed axis/angle, you can now:
- Traverse via all cube symmetries (24 steps)
- Traverse via icosahedral symmetries (60 steps)
- Use a uniform grid or user-defined set of orientations
- Model symmetry group traversal, sphere grids, or custom grids

## API

```python
from spinstep.discrete import DiscreteOrientationSet
from spinstep.discrete_iterator import DiscreteQuaternionIterator

# Cube group (default)
orientation_set = DiscreteOrientationSet.from_cube()

# Or icosahedral group, or user-defined set
# orientation_set = DiscreteOrientationSet.from_icosahedron()
# orientation_set = DiscreteOrientationSet.from_custom([...])

itr = DiscreteQuaternionIterator(root, orientation_set, angle_threshold=np.pi/8)
for node in itr:
    print(node.name)
```

## Use Cases

- **Robot planning**: Restrict to valid actuator orientations.
- **Crystal/molecular modeling**: Traverse only symmetry-equivalent orientations.
- **3D search/AI**: Explore all directions in a grid.
- **Procedural content**: Generate structures along discrete orientation paths.

## Demo

See `demo4_discrete_traversal.py` for an example.

---

## Implementation Notes

- Under the hood, discrete traversal tries all possible steps in the orientation set at each node.
- For very large sets (e.g. >1000), tune the angle threshold and/or max_depth to control performance.
- Extend `DiscreteOrientationSet` for more symmetry groups or sampling schemes as needed.

---
