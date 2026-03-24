# FAQ

## Installation & Setup

### How do I install SpinStep?

```bash
git clone https://github.com/VoxleOne/SpinStep.git
cd SpinStep
pip install .
```

### What Python versions are supported?

Python 3.9, 3.10, 3.11, and 3.12.

### What are the required dependencies?

- `numpy>=1.22`
- `scipy>=1.10`
- `scikit-learn>=1.2`

These are installed automatically when you run `pip install .`.

## Common Pitfalls

### ValueError about quaternions?

All quaternions must be non-zero. SpinStep normalises them automatically, but zero vectors (e.g. `[0, 0, 0, 0]`) will raise a `ValueError`:

```python
from spinstep import Node
Node("bad", [0, 0, 0, 0])  # Raises ValueError
```

### AttributeError about .orientation or .children?

`DiscreteQuaternionIterator` requires nodes with `.orientation` and `.children` attributes. Use the `Node` class or provide objects with these attributes.

### Why is my traversal only visiting the root?

The `angle_threshold` may be too small. Try increasing it:

```python
import numpy as np
from spinstep import Node, QuaternionDepthIterator

root = Node("root", [0, 0, 0, 1], [
    Node("child", [0.2588, 0, 0, 0.9659])
])

# Use a larger threshold
for node in QuaternionDepthIterator(root, [0.2588, 0, 0, 0.9659], angle_threshold=np.deg2rad(45)):
    print(node.name)
```

## GPU vs CPU Behaviour

### When should I use `use_cuda=True`?

Only when you have CuPy installed and a CUDA-capable GPU. GPU acceleration benefits large orientation sets (thousands of quaternions) where batch angular distance computation is the bottleneck.

```python
from spinstep import DiscreteOrientationSet

# CPU (default)
cpu_set = DiscreteOrientationSet.from_cube()

# GPU (requires CuPy)
import numpy as np
gpu_set = DiscreteOrientationSet(np.random.randn(10000, 4), use_cuda=True)
```

### What happens if CuPy is not installed?

If `use_cuda=True` is specified but CuPy is not available, SpinStep falls back to NumPy. No error is raised.

### Does the tree traversal itself run on GPU?

No. The traversal logic (stack operations, depth tracking) always runs on CPU. GPU acceleration applies only to orientation storage and batch angular distance computations in `DiscreteOrientationSet`.

## Optional Dependencies

### What is CuPy used for?

[CuPy](https://cupy.dev/) enables GPU-accelerated storage and computation for `DiscreteOrientationSet`. Install with:

```bash
pip install cupy-cuda12x
```

### What is healpy used for?

[healpy](https://healpy.readthedocs.io/) enables the `get_unique_relative_spins()` utility function for HEALPix-based unique relative spin detection. Install with:

```bash
pip install healpy
```

## Quaternion Format

### What quaternion convention does SpinStep use?

SpinStep uses `[x, y, z, w]` format (scalar-last), matching [SciPy's convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html).

### Can I use Euler angles or axis-angle?

Yes. Convert to quaternions first using SciPy:

```python
from scipy.spatial.transform import Rotation as R

# From Euler angles (in degrees)
quat = R.from_euler("xyz", [30, 0, 0], degrees=True).as_quat()

# From axis-angle
quat = R.from_rotvec([0.5236, 0, 0]).as_quat()  # 30° around X
```

### Are quaternions automatically normalised?

Yes. Both `Node` and `DiscreteOrientationSet` normalise quaternions on construction.

---

[⬅️ Discrete Traversal](discrete-traversal.md) | [🏠 Home](index.md) | [API Reference ➡️](09-api-reference.md)
