# Getting Started

## Installation

**Requirements:** Python 3.9+

Install SpinStep from source:

```bash
git clone https://github.com/VoxleOne/SpinStep.git
cd SpinStep
pip install .
```

This installs SpinStep and its core dependencies (`numpy`, `scipy`, `scikit-learn`).

For development (includes `pytest`, `ruff`, `mypy`, `black`):

```bash
pip install -e ".[dev]"
```

## First Example

SpinStep uses quaternions `[x, y, z, w]` to represent orientations.
Here is a minimal working example using continuous traversal:

```python
from spinstep import Node, QuaternionDepthIterator

# Identity quaternion: [0, 0, 0, 1] (no rotation)
root = Node("root", [0, 0, 0, 1], [
    Node("child", [0.2588, 0, 0, 0.9659])  # ~30° around Z
])

# Traverse with a 30° rotation step
for node in QuaternionDepthIterator(root, [0.2588, 0, 0, 0.9659]):
    print("Visited:", node.name)
# Output:
# Visited: root
# Visited: child
```

### How It Works

1. You create `Node` objects, each with a quaternion orientation.
2. You choose a traversal mode:
   - `QuaternionDepthIterator` for continuous single-step traversal.
   - `DiscreteQuaternionIterator` for multi-step discrete traversal.
3. The iterator visits nodes whose orientations are reachable from the current state.

## Key Concepts

- **Quaternion format:** SpinStep uses `[x, y, z, w]` format (scalar-last), matching SciPy's convention.
- **Automatic normalisation:** All quaternions are normalised on construction. You do not need to pre-normalise.
- **Angle threshold:** Controls how close a child's orientation must be to the rotated state to be visited. Measured in radians.

## Next Steps

- [Continuous Traversal Guide](continuous-traversal.md) — detailed `QuaternionDepthIterator` usage.
- [Discrete Traversal Guide](discrete-traversal.md) — using `DiscreteOrientationSet` and `DiscreteQuaternionIterator`.
- [FAQ](faq.md) — common pitfalls and tips.
- [API Reference](09-api-reference.md) — full class and method documentation.
