# SpinStep - [Read the Docs](https://github.com/VoxLeone/SpinStep/tree/main/docs/index.md)

**SpinStep** is a proof-of-concept quaternion-driven traversal framework for trees and orientation-based data structures.

By leveraging the power of 3D rotation math, SpinStep enables traversal based not on position or order — but on orientation. This makes it ideal for spatial reasoning, robotics, 3D scene graphs, and anywhere quaternion math naturally applies.

<div align="center">
  <img src="assets/quaternion-tree.png" alt="A 3D Graph concept image" style="max-width: 100% style="margin: 20px;" />
</div>

---

## ✨ Features

- Quaternion-based stepping and branching  
- Full support for yaw, pitch, and roll rotations  
- Configurable angular thresholds for precision control  
- Easily extendable to N-ary trees or orientation graphs  
- Written in Python with SciPy's rotation engine  

---

## Example Use Case

```python
from spinstep import Node, QuaternionDepthIterator

# Define your tree with orientation quaternions
root = Node("root", [0, 0, 0, 1], [
    Node("child", [0.2588, 0, 0, 0.9659])  # ~30 degrees around Z
])

# Step using a 30-degree rotation
iterator = QuaternionDepthIterator(root, [0.2588, 0, 0, 0.9659])

for node in iterator:
    print("Visited:", node.name)
```

---

## Requirements

- Python 3.8+  
- `numpy`  
- `scipy`  

Install dependencies via pip:

```bash
pip install numpy scipy
```
## Node Requirements

- `.orientation`: Quaternion as `[x, y, z, w]`, always normalized.
- `.children`: Iterable of nodes.
- Node constructor and orientation set utilities always normalize quaternions and check for zero-norm.
- `angle_threshold` parameters are always in radians.

All core functions will raise `ValueError` or `AttributeError` if these invariants are violated.

---

## Concepts

SpinStep uses quaternion rotation to determine if a child node is reachable from a given orientation. Only children whose orientations lie within a defined angular threshold (default: 45°) of the current rotation state are traversed.

This mimics rotational motion or attention in physical and virtual spaces — ideal for:

- Orientation trees  
- 3D pose search  
- Animation graph traversal  
- Spatial AI and robotics

<div align="center">
  <img src="https://raw.githubusercontent.com/VoxLeone/SpinStep/main/docs/assets/spinstep-quaternion-diagram.png" alt="A 3D Graph concept image" style="max-width: 100% style="margin: 20px;" />
</div>

---

## What Would It Mean to “Rotate into Branches”?

### 1. Quaternion as a Branch Selector

- Each node in a graph or tree encodes a rotational state (quaternion)  
- Traversal is guided by a current quaternion state  
- At each step, you rotate your state and select the next node based on geometric orientation  

🔸 *Use Cases*: Scene graphs, spatial indexing, directional AI traversal, robot path planning  

---

### 2. Quaternion-Based Traversal Heuristics

Instead of:

```python
next = left or right
```

You define:

```python
next_node = rotate(current_orientation, branch_orientation)
```

- Rotation (quaternion multiplication) becomes the “step” function  
- Orientation and direction are first-class traversal parameters  

🔸 *Use Cases*: Game engines, camera control, 3D modeling, procedural generation  

---

### 3. Multi-Dimensional Trees with Quaternion Keys

- Use quaternion distance (angle) to decide which branches to explore or when to stop  
- Think of it like a quaternion-aware k-d tree  

---

### Visual Metaphor

Imagine walking through a tree **not** left/right — but by **rotating** in space:

- Rotate **pitch** to descend to one child  
- Rotate **yaw** to reach another  
- Traverse hierarchies by change in orientation, not position  

---

## Project Structure

```
spinstep/
├── __init__.py
├── node.py
├── traversal.py         # Core iterator logic
├── quaternion_utils.py  # Quaternion math helpers
├── demo.py
├── demo1_tree_traversal.py
├── demo2_full_depth_traversal.py
├── demo3_spatial_traversal.py
README.md
LICENSE
pyproject.toml
MANIFEST.in
setup.cfg
setup.py
```

---

## To Build and Install Locally

First, clone the repository:

```bash
git clone https://github.com/VoxLeone/spinstep.git
cd spinstep
```

Then, install it:

```bash
pip install .
```

To build a wheel distribution:

```bash
python -m build
```

---

## 📜 License

MIT — free to use, fork, and adapt.

---

## 💬 Feedback & Contributions

PRs and issues are welcome!  
If you're using SpinStep in a cool project, let us know.
