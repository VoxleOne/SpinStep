# SpinStep

**SpinStep** is a lightweight, quaternion-driven traversal framework for trees and orientation-based data structures.

By leveraging the power of 3D rotation math, SpinStep enables traversal based not on position or order — but on **orientation**. This makes it ideal for spatial reasoning, robotics, 3D scene graphs, and anywhere quaternion math naturally applies.

---

## ✨ Features

- Quaternion-based stepping and branching
- Full support for yaw, pitch, and roll rotations
- Configurable angular thresholds for precision control
- Easily extendable to N-ary trees or orientation graphs
- Written in Python with `scipy`'s rotation engine

---

## 🔧 Example Use Case

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

## 📦 Requirements

- Python 3.8+
- `numpy`
- `scipy`

Install via pip:

```bash
pip install numpy scipy
```

---

## 🧠 Concepts

SpinStep uses **quaternion rotation** to determine if a child node is reachable from a given orientation. Only those children whose orientations lie within a defined angular threshold (default: 45°) of the current rotation state are traversed.

This mimics rotational motion or attention in physical and virtual spaces — ideal for:

- Orientation trees
- 3D pose search
- Animation graph traversal
- Spatial AI and robotics

---

## 🧭 What Would It Mean to “Rotate into Branches”?

Let’s unpack it:

✅ 1. Quaternion as a Branch Selector

    Imagine each node in a graph or tree encodes rotational states (quaternions).

    Traversal is guided by a current quaternion state.

    At each step, you rotate your state and select the next node based on geometric orientation — like rotating through a field of possibilities.

🔸 Use Case: Scene graphs, spatial indexing, directional AI traversal, robot path planning.

✅ 2. Quaternion-Based Traversal Heuristics

    Instead of "next = left/right", you define:

    next_node = rotate(current_orientation, branch_orientation);

    Rotation (quaternion multiplication) becomes your “step” function in the iterator.

    This makes orientation and direction first-class traversal parameters.

🔸 Use Case: Game engines (e.g., cameras rotating into nearby zones), 3D modeling (e.g., mesh walks), or procedural generation.

✅ 3. Multi-Dimensional Trees with Quaternion Keys

    In a tree where nodes have orientation data, you could use quaternion distance (angle) to decide:

        Which branches to explore

        When to stop

    Think of this like a quaternion-aware k-d tree.

✨ Visual Metaphor:

Imagine walking through a tree not left/right, but by rotating in space:

    Rotate “pitch” to go down to one child.

    Rotate “yaw” to go to another.

    Traverse a hierarchy of nodes not by position, but by change in orientation.

---

## 📁 Structure

```
spinstep/
│
├── __init__.py
├── node.py
├── traversal.py         # Core Iterator logic
├── quaternion_utils.py  # Quaternion math helpers
├── demo.py
├── demo1-tree-traversal.py
├── demo2-full-depth-traversal.py
├── demo3-spatial-traversal.py
README.md
LICENSE
pyproject.toml
MANIFEST.in
setup.cfg
setup.py
```

---
## 🚀 To Build and Install Locally

From the root of your project:

```bash
pip install .
```

Or to build a wheel:

```
python -m build
```

## 📜 License

MIT — free to use, fork, and adapt.

---

## 💬 Feedback & Contributions

PRs and issues are welcome! If you're using SpinStep in a cool project, let us know.
