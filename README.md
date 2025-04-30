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

## 📁 Structure

```
spinstep/
│
├── __init__.py
├── traversal.py         # Core iterator logic
├── quaternion_utils.py  # Quaternion math helpers
```

---

## 📜 License

MIT — free to use, fork, and adapt.

---

## 💬 Feedback & Contributions

PRs and issues are welcome! If you're using SpinStep in a cool project, let us know.

```

---

Would you like this as a downloadable file or to expand into a full package scaffold?
