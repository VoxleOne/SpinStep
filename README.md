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
