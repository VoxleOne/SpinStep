# SpinStep Architecture

SpinStep is an extensible Python library for traversal of orientation-based trees and graphs using quaternion mathematics. Its architecture is designed to support both continuous and discrete rotational traversal, making it adaptable for a wide range of domains—robotics, graphics, AI, molecular modeling, and beyond.

---

## Core Concepts

### 1. **Node Structure**
Each node represents an entity in the tree/graph with:
- A unique name or label
- An orientation (quaternion: `[x, y, z, w]`)
- Zero or more child nodes

Nodes can be freely composed to build arbitrary tree or graph structures.

---

### 2. **Traversal Engines**
SpinStep provides iterator classes that traverse the tree by stepping through orientations:

#### **Continuous Traversal**
- Uses a fixed quaternion rotation step (e.g., rotate 30° around Z each step)
- At each node, children whose orientations are within a configurable angular threshold of the current orientation are visited
- Supports depth-first, breadth-first, and spatial traversal variants

#### **Discrete Traversal** (Extension)
- Allows traversal to use a **discrete set of orientations** (e.g., all cube symmetries, user-defined grids)
- At each step, the iterator tries all possible discrete rotations from the current orientation
- Enables traversal over symmetry groups, rotational grids, or custom orientation sets

---

### 3. **Quaternion Utilities**
All quaternion operations (creation, rotation, distance, normalization) are abstracted via utility functions relying on SciPy’s robust rotation API.

---

## Extensibility

SpinStep is designed for extension:
- **Custom Traversal Modes:** Implement your own iterator (inheriting from `Iterator` or using SpinStep utilities)
- **Pluggable Discrete Sets:** Use `DiscreteOrientationSet` to define any set of orientations for traversal
- **Custom Node Types:** Extend the node class to include additional metadata, weights, or payloads

---

## Typical Workflow

1. **Tree Construction:** Build a tree using `Node`, assigning each an orientation quaternion.
2. **Choose Traversal:** Select an iterator:
   - `QuaternionDepthIterator` for continuous, fixed-step traversal
   - `DiscreteQuaternionIterator` for group/grid-based traversal
3. **Configure Parameters:** Set the rotation step, angle threshold, and/or discrete orientation set.
4. **Traverse:** Iterate over nodes as needed for your application—search, planning, generation, etc.

---

## Example: Discrete Traversal with Cube Symmetries

```python
from spinstep.node import Node
from spinstep.discrete import DiscreteOrientationSet
from spinstep.discrete_iterator import DiscreteQuaternionIterator

# Create your tree (orientations as quaternions)
root = Node("root", [0,0,0,1], [...])

# Use the cube group for discrete steps
orientation_set = DiscreteOrientationSet.from_cube()

iterator = DiscreteQuaternionIterator(root, orientation_set, angle_threshold=0.3)
for node in iterator:
    print(node.name)
```

---

## Design Motivation

- **Orientation-first:** Traversal is always guided by rotational relationships, not just topology or position.
- **General-purpose:** Works for kinematic chains, scene graphs, spatial search, and more.
- **Mathematically robust:** By using quaternions, SpinStep avoids gimbal lock and enables smooth, consistent orientation operations.

For deeper rationale and use cases, see [docs/01_rationale.md](01_rationale.md) and [docs/04_use_cases.md](04_use_cases.md).

---
[⬅️ 01. Rationale](01-rationale.md) | [🏠 Home](index.md) | [03. Basics ➡️](03-basics.md)

