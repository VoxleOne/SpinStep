# SpinStep — Repository Status Report

> **Version:** 0.1.0 (Alpha)  
> **Date:** 2026-03-24  
> **License:** MIT  

---

## Table of Contents

1. [Current Status](#1-current-status)
2. [Functionalities](#2-functionalities)
3. [Capabilities](#3-capabilities)
4. [Suggested Further Functionalities](#4-suggested-further-functionalities)

---

## 1. Current Status

### Overview

SpinStep is an alpha-stage Python library that implements quaternion-driven tree
traversal.  Instead of traversing by position or order, SpinStep uses 3D
orientation (quaternion rotation) to decide which branches to explore.  The core
library is functional, well-typed, and has no import-time side effects.

### Repository Layout

| Directory | Role | Distributed |
|-----------|------|:-----------:|
| `spinstep/` | Core library (≈ 680 LOC) | ✅ |
| `tests/` | Unit & integration tests (≈ 390 LOC) | ❌ |
| `docs/` | Markdown documentation (≈ 1 500 LOC) | ❌ |
| `demos/` | Educational scripts (5 files) | ❌ |
| `examples/` | Real-world examples (1 file) | ❌ |
| `benchmark/` | Performance benchmarks & QGNN examples | ❌ |

### Packaging

* **Build system:** `pyproject.toml` (PEP 517 / 621), built with
  `setuptools ≥ 61.0`.
* **Dependencies (core):** `numpy ≥ 1.22`, `scipy ≥ 1.10`,
  `scikit-learn ≥ 1.2`.
* **Optional extras:** `[gpu]` → `cupy-cuda12x`, `[dev]` → `pytest`, `black`,
  `ruff`, `mypy`.
* **Distribution:** `MANIFEST.in` excludes benchmarks, demos, examples, tests,
  and docs from the sdist/wheel.
* **Install:** `pip install .` or `pip install -e .` works correctly.

### Test Suite

* **Framework:** pytest (17 tests in `tests/`).
* **Results:** 13 passing, 4 skipped (CuPy/CUDA not available in CI).
* **Covered:** `DiscreteOrientationSet` (init, factories, queries, GPU path),
  `DiscreteQuaternionIterator` (creation, traversal, depth limits), integration
  pipeline.
* **Not covered:** `QuaternionDepthIterator` (continuous traversal) — tests are
  commented out.  Utility functions in `spinstep/utils/` are untested
  directly.

### Code Quality

| Metric | Status |
|--------|--------|
| Type hints | 100 % (all public functions/methods) |
| Docstrings | ≈ 71 % (magic methods mostly missing) |
| Side effects at import | None (lazy imports for cupy, sklearn, healpy) |
| Linting config | `ruff` and `black` configured (88-char lines) |
| Type checking config | `mypy` strict mode configured |

### Known Issues

| # | Severity | Description |
|---|----------|-------------|
| 1 | Medium | `QuaternionDepthIterator` has **no test coverage** (tests commented out) |
| 2 | Low | `requirements.txt` pins `scikit-learn ≥ 1.5` while `pyproject.toml` pins `≥ 1.2` |
| 3 | Low | `healpy` is used by `get_unique_relative_spins()` but is not declared in optional dependencies |
| 4 | Low | README states "Python 3.8+" but `pyproject.toml` requires `≥ 3.9` |
| 5 | Low | `DiscreteQuaternionIterator` multiplies `magnitude()` by 2 — may over-estimate angular distances |

---

## 2. Functionalities

### Public API (`spinstep/__init__.py`)

The library exports four classes:

```python
from spinstep import (
    Node,                        # Tree node with quaternion orientation
    QuaternionDepthIterator,     # Continuous rotation-step traversal
    DiscreteOrientationSet,      # Queryable set of discrete orientations
    DiscreteQuaternionIterator,  # Discrete rotation-step traversal
)
```

#### `Node`
Tree node carrying a name, a unit quaternion `[x, y, z, w]`, and a list of
children.  Orientations are auto-normalised on construction.

```python
root = Node("root", [0, 0, 0, 1])
child = Node("child", [0.2588, 0, 0, 0.9659])
root.children.append(child)
```

#### `QuaternionDepthIterator`
Depth-first traversal that applies a **continuous** rotation step at each
visited node.  Only children within a configurable angular threshold of the
rotated state are pushed onto the stack.

```python
step = [0.2588, 0, 0, 0.9659]        # ≈ 30° around X
for node in QuaternionDepthIterator(root, step):
    print(node.name)
```

Key behaviours:
* Dynamic threshold: when `angle_threshold` is omitted, defaults to 30 % of the
  step angle (minimum 1°).
* Stack-based iteration — memory-efficient for large trees.

#### `DiscreteOrientationSet`
Container for a finite set of quaternion orientations with spatial querying.

```python
oset = DiscreteOrientationSet.from_cube()       # 24 octahedral orientations
oset = DiscreteOrientationSet.from_icosahedron() # 60 icosahedral orientations
oset = DiscreteOrientationSet.from_sphere_grid(200)  # Fibonacci-sphere
oset = DiscreteOrientationSet.from_custom(my_quats)  # user-supplied

indices = oset.query_within_angle([0, 0, 0, 1], angle=0.5)
```

Key behaviours:
* Auto-normalises all quaternions.
* Builds a `BallTree` (scikit-learn) when set size > 100 for O(log N) queries.
* Optional GPU path via CuPy (`use_cuda=True`).

#### `DiscreteQuaternionIterator`
Depth-first traversal that tries **every** orientation in a
`DiscreteOrientationSet` as a potential rotation step at each node.

```python
for node in DiscreteQuaternionIterator(root, oset, angle_threshold=0.4, max_depth=5):
    print(node.name)
```

Key behaviours:
* Cycle prevention via visited-node tracking.
* Configurable maximum depth.
* Supports any object with `.orientation` and `.children` attributes.

### Internal Utilities (`spinstep/utils/`)

These are **not** part of the public API but provide essential building blocks:

| Module | Key Functions |
|--------|---------------|
| `array_backend` | `get_array_module(use_cuda)` — returns NumPy or CuPy |
| `quaternion_math` | `batch_quaternion_angle(qs1, qs2, xp)` — pairwise angular distances |
| `quaternion_utils` | `quaternion_from_euler`, `quaternion_distance`, `rotate_quaternion`, `is_within_angle_threshold`, `quaternion_conjugate`, `quaternion_multiply`, `rotation_matrix_to_quaternion`, `get_relative_spin`, `get_unique_relative_spins` (HEALPix) |

---

## 3. Capabilities

### What SpinStep Can Do Today

1. **Build orientation-aware trees.**  
   Create tree structures where every node carries a 3D orientation (quaternion)
   rather than just positional data.

2. **Traverse by rotation.**  
   Walk a tree depth-first by *rotating* into branches: the traversal decision
   is based on angular proximity between the current orientation state and child
   orientations.

3. **Continuous traversal.**  
   Apply a single, fixed rotation step at every level
   (`QuaternionDepthIterator`).  Useful when the "direction of attention" rotates
   smoothly through the tree.

4. **Discrete traversal.**  
   Try a predefined catalogue of orientation steps at every node
   (`DiscreteQuaternionIterator`).  Useful for exploring orientation graphs with
   finite, known symmetry groups.

5. **Predefined symmetry groups.**  
   Quickly create orientation sets from the octahedral (24) and icosahedral (60)
   rotation groups, or from a Fibonacci-sphere sampling of arbitrary size.

6. **Efficient spatial queries.**  
   Find all orientations within a given angular distance.  Uses a BallTree for
   large sets (O(log N)) and brute-force for small sets.

7. **GPU acceleration.**  
   Offload orientation storage and distance computation to NVIDIA GPUs via CuPy
   (optional).  Transparent fallback to NumPy when CuPy is absent.

8. **Quaternion math utilities.**  
   Euler-to-quaternion conversion, Hamilton product, angular distance, threshold
   checking, rotation matrix conversion, and relative-spin computation between
   nodes.

9. **HEALPix integration.**  
   Compute unique relative rotations between neighbouring HEALPix pixels
   (requires optional `healpy` dependency).

### Example Use Cases

| Domain | Application |
|--------|-------------|
| **Robotics** | Joint-space search, end-effector reachability |
| **3D Scene Graphs** | Orientation-aware scene hierarchy traversal |
| **Spatial AI** | Directional attention, quaternion-based search |
| **Crystallography** | Orientation distribution sampling |
| **Animation** | Keyframe graph traversal, rotation blending paths |
| **Game Engines** | Camera orientation trees, procedural content |

---

## 4. Suggested Further Functionalities

### Short-term Improvements (patch releases)

| # | Suggestion | Rationale |
|---|------------|-----------|
| S1 | **Add tests for `QuaternionDepthIterator`** | The main continuous traversal API is currently untested (tests are commented out). |
| S2 | **Add tests for `spinstep/utils/` functions** | `quaternion_from_euler`, `quaternion_distance`, `rotate_quaternion`, etc. have no direct tests. |
| S3 | **Harmonise dependency versions** | `requirements.txt` (`scikit-learn ≥ 1.5`) and `pyproject.toml` (`≥ 1.2`) disagree. |
| S4 | **Declare `healpy` as an optional dependency** | `get_unique_relative_spins()` imports healpy at runtime but pyproject.toml does not list it. |
| S5 | **Fix Python version in README** | README says "Python 3.8+" but pyproject.toml requires ≥ 3.9. |

### Medium-term Enhancements (minor releases)

| # | Suggestion | Rationale |
|---|------------|-----------|
| M1 | **Breadth-first traversal iterator** | Currently only depth-first is available. BFS would serve level-order use cases (e.g., layer-by-layer 3D scene processing). |
| M2 | **A\* / best-first orientation search** | Priority-queue traversal ordered by angular proximity to a target orientation. Essential for path-planning in robotics. |
| M3 | **Interpolation (SLERP) stepping** | Allow smooth interpolation between orientations instead of discrete jumps. Useful for animation and smooth camera paths. |
| M4 | **Export utility functions in public API** | Functions like `quaternion_from_euler`, `quaternion_distance`, and `rotate_quaternion` are broadly useful; expose them as `spinstep.utils.*` or in `__init__.py`. |
| M5 | **Visualization module** | Matplotlib / Plotly helpers to visualize orientation trees, traversal paths, and orientation distributions on the sphere. |
| M6 | **Serialisation (save/load trees)** | JSON/pickle serialisation for Node trees so that orientation structures can be persisted and shared. |
| M7 | **Edge-weighted orientation graphs** | Generalise from trees to directed graphs where edges carry rotation costs or weights. |

### Long-term Extensions (major releases)

| # | Suggestion | Rationale |
|---|------------|-----------|
| L1 | **Dual-quaternion support** | Dual quaternions encode both rotation *and* translation. This would extend SpinStep from pure orientation to full rigid-body pose traversal. |
| L2 | **Quaternion Graph Neural Network (QGNN) integration** | Build on the existing `benchmark/qgnn_example.py` to provide a first-class QGNN layer that consumes SpinStep orientation sets. |
| L3 | **Streaming / incremental traversal** | Yield results lazily from very large or dynamically expanding trees (e.g., online robotics planning). |
| L4 | **Multi-resolution orientation grids** | Hierarchical orientation sets (coarse → fine) for adaptive-precision queries, similar to multi-level HEALPix. |
| L5 | **Parallel traversal back-end** | Use `multiprocessing` or `joblib` to parallelise independent sub-tree traversals for large-scale trees. |
| L6 | **ROS / USD integration** | Provide adapters for ROS `geometry_msgs/Quaternion` and Universal Scene Description (USD) schemas so SpinStep can be plugged into existing robotics and 3D pipelines. |
| L7 | **Benchmarking CI** | Integrate performance regression benchmarks (`benchmark/`) into CI/CD so that PRs report timing changes automatically. |

---

*End of report.*
