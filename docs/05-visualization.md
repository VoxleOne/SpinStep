# Core Theory and Control in SpinStep

SpinStep models traversal as movement through **orientations**, not positions.
Every node in the graph represents a quaternion — a rotation state — and edges
between nodes are defined by **angular compatibility**, not Euclidean distance.

> The visualization is not a map of positions.
> It is a representation of orientation relationships.

---

## Observer-Centered Model

SpinStep uses an observer-centered spherical model:

- The **observer** sits at the origin.
- **Concentric layers** (radii) encode distance from the observer.
- A guided vehicle (node) is located by a **quaternion** (direction from the
  observer) and a **distance** (radial layer).

Reachability depends on the observer's current orientation. Changing orientation
changes which nodes are reachable — traversal is **observer-dependent**.

---

## Nodes, Edges, and Distance

| Term     | Meaning in SpinStep         |
| -------- | --------------------------- |
| Node     | Orientation (quaternion)    |
| Edge     | Angular compatibility       |
| Distance | Angular distance (radians)  |
| Movement | Orientation change          |

### Edge Construction

Two nodes are connected when their angular distance falls within a threshold.
SpinStep provides `is_within_angle_threshold()` for this:

```python
from spinstep.math import is_within_angle_threshold

connected = is_within_angle_threshold(q_current, q_target, threshold_rad=0.3)
```

To build connections across a set of orientations:

```python
def connect_by_angle(src_quats, dst_quats, threshold):
    """Connect orientations whose angular distance is within threshold."""
    connections = []
    for q1 in src_quats:
        for q2 in dst_quats:
            if is_within_angle_threshold(q1, q2, threshold):
                connections.append((q1, q2))
    return connections
```

Edges are never determined by spatial proximity or nearest-neighbor searches.
They are defined solely by **orientation-based adjacency** using angular
constraints.

---

## Minimal Example

1. Start at orientation A
2. Compute all orientations within angular threshold → reachable set
3. Move to orientation B
4. Reachable set changes

Traversal is defined by orientation, not position.

---

## Quaternion-to-Vector Mapping

For rendering purposes only, a quaternion can be projected to a 3D point using
`forward_vector_from_quaternion()`:

```python
from spinstep.math import forward_vector_from_quaternion

def quat_to_point(q, radius):
    """Map a quaternion to a 3D point for plotting."""
    v = forward_vector_from_quaternion(q)
    return radius * v
```

This converts an orientation into a direction vector scaled by the layer radius.
The resulting point is **not** the node's position — it is a rendering
convenience.

---

## Visualization (Illustrative Only)

The plots below project orientation nodes onto Euclidean space so the graph
structure can be inspected visually.

> **Important:**
>
> This visualization uses Euclidean positions only as a rendering convenience.
>
> - Nodes are NOT positions in space
> - Nodes represent orientations (quaternions)
> - Edges shown here are approximate
>
> In SpinStep, edges must be defined using angular constraints
> (e.g., `is_within_angle_threshold()`), not Euclidean distance.

### Static Graph

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from spinstep.math import forward_vector_from_quaternion, is_within_angle_threshold


def generate_orientation_nodes(num_lat, num_lon):
    """Generate quaternion nodes sampled over a sphere."""
    nodes = []
    for i in range(1, num_lat):  # avoid poles for simplicity
        theta = np.pi * i / num_lat
        for j in range(num_lon):
            phi = 2 * np.pi * j / num_lon
            # Build a quaternion from spherical angles
            rotvec = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]) * theta  # scale by polar angle
            q = Rotation.from_rotvec(rotvec).as_quat()  # [x, y, z, w]
            nodes.append(q)
    return nodes


def quat_to_point(q, radius):
    """Map a quaternion to a 3D point for plotting."""
    v = forward_vector_from_quaternion(q)
    return radius * v


def connect_by_angle(src_quats, dst_quats, threshold):
    """Connect orientations whose angular distance is within threshold."""
    connections = []
    for q1 in src_quats:
        for q2 in dst_quats:
            if is_within_angle_threshold(q1, q2, threshold):
                connections.append((q1, q2))
    return connections


# Parameters
radii = [1, 2, 3]
num_lat, num_lon = 6, 12
threshold = 0.5  # radians
colors = ['r', 'g', 'b']
all_quats = []   # quaternion nodes per layer
all_points = []  # projected 3D points per layer (for plotting only)
all_edges = []

# Generate orientation nodes per layer
for r in radii:
    quats = generate_orientation_nodes(num_lat, num_lon)
    points = np.array([quat_to_point(q, r) for q in quats])
    all_quats.append(quats)
    all_points.append(points)

# Connect adjacent layers by angular compatibility
for i in range(len(all_quats) - 1):
    edges = connect_by_angle(all_quats[i], all_quats[i + 1], threshold)
    all_edges.extend([
        (quat_to_point(q1, radii[i]), quat_to_point(q2, radii[i + 1]))
        for q1, q2 in edges
    ])

# Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for idx, points in enumerate(all_points):
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, color=colors[idx], label=f'Layer {idx+1}', alpha=0.6)

ax.scatter(0, 0, 0, color='k', s=100, label='Observer')

for a, b in all_edges:
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='gray', alpha=0.4)

ax.set_title("Orientation Graph — Angular Compatibility Edges")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
```

### Animated Traversal

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from spinstep.math import forward_vector_from_quaternion, is_within_angle_threshold


def generate_orientation_nodes(num_lat, num_lon):
    """Generate quaternion nodes sampled over a sphere."""
    nodes = []
    for i in range(1, num_lat):
        theta = np.pi * i / num_lat
        for j in range(num_lon):
            phi = 2 * np.pi * j / num_lon
            rotvec = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]) * theta
            q = Rotation.from_rotvec(rotvec).as_quat()
            nodes.append(q)
    return nodes


def quat_to_point(q, radius):
    """Map a quaternion to a 3D point for plotting."""
    v = forward_vector_from_quaternion(q)
    return radius * v


def connect_by_angle(src_quats, dst_quats, threshold):
    """Connect orientations whose angular distance is within threshold."""
    connections = []
    for q1 in src_quats:
        for q2 in dst_quats:
            if is_within_angle_threshold(q1, q2, threshold):
                connections.append((q1, q2))
    return connections


# Generate orientation nodes and angular-compatibility edges
radii = [1, 2, 3]
num_lat, num_lon = 6, 12
threshold = 0.5  # radians
colors = ['r', 'g', 'b']
all_quats, all_points, all_edges = [], [], []

for r in radii:
    quats = generate_orientation_nodes(num_lat, num_lon)
    points = np.array([quat_to_point(q, r) for q in quats])
    all_quats.append(quats)
    all_points.append(points)

for i in range(len(all_quats) - 1):
    edges = connect_by_angle(all_quats[i], all_quats[i + 1], threshold)
    all_edges.extend([
        (quat_to_point(q1, radii[i]), quat_to_point(q2, radii[i + 1]))
        for q1, q2 in edges
    ])

# Set up 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for idx, points in enumerate(all_points):
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, color=colors[idx], label=f'Layer {idx+1}', alpha=0.6)
ax.scatter(0, 0, 0, color='k', s=100, label='Observer')
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3.5, 3.5])
ax.set_zlim([-3.5, 3.5])
ax.set_title("Animated Traversal — Orientation Graph")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Animation logic
lines = []

def init():
    return lines

def animate(i):
    if i < len(all_edges):
        a, b = all_edges[i]
        line, = ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='orange', lw=2)
        lines.append(line)
    return lines

ani = animation.FuncAnimation(fig, animate, frames=len(all_edges), init_func=init,
                              interval=100, blit=False, repeat=False)

# To display in Jupyter Notebook:
from IPython.display import HTML
HTML(ani.to_jshtml())
```

**Tip**: Export as `.mp4` or `.gif`:

```python
ani.save("orientation_traversal.mp4", writer="ffmpeg", fps=10)
```

---
[⬅️ 04. Use Cases](04-use-cases.md) | [🏠 Home](index.md) | [06. Discrete Traversal ➡️](06-discrete-traversal.md)
