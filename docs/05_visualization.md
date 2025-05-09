# Visualization

Example of concentric spherical graph with **edges** and a possible **traversal visualization**.  We can:

1. **Connect each node to the trunk** (center node),
2. **Connect nodes between layers** (e.g., closest node in next layer),
3. Optionally **animate a traversal**, like BFS or DFS.

Here's a code that adds edges from each node to the center and to corresponding nodes on adjacent layers for a visual traversal structure:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_sphere_nodes(num_lat, num_lon, radius=1.0):
    """Generate nodes over a sphere using latitude/longitude-like placement."""
    nodes = []
    for i in range(1, num_lat):  # avoid poles
        theta = np.pi * i / num_lat
        z = radius * np.cos(theta)
        r_xy = radius * np.sin(theta)
        for j in range(num_lon):
            phi = 2 * np.pi * j / num_lon
            x = r_xy * np.cos(phi)
            y = r_xy * np.sin(phi)
            nodes.append((x, y, z))
    return np.array(nodes)

# Parameters
radii = [1, 2, 3]
num_lat, num_lon = 6, 12
colors = ['r', 'g', 'b']
center = np.array([0, 0, 0])

# Generate nodes
all_nodes = [generate_sphere_nodes(num_lat, num_lon, radius=r) for r in radii]

# Create plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
for idx, nodes in enumerate(all_nodes):
    xs, ys, zs = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    ax.scatter(xs, ys, zs, color=colors[idx], label=f'Radius {radii[idx]}', alpha=0.6)

# Plot center node (trunk)
ax.scatter(*center, color='k', s=100, label='Center (Trunk)')

# Draw edges to trunk
for layer in all_nodes:
    for node in layer:
        ax.plot([node[0], 0], [node[1], 0], [node[2], 0], color='gray', alpha=0.3)

# Draw edges between layers
for i in range(len(all_nodes) - 1):
    inner_layer = all_nodes[i]
    outer_layer = all_nodes[i + 1]
    for idx, node in enumerate(inner_layer):
        # Connect to the corresponding node in the next layer (mod to handle mismatches)
        outer_node = outer_layer[idx % len(outer_layer)]
        ax.plot(
            [node[0], outer_node[0]],
            [node[1], outer_node[1]],
            [node[2], outer_node[2]],
            color='black',
            linewidth=0.5,
            alpha=0.5
        )

# Labels and layout
ax.set_title("Spherical Graph with Edges and Traversal Structure")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
```

### Output:

* **Edges to the center**: Simulates tree "roots" from outer nodes.
* **Edges between layers**: Simulates upward traversal or graph expansion.
* Light `alpha` values keep the plot readable even with many lines.
  
## Adding a third dimension - FS Analogy

In the SpinStep traversal framework, where movement is represented as rotations using quaternions rather than positional steps, a file system modeled as a concentric spherical graph introduces a unique analogy for a "directory." Let’s address the equivalent of a directory in this context, considering the provided code and visualization.

### Analogy of a Directory in a Spherical Graph File System
In a traditional file system, a directory is a container that organizes files and other directories, forming a hierarchical structure. In a spherical graph where nodes are arranged on concentric spheres (layers) and connected to a central "trunk" node and adjacent layers, the equivalent of a directory could be conceptualized as:

1. **A Layer (Sphere) as a Directory**:
   - Each concentric sphere (defined by a radius in the code) can represent a directory. Nodes on a given sphere are analogous to files or subdirectories within that directory.
   - The radius of the sphere corresponds to the "depth" or level in the hierarchy, with inner spheres representing higher-level directories and outer spheres representing nested subdirectories.
   - For example, in the code, `radii = [1, 2, 3]` defines three layers. The sphere at radius 1 could be the root directory, radius 2 a subdirectory, and radius 3 a further nested subdirectory.

2. **A Node Cluster or Sector as a Directory**:
   - Alternatively, a directory could be a subset of nodes on a sphere, grouped by their angular coordinates (e.g., a sector defined by ranges of theta and phi in the `generate_sphere_nodes` function). This is akin to partitioning a directory into subfolders based on some criterion.
   - For instance, nodes with similar longitude (phi) values could represent files grouped under a common directory, with edges connecting them to indicate relationships.

3. **The Trunk Connection as a Directory Reference**:
   - Since each node is connected to the central trunk (as implemented in the code with edges drawn to the center `[0, 0, 0]`), the trunk could represent the root of the file system. A directory might then be defined as a collection of nodes that share a common path back to the trunk, with the path encoded as a sequence of rotations (quaternions).
   - Traversing from a node to the trunk and then to another node on a different layer mimics navigating up to a parent directory and down to a subdirectory.

4. **Quaternion-Based Traversal as Directory Navigation**:
   - In SpinStep, movement is a rotation defined by quaternions. A directory could be analogous to a "rotational context" — a set of nodes reachable by applying a specific sequence of rotations from the trunk or another reference node.
   - For example, navigating to a directory might involve rotating from the trunk to a node on a sphere (e.g., a specific theta/phi coordinate) and then exploring connected nodes on that sphere or the next layer, as shown in the code’s inter-layer edges.

### Mapping to the Code and Visualization
The provided code generates a spherical graph with:
- **Nodes** on concentric spheres at radii `[1, 2, 3]`, created using `generate_sphere_nodes`.
- **Edges** connecting each node to the trunk (center) and to corresponding nodes in adjacent layers.
- **Visualization** showing nodes as scattered points and edges as lines, with different colors for each layer.

In this structure:
- **Directory as a Sphere**: Each sphere (e.g., radius 1, 2, or 3) could be a directory. The nodes on the sphere are files or subdirectories, and edges to the trunk represent the path to the root. Inter-layer edges (e.g., from radius 1 to radius 2) represent navigation to a subdirectory.
- **Traversal**: A traversal like BFS or DFS, as mentioned, could simulate navigating the file system. For example:
  - **BFS** might explore all nodes (files) on the current sphere (directory) before moving to the next sphere (subdirectory) via inter-layer edges.
  - **DFS** might follow a path from the trunk to a node, then to a connected node on the next layer, diving deeper into the hierarchy.
- **Quaternion Representation**: Each edge traversal could be encoded as a quaternion rotation. For instance, moving from the trunk to a node at `(x, y, z)` on radius 1 involves a rotation defined by the node’s spherical coordinates (theta, phi). Moving to the next layer (radius 2) involves an additional rotation, scaling the radius while adjusting angular positions.
### Example: Directory Navigation in the Graph
Suppose the trunk is the root directory (`/`), and the sphere at radius 1 is a directory `/home`. Nodes on this sphere are files or subdirectories (e.g., `/home/user1`, `/home/user2`). The sphere at radius 2 might represent `/home/user1/docs`, with nodes as files like `doc1.txt`, `doc2.txt`.

- **Navigating to `/home`**:
  - Start at the trunk (center).
  - Apply a quaternion rotation to reach a node on the radius 1 sphere (e.g., a node at theta = π/3, phi = π/2).
- **Navigating to `/home/user1/docs`**:
  - From the node on radius 1, follow an inter-layer edge to a corresponding node on the radius 2 sphere (as coded in the loop connecting `inner_layer` to `outer_layer`).
  - This traversal is a rotation that scales the radius and adjusts the angular position.
- **Listing Files**:
  - Explore all nodes on the radius 2 sphere (e.g., using BFS to visit all nodes at that level).
- **Returning to Root**:
  - Follow the edge back to the trunk (a rotation reversing the path to the center).

### Animation of Traversal
The code could be extended to animate traversal (e.g., BFS or DFS):
- **BFS Animation**: Highlight nodes on one sphere at a time, then move to the next sphere, visualizing the exploration of directories layer by layer.
- **DFS Animation**: Trace a path from the trunk to a node, then to a connected node on the next layer, diving deeper, and backtrack to explore other branches.
- Use Matplotlib’s `FuncAnimation` to update the plot, highlighting visited nodes and edges in sequence.

### Challenges and Considerations
- **Node-to-File Mapping**: In a real file system, files have names and metadata. You’d need a way to map nodes (defined by `(x, y, z)`) to file names, perhaps using a dictionary or metadata stored at each node.
- **Quaternion Precision**: Since SpinStep uses quaternions, traversals must account for numerical precision in rotations to avoid drift when navigating between nodes.
- **Scalability**: The code generates a fixed number of nodes per sphere (`num_lat * num_lon`). A real file system might require dynamic node creation as files are added.

### Wrap-up
In a spherical graph file system using the SpinStep framework:
- A **directory** is analogous to a **concentric sphere** (e.g., radius 1 = `/home`, radius 2 = `/home/user1/docs`) or a **sector of nodes** on a sphere grouped by angular coordinates.
- **Nodes** on a sphere represent files or subdirectories.
- **Edges** to the trunk represent paths to the root, and inter-layer edges represent navigation to subdirectories.
- **Traversal** (e.g., BFS or DFS) mimics exploring directories, with each step encoded as a quaternion rotation.
- The provided code visualizes this structure, with spheres as directories, nodes as files, and edges as navigational paths. Extend it with animations or metadata to simulate file system operations.
