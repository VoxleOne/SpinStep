# Basic Example

Let’s walk through a simple example of how to use quaternions to place nodes evenly on a sphere, and then use those rotations to move between them — all while building the structure of our spherical graph.

## Setup: Sphere with Nodes at Radius r

Assume:

    We start with a unit vector, e.g., pointing “up” → v₀ = (0, 0, 1)

    Our goal: Rotate this vector in various directions to generate other points on a sphere of radius r

### Step 1: Rotate Around Axes Using Quaternions

To place a node at angle θ from the initial vector, do:

import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_vector(vector, axis, angle_deg):
    """Rotate a vector around an axis by a given angle in degrees using quaternions."""
    r = R.from_rotvec(np.radians(angle_deg) * np.array(axis))
    return r.apply(vector)

    vector: Your current direction (e.g., [0, 0, 1])

    axis: Axis to rotate around (e.g., X or Y or any tangent axis)

    angle_deg: Angle of rotation

### Step 2: Generate Nodes on Sphere Surface

Here’s how to generate points on a single spherical layer:

def generate_sphere_nodes(num_lat, num_lon, radius=1.0):
    """Generate nodes over a sphere using latitude/longitude-like placement."""
    nodes = []
    for i in range(1, num_lat):  # skip poles for now
        theta = np.pi * i / num_lat
        z = radius * np.cos(theta)
        r_xy = radius * np.sin(theta)
        for j in range(num_lon):
            phi = 2 * np.pi * j / num_lon
            x = r_xy * np.cos(phi)
            y = r_xy * np.sin(phi)
            nodes.append((x, y, z))
    return nodes

This gives us a structured grid of nodes on a spherical shell. We could then connect neighbors based on proximity or indexing.

### Step 3: Use Quaternions to Traverse Between Directions

To move from one direction to another:

    Get the direction vectors v1, v2

    Compute the quaternion that rotates v1 → v2:

def get_rotation_quaternion(v1, v2):
    """Find the quaternion that rotates v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if np.isclose(dot, -1.0):
        # 180-degree rotation
        orthogonal = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        return R.from_rotvec(np.pi * orthogonal)
    elif np.isclose(dot, 1.0):
        return R.identity()
    else:
        q = R.from_rotvec(np.arccos(dot) * cross / np.linalg.norm(cross))
        return q

Then apply this rotation to the structure to traverse or generate new orientations.

Connecting It All: Spherical Graph

    Layers = spheres of increasing radius

    Nodes on each sphere = use generate_sphere_nodes(...)

    Edges:

        Radial: From layer n to layer n+1 at same direction

        Tangential: Between nearby nodes on the same sphere, using quaternion rotations

        Global rotations: Use get_rotation_quaternion(...) to move in 3D directions

