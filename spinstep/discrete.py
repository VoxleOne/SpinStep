import numpy as np
from scipy.spatial.transform import Rotation as R

class DiscreteOrientationSet:
    """
    Represents a set of discrete orientations (as quaternions) for traversal.
    Useful for cube/icosahedral symmetry, grid sampling, or user-defined sets.
    """

    def __init__(self, orientations):
        """
        orientations: (N, 4) array-like of quaternions [x, y, z, w]
        """
        self.orientations = np.array(orientations)

    @classmethod
    def from_cube(cls):
        """
        24 rotational symmetries of the cube (the rotational octahedral group).
        """
        # Cube symmetries: axis-angle (90, 180, 270 about x, y, z), identity, etc.
        # For brevity, we use scipy's built-in list for the octahedral group.
        group = R.create_group('O')
        return cls(group.as_quat())

    @classmethod
    def from_icosahedron(cls):
        """
        60 rotational symmetries of the icosahedron (the full icosahedral group).
        """
        group = R.create_group('I')
        return cls(group.as_quat())

    @classmethod
    def from_custom(cls, quat_list):
        """
        Accept a user-supplied list of quaternions [x, y, z, w].
        """
        return cls(quat_list)

    @classmethod
    def from_sphere_grid(cls, n_points=100):
        """
        Uniformly sample n_points orientations on the sphere (using random or deterministic grid).
        """
        # Fibonacci sphere sampling
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        vecs = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ], axis=-1)
        quats = [R.from_rotvec(v * np.pi).as_quat() for v in vecs]
        return cls(quats)

    def __len__(self):
        return len(self.orientations)