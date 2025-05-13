import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import BallTree

class DiscreteOrientationSet:
    def __init__(self, orientations):
        arr = np.array(orientations)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Each orientation must be a quaternion [x, y, z, w]")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Zero or near-zero quaternion in orientation set")
        arr = arr / norms[:, None]
        self.orientations = arr

        # Precompute rotation vectors for BallTree
        self.rotvecs = R.from_quat(arr).as_rotvec()
        if len(arr) > 100:
            self._balltree = BallTree(self.rotvecs)
        else:
            self._balltree = None

    def query_within_angle(self, quat, angle):
        """Return indices of orientations within the given angle of quat."""
        rv = R.from_quat(quat).as_rotvec().reshape(1, -1)
        if self._balltree is not None:
            # BallTree in rotation vector space, Euclidean distance ≈ angle for small rotations
            inds = self._balltree.query_radius(rv, r=angle)[0]
        else:
            # Brute force for small sets
            dists = np.linalg.norm(self.rotvecs - rv, axis=1)
            inds = np.where(dists < angle)[0]
        return inds

    # ... rest unchanged ...

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
