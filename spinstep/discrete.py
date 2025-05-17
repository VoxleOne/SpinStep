# discrete.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

from spinstep.utils.array_backend import get_array_module

class DiscreteOrientationSet:
    def __init__(self, orientations, use_cuda=False):
        xp = get_array_module(use_cuda)
        arr = xp.array(orientations)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Each orientation must be a quaternion [x, y, z, w]")
        norms = xp.linalg.norm(arr, axis=1)
        if xp.any(norms < 1e-8):
            raise ValueError("Zero or near-zero quaternion in orientation set")
        arr = arr / norms[:, None]
        self.orientations = arr
        self.xp = xp
        self.use_cuda = use_cuda

        # Only for CPU/NumPy mode: BallTree for fast queries
        self._balltree = None
        if not use_cuda:
            from scipy.spatial.transform import Rotation as R
            from sklearn.neighbors import BallTree
            self.rotvecs = R.from_quat(arr).as_rotvec()
            if len(arr) > 100:
                self._balltree = BallTree(self.rotvecs)
            else:
                self._balltree = None

    def query_within_angle(self, quat, angle):
        """Return indices of orientations within the given angle of quat."""
        if self.use_cuda:
            # Brute force: batch math on GPU
            # Convert quat to rotvec on CPU, then broadcast to GPU
            import numpy as np
            from scipy.spatial.transform import Rotation as R
            rv = R.from_quat(np.array(quat)).as_rotvec().reshape(1, -1)
            rv_gpu = self.xp.array(rv)
            dists = self.xp.linalg.norm(self.orientations - rv_gpu, axis=1)
            inds = self.xp.where(dists < angle)[0]
            return inds
        else:
            from scipy.spatial.transform import Rotation as R
            rv = R.from_quat(quat).as_rotvec().reshape(1, -1)
            if self._balltree is not None:
                inds = self._balltree.query_radius(rv, r=angle)[0]
            else:
                dists = self.xp.linalg.norm(self.rotvecs - rv, axis=1)
                inds = self.xp.where(dists < angle)[0]
            return inds

    def as_numpy(self):
        if self.use_cuda:
            return self.xp.asnumpy(self.orientations)
        return self.orientations

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
