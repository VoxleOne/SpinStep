# discrete.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
# from scipy.spatial.transform import Rotation as R # Keep this, but we'll also try importing inside the method for test
from spinstep.utils.array_backend import get_array_module
import sys # For diagnostics

class DiscreteOrientationSet:
    def __init__(self, orientations, use_cuda=False):
        # Try importing R here too, just to see if scope is an issue for __init__
        from scipy.spatial.transform import Rotation as R_init_check 
        print(f"[DiscreteOrientationSet __init__] R_init_check is: {R_init_check}")

        xp = get_array_module(use_cuda)
        arr = xp.array(orientations)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Each orientation must be a quaternion [x, y, z, w]")
        norms = xp.linalg.norm(arr, axis=1)
        if xp.any(norms < 1e-8):
            raise ValueError("Zero or near-zero quaternion in orientation set")
        valid_norms = norms > 1e-8
        arr[valid_norms] = arr[valid_norms] / norms[valid_norms][:, None]
        self.orientations = arr
        self.xp = xp
        self.use_cuda = use_cuda

        self._balltree = None
        if not use_cuda:
            from sklearn.neighbors import BallTree
            # Use the R imported at module level if available, or the local one for test
            try:
                from scipy.spatial.transform import Rotation as R_balltree_check
                print(f"[DiscreteOrientationSet __init__ BallTree] R_balltree_check is: {R_balltree_check}")
                self.rotvecs = R_balltree_check.from_quat(self.orientations).as_rotvec()
            except NameError:
                 print("[DiscreteOrientationSet __init__ BallTree] R was NOT defined here trying module R")
                 # This would indicate the module level R failed.
                 # As a fallback for extreme debugging, import it directly:
                 from scipy.spatial.transform import Rotation as R_local_bt
                 self.rotvecs = R_local_bt.from_quat(self.orientations).as_rotvec()


            if len(self.orientations) > 100:
                self._balltree = BallTree(self.rotvecs)


    # ... (other methods like query_within_angle, as_numpy - ensure they also use a defined R) ...
    # For query_within_angle, if it uses R, ensure it's defined.
    def query_within_angle(self, quat, angle):
        from scipy.spatial.transform import Rotation as R_query_check # Test import
        print(f"[DiscreteOrientationSet query_within_angle] R_query_check is: {R_query_check}")
        # ... rest of query_within_angle
        query_quat_np = np.asarray(quat)
        if query_quat_np.shape == (4,):
             query_rv = R_query_check.from_quat(query_quat_np).as_rotvec().reshape(1, -1)
        elif query_quat_np.ndim == 2 and query_quat_np.shape[1] == 4:
             query_rv = R_query_check.from_quat(query_quat_np).as_rotvec()
        else:
            raise ValueError("Input quat must be a single quaternion [x,y,z,w] or an array of quaternions.")
        # ... (the rest of the method for CPU/GPU)

        if self.use_cuda:
            rv_gpu = self.xp.array(query_rv)
            if not hasattr(self, 'orientations_rotvecs_gpu'):
                 self.orientations_rotvecs_gpu = self.xp.array(R_query_check.from_quat(self.xp.asnumpy(self.orientations)).as_rotvec())
            dists = self.xp.linalg.norm(self.orientations_rotvecs_gpu - rv_gpu, axis=1)
            inds = self.xp.where(dists < angle)[0]
            return inds
        else:
            if self._balltree is not None:
                inds = self._balltree.query_radius(query_rv, r=angle)[0]
            else:
                dists = np.linalg.norm(self.rotvecs - query_rv, axis=1)
                inds = np.where(dists < angle)[0]
            return inds


    @classmethod
    def from_cube(cls):
        print(f"[DiscreteOrientationSet from_cube] Attempting to use R. Python sys.path: {sys.path}")
        print(f"[DiscreteOrientationSet from_cube] scipy.spatial.transform.Rotation is available as R? {'R' in globals()}")
        # ---- VERY EXPLICIT IMPORT FOR TESTING ----
        from scipy.spatial.transform import Rotation as R_local_cube 
        print(f"[DiscreteOrientationSet from_cube] R_local_cube is: {R_local_cube}")
        # ---- END EXPLICIT IMPORT ----
        group = R_local_cube.create_group('O') # Use the locally imported R_local_cube
        return cls(group.as_quat())

    @classmethod
    def from_icosahedron(cls):
        from scipy.spatial.transform import Rotation as R_local_ico # Test import
        group = R_local_ico.create_group('I')
        return cls(group.as_quat())

    @classmethod
    def from_custom(cls, quat_list):
        return cls(quat_list)

    @classmethod
    def from_sphere_grid(cls, n_points=100):
        from scipy.spatial.transform import Rotation as R_local_sphere # Test import
        if n_points <= 0:
            return cls(np.empty((0,4)))
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z_vals = np.cos(phi)
        vecs = np.stack([x, y, z_vals], axis=-1)
        quats = R_local_sphere.from_rotvec(vecs * np.pi).as_quat()
        return cls(quats)

    def __len__(self):
        return len(self.orientations)
