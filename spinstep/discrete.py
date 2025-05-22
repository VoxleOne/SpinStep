print("!!!!!!!!!!!!!! EXECUTING THIS CLEANED VERSION OF discrete.py !!!!!!!!!!!!!!")
print(f"!!!!!!!!!!!!!! Date: { '2025-05-22 14:50:00 UTC (approx)' } !!!!!!!!!!!!!!")

import numpy as np
from scipy.spatial.transform import Rotation as R # Module-level import should be sufficient now
from spinstep.utils.array_backend import get_array_module
# import sys # Not needed for sys.path anymore for this issue

class DiscreteOrientationSet:
    def __init__(self, orientations, use_cuda=False): # Ensure this signature is correct
        print(f"[DiscreteOrientationSet __init__] Received orientations type: {type(orientations)}, use_cuda: {use_cuda}")
        if orientations is None:
            raise ValueError("Orientations cannot be None in DiscreteOrientationSet constructor")
        
        # Ensure R is available in __init__ if needed
        # print(f"[DiscreteOrientationSet __init__] R is: {R}") 

        xp = get_array_module(use_cuda)
        arr = xp.array(orientations) # This is where orientations is used
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Each orientation must be a quaternion [x, y, z, w]")
        
        norms = xp.linalg.norm(arr, axis=1)
        # Handle cases where all norms are zero to avoid division by zero if norms is scalar 0.
        if xp.all(norms < 1e-8): # Check if all norms are effectively zero
            # If all orientations are zero vectors, they remain zero vectors.
            # Or, raise an error if zero quaternions are not allowed.
            # For now, let's assume they remain as is, or consider raising ValueError.
             print("[DiscreteOrientationSet __init__] Warning: All input orientations are zero or near-zero vectors.")
        else: # Proceed with normalization for non-zero vectors
            valid_norms_mask = norms > 1e-8
            # Ensure arr[valid_norms_mask] is not empty before division
            if xp.any(valid_norms_mask):
                 arr[valid_norms_mask] = arr[valid_norms_mask] / norms[valid_norms_mask][:, None]
            else:
                # This case means all norms were < 1e-8 but not caught by xp.all above (e.g. mixed small/zero)
                # Behavior might need refinement if this state is problematic.
                print("[DiscreteOrientationSet __init__] Warning: All input orientations are effectively zero after filtering valid norms.")


        self.orientations = arr
        self.xp = xp
        self.use_cuda = use_cuda
        self._balltree = None

        if not use_cuda:
            from sklearn.neighbors import BallTree
            # Check if there are any orientations to process
            if self.orientations.shape[0] > 0:
                # Convert to NumPy array if it's a CuPy array for SciPy/Scikit-learn
                orientations_for_rotvec = self.orientations
                if hasattr(orientations_for_rotvec, 'get'): # Check if it's a CuPy array
                    orientations_for_rotvec = orientations_for_rotvec.get()
                
                # Ensure orientations_for_rotvec is not empty before R.from_quat
                if orientations_for_rotvec.shape[0] > 0:
                    self.rotvecs = R.from_quat(orientations_for_rotvec).as_rotvec()
                    if len(self.orientations) > 100: # Use self.orientations for original count
                        self._balltree = BallTree(self.rotvecs)
                else:
                    self.rotvecs = np.empty((0,3)) # Empty rotvecs if no orientations
            else:
                self.rotvecs = np.empty((0,3)) # Empty rotvecs if no orientations


    def query_within_angle(self, quat, angle):
        # print(f"[DiscreteOrientationSet query_within_angle] R is: {R}")
        query_quat_np = np.asarray(quat)
        if query_quat_np.shape == (4,):
             query_rv = R.from_quat(query_quat_np).as_rotvec().reshape(1, -1)
        elif query_quat_np.ndim == 2 and query_quat_np.shape[1] == 4:
             query_rv = R.from_quat(query_quat_np).as_rotvec()
        else:
            raise ValueError("Input quat must be a single quaternion [x,y,z,w] or an array of quaternions.")

        if self.use_cuda:
            # Ensure self.orientations is on GPU if it's not already
            # This logic assumes self.orientations might be CPU or GPU
            # For consistency, ensure rotvecs are derived from the current self.orientations
            
            # Convert query_rv to GPU
            rv_gpu = self.xp.array(query_rv)

            # Ensure orientations_rotvecs_gpu is created if not present or if orientations changed
            # This part needs careful handling if self.orientations can change after __init__
            # For now, assume it's based on __init__ state.
            current_orientations_for_rotvec = self.orientations
            if hasattr(current_orientations_for_rotvec, 'get'): # if cupy array
                current_orientations_for_rotvec_np = current_orientations_for_rotvec.get()
            else: # if numpy array
                current_orientations_for_rotvec_np = current_orientations_for_rotvec

            if current_orientations_for_rotvec_np.shape[0] > 0:
                orientations_rotvecs_np = R.from_quat(current_orientations_for_rotvec_np).as_rotvec()
                orientations_rotvecs_gpu = self.xp.array(orientations_rotvecs_np)
                dists = self.xp.linalg.norm(orientations_rotvecs_gpu - rv_gpu, axis=1) # rv_gpu might need broadcasting if query_rv was single
                inds = self.xp.where(dists < angle)[0]
                return inds
            else:
                return self.xp.array([], dtype=int) # No orientations to compare against

        else: # CPU path
            if not hasattr(self, 'rotvecs') or self.rotvecs.shape[0] == 0 : # Check if rotvecs exist and are not empty
                 return np.array([], dtype=int) # No orientations to compare against

            if self._balltree is not None:
                inds = self._balltree.query_radius(query_rv, r=angle)[0]
            else:
                dists = np.linalg.norm(self.rotvecs - query_rv, axis=1) # query_rv might need broadcasting
                inds = np.where(dists < angle)[0]
            return inds

    @classmethod
    def from_cube(cls):
        print("[DiscreteOrientationSet from_cube] Creating group...")
        group = R.create_group('O') 
        print("[DiscreteOrientationSet from_cube] Group created. Calling constructor.")
        return cls(group.as_quat()) # This should pass the quaternions to __init__

    @classmethod
    def from_icosahedron(cls):
        group = R.create_group('I')
        return cls(group.as_quat())

    @classmethod
    def from_custom(cls, quat_list):
        return cls(quat_list)

    @classmethod
    def from_sphere_grid(cls, n_points=100):
        if n_points <= 0:
            # Return an instance with empty orientations
            return cls(np.empty((0,4))) 
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z_vals = np.cos(phi)
        vecs = np.stack([x, y, z_vals], axis=-1)
        # For from_rotvec, vectors are scaled by angle. If these are direction vectors, angle is effectively pi?
        # Or are these axis vectors for a fixed rotation? The original paper for Fibonacci lattice is for points.
        # Assuming these vectors represent rotation axes and a common angle (e.g. pi for 180-degree symmetry, or specific angle)
        # If they are just directions, R.align_vectors might be more appropriate, or R.from_rotvec(vecs * some_angle)
        # For now, let's assume they are rotation vectors scaled by pi (180-degree rotations around these axes)
        # This part might need clarification based on intended use of "sphere grid" for orientations.
        # If it's just to get diverse quaternions, R.from_rotvec(vecs) might be enough if vecs are already scaled by angle.
        # A common way to get somewhat uniform quaternions is R.random(n_points, random_state=...).
        # Let's assume vecs * np.pi is the intended rotation vector.
        quats = R.from_rotvec(vecs * np.pi).as_quat()
        return cls(quats)

    def __len__(self):
        if hasattr(self, 'orientations') and self.orientations is not None:
            return self.orientations.shape[0]
        return 0
