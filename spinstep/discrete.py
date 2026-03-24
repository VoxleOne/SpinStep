# discrete.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from spinstep.utils.array_backend import get_array_module


class DiscreteOrientationSet:
    """A set of discrete quaternion orientations with spatial querying.

    Orientations are stored as unit quaternions ``[x, y, z, w]`` and can be
    queried efficiently by angular distance.  On CPU a
    :class:`~sklearn.neighbors.BallTree` is used for large sets; an optional
    CUDA path is available via CuPy.

    Parameters
    ----------
    orientations:
        Array of shape ``(N, 4)`` — one quaternion per row.
    use_cuda:
        When *True*, store orientations on the GPU using CuPy.
    """

    def __init__(
        self,
        orientations: ArrayLike,
        use_cuda: bool = False,
    ) -> None:
        if orientations is None:
            raise ValueError("Orientations cannot be None in DiscreteOrientationSet constructor")

        xp: Any = get_array_module(use_cuda)
        arr = xp.array(orientations)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Each orientation must be a quaternion [x, y, z, w]")

        norms = xp.linalg.norm(arr, axis=1)
        if xp.all(norms < 1e-8):
            raise ValueError(
                "All input orientations are zero or near-zero vectors. "
                "Quaternions must have non-zero magnitude."
            )
        else:
            valid_norms_mask = norms > 1e-8
            if xp.any(valid_norms_mask):
                arr[valid_norms_mask] = arr[valid_norms_mask] / norms[valid_norms_mask][:, None]

        self.orientations = arr
        self.xp: Any = xp
        self.use_cuda: bool = use_cuda
        self._balltree: Any = None

        if not use_cuda:
            from sklearn.neighbors import BallTree

            if self.orientations.shape[0] > 0:
                orientations_for_rotvec = self.orientations
                if hasattr(orientations_for_rotvec, "get"):
                    orientations_for_rotvec = orientations_for_rotvec.get()

                if orientations_for_rotvec.shape[0] > 0:
                    self.rotvecs: np.ndarray = R.from_quat(orientations_for_rotvec).as_rotvec()
                    if len(self.orientations) > 100:
                        self._balltree = BallTree(self.rotvecs)
                else:
                    self.rotvecs = np.empty((0, 3))
            else:
                self.rotvecs = np.empty((0, 3))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_within_angle(
        self,
        quat: ArrayLike,
        angle: float,
    ) -> np.ndarray:
        """Return indices of orientations within *angle* radians of *quat*.

        Parameters
        ----------
        quat:
            Query quaternion ``[x, y, z, w]`` or batch of shape ``(N, 4)``.
        angle:
            Maximum angular distance in radians.

        Returns
        -------
        numpy.ndarray
            Integer indices into :attr:`orientations`.
        """
        query_quat_np = np.asarray(quat)
        if query_quat_np.shape == (4,):
            query_rv = R.from_quat(query_quat_np).as_rotvec().reshape(1, -1)
        elif query_quat_np.ndim == 2 and query_quat_np.shape[1] == 4:
            query_rv = R.from_quat(query_quat_np).as_rotvec()
        else:
            raise ValueError(
                "Input quat must be a single quaternion [x,y,z,w] or an array of quaternions."
            )

        if self.use_cuda:
            rv_gpu = self.xp.array(query_rv)

            current_orientations_for_rotvec = self.orientations
            if hasattr(current_orientations_for_rotvec, "get"):
                current_orientations_for_rotvec_np = current_orientations_for_rotvec.get()
            else:
                current_orientations_for_rotvec_np = current_orientations_for_rotvec

            if current_orientations_for_rotvec_np.shape[0] > 0:
                orientations_rotvecs_np = R.from_quat(current_orientations_for_rotvec_np).as_rotvec()
                orientations_rotvecs_gpu = self.xp.array(orientations_rotvecs_np)
                dists = self.xp.linalg.norm(orientations_rotvecs_gpu - rv_gpu, axis=1)
                inds = self.xp.where(dists < angle)[0]
                return inds
            else:
                return self.xp.array([], dtype=int)

        else:
            if not hasattr(self, "rotvecs") or self.rotvecs.shape[0] == 0:
                return np.array([], dtype=int)

            if self._balltree is not None:
                inds = self._balltree.query_radius(query_rv, r=angle)[0]
            else:
                dists = np.linalg.norm(self.rotvecs - query_rv, axis=1)
                inds = np.where(dists < angle)[0]
            return inds

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_cube(cls) -> "DiscreteOrientationSet":
        """Create orientation set from octahedral symmetry group (24 orientations)."""
        group = R.create_group("O")
        return cls(group.as_quat())

    @classmethod
    def from_icosahedron(cls) -> "DiscreteOrientationSet":
        """Create orientation set from icosahedral symmetry group (60 orientations)."""
        group = R.create_group("I")
        return cls(group.as_quat())

    @classmethod
    def from_custom(cls, quat_list: ArrayLike) -> "DiscreteOrientationSet":
        """Create orientation set from a user-supplied list of quaternions.

        Parameters
        ----------
        quat_list:
            Array of shape ``(N, 4)`` — quaternions ``[x, y, z, w]``.
        """
        return cls(quat_list)

    @classmethod
    def from_sphere_grid(cls, n_points: int = 100) -> "DiscreteOrientationSet":
        """Create orientation set by Fibonacci-sphere sampling.

        Parameters
        ----------
        n_points:
            Number of orientations to generate.
        """
        if n_points <= 0:
            return cls(np.empty((0, 4)))
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z_vals = np.cos(phi)
        vecs = np.stack([x, y, z_vals], axis=-1)
        quats = R.from_rotvec(vecs * np.pi).as_quat()
        return cls(quats)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def as_numpy(self) -> np.ndarray:
        """Convert orientations to a NumPy array.

        Returns
        -------
        numpy.ndarray
            The orientations as a NumPy array.  If stored as a CuPy array
            on GPU, transfers to CPU first.
        """
        if hasattr(self.orientations, "get"):  # CuPy array
            return self.orientations.get()
        return np.asarray(self.orientations)

    def __len__(self) -> int:
        if hasattr(self, "orientations") and self.orientations is not None:
            return self.orientations.shape[0]
        return 0
