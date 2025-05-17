# node.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np

class Node:
    def __init__(self, name, orientation, children=None):
        arr = np.array(orientation, dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"Orientation must be a quaternion [x,y,z,w], got shape {arr.shape}")
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Orientation quaternion must be non-zero")
        self.orientation = arr / norm
        self.name = name
        self.children = list(children) if children else []
