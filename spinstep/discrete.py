print("!!!!!!!!!!!!!! EXECUTING THIS VERSION OF discrete.py !!!!!!!!!!!!!!") # VERY TOP LINE
print("!!!!!!!!!!!!!! Date: 2025-05-22 13:35:00 UTC (approx) !!!!!!!!!!!!!!") # Add a timestamp

# discrete.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R 
from spinstep.utils.array_backend import get_array_module
import sys 

class DiscreteOrientationSet:
    # ... (rest of the class, including the from_cube method with R_local_cube)
    @classmethod
    def from_cube(cls):
        print(f"[DiscreteOrientationSet from_cube] TOP OF METHOD. Python sys.path: {sys.path}")
        print(f"[DiscreteOrientationSet from_cube] scipy.spatial.transform.Rotation is available as R? {'R' in globals()}")
        # ---- VERY EXPLICIT IMPORT FOR TESTING ----
        from scipy.spatial.transform import Rotation as R_local_cube 
        print(f"[DiscreteOrientationSet from_cube] R_local_cube is: {R_local_cube}")
        # ---- END EXPLICIT IMPORT ----
        group = R_local_cube.create_group('O') # Use the locally imported R_local_cube
        print("[DiscreteOrientationSet from_cube] Successfully created group.")
        return cls(group.as_quat())
    # ... (rest of the methods)
