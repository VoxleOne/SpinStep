# array_backend.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

def get_array_module(use_cuda=False):
    if use_cuda:
        try:
            import cupy as cp
            return cp
        except ImportError:
            print("[SpinStep] CuPy not found, falling back to NumPy.")
    import numpy as np
    return np
