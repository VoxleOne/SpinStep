# quaternion_math.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

def batch_quaternion_angle(qs1, qs2, xp):
    """
    qs1: (N, 4) array
    qs2: (M, 4) array
    xp: array module (np or cp)
    Returns (N, M) array of angular distances.
    """
    # Quaternion inner product: angle = 2*arccos(|dot(q1, q2)|)
    dots = xp.abs(xp.dot(qs1, qs2.T))
    dots = xp.clip(dots, -1.0, 1.0)
    angles = 2 * xp.arccos(dots)
    return angles
