# quaternion_utils.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_from_euler(angles, order='zyx', degrees=True):
    """Convert Euler angles to a quaternion."""
    return R.from_euler(order, angles, degrees=degrees).as_quat()

def quaternion_distance(q1, q2):
    """Calculate angular distance between two quaternions (in radians)."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1.inv() * r2).magnitude()

def rotate_quaternion(q, rotation_step):
    """Apply a rotation step to quaternion q."""
    r1 = R.from_quat(q)
    step = R.from_quat(rotation_step)
    return (r1 * step).as_quat()

def is_within_angle_threshold(q_current, q_target, threshold_rad):
    """Check if two quaternions are within a given angular distance."""
    return quaternion_distance(q_current, q_target) < threshold_rad

# More utils

def quaternion_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quaternion_multiply(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])

def rotation_matrix_to_quaternion(m):
    t=np.trace(m)
    if(t>0): s=np.sqrt(t+1)*2; qw=0.25*s; qx=(m[2,1]-m[1,2])/s; qy=(m[0,2]-m[2,0])/s; qz=(m[1,0]-m[0,1])/s
    elif((m[0,0]>m[1,1])and(m[0,0]>m[2,2])): s=np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; qx=0.25*s; qw=(m[2,1]-m[1,2])/s; qy=(m[0,1]+m[1,0])/s; qz=(m[0,2]+m[2,0])/s
    elif(m[1,1]>m[2,2]): s=np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; qy=0.25*s; qw=(m[0,2]-m[2,0])/s; qx=(m[0,1]+m[1,0])/s; qz=(m[1,2]+m[2,1])/s
    else: s=np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; qz=0.25*s; qw=(m[1,0]-m[0,1])/s; qx=(m[0,2]+m[2,0])/s; qy=(m[1,2]+m[2,1])/s
    q=np.array([qx,qy,qz,qw]); n=np.linalg.norm(q); return q/n if n>1e-8 else np.array([0.,0.,0.,1.])

def get_relative_spin(nf,nt):
    qfc=quaternion_conjugate(nf.orientation); qr=quaternion_multiply(qfc,nt.orientation); n=np.linalg.norm(qr)
    return qr/n if n>1e-8 else np.array([0.,0.,0.,1.])

def get_unique_relative_spins(nodes, nside, nest, threshold=1e-3):
    spins, NPIX = [], hp.nside2npix(nside)
    for i in range(NPIX):
        nf=nodes[i]; nidx=hp.get_all_neighbours(nside,i,nest=nest)
        for idx in nidx:
            if idx!=-1:
                q=get_relative_spin(nf,nodes[idx])
                if q[3]<0: q=-q # Canonical form (w >= 0)
                is_uniq=True
                for s_q in spins:
                    dot=np.abs(np.dot(q,s_q)); dot=np.clip(dot,-1,1); angle=2*np.arccos(dot)
                    if angle<threshold: is_uniq=False; break
                if is_uniq: spins.append(q)
    print(f"Found {len(spins)} unique relative spins (approx threshold: {threshold*180/np.pi:.2f} deg).")
    return spins
