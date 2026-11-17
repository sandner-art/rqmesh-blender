# SPDX-License-Identifier: Apache-2.0
"""QuadWild .rosy cross-field I/O (one representative tangent vector per face, N=4)."""
import numpy as np

def read_rosy(path):
    with open(path) as fh:
        n = int(fh.readline()); _ = int(fh.readline())
        return np.array([[float(x) for x in fh.readline().split()] for _ in range(n)])

def write_rosy(path, D):
    with open(path, 'w') as fh:
        fh.write(f"{len(D)}\n4\n")
        for d in D:
            fh.write(f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}\n")

def rotate_in_plane(D, N, ang):
    """Rotate per-face tangent vectors D about face normals N by ang (radians, scalar or per face)."""
    c, s = np.cos(ang), np.sin(ang)
    c = np.broadcast_to(c, len(D))[:, None]; s = np.broadcast_to(s, len(D))[:, None]
    return D * c + np.cross(N, D) * s + N * np.einsum('ij,ij->i', N, D)[:, None] * (1 - c)
