# SPDX-License-Identifier: Apache-2.0
"""Skin weights as a field on the input surface: for keeping joint loops when simplifying a rigged mesh, and for
carrying the weights to any new mesh that lies on (or near) the input surface.

Why joint loops matter (RESULTS-round16): at a tight joint a low-poly mesh that keeps the cage's loops follows the
posed high-poly 2.7-5.5x more closely than a Decimate LOD of the same size, although Decimate is more accurate at
rest; a ring simplifier that removes the densest loops first removes exactly the joint loops (-23% elbow volume).
Under a pose, two neighbouring loops whose weights differ by dw rotate apart by about joint_deg * dw, so a ring's
removal cost counts that as extra bend (`structsimp.ring_collapse(skin=...)`)."""
import numpy as np
from .compat import BallIndex


def _closest_on_tris(P, A, B, C):
    """Closest points on triangles (A, B, C) to points P, all (n, 3) (Ericson's region test); barycentrics."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - B; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - C; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = va + vb + vc; den = np.where(den == 0, 1e-30, den)
    v = vb / den; w = vc / den; u = 1 - v - w                       # inside the triangle by default
    nz = lambda x: np.where(x == 0, 1e-30, x)
    m = (vc <= 0) & (d1 - d3 >= 0) & (d3 <= 0)                       # edge AB
    t = d1 / nz(d1 - d3); u[m], v[m], w[m] = 1 - t[m], t[m], 0.0
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                            # edge AC
    t = d2 / nz(d2 - d6); u[m], v[m], w[m] = 1 - t[m], 0.0, t[m]
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)                   # edge BC
    t = (d4 - d3) / nz((d4 - d3) + (d5 - d6)); u[m], v[m], w[m] = 0.0, 1 - t[m], t[m]
    m = (d1 <= 0) & (d2 <= 0); u[m], v[m], w[m] = 1.0, 0.0, 0.0       # vertex regions
    m = (d3 >= 0) & (d4 <= d3); u[m], v[m], w[m] = 0.0, 1.0, 0.0
    m = (d6 >= 0) & (d5 <= d6); u[m], v[m], w[m] = 0.0, 0.0, 1.0
    return np.c_[u, v, w]


class SkinField:
    """Weights W (n_vertices x n_groups) of the input mesh (V, F), queried anywhere near its surface."""

    def __init__(self, V, F, W):
        self.V = np.asarray(V, float); self.W = np.asarray(W, float)
        self.T = np.array([[f[0], f[k], f[k + 1]] for f in F for k in range(1, len(f) - 1)], np.int64)
        self.index = BallIndex(self.V)
        vt = [[] for _ in range(len(self.V))]
        for t, tri in enumerate(self.T):
            for v in tri: vt[v].append(t)
        # the triangles around each vertex and around its neighbours: where the closest surface point lies
        nb = [set() for _ in range(len(self.V))]
        for tri in self.T:
            for v in tri: nb[v].update(tri)
        cand = [sorted({t for u in nb[v] for t in vt[u]}) for v in range(len(self.V))]
        k = max((len(c) for c in cand), default=1)
        self.cand = np.array([c + [c[0]] * (k - len(c)) if c else [0] * k for c in cand], np.int64)

    def nearest_vertex(self, P):
        return np.asarray(self.index.query(np.asarray(P, float))[1], np.int64)

    def dw(self, Pa, Pb):
        """Weight change between point pairs (0..1: half the L1 distance of the weight vectors, nearest vertex)."""
        Wa, Wb = self.W[self.nearest_vertex(Pa)], self.W[self.nearest_vertex(Pb)]
        return 0.5 * np.abs(Wa - Wb).sum(1)

    def at(self, P):
        """Weights at points P, interpolated on the closest input triangle (as Blender's Data Transfer
        'nearest face interpolated')."""
        P = np.asarray(P, float); n = len(P)
        cand = self.cand[self.nearest_vertex(P)]
        best = np.full(n, np.inf); bt = np.zeros(n, np.int64); bb = np.zeros((n, 3))
        for j in range(cand.shape[1]):
            t = cand[:, j]; A, B, C = self.V[self.T[t, 0]], self.V[self.T[t, 1]], self.V[self.T[t, 2]]
            bary = _closest_on_tris(P, A, B, C)
            Q = A * bary[:, :1] + B * bary[:, 1:2] + C * bary[:, 2:]
            d = np.linalg.norm(P - Q, axis=1)
            m = d < best; best[m] = d[m]; bt[m] = t[m]; bb[m] = bary[m]
        return (self.W[self.T[bt]] * bb[:, :, None]).sum(1)
