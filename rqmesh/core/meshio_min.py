# SPDX-License-Identifier: Apache-2.0
"""Minimal OBJ I/O and triangle-mesh adjacency helpers (NumPy only)."""
import numpy as np


def read_obj(path):
    V, F = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('v '):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('f '):
                idx = [int(t.split('/')[0]) for t in ln.split()[1:]]
                F.append([i - 1 if i > 0 else len(V) + i for i in idx])
    return np.asarray(V, float), F


def write_obj(path, V, F):
    with open(path, 'w') as fh:
        for v in V:
            fh.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for f in F:
            fh.write("f " + " ".join(str(int(i) + 1) for i in f) + "\n")


def read_obj_uv(path):
    """V, F, UV, FUV (FUV None when the file has no texture coordinates on every face)."""
    V, F, UV, FUV = [], [], [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('v '):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith('vt '):
                UV.append([float(x) for x in ln.split()[1:3]])
            elif ln.startswith('f '):
                tok = [t.split('/') for t in ln.split()[1:]]
                F.append([int(t[0]) - 1 if int(t[0]) > 0 else len(V) + int(t[0]) for t in tok])
                FUV.append([(int(t[1]) - 1 if int(t[1]) > 0 else len(UV) + int(t[1])) if len(t) > 1 and t[1] else -1
                            for t in tok])
    ok = UV and all(x >= 0 for f in FUV for x in f)
    return np.asarray(V, float), F, (np.asarray(UV, float) if ok else None), (FUV if ok else None)


def write_obj_uv(path, V, F, UV=None, FUV=None):
    with open(path, 'w') as fh:
        for v in V:
            fh.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if UV is not None:
            for t in UV:
                fh.write(f"vt {t[0]:.8f} {t[1]:.8f}\n")
        for i, f in enumerate(F):
            if UV is not None:
                fh.write("f " + " ".join(f"{int(a) + 1}/{int(b) + 1}" for a, b in zip(f, FUV[i])) + "\n")
            else:
                fh.write("f " + " ".join(str(int(a) + 1) for a in f) + "\n")


def weld_uv(F, FUV, UV, tol=1e-7):
    """Re-index UVs so that corners of the same vertex with the same UV share one index (one index per vertex and
    island) -- what the ring collapse needs."""
    key, out, NF = {}, [], []
    for f, fu in zip(F, FUV):
        g = []
        for v, t in zip(f, fu):
            k = (int(v), int(round(UV[t][0] / tol)), int(round(UV[t][1] / tol)))
            if k not in key: key[k] = len(out); out.append(UV[t])
            g.append(key[k])
        NF.append(g)
    return NF, np.asarray(out, float)


def triangulate(F):
    return np.array([[f[0], f[k], f[k + 1]] for f in F for k in range(1, len(f) - 1)], int)


class TriMesh:
    """Triangle mesh with face normals, areas and a manifold edge->face map."""

    def __init__(self, V, T):
        self.V = np.asarray(V, float)
        self.T = np.asarray(T, int)
        a, b, c = (self.V[self.T[:, i]] for i in range(3))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        self.area = 0.5 * ln
        self.N = n / np.maximum(ln, 1e-30)[:, None]
        self.C = (a + b + c) / 3.0
        # half-edges: face f, local k -> edge (T[f,k], T[f,(k+1)%3])
        he = np.stack([self.T, np.roll(self.T, -1, axis=1)], axis=2).reshape(-1, 2)
        key = np.sort(he, axis=1)
        self.E, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        nE = len(self.E)
        self.he_edge = inv.reshape(-1, 3)                       # (F,3) edge id of local edge k
        cnt = np.bincount(inv, minlength=nE)
        self.edge_nfaces = cnt
        ef = -np.ones((nE, 2), int)
        ek = -np.ones((nE, 2), int)
        order = np.argsort(inv, kind='stable')
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        for slot in range(2):
            m = cnt > slot
            hidx = order[start[m] + slot]
            ef[m, slot] = hidx // 3
            ek[m, slot] = hidx % 3
        self.EF, self.EK = ef, ek                               # first two incident faces / local index
        self.interior = cnt == 2
        self.boundary = cnt == 1
        self.nonmanifold = cnt > 2
        e = self.V[self.E[:, 1]] - self.V[self.E[:, 0]]
        self.elen = np.linalg.norm(e, axis=1)
        self.edir = e / np.maximum(self.elen, 1e-30)[:, None]
        self.emid = 0.5 * (self.V[self.E[:, 0]] + self.V[self.E[:, 1]])
        # signed dihedral (turning) angle, positive = convex
        th = np.zeros(nE)
        cvx = np.zeros(nE, bool)
        m = self.interior
        n1, n2 = self.N[ef[m, 0]], self.N[ef[m, 1]]
        th[m] = np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2), -1, 1))
        cvx[m] = np.einsum('ij,ij->i', n1, self.C[ef[m, 1]] - self.C[ef[m, 0]]) < 0
        self.theta = th
        self.convex = cvx

    @property
    def diag(self):
        return float(np.linalg.norm(self.V.max(0) - self.V.min(0)))

    @property
    def total_area(self):
        return float(self.area.sum())
