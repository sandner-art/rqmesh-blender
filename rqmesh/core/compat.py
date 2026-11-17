# SPDX-License-Identifier: Apache-2.0
"""SciPy-free fallbacks so the core runs inside Blender's bundled Python (NumPy only).
Uses SciPy when it is importable, unless RQM_NO_SCIPY=1."""
import os
import numpy as np

try:
    if os.environ.get('RQM_NO_SCIPY') == '1':
        raise ImportError
    from scipy.spatial import cKDTree as _KD
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class _SortedGrid:
    """Vectorized exact nearest neighbour without SciPy: points sorted by grid cell; a query looks at its 27
    neighbouring cells and accepts the nearest candidate if it lies within one cell size (then nothing outside can
    be nearer).  Unresolved queries go to a coarser grid, then to chunked brute force."""

    def __init__(self, P, h):
        self.P, self.h = P, float(h); self.o = P.min(0) - h
        c = np.floor((P - self.o) / h).astype(np.int64); self.dim = c.max(0) + 3
        k = self._key(c); self.order = np.argsort(k, kind='stable'); self.sk = k[self.order]

    def _key(self, c):
        return (c[:, 0] * self.dim[1] + c[:, 1]) * self.dim[2] + c[:, 2]

    def nn(self, X):
        m = len(X); c = np.floor((X - self.o) / self.h).astype(np.int64)
        off = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        cc = (c[:, None, :] + off[None]).reshape(-1, 3)
        inside = np.all((cc >= 0) & (cc < self.dim), axis=1)
        key = np.where(inside, self._key(np.clip(cc, 0, self.dim - 1)), -1)
        lo = np.searchsorted(self.sk, key, 'left'); hi = np.searchsorted(self.sk, key, 'right')
        cnt = np.where(inside, hi - lo, 0); tot = int(cnt.sum())
        bd = np.full(m, np.inf); bi = np.full(m, -1, np.int64)
        if tot:                                         # candidates come grouped by query: segmented minima
            qi = np.repeat(np.arange(m * 27) // 27, cnt)
            start = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
            idx = self.order[start]; diff = self.P[idx] - X[qi]
            d = np.sqrt(np.einsum('ij,ij->i', diff, diff))
            qc = cnt.reshape(m, 27).sum(1); has = qc > 0; seg = np.r_[0, np.cumsum(qc[has])[:-1]]
            dmin = np.minimum.reduceat(d, seg)
            big = np.iinfo(np.int64).max
            imin = np.minimum.reduceat(np.where(d == np.repeat(dmin, qc[has]), idx, big), seg)
            bd[has] = dmin; bi[has] = imin
        return bd, bi, bd <= self.h


def _nn_exact(P, X, grids):
    X = np.atleast_2d(np.asarray(X, float)); d = np.full(len(X), np.inf); i = np.full(len(X), -1, np.int64)
    todo = np.arange(len(X))
    for g in grids:
        if not len(todo): break
        bd, bi, ok = g.nn(X[todo]); d[todo[ok]] = bd[ok]; i[todo[ok]] = bi[ok]; todo = todo[~ok]
    if len(todo):                                       # far queries: brute force (BLAS) in chunks
        pp = np.einsum('ij,ij->i', P, P)
        for s in range(0, len(todo), 1024):
            t = todo[s:s + 1024]; Xt = X[t]
            D2 = pp[None, :] - 2.0 * (Xt @ P.T)
            j = np.argmin(D2, axis=1); i[t] = j; d[t] = np.linalg.norm(P[j] - Xt, axis=1)
    return d, i


class BallIndex:
    """Ball queries on a point set: cKDTree if available, else a uniform grid hash."""

    def __init__(self, P, cell=None):
        self.P = np.asarray(P, float)
        if HAVE_SCIPY:
            self.tree = _KD(self.P); return
        self.tree = None
        ext = self.P.max(0) - self.P.min(0)
        self.cell = cell or float(max(ext.max() / 64.0, 1e-9))
        self.o = self.P.min(0)
        key = np.floor((self.P - self.o) / self.cell).astype(np.int64)
        self.keys = {}
        for i, k in enumerate(map(tuple, key)):
            self.keys.setdefault(k, []).append(i)

    def query_ball_point(self, x, r):
        if self.tree is not None:
            return self.tree.query_ball_point(x, r)
        if len(self.P) <= 300000:                       # exact vectorized scan beats a Python cell loop
            d = self.P - np.asarray(x, float)
            return np.flatnonzero(np.einsum('ij,ij->i', d, d) <= r * r).tolist()
        lo = np.floor((x - r - self.o) / self.cell).astype(int); hi = np.floor((x + r - self.o) / self.cell).astype(int)
        cand = []
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    cand += self.keys.get((i, j, k), [])
        if not cand:
            return []
        cand = np.array(cand)
        return cand[np.linalg.norm(self.P[cand] - x, axis=1) <= r].tolist()

    def query(self, X, distance_upper_bound=None):
        """Nearest neighbour.  With distance_upper_bound, points farther than it get (inf, len(P)) as in SciPy."""
        ub = distance_upper_bound
        if self.tree is not None:
            return self.tree.query(X) if ub is None else self.tree.query(X, distance_upper_bound=ub)
        single = np.asarray(X).ndim == 1
        if ub is not None:                              # one sorted grid with cell >= ub is exact up to ub
            ext = float((self.P.max(0) - self.P.min(0)).max()) or 1.0
            h = max(2.0 * ext / max(1.0, np.sqrt(len(self.P))), float(ub))
            if getattr(self, '_ubgrid', None) is None or self._ubgrid.h != h: self._ubgrid = _SortedGrid(self.P, h)
            bd, bi, _ = self._ubgrid.nn(np.atleast_2d(np.asarray(X, float)))
            miss = ~(bd <= ub); bd[miss] = np.inf; bi[miss] = len(self.P)
            return (float(bd[0]), int(bi[0])) if single else (bd, bi)
        if getattr(self, '_grids', None) is None:
            ext = float((self.P.max(0) - self.P.min(0)).max()) or 1.0
            h = 2.0 * ext / max(1.0, np.sqrt(len(self.P)))
            self._grids = [_SortedGrid(self.P, h)]
        d, i = _nn_exact(self.P, X, self._grids)
        return (float(d[0]), int(i[0])) if single else (d, i)


def solve_hermitian(rows, cols, vals, diag, rhs, n, tol=1e-8, maxit=4000):
    """Solve (A + diag) u = rhs for Hermitian PSD A given as COO triplets (complex)."""
    if HAVE_SCIPY:
        A = _sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + _sp.diags(diag)
        return _spla.spsolve(A.tocsc(), rhs)
    def mv(x):
        y = np.zeros(n, complex)
        np.add.at(y, rows, vals * x[cols])
        return y + diag * x
    x = rhs / np.maximum(np.abs(diag), 1e-12)        # Jacobi-preconditioned conjugate gradient
    dinv = 1.0 / np.maximum(np.real(diag) + np.bincount(rows, weights=np.real(vals) * (rows == cols), minlength=n), 1e-12)
    r = rhs - mv(x); z = dinv * r; p = z.copy(); rz = np.vdot(r, z)
    nb = np.linalg.norm(rhs) + 1e-30
    for _ in range(maxit):
        Ap = mv(p); a = rz / np.vdot(p, Ap); x += a * p; r -= a * Ap
        if np.linalg.norm(r) / nb < tol:
            break
        z = dinv * r; rz2 = np.vdot(r, z); p = z + (rz2 / rz) * p; rz = rz2
    return x


def knn(index, x, k):
    """k nearest indices of point x in a BallIndex (SciPy tree, else an exact vectorized scan)."""
    k = min(k, len(index.P))
    if index.tree is not None:
        _, idx = index.tree.query(x, k=k); return np.atleast_1d(idx).tolist()
    if k <= 0: return []
    d = index.P - np.asarray(x, float); d = np.einsum('ij,ij->i', d, d)
    part = np.argpartition(d, k - 1)[:k] if k < len(d) else np.arange(len(d))
    return part[np.lexsort((part, d[part]))].tolist()
