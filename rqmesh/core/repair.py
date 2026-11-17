# SPDX-License-Identifier: Apache-2.0
"""Feature-aware fold repair for quad meshes (post-process after any remesher).

Folded quads (scaled Jacobian <= 0) after QuadWild sit almost always on feature lines (94-97 % in the round-2
panels), mostly where two feature lines meet.  This pass only moves vertices:
  * corner vertices (on two non-parallel feature lines) stay fixed,
  * feature vertices slide along the input feature polyline,
  * other vertices move in the tangent plane and are projected back to the input surface,
and a move is accepted only if the minimum scaled Jacobian of the incident quads improves.
usage: python repair.py REF_TRI.obj IN_QUAD.obj OUT_QUAD.obj [--feature-deg 35]  -> JSON line
"""
import sys, os, json, math, argparse
import numpy as np

try:
    from .meshio_min import read_obj, write_obj, triangulate, TriMesh
except ImportError:                     # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj, write_obj, triangulate, TriMesh
try:                                    # Blender's bundled Python has NumPy but not SciPy
    if os.environ.get('RQM_NO_SCIPY') == '1': raise ImportError
    from scipy.spatial import cKDTree
except ImportError:
    try:
        from .compat import BallIndex, knn
    except ImportError:
        from compat import BallIndex, knn

    class cKDTree:
        def __init__(self, P): self.ix = BallIndex(P)

        def query(self, X, k=1):
            X = np.atleast_2d(np.asarray(X, float))
            if k == 1: return self.ix.query(X)
            idx = np.array([knn(self.ix, x, k) for x in X])
            return np.linalg.norm(self.ix.P[idx] - X[:, None, :], axis=2), idx


def closest_on_tris(P, A, B, C):
    """Closest points on triangles (A,B,C) to points P, vectorised (Ericson, RTCD 5.1.5)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - B; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - C; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / den; w = vc / den
    res = A + ab * v[:, None] + ac * w[:, None]
    m = (d1 <= 0) & (d2 <= 0); res[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3); res[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6); res[m] = C[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    res[m] = (A + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    res[m] = (A + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
    res[m] = (B + (C - B) * t[:, None])[m]
    return res


class Surface:
    def __init__(self, M, k=16):
        self.M = M; self.tree = cKDTree(M.C); self.k = min(k, len(M.T))
        self.A, self.B, self.Cc = (M.V[M.T[:, i]] for i in range(3))

    def project(self, P):
        P = np.atleast_2d(P); _, idx = self.tree.query(P, k=self.k)
        best = np.zeros_like(P); bd = np.full(len(P), np.inf)
        for j in range(idx.shape[1]):
            f = idx[:, j]; q = closest_on_tris(P, self.A[f], self.B[f], self.Cc[f])
            d = np.linalg.norm(q - P, axis=1); m = d < bd; bd[m] = d[m]; best[m] = q[m]
        return best


class Features:
    def __init__(self, M, deg):
        e = np.where(M.interior & (M.theta > np.radians(deg)) | M.boundary)[0]
        self.S0, self.S1 = M.V[M.E[e, 0]], M.V[M.E[e, 1]]
        self.dir = M.edir[e]
        self.tree = cKDTree(0.5 * (self.S0 + self.S1)) if len(e) else None
        # corner input vertices: feature degree != 2 or a turn > 30 deg between two feature edges
        deg_v, dirs = {}, {}
        for i, (a, b) in enumerate(M.E[e]):
            for v in (int(a), int(b)):
                deg_v[v] = deg_v.get(v, 0) + 1; dirs.setdefault(v, []).append(self.dir[i])
        corners = [v for v, c in deg_v.items() if c != 2 or abs(dirs[v][0] @ dirs[v][1]) < math.cos(math.radians(30))]
        self.corners = M.V[corners] if corners else np.zeros((0, 3))
        self.ctree = cKDTree(self.corners) if len(corners) else None

    def project(self, P, k=8):
        P = np.atleast_2d(P); _, idx = self.tree.query(P, k=min(k, len(self.S0)))
        idx = np.atleast_2d(idx); best = np.zeros_like(P); bd = np.full(len(P), np.inf)
        for j in range(idx.shape[1]):
            s = idx[:, j]; a, b = self.S0[s], self.S1[s]; ab = b - a
            t = np.clip(((P - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0, 1)
            q = a + ab * t[:, None]; d = np.linalg.norm(q - P, axis=1); m = d < bd; bd[m] = d[m]; best[m] = q[m]
        return best, bd


def quad_sj(V, f):
    P = V[f]; n = np.cross(P[2] - P[0], P[3] - P[1]); ln = np.linalg.norm(n)
    if ln < 1e-20: return -1.0
    n /= ln; out = 1.0
    for k in range(4):
        e1 = P[(k + 1) % 4] - P[k]; e2 = P[k - 1] - P[k]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if l1 < 1e-20 or l2 < 1e-20: return -1.0
        out = min(out, float(np.dot(np.cross(e1, e2), n) / (l1 * l2)))
    return out


def repair(Vin, Tin, Vq, Fq, feature_deg=35.0, sweeps=40, bad=0.1):
    M = TriMesh(Vin, Tin); S = Surface(M); FE = Features(M, feature_deg)
    V = Vq.copy(); diag = M.diag
    quads = [i for i, f in enumerate(Fq) if len(f) == 4]
    vf = {}
    for i in quads:
        for v in Fq[i]: vf.setdefault(v, []).append(i)
    nb = {}
    for f in Fq:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    # vertex classes
    allv = np.array(sorted(vf)); tol = 1e-3 * diag
    _, dfeat = FE.project(V[allv]) if FE.tree is not None else (None, np.full(len(allv), np.inf))
    on_feat = {int(v): bool(d < tol) for v, d in zip(allv, dfeat)}
    fixed = set()
    if FE.ctree is not None:
        dc, _ = FE.ctree.query(V[allv])
        fixed = {int(v) for v, d in zip(allv, dc) if d < 2 * tol}
    sj = {i: quad_sj(V, Fq[i]) for i in quads}
    n0 = sum(1 for s in sj.values() if s <= 0)
    moved = 0
    for _ in range(sweeps):
        cand = sorted({v for i, s in sj.items() if s < bad for v in Fq[i]} - fixed)
        if not cand:
            break
        improved = 0
        for v in cand:
            inc = vf[v]; cur = min(sj[i] for i in inc); cur_n = sum(1 for i in inc if sj[i] <= 0)
            ring = list(nb[v])
            if on_feat[v]:
                fr = [u for u in ring if on_feat.get(u)]
                tgt = V[fr].mean(0) if len(fr) >= 2 else V[ring].mean(0)
            else:
                tgt = V[ring].mean(0)
            best, bestv = cur, None
            for t in (1.0, 0.5, 0.25):
                p = V[v] + t * (tgt - V[v])
                p = FE.project(p)[0][0] if on_feat[v] else S.project(p)[0]
                old = V[v].copy(); V[v] = p
                vals = [quad_sj(V, Fq[i]) for i in inc]; s = min(vals); n_f = sum(1 for x in vals if x <= 0)
                V[v] = old
                if s > best + 1e-6 and n_f <= cur_n:          # never create a new fold to fix another
                    best, bestv = s, p
            if bestv is not None:
                V[v] = bestv; improved += 1; moved += 1
                for i in inc: sj[i] = quad_sj(V, Fq[i])
        if not improved:
            break
    n1 = sum(1 for s in sj.values() if s <= 0)
    return V, dict(folds_before=n0, folds_after=n1, moved=moved, fixed_corners=len(fixed))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--feature-deg', type=float, default=35.0); a = ap.parse_args()
    Vi, Fi = read_obj(a.ref); Vq, Fq = read_obj(a.inp)
    V, info = repair(Vi, triangulate(Fi), Vq, Fq, a.feature_deg)
    write_obj(a.out, V, Fq); print(json.dumps(info))


if __name__ == '__main__':
    main()


def fill_new_holes(V_in, F_in, V_out, F_out, max_edges=16, near=2.0):
    """Fill small holes in a remesh result that the input did not have (QuadWild occasionally leaves one patch
    unfilled).  A boundary loop of the output is 'new' when no input boundary vertex lies within `near` mean edge
    lengths of its centroid.  Even loops get a strip of quads, odd loops quads plus one triangle.  Faces are oriented
    against their neighbours.  Returns (F_out with fills, info)."""
    import collections
    V_in = np.asarray(V_in, float); V_out = np.asarray(V_out, float)

    def loops_of(F):
        cnt = collections.Counter(); dirn = {}
        for f in F:
            for k in range(len(f)):
                a, b = int(f[k]), int(f[(k + 1) % len(f)])
                cnt[min(a, b), max(a, b)] += 1; dirn[a, b] = True
        nxt = {}
        for (a, b), c in cnt.items():
            if c != 1: continue
            # the hole is walked opposite to the face that owns the edge: face has a->b, hole goes b->a
            if (a, b) in dirn: nxt[b] = a
            else: nxt[a] = b
        out, seen = [], set()
        for s in list(nxt):
            if s in seen: continue
            L = [s]; seen.add(s); c = nxt[s]
            while c != s and c in nxt and c not in seen and len(L) <= len(nxt):
                L.append(c); seen.add(c); c = nxt[c]
            if c == s: out.append(L)
        return out
    bin_ = sorted({v for L in loops_of(F_in) for v in L})
    e = [np.linalg.norm(V_out[f[k]] - V_out[f[(k + 1) % len(f)]]) for f in F_out[:2000] for k in range(len(f))]
    h = float(np.mean(e)) if e else 1.0
    Pin = V_in[bin_] if bin_ else np.zeros((0, 3))
    F_new = [list(map(int, f)) for f in F_out]; info = dict(holes_new=0, filled_quads=0, filled_triangles=0, too_big=0)
    for L in loops_of(F_out):
        c = V_out[L].mean(0)
        if len(Pin) and np.min(np.linalg.norm(Pin - c, axis=1)) < near * h + 0.5 * np.max(np.linalg.norm(V_out[L] - c, axis=1)):
            continue                                       # an opening the input had
        info['holes_new'] += 1
        n = len(L)
        if n > max_edges: info['too_big'] += 1; continue
        # strip fill: quad (L0 L1 L2 L[-1]) leaves the polygon L2 .. L[-1]; repeat; 4 left -> quad, 3 -> triangle
        faces = []
        while len(L) > 4:
            faces.append([L[0], L[1], L[2], L[-1]]); L = L[2:]
        faces.append(list(L))
        for f in faces:
            F_new.append(f)
            info['filled_triangles' if len(f) == 3 else 'filled_quads'] += 1
    return F_new, info
