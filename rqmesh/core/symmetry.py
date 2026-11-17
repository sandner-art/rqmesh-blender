# SPDX-License-Identifier: Apache-2.0
"""Reflective symmetry and principal axes.

detect_planes(V, F)       -> list of planes (n, d, score), n.x = d, score = fraction of surface samples whose mirror
                             image lies on the surface (within tol).  Candidates: the bounding-box axes (game/CAD models
                             are usually modelled axis-aligned) and the PCA axes of the surface, each refined by a few
                             ICP steps (Mitra et al. 2006 use transform voting; this is the cheap special case).
part_axes(V, F)           -> per connected component: centre, 3 principal axes, extents (for compound objects whose
                             parts are symmetric or axis-aligned on their own, not globally).
mirror_map(V, plane, tol) -> for each vertex the index of its mirror vertex, or -1.
symmetry_error(V, plane)  -> (mean, max, matched fraction) of |reflect(v) - nearest vertex| in % of the bbox diagonal.

  python symmetry.py MESH.obj    -> JSON report
"""
import sys, os, json
import numpy as np
try:
    from .meshio_min import read_obj
    from .compat import BallIndex, knn
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj
    from compat import BallIndex, knn


def _tris(F):
    return np.array([[f[0], f[k], f[k + 1]] for f in F for k in range(1, len(f) - 1)], dtype=np.int64)


def surface_samples(V, F, n=4000, seed=0):
    T = _tris(F); A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    rng = np.random.default_rng(seed); p = A / A.sum(); idx = rng.choice(len(T), size=n, p=p)
    r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)
    a, b, c = V[T[idx, 0]], V[T[idx, 1]], V[T[idx, 2]]
    return (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c, A.sum()


def reflect(P, n, d):
    return P - 2.0 * ((P @ n) - d)[:, None] * n[None, :]


def _dense_points(V, F, n=40000):
    S, area = surface_samples(V, F, n, seed=1)
    P = np.vstack([V, S])
    return P, float(np.sqrt(area / len(P)))          # mean sample spacing


def detect_planes(V, F, tol_rel=0.004, min_score=0.9, iters=6):
    V = np.asarray(V, float); diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    S, _ = surface_samples(V, F, 3000)
    ref, spacing = _dense_points(V, F); idx = BallIndex(ref)
    tol = max(tol_rel * diag, 1.2 * spacing)        # point-to-sample distance overestimates point-to-surface
    c_box = 0.5 * (V.min(0) + V.max(0)); c_mass = S.mean(0)
    C = np.cov((S - c_mass).T); w, U = np.linalg.eigh(C)
    cands = [(np.eye(3)[k], c_box) for k in range(3)] + [(U[:, k], c_mass) for k in range(3)]
    out = []; ub = 4 * tol * (1 + 1e-9); nref = len(ref)

    def near(R):
        """Partner in ref and distance; farther than 4 tol -> inf (never used beyond that)."""
        _, j = idx.query(R, distance_upper_bound=ub); j = np.asarray(j); hit = j < nref
        Q = ref[np.minimum(j, nref - 1)]
        return Q, np.where(hit, np.linalg.norm(R - Q, axis=1), np.inf)
    for n, c in cands:
        n = n / np.linalg.norm(n); d = float(n @ c)
        for _ in range(iters):                       # ICP refinement of the plane
            R = reflect(S, n, d); Q, dist = near(R); good = dist < 4 * tol
            if good.sum() < 10: break
            D = S[good] - Q[good]                    # S and its partner Q: n ~ (S - Q), plane through midpoints
            sgn = np.sign(D @ n); sgn[sgn == 0] = 1; D = D * sgn[:, None]
            ln = np.linalg.norm(D, axis=1); keep = ln > 1e-3 * diag
            if keep.sum() >= 10:
                n2 = D[keep].sum(0); n2 /= np.linalg.norm(n2)
                if n2 @ n > 0.99: n = n2
            d = float(np.median(0.5 * (S[good] + Q[good]) @ n))
        ax = int(np.argmax(np.abs(n)))
        if abs(n[ax]) > 0.9995:                      # snap to a world axis (exact mirroring for modelled assets)
            n = np.eye(3)[ax] * np.sign(n[ax]); R = reflect(S, n, d); Q, dist = near(R)
            good = dist < 4 * tol
            if good.sum() >= 10: d = float(np.median(0.5 * (S[good] + Q[good]) @ n))
            best = None                              # the bbox centre often IS the plane: keep whichever fits better
            for dd in (d, float(n @ c_box)):
                r = np.mean(np.minimum(near(reflect(S, n, dd))[1], 4 * tol))
                if best is None or r < best[0]: best = (r, dd)
            d = best[1]
        R = reflect(S, n, d)
        score = float(np.mean(near(R)[1] < tol))
        if score >= min_score and all(abs(n @ m) < 0.98 for m, _, _ in out):
            out.append((n, d, score))
        elif score >= min_score:                      # duplicate direction: keep the better one
            for i, (m, dm, sm) in enumerate(out):
                if abs(n @ m) >= 0.98 and score > sm: out[i] = (n, d, score)
    return sorted(out, key=lambda t: -t[2])


def mirror_map(V, plane, tol_rel=0.002):
    n, d = plane[0], plane[1]; V = np.asarray(V, float)
    diag = float(np.linalg.norm(V.max(0) - V.min(0))); idx = BallIndex(V)
    R = reflect(V, n, d); _, j = idx.query(R, distance_upper_bound=tol_rel * diag * (1 + 1e-9)); j = np.asarray(j)
    hit = j < len(V); jc = np.minimum(j, len(V) - 1)
    dist = np.where(hit, np.linalg.norm(R - V[jc], axis=1), np.inf)
    m = np.where(dist < tol_rel * diag, jc, -1)
    ok = m >= 0; mm = m.copy(); mm[ok & (m[np.clip(m, 0, None)] != np.arange(len(V)))] = -1   # must be mutual
    return mm


def symmetry_error(V, plane):
    n, d = plane[0], plane[1]; V = np.asarray(V, float)
    diag = float(np.linalg.norm(V.max(0) - V.min(0))); idx = BallIndex(V)
    R = reflect(V, n, d); _, j = idx.query(R)
    e = np.linalg.norm(R - V[np.asarray(j)], axis=1) / diag * 100
    return float(e.mean()), float(e.max()), float(np.mean(e < 0.2))


def components(F, nv):
    parent = list(range(nv))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for f in F:
        for v in f[1:]:
            a, b = find(f[0]), find(v)
            if a != b: parent[a] = b
    lab = {}
    return [lab.setdefault(find(v), len(lab)) for v in range(nv)]


def part_axes(V, F):
    V = np.asarray(V, float); lab = np.array(components(F, len(V))); out = []
    for k in range(lab.max() + 1):
        P = V[lab == k]
        if len(P) < 4: continue
        c = P.mean(0); w, U = np.linalg.eigh(np.cov((P - c).T)); U = U[:, ::-1]
        ext = np.ptp((P - c) @ U, axis=0)
        out.append(dict(part=int(k), n_verts=int(len(P)), centre=c.round(5).tolist(),
                        axes=U.T.round(4).tolist(), extents=ext.round(5).tolist()))
    return out


def main():
    V, F = read_obj(sys.argv[1])
    pl = detect_planes(V, F)
    rep = dict(planes=[dict(normal=n.round(4).tolist(), offset=round(d, 5), score=round(s, 3),
                            sym_err_mean=round(symmetry_error(V, (n, d))[0], 4)) for n, d, s in pl],
               parts=len(part_axes(V, F)))
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
