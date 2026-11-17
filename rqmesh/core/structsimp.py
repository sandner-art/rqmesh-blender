# SPDX-License-Identifier: Apache-2.0
"""Structure-preserving simplification (prototype).

When the input still carries its modelling structure (a subdivided / beveled game model exported as triangles),
don't re-sample the surface: recover the quads, then remove whole rows of quads (poly-chord / ring collapse,
Daniels et al. 2008; Tarini et al. 2010) until the face budget is met.  Rules:
  * a rung whose endpoints lie on two different feature lines is never collapsed (creases, bevel and fillet
    facet loops survive by construction);
  * a feature vertex wins over a smooth one; corners (feature valence != 2) never move;
  * a regular vertex stays regular (ring collapse preserves valence 4), so no new poles are created;
  * each candidate is checked locally (degenerate faces, doublets, non-manifold edges, valence-2 vertices,
    normal flips) and skipped if invalid;
  * rings are ranked by width x (1 + bend) so thin, flat rows go first; several disjoint rings per pass.
Moved vertices are projected back to the input (nearest point on input triangles, feature vertices onto input
feature edges).

  python structsimp.py IN.obj OUT.obj TARGET_FACES [--feature-deg 8] [--bend-w 0.1] [--no-project]  -> JSON line
  IN may be triangles (quads are recovered), quads or a mix.
"""
import sys, os, json, math, argparse, time
import numpy as np
try:
    from .meshio_min import read_obj, write_obj
    from .compat import BallIndex, knn
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj, write_obj
    from compat import BallIndex, knn


def _ekey(a, b):
    return (a, b) if a < b else (b, a)


# ----------------------------------------------------------------------------- triangles -> quads
def tris_to_quads(V, T, max_corner_dev=89.5, max_bend=30.0, order_hint=False, flood=False, straight_max=50.0):
    """Recover quads from a triangulated quad-dominant mesh.

    order_hint: first pair consecutive triangles in file order (DCC exporters split each quad into two consecutive
    triangles; engines that reorder index buffers destroy this cue).
    Geometric part: greedy best-first by corner-angle deviation.  flood=True instead grows a grid-continuation
    front from each seed (partner chosen by straightness of the continued side edges); it propagates a wrong seed,
    so it is off by default.  Geometry alone is ambiguous on skewed cells: pairing across rows of 63/117-degree
    parallelograms gives MORE rectangular quads than the modeller's own, so neither rule recovers the original
    structure there (round-5 finding); quads or the export order are needed."""
    import heapq
    T = np.asarray(T, dtype=np.int64)
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
    emap = {}
    for i, t in enumerate(T):
        for k in range(3):
            emap.setdefault(_ekey(t[k], t[(k + 1) % 3]), []).append(i)
    cb = math.cos(math.radians(max_bend)); cand = []; pairs = {}
    for e, fs in emap.items():
        if len(fs) != 2: continue
        i, j = fs
        ti = list(T[i]); a = [v for v in ti if v not in e][0]; b = [v for v in T[j] if v not in e][0]
        k = ti.index(a); q = [a, ti[(k + 1) % 3], b, ti[(k + 2) % 3]]
        hinted = order_hint and abs(i - j) == 1
        P = V[q]; n = N[i] + N[j]; dev = 0.0; ok = hinted or N[i] @ N[j] >= cb
        for m in range(4):
            if not ok: break
            d0 = P[(m - 1) % 4] - P[m]; d1 = P[(m + 1) % 4] - P[m]
            l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
            if not hinted and np.dot(np.cross(d1, d0), n) < -0.02 * l0 * l1 * np.linalg.norm(n): ok = False; break
            c = d0 @ d1 / (l0 * l1 + 1e-30)
            dev = max(dev, abs(math.degrees(math.acos(max(-1, min(1, c)))) - 90))
        if ok and (hinted or dev <= max_corner_dev):
            cand.append((dev, i, j, q)); pairs[(min(i, j), max(i, j))] = q
    cand.sort(key=lambda c: c[0])
    tri_pairs = {}
    for dev, i, j, q in cand:
        tri_pairs.setdefault(i, []).append((j, q)); tri_pairs.setdefault(j, []).append((i, q))
    used = np.zeros(len(T), bool); F = []
    if order_hint:
        i = 0
        while i + 1 < len(T):
            q = pairs.get((i, i + 1))
            if q is not None: used[i] = used[i + 1] = True; F.append(q); i += 2
            else: i += 1

    def nb(q, v, other):
        k = q.index(v); a, b = q[(k - 1) % 4], q[(k + 1) % 4]
        return a if b == other else b

    def ang(u, w):
        return math.degrees(math.acos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-30), -1, 1)))

    heap = []

    def push_front(q, i, j):
        if not flood: return
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            for t in emap.get(_ekey(a, b), ()):
                if t == i or t == j or used[t]: continue
                for s, qn in tri_pairs.get(t, ()):
                    if used[s] or a not in qn or b not in qn: continue
                    ka, kb = qn.index(a), qn.index(b)
                    if (ka - kb) % 4 not in (1, 3): continue          # a-b must be a side of the new quad
                    sc = ang(V[a] - V[nb(q, a, b)], V[nb(qn, a, b)] - V[a]) + ang(V[b] - V[nb(q, b, a)], V[nb(qn, b, a)] - V[b])
                    heapq.heappush(heap, (sc, t, s, qn))

    # seed the front from hinted quads too
    if F:
        idx_of = {}
        for (i, j), q in pairs.items():
            if used[i] and used[j]: idx_of[tuple(q)] = (i, j)
        for q in F:
            ij = idx_of.get(tuple(q))
            if ij: push_front(q, *ij)
    ptr = 0
    while True:
        while heap:
            sc, t, s, qn = heapq.heappop(heap)
            if used[t] or used[s] or sc > 2 * straight_max: continue
            used[t] = used[s] = True; F.append(qn); push_front(qn, t, s)
        while ptr < len(cand) and (used[cand[ptr][1]] or used[cand[ptr][2]]): ptr += 1
        if ptr >= len(cand): break
        dev, i, j, q = cand[ptr]
        used[i] = used[j] = True; F.append(q); push_front(q, i, j)
    F += [list(T[i]) for i in range(len(T)) if not used[i]]
    return F


# ----------------------------------------------------------------------------- helpers
def face_normal(P):
    """Unit area normal of a polygon (pure-float arithmetic: np.cross on 3-vectors was 60% of the run time)."""
    P = P.tolist() if hasattr(P, 'tolist') else P
    nx = ny = nz = 0.0; m = len(P)
    for k in range(m):
        ax, ay, az = P[k]; bx, by, bz = P[(k + 1) % m]
        nx += ay * bz - az * by; ny += az * bx - ax * bz; nz += ax * by - ay * bx
    L = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-30
    return np.array((nx / L, ny / L, nz / L))


def face_normals_all(V, F, alive=None):
    """Vectorized normals of many faces (quads and triangles in one go)."""
    out = np.zeros((len(F), 3))
    for k_len in (3, 4):
        idx = [i for i, f in enumerate(F) if len(f) == k_len and (alive is None or alive[i])]
        if not idx: continue
        A = np.array([F[i] for i in idx]); P = V[A]; Pn = np.roll(P, -1, axis=1)
        out[idx] = np.cross(P, Pn).sum(axis=1)
    for i, f in enumerate(F):
        if len(f) > 4 and (alive is None or alive[i]): out[i] = face_normal(V[f])
    L = np.linalg.norm(out, axis=1, keepdims=True) + 1e-30
    return out / L


def feature_edges(V, F, deg):
    em = {}
    for fi, f in enumerate(F):
        for k in range(len(f)): em.setdefault(_ekey(f[k], f[(k + 1) % len(f)]), []).append(fi)
    N = np.array([face_normal(V[f]) for f in F]); feat = set(); c = math.cos(math.radians(deg))
    for e, fs in em.items():
        if len(fs) == 1 or len(fs) > 2 or N[fs[0]] @ N[fs[1]] < c: feat.add(e)
    return feat


class Mesh:
    def __init__(s, V, F):
        s.V = V.copy(); s.F = [list(f) for f in F]; s.alive = [True] * len(F)
        s.vf = {}
        for fi, f in enumerate(s.F):
            for v in f: s.vf.setdefault(v, set()).add(fi)

    def nfaces(s):
        return sum(s.alive)

    def edge_faces(s, a, b):
        return s.vf.get(a, set()) & s.vf.get(b, set())

    def edges(s):
        em = {}
        for fi, f in enumerate(s.F):
            if not s.alive[fi]: continue
            for k in range(len(f)): em.setdefault(_ekey(f[k], f[(k + 1) % len(f)]), []).append(fi)
        return em


def sj_min(P):
    """Minimum scaled Jacobian of a quad (pure-float arithmetic)."""
    P = P.tolist() if hasattr(P, 'tolist') else P
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = P
    ax, ay, az = x2 - x0, y2 - y0, z2 - z0; bx, by, bz = x3 - x1, y3 - y1, z3 - z1
    nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-20: return -1.0
    nx /= ln; ny /= ln; nz /= ln; out = 1.0
    for k in range(4):
        px, py, pz = P[k]; qx, qy, qz = P[(k + 1) % 4]; rx, ry, rz = P[k - 1]
        e1x, e1y, e1z = qx - px, qy - py, qz - pz; e2x, e2y, e2z = rx - px, ry - py, rz - pz
        l1 = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z); l2 = math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
        if l1 < 1e-20 or l2 < 1e-20: return -1.0
        cx, cy, cz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
        out = min(out, (cx * nx + cy * ny + cz * nz) / (l1 * l2))
    return out


def opposite_edge(f, e):
    i = f.index(e[0]); j = f.index(e[1])
    if (i + 1) % 4 == j: return _ekey(f[(j + 1) % 4], f[(j + 2) % 4])
    return _ekey(f[(i + 1) % 4], f[(i + 2) % 4])


def chords_fast(M):
    """Same chords as chords(M, M.edges()) -- same order, same validity -- from a sort-based edge table.
    A chord is a connected component of the 'opposite edges of a quad' graph (every edge has at most two such links,
    so components are paths or cycles).  Components touching a non-quad face or a non-manifold edge are invalid
    (faces get -1).  Order: by the first appearance of any of their edges in face order, as chords() iterates."""
    alive = [g for g in range(len(M.F)) if M.alive[g]]
    if not alive: return []
    deg = np.fromiter((len(M.F[g]) for g in alive), np.int64, len(alive))
    HF = np.repeat(np.asarray(alive, np.int64), deg)
    HA = np.fromiter((v for g in alive for v in M.F[g]), np.int64, int(deg.sum()))
    off = np.r_[0, np.cumsum(deg)[:-1]]; pos = np.arange(len(HA)) - np.repeat(off, deg)
    nxt = np.where(pos + 1 == np.repeat(deg, deg), np.repeat(off, deg), np.arange(len(HA)) + 1)
    HB = HA[nxt]
    lo, hi = np.minimum(HA, HB), np.maximum(HA, HB); key = lo * (int(HA.max()) + 1) + hi
    order = np.argsort(key, kind='stable'); ks = key[order]
    start = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]; cnt = np.diff(np.r_[start, len(ks)])
    eid = np.empty(len(HA), np.int64); eid[order] = np.repeat(np.arange(len(start)), cnt)
    first = order[start]                                     # first half-edge (face order) of each edge
    E = len(start); bad = cnt > 2
    nq = np.repeat(deg != 4, deg); bad[eid[nq]] = True        # edges of non-quad faces
    q = np.flatnonzero(deg == 4); qo = off[q]
    A = np.r_[eid[qo], eid[qo + 1]]; B = np.r_[eid[qo + 2], eid[qo + 3]]; LF = np.r_[np.asarray(alive)[q], np.asarray(alive)[q]]
    lab = np.arange(E)
    while True:                                              # min-label propagation with pointer jumping
        m = np.minimum(lab[A], lab[B]); old = lab.copy()
        np.minimum.at(lab, lab[A], m); np.minimum.at(lab, lab[B], m)
        while True:
            l2 = lab[lab]
            if np.array_equal(l2, lab): break
            lab = l2
        if np.array_equal(lab, old): break
    comp_first = np.full(E, np.iinfo(np.int64).max); np.minimum.at(comp_first, lab, first)
    eo = np.lexsort((first, comp_first[lab])); el = lab[eo]
    cs = np.r_[0, np.flatnonzero(el[1:] != el[:-1]) + 1]; ce = np.r_[cs[1:], len(eo)]
    lo_e, hi_e = lo[first].tolist(), hi[first].tolist()
    lk = lab[A]; lo_l = np.argsort(lk, kind='stable'); lks = lk[lo_l]
    ls = np.searchsorted(lks, el[cs]); le = np.searchsorted(lks, el[cs], side='right')
    badc = np.zeros(E, bool); np.logical_or.at(badc, lab, bad)
    out = []
    LFl = LF[lo_l].tolist()
    for i in range(len(cs)):
        es = eo[cs[i]:ce[i]].tolist()
        faces = LFl[ls[i]:le[i]]
        if badc[el[cs[i]]]: faces = faces + [-1]
        out.append(([(lo_e[e], hi_e[e]) for e in es], faces, len(faces) == len(es)))
    return out


def chords(M, em):
    """Poly-chords (rows of quads).  A chord that runs into a non-quad face or a non-manifold edge is marked
    with -1 in its face list (not collapsible)."""
    seen = set(); out = []
    for e0, fs0 in em.items():
        if e0 in seen: continue
        seen.add(e0); seq = {0: [], 1: []}; fseq = {0: [], 1: []}; closed = False
        for d in (0, 1):
            if d >= len(fs0): break
            e, f = e0, fs0[d]
            while True:
                if len(M.F[f]) != 4: fseq[d].append(-1); break
                fseq[d].append(f)
                e2 = opposite_edge(M.F[f], e)
                if e2 == e0: closed = True; break
                if e2 in seen: fseq[d].append(-2); break
                seen.add(e2); seq[d].append(e2)
                nf = [g for g in em[e2] if g != f]
                if len(nf) > 1: fseq[d].append(-3); break
                if not nf: break
                e, f = e2, nf[0]
            if closed: break
        edges = seq[1][::-1] + [e0] + seq[0]
        faces = fseq[1][::-1] + fseq[0]
        out.append((edges, faces, closed))
    return out


# ----------------------------------------------------------------------------- projection to the input
class Projector:
    def __init__(s, V, T, feat_edges):
        s.V = V; s.T = np.asarray(T); s.kd = BallIndex(V)
        s.vt = [[] for _ in range(len(V))]
        for i, t in enumerate(s.T):
            for v in t: s.vt[v].append(i)
        fe = np.array(sorted(feat_edges)) if feat_edges else np.zeros((0, 2), int)
        s.fe = fe
        s.fkd = BallIndex(0.5 * (V[fe[:, 0]] + V[fe[:, 1]])) if len(fe) else None

    @staticmethod
    def _pt_tri(p, a, b, c):
        ab, ac, ap = b - a, c - a, p - a
        d1, d2 = ab @ ap, ac @ ap
        if d1 <= 0 and d2 <= 0: return a
        bp = p - b; d3, d4 = ab @ bp, ac @ bp
        if d3 >= 0 and d4 <= d3: return b
        vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0: return a + d1 / (d1 - d3) * ab
        cp = p - c; d5, d6 = ab @ cp, ac @ cp
        if d6 >= 0 and d5 <= d6: return c
        vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0: return a + d2 / (d2 - d6) * ac
        va = d3 * d6 - d5 * d4
        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
            return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
        den = 1.0 / (va + vb + vc); return a + ab * vb * den + ac * vc * den

    has_uv = False

    def set_uv(s, Tuv, UVin):
        """Enable UV-space placement: Tuv (n_tri x 3) indices into UVin, aligned with the triangles T."""
        s.Tuv = np.asarray(Tuv); s.UVin = np.asarray(UVin, float); s.has_uv = True
        C = s.UVin[s.Tuv].mean(1); s.uvkd = BallIndex(np.c_[C, np.zeros(len(C))])
        e = np.linalg.norm(s.UVin[s.Tuv[:, 1]] - s.UVin[s.Tuv[:, 0]], axis=1); s.uv_r = float(np.percentile(e, 99)) * 2 + 1e-12

    def locate_uv(s, uv, hint=None, k=24):
        """Input triangles whose UV triangle contains uv -> best (3D point) by closeness to hint."""
        best, bd = None, 1e30
        for t in knn(s.uvkd, np.array([uv[0], uv[1], 0.0]), k):
            a, b, c = s.UVin[s.Tuv[t]]
            m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
            det = np.linalg.det(m)
            if abs(det) < 1e-18: continue
            l1, l2 = np.linalg.solve(m, np.asarray(uv[:2]) - a); l0 = 1 - l1 - l2
            if min(l0, l1, l2) < -1e-6: continue
            P = l0 * s.V[s.T[t, 0]] + l1 * s.V[s.T[t, 1]] + l2 * s.V[s.T[t, 2]]
            d = 0.0 if hint is None else float(np.sum((P - hint) ** 2))
            if d < bd: bd, best = d, P
        return best

    def from_uv(s, uv, hint, max_jump=None):
        P = s.locate_uv(uv, hint)
        if P is None: return None
        if max_jump is not None and np.linalg.norm(P - hint) > max_jump: return None
        return P

    def surface_tri(s, p, k=8):
        """Nearest input point and its triangle index."""
        best, bd, bt = p, 1e30, None
        for t in {t for v in knn(s.kd, p, k) for t in s.vt[v]}:
            a, b, c = s.V[s.T[t]]; q = s._pt_tri(p, a, b, c); d = np.sum((q - p) ** 2)
            if d < bd: bd, best, bt = d, q, t
        return best, bt

    def uv_at(s, t, q):
        if not s.has_uv: return None
        a, b, c = s.V[s.T[t]]; v0, v1, v2 = b - a, c - a, q - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-30: return None
        l1 = (d11 * d20 - d01 * d21) / den; l2 = (d00 * d21 - d01 * d20) / den
        U = s.UVin[s.Tuv[t]]; return (1 - l1 - l2) * U[0] + l1 * U[1] + l2 * U[2]

    def surface(s, p, k=8):
        best, bd = p, 1e30
        for t in {t for v in knn(s.kd, p, k) for t in s.vt[v]}:
            a, b, c = s.V[s.T[t]]; q = s._pt_tri(p, a, b, c); d = np.sum((q - p) ** 2)
            if d < bd: bd, best = d, q
        return best

    def feature(s, p, k=6):
        if s.fkd is None: return s.surface(p)
        best, bd = p, 1e30
        for i in knn(s.fkd, p, k):
            a, b = s.V[s.fe[i]]; ab = b - a; t = np.clip((p - a) @ ab / (ab @ ab + 1e-30), 0, 1); q = a + t * ab
            d = np.sum((q - p) ** 2)
            if d < bd: bd, best = d, q
        return best



# ----------------------------------------------------------------------------- quadric error metric (Garland-Heckbert)
def vertex_quadrics(V, F):
    """Area-weighted plane quadrics of the input surface per vertex: (n,4,4) and summed areas (n,)."""
    V = np.asarray(V, float); n = len(V)
    T = np.array([[f[0], f[k], f[k + 1]] for f in F for k in range(1, len(f) - 1)], dtype=np.int64)
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); A2 = np.linalg.norm(N, axis=1)
    ok = A2 > 1e-20; N[ok] /= A2[ok][:, None]; A = 0.5 * A2
    P = np.c_[N, -np.einsum('ij,ij->i', N, V[T[:, 0]])]
    K = A[:, None, None] * np.einsum('ij,ik->ijk', P, P)
    Q = np.zeros((n, 4, 4)); W = np.zeros(n)
    for k in range(3):
        np.add.at(Q, T[:, k], K / 3.0); np.add.at(W, T[:, k], A / 3.0)
    return Q, W


def qem_eval(Q, x):
    h = np.r_[x, 1.0]; return float(h @ Q @ h)


def qem_opt(Q, xbar, rel=1e-3):
    """Regularized optimum (Lindstrom): solve A v = -b in the well-determined directions only, stay at xbar in
    the flat ones (keeps flat regions where they are, moves curved ones off the chord)."""
    A = Q[:3, :3]; b = Q[:3, 3]
    U, S, Vt = np.linalg.svd(A)
    if S[0] <= 0: return np.asarray(xbar, float)
    inv = np.where(S > rel * S[0], 1.0 / np.maximum(S, 1e-300), 0.0)
    r = -(A @ xbar + b)
    return xbar + Vt.T @ (inv * (U.T @ r))


def qem_opt_batch(Q, xbar, rel=1e-3):
    """qem_opt for many quadrics at once: Q (n,4,4), xbar (n,3)."""
    A = Q[:, :3, :3]; b = Q[:, :3, 3]
    U, S, Vt = np.linalg.svd(A)
    inv = np.where(S > rel * S[:, :1], 1.0 / np.maximum(S, 1e-300), 0.0)
    r = -(np.einsum('nij,nj->ni', A, xbar) + b)
    x = xbar + np.einsum('nji,nj->ni', Vt, inv * np.einsum('nji,nj->ni', U, r))
    return np.where((S[:, 0] > 0)[:, None], x, xbar)


def qem_line(Q, a, d, t0, t1):
    """Minimize the quadric along a + t d, t in [t0, t1] (feature lines)."""
    A = Q[:3, :3]; b = Q[:3, 3]; den = float(d @ A @ d)
    t = 0.5 * (t0 + t1) if den <= 1e-30 else -float(d @ (A @ a + b)) / den
    return a + min(max(t, t0), t1) * d

# ----------------------------------------------------------------------------- ring collapse
def ring_collapse(V, F, target, feat_deg=8.0, bend_w=0.1, proj=None, log=None, extra_feat=None,
                  FUV=None, UV=None, mirrors=None, stats=None, cost_mode='width', size_w=0.02, max_error=None,
                  placement='surface', quadrics=None):
    """Remove whole poly-chords until <= target faces.
    extra_feat: edges to protect besides dihedral features (UV seams, sharp marks).
    FUV, UV:    per-face UV indices into UV (welded per vertex and island).  Carried through every collapse: each rung
                merges its two UV indices per island with the same weights as the 3D merge, so seams stay seams.
                With a UV-aware projector, merged smooth vertices are placed at the input surface point whose input UV
                equals their merged UV (zero texture deviation at vertices).
    mirrors:    list of vertex->mirror maps (-1 = none), one per symmetry plane.  A chord whose vertices all have
                mirrors is collapsed only together with its mirror chords, so symmetric input stays symmetric.
    cost_mode:  'width' = width x (1 + bend_w * bend_deg) (hard-surface default); 'error' = estimated deviation of the
                removed row (rung length x bend / 8) + size_w x width, better for curved/organic shapes.
    max_error:  stop when the cheapest remaining row costs more than this (absolute units), whatever the target.
    placement:  'surface' = midpoint projected onto the input (round 5-7); 'qem' = quadric-optimal position of each
                merged vertex (input plane quadrics accumulated through merges; regularized so flat areas stay put;
                feature clusters optimized along their own line).  cost_mode 'qem' ranks rows by the RMS plane distance
                of their merged vertices.
    quadrics:   (Q, W) per input vertex to continue from (adaptive rounds); default computed from V, F.
    Returns V, F (and FUV, UV when given), number of collapses.  stats['Q_out'] holds the carried quadrics."""
    M = Mesh(V, F)
    has_uv = FUV is not None
    if has_uv:
        FUV = [list(x) for x in FUV]; UV = np.asarray(UV, float).copy()
    mir = [np.asarray(m).copy() for m in (mirrors or [])]
    feat = feature_edges(M.V, M.F, feat_deg) | {_ekey(int(a), int(b)) for a, b in (extra_feat or ())}
    fdeg = {}
    for a, b in feat: fdeg[a] = fdeg.get(a, 0) + 1; fdeg[b] = fdeg.get(b, 0) + 1
    fv = set(fdeg)
    corner = {v for v, d in fdeg.items() if d != 2}
    st = stats if stats is not None else {}
    st.update(collapses=0, groups=0, sym_skipped=0)
    passes, stalled = 0, False
    use_q = placement == 'qem' or cost_mode == 'qem'
    if use_q:
        QQ, QW = (np.array(quadrics[0], float), np.array(quadrics[1], float)) if quadrics is not None else vertex_quadrics(V, F)

    def place(C, cf, cc):
        """Kept vertex, position, 'position is the kept vertex' flag for one merge cluster."""
        if cc: u = cc[0]; return u, M.V[u].copy(), True
        if len(cf) == 1: u = cf[0]; return u, M.V[u].copy(), True
        u = cf[0] if cf else C[0]
        P = M.V[cf] if cf else M.V[C]; xbar = P.mean(0)
        if placement != 'qem': return u, xbar, False
        Qs = QQ[C].sum(0)
        if cf:
            if has_uv: return u, xbar, False           # keep 3D and UV consistent along seams / creases
            d = P[-1] - P[0]; L = np.linalg.norm(d)    # no UVs: optimize along the feature line
            if L < 1e-20: return u, xbar, False
            d /= L; ts = (P - P[0]) @ d
            return u, qem_line(Qs, P[0], d, float(ts.min()), float(ts.max())), False
        return u, qem_opt(Qs, xbar), False

    def boundary(v):
        return any(len(M.edge_faces(v, w)) == 1 for f in M.vf[v] for w in M.F[f] if w != v)

    def make_plan(edges, faces):
        """Plan for collapsing a set of rungs (one chord, or a chord and its mirror images which may touch: rung
        chains u-x-y then merge into one vertex, at the cluster mean, so symmetric pairs stay symmetric)."""
        ring_v = {v for e in edges for v in e}
        if any(v not in M.vf for v in ring_v): return None, 'stale'
        fset = set(faces)
        if len(fset) != len(faces): return None, 'topology'
        region = {g for v in ring_v for g in M.vf[v]}
        par = {v: v for v in ring_v}

        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for u, v in edges: par[find(u)] = find(v)
        clus = {}
        for v in ring_v: clus.setdefault(find(v), []).append(v)
        rep, pos, keep = {}, {}, {}
        for C in clus.values():
            C = sorted(C); cf = [v for v in C if v in fv]; cc = [v for v in C if v in corner]
            if len(cc) > 1: return None, 'feature'
            if len(cf) >= 2:                          # feature vertices may only merge along their own line
                fe = [(x, y) for x, y in edges if x in cf and y in cf and _ekey(x, y) in feat]
                pp = {v: v for v in cf}

                def f2(x):
                    while pp[x] != x: x = pp[x]
                    return x
                for x, y in fe: pp[f2(x)] = f2(y)
                if len({f2(v) for v in cf}) > 1: return None, 'feature'
            u, p, k = place(C, cf, cc)
            for v in C:
                if v != u: rep[v] = u
            pos[u] = p; keep[u] = (k, C)
        newfaces = {}
        for g in region - fset:
            h = [rep.get(w, w) for w in M.F[g]]
            if len(set(h)) != len(h): return None, 'topology'
            newfaces[g] = h
        vf2 = {}
        for g in {gg for w in ring_v for h in [newfaces.get(x, M.F[x]) for x in M.vf.get(w, ())]
                  for z in h for gg in M.vf.get(z, ())} - fset:
            for w in newfaces.get(g, M.F[g]): vf2.setdefault(w, set()).add(g)
        for g, h in newfaces.items():
            for k in range(len(h)):
                x, y = h[k], h[(k + 1) % len(h)]
                fa = vf2.get(x, set()) & vf2.get(y, set())
                if len(fa) > 2: return None, 'topology'
                for o in fa:
                    if o != g and len(set(h) & set(newfaces.get(o, M.F[o]))) > 2: return None, 'topology'
        for u in set(rep.values()):
            if len(vf2.get(u, ())) <= 2 and not boundary(u): return None, 'valence2'
        Vt = M.V.copy()
        for u, p in pos.items(): Vt[u] = p
        for g, h in newfaces.items():
            if face_normal(Vt[h]) @ face_normal(M.V[M.F[g]]) < 0.5: return None, 'geom'
            if len(h) == 4 and sj_min(Vt[h]) < min(0.05, sj_min(M.V[M.F[g]])): return None, 'geom'
        uvrep, uvpos = {}, {}
        if has_uv:                                    # merge UV indices per cluster and island
            up, owner = {}, {}

            def uf(x):
                while up[x] != x: up[x] = up[up[x]]; x = up[x]
                return x
            for q in fset:
                f = M.F[q]
                for k, w in enumerate(f):
                    if w in ring_v:
                        t = FUV[q][k]; up.setdefault(t, t); owner[t] = w
            for q in fset:
                f = M.F[q]; byc = {}
                for k, w in enumerate(f):
                    if w in ring_v: byc.setdefault(find(w), []).append(FUV[q][k])
                for L in byc.values():
                    for t in L[1:]: up[uf(t)] = uf(L[0])
            groups = {}
            for t in up: groups.setdefault(uf(t), []).append(t)
            for L in groups.values():
                if len(L) < 2: continue
                w0 = owner[L[0]]; u = rep.get(w0, w0); k = keep[u][0]
                ku = [t for t in L if owner[t] == u]
                ia = ku[0] if ku else L[0]
                uvpos[ia] = UV[ia].copy() if (k and ku) else UV[L].mean(0)
                for t in L:
                    if t != ia: uvrep[t] = ia
        return dict(ring_v=ring_v, region=region, rep=rep, pos=pos, keep=keep, fset=fset, newfaces=newfaces,
                    uvrep=uvrep, uvpos=uvpos, clusters={u: kc[1] for u, kc in keep.items()}), None

    def apply(P):
        nonlocal feat, fv, corner
        rep = P['rep']
        for u, p in P['pos'].items(): M.V[u] = p
        for f in P['fset']:
            M.alive[f] = False
            for w in M.F[f]: M.vf[w].discard(f)
        for g, h in P['newfaces'].items():
            for w in M.F[g]: M.vf[w].discard(g)
            M.F[g] = h
            for w in h: M.vf.setdefault(w, set()).add(g)
            if has_uv:
                FUV[g] = [P['uvrep'].get(x, x) for x in FUV[g]]
        if has_uv:
            for ia, p in P['uvpos'].items(): UV[ia] = p
        for v in rep: M.vf.pop(v, None)
        if use_q:
            for u, C in P['clusters'].items():
                others = [c for c in C if c != u]
                if others: QQ[u] = QQ[u] + QQ[others].sum(0); QW[u] = QW[u] + QW[others].sum()
        feat = {_ekey(rep.get(a, a), rep.get(b, b)) for a, b in feat if rep.get(a, a) != rep.get(b, b)}
        fv = {rep.get(v, v) for v in fv}
        corner = {rep.get(v, v) for v in corner}
        if placement == 'qem':
            if has_uv and proj is not None and getattr(proj, 'has_uv', False):
                for u in P['pos']:                 # UV of the input point nearest to the optimal position
                    if u in fv or P['keep'][u][0]: continue
                    g0 = next(iter(M.vf[u])); ia = FUV[g0][M.F[g0].index(u)]
                    q, t = proj.surface_tri(M.V[u])
                    if t is None: continue
                    uvq = proj.uv_at(t, q)
                    span = max(np.linalg.norm(UV[ia] - UV[FUV[g][M.F[g].index(w)]]) for g in M.vf[u] for w in M.F[g])
                    if uvq is not None and np.linalg.norm(uvq - UV[ia]) <= span: UV[ia] = uvq
        elif proj is not None:                     # project, but never at the price of a twisted quad
            for u in P['pos']:
                if u in fv: continue               # feature vertices stay on their own polyline
                old = M.V[u].copy()
                before = min((sj_min(M.V[M.F[g]]) for g in M.vf[u] if len(M.F[g]) == 4), default=1.0)
                newp, newuv = None, None
                if has_uv and getattr(proj, 'has_uv', False):
                    g0 = next(iter(M.vf[u])); ia = FUV[g0][M.F[g0].index(u)]
                    lmax = max(np.linalg.norm(M.V[w] - old) for g in M.vf[u] for w in M.F[g])
                    newp = proj.from_uv(UV[ia], old, max_jump=lmax)
                    if newp is not None: newuv = ia
                if newp is None: newp = proj.surface(old)
                M.V[u] = newp
                after = min((sj_min(M.V[M.F[g]]) for g in M.vf[u] if len(M.F[g]) == 4), default=1.0)
                if after < min(0.05, before): M.V[u] = old

    def remap_mirrors(rep):
        for m in mir:
            for v, u in rep.items():
                if m[v] >= 0 and m[u] < 0: m[u] = m[v]
            for v in rep: m[v] = -1
            if rep:
                r = np.arange(len(m)); ks = np.fromiter(rep.keys(), np.int64); r[ks] = np.fromiter(rep.values(), np.int64)
                x = np.flatnonzero(m >= 0); m[x] = r[m[x]]

    while M.nfaces() > target:
        passes += 1
        FN = face_normals_all(M.V, M.F, M.alive); VN = np.zeros((len(M.V), 3))
        al = [g for g in range(len(M.F)) if M.alive[g]]
        hv = np.fromiter((v for g in al for v in M.F[g]), np.int64)
        hf = np.repeat(np.asarray(al, np.int64), [len(M.F[g]) for g in al])
        for k in range(3): VN[:, k] = np.bincount(hv, weights=FN[hf, k], minlength=len(M.V))
        n_alive = len(al)
        chs = []
        for edges, faces, closed in chords_fast(M):
            if not edges or not faces or any(f < 0 for f in faces) or len(set(faces)) != len(faces): continue
            verts = [v for e in edges for v in e]
            if len(set(verts)) != len(verts): continue
            chs.append((edges, faces))
        cands = []
        if chs:                                       # all rungs of all chords costed at once (NumPy)
            ru = np.fromiter((u for e, _ in chs for u, _v in e), np.int64)
            rv = np.fromiter((v for e, _ in chs for _u, v in e), np.int64)
            okr = np.fromiter((not ((u in fv and v in fv and _ekey(u, v) not in feat) or (u in corner and v in corner))
                               for e, _ in chs for u, v in e), bool, len(ru))
            start = np.cumsum([0] + [len(e) for e, _ in chs[:-1]])
            Lr = np.linalg.norm(M.V[ru] - M.V[rv], axis=1)
            nu, nv = VN[ru], VN[rv]
            cosb = np.einsum('ij,ij->i', nu, nv) / (np.linalg.norm(nu, axis=1) * np.linalg.norm(nv, axis=1) + 1e-30)
            bend = np.degrees(np.arccos(np.clip(cosb, -1, 1)))
            if cost_mode == 'qem':                # RMS plane distance of the merged vertex + small size term
                Qs = QQ[ru] + QQ[rv]
                fu = np.fromiter((u in fv for u in ru), bool, len(ru)); fw = np.fromiter((v in fv for v in rv), bool, len(rv))
                x = qem_opt_batch(Qs, 0.5 * (M.V[ru] + M.V[rv]))
                one = fu != fw
                x[one] = np.where(fu[one][:, None], M.V[ru[one]], M.V[rv[one]])
                h = np.c_[x, np.ones(len(x))]
                e = np.maximum(np.einsum('ni,nij,nj->n', h, Qs, h), 0.0) / np.maximum(QW[ru] + QW[rv], 1e-30)
                rc = np.sqrt(e) + size_w * Lr
            elif cost_mode == 'error':            # sagitta-like deviation of the removed row + small size term
                rc = Lr * np.radians(bend) / 8.0 + size_w * Lr
            else:
                rc = Lr * (1 + bend_w * bend)
            cc = np.maximum.reduceat(rc, start); ok = np.logical_and.reduceat(okr, start)
            cands = [(round(float(cc[i]), 12), chs[i][0], chs[i][1]) for i in np.flatnonzero(ok)]
        if not cands: st['next_cost'] = float('inf'); break
        cands.sort(key=lambda c: c[0])
        if max_error is not None and cands[0][0] > max_error:
            st['stopped_by_error'] = True; st['next_cost'] = cands[0][0]; break
        edge2c = {e: i for i, (_, edges, _) in enumerate(cands) for e in edges}
        touched = set(); applied = 0; done = set()
        cut = float('inf') if stalled else cands[0][0] * 1.5   # near-best rings first; all of them if stalled
        rej = {'geom': 0, 'valence2': 0, 'topology': 0, 'stale': 0, 'mirror': 0, 'feature': 0}
        for ci, (cost, edges, faces) in enumerate(cands):
            if n_alive <= target or cost > cut: break
            if max_error is not None and cost > max_error: break
            if ci in done: continue
            group = [ci]
            if mir:                                   # closure of the chord under all mirror planes
                k = 0; okm = True
                while k < len(group) and okm:
                    for m in mir:
                        e0 = cands[group[k]][1]
                        if any(m[x] < 0 for e in e0 for x in e):
                            continue                  # asymmetric region: no constraint from this plane
                        mapped = {_ekey(int(m[x]), int(m[y])) for x, y in e0}
                        js = {edge2c.get(e, -1) for e in mapped}
                        if len(js) != 1 or -1 in js: okm = False; break
                        j = js.pop()
                        if j not in group: group.append(j)
                    k += 1
                if not okm or len(group) > 8:
                    rej['mirror'] += 1; st['sym_skipped'] += 1; done.update(group); continue
            gv = {v for j in group for e in cands[j][1] for v in e}
            if gv & touched: done.update(group); continue
            P, why = make_plan([e for j in group for e in cands[j][1]], [f for j in group for f in cands[j][2]])
            done.update(group)
            if P is None: rej[why] = rej.get(why, 0) + 1; continue
            if any(w in touched for g in P['region'] for w in M.F[g]): continue
            apply(P); allrep = P['rep']; n_alive -= len(P['fset'])
            touched |= {w for g in P['region'] if M.alive[g] for w in M.F[g]} | P['ring_v']
            applied += len(group); st['collapses'] += len(group)
            if mir: remap_mirrors(allrep)
            st['groups'] += 1
        if log: print('pass', passes, 'cands', len(cands), 'applied', applied, 'faces', M.nfaces(), rej, file=log, flush=True)
        if applied == 0:
            if not stalled:                        # nothing near-best was valid: retry with all candidates
                stalled = True; continue
            st['next_cost'] = float('inf')         # every remaining ring is invalid
            break
        stalled = False
    keepf = [fi for fi in range(len(M.F)) if M.alive[fi]]
    F2 = [M.F[fi] for fi in keepf]
    used = sorted({v for f in F2 for v in f}); remap = {v: i for i, v in enumerate(used)}
    Fo = [[remap[v] for v in f] for f in F2]
    st['feat_out'] = [(remap[a], remap[b]) for a, b in feat if a in remap and b in remap]   # carried feature lines
    if use_q: st['Q_out'] = (QQ[used], QW[used])
    if has_uv:
        U2 = [FUV[fi] for fi in keepf]; uused = sorted({x for f in U2 for x in f}); um = {x: i for i, x in enumerate(uused)}
        return M.V[used], Fo, [[um[x] for x in f] for f in U2], UV[uused], st['collapses']
    return M.V[used], Fo, st['collapses']


def prepare(V, F, UV=None, FUV=None, order_hint=False, feature_deg=8.0, project=True, symmetry='none',
            protect_seams=True, extra_feat=None):
    """Shared setup for the CLI and the Blender operator.  Returns kwargs for ring_collapse and an info dict."""
    try:
        from . import symmetry as SYM
        from .meshio_min import weld_uv
    except ImportError:
        import symmetry as SYM
        from meshio_min import weld_uv
    info = {}
    T = [f for f in F if len(f) == 3]
    if T and UV is None:
        Q = tris_to_quads(V, T, order_hint=order_hint) + [f for f in F if len(f) != 3]; Quv = None
    elif T:                                           # keep UV indices aligned with recovered quads
        idx = {tuple(f): i for i, f in enumerate(F)}
        Q = tris_to_quads(V, T, order_hint=order_hint) + [f for f in F if len(f) != 3]
        tri_uv = {}
        for f, fu in zip(F, FUV):
            if len(f) == 3:
                for k in range(3): tri_uv.setdefault(f[k], {})[(f[(k + 1) % 3], f[(k + 2) % 3])] = fu[k]
        Quv = []
        for q in Q:
            if len(q) == 3 or len(q) > 4:
                Quv.append(FUV[idx[tuple(q)]] if tuple(q) in idx else [0] * len(q)); continue
            # corner uv of quad vertex = uv of that vertex in one of the two source triangles
            row = []
            for k, v in enumerate(q):
                cand = tri_uv.get(v, {})
                nxt, prv = q[(k + 1) % 4], q[(k - 1) % 4]
                hit = [t for (x, y), t in cand.items() if x in (nxt, prv) or y in (nxt, prv)]
                row.append(hit[0] if hit else 0)
            Quv.append(row)
    else:
        Q, Quv = F, FUV
    info['recovered_quads'] = sum(1 for f in Q if len(f) == 4); info['faces_in'] = len(Q)
    kw = {}
    seams = set()
    if UV is not None:
        Fw, Uw = weld_uv(Q, Quv, UV)
        kw.update(FUV=Fw, UV=Uw)
        if protect_seams:                             # an edge is a seam if its two faces disagree on its UVs
            em = {}
            for fi, (f, fu) in enumerate(zip(Q, Fw)):
                for k in range(len(f)):
                    e = _ekey(f[k], f[(k + 1) % len(f)]); em.setdefault(e, []).append({f[k]: fu[k], f[(k + 1) % len(f)]: fu[(k + 1) % len(f)]})
            seams = {e for e, L in em.items() if len(L) == 2 and L[0] != L[1]}
        info['uv_seam_edges'] = len(seams)
    kw['extra_feat'] = list(seams) + list(extra_feat or [])
    if project:
        Tin, Tuv = [], []
        for f, fu in zip(F, FUV if FUV is not None else [None] * len(F)):
            for k in range(1, len(f) - 1):
                Tin.append([f[0], f[k], f[k + 1]])
                if fu is not None: Tuv.append([fu[0], fu[k], fu[k + 1]])
        proj = Projector(V, Tin, feature_edges(V, Q, feature_deg) | set(seams))
        if UV is not None: proj.set_uv(Tuv, UV)
        kw['proj'] = proj
    planes = []
    if symmetry == 'auto':
        planes = [(n, d) for n, d, sc in SYM.detect_planes(V, Q)]
    elif symmetry in ('x', 'y', 'z'):
        n = np.eye(3)['xyz'.index(symmetry)]; c = 0.5 * (V.min(0) + V.max(0)); planes = [(n, float(n @ c))]
    if planes:
        kw['mirrors'] = [SYM.mirror_map(V, pl) for pl in planes]
        info['symmetry_planes'] = [dict(normal=np.round(n, 4).tolist(), offset=round(float(d), 5),
                                        mirrored=float(np.mean(m >= 0))) for (n, d), m in zip(planes, kw['mirrors'])]
    kw['feat_deg'] = feature_deg
    return Q, kw, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('target', type=int)
    ap.add_argument('--feature-deg', type=float, default=8.0); ap.add_argument('--bend-w', type=float, default=0.1)
    ap.add_argument('--no-project', action='store_true'); ap.add_argument('--order-hint', action='store_true')
    ap.add_argument('--symmetry', default='none', choices=['none', 'auto', 'x', 'y', 'z'])
    ap.add_argument('--ignore-uv', action='store_true'); ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--cost', default='width', choices=['width', 'error', 'qem']);
    ap.add_argument('--placement', default='surface', choices=['surface', 'qem']); ap.add_argument('--max-error', type=float, default=None,
                    help='stop when the estimated row error exceeds this %% of the bbox diagonal')
    a = ap.parse_args(); t0 = time.time()
    try:
        from .meshio_min import read_obj_uv, write_obj_uv
    except ImportError:
        from meshio_min import read_obj_uv, write_obj_uv
    V, F, UV, FUV = read_obj_uv(a.inp)
    if a.ignore_uv: UV = FUV = None
    Q, kw, info = prepare(V, F, UV, FUV, a.order_hint, a.feature_deg, not a.no_project, a.symmetry)
    st = {}
    me = None if a.max_error is None else a.max_error / 100.0 * float(np.linalg.norm(V.max(0) - V.min(0)))
    res = ring_collapse(V, Q, a.target, bend_w=a.bend_w, log=sys.stderr if a.verbose else None, stats=st,
                        cost_mode=a.cost, max_error=me, placement=a.placement, **kw)
    if UV is not None:
        V2, F2, FUV2, UV2, n = res; write_obj_uv(a.out, V2, F2, UV2, FUV2)
    else:
        V2, F2, n = res; write_obj_uv(a.out, V2, F2)
    st.pop('Q_out', None); st.pop('feat_out', None)
    info.update(st); info.update(faces=len(F2), quads=sum(1 for f in F2 if len(f) == 4), seconds=round(time.time() - t0, 2))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
