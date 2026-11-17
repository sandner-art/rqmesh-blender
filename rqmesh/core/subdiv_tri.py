# SPDX-License-Identifier: Apache-2.0
"""Recover the base of a triangle mesh made by 1-to-4 subdivision: Loop, or the midpoint split that Blender's
Subdivide (edit mode) applies to triangles.

Roles: even (old) vertices are never adjacent; an edge point (odd) has valence 6 inside, 4 on the boundary, and its
two even neighbours sit opposite each other in its 1-ring; every triangle is a centre (3 odd) or a corner (1 even).
Labels propagate from any vertex that cannot be odd (valence != 6 inside, != 4 on the boundary); each base triangle
is read off around a centre triangle.  Positions are fitted by least squares (midpoint or Loop rules), with the same
residual-based stopping rule as the Catmull-Clark recovery.  (Loop uses Loop's original weights; it is checked here
against our own forward Loop, not against an external tool.)"""
import math
import numpy as np
from collections import defaultdict, deque
try:
    from .subdiv import Coo, cgls
except ImportError:
    from subdiv import Coo, cgls


def _rings(T, nV):
    e2t = defaultdict(list); nbr = [set() for _ in range(nV)]
    for i, t in enumerate(T):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]; e2t[(min(a, b), max(a, b))].append(i); nbr[a].add(b); nbr[b].add(a)
    bnd = {v for e, ts in e2t.items() if len(ts) == 1 for v in e}
    return e2t, nbr, bnd


def _opposite_in_ring(T, e2t, v, u):
    """For an interior valence-6 vertex v: the neighbour three steps from u around v (u's opposite)."""
    ring = [u]; prev = None; cur = u
    for _ in range(5):
        nxt = None
        for ti in e2t[(min(v, cur), max(v, cur))]:
            w = next(x for x in T[ti] if x not in (v, cur))
            if w != prev and w not in ring: nxt = w; break
        if nxt is None: return None
        prev, cur = cur, nxt; ring.append(cur)
    return ring[3] if len(ring) == 6 else None


def labelling(T, nV):
    """even flags (bool array) or None."""
    if not T or any(len(t) != 3 for t in T): return None
    e2t, nbr, bnd = _rings(T, nV)
    if any(len(x) > 2 for x in e2t.values()): return None
    used = [v for v in range(nV) if nbr[v]]
    seeds = [v for v in used if (len(nbr[v]) != 6 and v not in bnd) or (v in bnd and len(nbr[v]) != 4)]
    lab = np.full(nV, -1, np.int8)                     # 1 even, 0 odd
    for s in seeds + used:
        if lab[s] >= 0: continue
        lab[s] = 1; dq = deque([s]); ok = True
        while dq and ok:
            u = dq.popleft()
            if lab[u] == 1:
                for w in nbr[u]:
                    if lab[w] == 1: ok = False; break
                    if lab[w] < 0: lab[w] = 0; dq.append(w)
            else:                                     # odd: its two even neighbours are opposite; others odd
                if u in bnd:
                    if len(nbr[u]) != 4: ok = False; break
                    evens = [w for w in nbr[u] if w in bnd and len(e2t[(min(u, w), max(u, w))]) == 1]
                else:
                    if len(nbr[u]) != 6: ok = False; break
                    e0 = next((w for w in nbr[u] if lab[w] == 1), None)
                    if e0 is None: continue
                    o = _opposite_in_ring(T, e2t, u, e0)
                    if o is None: ok = False; break
                    evens = [e0, o]
                if len(evens) != 2: ok = False; break
                for w in nbr[u]:
                    want = 1 if w in evens else 0
                    if lab[w] >= 0 and lab[w] != want: ok = False; break
                    if lab[w] < 0: lab[w] = want; dq.append(w)
        if not ok: return None
    for t in T:
        if sum(lab[x] for x in t) > 1: return None
    return lab == 1


def base_of(T, nV, even):
    """Base triangles (old vertex ids), and edge point of each base edge."""
    e2t, nbr, bnd = _rings(T, nV)
    base, Ev = [], {}
    for i, t in enumerate(T):
        if any(even[x] for x in t): continue
        m = list(t); corner = []
        for k in range(3):
            a, b = m[k], m[(k + 1) % 3]
            ts = [j for j in e2t[(min(a, b), max(a, b))] if j != i]
            if len(ts) != 1: return None
            c = next(x for x in T[ts[0]] if x not in (a, b))
            if not even[c]: return None
            corner.append(c)
        A, B, C = corner[2], corner[0], corner[1]     # across (m3,m1), (m1,m2), (m2,m3)
        base.append([A, B, C])
        for (x, y), mm in (((A, B), m[0]), ((B, C), m[1]), ((C, A), m[2])):
            Ev[(min(x, y), max(x, y))] = mm
    if 4 * len(base) != len(T): return None
    return base, Ev


def operator(base, nb, Ev, fine_of, nfine, scheme):
    """Rows: fine vertices; cols: base vertices.  fine_of[v] = fine row of base vertex v."""
    R, C, W = [], [], []
    add = lambda r, c, w: (R.append(r), C.append(c), W.append(w))
    e2f = defaultdict(list); vn = defaultdict(set)
    for bi, f in enumerate(base):
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]; e2f[(min(a, b), max(a, b))].append(bi); vn[a].add(b); vn[b].add(a)
    bnd_v = {v for e, fs in e2f.items() if len(fs) == 1 for v in e}
    for e, fs in e2f.items():
        r = Ev[e]; a, b = e
        if scheme == 'midpoint' or len(fs) == 1:
            add(r, a, 0.5); add(r, b, 0.5); continue
        add(r, a, 0.375); add(r, b, 0.375)
        for bi in fs:
            c = next(x for x in base[bi] if x not in e); add(r, c, 0.125)
    for v in range(nb):
        r = fine_of[v]
        if scheme == 'midpoint': add(r, v, 1.0); continue
        if v in bnd_v:
            bn = [u for u in vn[v] if len(e2f[(min(u, v), max(u, v))]) == 1]
            if len(bn) == 2: add(r, v, 0.75); add(r, bn[0], 0.125); add(r, bn[1], 0.125)
            else: add(r, v, 1.0)
            continue
        n = len(vn[v]); beta = (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * math.cos(2 * math.pi / n)) ** 2) / n
        add(r, v, 1.0 - n * beta)
        for u in vn[v]: add(r, u, beta)
    return Coo(R, C, W, (nfine, nb))


def unsubdivide_tri_once(V, T, schemes=('midpoint', 'loop')):
    V = np.asarray(V, float); nV = len(V)
    even = labelling(T, nV)
    if even is None: return None
    bf = base_of(T, nV, even)
    if bf is None: return None
    base, Ev = bf
    vb = sorted({v for f in base for v in f}); loc = {v: i for i, v in enumerate(vb)}
    fine = sorted({v for t in T for v in t}); fl = {v: i for i, v in enumerate(fine)}
    bl = [[loc[v] for v in f] for f in base]
    Evl = {(min(loc[a], loc[b]), max(loc[a], loc[b])): fl[m] for (a, b), m in Ev.items()}
    X = V[fine]; mel = float(np.mean([np.linalg.norm(V[t[0]] - V[t[1]]) for t in T[:2000]])) + 1e-30
    best = None
    for sc in schemes:
        S = operator(bl, len(vb), Evl, {loc[v]: fl[v] for v in vb}, len(fine), sc)
        P = V[vb].copy()
        if sc != 'midpoint': P = cgls(S.dot, S.tdot, X, P)
        res = float(np.sqrt(np.mean(np.sum((S.dot(P) - X) ** 2, 1)))) / mel
        if best is None or res < best[0]: best = (res, sc, P)
    res, sc, P = best
    return dict(V=P, F=bl, vmap=vb, scheme=sc, residual_rel=res)


def unsubdivide_tri(V, T, levels=None, tol=0.05, log=None):
    V = np.asarray(V, float); orig = list(range(len(V))); curV, curT = V, [list(t) for t in T]; info = []
    while levels is None or len(info) < levels:
        r = unsubdivide_tri_once(curV, curT)
        if r is None: break
        lim = tol if not info else max(0.01, 5.0 * info[-1]['residual_rel'])
        if tol is not None and r['residual_rel'] > lim: break
        orig = [orig[v] for v in r['vmap']]
        info.append(dict(scheme=r['scheme'], residual_rel=round(r['residual_rel'], 6), faces=len(r['F'])))
        if log: log(f"level {len(info)}: {len(curT)} -> {len(r['F'])} triangles, {r['scheme']}")
        curV, curT = r['V'], r['F']
    return dict(cage=(curV, curT), surface=(V[orig], curT), orig=orig, levels=info)


def subdivide_tri(V, T, scheme='loop', levels=1):
    """Forward 1-to-4 subdivision (for tests)."""
    V = np.asarray(V, float)
    for _ in range(levels):
        nb = len(V); e2f = defaultdict(list)
        for bi, f in enumerate(T):
            for k in range(3): a, b = f[k], f[(k + 1) % 3]; e2f[(min(a, b), max(a, b))].append(bi)
        Ev = {e: nb + i for i, e in enumerate(sorted(e2f))}
        S = operator(T, nb, Ev, {v: v for v in range(nb)}, nb + len(Ev), scheme)
        NT = []
        for f in T:
            a, b, c = f; ab, bc, ca = Ev[(min(a, b), max(a, b))], Ev[(min(b, c), max(b, c))], Ev[(min(c, a), max(c, a))]
            NT += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = S.dot(V); T = NT
    return V, T
