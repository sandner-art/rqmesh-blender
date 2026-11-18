# SPDX-License-Identifier: Apache-2.0
"""Recover the control cage of a subdivided mesh ("un-subdivide").

A mesh made by Catmull-Clark or simple (linear) subdivision of ANY polygon cage -- triangles, quads, n-gons -- is
all quads, and its vertices fall into three roles:  V (old vertices), E (edge points), F (face points).
    * every edge joins an E to a V or an F         -> the vertex graph is bipartite: {E} | {V u F}
    * every quad is (V, E, F, E)                    -> the quad diagonals pair V with F: a second 2-colouring
    * interior E have valence 4, boundary E 3, F are never on the boundary, each E has exactly two V neighbours
The two 2-colourings leave at most 4 labellings per connected component; the checks above usually leave one.
Each base face is read off around its F vertex.  Positions are then FITTED, not copied: we solve
        min_P || L S P - X ||^2
where S is the subdivision operator (simple, Catmull-Clark) and L the optional limit-surface projection (Blender's
"Use Limit Surface"), by CGLS with NumPy only.  A residual near zero proves the scheme and returns the exact cage;
a larger one means the mesh was edited after subdividing (the cage is then the best least-squares cage).
Remaining ambiguity (closed regular grids) is settled by the residual.  Repeat for several levels.

Blender's Un-Subdivide keeps the surviving vertices where they are, so its cage shrinks (CC pulls V points inward)
and it does not recognise triangle or n-gon cages.  Loop / sqrt(3) triangle subdivision is not handled here.
"""
import math
import numpy as np
from collections import defaultdict, deque

SCHEMES = ('simple', 'cc', 'cc_limit')


class Coo:
    """Minimal sparse matrix (COO) with matvec and transpose matvec on (n, k) arrays."""
    def __init__(self, r, c, v, shape):
        self.r = np.asarray(r, np.int64); self.c = np.asarray(c, np.int64); self.v = np.asarray(v, float)
        self.shape = shape

    def _mv(self, rows, cols, n, x):
        w = self.v[:, None] * x[cols]
        return np.stack([np.bincount(rows, weights=w[:, k], minlength=n) for k in range(x.shape[1])], 1)

    def dot(self, x): return self._mv(self.r, self.c, self.shape[0], x)
    def tdot(self, y): return self._mv(self.c, self.r, self.shape[1], y)


def cgls(mv, rmv, b, x0, iters=400, tol=1e-12):
    x = x0.copy(); r = b - mv(x); s = rmv(r); p = s.copy(); g = float((s * s).sum()); g0 = g
    for _ in range(iters):
        if g <= tol * tol * max(g0, 1e-300): break
        q = mv(p); qq = float((q * q).sum())
        if qq <= 0: break
        a = g / qq; x += a * p; r -= a * q; s = rmv(r); gn = float((s * s).sum()); p = s + (gn / g) * p; g = gn
    return x


# ---------------------------------------------------------------- topology
def _quad_topology(F, nV):
    E2F = defaultdict(list); nbr = [set() for _ in range(nV)]
    for fi, f in enumerate(F):
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]; E2F[(min(a, b), max(a, b))].append(fi); nbr[a].add(b); nbr[b].add(a)
    bnd = set()
    for (a, b), fs in E2F.items():
        if len(fs) == 1: bnd.add(a); bnd.add(b)
        elif len(fs) > 2: return None
    return E2F, nbr, bnd


def _two_colour(nodes, adj):
    col = {}
    for s in nodes:
        if s in col: continue
        col[s] = 0; dq = deque([s])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in col: col[w] = 1 - col[u]; dq.append(w)
                elif col[w] == col[u]: return None
    return col


def _components(F, nV):
    par = list(range(nV))

    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for f in F:
        r0 = find(f[0])
        for v in f[1:]:
            r = find(v)
            if r != r0: par[r] = r0
    comp = defaultdict(list)
    for fi, f in enumerate(F): comp[find(f[0])].append(fi)
    return list(comp.values())


def labellings(F, nV):
    """All valid (E, Fp, Vb, base_faces, corner) labellings, per connected component.  corner[(bf, k)] is the fine
    face index whose V-corner is base corner k of base face bf.  Returns list (per component) of lists, or None."""
    if not F or any(len(f) != 4 for f in F): return None
    top = _quad_topology(F, nV)
    if top is None: return None
    E2F, nbr, bnd = top
    out = []
    for faces in _components(F, nV):
        verts = sorted({v for fi in faces for v in F[fi]})
        col = _two_colour(verts, nbr)
        if col is None: return None
        opts = []
        for ecol in (0, 1):
            Es = {v for v in verts if col[v] == ecol}
            if any(len(nbr[e]) != (3 if e in bnd else 4) for e in Es): continue
            diag = defaultdict(set)
            for fi in faces:
                f = F[fi]; a, c = (f[1], f[3]) if col[f[0]] == ecol else (f[0], f[2]); diag[a].add(c); diag[c].add(a)
            A = [v for v in verts if col[v] != ecol]
            dcol = _two_colour(A, diag)
            if dcol is None: continue
            for fcol in (0, 1):
                Fp = {v for v in A if dcol[v] == fcol}
                if Fp & bnd or any(len(nbr[f]) < 3 for f in Fp): continue
                if any(sum(1 for w in nbr[e] if w not in Fp) != 2 for e in Es): continue
                res = _base_faces(F, faces, Es, Fp, nbr)
                if res is not None: opts.append((Es, Fp, set(A) - Fp, res[0], res[1]))
        if not opts: return None
        opts.sort(key=lambda o: (np.mean([len(nbr[f]) for f in o[1]]) if o[1] else 0.0))
        out.append(opts)
    return out


def _base_faces(F, faces, Es, Fp, nbr):
    around = defaultdict(list)
    for fi in faces:
        f = F[fi]
        k = next((i for i in range(4) if f[i] in Fp), None)
        if k is None or f[(k + 2) % 4] in Fp or f[(k + 2) % 4] in Es: return None
        around[f[k]].append(fi)
    base, corner, edge_seen = [], [], {}
    for fp, fis in around.items():
        step = {}
        for fi in fis:
            f = F[fi]; k = f.index(fp); step[f[(k + 1) % 4]] = (f[(k + 2) % 4], f[(k + 3) % 4], fi)  # e_prev->(v,e_next)
        if len(step) != len(fis): return None
        e0 = next(iter(step)); e = e0; ring, cfs = [], []
        for _ in range(len(fis)):
            if e not in step: return None
            v, en, fi = step[e]; ring.append(v); cfs.append(fi); e = en
        if e != e0 or len(set(ring)) != len(ring): return None
        for i in range(len(ring)):                       # E between consecutive V must be unique per base edge
            a, b = ring[i], ring[(i + 1) % len(ring)]
            ed = (min(a, b), max(a, b))
            f = F[cfs[i]]; k = f.index(fp); step_e = f[(k + 3) % 4]
            if ed in edge_seen and edge_seen[ed] != step_e: return None
            edge_seen[ed] = step_e
        base.append(ring); corner.append(cfs)
    return base, corner


# ---------------------------------------------------------------- operators
def _base_adjacency(base):
    e2f = defaultdict(list); vf = defaultdict(list); vn = defaultdict(set)
    for bi, f in enumerate(base):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; e2f[(min(a, b), max(a, b))].append(bi); vf[a].append(bi)
            vn[a].add(b); vn[b].add(a)
    return e2f, vf, vn


def subdiv_operator(base, nb, scheme, Ev, Fv, Vv, nfine, sharp=None, boundary='smooth'):
    """Rows = fine vertices (indices given by Ev[edge], Fv[face], Vv[base vertex]), cols = base vertices.
    sharp: base edges (min, max) with infinite crease (Blender crease 1.0).  boundary: 'smooth' (crease rule along
    the boundary, corners kept), 'all' (corners smoothed too) or 'linear' (boundary vertices kept -- Blender's UV
    smoothing 'Keep Boundaries')."""
    R, C, W = [], [], []
    add = lambda r, c, w: (R.append(r), C.append(c), W.append(w))
    e2f, vf, vn = _base_adjacency(base)
    sharp = sharp or set()
    for bi, f in enumerate(base):
        for v in f: add(Fv[bi], v, 1.0 / len(f))
    bnd_v = set()
    for (a, b), fs in e2f.items():
        if len(fs) == 1: bnd_v.update((a, b))
        if scheme == 'simple' or len(fs) == 1 or (a, b) in sharp:
            add(Ev[(a, b)], a, 0.5); add(Ev[(a, b)], b, 0.5)
        else:
            add(Ev[(a, b)], a, 0.25); add(Ev[(a, b)], b, 0.25)
            for bi in fs:
                for v in base[bi]: add(Ev[(a, b)], v, 0.25 / len(base[bi]))
    for v in range(nb):
        r = Vv[v]
        if scheme == 'simple': add(r, v, 1.0); continue
        hard = [u for u in vn[v] if len(e2f[(min(u, v), max(u, v))]) == 1 or (min(u, v), max(u, v)) in sharp]
        if v in bnd_v:
            if boundary == 'linear': add(r, v, 1.0); continue
            if (len(vf[v]) == 1 and boundary != 'all') or len(hard) != 2: add(r, v, 1.0)   # corner kept ("Keep Corners")
            else: add(r, v, 0.75); add(r, hard[0], 0.125); add(r, hard[1], 0.125)
            continue
        if len(hard) >= 3: add(r, v, 1.0); continue                       # corner of creases
        if len(hard) == 2: add(r, v, 0.75); add(r, hard[0], 0.125); add(r, hard[1], 0.125); continue
        n = len(vn[v])                                                    # smooth (also darts: one crease edge)
        add(r, v, (n - 2.0) / n)
        for u in vn[v]: add(r, u, 1.0 / (n * n))
        for bi in vf[v]:
            for u in base[bi]: add(r, u, 1.0 / (n * n * len(base[bi])))
    return Coo(R, C, W, (nfine, nb))


def limit_operator(F, nV, sharp=None, boundary='smooth'):
    """Catmull-Clark limit-position stencils on an all-quad mesh (fine x fine); crease edges and corners as in
    subdiv_operator."""
    E2F, nbr, bnd = _quad_topology(F, nV)
    sharp = sharp or set()
    diag = defaultdict(set); vf = defaultdict(int)
    for f in F:
        for k in range(4): diag[f[k]].add(f[(k + 2) % 4]); vf[f[k]] += 1
    R, C, W = [], [], []
    for v in range(nV):
        if not nbr[v]: R.append(v); C.append(v); W.append(1.0); continue
        hard = [u for u in nbr[v] if len(E2F[(min(u, v), max(u, v))]) == 1 or (min(u, v), max(u, v)) in sharp]
        if v in bnd and boundary == 'linear': R.append(v); C.append(v); W.append(1.0); continue
        if (v in bnd and ((vf[v] == 1 and boundary != 'all') or len(hard) != 2)) or len(hard) >= 3:
            R.append(v); C.append(v); W.append(1.0); continue
        if len(hard) == 2:
            R += [v, v, v]; C += [v, hard[0], hard[1]]; W += [4 / 6, 1 / 6, 1 / 6]
            continue
        n = len(nbr[v]); s = n * (n + 5.0)
        R.append(v); C.append(v); W.append(n * n / s)
        for u in nbr[v]: R.append(v); C.append(u); W.append(4.0 / s)
        for u in diag[v]: R.append(v); C.append(u); W.append(1.0 / s)
    return Coo(R, C, W, (nV, nV))


# ---------------------------------------------------------------- one level
def unsubdivide_once(V, F, schemes=SCHEMES, iters=400, fine_sharp=None, creases=True, more_below=None):
    """Returns dict(V, F, scheme, residual_rel, vmap, corner_face, sharp, per_component) or None if F is not subdivided.
    vmap[i] = fine vertex that sits on base vertex i;  corner_face[b][k] = fine face holding that corner.
    fine_sharp: crease edges already known on F (from the previous level), else creases are detected when a smooth fit
    leaves a residual: an edge is a candidate if its edge point's residual is high and a local maximum across the
    edge (creases are lines); a candidate set is kept only if it makes the fit at least 5x better."""
    V = np.asarray(V, float); nV = len(V)
    labs = labellings(F, nV)
    if labs is None: return None
    mel = float(np.mean([np.linalg.norm(V[f[0]] - V[f[1]]) for f in F[:2000]])) + 1e-30
    _, nbr_, bnd_ = _quad_topology(F, nV)
    skip_rows = set()
    if 'cc_limit' in schemes:
        for v in range(nV):
            if len(nbr_[v]) == 2 and v not in bnd_: skip_rows |= nbr_[v]
    fsh = {(min(a, b), max(a, b)) for a, b in fine_sharp} if fine_sharp is not None else None
    lop_cache = {}

    def lop_for(sh):
        k = frozenset(sh)
        if k not in lop_cache: lop_cache[k] = limit_operator(F, nV, set(sh))
        return lop_cache[k]

    def context(opt):
        Es, Fp, Vb, base, corner = opt
        vb = sorted(Vb); loc = {v: i for i, v in enumerate(vb)}
        bl = [[loc[v] for v in f] for f in base]
        fine = sorted(Es | Fp | Vb); fl = {v: i for i, v in enumerate(fine)}
        Ev = {}
        for bi, f in enumerate(base):                 # E vertex of base edge = shared neighbour of its two V
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                q = F[corner[bi][k]]; j = q.index(a)
                Ev[(min(loc[a], loc[b]), max(loc[a], loc[b]))] = fl[q[(j + 1) % 4]]
        Fv = {}
        for bi, f in enumerate(base):
            q = F[corner[bi][0]]; Fv[bi] = fl[q[(q.index(f[0]) + 2) % 4]]
        Vv = {loc[v]: fl[v] for v in vb}
        g2l = np.full(nV, -1, np.int64); g2l[fine] = np.arange(len(fine))
        return dict(vb=vb, base=base, corner=corner, bl=bl, fine=fine, Ev=Ev, Fv=Fv, Vv=Vv, X=V[fine], g2l=g2l)

    def children(c, bsh):                             # fine (global) crease edges of base crease edges
        out = set()
        for (a, b) in bsh:
            m = c['fine'][c['Ev'][(a, b)]]; va, vb_ = c['fine'][c['Vv'][a]], c['fine'][c['Vv'][b]]
            out |= {(min(va, m), max(va, m)), (min(vb_, m), max(vb_, m))}
        return out

    def fit(c, sc, bsh, P0=None):
        nf = len(c['fine'])
        S = subdiv_operator(c['bl'], len(c['vb']), 'simple' if sc == 'simple' else 'cc', c['Ev'], c['Fv'], c['Vv'], nf,
                            bsh)
        if sc == 'cc_limit':
            Lop = lop_for(children(c, bsh)); g2l = c['g2l']; m = g2l[Lop.r] >= 0
            Ll = Coo(g2l[Lop.r[m]], g2l[Lop.c[m]], Lop.v[m], (nf, nf))
            fw = lambda P: Ll.dot(S.dot(P)); bw = lambda Y: S.tdot(Ll.tdot(Y))
        else:
            fw = lambda P: S.dot(P); bw = lambda Y: S.tdot(Y)
        P = (V[c['vb']] if P0 is None else P0).copy(); w = np.ones((nf, 1))
        if sc == 'cc_limit' and skip_rows:            # Blender's limit patches next to valence-2 vertices
            w[[c['g2l'][v] for v in skip_rows if c['g2l'][v] >= 0]] = 0.0   # differ from the stencil: leave them out
        if sc != 'simple':
            P = cgls(lambda Q: w * fw(Q), lambda Y: bw(w * Y), w * c['X'], P, iters=iters)
        d = np.linalg.norm(fw(P) - c['X'], axis=1) / mel
        inl = w[:, 0] > 0
        return float(np.sqrt(np.mean(d[inl] ** 2))), P, int((~inl).sum()), d

    def detect(c, sc, res0, P0, d0):
        """Alternate: classify every base edge by which rule predicts its edge point better from the current cage --
        smooth (ends + adjacent face centres) or sharp (midpoint) -- then refit; stop when the set is stable.  An edge
        counts as sharp only with a clear margin, so flat areas (where both rules agree) never turn into creases."""
        e2f, _, _ = _base_adjacency(c['bl'])
        edges = [e for e, fs in e2f.items() if len(fs) == 2]
        ei = np.array([c['Ev'][e] for e in edges]); ea = np.array([e[0] for e in edges]); eb = np.array([e[1] for e in edges])
        # seed: a crease leaves a fold along both child edges at the fine level; smooth areas flatten out
        fn = {}
        for fi, q in enumerate(F):
            n_ = np.cross(V[q[2]] - V[q[0]], V[q[3]] - V[q[1]]); fn[fi] = n_ / (np.linalg.norm(n_) + 1e-30)
        e2f_fine = defaultdict(list)
        for fi, q in enumerate(F):
            for k in range(4): a, b = q[k], q[(k + 1) % 4]; e2f_fine[(min(a, b), max(a, b))].append(fi)
        dih = {e: math.degrees(math.acos(max(-1.0, min(1.0, float(fn[fs[0]] @ fn[fs[1]])))))
               for e, fs in e2f_fine.items() if len(fs) == 2}
        fold = {}
        for e in edges:
            m = c['fine'][c['Ev'][e]]; va, vb_ = c['fine'][c['Vv'][e[0]]], c['fine'][c['Vv'][e[1]]]
            fold[e] = min(dih.get((min(va, m), max(va, m)), 0.0), dih.get((min(vb_, m), max(vb_, m)), 0.0))
        med = float(np.median(list(fold.values()))) if fold else 0.0
        order = [e for e in sorted(fold, key=lambda e: -fold[e]) if fold[e] > max(3.0, 1.5 * med)]
        vn_b = defaultdict(set)
        for (a, b) in edges: vn_b[a].add(b); vn_b[b].add(a)

        def extend(sh):
            """Base edges continuing a crease line straight through a vertex with 4 edges (edge loops)."""
            out = set()
            for (a, b) in sh:
                for v, u in ((a, b), (b, a)):
                    if len(vn_b[v]) != 4: continue
                    d0 = P[v] - P[u]
                    w = max(vn_b[v] - {u}, key=lambda w: float((P[w] - P[v]) @ d0) / (np.linalg.norm(P[w] - P[v]) + 1e-30))
                    e = (min(v, w), max(v, w))
                    if e in fold and e not in sh and fold[e] > 3.0: out.add(e)
            return out
        sharp, res, P = set(), res0, P0
        thr_hi = max(15.0, 4.0 * med)
        seed = {e for e in order if fold[e] > thr_hi}
        if not seed: return set(), res0, P0           # creases leave clear folds; none here
        r_s, P_s, _, _ = fit(c, sc, seed, P0)
        if r_s > 0.8 * res0: return set(), res0, P0   # real creases explain much of the residual at once
        sharp, res, P = seed, r_s, P_s
        rest = [e for e in order if e not in sharp]
        k = 0
        for _ in range(16):                           # grow: crease lines first, then the next edges by fold angle
            improved = False
            for cand in (extend(sharp), set(rest[k:k + max(4, len(sharp) // 2)])):
                cand = cand - sharp
                if not cand: continue
                r_t, P_t, _, _ = fit(c, sc, sharp | cand, P)
                if r_t < 0.95 * res:
                    sharp, res, P = sharp | cand, r_t, P_t; improved = True; break
            k += max(4, len(sharp) // 2)
            if (not improved and k >= len(rest)) or res < 1e-6: break
        for _ in range(8):
            cen = np.array([P[f].mean(0) for f in c['bl']])
            f1 = np.array([e2f[e][0] for e in edges]); f2 = np.array([e2f[e][1] for e in edges])
            sm = 0.25 * (P[ea] + P[eb] + cen[f1] + cen[f2]); sp = 0.5 * (P[ea] + P[eb]); x = c['X'][ei]
            gap = np.linalg.norm(sm - sp, axis=1)
            ds, dp = np.linalg.norm(x - sm, axis=1), np.linalg.norm(x - sp, axis=1)
            new = {edges[i] for i in np.flatnonzero((dp < 0.5 * ds) & (gap > 1e-3 * mel))}
            if new == sharp: break
            r_t, P_t, _, _ = fit(c, sc, new, P)
            if r_t >= res: break
            sharp, res, P = new, r_t, P_t
        if sharp and (res < 0.2 * res0 or res < 1e-4): return sharp, res, P
        return set(), res0, P0

    outV, outF, vmap, cface, comps, out_sharp = [], [], [], [], [], set()
    overruled = []                                   # readings where the clear-fit rule overruled the shape prior

    def peelable(c):
        # a Catmull-Clark level above the cage is all quads and is itself a subdivision (cached on the context:
        # contexts are rebuilt, so an id() would be reused by another one)
        if '_peelable' not in c:
            if any(len(f) != 4 for f in c['base']):
                c['_peelable'] = 0
            else:
                lb = {v: i for i, v in enumerate(c['vb'])}
                c['_peelable'] = 1 if labellings([[lb[v] for v in f] for f in c['base']], len(c['vb'])) is not None else 0
        return c['_peelable']

    def ngon_excess(c):
        return round(float(np.mean([max(0, len(f) - 4) for f in c['base']])), 3)

    def prior(c):
        # inexact fits: a level that can be peeled again first, then fewer n-gons (artist cages are mostly
        # triangles and quads; the dual of a closed cage is the usual wrong reading)
        return (-peelable(c), ngon_excess(c))

    def choose(cands, ambiguous):
        """cands: (res, scheme, P, context, outliers, sharp, d).  An exact fit decides; otherwise the shape prior,
        then the fit.  Under heavy edits both readings fit badly and the prior is right more often (fox, panels).
        But the prior must not pick a reading that could not pass as a subdivision (fit > 2 LEVEL_TOL) when
        another one fits within LEVEL_TOL: a flat grid over an n-gon model fits its true reading at 0.005-0.06,
        and the all-quad reading the prior prefers at 0.18-0.49 (RESULTS-round14b)."""
        ex = [t for t in cands if t[0] < 1e-4]
        if ex:
            return min(ex, key=lambda t: (round(t[0], 6), prior(t[3]) if ambiguous else 0, SCHEMES.index(t[1])))
        if not ambiguous:
            return min(cands, key=lambda t: (t[0], SCHEMES.index(t[1])))
        pick = min(cands, key=lambda t: (prior(t[3]), round(t[0], 6), SCHEMES.index(t[1])))
        pool = [t for t in cands if not more_below or peelable(t[3])] or cands
        best = min(pool, key=lambda t: (t[0], SCHEMES.index(t[1])))
        if pick[0] > 2 * LEVEL_TOL and best[0] <= LEVEL_TOL:
            overruled.append(1); return best
        return pick

    for opts in labs:
        tried = []
        if len(opts) > 1:                             # several exact fits: the first one tried wins, so try the
            ctx = [(prior(context(o)), k, o) for k, o in enumerate(opts)]   # likelier shape first
            opts = [o for _, _, o in sorted(ctx, key=lambda t: (t[0], t[1]))]
        for opt in opts:
            c = context(opt)
            bsh = set()
            if fsh:                                   # known creases: a base edge is sharp if both halves are
                for (a, b), ei in c['Ev'].items():
                    m = c['fine'][ei]; va, vb_ = c['fine'][c['Vv'][a]], c['fine'][c['Vv'][b]]
                    if (min(va, m), max(va, m)) in fsh and (min(vb_, m), max(vb_, m)) in fsh: bsh.add((a, b))
            exact = False
            for sc in schemes:
                res, P, nout, d = fit(c, sc, bsh)
                tried.append((res, sc, P, c, nout, bsh, d))
                if res < 1e-4:
                    exact = True
                    if nout == 0: break
            if exact: break
        best = choose(tried, len(opts) > 1)
        if creases and fsh is None and best[0] > 1e-4:     # no exact smooth fit: try creases on every labelling and
            cands = list(tried)                            # scheme (a smooth fit can prefer the wrong labelling)
            for res, sc, P, c, nout, bsh, d in tried:
                if sc == 'simple': continue
                bsh2, res2, P2 = detect(c, sc, res, P, d)
                if bsh2: cands.append((res2, sc, P2, c, nout, bsh2, d))
            overruled.clear()
            best = choose(cands, len(opts) > 1)
        res, sc, P, c, nout, bsh, d = best
        off = len(outV); loc = {i: i + off for i in range(len(c['vb']))}
        outV.extend(P); vmap.extend(c['vb'])
        lb = {v: i for i, v in enumerate(c['vb'])}
        for f, cf in zip(c['base'], c['corner']): outF.append([lb[v] + off for v in f]); cface.append(list(cf))
        out_sharp |= {(a + off, b + off) for a, b in bsh}
        comps.append(dict(scheme=sc, residual_rel=res, outliers=nout, faces=len(c['base']), options=len(opts),
                          creases=len(bsh), overruled=len(overruled)))
        overruled.clear()
    worst = max(x['residual_rel'] for x in comps)
    scs = sorted({x['scheme'] for x in comps})
    return dict(V=np.array(outV), F=outF, vmap=vmap, corner_face=cface, scheme='+'.join(scs), residual_rel=worst,
                outliers=sum(x['outliers'] for x in comps), sharp=out_sharp, per_component=comps)


def weld_uv_cluster(F, FUV, UV, tol=1e-5):
    """One UV index per vertex and island: per vertex, corner UVs closer than tol are merged (distance clustering --
    a rounding grid would split points that straddle a cell edge, e.g. float32 noise of evaluated UVs)."""
    UV = np.asarray(UV, float); per_v = defaultdict(list); out = []; NF = []
    for f, fu in zip(F, FUV):
        g = []
        for v, t in zip(f, fu):
            p = UV[t]; hit = None
            for idx in per_v[v]:
                if abs(out[idx][0] - p[0]) <= tol and abs(out[idx][1] - p[1]) <= tol: hit = idx; break
            if hit is None: hit = len(out); out.append(p); per_v[v].append(hit)
            g.append(hit)
        NF.append(g)
    return NF, np.asarray(out, float)


UV_MODES = (('cc', 'linear'), ('cc', 'smooth'), ('cc', 'all'), ('simple', 'linear'))
# Blender UV Smooth: Keep Boundaries (default) / Keep Corners / All / None


def fit_uv_level(F, FUV, UV, r, limit, iters=400):
    """Cage UVs for one level, fitted like the positions but on the UV mesh: UV seams are boundaries of that mesh.
    Tries Blender's UV smoothing modes (Keep Boundaries = linear seams, Keep Corners = smooth seams, None = linear)
    with or without the limit projection, and keeps the best.  Returns (UVbase, FUVbase, info)."""
    UV = np.asarray(UV, float)
    corner_uv = []                                    # per base face: (uv at V-corner, uv of edge point, uv of face point)
    for bf, cf in zip(r['F'], r['corner_face']):
        row = []
        for k, fi in enumerate(cf):
            q = F[fi]; j = q.index(r['vmap'][bf[k]]); row.append((FUV[fi][j], FUV[fi][(j + 1) % 4], FUV[fi][(j + 2) % 4]))
        corner_uv.append(row)
    uids = sorted({c[0] for row in corner_uv for c in row}); loc = {u: i for i, u in enumerate(uids)}
    bl = [[loc[c[0]] for c in row] for row in corner_uv]
    fine = sorted({u for f in FUV for u in f}); fl = {u: i for i, u in enumerate(fine)}
    Ev, Fv = {}, {}
    for bi, row in enumerate(corner_uv):
        n = len(row)
        for k in range(n):
            a, b = loc[row[k][0]], loc[row[(k + 1) % n][0]]
            Ev[(min(a, b), max(a, b))] = fl[row[k][1]]
        Fv[bi] = fl[row[0][2]]
    Vv = {loc[u]: fl[u] for u in uids}
    X = UV[fine]; FUVl = [[fl[u] for u in f] for f in FUV]
    mel = float(np.mean([np.linalg.norm(X[f[0]] - X[f[1]]) for f in FUVl[:2000]])) + 1e-30
    best = None
    for sc, bd in UV_MODES:
        S = subdiv_operator(bl, len(uids), sc, Ev, Fv, Vv, len(fine), boundary=bd)
        for lim in ((False, True) if (limit and sc != 'simple') else (False,)):
            if lim:
                Lop = limit_operator(FUVl, len(fine), boundary=bd)
                fw = lambda P, S=S, L=Lop: L.dot(S.dot(P)); bw = lambda Y, S=S, L=Lop: S.tdot(L.tdot(Y))
            else:
                fw = lambda P, S=S: S.dot(P); bw = lambda Y, S=S: S.tdot(Y)
            P0 = UV[uids].copy()
            P = P0 if sc == 'simple' else cgls(fw, bw, X, P0, iters=iters)
            res = float(np.sqrt(np.mean(np.sum((fw(P) - X) ** 2, 1)))) / mel
            if best is None or res < best[0]: best = (res, P, f"{sc}/{bd}" + ("+limit" if lim else ""))
    return best[1], bl, dict(uv_mode=best[2], uv_residual_rel=round(best[0], 6))


def detect_levels(F, nV, max_levels=4, budget=24):
    """Topology-only test (no fitting): the largest number of levels the connectivity carries.  Labellings are tried
    lower-F-valence first (artist cages are mostly tris/quads); a closed mesh's dual cage is also valid, so a few
    alternatives are explored depth-first within `budget` calls."""
    calls = [0]

    def rec(FF, n, depth):
        if depth >= max_levels or calls[0] >= budget: return depth
        calls[0] += 1
        labs = labellings(FF, n)
        if labs is None: return depth
        best = depth + 1
        combos = [tuple(0 for _ in labs)] + [tuple(1 if i == j else 0 for i in range(len(labs)))
                                             for j in range(len(labs)) if len(labs[j]) > 1][:2]
        for combo in combos:
            nb, FF2 = 0, []
            for opts, c in zip(labs, combo):
                Es, Fp, Vb, base, _ = opts[min(c, len(opts) - 1)]
                loc = {v: i + nb for i, v in enumerate(sorted(Vb))}; nb += len(Vb)
                FF2.extend([loc[v] for v in f] for f in base)
            if all(len(f) == 4 for f in FF2): best = max(best, rec(FF2, nb, depth + 1))
            if best >= max_levels: break
        return best
    return rec([list(f) for f in F], nV, 0)


LEVEL_TOL = 0.075   # fit residual (RMS, fine edge lengths) that a real level may have; see RESULTS-round14


def unsubdivide(V, F, levels=None, FUV=None, UV=None, tol=LEVEL_TOL, schemes=SCHEMES, log=None, creases=True, trace=None):
    """Peel up to `levels` subdivision levels (None = as many as found).  A level is kept only if its fit residual
    (RMS, in fine edge lengths) is <= tol at every level.  Some hand-made cages are topologically "subdividable"
    too (5 of 23 designer cages): unedited, peeling them gives residuals of 0.094-0.2.  Edits after subdividing
    raise the residual of the real levels (0.1-0.5 under strong bends, sculpts and detail) and heavy smoothing
    lowers the fake one (0.047), so on edited meshes no threshold separates them.  A fixed 0.075 added no false
    level on 69 unedited designer-cage inputs or 133 edited meshes, and one (after 30 smoothing passes) on 76
    held-out edited meshes.  A rule that loosened the limit after a noisy level recovered more cages on the data it
    was tuned on (90 vs 80 of 133) but none more on held-out edits (43 vs 44 of 76), and added 4 false levels there:
    the noisy real levels loosened it enough to pass a fake level at 0.093 (RESULTS-round14).  A false level gives
    a wrong cage, so the default stays fixed and conservative.  tol=None keeps every
    topologically valid level: surface edited after subdividing -- set `levels` too, since a hand-made cage under
    the true levels may look like one more level.  Returns dict with
        cage   : (V, F)          least-squares control cage (re-subdivides to the input)
        surface: (V, F)          same faces, vertices at the input positions they correspond to (for LOD use)
        uv     : (UV, FUV) or None
        levels : list of per-level info."""
    V = np.asarray(V, float); F = [list(f) for f in F]
    orig = list(range(len(V)))                             # cage vertex -> original vertex
    curV, curF = V, F; info = []
    cur_fuv = cur_uv = None
    if FUV is not None:                                # one UV index per vertex and island (files often store one
        cur_fuv, cur_uv = weld_uv_cluster(F, FUV, UV)  # per face corner, which would cut the UV mesh apart)
        cur_fuv = [list(f) for f in cur_fuv]
    cur_sharp = set()
    while levels is None or len(info) < levels:
        r = unsubdivide_once(curV, curF, schemes=schemes if not info else ('simple', 'cc'),
                             fine_sharp=cur_sharp if info else None, creases=creases,
                             more_below=None if levels is None else len(info) + 1 < levels)
        if r is None: break
        if tol is not None:
            lim = tol
            if r['residual_rel'] > lim:
                if log: log(f"stop: level {len(info) + 1} residual {r['residual_rel']:.3f} > {lim:.3f} (not a subdivision)")
                break
        uvinfo = {}
        if cur_fuv is not None:                            # cage UVs fitted on the UV mesh (seams = its boundaries)
            cur_uv, cur_fuv, uvinfo = fit_uv_level(curF, cur_fuv, cur_uv, r, limit='limit' in r['scheme'])
        orig = [orig[v] for v in r['vmap']]
        if trace is not None:                              # per level: which fine face holds each coarse corner
            trace.append(dict(fineF=[list(f) for f in curF], F=r['F'], vmap=r['vmap'], corner_face=r['corner_face']))
        cur_sharp = r['sharp']
        info.append(dict(scheme=r['scheme'], residual_rel=round(r['residual_rel'], 6), outliers=r['outliers'],
                         creases=len(r['sharp']), faces=len(r['F']),
                         verts=len(r['V']), components=len(r['per_component']),
                         ambiguous=sum(c['options'] > 1 for c in r['per_component']),
                         overruled=sum(c.get('overruled', 0) > 0 for c in r['per_component']), **uvinfo))
        if log: log(f"level {len(info)}: {len(curF)} -> {len(r['F'])} faces, {r['scheme']}, residual {r['residual_rel']:.2e}")
        curV, curF = r['V'], r['F']
        if any(len(f) != 4 for f in curF): break
    return dict(cage=(curV, curF), surface=(V[orig], curF), creases=sorted(cur_sharp),
                fit=(guard_folds(fit_placement(V, F, orig), V[orig], curF), curF) if info else (V, F),
                uv=(cur_uv, cur_fuv) if cur_fuv is not None else None,
                orig=orig, levels=info)


def _pair_file_order(T):
    """Exporters write each quad as two consecutive triangles: pair (2i, 2i+1) when they share an edge."""
    Q = []
    for i in range(0, len(T) - 1, 2):
        t1, t2 = list(T[i]), list(T[i + 1])
        sh = [v for v in t1 if v in t2]
        if len(sh) != 2: return None
        c = next(v for v in t1 if v not in sh); d = next(v for v in t2 if v not in sh)
        k = t1.index(c); x, y = t1[(k + 1) % 3], t1[(k + 2) % 3]          # t1 = (c, x, y): diagonal x-y
        Q.append([c, x, d, y])
    return Q if len(T) % 2 == 0 else None


def pair_by_colouring(V, T, Q0, max_sweeps=60, seed=0):
    """Order-free re-pairing of a triangulated subdivided mesh.  In such a mesh the vertices 2-colour into edge points
    (colour 1) and the rest, every triangle has exactly one same-colour edge (its quad diagonal), and an edge point
    has exactly 4 (boundary: 3) neighbours of the other colour.  Start from the colouring of a geometric pairing Q0,
    repair it by local search on those constraints, pair triangles along their same-colour edges, and accept the
    result only if it passes `labellings`.  Returns quads or None."""
    T = [list(t) for t in T]; nV = len(V)
    nbr = [set() for _ in range(nV)]; e2t = defaultdict(list); vt = [[] for _ in range(nV)]
    for i, t in enumerate(T):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]; nbr[a].add(b); nbr[b].add(a); e2t[(min(a, b), max(a, b))].append(i)
            vt[a].append(i)
    bnd = {v for e, ts in e2t.items() if len(ts) == 1 for v in e}
    # initial colouring: BFS over the quad edges (not diagonals) of the geometric pairing
    qadj = [set() for _ in range(nV)]
    for q in Q0:
        if len(q) == 4:
            for k in range(4): a, b = q[k], q[(k + 1) % 4]; qadj[a].add(b); qadj[b].add(a)
    col0 = np.zeros(nV, np.int8); seen = np.zeros(nV, bool)
    for s0 in range(nV):
        if seen[s0]: continue
        seen[s0] = True; dq = deque([s0])
        while dq:
            u = dq.popleft()
            for w in qadj[u]:
                if not seen[w]: seen[w] = True; col0[w] = 1 - col0[u]; dq.append(w)
    # second start: parity votes.  Edge e is likely a diagonal if the quad across it is the best-shaped choice for
    # both its triangles (margin d > 0), likely a quad edge otherwise; merge votes most-confident first.
    try:
        from .structsimp import sj_min
    except ImportError:
        try:
            from structsimp import sj_min
        except ImportError:
            from rqmesh.structsimp import sj_min
    Va = np.asarray(V, float); qs = {}
    for e, ts in e2t.items():
        if len(ts) != 2: continue                    # (an edge shared by 3+ triangles -- doublets -- gets no vote)
        a, b = e; t1, t2 = T[ts[0]], T[ts[1]]
        c1 = next(v for v in t1 if v not in e); c2 = next(v for v in t2 if v not in e)
        qs[e] = sj_min(Va[[c1, a, c2, b]])
    d = {}
    for e, ts in e2t.items():
        if e not in qs: d[e] = -1.0; continue
        m = []
        for i in ts:
            t = T[i]; others = [(min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3])) for k in range(3)]
            others = [o for o in others if o != e]
            m.append(qs[e] - max(qs.get(o, -1.0) for o in others))
        d[e] = min(m)
    par = list(range(nV)); pty = [0] * nV

    def find(x):
        p_ = 0
        while par[x] != x:
            p_ ^= pty[x]; x = par[x]
        return x, p_
    for e in sorted(d, key=lambda e: -abs(d[e])):
        (ra, pa), (rb, pb) = find(e[0]), find(e[1])
        if ra != rb: par[ra] = rb; pty[ra] = pa ^ pb ^ (0 if d[e] > 0 else 1)
    col1 = np.array([find(v)[1] for v in range(nV)], np.int8)
    rng = np.random.default_rng(seed)

    def tri_bad(c, i):
        a, b, d = T[i]; return 1 if c[a] == c[b] == c[d] else 0

    def v_bad(c, v):
        other = sum(1 for w in nbr[v] if c[w] != c[v])
        if c[v] == 1: return abs(other - (3 if v in bnd else 4))
        return 0 if other >= (2 if v in bnd else 3) else 1

    def local(c, v):                                    # cost terms that involve v
        return sum(tri_bad(c, i) for i in vt[v]) + v_bad(c, v) + sum(v_bad(c, w) for w in nbr[v])
    for start in (col1, 1 - col1, col0, 1 - col0):
        c = start.copy()
        bad = {v for v in range(nV) if v_bad(c, v)} | {x for i in range(len(T)) if tri_bad(c, i) for x in T[i]}
        if len(bad) > 0.03 * nV: continue            # not (or not recognisably) a subdivided mesh: skip the search
        for _ in range(max_sweeps):
            if not bad: break
            changed = False; cand = list(bad); rng.shuffle(cand); nb_ = set()
            for v in cand:
                e0 = local(c, v); c[v] ^= 1; e1 = local(c, v)
                if e1 < e0 or (e1 == e0 and rng.random() < 0.1):
                    changed = changed or e1 < e0; nb_ |= {v} | nbr[v]
                else:
                    c[v] ^= 1
            bad = {v for v in nb_ | bad if v_bad(c, v) or any(tri_bad(c, i) for i in vt[v])}
            if not changed and not bad: break
        if bad:                                       # anneal over the conflict zones (2 rings around them):
            act = set(bad)                            # wrong vertices can look locally fine, so flip neighbours too
            for _ in range(2): act |= {w for u in list(act) for w in nbr[u]}
            act = list(act); n_it = 60 * len(act)
            for it in range(n_it):
                temp = 2.0 * (0.02 / 2.0) ** (it / max(1, n_it - 1))
                v = act[int(rng.integers(len(act)))]
                e0 = local(c, v); c[v] ^= 1; e1 = local(c, v)
                if e1 > e0 and rng.random() >= math.exp(-(e1 - e0) / temp): c[v] ^= 1
            bad = {v for v in range(nV) if v_bad(c, v)} | {x for i in range(len(T)) if tri_bad(c, i) for x in T[i]}
        if bad: continue
        # pair triangles across their single same-colour edge
        diag = {}
        for i, t in enumerate(T):
            ks = [k for k in range(3) if c[t[k]] == c[t[(k + 1) % 3]]]
            if len(ks) != 1: diag = None; break
            diag[i] = ks[0]
        if diag is None: continue
        Q, used = [], np.zeros(len(T), bool); ok = True
        for i, t in enumerate(T):
            if used[i]: continue
            k = diag[i]; x, y, cc = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            other = [j for j in e2t[(min(x, y), max(x, y))] if j != i and not used[j]
                     and {T[j][diag[j]], T[j][(diag[j] + 1) % 3]} == {x, y}]
            if not other: ok = False; break
            j = other[0]; tj = T[j]; d = next(v for v in tj if v not in (x, y))
            Q.append([cc, x, d, y]); used[i] = used[j] = True
        if ok and labellings(Q, nV) is not None: return Q
    return None


def to_quads_if_subdivided(V, F):
    """All-quad input is returned as is.  Triangles are re-paired (file order first, then geometric best-first) and
    accepted only if the result is a subdivided quad mesh.  Returns (quads, how) or (None, reason).
    Index buffers shuffled by an engine defeat both pairings on skewed cells (an order-free solver is future work)."""
    if F and all(len(f) == 4 for f in F):
        Q = [list(f) for f in F]
        return (Q, 'quads') if labellings(Q, len(V)) is not None else (None, 'connectivity is not that of a subdivided mesh')
    if not F or not all(len(f) == 3 for f in F): return None, 'mixed polygons'
    Q = _pair_file_order(F)
    if Q is not None and labellings(Q, len(V)) is not None: return Q, 'paired (file order)'
    try:
        from .structsimp import tris_to_quads
    except ImportError:
        from structsimp import tris_to_quads
    r = tris_to_quads(np.asarray(V, float), F)
    Q = r[0] if isinstance(r, tuple) else r
    if Q and all(len(f) == 4 for f in Q) and labellings([list(f) for f in Q], len(V)) is not None:
        return [list(f) for f in Q], 'paired (geometry)'
    Q2 = pair_by_colouring(V, F, Q or [])
    if Q2 is not None: return Q2, 'paired (colouring search)'
    return None, 'triangles do not pair into a subdivided quad mesh'


def fit_placement(V, F, orig, rel=1e-3):
    """LOD placement for the recovered faces: each base vertex gathers the area-weighted plane quadrics of the fine
    vertices in its graph-Voronoi region (multi-source BFS from the fine vertex it sits on) and moves to the
    regularized QEM optimum (flat directions stay put).  Lower surface error than the on-surface positions."""
    try:
        from .structsimp import vertex_quadrics, qem_opt
    except ImportError:
        from structsimp import vertex_quadrics, qem_opt
    V = np.asarray(V, float); n = len(V)
    Qv = vertex_quadrics(V, F); Qv = Qv[0] if isinstance(Qv, tuple) else Qv
    nbr = [[] for _ in range(n)]
    for f in F:
        for k in range(len(f)): a, b = f[k], f[(k + 1) % len(f)]; nbr[a].append(b); nbr[b].append(a)
    owner = np.full(n, -1, np.int64); dq = deque()
    for i, v in enumerate(orig): owner[v] = i; dq.append(v)
    while dq:
        u = dq.popleft()
        for w in nbr[u]:
            if owner[w] < 0: owner[w] = owner[u]; dq.append(w)
    Qb = np.zeros((len(orig), 4, 4)); m = owner >= 0
    np.add.at(Qb, owner[m], Qv[m])
    return np.array([qem_opt(Qb[i], V[v], rel) for i, v in enumerate(orig)])


def _corner_ok(P, F, ref_n, min_cos=0.2):
    """Per face: every corner's turn agrees with the face's reference normal (no fold, no flip)."""
    bad = np.zeros(len(F), bool)
    for i, f in enumerate(F):
        k = len(f)
        for j in range(k):
            a, b, c = P[f[j - 1]], P[f[j]], P[f[(j + 1) % k]]
            nrm = np.cross(c - b, a - b); L = np.linalg.norm(nrm)
            if L < 1e-30 or float(nrm @ ref_n[i]) / L < min_cos: bad[i] = True; break
    return bad


def guard_folds(P_fit, P_safe, F, rounds=6):
    """Pull vertices of folded faces back toward the safe (on-surface) placement until no face folds."""
    P = np.array(P_fit, float); S = np.asarray(P_safe, float)
    ref = []
    for f in F:
        c = S[f].mean(0); n = sum(np.cross(S[f[j]] - c, S[f[(j + 1) % len(f)]] - c) for j in range(len(f)))
        ref.append(n / (np.linalg.norm(n) + 1e-30))
    ref = np.array(ref); t = np.ones(len(P))
    for _ in range(rounds):
        bad = _corner_ok(P, F, ref)
        if not bad.any(): break
        vs = np.unique(np.concatenate([F[i] for i in np.flatnonzero(bad)]))
        t[vs] *= 0.5; t[vs[t[vs] < 0.1]] = 0.0
        P[vs] = S[vs] + t[vs, None] * (np.asarray(P_fit)[vs] - S[vs])
    return P


def quad_uv_from_tris(Q, T, TFUV):
    """Per-corner UV indices for quads re-paired from triangles (each quad's two triangles are found by vertex set)."""
    look = {}
    for t, f in enumerate(T): look[frozenset(f)] = t
    out = []
    for q in Q:
        row = None
        for a, b, c, d in ((q[0], q[1], q[2], q[3]), (q[1], q[2], q[3], q[0])):
            t1, t2 = look.get(frozenset((a, b, c))), look.get(frozenset((a, c, d)))
            if t1 is not None and t2 is not None:
                uv = {}
                for t in (t1, t2):
                    for v, u in zip(T[t], TFUV[t]): uv.setdefault(v, u)
                row = [uv[v] for v in q]; break
        if row is None: return None
        out.append(row)
    return out


def resubdivide(V, F, scheme, levels=1, sharp=None, boundary='smooth'):
    """Apply `levels` of simple / Catmull-Clark subdivision (for checking a recovered cage).  sharp: crease edges of the
    input; their children stay sharp.  Returns (V, F) -- or (V, F, sharp) when sharp is given."""
    V = np.asarray(V, float); sh = {(min(a, b), max(a, b)) for a, b in (sharp or ())}
    for _ in range(levels):
        nb = len(V); e2f, vf, vn = _base_adjacency(F)
        Ev = {e: nb + len(F) + i for i, e in enumerate(sorted(e2f))}; Fv = {bi: nb + bi for bi in range(len(F))}
        Vv = {v: v for v in range(nb)}; n = nb + len(F) + len(Ev)
        S = subdiv_operator(F, nb, 'simple' if scheme == 'simple' else 'cc', Ev, Fv, Vv, n, sh, boundary)
        NF = []
        for bi, f in enumerate(F):
            k = len(f)
            for i in range(k):
                a, b, p = f[i], f[(i + 1) % k], f[i - 1]
                NF.append([a, Ev[(min(a, b), max(a, b))], Fv[bi], Ev[(min(a, p), max(a, p))]])
        nsh = set()
        for (a, b) in sh:
            m = Ev[(a, b)]; nsh |= {(min(a, m), max(a, m)), (min(b, m), max(b, m))}
        V = S.dot(V); F = NF; sh = nsh
    if scheme == 'cc_limit':
        V = limit_operator(F, len(V), sh, boundary).dot(V)
    return (V, F, sh) if sharp is not None else (V, F)


if __name__ == '__main__':
    import argparse, json, os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rqmesh.meshio_min import read_obj_uv, write_obj_uv
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--levels', type=int)
    ap.add_argument('--placement', choices=('cage', 'surface', 'fit'), default='cage')
    ap.add_argument('--ignore-uv', action='store_true')
    a = ap.parse_args()
    V, F, UV, FUV = read_obj_uv(a.inp)
    if a.ignore_uv: UV = FUV = None
    t = time.time()
    r = unsubdivide(V, F, levels=a.levels, FUV=FUV, UV=UV, log=lambda s: print(s, file=sys.stderr))
    Vo, Fo = r[a.placement]
    uv = r['uv']
    write_obj_uv(a.out, Vo, Fo, uv[0] if uv else None, uv[1] if uv else None)
    print(json.dumps(dict(levels=r['levels'], faces=len(Fo), verts=len(Vo), seconds=round(time.time() - t, 2))))
