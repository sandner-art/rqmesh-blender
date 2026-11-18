# SPDX-License-Identifier: Apache-2.0
"""UVs and panel data from structure (round 19).

Cage panels.  After a subdivision is reversed, every face of the dense mesh lies in exactly one cage face (its panel).
`push_corners` carries any per-corner value of the cage down the subdivision levels the way the subdivision split the
faces (corner, edge midpoint, face centre), so:
  - seeded with a regular polygon per cage face it gives panel coordinates (a quad panel is the unit square);
  - seeded with cage UVs it gives dense UVs that are exactly the cage layout, bilinear inside every panel.

Cage UV layouts.
  'panels'   every cage face its own chart (a rectangle or regular polygon at its true size): no distortion from
             joining, grain exactly along the loops, many seams.
  'patches'  the cage is cut along separatrices traced straight through regular vertices from every pole, n-gon,
             boundary and crease (stopping where they meet another one), so every patch is a grid of quads; each
             patch becomes a rectangle grid with per-column widths and per-row heights at true size.  Grain along
             the loops, seams only at patch borders.

Shape-aware part unwrapping (`part_uvs`): pieces (separate, or split at concave creases) that are cylinders get a
cylindrical unwrap (sides) and planar caps, everything else a box projection in its own principal frame.

`uv_metrics` measures any UV layout: angle and area distortion, seams, islands, flips and grain alignment."""
import math
import collections
import numpy as np


# ---------------------------------------------------------------- carrying values down the subdivision levels
def push_corners(trace, seed):
    """trace: subdiv.unsubdivide(trace=[...]) levels (dense -> cage order).  seed[b]: (n_b, d) array, the value at
    each corner of cage face b.  Returns (per dense face (4, d) arrays, cage face of each dense face)."""
    vals = [np.asarray(x, float) for x in seed]; pid = list(range(len(seed)))
    for lev in reversed(trace):
        Fc, Ff, cfm, vmap = lev['F'], lev['fineF'], lev['corner_face'], lev['vmap']
        new, npid = [None] * len(Ff), [None] * len(Ff)
        for b, fb in enumerate(Fc):
            n = len(fb); P = vals[b]; qs = cfm[b]
            common = set(Ff[qs[0]])
            for q in qs[1:]:
                common &= set(Ff[q])
            fpt = next(iter(common)); cen = P.mean(0)
            for k in range(n):
                q = qs[k]; fq = Ff[q]
                en = (set(fq) & set(Ff[qs[(k + 1) % n]])) - {fpt}; ep = (set(fq) & set(Ff[qs[(k - 1) % n]])) - {fpt}
                cv = {vmap[fb[k]]: P[k], fpt: cen, next(iter(en)): (P[k] + P[(k + 1) % n]) / 2,
                      next(iter(ep)): (P[k] + P[(k - 1) % n]) / 2}
                new[q] = np.array([cv[v] for v in fq]); npid[q] = pid[b]
        vals, pid = new, npid
    return vals, np.array(pid)


def polygon_corners(n, R=0.5, c=(0.5, 0.5)):
    """Corners of a quad (the unit square) or a regular n-gon, counter-clockwise."""
    if n == 4:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) * (2 * R) + (np.asarray(c) - R)
    t = 2 * np.pi * np.arange(n) / n - np.pi / 2 - np.pi / n
    return np.stack([np.cos(t), np.sin(t)], 1) * R + np.asarray(c)


def panels(trace, cageF):
    """Panel (cage face) of every dense face and panel coordinates per corner (quads: unit square)."""
    return push_corners(trace, [polygon_corners(len(f)) for f in cageF])


# ---------------------------------------------------------------- cage layouts
def _edge_faces(F):
    ef = collections.defaultdict(list)
    for i, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; ef[(min(a, b), max(a, b))].append((i, k))
    return ef


def _ngon_chart(V, f):
    L = np.mean([np.linalg.norm(V[f[(k + 1) % len(f)]] - V[f[k]]) for k in range(len(f))])
    n = len(f)
    if n == 4:
        a = (np.linalg.norm(V[f[1]] - V[f[0]]) + np.linalg.norm(V[f[2]] - V[f[3]])) / 2
        b = (np.linalg.norm(V[f[3]] - V[f[0]]) + np.linalg.norm(V[f[2]] - V[f[1]])) / 2
        return np.array([[0, 0], [a, 0], [a, b], [0, b]])
    return polygon_corners(n, R=L / (2 * math.sin(math.pi / n)), c=(0, 0))


def cage_layout(V, F, mode='patches', cuts=(), return_grid=False):
    """UV per corner of every cage face (list of (n, 2) arrays) at true size, and the chart of each face."""
    V = np.asarray(V, float); nF = len(F)
    if mode == 'panels':
        return [_ngon_chart(V, f) for f in F], np.arange(nF)
    ef = _edge_faces(F)
    quad = np.array([len(f) == 4 for f in F])
    vf = collections.defaultdict(list)
    for i, f in enumerate(F):
        for v in f:
            vf[v].append(i)
    bnd = {e for e, fs in ef.items() if len(fs) == 1}
    bverts = {v for e in bnd for v in e}
    val = collections.Counter(v for e in ef for v in e)
    cut = set(bnd) | {(min(a, b), max(a, b)) for a, b in cuts}
    for i, f in enumerate(F):                        # n-gons and triangles are panels of their own
        if not quad[i]:
            cut |= {(min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)])) for k in range(len(f))}

    ncut = collections.Counter(v for e in cut for v in e)

    def regular(v):
        """Inside: 4 quads, 4 edges.  On a border (boundary, n-gon edge, crease): like a boundary vertex (2 quads,
        3 edges), or a border passing straight through 4 quads."""
        q = sum(quad[i] for i in vf[v]); c = ncut[v]
        return (c == 0 and q == 4 and val[v] == 4) or (c == 2 and q == 2 and val[v] == 3) or (c == 2 and q == 4 and val[v] == 4)
    on_cut = {v for e in cut for v in e}

    def opposite(a, b):
        """The edge leaving b straight on from a->b, through a regular interior vertex (shares no face with a-b)."""
        fa = {i for i, _ in ef[(min(a, b), max(a, b))]}
        for i in vf[b]:
            if i in fa:
                continue
            f = F[i]; k = f.index(b)
            for c in (f[(k + 1) % 4], f[(k - 1) % 4]):
                e = (min(b, c), max(b, c))
                if not ({j for j, _ in ef[e]} & fa):
                    return c
        return None
    starts = [v for v in vf if not regular(v)]
    for v in starts:                                 # separatrices: straight on until they meet another
        for e in [e for e in ef if v in e]:
            a, b = v, e[0] if e[1] == v else e[1]
            while True:
                key = (min(a, b), max(a, b))
                if key in cut:
                    break
                cut.add(key)
                if b in on_cut or not regular(b) or b in bverts:
                    on_cut.add(b); break
                on_cut.add(b); c = opposite(a, b)
                if c is None:
                    break
                a, b = b, c
    # patches: quads joined across uncut edges; integer grid coordinates per corner by breadth-first propagation
    chart = -np.ones(nF, np.int64); grid = [None] * nF; nchart = 0
    for s in range(nF):
        if chart[s] >= 0 or not quad[s]:
            continue
        chart[s] = nchart; grid[s] = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float); q = collections.deque([s])
        while q:
            i = q.popleft(); f = F[i]
            for k in range(4):
                a, b = f[k], f[(k + 1) % 4]; key = (min(a, b), max(a, b))
                if key in cut:
                    continue
                for j, kk in ef[key]:
                    if j == i or chart[j] >= 0 or not quad[j]:
                        continue
                    A, B = grid[i][k], grid[i][(k + 1) % 4]; D = B - A; Lft = np.array([D[1], -D[0]])
                    g = F[j]; kb = g.index(b)            # g runs b -> a -> c -> d
                    G = np.zeros((4, 2)); G[kb] = B; G[(kb + 1) % 4] = A; G[(kb + 2) % 4] = A + Lft; G[(kb + 3) % 4] = B + Lft
                    chart[j] = nchart; grid[j] = G; q.append(j)
        nchart += 1
    uv = [None] * nF
    for c in range(nchart):                          # true size: per-column widths, per-row heights
        fs = np.flatnonzero(chart == c); w = collections.defaultdict(list); h = collections.defaultdict(list)
        for i in fs:
            f = F[i]; G = grid[i]
            for k in range(4):
                d = G[(k + 1) % 4] - G[k]; L = float(np.linalg.norm(V[f[(k + 1) % 4]] - V[f[k]]))
                if abs(d[0]) > 0.5:
                    w[int(round(min(G[k][0], G[(k + 1) % 4][0])))].append(L)
                else:
                    h[int(round(min(G[k][1], G[(k + 1) % 4][1])))].append(L)
        def cum(dct):
            ks = sorted(dct); lo = ks[0]; out = {lo: 0.0}
            for k in range(lo, ks[-1] + 1):
                out[k + 1] = out[k] + (np.mean(dct[k]) if k in dct else 0.0)
            return out
        U, Vv = cum(w), cum(h)
        for i in fs:
            uv[i] = np.array([[U[int(round(g[0]))], Vv[int(round(g[1]))]] for g in grid[i]])
    for i in range(nF):                              # n-gons, triangles: their own chart
        if uv[i] is None:
            uv[i] = _ngon_chart(V, F[i]); chart[i] = nchart; nchart += 1
    return (uv, chart, grid) if return_grid else (uv, chart)


def trim_layout(V, F, rows, size=None, unit='strip', cuts=(), seed=0, group=False):
    """The straight-loop layout placed on a trim sheet.  rows: (v0, v1) bands of the sheet; size: metres per UV unit
    along a band.  unit 'strip' (one row of cage quads along the loops, the usual trim unit) or 'patch'.  Each unit
    is turned so its long side runs along u, goes to the band whose height in metres (band height x size) is
    closest to its own, scaled uniformly to fill the band's height, and starts at a random u (bands tile in u).
    size None: chosen so the median unit fits the median band at its true size.
    Returns (uv per corner, unit id per face, band per face, log2 texel-density error per face)."""
    uv, chart, grid = cage_layout(V, F, 'patches', cuts=cuts, return_grid=True)
    rng = np.random.default_rng(seed); nF = len(F); key = [None] * nF
    for c in np.unique(chart):
        fs = np.flatnonzero(chart == c); P = np.vstack([uv[i] for i in fs]); w, h = np.ptp(P, axis=0)
        along = 0 if w >= h else 1
        for i in fs:
            if unit == 'strip' and grid[i] is not None:
                key[i] = (int(c), int(round(grid[i][:, 1 - along].min())), along)
            else:
                key[i] = (int(c), 0, along)
    units = {k: n for n, k in enumerate(sorted(set(key)))}
    uid = np.array([units[k] for k in key]); band = np.zeros(nF, np.int64); err = np.zeros(nF); out = [None] * nF
    hgt = {}
    for k, n in units.items():
        P = np.vstack([uv[i] for i in np.flatnonzero(uid == n)]); hgt[k] = float(np.ptp(P[:, 1 - k[2]]))
    bh = [r1 - r0 for r0, r1 in rows]
    if size is None:                                 # the median strip fills the median band (the smallest one when
        size = float(np.median(list(hgt.values()))) / float(min(bh) if group else np.median(bh))   # grouping)
    if group and unit == 'strip':
        # adjacent strips of a patch join to fill taller bands: each patch's strips split into runs that minimise the
        # height-weighted density error |log2(run height / band height)| (dynamic programming, runs of up to 8)
        B = np.array(bh) * size; newkey = {}
        for c in set(k[0] for k in units):
            ks = sorted([k for k in units if k[0] == c], key=lambda k: k[1]); h = [hgt[k] for k in ks]; m = len(ks)
            cost = lambda i, j: sum(h[i:j]) * float(np.min(np.abs(np.log2(max(sum(h[i:j]), 1e-12) / B))))
            best = [0.0] + [math.inf] * m; cut = [0] * (m + 1)
            for j in range(1, m + 1):
                for i in range(max(0, j - 8), j):
                    if best[i] + cost(i, j) < best[j]:
                        best[j], cut[j] = best[i] + cost(i, j), i
            j, runs = m, []
            while j > 0:
                runs.append((cut[j], j)); j = cut[j]
            for g, (i, j) in enumerate(reversed(runs)):
                for k in ks[i:j]:
                    newkey[k] = (c, g, k[2])
        key = [newkey[k] for k in key]; units = {k: n for n, k in enumerate(sorted(set(key)))}
        uid = np.array([units[k] for k in key])
    for k, n in units.items():
        fs = np.flatnonzero(uid == n); along = k[2]
        R = np.eye(2) if along == 0 else np.array([[0.0, 1.0], [-1.0, 0.0]])     # a turn, never a mirror
        Q = [uv[i] @ R.T for i in fs]; P = np.vstack(Q); lo = P.min(0); h = max(float(np.ptp(P[:, 1])), 1e-12)
        r = min(range(len(rows)), key=lambda j: abs(math.log(h / ((rows[j][1] - rows[j][0]) * size))))
        sc = (rows[r][1] - rows[r][0]) / h; off = float(rng.random())
        for i, q in zip(fs, Q):
            out[i] = np.stack([(q[:, 0] - lo[0]) * sc + off, rows[r][0] + (q[:, 1] - lo[1]) * sc], 1)
            band[i] = r; err[i] = math.log2(sc * size)
    return out, uid, band, err


def angle_deficits(V, F):
    """Integrated Gaussian curvature per vertex: 2 pi minus its corner angles (pi minus them on the boundary)."""
    V = np.asarray(V, float); tot = np.zeros(len(V)); ef = _edge_faces(F)
    for f in F:
        n = len(f)
        for k in range(n):
            a, b, c = V[f[k - 1]], V[f[k]], V[f[(k + 1) % n]]; u, w = a - b, c - b
            tot[f[k]] += math.acos(max(-1.0, min(1.0, float(u @ w) / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-30))))
    bnd = {v for e, fs in ef.items() if len(fs) == 1 for v in e}
    return np.array([(math.pi if v in bnd else 2 * math.pi) - tot[v] for v in range(len(V))])


def merge_charts(V, F, chart, cuts=(), max_curv=0.5):
    """Merge neighbouring charts while the result stays a disc (Euler characteristic 1) whose total absolute
    curvature inside (sum of |angle deficit| of its interior vertices, radians) stays under max_curv, so it
    flattens with little distortion (Gauss-Bonnet).  Never across `cuts` (creases) or the boundary.  Cheapest
    merge first."""
    import heapq
    K = np.abs(angle_deficits(V, F)); ef = _edge_faces(F); nogo = {(min(a, b), max(a, b)) for a, b in cuts}
    lab = np.asarray(chart).copy(); members = collections.defaultdict(set)
    for i, c in enumerate(lab):
        members[c].add(i)
    vf = collections.defaultdict(set)
    for i, f in enumerate(F):
        for v in f:
            vf[v].add(i)
    bnd = {v for e, fs in ef.items() if len(fs) == 1 for v in e}

    def stats(fs):
        vs = {v for i in fs for v in F[i]}; es = {(min(F[i][k], F[i][(k + 1) % len(F[i])]), max(F[i][k], F[i][(k + 1) % len(F[i])]))
                                                 for i in fs for k in range(len(F[i]))}
        inner = [v for v in vs if v not in bnd and vf[v] <= fs]
        return len(vs) - len(es) + len(fs), float(K[inner].sum()) if inner else 0.0

    def pairs(c):
        out = set()
        for i in members[c]:
            f = F[i]
            for k in range(len(f)):
                e = (min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))
                if e in nogo:
                    continue
                for j, _ in ef[e]:
                    if lab[j] != c:
                        out.add(int(lab[j]))
        return out
    heap, ver = [], collections.Counter()

    def push(a, b):
        chi, k = stats(members[a] | members[b])
        if chi == 1 and k <= max_curv:
            heapq.heappush(heap, (k, a, b, ver[a], ver[b]))
    for c in list(members):
        for d in pairs(c):
            if c < d:
                push(c, d)
    while heap:
        k, a, b, va, vb = heapq.heappop(heap)
        if va != ver[a] or vb != ver[b] or a not in members or b not in members:
            continue
        members[a] |= members.pop(b)
        for i in members[a]:
            lab[i] = a
        ver[a] += 1
        for d in pairs(a):
            push(min(a, d), max(a, d))
    _, lab = np.unique(lab, return_inverse=True)
    return lab


# ---------------------------------------------------------------- shape-aware unwrapping of parts
def part_uvs(V, F, crease_deg=30.0):
    """UV per corner for every face: cylinders unwrapped around their axis (seam on the side facing the model's
    centre), their caps and everything else box-projected in the piece's own principal frame.  True size."""
    from .rigid_rig import face_geometry, segment, cylinder_fit
    V = np.asarray(V, float); N, A, C, T, tfi = face_geometry(V, F)
    seg = segment(V, F, N, crease_deg)
    # pieces that share no vertex stay apart even if segment joined them? (segment only joins across shared edges)
    centre = (C * A[:, None]).sum(0) / A.sum()
    uv = [None] * len(F); kind = {}
    for s in range(int(seg.max()) + 1):
        fs = np.flatnonzero(seg == s); vs = np.unique(np.concatenate([np.asarray(F[f]) for f in fs]))
        cy = cylinder_fit(V[vs], N[fs], A[fs], C[fs])
        if cy is not None:
            a = cy['axis']; c = cy['center']; d = centre - c; d = d - (d @ a) * a
            e1 = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
            e1 = e1 / np.linalg.norm(e1); e2 = np.cross(a, e1); r = cy['radius']; kind[s] = 'cylinder'
            for f in fs:
                P = V[F[f]] - c
                if abs(N[f] @ a) < 0.7:              # side: angle measured from the seam (facing the centre)
                    th = np.arctan2(P @ e2, P @ e1) % (2 * np.pi)
                    if th.max() - th.min() > np.pi:  # the face straddles the seam: keep it on one side
                        th[th < np.pi] += 2 * np.pi
                    uv[f] = np.stack([r * th, P @ a], 1)
                else:
                    sgn = 1.0 if N[f] @ a > 0 else -1.0
                    uv[f] = np.stack([P @ e1, sgn * (P @ e2)], 1)
            continue
        _, U = np.linalg.eigh((N[fs] * A[fs, None]).T @ N[fs]); kind[s] = 'box'
        for f in fs:                                 # box projection in the piece's frame, right-handed (no mirroring)
            k = int(np.argmax(np.abs(N[f] @ U))); nd = U[:, k] * (np.sign(N[f] @ U[:, k]) or 1.0)
            ax1 = U[:, (k + 1) % 3]; ax2 = np.cross(nd, ax1)
            P = V[F[f]]; uv[f] = np.stack([P @ ax1, P @ ax2], 1)
    return uv, seg, kind


# ---------------------------------------------------------------- measuring a layout
def uv_metrics(V, F, uv):
    """uv: per face (n, 2) corner UVs.  Area-weighted over fan triangles:
    angle   sigma1/sigma2 of the 3D->UV map: area-weighted median and 90th percentile, and % of area above 1.25
            (a mean is ruled by a few collapsed faces)
    density spread of texel density: area-weighted std of log2(scale) after removing its median (stops), and the
            5-95 % range ratio
    seams   % of edge length where the UVs part, islands, flipped area %
    grain   mean angle (deg) of UV edges of quads off their island's best axis pair (0 = loops run straight)."""
    V = np.asarray(V, float)
    ef = _edge_faces(F); par = list(range(len(F)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    seam_len = tot_len = 0.0
    for (a, b), fs in ef.items():
        L = float(np.linalg.norm(V[a] - V[b])); tot_len += L
        if len(fs) != 2:
            continue
        (i, ki), (j, kj) = fs; fi, fj = F[i], F[j]
        ua = {fi[k]: uv[i][k] for k in range(len(fi))}; ub = {fj[k]: uv[j][k] for k in range(len(fj))}
        if np.linalg.norm(ua[a] - ub[a]) > 1e-6 or np.linalg.norm(ua[b] - ub[b]) > 1e-6:
            seam_len += L
        else:
            par[find(i)] = find(j)
    island = np.array([find(i) for i in range(len(F))])
    ratios, scales, areas, flips = [], [], [], 0.0
    for i, f in enumerate(F):
        P, Q = V[f], np.asarray(uv[i], float)
        for k in range(1, len(f) - 1):
            e1, e2 = P[k] - P[0], P[k + 1] - P[0]; n = np.cross(e1, e2); a3 = np.linalg.norm(n) / 2
            if a3 < 1e-14:
                continue
            x = e1 / np.linalg.norm(e1); y = np.cross(n / (2 * a3), x)
            M = np.array([[e1 @ x, e2 @ x], [e1 @ y, e2 @ y]]); Qm = np.stack([Q[k] - Q[0], Q[k + 1] - Q[0]], 1)
            J = Qm @ np.linalg.inv(M); sv = np.linalg.svd(J, compute_uv=False); det = np.linalg.det(J)
            if det <= 0:
                flips += a3
            ratios.append(sv[0] / max(sv[1], 1e-12)); scales.append(math.sqrt(max(abs(det), 1e-30))); areas.append(a3)
    ratios, scales, areas = np.array(ratios), np.array(scales), np.array(areas); w = areas / areas.sum()
    ls = np.log2(scales); order = np.argsort(ls); cw = np.cumsum(w[order])
    med = ls[order][np.searchsorted(cw, 0.5)]; p5 = ls[order][np.searchsorted(cw, 0.05)]; p95 = ls[order][min(np.searchsorted(cw, 0.95), len(ls) - 1)]
    # grain: quads only
    ang = collections.defaultdict(list)
    for i, f in enumerate(F):
        if len(f) != 4:
            continue
        Q = np.asarray(uv[i], float); A3 = np.linalg.norm(np.cross(V[f[2]] - V[f[0]], V[f[3]] - V[f[1]])) / 2
        for k in range(4):
            d = Q[(k + 1) % 4] - Q[k]
            if np.linalg.norm(d) > 1e-12:
                ang[island[i]].append((math.atan2(d[1], d[0]), A3))
    dev = wsum = 0.0
    for isl, lst in ang.items():
        th = np.array([t for t, _ in lst]); wt = np.array([x for _, x in lst])
        phi = np.angle((wt * np.exp(4j * th)).sum()) / 4
        dlt = np.abs(((th - phi + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
        dev += float((np.degrees(dlt) * wt).sum()); wsum += float(wt.sum())
    o2 = np.argsort(ratios); c2 = np.cumsum(w[o2])
    q = lambda p: float(ratios[o2][min(np.searchsorted(c2, p), len(o2) - 1)])
    return dict(angle=q(0.5), angle_p90=q(0.9), angle_over_1_25=float(100 * w[ratios > 1.25].sum()),
                density_stops=float(np.sqrt((w * (ls - med) ** 2).sum())), density_5_95=float(2 ** (p95 - p5)),
                seams=float(100 * seam_len / tot_len), islands=int(len(set(island))), flipped=float(100 * flips / areas.sum()),
                grain_deg=float(dev / wsum) if wsum else float('nan'))
