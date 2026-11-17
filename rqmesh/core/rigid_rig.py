# SPDX-License-Identifier: Apache-2.0
"""Rigging hard-surface models from their structure.

Robots, lamps, machines, doors and hatches are rigid parts that turn on pins, slide in sleeves or rails, or swivel on
balls.  `analyze` finds from the mesh alone the parts (separate pieces, and welded pieces split at concave creases), the
joints between them with their kind, pivot and axis, and a bone hierarchy.  Each part then gets one bone with 100 %
weights.  Joint evidence, strongest first:

  pin    a slender cylinder touching two parts: a hinge about the pin's axis
  ball   a sphere touching two parts, smaller than both: a ball joint at its centre
  slide  a rod in a sleeve (coaxial cylinders, the thinner one entering one end): a slide along the axis.  If the
         sleeve and the rod are each pinned to another part, a piston that follows those two parts
  axis   a cylinder whose axis runs through the contact (a turret on a drum): a hinge about that axis
  line   the parts touch along one straight line (a door's hinge edge): a hinge about it; along two parallel lines
         (a panel between rails): a slide along them

A contact without evidence fixes a small piece (a panel, a plate, a greeble) to the part it sits on; two large parts
touching without evidence get a free joint.  Helper faces (a cylinder the user adds where a hinge is hidden) count as
evidence but belong to no part.  Measured in RESULTS-round17 and -round18."""
import math
import collections
import numpy as np
from .compat import BallIndex


def face_geometry(V, F):
    """Unit normals, areas and vertex-mean centroids of polygons (Newell normals: fine for non-convex n-gons)."""
    tri, fi = [], []
    for i, f in enumerate(F):
        for k in range(1, len(f) - 1):
            tri.append((f[0], f[k], f[k + 1])); fi.append(i)
    T = np.asarray(tri, np.int64).reshape(-1, 3); fi = np.asarray(fi, np.int64)
    N = np.zeros((len(F), 3))
    np.add.at(N, fi, np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]))
    A = np.linalg.norm(N, axis=1) / 2
    lens = np.array([len(f) for f in F]); flat = np.concatenate([np.asarray(f, np.int64) for f in F]) if F else np.zeros(0, np.int64)
    C = np.zeros((len(F), 3)); np.add.at(C, np.repeat(np.arange(len(F)), lens), V[flat])
    return N / np.maximum(2 * A[:, None], 1e-30), A, C / lens[:, None], T, fi


def segment(V, F, N, crease_deg=30.0):
    """Faces grouped across every edge that is not a concave crease sharper than crease_deg (where welded parts meet).
    Convexity from the edge itself: with a->b in face i's winding, convex iff (Ni x Nj).(b-a) > 0."""
    ef = collections.defaultdict(list)
    for i, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; ef[(min(a, b), max(a, b))].append((i, a, b))
    par = list(range(len(F)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    cos_c = math.cos(math.radians(crease_deg))
    for fs in ef.values():
        if len(fs) != 2:
            continue
        (i, a, b), (j, _, _) = fs
        if float(N[i] @ N[j]) < cos_c and float(np.cross(N[i], N[j]) @ (V[b] - V[a])) < 0:
            continue
        par[find(i)] = find(j)
    return np.unique([find(i) for i in range(len(F))], return_inverse=True)[1]


def cylinder_fit(Vs, Ns, As, Cs):
    """A cylinder (pin, drum, rod, sleeve): for one principal direction of the face normals, nearly all area is side
    faces (normal perpendicular, spread around it) or caps, small bevels allowed, and the side faces sit at one radius
    (a bevelled box has 8 side directions but two radii)."""
    if len(Ns) < 8:
        return None
    _, U = np.linalg.eigh((Ns * As[:, None]).T @ Ns)
    for k in range(3):
        a = U[:, k]; d = np.abs(Ns @ a); side = d < 0.15; cap = d > 0.95
        if As[side | cap].sum() < 0.85 * As.sum() or As[side].sum() < 0.1 * As.sum():
            continue
        e1 = U[:, (k + 1) % 3]; e2 = np.cross(a, e1)
        if len(np.unique(np.round(np.degrees(np.arctan2(Ns[side] @ e2, Ns[side] @ e1)) / 10.0))) < 8:
            continue
        c0 = Cs[side].mean(0); X = Cs[side] - c0; rad = np.linalg.norm(X - np.outer(X @ a, a), axis=1)
        if rad.std() > 0.05 * rad.mean():
            continue
        t = Vs @ a; c = c0 + (0.5 * (t.min() + t.max()) - c0 @ a) * a
        return dict(axis=a, center=c, radius=float(rad.mean()), length=float(np.ptp(t)), span=(float(t.min()), float(t.max())))
    return None


def sphere_fit(Ns, As, Cs):
    """A sphere (ball joint): every face centroid lies one radius out along its normal from one centre, the faces
    point every way, and there are enough of them (a cube passes the first test, not the others)."""
    if len(Ns) < 20:
        return None
    w = np.sqrt(As / As.sum())
    M = np.zeros((3 * len(Ns), 4)); M[:, :3] = np.tile(np.eye(3), (len(Ns), 1)); M[:, 3] = Ns.ravel()
    sw = np.repeat(w, 3); x = np.linalg.lstsq(M * sw[:, None], Cs.ravel() * sw, rcond=None)[0]
    c, r = x[:3], x[3]
    if r <= 0:
        return None
    dist = np.linalg.norm(Cs - c, axis=1)
    if dist.std() > 0.05 * dist.mean() or np.linalg.norm((Ns * As[:, None]).sum(0)) > 0.35 * As.sum():
        return None
    _, U = np.linalg.eigh((Ns * As[:, None]).T @ Ns)
    if any(len(np.unique(np.round(np.abs(Ns @ U[:, k]) * 10))) < 4 for k in range(3)):
        return None                                      # a short cylinder: normals at only two tilts to its axis
    return dict(center=c, radius=float(r))


def _clusters_2d(Q, h):
    """Connected groups of 2D points on a grid of cell h (8-neighbourhood)."""
    key = np.floor(Q / h).astype(np.int64); cells = {}
    for i, k in enumerate(map(tuple, key)):
        cells.setdefault(k, []).append(i)
    par = {k: k for k in cells}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for (a, b) in cells:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                n = (a + da, b + db)
                if n in cells:
                    par[find((a, b))] = find(n)
    groups = collections.defaultdict(list)
    for k, idx in cells.items():
        groups[find(k)] += idx
    return [np.array(g) for g in groups.values()]


def line_evidence(P, tol):
    """Contact points along one straight line: a hinge about it; along two parallel lines: a slide along them."""
    if len(P) < 12:
        return None
    c = P.mean(0); X = P - c; _, _, Vt = np.linalg.svd(X, full_matrices=False)
    best = None
    for k in range(3):
        d = Vt[k]; along = X @ d
        if np.ptp(along) < 6 * tol:
            continue
        e1, e2 = Vt[(k + 1) % 3], Vt[(k + 2) % 3]; Q = np.stack([X @ e1, X @ e2], 1)
        groups = _clusters_2d(Q, tol)
        if len(groups) > 2 or any(np.ptp(Q[g], axis=0).max() > 5 * tol for g in groups):
            continue
        if len(groups) == 2 and any(np.ptp(along[g]) < 6 * tol for g in groups):
            continue
        if best is None or len(groups) < best[0]:
            q = Q[groups[0]].mean(0)
            gap = Q[groups[0]].mean(0) - Q[groups[-1]].mean(0)
            best = (len(groups), d, c + q[0] * e1 + q[1] * e2, gap)
    if best is None:
        return None
    if best[0] == 1:
        return dict(kind='hinge', axis=best[1], pivot=best[2])
    return dict(kind='slide', axis=best[1], pivot=c, sep=float(np.linalg.norm(best[3])))


def _samples(V, T, A, n, rng):
    """n points on the triangles, area-uniform, and the triangle of each."""
    if len(T) == 0:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    t = rng.choice(len(T), size=n, p=A / A.sum()) if A.sum() > 0 else rng.integers(len(T), size=n)
    r = rng.random((n, 2)); m = r.sum(1) > 1; r[m] = 1 - r[m]
    return V[T[t, 0]] + r[:, :1] * (V[T[t, 1]] - V[T[t, 0]]) + r[:, 1:] * (V[T[t, 2]] - V[T[t, 0]]), t


def _closest_on_line(p0, a, q0, b):
    """Point on line (p0, a) closest to line (q0, b)."""
    w = p0 - q0; aa, bb, ab = a @ a, b @ b, a @ b; den = aa * bb - ab * ab
    if abs(den) < 1e-12:
        return p0
    s = (ab * (b @ w) - bb * (a @ w)) / den
    return p0 + s * a


def analyze(V, F, helper=None, crease_deg=30.0, contact=0.01, detail=0.5, samples=40000, seed=0):
    """V (n,3) world coordinates; F polygons; helper: per-face flag of helper geometry.
    Returns dict: face_body (body of each face, -1 for helpers), bodies, root, parent {body: parent or None},
    joint {body: dict(kind=root|hinge|slide|ball|free|track, pivot, axis, evidence, ...)}, order, centroid {body},
    counts.  'track' bodies are the two halves of a piston: each turns about its pin to point at target_point, which
    moves with target_parent."""
    V = np.asarray(V, float); nF = len(F)
    helper = np.zeros(nF, bool) if helper is None else np.asarray(helper, bool)
    diag = float(np.linalg.norm(np.ptp(V, axis=0))) or 1.0; tol = contact * diag
    N, A, C, T, tfi = face_geometry(V, F)
    seg = segment(V, F, N, crease_deg); nseg = int(seg.max()) + 1 if nF else 0
    rng = np.random.default_rng(seed)
    faces = [np.flatnonzero(seg == s) for s in range(nseg)]
    tarea = np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1) / 2
    info, samp, snrm, idx, box = [], [], [], [], []
    total = A.sum() or 1.0
    for s in range(nseg):
        fs = faces[s]; vs = np.unique(np.concatenate([np.asarray(F[f]) for f in fs]))
        info.append(dict(area=float(A[fs].sum()), verts=vs, zmin=float(V[vs, 2].min()),
                         centroid=(C[fs] * A[fs, None]).sum(0) / max(A[fs].sum(), 1e-30),
                         cyl=cylinder_fit(V[vs], N[fs], A[fs], C[fs]), sph=sphere_fit(N[fs], A[fs], C[fs]),
                         helper=bool(helper[fs].all())))
        tm = np.isin(tfi, fs)
        n = int(np.clip(samples * A[fs].sum() / total, 400, 6000))
        P, t = _samples(V, T[tm], tarea[tm], n, rng); samp.append(P); idx.append(BallIndex(P))
        snrm.append(N[tfi[tm][t]] if len(t) else np.zeros((0, 3)))
        box.append((V[vs].min(0) - tol, V[vs].max(0) + tol))
    for x in info:                                       # a sphere is not a cylinder
        if x['sph'] is not None:
            x['cyl'] = None

    # contacts: surfaces closer than tol (a pin through a plate meets it along the cut)
    adj = collections.defaultdict(dict); cnt = {}; cpts = {}
    for i in range(nseg):
        for j in range(i + 1, nseg):
            if np.any(box[i][1] < box[j][0]) or np.any(box[j][1] < box[i][0]):
                continue
            di, ki = idx[j].query(samp[i], distance_upper_bound=tol); dj, kj = idx[i].query(samp[j], distance_upper_bound=tol)
            fi_, fj_ = np.isfinite(di), np.isfinite(dj)
            P = np.vstack([samp[i][fi_], samp[j][fj_]])
            if len(P) >= 3:
                # how much they touch: the area lying on the other surface (within a quarter of the gap of its tangent
                # plane at the nearest sample, which is exact for faces lying on each other however sparse the
                # samples), not the number of samples (density differs per piece; the gap band inflates thin contacts)
                pi_ = np.abs(np.einsum('ij,ij->i', samp[i][fi_] - samp[j][ki[fi_]], snrm[j][ki[fi_]])) < tol / 4
                pj_ = np.abs(np.einsum('ij,ij->i', samp[j][fj_] - samp[i][kj[fj_]], snrm[i][kj[fj_]])) < tol / 4
                area = 0.5 * (pi_.sum() / len(di) * info[i]['area'] + pj_.sum() / len(dj) * info[j]['area'])
                adj[i][j] = adj[j][i] = P.mean(0); cnt[(i, j)] = cnt[(j, i)] = area + 1e-12 * len(P)
                cpts[(i, j)] = cpts[(j, i)] = P

    # rods in sleeves: coaxial cylinders of different radius, the thinner one entering one end of the slender other
    slide = {}
    for i in range(nseg):
        for j in adj[i]:
            a, b = info[i]['cyl'], info[j]['cyl']
            if j <= i or a is None or b is None or abs(a['axis'] @ b['axis']) < math.cos(math.radians(3)):
                continue
            inner, outer = (i, j) if a['radius'] < b['radius'] else (j, i)
            ci, co = info[inner]['cyl'], info[outer]['cyl']; ax = co['axis']
            d = ci['center'] - co['center']
            if ci['radius'] > 0.9 * co['radius'] or np.linalg.norm(d - (d @ ax) * ax) > 0.3 * ci['radius']:
                continue
            if co['length'] < 1.5 * co['radius']:
                continue                                 # a wheel on an axle turns, it doesn't slide
            ti = info[inner]['verts']; s = V[ti] @ ax; t = V[info[outer]['verts']] @ ax
            ins = [(t.min() - tol < e < t.max() + tol) for e in (s.min(), s.max())]
            if ins[0] != ins[1]:
                slide[(inner, outer)] = ax
    in_slide = {x for p in slide for x in p}

    pins = {s for s in range(nseg) if info[s]['cyl'] is not None and s not in in_slide and len(adj[s]) >= 2
            and info[s]['cyl']['length'] >= 2.5 * info[s]['cyl']['radius']}
    balls = {s for s in range(nseg) if info[s]['sph'] is not None and len(adj[s]) >= 2
             and all(info[s]['area'] < info[j]['area'] for j in adj[s] if not info[j]['helper'])}
    knobs = {s for s in range(nseg) if (info[s]['cyl'] is not None or info[s]['sph'] is not None)
             and s not in pins | balls | in_slide and adj[s] and all(info[s]['area'] < 0.5 * info[j]['area'] for j in adj[s])}
    helpers = {s for s in range(nseg) if info[s]['helper']}
    bodies = [s for s in range(nseg) if s not in pins | balls | knobs | helpers]

    def axis_through(u, v, c):
        best = None
        for x in {u, v} | {k for k in pins | knobs if u in adj[k] or v in adj[k]}:
            cy = info[x]['cyl']
            if cy is None or (x in (u, v) and cy['length'] > 2 * cy['radius']):
                continue                                 # a turntable is a disc or a drum, not a post
            d = c - cy['center']; along = abs(float(d @ cy['axis'])); off = float(np.linalg.norm(d - (d @ cy['axis']) * cy['axis']))
            if along <= cy['length'] / 2 + max(cy['radius'], 0.05 * diag) and off <= max(1.5 * cy['radius'], 0.02 * diag):
                if best is None or off < best[0]:
                    best = (off, cy)
        return None if best is None else best[1]

    def evidence(u, v, c, P, slides):
        for (ri, ro), ax in slides.items():
            if {ri, ro} == {u, v}:
                return dict(kind='slide', axis=ax, pivot=c, evidence='slide')
        cy = axis_through(u, v, c)
        if cy is not None:
            return dict(kind='hinge', axis=cy['axis'], pivot=cy['center'] + ((c - cy['center']) @ cy['axis']) * cy['axis'],
                        evidence='axis')
        le = line_evidence(P, tol) if P is not None else None
        if le is None:
            return None
        if le['kind'] == 'hinge':                        # both parts must reach well away from the line: a door does,
            for x in (u, v):                             # a rail glued flat on a wall doesn't
                X = samp[x] - le['pivot']
                if np.linalg.norm(X - np.outer(X @ le['axis'], le['axis']), axis=1).max() < 0.1 * diag:
                    return None
            le['evidence'] = 'line'
        else:
            if le['sep'] < 0.1 * diag:
                return None
            le['evidence'] = 'rails'
        return le

    # 1. details: no evidence at the contact, under `detail` times that neighbour's size -> fixed to it
    grp = {b: b for b in bodies}

    def gfind(x):
        while grp[x] != x:
            grp[x] = grp[grp[x]]; x = grp[x]
        return x
    lowest = min(bodies, key=lambda s: (round(info[s]['zmin'] / max(tol, 1e-12)), -info[s]['area'])) if bodies else None
    for i in sorted(bodies, key=lambda s: info[s]['area']):
        if i == lowest:
            continue
        nb = [j for j, c in adj[i].items() if j in grp and info[i]['area'] < detail * info[j]['area']
              and evidence(i, j, c, cpts[(i, j)], slide) is None]
        if nb:
            grp[gfind(i)] = gfind(max(nb, key=lambda j: cnt[(i, j)]))
    rep = {b: gfind(b) for b in bodies}
    R = lambda x: rep.get(x, x)
    members = collections.defaultdict(list)
    for b in bodies:
        members[rep[b]].append(b)
    for r_, ms in members.items():
        if len(ms) > 1:
            info[r_] = dict(info[r_], area=sum(info[m]['area'] for m in ms), cyl=None, sph=None,
                            zmin=min(info[m]['zmin'] for m in ms),
                            centroid=sum(info[m]['centroid'] * info[m]['area'] for m in ms) / sum(info[m]['area'] for m in ms))
            samp[r_] = np.vstack([samp[m] for m in ms]); snrm[r_] = np.vstack([snrm[m] for m in ms])
    madj = collections.defaultdict(dict); mcnt = {}; mpts = collections.defaultdict(list)
    for (i, j), k in cnt.items():
        a_, b_ = R(i), R(j)
        if a_ == b_:
            continue
        if mcnt.get((a_, b_), 0) < k:
            madj[a_][b_] = adj[i][j]
        mcnt[(a_, b_)] = mcnt.get((a_, b_), 0) + k; mpts[(a_, b_)].append(cpts[(i, j)])
    mpts = {k: np.vstack(v) for k, v in mpts.items()}
    for k in mpts:                                       # contact centroid of the merged pair
        madj[k[0]][k[1]] = mpts[k].mean(0)
    adj, cnt = madj, mcnt
    bodies = sorted(members)
    mslide = {(R(i), R(o)): ax for (i, o), ax in slide.items() if R(i) != R(o) and R(i) in members and R(o) in members}

    # 2. joint graph: pins and balls join the two bodies they touch most; direct contacts
    edges = collections.defaultdict(dict)
    for p in sorted(pins | balls):
        nb = sorted([j for j in adj[p] if j in members], key=lambda j: -cnt[(p, j)])[:2]
        if len(nb) < 2:
            continue
        u, v = nb
        if p in balls:
            e = dict(kind='ball', pivot=info[p]['sph']['center'], axis=None, evidence='ball', via=p)
        else:
            cy = info[p]['cyl']; c = adj[u].get(v)
            piv = cy['center'] if c is None else cy['center'] + ((c - cy['center']) @ cy['axis']) * cy['axis']
            e = dict(kind='hinge', pivot=piv, axis=cy['axis'], evidence='pin', via=p)
        edges[u][v] = edges[v][u] = e
    for u in bodies:
        for v, c in adj[u].items():
            if v in members and v not in edges[u]:
                ev = evidence(u, v, c, mpts.get((u, v)), mslide)
                edges[u][v] = edges[v][u] = ev or dict(kind='free', pivot=c, axis=None, evidence='none')

    # pistons: a slide pair whose sleeve and rod are each pinned to another, different body
    pistons = []
    for (ri, ro), ax in mslide.items():
        X = [w for w, e in edges[ro].items() if e['evidence'] == 'pin' and w not in (ri, ro)]
        Y = [w for w, e in edges[ri].items() if e['evidence'] == 'pin' and w not in (ri, ro)]
        pairs = [(x, y) for x in X for y in Y if x != y]
        if pairs:
            x, y = pairs[0]; ex, ey = edges[ro][x], edges[ri][y]
            c0 = info[ro]['cyl']['center'] if info[ro]['cyl'] is not None else adj[ro][ri]
            pa = _closest_on_line(c0, ax, ex['pivot'], ex['axis']); pb = _closest_on_line(c0, ax, ey['pivot'], ey['axis'])
            pistons.append(dict(sleeve=ro, rod=ri, sleeve_parent=x, rod_parent=y, a=pa, b=pb))
    in_piston = {p['sleeve'] for p in pistons} | {p['rod'] for p in pistons}

    # 3. hierarchy: breadth first from the lowest (then largest) body, strongest evidence first
    rank_ev = {'pin': 0, 'ball': 0, 'slide': 1, 'axis': 1, 'line': 2, 'rails': 2, 'none': 3}
    tree = [b for b in bodies if b not in in_piston]
    root = min(tree, key=lambda s: (round(info[s]['zmin'] / max(tol, 1e-12)), -info[s]['area'])) if tree else None
    parent, joint, order = {}, {}, []
    if root is not None:
        parent[root] = None; order.append(root)
        joint[root] = dict(kind='root', pivot=np.array([info[root]['centroid'][0], info[root]['centroid'][1], info[root]['zmin']]),
                           axis=None, evidence='root')
        q = collections.deque([root])
        while q:
            u = q.popleft()
            for v, e in sorted(edges[u].items(), key=lambda kv: (rank_ev[kv[1]['evidence']], -cnt.get((u, kv[0]), 0))):
                if v in parent or v in in_piston:
                    continue
                parent[v] = u; order.append(v); q.append(v); joint[v] = dict(e)
        while len(parent) < len(tree):                   # parts that touch nothing reached: hang from the nearest one
            best = None
            for v in tree:
                if v in parent:
                    continue
                iv = BallIndex(samp[v])
                for u in order:
                    d, k = iv.query(samp[u]); i = int(np.argmin(d))
                    if best is None or d[i] < best[0]:
                        best = (d[i], u, v, (samp[u][i] + samp[v][k[i]]) / 2)
            _, u, v, c = best
            parent[v] = u; order.append(v); joint[v] = dict(kind='free', pivot=c, axis=None, evidence='stray')
    for p in pistons:
        u = (p['b'] - p['a']) / max(np.linalg.norm(p['b'] - p['a']), 1e-30)
        parent[p['sleeve']] = p['sleeve_parent']; parent[p['rod']] = p['rod_parent']
        joint[p['sleeve']] = dict(kind='track', pivot=p['a'], axis=u, target_point=p['b'], target_parent=p['rod_parent'],
                                  evidence='piston', partner=p['rod'])
        joint[p['rod']] = dict(kind='track', pivot=p['b'], axis=-u, target_point=p['a'], target_parent=p['sleeve_parent'],
                               evidence='piston', partner=p['sleeve'])
        order += [p['sleeve'], p['rod']]
    rank = {b: k for k, b in enumerate(order)}
    owner = {b: b for b in bodies}
    for s in range(nseg):                                # pins, balls, knobs: with their parent-most neighbour
        if s in owner or s in helpers:
            continue
        if s in rep:
            owner[s] = rep[s]; continue
        nb = [R(j) for j in adj.get(s, {})] or [R(j) for j in range(nseg) if (s, j) in cnt]
        nb = [j for j in nb if j in rank]
        owner[s] = min(nb, key=rank.get) if nb else root
    for s in list(owner):
        if s in rep:
            owner[s] = rep[s]
    face_body = np.array([owner.get(int(s), -1) if not info[int(s)]['helper'] else -1 for s in seg], np.int64)
    kinds = collections.Counter(j['kind'] for b, j in joint.items() if b != root)
    return dict(face_body=face_body, bodies=order, root=root, parent=parent, joint=joint, order=order,
                centroid={b: info[b]['centroid'] for b in order}, area={b: info[b]['area'] for b in order},
                counts=dict(pieces=nseg, bodies=len(order), pins=len(pins), spheres=len(balls), knobs=len(knobs),
                            rods_in_sleeves=len(slide), pistons=len(pistons), helpers=len(helpers),
                            merged=sum(len(m) - 1 for m in members.values())),
                joints=dict(kinds),
                diag=diag, seg=seg)
