# SPDX-License-Identifier: Apache-2.0
"""Surface masks for shading (round 19b): where paint wears off, where dirt collects, and a smooth colour variation
that follows the model's structure.  Per vertex, 0..1, for materials (Attribute node) and Geometry Nodes.

wear / cavity   signed mean curvature from the dihedral angle of every edge (convex where (Ni x Nj).(b-a) > 0 with
                a->b in face i's winding), lightly smoothed; the most convex `amount` of the surface (by area) ramps
                to 1 (wear), the most concave likewise (cavity).  Set by area share, so it doesn't depend on the
                mesh's resolution or size.  Works in EEVEE too (Cycles' Pointiness is Cycles-only).
variance        on a subdivided mesh: midpoint-displacement noise through the recovered levels: random at the cage's
                vertices, then each new edge and face point gets the mean of its parents plus a random kick that
                shrinks by `falloff` per level.  Smooth, continuous, and aligned with the cage's panels and loops.
                On any other mesh: random values diffused over the surface."""
import math
import collections
import numpy as np


def _vertex_area(V, F):
    a = np.zeros(len(V))
    for f in F:
        P = V[f]; n = np.zeros(3)
        for k in range(1, len(f) - 1):
            n += np.cross(P[k] - P[0], P[k + 1] - P[0])
        a[f] += np.linalg.norm(n) / 2 / len(f)
    return np.maximum(a, 1e-30)


def _neighbours(F, nV):
    nb = [set() for _ in range(nV)]
    for f in F:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; nb[a].add(b); nb[b].add(a)
    return [np.fromiter(s, np.int64) if s else np.zeros(0, np.int64) for s in nb]


def _smooth(x, nb, iters):
    for _ in range(iters):
        x = np.array([0.5 * x[i] + 0.5 * x[n].mean() if len(n) else x[i] for i, n in enumerate(nb)])
    return x


def curvature(V, F, smooth=2):
    """Signed mean curvature per vertex (convex > 0), from edge dihedral angles, times the model's size."""
    V = np.asarray(V, float); ef = collections.defaultdict(list)
    for i, f in enumerate(F):
        for k in range(len(f)):
            ef[(min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))].append((i, f[k], f[(k + 1) % len(f)]))
    N = np.zeros((len(F), 3))
    for i, f in enumerate(F):
        P = V[f]
        for k in range(1, len(f) - 1):
            N[i] += np.cross(P[k] - P[0], P[k + 1] - P[0])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-30)
    H = np.zeros(len(V))
    for e, fs in ef.items():
        if len(fs) != 2:
            continue
        (i, a, b), (j, _, _) = fs
        th = math.acos(max(-1.0, min(1.0, float(N[i] @ N[j])))); L = float(np.linalg.norm(V[b] - V[a]))
        sgn = 1.0 if float(np.cross(N[i], N[j]) @ (V[b] - V[a])) > 0 else -1.0
        H[a] += sgn * th * L / 2; H[b] += sgn * th * L / 2
    H = H / (2 * _vertex_area(V, F)) * float(np.linalg.norm(np.ptp(V, axis=0)))
    return _smooth(H, _neighbours(F, len(V)), smooth)


def _ramp_top(x, area, amount):
    """0..1: the top `amount` share of the area (by x) ramps up from the share 2*amount."""
    o = np.argsort(-x); cw = np.cumsum(area[o]) / area.sum()
    hi = x[o][min(np.searchsorted(cw, amount * 0.5), len(o) - 1)]; lo = x[o][min(np.searchsorted(cw, amount * 2), len(o) - 1)]
    lo = max(lo, 0.0)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)


def wear_cavity(V, F, wear=0.08, cavity=0.08, smooth=2):
    V = np.asarray(V, float); H = curvature(V, F, smooth); A = _vertex_area(V, F)
    return _ramp_top(H, A, wear), _ramp_top(-H, A, cavity)


def structure_noise(trace, nV, falloff=0.35, seed=0):
    """Midpoint-displacement noise through the recovered subdivision levels (see the module note).  0..1."""
    rng = np.random.default_rng(seed)
    nVc = 1 + max(max(f) for f in trace[-1]['F'])
    val = dict(enumerate(rng.random(nVc))); amp = 0.5
    for lev in reversed(trace):
        Fc, Ff, cfm, vmap = lev['F'], lev['fineF'], lev['corner_face'], lev['vmap']
        kick = rng.random(1 + max(max(f) for f in Ff)) - 0.5
        new = {vmap[i]: x for i, x in val.items()}
        for b, fb in enumerate(Fc):
            n = len(fb); qs = cfm[b]; common = set(Ff[qs[0]])
            for q in qs[1:]:
                common &= set(Ff[q])
            fpt = next(iter(common))
            new.setdefault(fpt, float(np.mean([val[v] for v in fb])) + amp * kick[fpt])
            for k in range(n):
                en = next(iter((set(Ff[qs[k]]) & set(Ff[qs[(k + 1) % n]])) - {fpt}))
                new.setdefault(en, (val[fb[k]] + val[fb[(k + 1) % n]]) / 2 + amp * kick[en])
        val = new; amp *= falloff
    x = np.array([val.get(v, 0.5) for v in range(nV)])
    return (x - x.min()) / max(np.ptp(x), 1e-12)


def diffusion_noise(V, F, iters=25, seed=0):
    """Random per vertex, diffused over the surface, rescaled to 0..1 (for meshes without a cage)."""
    nb = _neighbours(F, len(V)); x = _smooth(np.random.default_rng(seed).random(len(V)), nb, iters)
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)


def vertex_normals(V, F):
    n = np.zeros((len(V), 3))
    for f in F:
        P = V[f]; fn = np.zeros(3)
        for k in range(1, len(f) - 1):
            fn += np.cross(P[k] - P[0], P[k + 1] - P[0])
        n[f] += fn
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def exposure(V, F, radius=0.15, res=96, rays=48, seed=0):
    """How open each vertex is to its surroundings, 0..1 (ambient occlusion: the cosine-weighted share of rays that
    leave within `radius` x the model's size without hitting it).  Rays march through a voxel grid of the solid:
    surface voxels, plus the inside where the surface is closed (flood fill from outside).  NumPy only."""
    V = np.asarray(V, float); lo = V.min(0); ext = float(np.ptp(V, axis=0).max()) or 1.0; h = ext / res
    pad = int(math.ceil(radius * ext / h)) + 3; org = lo - pad * h
    dims = np.ceil(np.ptp(V, axis=0) / h).astype(int) + 2 * pad + 1
    occ = np.zeros(dims, bool); rng = np.random.default_rng(seed)
    for f in F:                                      # surface voxels: points on every triangle, spaced under h/2
        for k in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[k]], V[f[k + 1]]
            n = int(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / (0.4 * h)) + 2
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1)); m = i + j <= n
            P = a + np.outer(i[m] / n, b - a) + np.outer(j[m] / n, c - a)
            ix = np.floor((P - org) / h).astype(int); occ[ix[:, 0], ix[:, 1], ix[:, 2]] = True
    out = np.zeros(dims, bool)
    out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1] = True, True, True, True, True, True
    out &= ~occ
    while True:                                      # flood fill the outside (6-neighbours) through empty voxels
        g = out.copy()
        g[1:] |= out[:-1]; g[:-1] |= out[1:]; g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        g[:, :, 1:] |= out[:, :, :-1]; g[:, :, :-1] |= out[:, :, 1:]; g &= ~occ
        if (g == out).all():
            break
        out = g
    solid = ~out; N = vertex_normals(V, F)
    u = rng.random((rays, 2)); r = np.sqrt(u[:, 0]); t = 2 * np.pi * u[:, 1]     # cosine-weighted hemisphere
    D = np.stack([r * np.cos(t), r * np.sin(t), np.sqrt(1 - u[:, 0])], 1)
    steps = int(math.ceil(radius * ext / (0.5 * h))); res_ = np.zeros(len(V))
    for s0 in range(0, len(V), 2048):
        n = N[s0:s0 + 2048]; p = V[s0:s0 + 2048] + n * 1.5 * h
        a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(n, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(n, e1)
        W = D[None, :, 0:1] * e1[:, None] + D[None, :, 1:2] * e2[:, None] + D[None, :, 2:3] * n[:, None]
        free = np.ones(W.shape[:2], bool)
        for st in range(1, steps + 1):
            q = np.clip(np.floor((p[:, None] + W * (st * 0.5 * h) - org) / h).astype(int), 0, dims - 1)
            free &= ~solid[q[..., 0], q[..., 1], q[..., 2]]
        res_[s0:s0 + 2048] = free.mean(1)
    return res_
