# SPDX-License-Identifier: Apache-2.0
"""Resolution-aware feature classification for quad remeshing (prototype for idea 1).

A dihedral-angle threshold cannot tell a true crease from one facet of a tessellated fillet or from scan
noise: all three can show the same local angle.  The difference is *where the turning happens* at the
scale of the target quad size L:

  * crease  - the surface turns by Theta over a band of width ~r, and almost all of it at this one edge
  * blend   - Theta is spread over several parallel facet edges (fillet / multi-segment bevel)
  * noise   - large local angle but the band turns very little (bumps on a scan)

For every edge we compare the local dihedral angle theta with the band turn Theta measured between the
area-weighted mean normals of the faces on each side at distance r/2..r.  Concentration c = theta/Theta.
"""
import numpy as np
try:
    from .compat import BallIndex
except ImportError:
    from compat import BallIndex

CREASE, BLEND, NOISE, SMOOTH = 1, 2, 3, 0


def target_edge_length(mesh, target_quads):
    return float(np.sqrt(mesh.total_area / max(target_quads, 1)))


def band_turn(mesh, edges, r):
    """Band turn Theta (radians) for the given edge ids at radius r."""
    tree = BallIndex(mesh.C)
    out = np.zeros(len(edges))
    for i, e in enumerate(edges):
        f1, f2 = mesh.EF[e]
        m, d = mesh.emid[e], mesh.edir[e]
        n1, n2 = mesh.N[f1], mesh.N[f2]
        p1 = np.cross(n1, d); p1 *= np.sign(np.dot(p1, mesh.C[f1] - m)) or 1.0
        p2 = np.cross(n2, d); p2 *= np.sign(np.dot(p2, mesh.C[f2] - m)) or 1.0
        idx = np.asarray(tree.query_ball_point(m, r), int)
        if len(idx) == 0:
            out[i] = mesh.theta[e]; continue
        nj = mesh.N[idx]
        # drop faces from other sheets (e.g. the back of a thin plate): normal far from both sides
        keep = (nj @ n1 > -0.2) | (nj @ n2 > -0.2)
        idx, nj = idx[keep], nj[keep]
        rel = mesh.C[idx] - m
        s1, s2 = rel @ p1, rel @ p2
        far1 = (s1 >= s2) & (s1 > 0.4 * r)
        far2 = (s2 > s1) & (s2 > 0.4 * r)
        def mean_n(mask, fallback):
            if not mask.any():
                return fallback
            v = (nj[mask] * mesh.area[idx[mask]][:, None]).sum(0)
            ln = np.linalg.norm(v)
            return v / ln if ln > 1e-20 else fallback
        N1, N2 = mean_n(far1, n1), mean_n(far2, n2)
        out[i] = np.arccos(np.clip(N1 @ N2, -1, 1))
    return out


def classify(mesh, L, tau_deg=35.0, c_min=0.8, c_max=1.3, theta_min_deg=8.0, r_factor=0.75, min_chain=2.0,
             step_chain=3.0):
    """Return per-edge labels and the band turn. L = target quad edge length."""
    r = r_factor * L
    lab = np.full(len(mesh.E), SMOOTH, int)
    Theta = np.zeros(len(mesh.E))
    cand = np.where(mesh.interior & (mesh.theta >= np.radians(theta_min_deg)))[0]
    if len(cand):
        Th = band_turn(mesh, cand, r)
        Theta[cand] = Th
        th = mesh.theta[cand]
        c = th / np.maximum(Th, 1e-9)
        tau = np.radians(tau_deg)
        # crease: the local edge carries (almost) the whole band turn; c > 1 = step/groove edge pair
        is_crease = (th >= tau) & (c >= c_min) & (c <= c_max)
        is_step = (th >= tau) & (c > c_max)          # groove / step edge, or a noise spike
        # blend facet: the band turns a lot but this edge carries only part of it (fillet, multi-segment bevel)
        is_blend = (Th >= tau) & (c < c_min)
        lab[cand[is_blend]] = BLEND
        lab[cand[is_step]] = NOISE
        lab[cand[is_crease]] = CREASE
    if min_chain > 0:
        lab = _drop_short_chains(mesh, lab, min_chain * L)
        # steps/grooves are kept only as long chains (scan noise forms short, ragged chains)
        steps = np.where(lab == NOISE, CREASE, SMOOTH)
        steps = _drop_short_chains(mesh, steps, step_chain * L)
        lab[steps == CREASE] = CREASE
    return lab, Theta


def _drop_short_chains(mesh, lab, min_len):
    """Remove crease chains (connected sets of crease edges) whose total length is below min_len."""
    ids = np.where(lab == CREASE)[0]
    if len(ids) == 0:
        return lab
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    by_v = {}
    for e in ids:
        for v in mesh.E[e]:
            by_v.setdefault(int(v), []).append(e)
    for lst in by_v.values():
        for e in lst[1:]:
            ra, rb = find(lst[0]), find(e)
            if ra != rb:
                parent[ra] = rb
    tot = {}
    for e in ids:
        tot[find(e)] = tot.get(find(e), 0.0) + mesh.elen[e]
    out = lab.copy()
    for e in ids:
        if tot[find(e)] < min_len:
            out[e] = SMOOTH
    return out


def write_quadwild_sharp(mesh, edge_mask, path, include_boundary=True):
    """QuadWild .sharp format: count, then 'convex,face,local_edge' per feature edge."""
    lines = []
    for e in np.where(edge_mask & mesh.interior)[0]:
        f, k = mesh.EF[e, 0], mesh.EK[e, 0]
        lines.append(f"{1 if mesh.convex[e] else 0},{f},{k}")
    if include_boundary:
        for e in np.where(mesh.boundary)[0]:
            lines.append(f"1,{mesh.EF[e, 0]},{mesh.EK[e, 0]}")
    with open(path, 'w') as fh:
        fh.write(f"{len(lines)}\n" + "".join(s + "\n" for s in lines))
    return len(lines)


LOOP = 4   # facet line of a modelled bevel / tessellated fillet, kept as a quad loop after thinning


def _chains(mesh, ids, max_turn_deg=30.0):
    """Group edges into chains: neighbours share a vertex and run within max_turn_deg of each other."""
    ids = list(map(int, ids)); parent = {e: e for e in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    by_v = {}
    for e in ids:
        for v in mesh.E[e]:
            by_v.setdefault(int(v), []).append(e)
    ct = np.cos(np.radians(max_turn_deg))
    for lst in by_v.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                if abs(mesh.edir[a] @ mesh.edir[b]) >= ct:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
    groups = {}
    for e in ids:
        groups.setdefault(find(e), []).append(e)
    return list(groups.values())


def classify_v2(mesh, L, tau_deg=35.0, line_deg=8.0, contrast=4.0, spacing=0.6, min_chain=2.0, cv_max=0.35,
                min_flat=0.4, **kw):
    """v1 creases plus thinned facet lines.  Facet lines are high-contrast dihedral lines (a crease-like line
    in an otherwise flat neighbourhood) forming long chains of near-constant angle; they are kept with a
    minimum spacing of `spacing`*L so a bevel/fillet strip gets about width/(spacing*L) quad rows, boundary
    lines first.  Narrow strips (all lines closer than spacing*L) keep one line."""
    lab, Theta = classify(mesh, L, tau_deg=tau_deg, **kw)
    r = 0.75 * L
    th = np.degrees(mesh.theta)
    cand = np.where(mesh.interior & (th >= line_deg) & (lab != CREASE))[0]
    if len(cand) == 0:
        return lab, Theta
    allE = np.where(mesh.interior)[0]
    idx_all = BallIndex(mesh.emid[allE])
    med = np.zeros(len(cand)); flat = np.zeros(len(cand))
    for i, e in enumerate(cand):
        nb = th[allE[idx_all.query_ball_point(mesh.emid[e], r)]]
        med[i] = np.median(nb); flat[i] = np.mean(nb < 0.5)
    # a facet line sits in a piecewise-flat neighbourhood (designed / polyhedral surface); smooth organic
    # meshes and scans have almost no exactly flat edges, so their dihedral lines are not facet lines
    line = cand[(th[cand] >= contrast * np.maximum(med, 1.0)) & (flat >= min_flat)]
    chains = []
    for ch in _chains(mesh, line):
        ch = np.array(ch)
        if mesh.elen[ch].sum() < min_chain * L:
            continue
        t = th[ch]
        if t.std() / max(t.mean(), 1e-9) > cv_max:
            continue
        chains.append(ch)
    if not chains:
        return lab, Theta
    line_ids = np.concatenate(chains)
    li = BallIndex(mesh.emid[line_ids])
    def n_parallel(e):
        nb = line_ids[li.query_ball_point(mesh.emid[e], r)]
        return int((np.abs(mesh.edir[nb] @ mesh.edir[e]) > 0.94).sum())
    prio = [np.mean([n_parallel(e) for e in ch]) for ch in chains]          # boundary lines have fewer neighbours
    order = np.lexsort((-np.array([mesh.elen[ch].sum() for ch in chains]), np.array(prio)))
    kept = list(np.where(lab == CREASE)[0])
    for k in order:
        ch = chains[k]
        if kept:
            kt = BallIndex(mesh.emid[np.array(kept)])
            d, nn = kt.query(mesh.emid[ch])
            par = np.abs(np.einsum('ij,ij->i', mesh.edir[np.array(kept)][nn], mesh.edir[ch])) > 0.94
            dist = np.where(par, d, np.inf)
            if np.median(dist) < spacing * L:
                continue
        kept += list(ch); lab[ch] = LOOP
    return lab, Theta
