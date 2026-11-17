# SPDX-License-Identifier: Apache-2.0
"""Partial reversal of subdivision: peel the parts of a mesh that are still a subdivision grid, keep the rest.

After modelling on a subdivided mesh with edits that change its topology (extrude, inset, loop cut, knife, bevel,
boolean), the whole mesh is no longer a subdivision and `subdiv.unsubdivide` finds nothing.  Most of it still is:
every quad of an untouched region has the corner pattern (V, E, F, E) -- cage vertex, edge point, face point, edge
point.  Here the labels are grown face by face from a seed quad.  Quads where the labels clash, quads touched by the
new geometry and every non-quad face are kept as they are; each complete fan of good quads around a face point is
peeled back to its cage face.  Where a peeled face meets kept geometry, the edge point stays as an extra vertex on
the peeled face's edge, so the result is a closed, consistently oriented polygon mesh with no T-junctions.

Positions: for simple (linear) subdivision the cage vertices are where the grid has them.  For Catmull-Clark they are
fitted by least squares on the rows whose whole stencil lies in the peeled region; cage vertices that are shared with
kept geometry keep their position, so the kept part is not moved.  Which reading of an ambiguous grid to peel (a
mesh can usually be read as its cage or as a "dual") is chosen as in `subdiv`: an exact fit decides, otherwise fewer
n-gons, unless that reading cannot pass as a subdivision while another can.
"""
import numpy as np
from collections import defaultdict, deque
from .subdiv import subdiv_operator, Coo, cgls, LEVEL_TOL

PAT = ('V', 'E', 'F', 'E')


def _rings(F):
    vf = defaultdict(list); ef = defaultdict(list)
    for i, f in enumerate(F):
        n = len(f)
        for k in range(n):
            vf[f[k]].append(i)
            a, b = f[k], f[(k + 1) % n]; ef[(min(a, b), max(a, b))].append(i)
    return vf, ef


def _grow(F, seed, rot, ef, deg=None, bnd=frozenset()):
    """Labels from one seed quad and pattern rotation; returns (labels per vertex, set of good faces).
    Quads whose edge-point corners have the valence of an edge point (4, or 3 on the boundary) are grown first; the
    others (next to an edit) get their labels from regular neighbours and pass them on only after the clean grid is
    labelled, so an edit cannot push wrong labels into the grid around it."""
    def regular(G, r):
        return deg is None or all(deg[G[j]] == (3 if G[j] in bnd else 4) for j in range(4) if PAT[(j + r) % 4] == 'E')
    lab = {}; good = set(); bad = set()
    f = F[seed]
    for k in range(4): lab[f[k]] = PAT[(k + rot) % 4]
    good.add(seed); q_reg, q_late = deque([seed]), deque()
    while q_reg or q_late:
        fi = q_reg.popleft() if q_reg else q_late.popleft(); f = F[fi]
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]
            for g in ef[(min(a, b), max(a, b))]:
                if g == fi or g in good or g in bad or len(F[g]) != 4: continue
                G = F[g]; ia = G.index(a)
                la = lab[a]
                if la == 'V': r = (0 - ia) % 4
                elif la == 'F': r = (2 - ia) % 4
                else:
                    ib = G.index(b); lb = lab[b]
                    want = 0 if lb == 'V' else 2                  # position of b's label in PAT
                    r = (want - ib) % 4
                    if PAT[(ia + r) % 4] != 'E': bad.add(g); continue
                ok = all(lab.get(G[j], PAT[(j + r) % 4]) == PAT[(j + r) % 4] for j in range(4))
                if not ok: bad.add(g); continue
                for j in range(4): lab.setdefault(G[j], PAT[(j + r) % 4])
                good.add(g); (q_reg if regular(G, r) else q_late).append(g)
    return lab, good


def _degrees(ef):
    deg = defaultdict(int); bnd = set()
    for (a, b), fs in ef.items():
        deg[a] += 1; deg[b] += 1
        if len(fs) == 1: bnd.update((a, b))
    return deg, bnd


def _fans(F, lab, good, vf, ef):
    """Complete fans of good quads around interior F-labelled vertices: {fp: ring of (V, E, face)}."""
    deg, bnd = _degrees(ef)
    fans = {}
    for p, l in lab.items():
        if l != 'F': continue
        fs = vf[p]
        if len(fs) < 3 or any(g not in good for g in fs): continue
        step = {}
        for g in fs:
            G = F[g]; k = G.index(p)
            if lab[G[(k + 2) % 4]] != 'V': step = None; break
            step[G[(k + 1) % 4]] = (G[(k + 2) % 4], G[(k + 3) % 4], g)     # e_in -> (v, e_out, face)
        if not step or len(step) != len(fs): continue
        e0 = next(iter(step)); e = e0; ring = []; ok = True
        for _ in range(len(fs)):
            if e not in step: ok = False; break
            v, en, g = step[e]; ring.append((v, en, g)); e = en
        if not ok or e != e0 or len({v for v, _, _ in ring}) != len(ring): continue
        if any(len(ef[(min(p, x), max(p, x))]) != 2 for x in step): continue          # p interior
        fans[p] = ring
    # an edge point that disappears (every face around it peeled) must have the valence of an edge point; one that
    # stays next to an edit may have any.  Removing a fan can only relax the test for its neighbours, so repeat.
    while True:
        in_fan = {g for r in fans.values() for _, _, g in r}
        bad = [p for p, r in fans.items()
               if any(all(g in in_fan for g in vf[e]) and deg[e] != (3 if e in bnd else 4) for _, e, _ in r)]
        if not bad: return fans
        for p in bad: del fans[p]


def _peel(V, F, lab, fans, vf, ef, scheme, iters=400):
    """Build the mixed mesh: one cage face per fan, every other face kept; returns (Vout, Fout, info)."""
    in_fan = {}
    for p, ring in fans.items():
        for _, _, g in ring: in_fan[g] = p
    kept = [i for i in range(len(F)) if i not in in_fan]
    # an edge point can go when every face around it is peeled
    drop_e = set()
    for p, ring in fans.items():
        for _, e, _ in ring:
            if all(g in in_fan for g in vf[e]): drop_e.add(e)
    base, src = [], []                          # cage faces as lists of input vertex ids (V points, kept E points);
    for p, ring in fans.items():                # src: the input face corner each output corner comes from
        face, fs = [], []
        for v, e, g in ring:
            k = F[g].index(p)
            face.append(v); fs.append((g, (k + 2) % 4))
            if e not in drop_e: face.append(e); fs.append((g, (k + 3) % 4))
        base.append(face); src.append(fs)
    Fout_in = base + [list(F[i]) for i in kept]
    src += [[(i, k) for k in range(len(F[i]))] for i in kept]
    used = sorted({v for f in Fout_in for v in f})
    # positions, fitted per scheme group (scheme: one name, or {face point: name} for pieces that differ)
    P = V.copy()
    info = dict(scheme=scheme if isinstance(scheme, str) else '+'.join(sorted(set(scheme.values()))),
                residual_rel=0.0, exact_fans=0)
    groups = defaultdict(dict)
    for p, ring in fans.items():
        groups[scheme if isinstance(scheme, str) else scheme[p]][p] = ring
    mel = np.mean([np.linalg.norm(V[a] - V[b]) for (a, b) in ef if a in lab and b in lab]) if ef else 1.0
    sq, nrow = 0.0, 0
    info['fan_res'] = {}
    for sc, gf in groups.items():
        Pb, pv, d, w, fan_res = _fit_group(V, gf, sc, vf, in_fan, drop_e, max(mel, 1e-30), iters)
        sq += float((d[w[:, 0] > 0] ** 2).sum()); nrow += int((w[:, 0] > 0).sum())
        info['exact_fans'] += sum(r < 1e-4 for r in fan_res.values()); info['fan_res'].update(fan_res)
        for i, v in enumerate(pv):
            if all(g in in_fan for g in vf[v]):          # shared with kept geometry: stays put
                P[v] = Pb[i]
    info['residual_rel'] = float(np.sqrt(sq / nrow)) if nrow else 0.0
    remap = {v: i for i, v in enumerate(used)}
    Fout = [[remap[v] for v in f] for f in Fout_in]
    info['valid'] = _valid(Fout, F)
    info.update(peeled=len(fans), kept=len(kept), stitch_points=sum(len(f) for f in base) - sum(len(r) for r in fans.values()))
    info['corner_src'] = src
    return P[used], Fout, used, len(base), info


def _valid(Fout, Fin):
    """The result must be a manifold, consistently oriented mesh with no more open edges than the input."""
    def edges(FF):
        und, dirs = defaultdict(int), defaultdict(int)
        for f in FF:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                if a == b: return None, None
                und[(min(a, b), max(a, b))] += 1; dirs[(a, b)] += 1
        return und, dirs
    u, d = edges(Fout)
    if u is None or any(c > 2 for c in u.values()) or any(c > 1 for c in d.values()): return False
    if any(len(set(f)) != len(f) or len(f) < 3 for f in Fout): return False
    u_in, _ = edges(Fin)
    return sum(c == 1 for c in u.values()) <= sum(c == 1 for c in (u_in or {}).values())


def _face_components(F):
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for f in F:
        r0 = find(f[0])
        for v in f[1:]:
            r = find(v)
            if r != r0: par[r] = r0
    comp = defaultdict(list)
    for i, f in enumerate(F): comp[find(f[0])].append(i)
    return list(comp.values())


def _choose_reading(V, F, quads, vf, ef, deg, bnd, schemes, seeds):
    """Grow labels from a few seeds and all four pattern rotations in one connected component and pick a reading:
    a whole exact fit first; then the reading with the most exactly fitting fans, when that is most of its fans
    (away from the edit the right reading fits exactly, a wrong one almost nowhere); otherwise fewer n-gons, unless
    that reading could not pass as a subdivision while another could (as in `subdiv`).  Returns (labels, fans,
    scheme, residual) or None."""
    cand_seeds = [quads[int(k * (len(quads) - 1) / max(1, seeds - 1))] for k in range(seeds)] if seeds > 1 else [quads[0]]
    uniq = {}
    for s_ in dict.fromkeys(cand_seeds):
        for rot in range(4):
            lab, good = _grow(F, s_, rot, ef, deg, bnd)
            fans = _fans(F, lab, good, vf, ef)
            if fans: uniq.setdefault(frozenset(fans), (lab, fans))
    if not uniq: return None
    top = max(len(f) for _, f in uniq.values())
    trials = []
    for lab, fans in uniq.values():
        if len(fans) < 0.5 * top: continue
        for sc in schemes:
            info = _peel(V, F, lab, fans, vf, ef, sc)[4]
            if info['valid']: trials.append((info['residual_rel'], sc, lab, fans, info))
    if not trials: return None
    ex = [t for t in trials if t[0] < 1e-4]
    if ex:
        t = min(ex, key=lambda t: (_ngon_excess(t[3]), -len(t[3]), round(t[0], 6), schemes.index(t[1])))
    else:
        t = max(trials, key=lambda t: (t[4]['exact_fans'], -t[0]))
        if not (t[4]['exact_fans'] > 0 and t[4]['exact_fans'] >= 0.5 * len(t[3])):
            t = min(trials, key=lambda t: (_ngon_excess(t[3]), t[0]))
            best = min(trials, key=lambda t: t[0])
            if t[0] > 2 * LEVEL_TOL and best[0] <= LEVEL_TOL: t = best
    # the scheme for that reading: an exact fit (simple first); else the best fit that could pass as a subdivision;
    # else simple, which leaves the cage vertices where they are (a Catmull-Clark fit that doesn't pass can inflate
    # the cage around sharp edges: a bent, gridded panel fitted CC at 0.22 and moved by 30% of its size)
    same = sorted([x for x in trials if x[3] is t[3]], key=lambda x: (round(x[0], 6), schemes.index(x[1])))
    exact = [x for x in same if x[0] < 1e-4]
    passing = [x for x in same if x[0] <= 2 * LEVEL_TOL]
    t = exact[0] if exact else passing[0] if passing else next((x for x in same if x[1] == 'simple'), same[0])
    fans = t[3]
    # in a reading backed by exactly fitting fans (no edit spread over the surface), a fan that fits badly is a
    # local pattern next to an edit that only looks like a grid (along a loop cut, say): keep it as it is
    fr = t[4]['fan_res']
    if t[4]['exact_fans'] >= 0.5 * len(fans):
        fans = {p: r for p, r in fans.items() if fr.get(p, 0.0) <= 2 * LEVEL_TOL}
    return (t[2], fans, t[1], t[0]) if fans else None


def _fit_group(V, fans, scheme, vf, in_fan, drop_e, mel, iters):
    """Least-squares cage positions for a set of fans under one scheme, using only the rows whose whole stencil
    lies in the peeled region.  Returns (positions, base vertex ids, row residuals, row weights, exact fans)."""
    pv = sorted({v for ring in fans.values() for v, _, _ in ring})
    loc = {v: i for i, v in enumerate(pv)}
    blf = [[loc[v] for v, _, _ in ring] for ring in fans.values()]
    Ev, Fv, Vv, rows_ok, fine = {}, {}, {}, [], []
    for bi, p in enumerate(fans):
        Fv[bi] = len(fine); fine.append(p); rows_ok.append(True)
    for p, ring in fans.items():
        n = len(ring)
        for i, (v, e, _) in enumerate(ring):
            a, b = loc[v], loc[ring[(i + 1) % n][0]]
            key = (min(a, b), max(a, b))
            if key not in Ev:
                Ev[key] = len(fine); fine.append(e); rows_ok.append(e in drop_e or len(vf[e]) == 2)
    for v in pv:
        Vv[loc[v]] = len(fine); fine.append(v); rows_ok.append(all(g in in_fan for g in vf[v]))
    X = V[fine]
    S = subdiv_operator(blf, len(pv), 'simple' if scheme == 'simple' else 'cc', Ev, Fv, Vv, len(fine))
    w = np.array(rows_ok, float)[:, None]
    Pb = V[pv].copy()
    if scheme != 'simple':
        Pb = cgls(lambda Q: w * S.dot(Q), lambda Y: S.tdot(w * Y), w * X, Pb, iters=iters)
    d = np.linalg.norm(S.dot(Pb) - X, axis=1) / mel
    fan_res = {}                  # a fan fits exactly when its face point and its fitted edge points do
    for bi, (p, ring) in enumerate(fans.items()):
        rows = [Fv[bi]]; n = len(ring)
        for i in range(n):
            a, b = loc[ring[i][0]], loc[ring[(i + 1) % n][0]]
            r_ = Ev[(min(a, b), max(a, b))]
            if w[r_, 0] > 0: rows.append(r_)
        fan_res[p] = float(d[rows].max())
    return Pb, pv, d, w, fan_res


def _ngon_excess(fans):
    return float(np.mean([max(0, len(r) - 4) for r in fans.values()])) if fans else 0.0


def unsubdivide_partial(V, F, levels=1, schemes=('simple', 'cc'), seeds=3, log=None):
    """Peel up to `levels` levels where the mesh is still a subdivision grid; keep the rest.
    Returns dict(V, F, orig (output vertex -> input vertex), corner_src (per output face, the (input face, corner)
    each corner comes from -- for UVs and materials), cage_faces (number of leading faces of F that were peeled at
    the last level), levels=[info per level])."""
    V = np.asarray(V, float); F = [list(map(int, f)) for f in F]
    orig = list(range(len(V))); out = []
    cur_cage = 0; csrc = [[(i, k) for k in range(len(f))] for i, f in enumerate(F)]
    for lev in range(levels):
        vf, ef = _rings(F); deg, bnd = _degrees(ef)
        all_lab, all_fans, picked, by_fan = {}, {}, [], {}
        for comp in _face_components(F):
            quads = [i for i in comp if len(F[i]) == 4]
            if not quads: continue
            got = _choose_reading(V, F, quads, vf, ef, deg, bnd, schemes, seeds)
            if got is None: continue
            lab, fans, sc, res = got
            all_lab.update(lab); all_fans.update(fans); picked.append((len(fans), sc))
            by_fan.update({p: sc for p in fans})
        if not all_fans: break
        Vo, Fo, used, nb, info = _peel(V, F, all_lab, all_fans, vf, ef, by_fan)
        if not info['valid']: break
        sc = info['scheme']
        res = info['residual_rel']; fans = all_fans
        pick = (res, sc, all_lab, fans, Vo, Fo, used, nb, info)
        res, sc, lab, fans, Vo, Fo, used, nb, info = pick
        info['components'] = len(picked); info['faces_in'] = len(F); info['faces_out'] = len(Fo)
        out.append(info)
        if log: log(f"partial level {lev + 1}: {len(F)} -> {len(Fo)} faces, {info['peeled']} peeled, {info['kept']} kept, "
                    f"{sc}, residual {res:.3f}")
        csrc = [[csrc[g][k] for g, k in fs] for fs in info.pop('corner_src')]
        orig = [orig[v] for v in used]; V, F = Vo, Fo; cur_cage = nb
    return dict(V=V, F=F, orig=orig, corner_src=csrc, cage_faces=cur_cage, levels=out)
