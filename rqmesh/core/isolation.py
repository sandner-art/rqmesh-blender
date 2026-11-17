# SPDX-License-Identifier: Apache-2.0
"""Isolation loops: closed rings of quads around an opening (eye socket, mouth) or a bump.

Modellers isolate a feature with a few concentric edge loops so it can be sculpted, cut or animated on its own.
Topologically each loop is a closed poly-chord (ring of quads) whose small side contains the feature.  A ring of
concentric loops is regular everywhere; the four singularities the index theorem asks for (an annulus in a quad
grid has interior index -4, i.e. typically four 5-poles) sit where the rings meet the surrounding grid -- the
"box corners" -- not on the loops themselves.

What this module gives:
  openings(F)                          boundary loops (ordered vertex lists)
  isolating_rings(F, seed, ...)        rings around one opening or one face region, innermost first
  ring_families(F, ...)                nested rings around bumps on closed surfaces (auto)
  protected_band(F, k, ...)            faces/vertices/side edges of the innermost k rings of every opening/region
  report(V, F, regions)                ring counts and poles near every opening/region (for benchmarks and the UI)
"""
import collections
import numpy as np

try:
    from .structsimp import Mesh, chords_fast
except ImportError:
    from structsimp import Mesh, chords_fast


def _ekey(a, b):
    return (a, b) if a < b else (b, a)


class _Topo:
    def __init__(s, F):
        s.F = [list(map(int, f)) for f in F]
        s.e2f = collections.defaultdict(list)
        for i, f in enumerate(s.F):
            for k in range(len(f)):
                s.e2f[_ekey(f[k], f[(k + 1) % len(f)])].append(i)
        s.fadj = [[] for _ in s.F]
        for e, fs in s.e2f.items():
            if len(fs) == 2:
                a, b = fs; s.fadj[a].append(b); s.fadj[b].append(a)
        par = list(range(len(s.F)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for a, nb in enumerate(s.fadj):
            for b in nb:
                ra, rb = find(a), find(b)
                if ra != rb: par[ra] = rb
        s.comp = [find(i) for i in range(len(s.F))]
        s.comp_size = collections.Counter(s.comp)


def openings(F, topo=None):
    """Boundary loops as ordered vertex lists (manifold boundaries only; branching boundaries are skipped)."""
    T = topo or _Topo(F)
    bE = [e for e, fs in T.e2f.items() if len(fs) == 1]
    nxt = collections.defaultdict(list)
    for a, b in bE:
        nxt[a].append(b); nxt[b].append(a)
    seen = set(); loops = []
    for a, b in bE:
        if a in seen: continue
        if len(nxt[a]) != 2: continue
        loop = [a]; prev, cur = a, nxt[a][0]; ok = True
        while cur != a:
            if len(nxt[cur]) != 2 or cur in seen: ok = False; break
            loop.append(cur); n2 = nxt[cur]
            prev, cur = cur, (n2[0] if n2[1] == prev else n2[1])
            if len(loop) > len(bE): ok = False; break
        seen.update(loop)
        if ok: loops.append(loop)
    return loops


def _closed_rings(V, F):
    """Closed, simple poly-chords: (face list, rung edges)."""
    M = Mesh(np.asarray(V, float), F)
    out = []
    for edges, faces, closed in chords_fast(M):
        if not closed or not faces or min(faces) < 0 or len(set(faces)) != len(faces): continue
        verts = [v for e in edges for v in e]
        if len(set(verts)) != len(verts): continue                # self-touching ring
        out.append((faces, edges))
    return out


def _flood(T, start, blocked, cap):
    """Faces reachable from `start` without entering `blocked`; None if more than cap."""
    seen = set(x for x in start if x not in blocked); st = list(seen)
    while st:
        g = st.pop()
        for h in T.fadj[g]:
            if h not in seen and h not in blocked:
                seen.add(h); st.append(h)
                if len(seen) > cap: return None
    return seen


def isolating_rings(F, seed_faces, rings=None, other_seeds=(), max_frac=0.45, topo=None, V=None):
    """Rings that cut seed_faces (an opening's adjacent faces, or a selected region) off from the rest of its
    component, with no other seed region on the inside.  Returns [(inner_faces, ring_faces, rung_edges)] sorted
    innermost first.  A ring may run through the seed itself (the first loop around an opening usually does)."""
    T = topo or _Topo(F)
    if rings is None: rings = _closed_rings(V, F)
    seed = set(seed_faces)
    if not seed: return []
    comp = T.comp[next(iter(seed))]; cap = int(max_frac * T.comp_size[comp])
    others = [set(o) for o in other_seeds]
    out = []
    for faces, edges in rings:
        S = set(faces)
        if T.comp[faces[0]] != comp: continue
        free = seed - S
        if free:
            inner = _flood(T, [next(iter(free))], S, cap)
            if inner is None or not free <= inner: continue      # one piece must hold every free seed face
        else:
            inner = set()
            touching = {h for g in S for h in T.fadj[g] if h not in S}
            # the ring hugs the seed: the seed side is the ring itself, provided the rest is one piece on the far side
            if not touching: continue
            far = _flood(T, [next(iter(touching))], S, T.comp_size[comp])
            if far is None or len(far) + len(S) != T.comp_size[comp]: continue
        if any(o & inner for o in others): continue
        out.append((inner, faces, edges))
    out.sort(key=lambda r: len(r[0]) + len(r[1]))
    return out


def ring_families(V, F, max_frac=0.2, min_nest=3, rings=None, topo=None, exclude=()):
    """Nested rings with a small side (a disk holding < max_frac of its component) around bumps on closed surfaces.
    Families with at least min_nest nested rings are returned as (inner_most_side, [rings innermost first])."""
    T = topo or _Topo(F)
    if rings is None: rings = _closed_rings(V, F)
    ex = set(exclude)
    small = []
    for faces, edges in rings:
        S = set(faces); comp = T.comp[faces[0]]; cap = int(max_frac * T.comp_size[comp])
        nb = [h for g in faces for h in T.fadj[g] if h not in S]
        if not nb: continue
        side = _flood(T, [nb[0]], S, cap)
        if side is None:
            other = [h for h in nb if h not in _flood(T, [nb[0]], S, T.comp_size[comp])]
            if not other: continue
            side = _flood(T, [other[0]], S, cap)
            if side is None: continue
        if side & ex: continue
        small.append((side, faces, edges))
    small.sort(key=lambda r: len(r[0]))
    fams = []                                                   # chains by inclusion of the small sides
    used = [False] * len(small)
    for i in range(len(small)):
        if used[i]: continue
        chain = [i]; cur = small[i][0] | set(small[i][1])
        for j in range(i + 1, len(small)):
            if used[j]: continue
            if cur <= small[j][0]:
                chain.append(j); cur = small[j][0] | set(small[j][1])
        if len(chain) >= min_nest:
            for j in chain: used[j] = True
            fams.append((small[chain[0]][0], [small[j] for j in chain]))
    return fams


def protected_band(V, F, k=2, regions=(), include_openings=True, families=False, max_frac=0.45):
    """Faces, vertices and side edges of the innermost k isolating rings of every opening (and of every face
    region given).  The side edges are the edge loops a modeller would see; passing them as feature edges to the
    simplifier keeps those loops."""
    T = _Topo(F); R = _closed_rings(V, F)
    seeds = _seeds(T, regions, include_openings)
    faces, verts, side = set(), set(), set(); per = []
    for j, (kind, ident, sf) in enumerate(seeds):
        others = [x[2] for m, x in enumerate(seeds) if m != j]
        rs = isolating_rings(F, sf, R, others, max_frac, T)
        kept = rs[:k]
        for inner, rf, re in kept:
            rung = set(re)
            faces |= set(rf)
            for g in rf:
                f = T.F[g]
                for q in range(4):
                    e = _ekey(f[q], f[(q + 1) % 4])
                    verts.add(f[q])
                    if e not in rung: side.add(e)
        per.append(dict(kind=kind, size=len(ident), rings=len(rs), protected=len(kept)))
    if families:
        ex = set().union(*[x[2] for x in seeds]) if seeds else set()
        for inner, chain in ring_families(V, F, rings=R, topo=T, exclude=ex):
            for side_f, rf, re in chain[:k]:
                rung = set(re); faces |= set(rf)
                for g in rf:
                    f = T.F[g]
                    for q in range(4):
                        e = _ekey(f[q], f[(q + 1) % 4]); verts.add(f[q])
                        if e not in rung: side.add(e)
            per.append(dict(kind='bump', size=len(inner), rings=len(chain), protected=min(k, len(chain))))
    return dict(faces=faces, verts=verts, side_edges=side, per=per)


def _seeds(T, regions=(), include_openings=True):
    """Seed face sets: faces along every opening, plus every region.  Seeds that overlap or touch are merged (an
    eye region selected together with its socket opening is one feature).  Returns [(kind, vertex set, face set)]."""
    raw = []
    if include_openings:
        for L in openings(T.F, T):
            raw.append(['opening', set(L), {g for i in range(len(L)) for g in T.e2f[_ekey(L[i], L[(i + 1) % len(L)])]}])
    for r in regions:
        r = {int(g) for g in r if 0 <= int(g) < len(T.F)}
        if r: raw.append(['region', {v for g in r for v in T.F[g]}, r])
    merged = True
    while merged:
        merged = False
        for i in range(len(raw)):
            grow = raw[i][2] | {h for g in raw[i][2] for h in T.fadj[g]}
            for j in range(i + 1, len(raw)):
                if raw[j][2] & grow:
                    raw[i][0] = 'opening' if 'opening' in (raw[i][0], raw[j][0]) else 'region'
                    raw[i][1] |= raw[j][1]; raw[i][2] |= raw[j][2]; del raw[j]; merged = True; break
            if merged: break
    return [tuple(x) for x in raw]


def _ring_loops(T, inner, rf, re):
    """(inner side edges, outer side edges) of a ring band."""
    rung = set(re); S = set(rf); ins, outs = [], []
    for g in rf:
        f = T.F[g]
        for q in range(4):
            e = _ekey(f[q], f[(q + 1) % 4])
            if e in rung: continue
            other = [h for h in T.e2f[e] if h != g]
            (ins if (not other or other[0] in inner) else outs).append(e)
    return ins, outs


def isolation_loops(V, F, k=2, spacing=None, target_faces=None, regions=(), families=False, max_frac=0.45,
                    include_openings=True):
    """Edge loops to keep around every opening (and region / bump): k loops at distances ~ spacing, 2 spacing, ...
    from the opening, chosen among the outer side loops of its isolating rings.  spacing defaults to the edge
    length of target_faces quads over the surface area.  Returns (edges, per-seed info).  Passed to the simplifier
    as feature edges, the loops stay and the rings between them collapse as usual -- isolation survives with the
    density the target asks for, instead of the input's thin subdivided bands."""
    V = np.asarray(V, float); T = _Topo(F); R = _closed_rings(V, F)
    if spacing is None:
        A = 0.0
        for f in T.F:
            P = V[f]
            for q in range(1, len(f) - 1): A += 0.5 * np.linalg.norm(np.cross(P[q] - P[0], P[q + 1] - P[0]))
        spacing = (A / max(target_faces or len(F), 1)) ** 0.5
    seeds = _seeds(T, regions, include_openings)
    chains = []
    for j, (kind, sv, sf) in enumerate(seeds):
        others = [x[2] for m, x in enumerate(seeds) if m != j]
        chains.append((kind, sv, isolating_rings(F, sf, R, others, max_frac, T)))
    if families:
        ex = set().union(*[x[2] for x in seeds]) if seeds else set()
        for inner, chain in ring_families(V, F, rings=R, topo=T, exclude=ex):
            core = inner or set(chain[0][1])
            chains.append(('bump', {v for g in core for v in T.F[g]}, chain))
    edges, info = set(), []
    for kind, sv, rs in chains:
        S = V[sorted(sv)]
        cand = []
        for inner, rf, re in rs:
            ins, outs = _ring_loops(T, inner, rf, re)
            lv = sorted({v for e in outs for v in e})
            if not lv: continue
            d = np.sqrt(((V[lv][:, None, :] - S[None, :, :]) ** 2).sum(-1)).min(1).mean()
            cand.append((float(d), outs))
        cand.sort(key=lambda c: c[0]); picked = []; last = 0.0
        for j in range(1, k + 1):
            want = j * spacing
            best = None
            for d, outs in cand:
                if d <= last + 0.5 * spacing: continue
                if best is None or abs(d - want) < abs(best[0] - want): best = (d, outs)
            if best is None: break
            picked.append(round(best[0], 4)); last = best[0]; edges |= set(best[1])
        info.append(dict(kind=kind, rings=len(rs), loops_kept=len(picked), distances=picked))
    return edges, dict(spacing=round(float(spacing), 5), seeds=info)


def report(V, F, regions=(), rings_near=4, families=False):
    """Per opening / region: isolating rings and the poles within `rings_near` vertex rings of it."""
    V = np.asarray(V, float); T = _Topo(F); R = _closed_rings(V, F)
    nbr = collections.defaultdict(set); bv = set()
    for e, fs in T.e2f.items():
        nbr[e[0]].add(e[1]); nbr[e[1]].add(e[0])
        if len(fs) == 1: bv.update(e)
    seeds = [(k, sv, sf, sv) for k, sv, sf in _seeds(T, regions, True)]
    out = []
    for j, (kind, ident, sf, sv) in enumerate(seeds):
        others = [x[2] for m, x in enumerate(seeds) if m != j]
        rs = isolating_rings(F, sf, R, others, topo=T)
        zone = set(sv)
        for _ in range(rings_near): zone |= {w for v in list(zone) for w in nbr[v]}
        poles = collections.Counter(len(nbr[v]) for v in zone if v not in bv and len(nbr[v]) != 4)
        # poles sitting on the innermost rings themselves (loops broken by a pole are not clean loops)
        on = collections.Counter()
        for inner, rf, re in rs[:2]:
            for v in {w for g in rf for w in T.F[g]}:
                if v not in bv and len(nbr[v]) != 4: on[len(nbr[v])] += 1
        c = V[list(sv)].mean(0)
        out.append(dict(kind=kind, size=len(ident), centre=[round(float(x), 4) for x in c],
                        component_faces=T.comp_size[T.comp[next(iter(sf))]], rings=len(rs),
                        poles_near=dict(sorted(poles.items())), poles_on_inner_rings=dict(sorted(on.items()))))
    if families:
        ex = set().union(*[x[2] for x in seeds]) if seeds else set()
        for inner, chain in ring_families(V, F, rings=R, topo=T, exclude=ex):
            c = V[list({v for g in (inner or chain[0][1]) for v in T.F[g]})].mean(0)
            out.append(dict(kind='bump', size=len(inner), centre=[round(float(x), 4) for x in c], rings=len(chain)))
    return out


if __name__ == '__main__':
    import sys, json
    try:
        from .meshio_min import read_obj
    except ImportError:
        from meshio_min import read_obj
    for p in [a for a in sys.argv[1:] if not a.startswith('--')]:
        V, F = read_obj(p)
        for r in report(V, F, families='--families' in sys.argv): print(p, json.dumps(r))
