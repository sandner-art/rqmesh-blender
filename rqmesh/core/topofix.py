# SPDX-License-Identifier: Apache-2.0
"""Local quad-topology repair: pole clean-up and cap filling.

Pole clean-up (Tarini et al. 2010 local operations; Peng et al. 2011 connectivity editing).  Every vertex gets an
IDEAL valence from the surface itself -- round(angle sum / 90 deg) (+1 on the boundary) -- so a valence-3 vertex on a
cube-like corner (270 deg) is correct and kept, while a valence-3 or -5 vertex on a flat or smoothly curved area is
an unwanted pole.  Energy E = sum_v (valence - ideal)^2.  Operations, each validated for topology (no degenerate
faces, doublets, non-manifold edges) and geometry (scaled Jacobian, normal agreement), never across feature lines:
  * edge rotation inside two quads (moves the shared edge to the other two hexagon diagonals),
  * diagonal collapse (merge two opposite corners of a quad),
  * doublet removal (valence-2 interior vertex).
Greedy best-first on dE < 0, with a depth-2 look-ahead (a neutral move followed by an improving one).  Index theory
fixes the total: sum(4 - valence) is set by the Euler characteristic, so poles can only be cancelled in pairs
(3 with 5) or moved to where the shape needs them; the energy encodes exactly that.  Finally a feature-preserving
tangential relaxation with projection to the input ("re-shape to smooth out the topology").

Cap filling.  A fan of triangles, an n-gon, or a high-valence pole (>= 6) on a flat or gently curved region is
replaced by a patterned quad grid: the region is grown over non-feature edges within an angle tolerance, its single
boundary loop (even length 2(a+b)) gets four corners chosen by boundary turning angle and side balance, and is filled
by Coons (transfinite) interpolation, projected onto the original surface and relaxed.  Poles end up at the four cap
corners, out of the flat face.  Odd boundary counts cannot be filled with quads only (parity), so one triangle is kept.

  python topofix.py IN.obj OUT.obj [--caps] [--poles] [--feature-deg 30] [--relax 10]  -> JSON line
"""
import sys, os, json, math, argparse, time
import numpy as np
try:
    from .meshio_min import read_obj, write_obj
    from .structsimp import Mesh, Projector, feature_edges, face_normal, sj_min, _ekey, chords
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj, write_obj
    from structsimp import Mesh, Projector, feature_edges, face_normal, sj_min, _ekey, chords


# ----------------------------------------------------------------------------- dynamic mesh with undo
class Edit(Mesh):
    def __init__(s, V, F):
        super().__init__(np.asarray(V, float), F); s.log = None

    def begin(s):
        s.log = dict(F={}, A={}, V={}, added=[])

    def _touch(s, fi):
        if s.log is not None and fi not in s.log['F'] and fi < len(s.F) and fi not in s.log['added']:
            s.log['F'][fi] = list(s.F[fi]); s.log['A'][fi] = s.alive[fi]

    def set_face(s, fi, f):
        s._touch(fi)
        for w in s.F[fi]: s.vf.get(w, set()).discard(fi)
        s.F[fi] = list(f); s.alive[fi] = True
        for w in f: s.vf.setdefault(w, set()).add(fi)

    def kill(s, fi):
        s._touch(fi)
        for w in s.F[fi]: s.vf.get(w, set()).discard(fi)
        s.alive[fi] = False

    def add(s, f):
        s.F.append(list(f)); s.alive.append(True); fi = len(s.F) - 1
        for w in f: s.vf.setdefault(w, set()).add(fi)
        if s.log is not None: s.log['added'].append(fi)
        return fi

    def add_vertex(s, p):
        s.V = np.vstack([s.V, p]); s.vf[len(s.V) - 1] = set(); return len(s.V) - 1

    def move(s, v, p):
        if s.log is not None and v not in s.log['V']: s.log['V'][v] = s.V[v].copy()
        s.V[v] = p

    def rollback(s):
        L = s.log; s.log = None
        for fi in sorted(L['added'], reverse=True):
            for w in s.F[fi]: s.vf.get(w, set()).discard(fi)
            s.alive[fi] = False
        for fi, f in L['F'].items():
            for w in s.F[fi]: s.vf.get(w, set()).discard(fi)
            s.F[fi] = f; s.alive[fi] = L['A'][fi]
            if s.alive[fi]:
                for w in f: s.vf.setdefault(w, set()).add(fi)
        for v, p in L['V'].items(): s.V[v] = p

    def commit(s):
        s.log = None

    def faces_of(s, v):
        return [g for g in s.vf.get(v, ()) if s.alive[g]]

    def nbrs(s, v):
        out = set()
        for g in s.faces_of(v):
            f = s.F[g]; k = f.index(v); out.add(f[k - 1]); out.add(f[(k + 1) % len(f)])
        return out

    def is_boundary_edge(s, a, b):
        return len([g for g in s.edge_faces(a, b) if s.alive[g]]) == 1

    def is_boundary(s, v):
        return any(s.is_boundary_edge(v, w) for w in s.nbrs(v))

    def valence(s, v):
        return len(s.nbrs(v))

    def ok_local(s, faces, old_normals=None, sj_floor=0.1):
        """Validity of the faces just created/changed."""
        for g in faces:
            if not s.alive[g]: continue
            f = s.F[g]
            if len(set(f)) != len(f): return False
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                ef = [x for x in s.edge_faces(a, b) if s.alive[x]]
                if len(ef) > 2: return False
                for o in ef:
                    if o != g and len(set(f) & set(s.F[o])) > 2: return False
            if len(f) == 4:
                if sj_min(s.V[f]) < sj_floor: return False
            if old_normals is not None and g in old_normals:
                if face_normal(s.V[f]) @ old_normals[g] < 0.5: return False
        for g in faces:
            if not s.alive[g]: continue
            for v in s.F[g]:
                if len(s.faces_of(v)) <= 2 and not s.is_boundary(v): return False
        return True


def angle_sum(V, F, vf, v):
    tot = 0.0
    for g in vf.get(v, ()):
        f = F[g]; k = f.index(v); a = V[f[k - 1]] - V[v]; b = V[f[(k + 1) % len(f)]] - V[v]
        c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30); tot += math.degrees(math.acos(np.clip(c, -1, 1)))
    return tot


# ----------------------------------------------------------------------------- pole clean-up
class PoleFixer:
    def __init__(s, V, F, feat_deg=30.0, extra_feat=None, proj=None, sj_floor=0.1):
        s.M = Edit(V, F); s.proj = proj; s.sj_floor = sj_floor
        s.feat = feature_edges(s.M.V, s.M.F, feat_deg) | {_ekey(int(a), int(b)) for a, b in (extra_feat or ())}
        fd = {}
        for a, b in s.feat: fd[a] = fd.get(a, 0) + 1; fd[b] = fd.get(b, 0) + 1
        s.fv = set(fd); s.corner = {v for v, d in fd.items() if d != 2}
        s.ideal = {}
        for v in range(len(s.M.V)):
            if v not in s.M.vf or not s.M.vf[v]: continue
            asum = angle_sum(s.M.V, s.M.F, s.M.vf, v)
            if s.M.is_boundary(v):
                s.ideal[v] = max(2, min(6, int(round(asum / 90.0)) + 1))
            else:                                     # wide dead band: only clear corners/saddles want 3 or 5
                s.ideal[v] = 3 if asum < 300 else 5 if asum > 420 else 4
        s.stats = dict(rotations=0, diag_collapses=0, doublets=0, lookahead=0)
        s.edited = set()                              # vertices touched by accepted edits (relax only around them)
        s.rings = []; s.ring_of = {}                  # guarded isolation rings (ordered face lists)
        s._nocache = bool(os.environ.get('RQM_NOCACHE'))

    # --- isolation rings: edits may change ring faces but never break the ring of quads
    def guard_rings(s, rings):
        for fl in rings:
            i = len(s.rings); s.rings.append(list(fl))
            for g in fl: s.ring_of.setdefault(g, []).append(i)
        s.stats['guarded_rings'] = len(s.rings)

    def guard_isolation(s, k=2, regions=()):
        """Guard the innermost k isolating rings of every opening (and face region)."""
        try:
            from . import isolation as ISO
        except ImportError:
            import isolation as ISO
        alive = [g for g in range(len(s.M.F)) if s.M.alive[g]]
        F = [s.M.F[g] for g in alive]; back = dict(enumerate(alive)); fwd = {g: i for i, g in enumerate(alive)}
        T = ISO._Topo(F); R = ISO._closed_rings(s.M.V, F)
        seeds = [sf for _, _, sf in ISO._seeds(T, [[fwd[g] for g in r if g in fwd] for r in regions], True)]
        out = []
        for j, sf in enumerate(seeds):
            rs = ISO.isolating_rings(F, sf, R, [x for m, x in enumerate(seeds) if m != j], topo=T)
            out += [[back[g] for g in rf] for inner, rf, re in rs[:k]]
        s.guard_rings(out)
        return len(out)

    def _ring_ok(s, i):
        M = s.M; fl = s.rings[i]; n = len(fl)
        for g in fl:
            if not M.alive[g] or len(M.F[g]) != 4: return False
        for j in range(n):
            a, b, c = fl[j - 1], fl[j], fl[(j + 1) % n]
            fb = M.F[b]; sa = set(fb) & set(M.F[a]); sc = set(fb) & set(M.F[c])
            if len(sa) != 2 or len(sc) != 2 or sa & sc: return False      # entry and exit edges opposite
            ka = sorted(fb.index(v) for v in sa)
            if ka not in ([0, 1], [1, 2], [2, 3], [0, 3]): return False
        return True

    def _guard_ok(s):
        if not s.rings or s.M.log is None: return True
        hit = {i for g in s.M.log['F'] for i in s.ring_of.get(g, ())}
        return all(s._ring_ok(i) for i in hit)

    def E(s, v):
        if v not in s.M.vf or not s.M.faces_of(v): return 0.0
        return float((s.M.valence(v) - s.ideal.get(v, 4)) ** 2)

    def energy(s):
        return sum(s.E(v) for v in s.ideal if v in s.M.vf and s.M.faces_of(v))

    def poles(s):
        return sum(1 for v in s.ideal if v in s.M.vf and s.M.faces_of(v) and s.M.valence(v) != s.ideal[v])

    # --- operations: each returns (affected vertices, changed faces) after applying, or None (state unchanged)
    def op_rotate(s, a, b, which):
        M = s.M
        if _ekey(a, b) in s.feat: return None
        fs = [g for g in M.edge_faces(a, b) if M.alive[g]]
        if len(fs) != 2 or any(len(M.F[g]) != 4 for g in fs): return None
        f1 = M.F[fs[0]]; k = f1.index(a)
        if f1[(k + 1) % 4] != b: fs = fs[::-1]; f1 = M.F[fs[0]]; k = f1.index(a)
        if f1[(k + 1) % 4] != b: return None
        f2 = M.F[fs[1]]; k2 = f2.index(b)
        if f2[(k2 + 1) % 4] != a: return None
        c, d = f1[(k + 2) % 4], f1[(k + 3) % 4]; e, f = f2[(k2 + 2) % 4], f2[(k2 + 3) % 4]
        H = [b, c, d, a, e, f]
        if len(set(H)) != 6: return None
        if which == 0: q1, q2 = [H[1], H[2], H[3], H[4]], [H[4], H[5], H[0], H[1]]
        else:          q1, q2 = [H[2], H[3], H[4], H[5]], [H[5], H[0], H[1], H[2]]
        if (q1[0] in M.nbrs(q1[3])) and len(M.edge_faces(q1[0], q1[3]) - set(fs)) > 0: return None   # edge exists
        old_n = {g: face_normal(M.V[M.F[g]]) for g in fs}
        M.begin(); M.set_face(fs[0], q1); M.set_face(fs[1], q2)
        old_n = {fs[0]: old_n[fs[0]], fs[1]: old_n[fs[1]]}
        if not M.ok_local(fs, old_n, s.sj_floor): M.rollback(); return None
        return set(H), fs

    def op_diag(s, g, k):
        M = s.M; f = M.F[g]
        if len(f) != 4 or not M.alive[g]: return None
        a, c = f[k], f[(k + 2) % 4]
        if a in s.corner and c in s.corner: return None
        if a in s.fv and c in s.fv: return None                     # would merge two feature vertices
        if M.is_boundary(a) and M.is_boundary(c): return None
        keep, gone = (a, c)
        if c in s.fv or c in s.corner or M.is_boundary(c): keep, gone = c, a
        p = M.V[keep].copy() if (keep in s.fv or M.is_boundary(keep)) else 0.5 * (M.V[a] + M.V[c])
        aff_f = set(M.faces_of(gone)) | set(M.faces_of(keep))
        old_n = {x: face_normal(M.V[M.F[x]]) for x in aff_f if x != g}
        M.begin(); M.kill(g)
        for x in M.faces_of(gone):
            M.set_face(x, [keep if w == gone else w for w in M.F[x]])
        M.move(keep, p)
        if s.proj is not None and keep not in s.fv and not M.is_boundary(keep): M.move(keep, s.proj.surface(p))
        ch = [x for x in aff_f if x != g]
        if not M.ok_local(ch, old_n, s.sj_floor): M.rollback(); return None
        s.ideal[keep] = s.ideal.get(keep, 4)
        return {keep, gone, f[(k + 1) % 4], f[(k + 3) % 4]}, ch

    def op_doublet(s, v):
        M = s.M; fs = M.faces_of(v)
        if len(fs) != 2 or M.is_boundary(v) or any(len(M.F[g]) != 4 for g in fs) or v in s.corner: return None
        f1 = M.F[fs[0]]; k = f1.index(v); a, x, b = f1[(k + 1) % 4], f1[(k + 2) % 4], f1[(k + 3) % 4]
        f2 = M.F[fs[1]]; k2 = f2.index(v); y = f2[(k2 + 2) % 4]
        q = [a, x, b, y]
        if len(set(q)) != 4: return None
        old_n = {fs[0]: face_normal(M.V[f1])}
        M.begin(); M.kill(fs[1]); M.set_face(fs[0], q)
        if not M.ok_local([fs[0]], old_n, 0.0): M.rollback(); return None
        return {v, a, b, x, y}, [fs[0]]

    # --- search
    def candidates(s, v):
        M = s.M; out = []
        if v not in M.vf: return out
        for g in M.faces_of(v):
            f = M.F[g]
            if len(f) != 4: continue
            for k in range(4):
                a, b = f[k], f[(k + 1) % 4]
                out.append(('rot', min(a, b), max(a, b), 0)); out.append(('rot', min(a, b), max(a, b), 1))
            out.append(('diag', g, f.index(v), 0))
            out.append(('diag', g, (f.index(v) + 1) % 4, 0))
        if len(M.faces_of(v)) == 2: out.append(('dbl', v, 0, 0))
        return sorted(set(out))                       # deterministic: set iteration order depends on edit history

    def run_op(s, op):
        if op[0] == 'rot': r = s.op_rotate(op[1], op[2], op[3])
        elif op[0] == 'diag': r = s.op_diag(op[1], op[2])
        else: r = s.op_doublet(op[1])
        if r is not None and not s._guard_ok(): s.M.rollback(); return None
        return r

    def local_E(s, verts):
        return sum(s.E(v) for v in verts)

    def try_op(s, op):
        """Apply op tentatively; return (dE, verts) with the op applied, or None."""
        M = s.M; pre_v = s._op_verts(op)
        if pre_v is None: return None
        e0 = s.local_E(pre_v)
        r = s.run_op(op)
        if r is None: return None
        verts = r[0] | pre_v
        return s.local_E(verts) - e0, verts

    def _op_verts(s, op):
        M = s.M
        if op[0] == 'rot':
            fs = [g for g in M.edge_faces(op[1], op[2]) if M.alive[g]]
            return {w for g in fs for w in M.F[g]} if len(fs) == 2 else None
        if op[0] == 'diag':
            if not M.alive[op[1]]: return None
            return set(M.F[op[1]])
        return {op[1]} | M.nbrs(op[1])

    def _op_keyverts(s, op):
        M = s.M
        if op[0] == 'rot': return {op[1], op[2]}
        if op[0] == 'diag': return set(M.F[op[1]]) if op[1] < len(M.F) else set()
        return {op[1]}

    def improve(s, max_iter=100000, lookahead=True):
        """Greedy descent on the valence energy.  Trial results are cached per operation and dropped only when an
        accepted edit lands within two rings of the vertices the trial touched (same choices as re-trying every
        candidate on every pass, at a fraction of the cost)."""
        M = s.M
        cache, vidx = {}, {}

        def evaluate(op):
            hit = None if s._nocache else cache.get(op)
            if hit is not None: return hit
            foot = s._op_keyverts(op)
            pre = s._op_verts(op)
            if pre is not None: foot |= pre
            r = s.try_op(op)
            if r is None: dE = None
            else:
                dE, verts = r; M.rollback(); foot |= verts
            cache[op] = (dE,)
            for w in foot: vidx.setdefault(w, set()).add(op)
            return cache[op]

        def invalidate(W):
            zone = set(W)
            for _ in range(2):
                zone |= {x for w in list(zone) if w in M.vf for x in M.nbrs(w)}
            for w in zone:
                for op in vidx.pop(w, ()): cache.pop(op, None)
        it = 0
        while it < max_iter:
            it += 1
            bad = [v for v in list(s.ideal) if v in M.vf and M.faces_of(v) and M.valence(v) != s.ideal[v]]
            best = None
            for v in bad:
                for op in s.candidates(v):
                    dE = evaluate(op)[0]
                    if dE is None: continue
                    if dE < -1e-9 and (best is None or dE < best[0]): best = (dE, op)
            if best is not None:
                W0 = s._op_verts(best[1]) or set()
                r = s.try_op(best[1]); M.commit()
                if r is not None: s.edited |= r[1]; W0 |= r[1]
                invalidate(W0 | s._op_keyverts(best[1]))
                s.stats[{'rot': 'rotations', 'diag': 'diag_collapses', 'dbl': 'doublets'}[best[1][0]]] += 1
                continue
            if not lookahead: break
            found = False
            for v in bad:                               # depth 2: neutral/slightly worse move then improving one
                for op in s.candidates(v):
                    r = s.try_op(op)
                    if r is None: continue
                    dE1, verts = r
                    if dE1 > 1.0 + 1e-9: M.rollback(); continue
                    log1 = M.log; M.log = None
                    best2 = None
                    for w in verts:
                        if w not in M.vf: continue
                        for op2 in s.candidates(w):
                            M.begin()
                            pre = s._op_verts(op2)
                            if pre is None: M.log = None; continue
                            e0 = s.local_E(pre)
                            r2 = s.run_op(op2)
                            if r2 is None: M.log = None; continue
                            dE2 = s.local_E(r2[0] | pre) - e0
                            M.rollback()
                            if dE1 + dE2 < -1e-9 and (best2 is None or dE2 < best2[0]): best2 = (dE2, op2)
                    if best2 is not None:
                        r2 = s.run_op(best2[1])
                        if r2 is not None:
                            M.commit(); s.stats['lookahead'] += 1; found = True; s.edited |= verts | r2[0]
                            cache.clear(); vidx.clear()
                            for o in (op, best2[1]):
                                s.stats[{'rot': 'rotations', 'diag': 'diag_collapses', 'dbl': 'doublets'}[o[0]]] += 1
                            break
                    M.log = log1; M.rollback()
                if found: break
            if not found: break

    # --- chord-level operations: the only way to cancel or move single poles in an all-quad mesh
    def _chord_ok(s, edges, faces):
        if not faces or any(f < 0 for f in faces) or len(set(faces)) != len(faces): return False
        verts = [v for e in edges for v in e]
        if len(set(verts)) != len(verts): return False
        for u, v in edges:
            if u in s.fv and v in s.fv and _ekey(u, v) not in s.feat: return False
            if u in s.corner and v in s.corner: return False
        return True

    def collapse_chord(s, edges, faces):
        """Dissolve an edge loop (ring collapse) as one undoable edit.  Merged valence = val(u) + val(v) - 4, so a
        rung joining a 3-pole and a 5-pole cancels both."""
        M = s.M; rep, pos = {}, {}
        for (u, v) in edges:
            if (v in s.corner) or (v in s.fv and u not in s.fv) or (M.is_boundary(v) and not M.is_boundary(u)): u, v = v, u
            keep = u in s.corner or (u in s.fv and v not in s.fv) or (M.is_boundary(u) and not M.is_boundary(v))
            rep[v] = u; pos[u] = M.V[u].copy() if keep else 0.5 * (M.V[u] + M.V[v])
        fset = set(faces); region = {g for w in list(rep) + list(rep.values()) for g in M.faces_of(w)} - fset
        old_n = {g: face_normal(M.V[M.F[g]]) for g in region}
        M.begin()
        for g in fset: M.kill(g)
        for g in region:
            M.set_face(g, [rep.get(w, w) for w in M.F[g]])
        for u, p in pos.items():
            if s.proj is not None and u not in s.fv and not M.is_boundary(u): p = s.proj.surface(p)
            M.move(u, p)
        if not M.ok_local(list(region), old_n, s.sj_floor) or not s._guard_ok(): M.rollback(); return None
        for v, u in rep.items(): s.ideal.setdefault(u, s.ideal.get(v, 4))
        return set(rep) | set(rep.values()), list(region)

    def split_chord(s, edges, faces):
        """Insert an edge loop through a chord (loop cut): pole-neutral, used to give back resolution."""
        M = s.M; mid = {}
        M.begin()
        for (u, v) in edges:
            p = 0.5 * (M.V[u] + M.V[v])
            if s.proj is not None and not (u in s.fv and v in s.fv): p = s.proj.surface(p)
            mid[_ekey(u, v)] = M.add_vertex(p)
        for g in faces:
            f = M.F[g]; es = [_ekey(f[k], f[(k + 1) % 4]) for k in range(4)]
            ks = [k for k in range(4) if es[k] in mid]
            if len(ks) != 2: M.rollback(); return None
            k0, k1 = ks; m0, m1 = mid[es[k0]], mid[es[k1]]
            q1 = [f[k0], m0, m1, f[(k1 + 1) % 4]] if (k1 - k0) % 4 == 2 else None
            if q1 is None: M.rollback(); return None
            q2 = [m0, f[(k0 + 1) % 4], f[k1], m1]
            M.set_face(g, q1); M.add(q2)
        # neighbours outside the chord that use a split rung edge: insert the midpoint (keeps the mesh conforming)
        for (u, v) in edges:
            for g in list(M.faces_of(u)):
                f = M.F[g]
                if v in f and len(f) >= 3:
                    k = f.index(u)
                    if f[(k + 1) % len(f)] == v: M.set_face(g, f[:k + 1] + [mid[_ekey(u, v)]] + f[k + 1:])
                    elif f[k - 1] == v: M.set_face(g, f[:k] + [mid[_ekey(u, v)]] + f[k:])
        if not s._guard_ok(): M.rollback(); return None
        for w in mid.values(): s.ideal[w] = 4
        s.edited |= set(mid.values()); M.commit(); return True

    def rung_dE(s, edges):
        M = s.M; d = 0.0
        for u, v in edges:
            if M.is_boundary(u) or M.is_boundary(v): continue
            vu, vv = M.valence(u), M.valence(v); iu, iv = s.ideal.get(u, 4), s.ideal.get(v, 4)
            keep_ideal = iu if (u in s.corner or u in s.fv) else iv if (v in s.corner or v in s.fv) else min(iu, iv, key=lambda x: abs(x - 4))
            d += (vu + vv - 4 - keep_ideal) ** 2 - (vu - iu) ** 2 - (vv - iv) ** 2
        return d

    def cancel_by_chords(s, max_ops=200, keep_count=True, min_gain=-1.0):
        """Greedy: dissolve the edge loop with the largest pole-energy drop (3-5 rungs); optionally give the faces
        back with a loop cut through the widest pole-free chord."""
        M = s.M; n0 = sum(M.alive)
        blocked = set()
        for _ in range(max_ops):
            em = M.edges(); cand = []
            for edges, faces, closed in chords(M, em):
                if not s._chord_ok(edges, faces) or frozenset(faces) in blocked: continue
                d = s.rung_dE(edges)
                if d <= min_gain: cand.append((d, len(faces), edges, faces))
            cand.sort(key=lambda c: (c[0], c[1])); done = False
            for d, _, edges, faces in cand:
                pre = {w for g in faces for w in M.F[g]} | {x for e in edges for w in e for x in M.nbrs(w)}
                e0 = s.local_E(pre)
                r = s.collapse_chord(edges, faces)
                if r is None: blocked.add(frozenset(faces)); continue
                if s.local_E(pre | r[0]) - e0 >= 0: M.rollback(); blocked.add(frozenset(faces)); continue
                M.commit(); s.stats['chord_collapses'] = s.stats.get('chord_collapses', 0) + 1; done = True
                s.edited |= r[0]; break
            if not done: break
        if keep_count:
            guard = 0
            while sum(M.alive) < n0 * 0.995 and guard < 50:
                guard += 1; em = M.edges(); best = None
                for edges, faces, closed in chords(M, em):
                    if not faces or any(f < 0 for f in faces) or len(set(faces)) != len(faces): continue
                    verts = {w for e in edges for w in e}
                    if any(M.valence(w) != s.ideal.get(w, 4) for g in faces for w in M.F[g]): continue
                    if sum(M.alive) + len(faces) > n0 * 1.01: continue
                    width = max(np.linalg.norm(M.V[u] - M.V[v]) for u, v in edges)
                    if best is None or width > best[0]: best = (width, edges, faces)
                if best is None or s.split_chord(best[1], best[2]) is None: break
                s.stats['loop_cuts'] = s.stats.get('loop_cuts', 0) + 1

    # --- region refill: replace a pole cluster by the minimal pattern its boundary allows
    def _region(s, seed, rings):
        """Faces within `rings` rings of vertex `seed` (all quads), or None."""
        M = s.M; verts = {seed}; faces = set()
        for _ in range(rings):
            fs = {g for v in verts for g in M.faces_of(v)}
            faces |= fs; verts |= {w for g in fs for w in M.F[g]}
        if any(len(M.F[g]) != 4 for g in faces): return None
        return faces

    def _bfs(s, a, bnd, max_hops=8):
        """Shortest-path tree from a over interior vertices (bnd: cached boundary flags)."""
        M = s.M; prev = {a: None}; front = [a]
        for _ in range(max_hops):
            nxt = []
            for u in front:
                for w in M.nbrs(u):
                    if w not in prev and not bnd(w): prev[w] = u; nxt.append(w)
            front = nxt
            if not front: break
        return prev

    def _region_path(s, prev, b, rings, max_faces=160):
        """Faces within `rings` rings of the tree path to b, then made convex (boundary vertices with 3+ region
        faces become interior).  None if unreachable, non-quad or too large."""
        M = s.M
        if b not in prev: return None
        path = [b]
        while prev[path[-1]] is not None: path.append(prev[path[-1]])
        verts = set(path); faces = set()
        for _ in range(rings):
            fs = {g for v in verts for g in M.faces_of(v)}
            faces |= fs; verts |= {w for g in fs for w in M.F[g]}
        for _ in range(20):                           # convexify
            if len(faces) > max_faces: return None
            k_in = {}
            for g in faces:
                for w in M.F[g]: k_in[w] = k_in.get(w, 0) + 1
            add = {g for w, k in k_in.items() if k >= 3 and len(M.faces_of(w)) > k for g in M.faces_of(w)} - faces
            if not add: break
            faces |= add
        if any(len(M.F[g]) != 4 for g in faces) or len(faces) > max_faces: return None
        return faces

    def refill_plan(s, faces):
        """Corners, sides and pattern for a disk of quads.  Returns dict or None.
        A boundary vertex with k region faces turns by (2 - k) x 90 deg in the patch; keeping k fixes the valences of
        every boundary vertex, so only the interior changes.  #corners = 4 + sum over interior (valence - 4)."""
        M = s.M
        loop = boundary_loop(M.F, faces)
        if loop is None or len(loop) < 6: return None
        ls = set(loop)
        inner = {w for g in faces for w in M.F[g]} - ls
        if not inner: return None
        for w in inner:                               # never destroy protected or wanted structure
            if w in s.fv or w in s.corner or M.is_boundary(w) or s.ideal.get(w, 4) != 4: return None
        for g in faces:
            f = M.F[g]
            for k in range(4):
                a, b = f[k], f[(k + 1) % 4]
                if _ekey(a, b) in s.feat and not (a in ls and b in ls): return None
        k_in = {v: sum(1 for g in M.faces_of(v) if g in faces) for v in loop}
        if any(k not in (1, 2) for k in k_in.values()): return None      # convex patches only
        ci = [i for i, v in enumerate(loop) if k_in[v] == 1]
        N = len(ci)
        if N < 3 or N > 5: return None
        n = len(loop); L = [((ci[(j + 1) % N] - ci[j]) % n) for j in range(N)]
        poles_in = sum(1 for w in inner if M.valence(w) != 4)
        if N == 4:
            if L[0] != L[2] or L[1] != L[3]: return None
            return dict(loop=loop, ci=ci, N=4, L=L, r=None, faces=faces, inner=inner, gain=poles_in, new_faces=L[0] * L[1])
        A = np.zeros((N, N))
        for j in range(N): A[j, (j - 1) % N] = 1; A[j, (j + 1) % N] = 1
        r = np.linalg.solve(A, np.array(L, float)); ri = np.rint(r).astype(int)
        if np.abs(r - ri).max() > 1e-6 or ri.min() < 1: return None
        nf = sum(int(ri[(j - 1) % N]) * int(ri[j]) for j in range(N))     # sub-patch j: x_j * y_{j-1} = r_{j-1} * r_j
        return dict(loop=loop, ci=ci, N=N, L=L, r=[int(x) for x in ri], faces=faces, inner=inner, gain=poles_in - 1,
                    new_faces=nf)

    def refill_apply(s, plan):
        """Build the pattern (Coons sub-grids sharing separatrices and centre), project, relax, check."""
        M = s.M; loop, ci, N, L = plan['loop'], plan['ci'], plan['N'], plan['L']; n = len(loop)
        faces = plan['faces']
        old_c = np.array([M.V[M.F[g]].mean(0) for g in faces])
        old_n = np.array([face_normal(M.V[M.F[g]]) for g in faces])
        pj = s.proj

        def proj(p):
            return pj.surface(p) if pj is not None else p

        def bchain(i0, m):                            # boundary vertex ids from loop index i0, m edges forward
            return [loop[(i0 + k) % n] for k in range(m + 1)]
        M.begin(); newf = []; inner_new = []
        if N == 4:
            P = M.V[loop]; a, b = L[0], L[1]
            G = coons_fill(P, ci, a, b)
            idx = np.full((a + 1, b + 1), -1, int)
            for k in range(a + 1): idx[k, 0] = loop[(ci[0] + k) % n]
            for k in range(b + 1): idx[a, k] = loop[(ci[1] + k) % n]
            for k in range(a + 1): idx[a - k, b] = loop[(ci[2] + k) % n]
            for k in range(b + 1): idx[0, b - k] = loop[(ci[3] + k) % n]
            for i in range(1, a):
                for j in range(1, b):
                    idx[i, j] = M.add_vertex(proj(G[i, j])); inner_new.append(int(idx[i, j]))
            for g in faces: M.kill(g)
            for i in range(a):
                for j in range(b):
                    newf.append(M.add([int(idx[i, j]), int(idx[i + 1, j]), int(idx[i + 1, j + 1]), int(idx[i, j + 1])]))
        else:
            r = plan['r']
            s_pos = [(ci[j] + r[(j - 1) % N]) % n for j in range(N)]   # x_j = r_{j-1} edges after corner j
            S = [loop[i] for i in s_pos]
            C = proj(np.mean([M.V[v] for v in S], axis=0)); cid = M.add_vertex(C); inner_new.append(cid)
            sep = []                                  # separatrix j: centre -> s_j, r_j edges
            for j in range(N):
                ids = [cid]
                for k in range(1, r[j]):
                    t = k / r[j]; v = M.add_vertex(proj((1 - t) * C + t * M.V[S[j]])); ids.append(v); inner_new.append(v)
                ids.append(S[j]); sep.append(ids)
            for g in faces: M.kill(g)
            for j in range(N):                        # sub-patch at corner j: s_{j-1} -> c_j -> s_j -> centre
                jm = (j - 1) % N
                a, b = r[jm], r[j]                    # a = x_j = r_{j-1} (c_j -> s_j), b = y_{j-1} = r_j (s_{j-1} -> c_j)
                A_ = bchain(ci[j], a)                 # c_j .. s_j
                D_ = bchain(s_pos[jm], b)             # s_{j-1} .. c_j
                B_ = sep[j][::-1]                     # s_j .. centre  (r_j = b edges)
                C_ = sep[jm]                          # centre .. s_{j-1} (r_{j-1} = a edges)
                ring = A_[:-1] + B_[:-1] + C_[:-1] + D_[:-1]
                P = M.V[ring]
                G = coons_fill(P, [0, a, a + b, 2 * a + b], a, b)
                idx = np.full((a + 1, b + 1), -1, int)
                for k in range(a + 1): idx[k, 0] = A_[k]
                for k in range(b + 1): idx[a, k] = B_[k]
                for k in range(a + 1): idx[a - k, b] = C_[k]
                for k in range(b + 1): idx[0, b - k] = D_[k]
                for i in range(1, a):
                    for k in range(1, b):
                        idx[i, k] = M.add_vertex(proj(G[i, k])); inner_new.append(int(idx[i, k]))
                for i in range(a):
                    for k in range(b):
                        newf.append(M.add([int(idx[i, k]), int(idx[i + 1, k]), int(idx[i + 1, k + 1]), int(idx[i, k + 1])]))
        # orientation: the old faces ran along the loop as (loop[i], loop[i+1]); new faces must too
        e0 = (loop[0], loop[1])
        if not any((M.F[g][k], M.F[g][(k + 1) % 4]) == e0 for g in newf for k in range(4)):
            for g in newf:
                M.F[g] = M.F[g][::-1]
        for _ in range(15):                           # relax the new interior on the surface
            for v in inner_new:
                nb = list(M.nbrs(v)); M.V[v] = proj(0.5 * M.V[v] + 0.5 * M.V[nb].mean(0))
        # normals: compare with the nearest old face
        okn = {}
        for g in newf:
            c = M.V[M.F[g]].mean(0); okn[g] = old_n[int(np.argmin(np.sum((old_c - c) ** 2, 1)))]
        if not M.ok_local(newf, okn, max(s.sj_floor, 0.2)) or not s._guard_ok():
            M.rollback(); return None
        for v in inner_new: s.ideal[v] = 4
        return set(inner_new) | set(loop), newf

    def refill_poles(s, max_rings=4, max_ops=500, path_reach=8.0):
        """Greedy: around every misplaced interior pole try regions of 1..max_rings rings; apply the refill that removes
        the most poles (ties: fewer new faces changed).  Face count may change slightly (patterns are boundary-driven)."""
        M = s.M; ops = 0; tried = set()
        al = [g for g in range(len(M.F)) if M.alive[g]]
        s.mel = float(np.mean([np.linalg.norm(M.V[M.F[g][0]] - M.V[M.F[g][1]]) for g in al[:4000]])) if al else 1.0
        while ops < max_ops:
            bad = [v for v in list(s.ideal) if v in M.vf and M.faces_of(v) and not M.is_boundary(v)
                   and v not in s.fv and M.valence(v) != s.ideal[v]]
            best = None
            regions = []
            for v in bad:
                for rg in range(1, max_rings + 1):
                    faces = s._region(v, rg)
                    if faces is None: break
                    regions.append(faces)
            pos = {v: M.V[v] for v in bad}; bcache = {}

            def bnd(w):
                if w not in bcache: bcache[w] = M.is_boundary(w)
                return bcache[w]
            for i, v in enumerate(bad):               # pairs of poles: their dislocations may cancel in one patch
                near = [u for u in bad[i + 1:] if np.linalg.norm(pos[v] - pos[u]) <= path_reach * s.mel]
                if not near: continue
                prev = s._bfs(v, bnd)
                for u in near:
                    for rg in (1, 2):
                        faces = s._region_path(prev, u, rg)
                        if faces is not None: regions.append(faces)
            for faces in regions:
                key = frozenset(faces)
                if key in tried: continue
                pl = s.refill_plan(faces)
                if pl is None or pl['gain'] < 1: continue
                score = (pl['gain'], -abs(pl['new_faces'] - len(faces)), -len(faces))
                if best is None or score > best[0]: best = (score, pl)
            if best is None: break
            r = s.refill_apply(best[1])
            if r is None:
                tried.add(frozenset(best[1]['faces'])); continue
            M.commit(); ops += 1; s.edited |= r[0]
            s.stats['refills'] = s.stats.get('refills', 0) + 1
            s.stats['refill_poles_removed'] = s.stats.get('refill_poles_removed', 0) + best[1]['gain']

    def relax(s, iters=10, w=0.5, rings=2):
        """Relax only around edited vertices (k rings), so untouched hand-made geometry is never smoothed."""
        M = s.M
        zone = {v for v in s.edited if v in M.vf}
        for _ in range(rings):
            zone |= {x for v in list(zone) for x in M.nbrs(v)}
        if not zone: return
        fv_nb = {}
        for a, b in s.feat:
            fv_nb.setdefault(a, []).append(b); fv_nb.setdefault(b, []).append(a)
        for _ in range(iters):
            newV = M.V.copy()
            for v in zone:
                if not M.faces_of(v) or v in s.corner or M.is_boundary(v): continue
                if v in s.fv:
                    nb = [x for x in fv_nb.get(v, []) if x in M.vf and M.faces_of(x)]
                    if len(nb) != 2: continue
                    p = (1 - w) * M.V[v] + w * 0.5 * (M.V[nb[0]] + M.V[nb[1]])
                    if s.proj is not None: p = s.proj.feature(p)
                else:
                    nb = list(M.nbrs(v)); p = (1 - w) * M.V[v] + w * M.V[nb].mean(0)
                    if s.proj is not None: p = s.proj.surface(p)
                newV[v] = p
            for v in range(len(M.V)):                # accept per vertex only if its quads stay healthy
                if np.allclose(newV[v], M.V[v]): continue
                old = M.V[v].copy(); before = min((sj_min(M.V[M.F[g]]) for g in M.faces_of(v) if len(M.F[g]) == 4), default=1)
                M.V[v] = newV[v]
                after = min((sj_min(M.V[M.F[g]]) for g in M.faces_of(v) if len(M.F[g]) == 4), default=1)
                if after < min(s.sj_floor, before): M.V[v] = old

    def result(s):
        F = [f for g, f in enumerate(s.M.F) if s.M.alive[g]]
        used = sorted({v for f in F for v in f}); rm = {v: i for i, v in enumerate(used)}
        return s.M.V[used], [[rm[v] for v in f] for f in F]


# ----------------------------------------------------------------------------- cap filling
def boundary_loop(F, faces):
    """Single boundary loop (ordered vertices) of a face set, or None if it is not a disk."""
    cnt = {}
    for g in faces:
        f = F[g]
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    bd = [(a, b) for (a, b) in cnt if (b, a) not in cnt]
    nxt = {}
    for a, b in bd:
        if a in nxt: return None
        nxt[a] = b
    if not nxt: return None
    start = next(iter(nxt)); loop = [start]; v = nxt[start]
    while v != start:
        loop.append(v); v = nxt.get(v)
        if v is None or len(loop) > len(nxt): return None
    if len(loop) != len(nxt): return None
    verts = {w for g in faces for w in F[g]}
    E = {_ekey(a, b) for (a, b) in cnt}
    if len(verts) - len(E) + len(faces) != 1: return None     # Euler: disk
    return loop


def choose_corners(P, loop_n, normal, max_eval=4000):
    """Four corners on a closed loop of n (even) points with side counts (a, b, a, b).  Every candidate is filled
    with a Coons grid and scored by its cells (mean and min scaled Jacobian, aspect), so a slot or a circle gets the
    'rounded rectangle' layout instead of a ladder."""
    n = loop_n; cands = []
    for a in range(1, n // 2):
        for i0 in range(n): cands.append((i0, a, n // 2 - a))
    if len(cands) > max_eval:                         # long loops: keep corners at the most turning vertices
        t = np.zeros(n)
        for i in range(n):
            d0 = P[i] - P[i - 1]; d1 = P[(i + 1) % n] - P[i]; t[i] = math.atan2(np.cross(d0, d1) @ normal, d0 @ d1)
        top = set(np.argsort(-t)[:max(8, int(math.sqrt(max_eval)))].tolist())
        cands = [c for c in cands if c[0] in top]
    best = None
    for i0, a, b in cands:
        ids = [i0, (i0 + a) % n, (i0 + a + b) % n, (i0 + 2 * a + b) % n]
        G = coons_fill(P, ids, a, b); sj, asp = [], []
        for i in range(a):
            for j in range(b):
                q = np.array([G[i, j], G[i + 1, j], G[i + 1, j + 1], G[i, j + 1]])
                sj.append(sj_min(q))
                e = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1); asp.append(math.log(max(e.max(), 1e-12) / max(e.min(), 1e-12)))
        score = float(np.mean(sj)) + 0.5 * float(np.min(sj)) - 0.1 * float(np.mean(asp))
        if best is None or score > best[0]: best = (score, ids, a, b)
    return best[1], best[2], best[3]


def coons_fill(P, ids, a, b):
    """Grid (a+1) x (b+1) from 4 boundary chains; returns grid of 3D points (boundary = P samples)."""
    n = len(P)
    def chain(i, m):
        return [P[(i + k) % n] for k in range(m + 1)]
    bot = chain(ids[0], a); right = chain(ids[1], b); top = chain(ids[2], a)[::-1]; left = chain(ids[3], b)[::-1]
    G = np.zeros((a + 1, b + 1, 3))
    for i in range(a + 1):
        for j in range(b + 1):
            u, v = i / a, j / b
            G[i, j] = ((1 - v) * bot[i] + v * top[i] + (1 - u) * left[j] + u * right[j]
                       - ((1 - u) * (1 - v) * bot[0] + u * (1 - v) * bot[a] + u * v * top[a] + (1 - u) * v * top[0]))
    return G


def find_caps(M, feat, ang_tol=12.0, max_faces=4000):
    """Seeds: n-gons, triangle fans (vertex with >= 4 incident triangles), poles of valence >= 6.  Each seed grows over
    non-feature edges while face normals stay within ang_tol of the seed's mean normal."""
    seeds = []
    for g, f in enumerate(M.F):
        if M.alive[g] and len(f) > 4: seeds.append(('ngon', [g]))
    for v in list(M.vf):
        fs = M.faces_of(v)
        if not fs: continue
        ntri = sum(1 for g in fs if len(M.F[g]) == 3)
        if ntri >= 4 or (len(fs) >= 6 and not M.is_boundary(v)): seeds.append(('fan' if ntri >= 4 else 'pole', fs))
    used, caps = set(), []
    for kind, fs in seeds:
        if any(g in used for g in fs): continue
        n0 = sum(face_normal(M.V[M.F[g]]) for g in fs); n0 /= np.linalg.norm(n0) + 1e-30
        region = set(fs); front = list(fs); c = math.cos(math.radians(ang_tol))
        while front and len(region) < max_faces:
            g = front.pop(); f = M.F[g]
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                if _ekey(a, b) in feat: continue
                for h in M.edge_faces(a, b):
                    if h in region or not M.alive[h] or h in used: continue
                    if face_normal(M.V[M.F[h]]) @ n0 < c: continue
                    region.add(h); front.append(h)
        loop = boundary_loop(M.F, region)
        if loop is None:                              # fall back to the seed's own faces
            region = set(fs); loop = boundary_loop(M.F, region)
            if loop is None: continue
        used |= region; caps.append(dict(kind=kind, faces=sorted(region), loop=loop, normal=n0))
    return caps


def fill_caps(V, F, feat_deg=30.0, ang_tol=12.0, relax_iters=20, density=1.0, proj=None):
    """Replace fan / n-gon / high-pole caps by quad grids.  density scales the number of rows (1 = boundary-driven)."""
    M = Edit(V, F)
    feat = feature_edges(M.V, M.F, feat_deg)
    caps = find_caps(M, feat, ang_tol)
    info = []
    for cap in caps:
        loop = cap['loop']; n = len(loop); P = M.V[loop]
        if n < 4: continue
        odd = n % 2 == 1
        if odd:                                       # parity: split the longest boundary edge of the cap... only
            info.append(dict(kind=cap['kind'], n=n, filled=False, reason='odd boundary (needs one triangle)'))
            continue                                  # inside the cap is impossible without touching neighbours
        ids, a, b = choose_corners(P, n, cap['normal'])
        G = coons_fill(P, ids, a, b)
        T = [[f[0], f[k], f[k + 1]] for g in cap['faces'] for f in [M.F[g]] for k in range(1, len(f) - 1)]
        pj = proj or Projector(M.V, T, set())
        idx = np.full((a + 1, b + 1), -1, int)
        for k in range(a + 1): idx[k, 0] = loop[(ids[0] + k) % n]
        for k in range(b + 1): idx[a, k] = loop[(ids[1] + k) % n]
        for k in range(a + 1): idx[a - k, b] = loop[(ids[2] + k) % n]
        for k in range(b + 1): idx[0, b - k] = loop[(ids[3] + k) % n]
        for i in range(1, a):
            for j in range(1, b):
                idx[i, j] = M.add_vertex(pj.surface(G[i, j]))
        for g in cap['faces']: M.kill(g)
        orient = np.cross(P[1] - P[0], P[2] - P[1]) @ cap['normal']
        newf = []
        for i in range(a):
            for j in range(b):
                q = [idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]]
                newf.append(M.add(q))
        # orientation: match the old faces' winding (boundary loop direction)
        e_loop = (loop[ids[0]], loop[(ids[0] + 1) % n])
        f0 = M.F[newf[0]]
        if not any((f0[k], f0[(k + 1) % 4]) == e_loop for k in range(4)):
            for g in newf: M.F[g] = M.F[g][::-1]
        inner = [int(idx[i, j]) for i in range(1, a) for j in range(1, b)]
        for _ in range(relax_iters):                  # Laplacian relax of the inner grid, projected
            for v in inner:
                nb = list(M.nbrs(v)); M.V[v] = pj.surface(0.5 * M.V[v] + 0.5 * M.V[nb].mean(0))
        info.append(dict(kind=cap['kind'], n=n, filled=True, grid=[a, b], old_faces=len(cap['faces']), new_faces=a * b))
    Fo = [f for g, f in enumerate(M.F) if M.alive[g]]
    used = sorted({v for f in Fo for v in f}); rm = {v: i for i, v in enumerate(used)}
    return M.V[used], [[rm[v] for v in f] for f in Fo], info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--caps', action='store_true'); ap.add_argument('--poles', action='store_true')
    ap.add_argument('--feature-deg', type=float, default=30.0); ap.add_argument('--relax', type=int, default=10)
    ap.add_argument('--no-lookahead', action='store_true'); ap.add_argument('--no-keep-count', action='store_true'); ap.add_argument('--loops', action='store_true', help='allow loop dissolve + loop cut (changes layout)'); ap.add_argument('--cap-angle', type=float, default=12.0)
    ap.add_argument('--refill', action='store_true', help='replace pole clusters by the minimal N-sided pattern')
    ap.add_argument('--keep-isolation', type=int, default=0, help='guard the innermost K isolation rings of every opening')
    a = ap.parse_args(); t0 = time.time()
    V, F = read_obj(a.inp); out = dict(faces_in=len(F))
    if a.caps:
        V, F, info = fill_caps(V, F, a.feature_deg, a.cap_angle)
        out['caps'] = info
    if a.poles:
        V0, F0 = read_obj(a.inp)
        T = [[f[0], f[k], f[k + 1]] for f in F0 for k in range(1, len(f) - 1)]
        proj = Projector(V0, T, feature_edges(V0, F0, a.feature_deg))
        pf = PoleFixer(V, F, a.feature_deg, proj=proj)
        if a.keep_isolation: pf.guard_isolation(a.keep_isolation)
        out['poles_before'] = pf.poles(); out['energy_before'] = pf.energy()
        if a.loops: pf.cancel_by_chords(keep_count=not a.no_keep_count)
        pf.improve(lookahead=not a.no_lookahead)
        if a.refill:
            pf.refill_poles(); pf.improve(lookahead=not a.no_lookahead)
        if a.relax: pf.relax(a.relax)
        out['poles_after'] = pf.poles(); out['energy_after'] = pf.energy(); out.update(pf.stats)
        V, F = pf.result()
    write_obj(a.out, V, F)
    out.update(faces=len(F), seconds=round(time.time() - t0, 2))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
