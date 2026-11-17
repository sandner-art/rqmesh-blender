# SPDX-License-Identifier: Apache-2.0
"""Transition patterns and local (partial) edge-loop removal.

Why a pattern is needed.  In an all-quad mesh every row of quads (poly-chord) is closed or runs boundary to boundary
-- it can never simply stop.  So "remove a loop only here" means re-routing rows: the 3-1 pattern turns the two outer
columns of a 3-wide strip into a U-turn around the middle one.  A quad-only patch needs an even number of boundary
edges, so a single band can only change the column count by an even number (3-1, 4-2, 5-3; 2-1 and 4-1 need a
triangle/pentagon or must be paired).

Local strip coarsening.  A regular a x L block of quads (a >= 3 columns, no poles, no feature or seam lines inside)
is refilled with: a top band (a -> a-2, one 3-1 at the centre), L-2 rows of width a-2, and a bottom band (a-2 -> a).
The block's boundary vertices stay, so the change is strictly local: 2 vertical lines disappear over the middle rows,
faces drop from a*L to (a-2)*L + 6, and the bands add two 3-poles + two 5-poles each (the standard density-transition
cost).  Error = max distance of each removed vertex from the straight line its row now follows (absolute units).

adaptive_simplify() combines both tools on an error ladder: at each tolerance it first removes whole loops (no new
poles), then local strips; the tolerance grows only when nothing more fits.  This keeps the coarsening where the
surface is flat instead of forcing every row of a blocked region to go (the Suzanne failure of round 6).

  python transitions.py IN.obj OUT.obj TARGET [--local adaptive|stuck|off] [--feature-deg 8] [--symmetry auto] -> JSON
"""
import sys, os, json, math, argparse, time
import numpy as np
try:
    from .meshio_min import read_obj_uv, write_obj_uv, weld_uv
    from .structsimp import (Projector, feature_edges, face_normal, sj_min, _ekey, ring_collapse, prepare,
                             vertex_quadrics)
    from .topofix import Edit
    from . import symmetry as SYM
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj_uv, write_obj_uv, weld_uv
    from structsimp import (Projector, feature_edges, face_normal, sj_min, _ekey, ring_collapse, prepare,
                            vertex_quadrics)
    from topofix import Edit
    import symmetry as SYM


# ----------------------------------------------------------------------------- transition patterns
def pattern(a):
    """Quad-only band for an a-wide block whose middle rows keep b = a-2 columns.
    '31': a 3-1 centred in the band (a odd: 3-1, 5-3);  '42': the symmetric six-quad 4-2 centred (a even: 4-2, 6-4).
    Returns (band, c0, removed vertex lines, faces saved over L rows minus this constant)."""
    if a % 2:
        c0 = (a - 3) // 2; return '31', c0, (c0 + 1, c0 + 2), 6
    c0 = (a - 4) // 2; return '42', c0, (c0 + 1, c0 + 3), 8


_WIDE_W = {}


def _wide_weights(a):
    """Fixed interpolation weights of the re-spaced row (b = a-2 columns) and of the error probes, per width."""
    if a not in _WIDE_W:
        b = a - 2
        def lin(sp, n):
            i = np.minimum(np.floor(sp).astype(int), n - 2); return i, (sp - i)[:, None]
        i0, t0 = lin(np.array([i * a / b for i in range(b + 1)], float), a + 1)
        j0, u0 = lin(np.array([x * b / a for x in range(1, a)], float), b + 1)
        _WIDE_W[a] = (i0, t0, j0, u0)
    return _WIDE_W[a]


def _polyline_at(P, s):
    """Point at column parameter s on the row polyline P[0..a] (unit spacing in parameter)."""
    i = min(int(math.floor(s)), len(P) - 2); t = s - i
    return (1 - t) * P[i] + t * P[i + 1]


# ----------------------------------------------------------------------------- grid walking
def cell_of(M, g, rot):
    """Face g as an oriented cell (tl, tr, br, bl) starting at corner `rot`."""
    f = M.F[g]
    return tuple(f[(rot + k) % 4] for k in range(4))


def right_of(M, cell):
    tl, tr, br, bl = cell
    for g in M.edge_faces(tr, br):
        if not M.alive[g] or len(M.F[g]) != 4: continue
        f = M.F[g]
        if tl in f: continue
        k = f.index(br)
        if f[(k + 1) % 4] == tr:                     # neighbour has br -> tr: cycle (br, tr, a, b)
            return (tr, f[(k + 2) % 4], f[(k + 3) % 4], br), g
    return None, None


def down_of(M, cell):
    tl, tr, br, bl = cell
    for g in M.edge_faces(bl, br):
        if not M.alive[g] or len(M.F[g]) != 4: continue
        f = M.F[g]
        if tl in f: continue
        k = f.index(bl)
        if f[(k + 1) % 4] == br:                     # neighbour has bl -> br: cycle (bl, br, c, d)
            return (bl, br, f[(k + 2) % 4], f[(k + 3) % 4]), g
    return None, None


class Strips:
    def __init__(s, M, feat, corner, boundary_v, proj=None, FUV=None, UV=None, size_w=0.03):
        s.M, s.feat, s.corner, s.bv, s.proj = M, feat, corner, boundary_v, proj
        s.fv = {v for e in feat for v in e}
        s.FUV, s.UV = FUV, UV                          # shared lists: appended UVs must survive across passes
        s.size_w = size_w
        s.QL = s.WL = None                             # carried quadrics (lists aligned with M.V), optional
        s._val = None
        s.jump_w = 0.0                                 # penalty on the width jump a/b (set for gentle/mixed modes)
        s.sj_floor = 0.05                              # quad quality floor of new faces (raised for gentle modes)

    def val(s, v):
        """Valence, cached for one candidate pass (the mesh does not change while candidates are scored)."""
        if s._val is None:
            nb = {}
            for g, f in enumerate(s.M.F):
                if not s.M.alive[g]: continue
                k = len(f)
                for i in range(k):
                    nb.setdefault(f[i], set()).update((f[i - 1], f[(i + 1) % k]))
            s._val = {u: len(w) for u, w in nb.items()}
        return s._val.get(v, 0)

    def grow(s, g0, rot, a=3, max_rows=64, tol=None):
        """Regular a-wide block starting at face g0 going down.  Returns (cells, faces) rows or None."""
        M = s.M
        c0 = cell_of(M, g0, rot); row = [c0]; rf = [g0]
        for _ in range(a - 1):
            c, g = right_of(M, row[-1])
            if c is None: return None
            row.append(c); rf.append(g)
        rows, faces = [row], [rf]
        while len(rows) < max_rows:
            nr, nf = [], []
            for c in rows[-1]:
                d, g = down_of(M, c)
                if d is None: break
                nr.append(d); nf.append(g)
            if len(nr) != a: break
            ok = all(nr[k][1] == nr[k + 1][0] and nr[k][2] == nr[k + 1][3] for k in range(a - 1))
            if not ok or len(set(nf) | {x for r in faces for x in r}) != len(nf) + sum(len(r) for r in faces): break
            rows.append(nr); faces.append(nf)
        return rows, faces

    def grow_eval(s, g0, rot, a=3, max_rows=64, min_rows=4, tol=None):
        """grow() + evaluate() in one pass that stops at the first vertex row evaluate() would reject (error is a
        running maximum, so later rows cannot succeed).  Same (rows, faces, L, err, cost) as the two-step version."""
        M = s.M; val = s.val
        c0 = cell_of(M, g0, rot); row = [c0]; rf = [g0]
        for _ in range(a - 1):
            c, g = right_of(M, row[-1])
            if c is None: return None
            row.append(c); rf.append(g)
        if a != 3: return s._grow_eval_wide(row, rf, a, max_rows, min_rows, tol)
        rows, faces, fset = [row], [rf], set(rf)
        G = [[c[0] for c in row] + [row[-1][1]], [c[3] for c in row] + [row[-1][2]]]
        width = float(np.linalg.norm(M.V[G[1][0]] - M.V[G[1][a]]))
        jump = s.jump_w * width * 2.0 / 3.0            # 0.0 unless mixed mode (then 3-1 pays for its 3x width jump)
        errs = []; err = 0.0
        while len(rows) < max_rows:
            nr, nf = [], []
            for c in rows[-1]:
                d, g = down_of(M, c)
                if d is None: break
                nr.append(d); nf.append(g)
            if len(nr) != a: break
            if not all(nr[k][1] == nr[k + 1][0] and nr[k][2] == nr[k + 1][3] for k in range(a - 1)): break
            if len(fset.union(nf)) != len(fset) + len(nf): break
            rows.append(nr); faces.append(nf); fset.update(nf)
            G.append([c[3] for c in nr] + [nr[-1][2]])
            y = len(rows) - 1                                  # vertex row y now has a cell row below it
            ok = True
            for x in range(1, a):
                v = G[y][x]
                if v in s.fv or v in s.corner or v in s.bv or val(v) != 4: ok = False; break
            if ok:
                for x in (1, 2):
                    if _ekey(G[y - 1][x], G[y][x]) in s.feat: ok = False
            if not ok: break
            p0, p1 = M.V[G[y][0]], M.V[G[y][a]]
            for x in (1, 2):
                t = x / a; d3 = M.V[G[y][x]] - ((1 - t) * p0 + t * p1)
                err = max(err, math.sqrt(float(d3[0] * d3[0] + d3[1] * d3[1] + d3[2] * d3[2])))
            errs.append(err)
            if tol is not None and err + s.size_w * width > tol: break
        Lmax = len(rows)
        if Lmax < min_rows: return None
        L = min(len(errs) + 1, Lmax)
        while L >= min_rows:
            bad = any(_ekey(G[L - 1][x], G[L][x]) in s.feat for x in (1, 2))
            e = errs[L - 2]; adm = e + s.size_w * width; cost = adm + jump   # admitted on error, ranked with jump
            if not bad and (tol is None or adm <= tol): return rows, faces, L, e, cost
            L -= 1
        return None

    def _grow_eval_wide(s, row, rf, a, max_rows, min_rows, tol):
        """grow_eval for a = 4..6: every interior vertex line is either removed or re-spaced, so all of them must be
        regular, unprotected and smooth; the error is the distance of every original row vertex from the re-spaced
        row (kept lines evenly spread along the original row)."""
        M = s.M; val = s.val; b = a - 2
        i0, t0, j0, u0 = _wide_weights(a)
        rows, faces, fset = [row], [rf], set(rf)
        G = [[c[0] for c in row] + [row[-1][1]], [c[3] for c in row] + [row[-1][2]]]
        width = float(np.linalg.norm(M.V[G[1][0]] - M.V[G[1][a]]))
        jump = s.jump_w * width * 2.0 / (a * b)
        errs = []; err = 0.0
        while len(rows) < max_rows:
            nr, nf = [], []
            for c in rows[-1]:
                d, g = down_of(M, c)
                if d is None: break
                nr.append(d); nf.append(g)
            if len(nr) != a: break
            if not all(nr[k][1] == nr[k + 1][0] and nr[k][2] == nr[k + 1][3] for k in range(a - 1)): break
            if len(fset.union(nf)) != len(fset) + len(nf): break
            rows.append(nr); faces.append(nf); fset.update(nf)
            G.append([c[3] for c in nr] + [nr[-1][2]])
            y = len(rows) - 1
            ok = True
            for x in range(1, a):
                v = G[y][x]
                if v in s.fv or v in s.corner or v in s.bv or val(v) != 4 or _ekey(G[y - 1][x], G[y][x]) in s.feat:
                    ok = False; break
            if not ok: break
            P = M.V[G[y]]
            newp = P[i0] * (1 - t0) + P[i0 + 1] * t0                      # kept lines spread evenly along the row
            q = newp[j0] * (1 - u0) + newp[j0 + 1] * u0                  # where each original vertex now lies
            d3 = P[1:a] - q; err = max(err, math.sqrt(float(np.max(np.einsum('ij,ij->i', d3, d3)))))
            errs.append(err)
            if tol is not None and err + s.size_w * width > tol: break
        Lmax = len(rows)
        if Lmax < min_rows: return None
        L = min(len(errs) + 1, Lmax)
        while L >= min_rows:
            bad = any(_ekey(G[L - 1][x], G[L][x]) in s.feat for x in range(1, a))
            e = errs[L - 2]; adm = e + s.size_w * width; cost = adm + jump   # admitted on error, ranked with jump
            if not bad and (tol is None or adm <= tol): return rows, faces, L, e, cost
            L -= 1
        return None

    def apply_wide(s, rows, faces, L):
        """Refill an a-wide block (a = 4..6): bands with a centred 3-1 (a odd) or symmetric 4-2 (a even), middle rows
        with b = a-2 columns spread evenly along each row.  Quads only; every new face must pass the quality floor."""
        M = s.M; a = len(rows[0]); b = a - 2; rows = rows[:L]; faces = faces[:L]; G = s.vgrid(rows)
        band, c0, rem, _ = pattern(a)
        keep = [x for x in range(a + 1) if x not in rem]
        allf = [g for r in faces for g in r]
        normals = {g: face_normal(M.V[M.F[g]]) for g in allf}
        nmean = np.mean(list(normals.values()), axis=0); nmean /= np.linalg.norm(nmean) + 1e-30
        uvcell = {}
        if s.FUV is not None:
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    for v in cell: uvcell[v] = s._uv_of(faces[r][c], v)
        M.begin(); new_faces = []; new_uv = {}; n_uv0 = len(s.UV) if s.FUV is not None else 0
        V0 = {}                                         # original positions of the block (for the error transfer)

        def lerp_v(x0, y0, x1, y1, t):
            return (1 - t) * M.V[G[y0][x0]] + t * M.V[G[y1][x1]]

        def newv(xa, ya, xb, yb, t):
            p = lerp_v(xa, ya, xb, yb, t)
            if s.proj is not None: p = s.proj.surface(p)
            v = M.add_vertex(p)
            if s.FUV is not None:
                s.UV.append((1 - t) * s.UV[uvi(G[ya][xa])] + t * s.UV[uvi(G[yb][xb])]); new_uv[v] = len(s.UV) - 1
            if s.QL is not None: s.QL.append(np.zeros((4, 4))); s.WL.append(0.0)
            return v

        def uvi(v):
            return new_uv[v] if v in new_uv else uvcell[v]

        def face(q):
            g = M.add(q); new_faces.append(g)
            if s.FUV is not None:
                while len(s.FUV) <= g: s.FUV.append(None)
                s.FUV[g] = [uvi(v) for v in q]
        # 1. re-space the kept interior lines of vertex rows 1..L-1 along their original rows (positions + UVs)
        moved = []
        for y in range(1, L):
            P = M.V[G[y]].copy(); V0[y] = P
            U = np.array([s.UV[uvcell[v]] for v in G[y]]) if s.FUV is not None else None
            for i, x in enumerate(keep[1:-1], start=1):
                v = G[y][x]; sp = i * a / b; p = _polyline_at(P, sp)
                if s.proj is not None: p = s.proj.surface(p)
                M.move(v, p); moved.append(v)
                if U is not None:
                    s.UV.append(_polyline_at(U, sp)); new_uv[v] = len(s.UV) - 1
        for g in allf: M.kill(g)
        bandv = []

        def pass_through(y, xs):
            for x in xs: face([G[y][x], G[y][x + 1], G[y + 1][x + 1], G[y + 1][x]])
        # 2. top band (cell row 0)
        if band == '31':
            M1 = newv(c0 + 1, 0, c0 + 1, 1, 0.6); M2 = newv(c0 + 2, 0, c0 + 2, 1, 0.6); bandv += [M1, M2]
            T0, T1, T2, T3 = (G[0][c0 + k] for k in range(4)); P_, Q_ = G[1][c0], G[1][c0 + 3]
            pass_through(0, range(0, c0))
            face([T0, T1, M1, P_]); face([T1, T2, M2, M1]); face([T2, T3, Q_, M2]); face([M1, M2, Q_, P_])
            pass_through(0, range(c0 + 3, a))
        else:
            p1, p2, p3 = (newv(c0 + k, 0, c0 + k, 1, 0.5) for k in (1, 2, 3)); bandv += [p1, p2, p3]
            T = [G[0][c0 + k] for k in range(5)]; B0, B1, B2 = G[1][c0], G[1][c0 + 2], G[1][c0 + 4]
            pass_through(0, range(0, c0))
            face([T[0], T[1], p1, B0]); face([T[1], T[2], p2, p1]); face([T[2], T[3], p3, p2]); face([T[3], T[4], B2, p3])
            face([p1, p2, B1, B0]); face([p2, p3, B2, B1])
            pass_through(0, range(c0 + 4, a))
        # 3. middle rows
        for y in range(1, L - 1):
            for i in range(len(keep) - 1):
                xa, xb = keep[i], keep[i + 1]; face([G[y][xa], G[y][xb], G[y + 1][xb], G[y + 1][xa]])
        # 4. bottom band (cell row L-1)
        y = L - 1
        if band == '31':
            M3 = newv(c0 + 1, y, c0 + 1, y + 1, 0.4); M4 = newv(c0 + 2, y, c0 + 2, y + 1, 0.4); bandv += [M3, M4]
            P2, Q2 = G[y][c0], G[y][c0 + 3]; B = [G[y + 1][c0 + k] for k in range(4)]
            pass_through(y, range(0, c0))
            face([P2, Q2, M4, M3]); face([P2, M3, B[1], B[0]]); face([M3, M4, B[2], B[1]]); face([M4, Q2, B[3], B[2]])
            pass_through(y, range(c0 + 3, a))
        else:
            q1, q2, q3 = (newv(c0 + k, y, c0 + k, y + 1, 0.5) for k in (1, 2, 3)); bandv += [q1, q2, q3]
            A0, A1, A2 = G[y][c0], G[y][c0 + 2], G[y][c0 + 4]; B = [G[y + 1][c0 + k] for k in range(5)]
            pass_through(y, range(0, c0))
            face([A0, A1, q2, q1]); face([A1, A2, q3, q2])
            face([A0, q1, B[1], B[0]]); face([q1, q2, B[2], B[1]]); face([q2, q3, B[3], B[2]]); face([q3, A2, B[4], B[3]])
            pass_through(y, range(c0 + 4, a))
        # 5. relax band vertices and the re-spaced lines next to them (surface-projected)
        for _ in range(3):
            for v in bandv + [G[yy][x] for yy in (1, L - 1) for x in keep[1:-1]]:
                nb = list(M.nbrs(v)); p = 0.5 * M.V[v] + 0.5 * M.V[nb].mean(0)
                M.move(v, s.proj.surface(p) if s.proj is not None else p)
        okn = {g: nmean for g in new_faces}
        if not M.ok_local(new_faces, okn, s.sj_floor):
            M.rollback()
            if s.FUV is not None: del s.UV[n_uv0:]
            return False
        if s.QL is not None:                           # removed vertices hand their quadrics to the nearest kept line
            for yy in range(1, L):
                for x in rem:
                    v = G[yy][x]
                    if yy == 1 or yy == L - 1:
                        bv = bandv[:len(bandv) // 2] if yy == 1 else bandv[len(bandv) // 2:]
                        tg = min(bv, key=lambda u: float(np.linalg.norm(M.V[u] - V0[yy][x])))
                    else:
                        tg = G[yy][min(keep, key=lambda k: abs(keep.index(k) * a / b - x))]
                    s.QL[tg] = s.QL[tg] + s.QL[v]; s.WL[tg] = s.WL[tg] + s.WL[v]
        M.commit(); return True

    def vgrid(s, rows):
        a = len(rows[0]); L = len(rows)
        G = [[None] * (a + 1) for _ in range(L + 1)]
        for r, row in enumerate(rows):
            for c, (tl, tr, br, bl) in enumerate(row):
                G[r][c], G[r][c + 1], G[r + 1][c + 1], G[r + 1][c] = tl, tr, br, bl
        return G

    def evaluate(s, rows, faces, min_rows=4, tol=None):
        """Longest prefix L >= min_rows whose removed vertices (lines x=1,2 of a 3-wide block, vertex rows 1..L-1)
        are smooth, regular, unprotected and within tol of their row's straight line.  Returns (L, err, cost)."""
        M = s.M; a = len(rows[0]); G = s.vgrid(rows); Lmax = len(rows)
        if Lmax < min_rows or a != 3: return None
        errs = []; err = 0.0
        for y in range(1, Lmax):
            ok = True
            for x in range(1, a):
                v = G[y][x]
                if v in s.fv or v in s.corner or v in s.bv or M.valence(v) != 4: ok = False; break
            if ok:
                for x in (1, 2):
                    if _ekey(G[y - 1][x], G[y][x]) in s.feat: ok = False
            if not ok: break
            p0, p1 = M.V[G[y][0]], M.V[G[y][a]]
            for x in (1, 2):
                t = x / a; err = max(err, float(np.linalg.norm(M.V[G[y][x]] - ((1 - t) * p0 + t * p1))))
            errs.append(err)
        L = min(len(errs) + 1, Lmax)
        while L >= min_rows:                          # bottom band's vertical edges + tolerance
            bad = any(_ekey(G[L - 1][x], G[L][x]) in s.feat for x in (1, 2))
            width = float(np.linalg.norm(M.V[G[1][0]] - M.V[G[1][a]]))
            e = errs[L - 2]; cost = e + s.size_w * width
            if not bad and (tol is None or cost <= tol): return L, e, cost
            L -= 1
        return None

    def _uv_of(s, g, v):
        f = s.M.F[g]; return s.FUV[g][f.index(v)]

    def apply(s, rows, faces, L):
        """Refill rows[0:L] with top band / middle rows / bottom band.  Returns True or False (rolled back)."""
        M = s.M; a = len(rows[0]); rows = rows[:L]; faces = faces[:L]; G = s.vgrid(rows)
        allf = [g for r in faces for g in r]
        normals = {g: face_normal(M.V[M.F[g]]) for g in allf}
        nmean = np.mean(list(normals.values()), axis=0); nmean /= np.linalg.norm(nmean) + 1e-30
        uvcell = {}
        if s.FUV is not None:                          # UV index of every grid vertex, as seen from the block
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    g = faces[r][c]
                    for v in cell: uvcell[v] = s._uv_of(g, v)

        def lerp_v(x0, y0, x1, y1, t):
            return (1 - t) * M.V[G[y0][x0]] + t * M.V[G[y1][x1]]

        def lerp_uv(x0, y0, x1, y1, t):
            return (1 - t) * s.UV[uvcell[G[y0][x0]]] + t * s.UV[uvcell[G[y1][x1]]]

        M.begin(); new_faces = []; new_uv = {}; new_q = []
        m = a // 2                                     # the 3-1 sits on columns m-1, m, m+1 (a >= 3)
        c0 = m - 1

        def newv(xa, ya, xb, yb, t):
            p = lerp_v(xa, ya, xb, yb, t)
            if s.proj is not None: p = s.proj.surface(p)
            v = M.add_vertex(p)
            if s.FUV is not None:
                s.UV.append(lerp_uv(xa, ya, xb, yb, t)); new_uv[v] = len(s.UV) - 1
            if s.QL is not None: s.QL.append(np.zeros((4, 4))); s.WL.append(0.0); new_q.append(v)
            return v

        def uvi(v):
            return new_uv[v] if v in new_uv else uvcell[v]

        def face(q):
            g = M.add(q); new_faces.append(g)
            if s.FUV is not None:
                while len(s.FUV) <= g: s.FUV.append(None)
                s.FUV[g] = [uvi(v) for v in q]
        for g in allf: M.kill(g)
        # columns kept in the middle rows: 0..c0, then c0+3..a  (lines c0+1 and c0+2 are removed)
        keep_x = list(range(0, c0 + 1)) + list(range(c0 + 3, a + 1))
        # top band (row 0)
        M1 = newv(c0 + 1, 0, c0 + 1, 1, 0.6); M2 = newv(c0 + 2, 0, c0 + 2, 1, 0.6)
        for x in range(0, c0): face([G[0][x], G[0][x + 1], G[1][x + 1], G[1][x]])
        T0, T1, T2, T3 = G[0][c0], G[0][c0 + 1], G[0][c0 + 2], G[0][c0 + 3]; P, Q = G[1][c0], G[1][c0 + 3]
        face([T0, T1, M1, P]); face([T1, T2, M2, M1]); face([T2, T3, Q, M2]); face([M1, M2, Q, P])
        for x in range(c0 + 3, a): face([G[0][x], G[0][x + 1], G[1][x + 1], G[1][x]])
        # middle rows
        for y in range(1, L - 1):
            for i in range(len(keep_x) - 1):
                xa, xb = keep_x[i], keep_x[i + 1]
                face([G[y][xa], G[y][xb], G[y + 1][xb], G[y + 1][xa]])
        # bottom band (row L-1)
        y = L - 1
        M3 = newv(c0 + 1, y, c0 + 1, y + 1, 0.4); M4 = newv(c0 + 2, y, c0 + 2, y + 1, 0.4)
        for x in range(0, c0): face([G[y][x], G[y][x + 1], G[y + 1][x + 1], G[y + 1][x]])
        P2, Q2 = G[y][c0], G[y][c0 + 3]; B0, B1, B2, B3 = G[y + 1][c0], G[y + 1][c0 + 1], G[y + 1][c0 + 2], G[y + 1][c0 + 3]
        face([P2, Q2, M4, M3]); face([P2, M3, B1, B0]); face([M3, M4, B2, B1]); face([M4, Q2, B3, B2])
        for x in range(c0 + 3, a): face([G[y][x], G[y][x + 1], G[y + 1][x + 1], G[y + 1][x]])
        # relax the 4 band vertices a little inside their bands (keeps the pattern quads convex)
        for v in (M1, M2, M3, M4):
            nb = list(M.nbrs(v)); p = 0.5 * M.V[v] + 0.5 * M.V[nb].mean(0)
            M.V[v] = s.proj.surface(p) if s.proj is not None else p
        okn = {g: nmean for g in new_faces}
        if not M.ok_local(new_faces, okn, 0.05):
            M.rollback()
            if s.FUV is not None:
                for _ in new_uv: s.UV.pop()
            # (orphan vertices stay in M.V after a rollback, so their zero quadrics stay too: indices remain aligned)
            return False
        if s.QL is not None:                           # removed vertices hand their quadrics to their replacements
            for y in range(1, L):
                for x in (1, 2):
                    v = G[y][x]
                    if y == 1: tg = M1 if x == 1 else M2
                    elif y == L - 1: tg = M3 if x == 1 else M4
                    else: tg = G[y][0] if x == 1 else G[y][3]
                    s.QL[tg] = s.QL[tg] + s.QL[v]; s.WL[tg] = s.WL[tg] + s.WL[v]
        M.commit(); return True


def seam_edges(F, FUV):
    em = {}
    for f, fu in zip(F, FUV):
        for k in range(len(f)):
            e = _ekey(f[k], f[(k + 1) % len(f)])
            em.setdefault(e, []).append({f[k]: fu[k], f[(k + 1) % len(f)]: fu[(k + 1) % len(f)]})
    return {e for e, Ls in em.items() if len(Ls) == 2 and Ls[0] != Ls[1]}


def coarsen_strips(V, F, target, feat_deg=8.0, extra_feat=(), proj=None, FUV=None, UV=None, max_error=None,
                   planes=None, a=3, min_rows=4, size_w=0.03, stats=None, max_rows=24, quadrics=None,
                   widths=None, jump_w=0.0, sj_floor=None):
    """Apply local strip coarsening (cheapest first, disjoint per pass) until target faces or max_error.
    widths: block widths to try -- (3,) = 3-1 strips (round 7); 4..6 add the gentler 4-2 / 5-3 / 6-4 transitions."""
    widths = tuple(widths) if widths else (a,)
    M = Edit(V, F)
    st = stats if stats is not None else {}
    st.setdefault('strips', 0); st.setdefault('strip_pairs', 0); st.setdefault('strip_widths', {})
    FUVl = None if FUV is None else [list(x) for x in FUV]
    UVl = None if UV is None else [np.asarray(u, float) for u in UV]
    QL = None if quadrics is None else [q.copy() for q in quadrics[0]]
    WL = None if quadrics is None else [float(w) for w in quadrics[1]]
    feat = feature_edges(M.V, M.F, feat_deg) | {_ekey(int(x), int(y)) for x, y in (extra_feat or ())}
    fd = {}
    for x, y in feat: fd[x] = fd.get(x, 0) + 1; fd[y] = fd.get(y, 0) + 1
    corner = {v for v, d in fd.items() if d != 2}
    while sum(M.alive) > target:
        bv = {v for v in M.vf if M.faces_of(v) and M.is_boundary(v)}
        S = Strips(M, feat, corner, bv, proj, FUVl, UVl, size_w); S.QL, S.WL = QL, WL; S.jump_w = jump_w
        if sj_floor is not None: S.sj_floor = sj_floor
        mir = [SYM.mirror_map(M.V, pl) for pl in planes] if planes else []
        cands = {}
        for g in range(len(M.F)):
            if not M.alive[g] or len(M.F[g]) != 4: continue
            for rot in (0, 1):
              v_in = cell_of(M, g, rot)[2]                # the block's first interior vertex: every width needs it
              if v_in in S.fv or v_in in S.corner or v_in in S.bv or S.val(v_in) != 4: continue
              for aw in widths:
                r = S.grow_eval(g, rot, aw, max_rows, min_rows, max_error)
                if r is None: continue
                rows, faces, L, err, cost = r
                key = frozenset(v for row in rows[:L] for c in row for v in c)
                gain = 2 * L - pattern(aw)[3]
                if gain <= 0: continue
                if key not in cands or cands[key][0] > cost / gain:
                    cands[key] = (cost / gain, rows, faces, L)
        if not cands: break
        S._val = None                                  # the mesh changes from here on
        order = sorted(cands.items(), key=lambda kv: kv[1][0])
        touched, applied = set(), 0
        for key, (c, rows, faces, L) in order:
            if sum(M.alive) <= target: break
            if key & touched: continue
            group = [(key, rows, faces, L)]
            if mir:
                mk = None
                ok = True
                for m in mir:
                    mm = frozenset(int(m[v]) for v in key)
                    if any(m[v] < 0 for v in key): continue
                    if mm == key: continue
                    if mm in cands: group.append((mm,) + tuple(cands[mm][1:]))
                    else: ok = False
                if not ok: continue
                if any(k2 & touched for k2, *_ in group[1:]): continue
                if len(group) > 1 and any(group[i][0] & group[j][0] for i in range(len(group)) for j in range(i + 1, len(group))):
                    continue
            done = 0
            for k2, rows2, faces2, L2 in group:
                if not all(M.alive[g] for r in faces2[:L2] for g in r): break
                aw = len(rows2[0])
                okk = S.apply(rows2, faces2, L2) if aw == 3 else S.apply_wide(rows2, faces2, L2)
                if okk:
                    done += 1; touched |= k2 | {x for v in k2 for x in M.nbrs(v)}
                    st['strip_widths'][aw] = st['strip_widths'].get(aw, 0) + 1
                else: break
            if done:
                applied += done; st['strips'] += done
                if done > 1: st['strip_pairs'] += 1
        if applied == 0: break
    keep = [g for g in range(len(M.F)) if M.alive[g]]
    Fo = [M.F[g] for g in keep]
    used = sorted({v for f in Fo for v in f}); rm = {v: i for i, v in enumerate(used)}
    Fo = [[rm[v] for v in f] for f in Fo]
    st['feat_out'] = [(rm[x], rm[y]) for x, y in feat if x in rm and y in rm]
    if QL is not None: st['Q_out'] = (np.array([QL[v] for v in used]), np.array([WL[v] for v in used]))
    if FUV is not None:
        U2 = [FUVl[g] for g in keep]; uu = sorted({x for f in U2 for x in f}); um = {x: i for i, x in enumerate(uu)}
        return M.V[used], Fo, [[um[x] for x in f] for f in U2], np.asarray([UVl[x] for x in uu]), st
    return M.V[used], Fo, None, None, st


def adaptive_simplify(V, F, target, UV=None, FUV=None, feature_deg=8.0, symmetry='none', local='adaptive',
                      order_hint=False, tol0_rel=2e-4, grow=1.6, max_rel=0.05, extra_feat=None, log=None, ring_margin=3.0,
                      cost_mode=None, max_error=None, placement='qem', transitions='31'):
    """cost_mode: ring ranking for 'off'/'stuck' ('width' default = hard-surface behaviour of round 5); 'adaptive'
    always ranks by estimated error.  max_error: absolute cap (stops early)."""
    """Error ladder over global ring removal and local strips.  local: 'off' | 'stuck' | 'adaptive'.
    transitions: '31' (3-1 strips only, rounds 7-9) | 'gentle' (4-2, 5-3, 6-4 only) | 'mixed' (gentle patterns
    first in every strip round, 3-1 only for what they cannot reach)."""
    if transitions == 'auto':                       # measured (round 10): gentle-first wins at moderate targets,
        transitions = 'mixed' if target >= 0.3 * len(F) else '31'   # 3-1 keeps error lowest at aggressive ones
    info_tr = transitions
    TW = {'31': [dict(widths=(3,))], 'gentle': [dict(widths=(5, 4, 6), sj_floor=0.2)],
          'mixed': [dict(widths=(5, 4, 6), sj_floor=0.2), dict(widths=(3,))]}[transitions]   # gentle first, 3-1 fallback
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    Q, kw, info = prepare(V, F, UV, FUV, order_hint, feature_deg, True, symmetry, True, extra_feat)
    proj = kw['proj']; planes = None
    if symmetry != 'none' and info.get('symmetry_planes'):
        planes = [(np.asarray(p['normal']), p['offset']) for p in info['symmetry_planes']]
    curV, curF = V, Q
    curFUV, curUV = kw.get('FUV'), kw.get('UV')
    st = dict(rings=0, strips=0, strip_pairs=0, levels=0)
    tol = tol0_rel * diag

    featE = set(map(tuple, kw.get('extra_feat', []))) | feature_edges(V, Q, feature_deg)
    curQ = vertex_quadrics(V, Q) if placement == 'qem' else None

    def ring_round(max_err):
        nonlocal curV, curF, curFUV, curUV, featE, curQ
        k = dict(feat_deg=180.0, proj=proj, extra_feat=list(featE))     # features carried from the input
        if curQ is not None: k.update(placement='qem', quadrics=curQ)
        if curFUV is not None: k.update(FUV=curFUV, UV=curUV)
        if planes: k['mirrors'] = [SYM.mirror_map(curV, pl) for pl in planes]
        s2 = {}
        cm = 'error' if local == 'adaptive' else (cost_mode or 'width')
        res = ring_collapse(curV, curF, target, stats=s2, cost_mode=cm, max_error=max_err, **k)
        if curFUV is not None: curV, curF, curFUV, curUV, n = res
        else: curV, curF, n = res
        featE = set(map(tuple, s2['feat_out']))
        if curQ is not None: curQ = s2['Q_out']
        st['rings'] += n; st['_next_ring'] = s2.get('next_cost', 0.0); return n

    def strip_round(max_err):
        nonlocal curV, curF, curFUV, curUV, featE, curQ
        n = 0
        for tw in TW:                                  # mixed: gentle patterns first, 3-1 only where none fits
            if len(curF) <= target: break
            s2 = {}
            V2, F2, FU2, U2, s2 = coarsen_strips(curV, curF, target, 180.0, list(featE), proj, curFUV, curUV, max_err,
                                                  planes, stats=s2, quadrics=curQ, **tw)
            n += s2['strips']; st['strips'] += s2['strips']; st['strip_pairs'] += s2['strip_pairs']
            for k_, v_ in s2.get('strip_widths', {}).items():
                st.setdefault('strip_widths', {})[k_] = st.get('strip_widths', {}).get(k_, 0) + v_
            curV, curF = V2, F2
            featE = set(map(tuple, s2['feat_out']))
            if curQ is not None: curQ = s2['Q_out']
            if curFUV is not None: curFUV, curUV = FU2, U2
        return n

    if local == 'off':
        ring_round(max_error)
    elif local == 'stuck':
        ring_round(max_error)
        if len(curF) > target: strip_round(max_error)
        if len(curF) > target: ring_round(max_error)
    else:
        cap = max_rel * diag if max_error is None else max_error
        while len(curF) > target and tol <= cap:
            st['levels'] += 1
            n1 = ring_round(tol)
            # strips only where whole rings cannot follow soon: they add poles and can block later rings
            nr = st.get('_next_ring', 0.0)
            stol = tol if nr == float('inf') else min(tol, nr / ring_margin)
            n2 = strip_round(stol) if len(curF) > target and stol > 0 else 0
            if log: print(f'tol {tol / diag * 100:.4f}% rings {n1} strips {n2} faces {len(curF)}', file=log, flush=True)
            if n1 == 0 and n2 == 0: tol *= grow
    st.pop('_next_ring', None)
    info.update(st); info['transitions'] = info_tr; info['faces'] = len(curF); info['quads'] = sum(1 for f in curF if len(f) == 4)
    return curV, curF, curFUV, curUV, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('target', type=int)
    ap.add_argument('--local', default='adaptive', choices=['off', 'stuck', 'adaptive'])
    ap.add_argument('--feature-deg', type=float, default=8.0); ap.add_argument('--symmetry', default='none')
    ap.add_argument('--order-hint', action='store_true'); ap.add_argument('--ignore-uv', action='store_true')
    ap.add_argument('--verbose', action='store_true'); ap.add_argument('--cost', default=None, choices=['width', 'error', 'qem'])
    ap.add_argument('--placement', default='qem', choices=['surface', 'qem'])
    ap.add_argument('--transitions', default='31', choices=['31', 'gentle', 'mixed', 'auto'])
    a = ap.parse_args(); t0 = time.time()
    V, F, UV, FUV = read_obj_uv(a.inp)
    if a.ignore_uv: UV = FUV = None
    V2, F2, FU2, U2, info = adaptive_simplify(V, F, a.target, UV, FUV, a.feature_deg, a.symmetry, a.local, a.order_hint,
                                              log=sys.stderr if a.verbose else None, cost_mode=a.cost,
                                              placement=a.placement, transitions=a.transitions)
    write_obj_uv(a.out, V2, F2, U2, FU2)
    info['seconds'] = round(time.time() - t0, 2); info.pop('symmetry_planes', None)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
