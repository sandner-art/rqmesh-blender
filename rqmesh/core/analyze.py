# SPDX-License-Identifier: Apache-2.0
"""Structure introspection -> method choice (the introsort / Timsort idea applied to retopology).

Hybrid sorts look at the input first (Timsort: existing runs; introsort: recursion depth as a sign of a bad pivot
sequence) and switch to the algorithm that fits it.  For simplification / retopology the "runs" are the modelling
structure: quads, regular vertices and clean edge loops.  Cheap scores, measured in one pass:

  quad_share    fraction of faces that are quads after rebuilding quads from triangles (export order first)
  regular_share fraction of interior vertices with valence 4
  loop_share    fraction of quad sides that lie on clean edge loops (no triangle, no self-crossing, no revisit)
  loop_len      median clean-loop length relative to sqrt(#faces)
  block_share   fraction of quads whose four corners are all regular (where 3-1 strips can work)
  subdiv_levels levels of Catmull-Clark / simple subdivision the mesh carries (topology test, confirmed by fitting
                one level: residual in fine edge lengths); such meshes go to Recover Base Cage first

recommend() turns them into a method, the way introsort turns a depth counter into a switch:
  * edge loops (+ transitions when stuck)  -- quad-modelled assets with long clean loops
  * adaptive loops + transitions           -- quad-modelled but loops are short/blocked (organic, partly symmetric)
  * QEM decimation (triangles) or a field-aligned remesh -- little or no loop structure (scans, iso-remeshes,
    meshes subdivided from triangles, remesher output with spiralling loops)

  python analyze.py MESH.obj  -> JSON
"""
import sys, os, json, math
import numpy as np
try:
    from .meshio_min import read_obj
    from .structsimp import tris_to_quads, Mesh, chords
    from . import subdiv as SD
    from . import subdiv_tri as STRI
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj
    from structsimp import tris_to_quads, Mesh, chords
    import subdiv as SD
    import subdiv_tri as STRI


def structure_scores(V, F, order_hint=True):
    V = np.asarray(V, float)
    T = [f for f in F if len(f) == 3]
    tri_at = {}
    for f in T:
        for v in f: tri_at[v] = tri_at.get(v, 0) + 1
    fans = sum(1 for v, c in tri_at.items() if c >= 5)
    ngons = sum(1 for f in F if len(f) > 4)
    Q = (tris_to_quads(V, T, order_hint=order_hint) + [f for f in F if len(f) != 3]) if T else [list(f) for f in F]
    nq = sum(1 for f in Q if len(f) == 4)
    M = Mesh(V, Q); em = M.edges()
    bnd = {v for e, fs in em.items() if len(fs) == 1 for v in e}
    val = {}
    for a, b in em: val[a] = val.get(a, 0) + 1; val[b] = val.get(b, 0) + 1
    interior = [v for v in val if v not in bnd]
    regular = sum(1 for v in interior if val[v] == 4)
    clean_sides, lens = 0, []
    for edges, faces, closed in chords(M, em):
        ok = faces and all(f >= 0 for f in faces) and len(set(faces)) == len(faces)
        if ok:
            verts = [v for e in edges for v in e]
            ok = len(set(verts)) == len(verts)
        if ok: clean_sides += len(faces); lens.append(len(faces))
    blocks = sum(1 for f in Q if len(f) == 4 and all(val.get(v, 0) == 4 and v not in bnd for v in f))
    n = max(len(Q), 1)
    sub = subdiv_scores(V, F)
    return dict(sub, faces=len(Q), quad_share=round(nq / n, 4), regular_share=round(regular / max(len(interior), 1), 4),
                loop_share=round(clean_sides / max(2 * nq, 1), 4),
                loop_len=round(float(np.median(lens)) / math.sqrt(n), 3) if lens else 0.0,
                block_share=round(blocks / max(nq, 1), 4), fans=fans, ngons=ngons)


def subdiv_scores(V, F, tol=0.05):
    Q, how = SD.to_quads_if_subdivided(V, [list(f) for f in F])
    if Q is None:
        if F and all(len(f) == 3 for f in F):        # triangle subdivision (Loop / Blender Subdivide)
            r = STRI.unsubdivide_tri_once(V, [list(f) for f in F])
            if r is not None and r['residual_rel'] <= tol:
                return dict(subdiv_levels=1, subdiv_scheme='tri-' + r['scheme'], subdiv_residual=round(r['residual_rel'], 6))
        return dict(subdiv_levels=0)
    lv = SD.detect_levels(Q, len(V))
    if lv == 0: return dict(subdiv_levels=0)
    r = SD.unsubdivide_once(V, Q)
    if r is None or r['residual_rel'] > tol:
        return dict(subdiv_levels=0, subdiv_topology_levels=lv, subdiv_residual=None if r is None else round(r['residual_rel'], 5))
    return dict(subdiv_levels=lv, subdiv_scheme=r['scheme'], subdiv_residual=round(r['residual_rel'], 6))


def recommend(s):
    """Rules checked against every case measured in rounds 5-9 (see RESULTS-round9)."""
    if s.get('subdiv_levels', 0) > 0:
        return 'recover_base_first', (f"subdivided (up to {s['subdiv_levels']} level(s); first level {s.get('subdiv_scheme')}, residual "
                                      f"{s.get('subdiv_residual')}): Recover Base Cage first - the exact cage - then "
                                      f"re-analyze the cage")
    if s.get('fans', 0) + s.get('ngons', 0) > 0 and s['faces'] < 20000:
        m, why = recommend(dict(s, fans=0, ngons=0))
        return 'quad_caps_first', (f"{s['fans']} triangle fan(s) and {s['ngons']} n-gon(s): run Quad Caps first, "
                                   f"then re-analyze (now: {why})")
    if s['quad_share'] < 0.9:
        return 'decimate_or_remesh', "few quads: not quad-modelled (scan, triangle export, sculpt)"
    if s['loop_share'] < 0.3 or s['loop_len'] < 0.15:
        return 'decimate_or_remesh', "quads without usable edge loops (iso-remesh, subdivided triangles)"
    if s['regular_share'] >= 0.95 and s['loop_share'] >= 0.9:
        return 'edge_loops', "quad-modelled with clean edge loops: remove whole loops, 3-1 transitions only when stuck"
    return 'adaptive_loops', ("quad-modelled but loops are blocked or poles are many: loops + 3-1 transitions on an "
                              "error ladder (expect 2-5x Decimate's geometric error, but quads, UVs and symmetry kept)")


def main():
    V, F = read_obj(sys.argv[1])
    s = structure_scores(V, F); m, why = recommend(s)
    s.update(method=m, why=why); print(json.dumps(s))


if __name__ == '__main__':
    main()
