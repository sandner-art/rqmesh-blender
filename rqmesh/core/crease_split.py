# SPDX-License-Identifier: Apache-2.0
"""Crease-wrapped quad fix.  Round-4 anatomy: ~91% of the folded quads left by field-aligned remeshing are quads
whose DIAGONAL lies on a sharp input crease (median bend ~140 deg): the quantized layout has a feature line cutting
a cell corner-to-corner.  No vertex smoothing can fix that (the quad is non-planar by construction), and an edge
rotation inside two quads cannot place the crease on an edge either (the diagonal endpoints are never opposite in
the hexagon), so this pass makes the minimal topological change: split such a quad into two triangles along the
crease diagonal.  For game assets this is exactly what export triangulation must do anyway (a quad folded across
a crease shades correctly only if triangulated along that diagonal).

  python crease_split.py REF_TRI.obj IN_QUAD.obj OUT.obj [--min-bend 40]  -> JSON line
"""
import sys, os, json, math, argparse
import numpy as np
try:
    from .meshio_min import read_obj, write_obj
    from .compat import BallIndex, knn
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj, write_obj
    from compat import BallIndex, knn


def scaled_jacobian_min(P):
    n = np.cross(P[2] - P[0], P[3] - P[1]); ln = np.linalg.norm(n)
    if ln < 1e-20: return -1.0
    n /= ln; out = 1.0
    for k in range(4):
        e1 = P[(k + 1) % 4] - P[k]; e2 = P[k - 1] - P[k]; l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if l1 < 1e-20 or l2 < 1e-20: return -1.0
        out = min(out, np.dot(np.cross(e1, e2), n) / (l1 * l2))
    return out


def tri_n(a, b, c):
    n = np.cross(b - a, c - a); return n / (np.linalg.norm(n) + 1e-30)


def split_folded(Vr, Tr, V, F, min_bend=40.0):
    """Split folded quads along the diagonal whose triangles agree with the reference surface normals."""
    Vr = np.asarray(Vr, float); Tr = np.asarray(Tr)
    Cr = Vr[Tr].mean(1); Nr = np.cross(Vr[Tr[:, 1]] - Vr[Tr[:, 0]], Vr[Tr[:, 2]] - Vr[Tr[:, 0]])
    Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-30; kd = BallIndex(Cr)
    out = []; stats = dict(quads=0, folded=0, split_crease=0, split_other=0, kept=0)
    for f in F:
        if len(f) != 4: out.append(list(f)); continue
        stats['quads'] += 1; P = V[f]
        if scaled_jacobian_min(P) > 0: out.append(list(f)); continue
        stats['folded'] += 1; best = None
        for d in (0, 1):                                   # diagonal f[d]-f[d+2]
            t1 = [f[d], f[d + 1], f[(d + 2) % 4]]; t2 = [f[d], f[(d + 2) % 4], f[(d + 3) % 4]]
            n1, n2 = tri_n(*V[t1]), tri_n(*V[t2]); s = 1e9
            for t, n in ((t1, n1), (t2, n2)):          # near a crease either side's input normal is acceptable
                s = min(s, float(np.max(Nr[knn(kd, V[t].mean(0), 8)] @ n)))
            bend = math.degrees(math.acos(np.clip(n1 @ n2, -1, 1)))
            if best is None or s > best[0]: best = (s, bend, t1, t2)
        s, bend, t1, t2 = best
        if s > 0.3:
            out += [t1, t2]; stats['split_crease' if bend >= min_bend else 'split_other'] += 1
        else:
            out.append(list(f)); stats['kept'] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--min-bend', type=float, default=40.0); a = ap.parse_args()
    Vr, Fr = read_obj(a.ref)
    Tr = np.array([[f[0], f[k], f[k + 1]] for f in Fr for k in range(1, len(f) - 1)])
    V, F = read_obj(a.inp)
    out, stats = split_folded(Vr, Tr, V, F, a.min_bend)
    write_obj(a.out, V, out)
    stats['faces'] = len(out); stats['quad_pct'] = round(100 * sum(1 for f in out if len(f) == 4) / len(out), 2)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
