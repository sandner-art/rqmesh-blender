# SPDX-License-Identifier: Apache-2.0
"""Angular-resolution bounds for Bi-MDF quantization (Paper A, part 2 prototype).

A subside that crosses a curved strip (fillet, multi-segment bevel) turns the surface normal by Theta.  Isometry
alone may quantize it to 1 segment when the strip is narrower than the target quad size, which flattens a 90-deg
fillet into a chamfer.  We require at least ceil(Theta / theta_max) segments, independent of the target size,
so curved strips keep their shape at any resolution (an artist's "fillets need >= 3 segments" rule).

  python strip_bounds.py WORK_DIR DUMP.txt BOUNDS.txt [--theta-max 30] [--theta-min 20]
WORK_DIR holds QuadWild's traced patch mesh (mesh_rem_p0.obj, .patch); DUMP.txt comes from the patched
quad_from_patches run with RQM_DUMP_SUBSIDES set.
"""
import sys, os, math, json, argparse
import numpy as np

try:
    from .meshio_min import read_obj, triangulate, TriMesh
except ImportError:                     # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meshio_min import read_obj, triangulate, TriMesh


def subside_turn(M, patch, verts, chart):
    """Angle between the surface normals at the two ends of a subside, measured on the side of `chart`."""
    if len(verts) < 2:
        return 0.0
    key = {}
    for e, (a, b) in enumerate(M.E):
        key[(int(a), int(b))] = e
    def face_on_side(a, b):
        e = key.get((min(a, b), max(a, b)))
        if e is None:
            return None
        fs = [f for f in M.EF[e] if f >= 0]
        side = [f for f in fs if patch[f] == chart] if chart >= 0 else []
        return (side or fs or [None])[0]
    f0 = face_on_side(verts[0], verts[1]); f1 = face_on_side(verts[-2], verts[-1])
    if f0 is None or f1 is None:
        return 0.0
    return math.degrees(math.acos(float(np.clip(M.N[f0] @ M.N[f1], -1, 1))))


def compute_bounds(work, dump, out, theta_max=30.0, theta_min=20.0, cap=8):
    V, F = read_obj(os.path.join(work, 'mesh_rem_p0.obj')); M = TriMesh(V, triangulate(F))
    patch = np.array([int(x) for x in open(os.path.join(work, 'mesh_rem_p0.patch')).read().split()[1:]])
    if len(patch) != len(M.T):                                     # file without a count header
        patch = np.array([int(x) for x in open(os.path.join(work, 'mesh_rem_p0.patch')).read().split()])
    n_b, hist = 0, {}
    with open(out, 'w') as fo:
        for ln in open(dump):
            t = ln.split()
            sid, c0 = int(t[0]), int(t[1]); nv = int(t[4]); verts = [int(x) for x in t[5:5 + nv]]
            th = subside_turn(M, patch, verts, c0)
            if th >= theta_min:
                lo = min(cap, max(1, math.ceil(th / theta_max)))
                if lo > 1:
                    fo.write(f"{sid} {lo} -1\n"); n_b += 1; hist[lo] = hist.get(lo, 0) + 1
    return dict(bounded_subsides=n_b, lower_bound_hist=hist)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('dump'); ap.add_argument('out')
    ap.add_argument('--theta-max', type=float, default=30.0); ap.add_argument('--theta-min', type=float, default=20.0)
    ap.add_argument('--cap', type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(compute_bounds(a.work, a.dump, a.out, a.theta_max, a.theta_min, a.cap)))


if __name__ == '__main__':
    main()
