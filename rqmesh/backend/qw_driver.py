# SPDX-License-Identifier: Apache-2.0
"""Experiment driver around the GPL-3 QuadWild-BiMDF libraries (loaded at runtime from the user's kit; not
redistributed here).  Adds what the stock runner does not expose:
  --features {angle,auto,none,file}  feature source (auto = resolution-aware classifier, features.py)
  --field FILE.rosy                  inject an external cross field (hasField=True, remesh off)
  --alpha, --flow {SIMPLE,HALF}, --satsuma {DEFAULT,MST,ROUND2EVEN,SYMMDC,EDGETHRU,NODETHRU,LEMON}
  --align-sing / --no-align-sing, --reg-weight
usage: python -I qw_driver.py KIT_QW_DIR IN.obj OUT.obj --target N [options]  -> JSON line on stdout
"""
import sys, os, json, time, math, shutil, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'core'))   # inside the extension: ../core
from meshio_min import read_obj, write_obj, triangulate, TriMesh   # noqa: E402
import features as FT                                               # noqa: E402


def _prepare(a, qw, info, has_field):
    from qwlib.data import Parameters
    from qwlib import create_string
    if has_field:
        shutil.copy(a.field, qw.field_path)
    t1 = time.time()
    enable_sharp = info['n_sharp'] > 0
    params = Parameters(remesh=not (a.no_remesh or has_field), sharpAngle=a.sharp if enable_sharp else -1,
                        hasFeature=enable_sharp, hasField=has_field, alpha=0.01, scaleFact=1)
    qw.quadwild.remeshAndField2(params, create_string(qw.mesh_path), create_string(qw.sharp_path),
                                create_string(qw.field_path))
    info['t_field'] = round(time.time() - t1, 2)
    t2 = time.time(); ok = qw.trace(); info['t_trace'] = round(time.time() - t2, 2)
    info['trace_ok'] = bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('qwdir'); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--target', type=int, required=True)
    ap.add_argument('--features', default='auto', choices=['angle', 'auto', 'auto2', 'none', 'file'])
    ap.add_argument('--sharp', type=float, default=35.0)
    ap.add_argument('--sharp-file')
    ap.add_argument('--field')
    ap.add_argument('--no-remesh', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.005)
    ap.add_argument('--flow', default='SIMPLE')
    ap.add_argument('--satsuma', default='DEFAULT')
    ap.add_argument('--no-align-sing', action='store_true')
    ap.add_argument('--reg-weight', type=float, default=0.9)
    ap.add_argument('--c-min', type=float, default=0.8)
    ap.add_argument('--tol', type=float, default=0.08)
    ap.add_argument('--step-chain', type=float, default=3.0)
    ap.add_argument('--no-smooth', action='store_true', help='skip QuadWild final smoothing')
    ap.add_argument('--dump-only', action='store_true', help='trace, dump subsides (patched lib) and stop')
    ap.add_argument('--reuse-work', action='store_true', help='skip remesh/field/trace; quantize an existing work dir')
    ap.add_argument('--bounds-file', help='precomputed per-subside bounds (patched lib)')
    ap.add_argument('--strip-theta', type=float, default=0.0,
                    help='>0: angular-resolution bounds (needs the RQM-patched quadpatches library)')
    ap.add_argument('--work')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.qwdir))
    from qwlib import Quadwild

    work = a.work or os.path.splitext(a.out)[0] + '_w'
    if not a.reuse_work:
        shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    V, F = read_obj(a.inp); T = triangulate(F)
    mesh_path = os.path.join(work, 'mesh.obj')
    if not a.reuse_work:
        write_obj(mesh_path, V, T)
    qw = Quadwild(mesh_path)
    M = TriMesh(V, T); L = FT.target_edge_length(M, a.target)
    t0 = time.time(); info = dict(L_pct=round(100 * L / M.diag, 3))
    if a.reuse_work:
        a.features = 'skip'
    if a.features == 'skip':
        info['n_sharp'] = 0
    elif a.features == 'angle':
        mask = M.interior & (M.theta > np.radians(a.sharp))
        info['n_sharp'] = FT.write_quadwild_sharp(M, mask, qw.sharp_path)
    elif a.features == 'auto':
        lab, _ = FT.classify(M, L, tau_deg=a.sharp, c_min=a.c_min, step_chain=a.step_chain)
        info.update(n_crease=int((lab == FT.CREASE).sum()), n_blend=int((lab == FT.BLEND).sum()),
                    n_noise=int((lab == FT.NOISE).sum()))
        info['n_sharp'] = FT.write_quadwild_sharp(M, lab == FT.CREASE, qw.sharp_path)
    elif a.features == 'auto2':
        lab, _ = FT.classify_v2(M, L, tau_deg=a.sharp, c_min=a.c_min, step_chain=a.step_chain)
        info.update(n_crease=int((lab == FT.CREASE).sum()), n_loop=int((lab == FT.LOOP).sum()),
                    n_blend=int((lab == FT.BLEND).sum()), n_noise=int((lab == FT.NOISE).sum()))
        info['n_sharp'] = FT.write_quadwild_sharp(M, (lab == FT.CREASE) | (lab == FT.LOOP), qw.sharp_path)
    elif a.features == 'file':
        shutil.copy(a.sharp_file, qw.sharp_path); info['n_sharp'] = int(open(qw.sharp_path).readline())
    else:
        info['n_sharp'] = FT.write_quadwild_sharp(M, np.zeros(len(M.E), bool), qw.sharp_path)
    info['t_features'] = round(time.time() - t0, 2)
    has_field = a.field is not None
    if not a.reuse_work:
        _prepare(a, qw, info, has_field)


    def quad(scale):
        ts = time.time()
        qw.quadrangulate(not a.no_smooth, scale, 0, a.alpha, 'LEASTSQUARES', 200, 0.0, 0.4, True, True, True, a.reg_weight,
                         not a.no_align_sing, 0.1, True, False, False, not a.no_align_sing, True, a.flow, a.satsuma,
                         [3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0],
                         [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3])
        res = qw.output_smoothed_path if not a.no_smooth else qw.output_smoothed_path.replace('_quadrangulation_smooth', '_quadrangulation')
        nf = sum(1 for ln in open(res) if ln.startswith('f ')) if os.path.exists(res) else 0
        return res, nf, time.time() - ts

    if a.dump_only:
        os.environ['RQM_DUMP_SUBSIDES'] = os.path.join(work, 'subsides.txt'); quad(1.0)
        print(json.dumps(dict(info, dumped=os.path.exists(os.path.join(work, 'subsides.txt'))))); return
    if a.bounds_file:
        os.environ['RQM_SUBSIDE_BOUNDS'] = os.path.abspath(a.bounds_file)
    if a.strip_theta > 0:
        import strip_bounds as SB
        dump = os.path.join(work, 'subsides.txt'); bnd = os.path.join(work, 'bounds.txt')
        os.environ['RQM_DUMP_SUBSIDES'] = dump; quad(1.0); del os.environ['RQM_DUMP_SUBSIDES']
        if os.path.exists(dump):
            info['strip'] = SB.compute_bounds(work, dump, bnd, theta_max=a.strip_theta)
            os.environ['RQM_SUBSIDE_BOUNDS'] = bnd
        else:                                   # stock library: no hook, run unconstrained
            info['strip'] = {'unsupported_library': True}
    scale, slope = 1.0, -2.0
    res, nf, tq = quad(scale); passes = [(scale, nf, round(tq, 2))]
    for _ in range(6):
        if nf and abs(nf - a.target) / a.target < a.tol:
            break
        if len(passes) >= 2:
            (s0, n0, _), (s1, n1, _) = passes[-2], passes[-1]
            if s1 != s0 and n1 != n0 and n0 > 0 and n1 > 0:
                slope = min(-0.3, math.log(n1 / n0) / math.log(s1 / s0))
        scale = scale * math.exp(math.log(a.target / max(nf, 1)) / slope)
        res, nf, tq = quad(scale); passes.append((scale, nf, round(tq, 2)))
    best = min(passes, key=lambda q: abs(q[1] - a.target))
    if best[0] != scale:
        res, nf, tq = quad(best[0])
    shutil.copy(res, a.out)
    info.update(passes=passes, t_quad_last=round(tq, 2), seconds=round(time.time() - t0, 2))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
