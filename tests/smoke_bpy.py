# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke test with Blender's Python module (pip install bpy): unpack a built extension zip, register it, run every
operator on a subdivided monkey and check the results.

usage: python tests/smoke_bpy.py dist/rqmesh-X.Y.Z-selfhosted.zip
       blender -b --factory-startup --python-exit-code 1 --python tests/smoke_bpy.py -- dist/rqmesh-....zip
       RQM_QWLIB=/path/to/qwlib_parent python tests/smoke_bpy.py ...   (also runs the QuadWild remesh)"""
import os, sys, json, time, zipfile, tempfile, importlib, traceback

zpath = os.path.abspath(sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else sys.argv[1])
root = tempfile.mkdtemp(prefix='rqmesh_smoke_')
with zipfile.ZipFile(zpath) as z: z.extractall(os.path.join(root, 'rqmesh'))
sys.path.insert(0, root)

import bpy  # noqa: E402
import addon_utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
addon = addon_utils.enable('rqmesh', default_set=True, handle_error=None)   # as Blender does: prefs exist too
assert addon is not None and 'rqmesh' in bpy.context.preferences.addons, "add-on did not enable"
print('registered', addon.__name__, 'from', os.path.dirname(addon.__file__))
for idn in ('rqmesh_structure_simplify', 'rqmesh_fix_poles', 'rqmesh_select_isolation', 'rqmesh_recover_base'):
    assert hasattr(bpy.ops.mesh, idn), f"operator mesh.{idn} missing"


def monkey(level=2):
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_monkey_add()
    o = bpy.context.active_object
    if level:
        m = o.modifiers.new('s', 'SUBSURF'); m.levels = level; bpy.ops.object.modifier_apply(modifier='s')
    o.select_set(True); bpy.context.view_layer.objects.active = o
    return o


def with_shape_key(level):
    o = monkey(level)
    o.shape_key_add(name='Basis'); k = o.shape_key_add(name='Bend')
    for i, v in enumerate(o.data.vertices):
        x, y, z = v.co; k.data[i].co = (x, y + 0.3 * x * x, z)
    return o


def quads_only(o):
    return all(len(p.vertices) == 4 for p in o.data.polygons)


failures = []


def run(name, setup_level, kwargs, check=None, setup=None):
    o = (setup or monkey)(setup_level); n0 = len(o.data.polygons); t = time.time()
    before = set(bpy.data.objects)
    try:
        r = getattr(bpy.ops.mesh, name)(**kwargs)
        new = [x for x in bpy.data.objects if x not in before]
        out = bpy.context.active_object
        if new and out is o: failures.append(name + ' (result not made active)')
        ok = 'FINISHED' in r and (check is None or check(o, out, n0))
        print(f"{'ok  ' if ok else 'FAIL'} {name:30s} {r} {n0} -> {len(out.data.polygons)} faces, {time.time() - t:.1f} s")
        if not ok: failures.append(name)
    except Exception:
        traceback.print_exc(); failures.append(name); print('FAIL', name, 'raised')


run('rqmesh_analyze', 2, {})
run('rqmesh_detect_features', 2, {})
run('rqmesh_detect_symmetry', 2, {})
run('rqmesh_select_poles', 2, {})
run('rqmesh_select_isolation', 2, {})
run('rqmesh_structure_simplify', 2, dict(preset='ORGANIC', ratio=0.35),
    lambda a, b, n0: quads_only(b) and len(b.data.polygons) < 0.45 * n0)
run('rqmesh_fix_poles', 1, dict(keep_isolation=2, refill=True), lambda a, b, n0: quads_only(b))
run('rqmesh_recover_base', 2, {}, lambda a, b, n0: len(b.data.polygons) == 500)   # Suzanne's own 500 faces
run('rqmesh_quad_caps', 1, {})
run('rqmesh_deformation_field', 1, {}, setup=with_shape_key)
if os.environ.get('RQM_QWLIB'):
    run('rqmesh_remesh_quadwild', 2, {}, lambda a, b, n0: b is not a and quads_only(b) and 100 < len(b.data.polygons) < n0)
addon_utils.disable('rqmesh')
print(json.dumps(dict(blender=bpy.app.version_string, failures=failures)))
if failures: raise RuntimeError(f"{len(failures)} operator(s) failed: {failures}")
