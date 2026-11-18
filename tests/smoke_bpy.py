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


def gridded_cube_extruded(level):
    """A cube gridded by one Simple subdivision, then one face extruded: the whole-mesh Recover finds nothing."""
    import bmesh
    for x in list(bpy.data.objects): bpy.data.objects.remove(x)
    bpy.ops.mesh.primitive_cube_add(); o = bpy.context.active_object
    m = o.modifiers.new('s', 'SUBSURF'); m.levels = 1; m.subdivision_type = 'SIMPLE'
    bpy.ops.object.modifier_apply(modifier='s')
    bm = bmesh.new(); bm.from_mesh(o.data); bm.faces.ensure_lookup_table()
    f = bm.faces[0]; n = f.normal.copy(); r = bmesh.ops.extrude_discrete_faces(bm, faces=[f])
    bmesh.ops.translate(bm, verts=list(r['faces'][0].verts), vec=n * 0.3)
    bm.to_mesh(o.data); bm.free(); o.data.update()
    o.select_set(True); bpy.context.view_layer.objects.active = o
    return o


# 24 grid quads + 4 extruded = 28 faces in; 5 cube faces come back, the edited face's fan stays (3 grid quads + 5)
run('rqmesh_recover_base', 0, dict(partial=True), lambda a, b, n0: len(b.data.polygons) == 13, setup=gridded_cube_extruded)


def skinned(level):
    """Suzanne subdivided twice, skinned to a one-bone armature with weights rising along z."""
    o = monkey(level)
    bpy.ops.object.armature_add(location=(0, 0, 0)); arm = bpy.context.active_object
    g = o.vertex_groups.new(name='Bone')
    for v in o.data.vertices: g.add([v.index], max(0.0, min(1.0, 0.5 + v.co.z)), 'REPLACE')
    md = o.modifiers.new('Armature', 'ARMATURE'); md.object = arm
    for x in bpy.context.selected_objects: x.select_set(False)
    o.select_set(True); bpy.context.view_layer.objects.active = o
    return o


def rig_carried(a, b, n0):
    """Recover keeps the vertex group and the rig. With 'on surface' placement each vertex sits on its source
    vertex, so its weight must be the source formula at that point."""
    if 'Bone' not in b.vertex_groups or not any(m.type == 'ARMATURE' and m.object for m in b.modifiers): return False
    gb = b.vertex_groups['Bone'].index
    for v in b.data.vertices:
        w = next((x.weight for x in v.groups if x.group == gb), None)
        if w is None or abs(w - max(0.0, min(1.0, 0.5 + v.co.z))) > 1e-4: return False
    return True


run('rqmesh_recover_base', 2, dict(placement='SURFACE'), rig_carried, setup=skinned)


def weights_follow(a, b, n0):
    """Simplify makes new vertices: their weights are interpolated on the input surface, so close to the source
    formula at the new vertex (not exact: a vertex sits a little off the input surface)."""
    if 'Bone' not in b.vertex_groups or not any(m.type == 'ARMATURE' and m.object for m in b.modifiers): return False
    gb = b.vertex_groups['Bone'].index
    err = [abs(next((x.weight for x in v.groups if x.group == gb), 0.0) - max(0.0, min(1.0, 0.5 + v.co.z)))
           for v in b.data.vertices]
    return max(err) < 0.05


run('rqmesh_structure_simplify', 2, dict(preset='ORGANIC', ratio=0.35, keep_joints=True), weights_follow,
    setup=skinned)
run('rqmesh_remesh_quadwild', 2, dict(backend='QUADRIFLOW', target_faces=1500), weights_follow, setup=skinned)


def skinned_tube(_level, n=48, m=12):
    """A tube along z with two bones blending over |z| < 0.1: an elbow."""
    import math
    V = [(0.25 * math.cos(2 * math.pi * j / m), 0.25 * math.sin(2 * math.pi * j / m), -1 + 2 * i / (n - 1))
         for i in range(n) for j in range(m)]
    F = [(i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j) for i in range(n - 1) for j in range(m)]
    me = bpy.data.meshes.new('tube'); me.from_pydata(V, [], F); me.update()
    o = bpy.data.objects.new('tube', me); bpy.context.collection.objects.link(o)
    bpy.ops.object.armature_add(location=(0, 0, 0)); arm = bpy.context.active_object
    ga, gb = o.vertex_groups.new(name='A'), o.vertex_groups.new(name='B')
    for v in o.data.vertices:
        w = max(0.0, min(1.0, (v.co.z + 0.1) / 0.2)); ga.add([v.index], 1 - w, 'REPLACE'); gb.add([v.index], w, 'REPLACE')
    o.modifiers.new('Armature', 'ARMATURE').object = arm
    for x in bpy.context.selected_objects: x.select_set(False)
    o.select_set(True); bpy.context.view_layer.objects.active = o
    return o


loops = {}
for keep in (False, True):
    def count(a, b, n0, keep=keep):
        zs = {round(v.co.z, 3) for v in b.data.vertices}; loops[keep] = (len(b.data.polygons), sum(abs(z) <= 0.15 for z in zs))
        return True
    run('rqmesh_structure_simplify', 0, dict(preset='HARD', ratio=0.25, keep_joints=keep), count, setup=skinned_tube)
print('joint loops kept (faces, loops within the blend):', loops)
if not (loops.get(True, (0, 0))[1] > loops.get(False, (0, 99))[1]): failures.append('keep_joints (no more loops kept)')
run('rqmesh_deformation_field', 1, {}, setup=with_shape_key)
if os.environ.get('RQM_QWLIB'):
    run('rqmesh_remesh_quadwild', 2, {}, lambda a, b, n0: b is not a and quads_only(b) and 100 < len(b.data.polygons) < n0)
    run('rqmesh_remesh_quadwild', 2, dict(backend='QUADWILD'), weights_follow, setup=skinned)
# Rig Hard Surface: a base, an arm on a pin, a hand on a ball, as separate objects
def _rig_model():
    import math
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)       # (a factory reset would unload the add-on)
    def box(name, size, loc):
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc); o = bpy.context.active_object; o.scale = size
        bpy.ops.object.transform_apply(scale=True); o.name = name; return o
    objs = [box('base', (1.0, 1.0, 0.2), (0, 0, 0.1)), box('post', (0.3, 0.3, 0.8), (0, 0, 0.6)),
            box('arm', (0.25, 0.25, 0.8), (0, 0, 1.45))]
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.06, depth=0.6, location=(0, 0, 1.0), rotation=(math.pi / 2, 0, 0))
    objs.append(bpy.context.active_object); objs[-1].name = 'pin'
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.1, location=(0, 0, 1.9))
    objs.append(bpy.context.active_object); objs[-1].name = 'ball'
    objs.append(box('hand', (0.12, 0.12, 0.5), (0, 0, 2.2)))
    for o in bpy.context.view_layer.objects: o.select_set(False)
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    return objs


try:
    objs = _rig_model(); t = time.time()
    r1 = bpy.ops.mesh.rqmesh_rig_hard_surface(preview=True)
    ok1 = 'FINISHED' in r1 and not any(o.type == 'ARMATURE' for o in bpy.data.objects) and \
        all('rqmesh_parts' in o.data.color_attributes for o in objs)
    print(f"{'ok  ' if ok1 else 'FAIL'} rig hard surface: preview colours the parts, makes no rig")
    if not ok1: failures.append('rig preview')
    r2 = bpy.ops.mesh.rqmesh_rig_hard_surface()
    arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    kinds = sorted(b['rqmesh_kind'] for b in arm.data.bones) if arm else []
    weighted = all(any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)
                   and all(len(v.groups) == 1 and v.groups[0].weight == 1.0 for v in o.data.vertices) for o in objs)
    hinge = next((arm.pose.bones[b.name] for b in arm.data.bones if b['rqmesh_kind'] == 'hinge'), None) if arm else None
    locked = hinge is not None and sum(hinge.lock_rotation) == 2 and all(hinge.lock_location)
    ok2 = 'FINISHED' in r2 and kinds == ['ball', 'hinge', 'root'] and weighted and locked
    print(f"{'ok  ' if ok2 else 'FAIL'} rig hard surface: {kinds}, rigid weights {weighted}, hinge locked {locked}, "
          f"{time.time() - t:.1f} s")
    if not ok2: failures.append('rig hard surface')
except Exception:
    traceback.print_exc(); failures.append('rig hard surface raised')

# UVs from Structure: a subdivided cube (cage seams, straight loops) and the rig model's parts
try:
    import numpy as np; UL = importlib.import_module('rqmesh.core.uvlayout')
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_cube_add(); cube = bpy.context.active_object
    m = cube.modifiers.new('s', 'SUBSURF'); m.levels = 2; bpy.ops.object.modifier_apply(modifier=m.name)
    for meth in ('SEAMS', 'STRAIGHT'):
        t = time.time(); r = bpy.ops.mesh.rqmesh_structure_uv(method=meth, demo=True)
        me = cube.data; F = [list(p.vertices) for p in me.polygons]; V = np.array([v.co[:] for v in me.vertices])
        uvl = me.uv_layers['RQMesh UV'].data
        uv = [np.array([uvl[i].uv[:] for i in range(p.loop_start, p.loop_start + p.loop_total)]) for p in me.polygons]
        mt = UL.uv_metrics(V, F, uv); pan = np.empty(len(F), np.int32); me.attributes['rqmesh_panel'].data.foreach_get('value', pan)
        ok = 'FINISHED' in r and mt['flipped'] == 0 and mt['islands'] <= 6 and len(set(pan.tolist())) == 6 and \
            'rqmesh_panel_uv' in me.uv_layers and (meth != 'STRAIGHT' or mt['grain_deg'] < 1e-6)
        print(f"{'ok  ' if ok else 'FAIL'} uvs from structure {meth}: {mt['islands']} islands, seams {mt['seams']:.1f}%, "
              f"grain {mt['grain_deg']:.2f} deg, 6 panels {len(set(pan.tolist())) == 6}, {time.time() - t:.1f} s")
        if not ok: failures.append(f'uvs {meth}')
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_cube_add(); cube = bpy.context.active_object
    m = cube.modifiers.new('s', 'SUBSURF'); m.levels = 2; bpy.ops.object.modifier_apply(modifier=m.name)
    r = bpy.ops.mesh.rqmesh_structure_uv(method='TRIM', demo=True); me = cube.data
    band = np.empty(len(me.polygons), np.int32); me.attributes['rqmesh_trim_band'].data.foreach_get('value', band)
    uvl = me.uv_layers['RQMesh UV'].data; vs = np.array([uvl[i].uv[1] for i in range(len(me.loops))])
    ok = 'FINISHED' in r and vs.min() >= -1e-6 and vs.max() <= 1 + 1e-6 and any(m.name.startswith('RQMesh trim') for m in me.materials)
    print(f"{'ok  ' if ok else 'FAIL'} uvs from structure TRIM: bands used {sorted(set(band.tolist()))}")
    if not ok: failures.append('uvs TRIM')
    r = bpy.ops.mesh.rqmesh_surface_masks(material=True)
    vals = {}
    for nm in ('rqmesh_wear', 'rqmesh_cavity', 'rqmesh_variance', 'rqmesh_exposure', 'rqmesh_weather'):
        x = np.empty(len(me.vertices), np.float32); me.attributes[nm].data.foreach_get('value', x); vals[nm] = (float(x.min()), float(x.max()))
    ok = 'FINISHED' in r and all(0 <= a and b <= 1 for a, b in vals.values()) and vals['rqmesh_wear'][1] > 0.5 \
        and me.materials[-1].name.startswith('RQMesh worn')
    print(f"{'ok  ' if ok else 'FAIL'} surface masks: ranges {vals}")
    if not ok: failures.append('surface masks')
    objs = _rig_model(); bpy.ops.object.select_all(action='DESELECT')
    o = objs[0]; bpy.context.view_layer.objects.active = o; o.select_set(True)
    r = bpy.ops.mesh.rqmesh_structure_uv(method='PARTS')
    ok = 'FINISHED' in r and 'RQMesh UV' in o.data.uv_layers and 'rqmesh_piece' in o.data.attributes
    print(f"{'ok  ' if ok else 'FAIL'} uvs from structure PARTS")
    if not ok: failures.append('uvs PARTS')
except Exception:
    traceback.print_exc(); failures.append('uvs from structure raised')

# the workflow panel: every plan draws, and every button points to a registered operator with valid presets
from rqmesh import workflow as WF  # noqa: E402


class _Layout:
    def __init__(s): s.ops = []; s.texts = []; s.scale_y = 1
    def box(s): return s
    def row(s, **k): return s
    def column(s, **k): return s
    def separator(s): pass
    def label(s, **k): s.texts.append(k.get('text', ''))
    def operator(s, idname, **k):
        s.ops.append(idname)
        if not idname.startswith('wm.'):
            mod, name = idname.split('.'); getattr(getattr(bpy.ops, mod), name).get_rna_type()   # registered?
        return type('Props', (), {})()


o = monkey(1)
for method in list(WF.PLANS) + [None]:
    lay = _Layout()
    try:
        WF.draw(lay, o, None if method is None else dict(method=method, faces=len(o.data.polygons)))
        alt = WF.ALTERNATIVE.get(method)
        for idname, _, _, props in WF.PLANS.get(method, []) + ([(alt[0], alt[1], '', alt[2])] if alt else []):
            mod, name = idname.split('.'); rna = getattr(getattr(bpy.ops, mod), name).get_rna_type()
            for k in props: assert k in rna.properties, f"{idname} has no property {k}"
        print(f"ok   workflow plan {method}: {len(lay.ops)} buttons")
    except Exception:
        traceback.print_exc(); failures.append(f'workflow {method}')
try:   # an unapplied Subdivision modifier: the box says the tools use the base mesh
    o.modifiers.new('Subdivision', 'SUBSURF'); lay = _Layout(); WF.draw(lay, o, None)
    assert any('base mesh' in t for t in lay.texts), lay.texts
    o.modifiers.clear(); lay = _Layout(); WF.draw(lay, o, None)
    assert not any('base mesh' in t for t in lay.texts), lay.texts
    print("ok   workflow box notes unapplied modifiers")
except Exception:
    traceback.print_exc(); failures.append('workflow modifier note')
addon_utils.disable('rqmesh')
print(json.dumps(dict(blender=bpy.app.version_string, failures=failures)))
if failures: raise RuntimeError(f"{len(failures)} operator(s) failed: {failures}")
