# SPDX-License-Identifier: GPL-3.0-or-later
# Blender wrapper (GPL, as Blender's extension platform requires). The algorithms live in ./core under Apache-2.0.
"""RQMesh: structure-aware quad remeshing, simplification and retopology for Blender.

* Detect features  - resolution-aware crease / facet-loop / blend / step classification, written to Blender sharp
                      marks and to the edge attribute 'rqmesh_feature' (1 crease, 2 blend facet, 3 step-or-noise, 4 facet loop).
* Deformation field - cross field aligned with principal strain across shape keys or the current armature
                      pose; face attributes 'rqmesh_field' (vector) and 'rqmesh_strain' (0..1).
* Remesh (QuadWild-BiMDF backend) - optional; runs the external GPL QuadWild-BiMDF libraries in a separate
                      process with the marks / field above. Set the library folder in the add-on preferences.
* Simplify by Edge Rings - structure-first retopology for models that still carry their modelling structure:
                      removes whole poly-chords, never merging feature lines; keeps UVs and symmetry.
* Analyze Structure - quads / regular vertices / clean loops -> recommends edge loops, adaptive or Decimate/remesh.
* Detect Symmetry   - reflective symmetry planes + per-part axes; enables Blender mesh symmetry.
* Quad Caps         - triangle fans / n-gons / high-valence poles on caps -> patterned quad grid.
* Fix Poles / Select Misplaced Poles - valence-energy clean-up and diagnosis (ideal valence from the surface angle).
(All structure-first and clean-up tools are pure Python + NumPy; no backend needed.)
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import bpy
from bpy.props import IntProperty, FloatProperty, BoolProperty, EnumProperty, StringProperty

from .core.meshio_min import TriMesh, write_obj, read_obj
from .core import features as FT
from .core import defofield as DF
from .core.rosy import write_rosy
from . import backends as BK
from . import workflow as WF
from bpy_extras.io_utils import ImportHelper

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- mesh access helpers
def mesh_arrays(me):
    """Vertices, loop-triangle indices, source polygon of each triangle (object-local coordinates)."""
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V)
    T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', T)
    P = np.empty(len(me.loop_triangles), np.int64); me.loop_triangles.foreach_get('polygon_index', P)
    return V.reshape(-1, 3), T.reshape(-1, 3), P


# modifiers that change what the viewport shows; RQMesh reads the base mesh, as Blender's own mesh tools do
SHAPE_MODIFIERS = {'SUBSURF', 'MULTIRES', 'REMESH', 'DECIMATE', 'BEVEL', 'BOOLEAN', 'SOLIDIFY', 'MIRROR', 'ARRAY',
                   'SCREW', 'SKIN', 'WELD', 'TRIANGULATE', 'WIREFRAME', 'NODES'}


def warn_modifiers(op, ob, recover=False):
    """Say so when the result was built from a base mesh that the viewport doesn't show as it is."""
    mods = [m for m in ob.modifiers if m.show_viewport and m.type in SHAPE_MODIFIERS]
    if not mods:
        return
    names = ", ".join(f"'{m.name}'" for m in mods)
    if recover and any(m.type in ('SUBSURF', 'MULTIRES') for m in mods):
        op.report({'WARNING'}, f"{ob.name} still has {names}: its base mesh is the cage already. Recover is for "
                               f"subdivision that was applied (and maybe edited afterwards)")
    else:
        op.report({'WARNING'}, f"Worked on the base mesh: {names} not applied. To use the shape you see, apply "
                               f"the modifier(s) first")


def edge_lookup(me):
    E = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get('vertices', E); E = E.reshape(-1, 2)
    return {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(E)}


def ensure_attr(me, name, kind, domain):
    a = me.attributes.get(name)
    if a is None or a.data_type != kind or a.domain != domain:
        if a is not None:
            me.attributes.remove(a)
        a = me.attributes.new(name, kind, domain)
    return a


# ----------------------------------------------------------------------------- operators
class RQM_OT_detect_features(bpy.types.Operator):
    """Classify edges into creases, blend (fillet/bevel) facets and steps at the scale of the target quad size"""
    bl_idname = "mesh.rqmesh_detect_features"
    bl_label = "Detect Features (resolution-aware)"
    bl_options = {'REGISTER', 'UNDO'}

    target_faces: IntProperty(name="Target quads", default=3000, min=50)
    tau: FloatProperty(name="Crease angle", default=35.0, min=5.0, max=170.0)
    c_min: FloatProperty(name="Concentration", default=0.8, min=0.1, max=1.0,
                         description="Share of the band turn an edge must carry to count as a crease")
    mark_sharp: BoolProperty(name="Write sharp marks", default=True)
    version: EnumProperty(name="Classifier", default='V2',
                          items=[('V2', "v2: creases + thinned facet loops", "Bevel/fillet facet lines kept with ~0.6 L spacing"),
                                 ('V1', "v1: creases only", "Blend facets are dropped")])

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        me = ctx.active_object.data
        V, T, _ = mesh_arrays(me)
        M = TriMesh(V, T)
        L = FT.target_edge_length(M, self.target_faces)
        if self.version == 'V2':
            lab, _ = FT.classify_v2(M, L, tau_deg=self.tau, c_min=self.c_min)
        else:
            lab, _ = FT.classify(M, L, tau_deg=self.tau, c_min=self.c_min)
        lut = edge_lookup(me)
        out = np.zeros(len(me.edges), np.int32)
        for e, (a, b) in enumerate(M.E):
            i = lut.get((int(a), int(b)))
            if i is not None:
                out[i] = lab[e]
        ensure_attr(me, 'rqmesh_feature', 'INT', 'EDGE').data.foreach_set('value', out)
        if self.mark_sharp:
            ensure_attr(me, 'sharp_edge', 'BOOLEAN', 'EDGE').data.foreach_set('value', (out == FT.CREASE) | (out == FT.LOOP))
        me.update()
        self.report({'INFO'}, f"creases {int((out == 1).sum())}, facet loops {int((out == 4).sum())}, "
                              f"blend facets {int((out == 2).sum())}, steps/noise {int((out == 3).sum())} (L = {L:.4g})")
        return {'FINISHED'}


class RQM_OT_deformation_field(bpy.types.Operator):
    """Cross field aligned with the principal strain of the shape keys or of the current armature pose"""
    bl_idname = "mesh.rqmesh_deformation_field"
    bl_label = "Deformation Field"
    bl_options = {'REGISTER', 'UNDO'}

    source: EnumProperty(name="Poses", items=[('SHAPE_KEYS', "Shape keys", "Every non-basis key is a pose"),
                                              ('EVALUATED', "Current pose", "Evaluated mesh (armature etc.)")])
    smoothness: FloatProperty(name="Data weight", default=1.0, min=0.01, max=100.0)
    percentile: FloatProperty(name="Strain percentile", default=98.0, min=50.0, max=100.0)
    repel: FloatProperty(name="Pole repulsion", default=10.0, min=0.0, max=100.0,
                         description="Extra field stiffness in high-strain zones; keeps poles out of joints")

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        ob = ctx.active_object; me = ob.data
        V, T, P = mesh_arrays(me)
        poses = []
        if self.source == 'SHAPE_KEYS':
            if not me.shape_keys or len(me.shape_keys.key_blocks) < 2:
                self.report({'ERROR'}, "Needs at least one shape key besides Basis"); return {'CANCELLED'}
            for kb in me.shape_keys.key_blocks[1:]:
                X = np.empty(len(me.vertices) * 3); kb.data.foreach_get('co', X); poses.append(X.reshape(-1, 3))
        else:
            ev = ob.evaluated_get(ctx.evaluated_depsgraph_get()).to_mesh()
            if len(ev.vertices) != len(me.vertices):
                ob.to_mesh_clear(); self.report({'ERROR'}, "Evaluated mesh changes vertex count"); return {'CANCELLED'}
            X = np.empty(len(ev.vertices) * 3); ev.vertices.foreach_get('co', X); ob.to_mesh_clear()
            poses.append(X.reshape(-1, 3))
        M = TriMesh(V, T)
        E = DF.strain_tensors(M, poses); _, smax, smin = DF.principal(E)
        s_ref = float(np.percentile(smax - smin, self.percentile)) + 1e-12
        dmax = DF.principal(E)[0]; e1, e2 = DF.face_frames(M)
        prior = dmax[:, :1] * e1 + dmax[:, 1:] * e2
        D, w = DF.deformation_aware_field(M, poses, prior, s_ref=s_ref, prior_w=0.01, lam=self.smoothness, repel=self.repel)
        nP = len(me.polygons)
        Fv = np.zeros((nP, 3)); Fs = np.zeros(nP)
        Fv[P] = D; np.maximum.at(Fs, P, w)
        ensure_attr(me, 'rqmesh_field', 'FLOAT_VECTOR', 'FACE').data.foreach_set('vector', Fv.ravel())
        ensure_attr(me, 'rqmesh_strain', 'FLOAT', 'FACE').data.foreach_set('value', Fs)
        me.update()
        self.report({'INFO'}, f"field on {len(T)} triangles; {int((w > 0.2).sum())} with significant strain")
        return {'FINISHED'}


class RQM_OT_remesh_quadwild(bpy.types.Operator):
    """Quad remesh with the external QuadWild-BiMDF libraries using RQMesh features and (optionally) the RQMesh field"""
    bl_idname = "mesh.rqmesh_remesh_quadwild"
    bl_label = "Remesh (QuadWild-BiMDF backend)"
    bl_options = {'REGISTER', 'UNDO'}

    target_faces: IntProperty(name="Target quads", default=3000, min=50)
    features: EnumProperty(name="Features", items=[('MARKS', "Sharp marks", "Edges marked sharp; without marks: automatic features for a coarse mesh, none for a dense one"), ('AUTO', "Auto v2 (RQMesh)", ""),
                                                   ('ANGLE', "Angle", ""), ('NONE', "None", "")], default='MARKS')
    angle: FloatProperty(name="Angle", default=35.0)
    alpha: FloatProperty(name="Isometry weight (alpha)", default=0.02, min=0.0005, max=0.5,
                         description="QuadWild/Bi-MDF quantization weight; 0.02 beat the 0.005 default in tests")
    use_field: BoolProperty(name="Use RQMesh field", default=False)
    angular_rows: FloatProperty(name="Max turn per row (deg)", default=30.0, min=0.0, max=90.0,
                                description="Curved strips get at least turn/value rows (0 = off). Needs the "
                                            "rqmesh-patched quadpatches library; ignored with the stock one")
    repair: BoolProperty(name="Repair folds", default=True,
                         description="Feature-aware untangling pass on the result")
    backend: EnumProperty(name="Back-end", default='AUTO', items=[
        ('AUTO', "Auto", "QuadWild-BiMDF when installed, otherwise Blender's built-in QuadriFlow"),
        ('QUADWILD', "QuadWild-BiMDF", "External GPL back-end (see add-on preferences)"),
        ('QUADRIFLOW', "QuadriFlow (built-in)", "Blender's own quad remesher: always available, weaker on hard surface")])

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def _quadriflow(self, ctx, ob, why):
        dup = ob.copy(); dup.data = ob.data.copy(); dup.name = ob.name + "_rqmesh"; ctx.collection.objects.link(dup)
        for o in ctx.selected_objects: o.select_set(False)
        dup.select_set(True); ctx.view_layer.objects.active = dup
        r = bpy.ops.object.quadriflow_remesh(target_faces=self.target_faces, use_preserve_sharp=True,
                                             use_preserve_boundary=True, use_mesh_symmetry=False)
        self.report({'INFO'} if 'FINISHED' in r else {'ERROR'},
                    f"QuadriFlow: {len(dup.data.polygons)} faces" + (f" ({why})" if why else ""))
        return {'FINISHED'} if 'FINISHED' in r else {'CANCELLED'}

    def execute(self, ctx):
        prefs = ctx.preferences.addons[__package__].preferences
        qwdir, src = BK.find_quadwild(bpy.path.abspath(prefs.qwlib_dir), __package__, HERE)
        if self.backend == 'QUADRIFLOW':
            return self._quadriflow(ctx, ctx.active_object, "")
        if qwdir is None:
            if self.backend == 'AUTO':
                return self._quadriflow(ctx, ctx.active_object, "QuadWild-BiMDF not installed - see add-on preferences")
            self.report({'ERROR'}, "QuadWild-BiMDF back-end not found: install it in the add-on preferences")
            return {'CANCELLED'}
        ob = ctx.active_object; me = ob.data
        V, T, P = mesh_arrays(me)
        tmp = tempfile.mkdtemp(prefix='rqm_')
        inp, out = os.path.join(tmp, 'in.obj'), os.path.join(tmp, 'out.obj')
        write_obj(inp, V, T)
        cmd = [prefs.python or sys.executable, '-I', os.path.join(HERE, 'backend', 'qw_driver.py'), qwdir, inp, out,
               '--target', str(self.target_faces), '--alpha', str(self.alpha), '--sharp', str(self.angle)]
        feat_note = ""
        features = self.features
        if features == 'MARKS':
            sharp = me.attributes.get('sharp_edge')
            flags = np.zeros(len(me.edges), bool)
            if sharp is not None:
                sharp.data.foreach_get('value', flags)
            if not flags.any():
                # no marks: a coarse, faceted mesh needs feature lines or its shape drifts; a dense sculpt does not --
                # automatic features there add poles and folds (round 14: 211 poles / 31 folds vs 90 / 1)
                if len(me.polygons) < self.target_faces:
                    features = 'AUTO'; feat_note = ", no sharp marks: automatic features (coarse input)"
                else:
                    features = 'NONE'; feat_note = ", no sharp marks: no features (dense input)"
        if features == 'MARKS':
            M = TriMesh(V, T); lut = edge_lookup(me)
            mask = np.array([flags[lut[(int(a), int(b))]] if (int(a), int(b)) in lut else False for a, b in M.E])
            sp = os.path.join(tmp, 'marks.sharp'); FT.write_quadwild_sharp(M, mask, sp)
            cmd += ['--features', 'file', '--sharp-file', sp]
        else:
            cmd += ['--features', {'AUTO': 'auto2', 'ANGLE': 'angle', 'NONE': 'none'}[features]]
        if self.angular_rows > 0:
            cmd += ['--strip-theta', str(self.angular_rows)]
        if self.use_field:
            a = me.attributes.get('rqmesh_field')
            if a is None:
                self.report({'ERROR'}, "Run 'Deformation Field' first"); return {'CANCELLED'}
            Fv = np.empty(len(me.polygons) * 3); a.data.foreach_get('vector', Fv); Fv = Fv.reshape(-1, 3)[P]
            rp = os.path.join(tmp, 'field.rosy'); write_rosy(rp, Fv); cmd += ['--field', rp]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=prefs.timeout)
        if r.returncode != 0 or not os.path.exists(out):
            self.report({'ERROR'}, "Backend failed: " + (r.stderr or r.stdout)[-300:]); return {'CANCELLED'}
        rep = ''
        if self.repair:
            fixed = os.path.join(tmp, 'out_repaired.obj')
            rr = subprocess.run([prefs.python or sys.executable, '-I', os.path.join(HERE, 'core', 'repair.py'), inp, out,
                                 fixed, '--feature-deg', '8' if features != 'NONE' else '35'],
                                capture_output=True, text=True, timeout=prefs.timeout)
            if rr.returncode == 0 and os.path.exists(fixed):
                out = fixed
                js = [l for l in rr.stdout.splitlines() if l.startswith('{')]
                if js:
                    d = json.loads(js[-1]); rep = f", folds {d['folds_before']} -> {d['folds_after']}"
        Vq, Fq = read_obj(out)
        from .core import repair as RP
        Fq, hole_info = RP.fill_new_holes(V, T, Vq, Fq)          # QuadWild can leave a patch unfilled
        if hole_info['holes_new']:
            rep += (f", filled {hole_info['holes_new'] - hole_info['too_big']} hole(s) the back-end left"
                    + (f" ({hole_info['filled_triangles']} triangle)" if hole_info['filled_triangles'] else "")
                    + (f"; {hole_info['too_big']} too large to fill" if hole_info['too_big'] else ""))
        rep += feat_note
        nm = bpy.data.meshes.new(ob.name + "_rqmesh"); nm.from_pydata(Vq.tolist(), [], [list(map(int, f)) for f in Fq])
        nm.update(); no = bpy.data.objects.new(nm.name, nm); no.matrix_world = ob.matrix_world.copy()
        ctx.collection.objects.link(no)
        ob.select_set(False); no.select_set(True); ctx.view_layer.objects.active = no   # like the other operators
        info = [l for l in r.stdout.splitlines() if l.startswith('{"')]
        self.report({'INFO'}, f"{len(Fq)} faces" + (f" ({json.loads(info[-1]).get('seconds')} s)" if info else "") + rep)
        warn_modifiers(self, ob)
        return {'FINISHED'}


def poly_data(me, with_uv=True):
    """Vertices, polygons and (optionally) per-corner UV indices of a mesh."""
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
    F = [list(p.vertices) for p in me.polygons]
    UV = FUV = None
    if with_uv and me.uv_layers.active is not None:
        L = np.empty(len(me.loops) * 2); me.uv_layers.active.data.foreach_get('uv', L); UV = L.reshape(-1, 2)
        FUV = [list(p.loop_indices) for p in me.polygons]
    return V, F, UV, FUV


def carry_rig(src, ob, vmap):
    """Skin weights and rig of `src` onto `ob`, whose vertex i is vertex vmap[i] of src: vertex groups (exact copies,
    no interpolation), Armature modifiers and the parent."""
    sw = {}
    for v in src.data.vertices:
        if v.groups: sw[v.index] = [(g.group, g.weight) for g in v.groups]
    if src.vertex_groups:
        groups = [ob.vertex_groups.new(name=g.name) for g in src.vertex_groups]
        by = {}
        for i, s_v in enumerate(vmap):
            for gi, w in sw.get(int(s_v), ()): by.setdefault((gi, round(w, 6)), []).append(i)
        for (gi, w), idx in by.items(): groups[gi].add(idx, w, 'REPLACE')
    for m in src.modifiers:
        if m.type == 'ARMATURE':
            nm = ob.modifiers.new(m.name, 'ARMATURE'); nm.object = m.object
            for k in ('use_vertex_groups', 'use_bone_envelopes', 'use_deform_preserve_volume', 'use_multi_modifier'):
                setattr(nm, k, getattr(m, k))
    if src.parent is not None:
        ob.parent = src.parent; ob.parent_type = src.parent_type; ob.parent_bone = src.parent_bone
        ob.matrix_parent_inverse = src.matrix_parent_inverse.copy(); ob.matrix_basis = src.matrix_basis.copy()


def new_object_from(ctx, src, name, V, F, UV=None, FUV=None, vmap=None, face_src=None):
    """vmap: source vertex of each new vertex (carries weights and rig); face_src: source face of each new face
    (carries material assignment)."""
    me = bpy.data.meshes.new(name); me.from_pydata(np.asarray(V).tolist(), [], [list(map(int, f)) for f in F])
    if UV is not None:
        lay = me.uv_layers.new(name='UVMap'); buf = np.empty(len(me.loops) * 2)
        for pi, p in enumerate(me.polygons):
            for k, li in enumerate(p.loop_indices):
                buf[2 * li: 2 * li + 2] = UV[FUV[pi][k]]
        lay.data.foreach_set('uv', buf)
    me.update(); ob = bpy.data.objects.new(name, me); ob.matrix_world = src.matrix_world.copy()
    for m in src.data.materials: me.materials.append(m)
    if face_src is not None and len(src.data.materials) > 1:
        mi = np.empty(len(src.data.polygons), np.int64); src.data.polygons.foreach_get('material_index', mi)
        me.polygons.foreach_set('material_index', mi[np.asarray(face_src, np.int64)])
    ctx.collection.objects.link(ob)
    if vmap is not None:
        carry_rig(src, ob, vmap)
    for o in ctx.selected_objects: o.select_set(False)
    ob.select_set(True); ctx.view_layer.objects.active = ob
    src.hide_set(True)
    return ob


def selected_regions(me):
    """Selected faces split into connected regions (each one gets its own isolation loops)."""
    sel = np.zeros(len(me.polygons), bool); me.polygons.foreach_get('select', sel)
    idx = np.flatnonzero(sel).tolist()
    if not idx or len(idx) == len(me.polygons): return []
    e2f = {}
    for g in idx:
        for k in me.polygons[g].edge_keys: e2f.setdefault(k, []).append(g)
    par = {g: g for g in idx}

    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for fs in e2f.values():
        for g in fs[1:]: par[find(g)] = find(fs[0])
    out = {}
    for g in idx: out.setdefault(find(g), []).append(g)
    return list(out.values())


ISO_SCOPE = [('AUTO', "Auto", "Organic: around openings; Hard surface / Mixed: off"),
             ('OPENINGS', "Openings", "Every opening (eye sockets, mouth, holes)"),
             ('ALL', "Openings + selection", "Every opening and every selected face region (select an eye or mouth area)"),
             ('SELECTED', "Selected faces", "Only around the selected face regions (select an eye or a mouth area)"),
             ('OFF', "Off", "")]


def protected_edges(me):
    sharp = me.attributes.get('sharp_edge'); sv = np.zeros(len(me.edges), bool)
    if sharp is not None and sharp.domain == 'EDGE': sharp.data.foreach_get('value', sv)
    return [tuple(e.vertices) for i, e in enumerate(me.edges) if sv[i] or e.use_seam]


PRESETS = [('HARD', "Hard surface", "Feature angle 8 deg: bevel and fillet rows are protected"),
           ('MIXED', "Mixed", "Feature angle 20 deg"),
           ('ORGANIC', "Organic", "Feature angle 35 deg, error-driven row choice")]
PRESET_DEG = {'HARD': 8.0, 'MIXED': 20.0, 'ORGANIC': 35.0}
SYM_ITEMS = [('AUTO', "Auto-detect", "Detect reflective symmetry planes"), ('NONE', "None", ""),
             ('X', "X", "Mirror across the object's X centre plane"), ('Y', "Y", ""), ('Z', "Z", "")]


class RQM_OT_structure_simplify(bpy.types.Operator):
    """Structure-first retopology: keep the model's own edge flow and remove whole edge rings (poly-chords) until the
    face budget is met. Feature lines, sharp marks and UV seams are never merged, so bevels and creases survive, no new
    poles appear, UVs are carried through (islands kept) and symmetric models stay symmetric"""
    bl_idname = "mesh.rqmesh_structure_simplify"
    bl_label = "Simplify by Edge Rings"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Budget", default='RATIO', items=[('RATIO', "Ratio", ""), ('COUNT', "Face count", "")])
    ratio: FloatProperty(name="Keep ratio", default=0.25, min=0.01, max=1.0, subtype='FACTOR')
    target_faces: IntProperty(name="Target faces", default=1000, min=4)
    preset: EnumProperty(name="Shape", items=PRESETS, default='HARD')
    symmetry: EnumProperty(name="Symmetry", items=SYM_ITEMS, default='AUTO')
    keep_uv: BoolProperty(name="Keep UVs", default=True, description="Carry UVs through; seams become protected lines")
    protect_marks: BoolProperty(name="Protect sharp marks and seams", default=True)
    max_error: FloatProperty(name="Max error %", default=0.0, min=0.0, max=10.0,
                             description="Stop early when the next row would deviate more than this % of the object size (0 = off)")
    project: BoolProperty(name="Project to original", default=True)
    order_hint: BoolProperty(name="Triangles in export order", default=True,
                             description="Pair consecutive triangles first when rebuilding quads from a triangulated mesh")
    priority: EnumProperty(name="Remove first", default='AUTO', items=[
        ('AUTO', "Auto", "Hard surface: thinnest rows (designer-like layout); Mixed/Organic: least shape change"),
        ('SHAPE', "Least shape change", "Rank rows by quadric (QEM) error: best geometric accuracy"),
        ('LAYOUT', "Thinnest rows", "Thin rows first (bevel and subdivision rows): closest to a designer cage")])
    placement: EnumProperty(name="Vertex placement", default='OPTIMAL', items=[
        ('OPTIMAL', "Optimal (QEM)", "Quadric-optimal position of each merged vertex, UV from the nearest surface point"),
        ('SURFACE', "On surface", "Midpoint projected onto the original (exact UV at vertices)")])
    local: EnumProperty(name="Local transitions", default='AUTO', items=[
        ('AUTO', "Auto", "Organic: adaptive; Hard/Mixed: only when whole loops are stuck"),
        ('OFF', "Off", "Whole edge loops only (no new poles)"),
        ('STUCK', "When stuck", "Whole loops first; 3-1 transition strips only when no loop can be removed"),
        ('ADAPTIVE', "Adaptive", "Error ladder over loops and 3-1 strips: coarsen where it is flat, keep detail elsewhere")])
    transitions: EnumProperty(name="Transition patterns", default='AUTO', items=[
        ('AUTO', "Auto", "Gentle first for moderate reductions (target >= 30% of the faces), classic 3-1 below"),
        ('MIXED', "Gentle first", "4-2, 5-3, 6-4 strips first (cells grow 2x / 1.7x / 1.5x), 3-1 only where none fits"),
        ('GENTLE', "Gentle only", "Best quad shapes and fewest poles per strip; may stop above the target"),
        ('CLASSIC', "Classic 3-1", "3-1 strips only (cells grow 3x); reaches the lowest error at strong reductions")])
    isolation: EnumProperty(name="Isolation loops around", items=ISO_SCOPE, default='AUTO',
                            description="Keep concentric edge loops around openings (eye sockets, mouth) and selected face "
                                        "regions, as a modeller isolates them for sculpting and animation")
    isolation_loops: IntProperty(name="Loops", default=2, min=1, max=8,
                                 description="Edge loops kept around each opening / region, spaced at the target edge length")

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import structsimp as SS
        from .core import transitions as TR
        from .core import isolation as ISO
        ob = ctx.active_object; me = ob.data
        V, F, UV, FUV = poly_data(me, self.keep_uv)
        extra = protected_edges(me) if self.protect_marks else []
        deg = PRESET_DEG[self.preset]
        n_in = len(F)
        tgt = self.target_faces if self.mode == 'COUNT' else max(4, int(round(n_in * self.ratio)))
        scope = self.isolation
        if scope == 'AUTO': scope = 'OPENINGS' if self.preset == 'ORGANIC' else 'OFF'
        iso_msg = ""
        if scope != 'OFF':
            regions = selected_regions(me) if scope in ('ALL', 'SELECTED') else []
            if scope != 'SELECTED' or regions:
                le, linfo = ISO.isolation_loops(V, F, k=self.isolation_loops, target_faces=tgt, regions=regions,
                                                include_openings=(scope != 'SELECTED'))
                extra = list(extra) + sorted(le)
                kept = [x for x in linfo['seeds'] if x['loops_kept']]
                if kept: iso_msg = f", isolation loops kept around {len(kept)} opening(s)/region(s)"
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        local = self.local.lower()
        if local == 'auto': local = 'adaptive' if self.preset == 'ORGANIC' else 'stuck'
        V2, F2, FUV2, UV2, info = TR.adaptive_simplify(
            V, F, tgt, UV, FUV, feature_deg=deg, symmetry=self.symmetry.lower(), local=local,
            order_hint=self.order_hint, extra_feat=extra,
            cost_mode='qem' if (self.priority == 'SHAPE' or (self.priority == 'AUTO' and self.preset != 'HARD')) else 'width',
            placement='qem' if self.placement == 'OPTIMAL' else 'surface',
            max_error=(self.max_error / 100.0 * diag) if self.max_error > 0 else None,
            transitions={'AUTO': 'auto', 'MIXED': 'mixed', 'GENTLE': 'gentle', 'CLASSIC': '31'}[self.transitions])
        Q = [None] * info.get('faces_in', n_in); n = info.get('rings', 0); st = {}
        new_object_from(ctx, ob, ob.name + "_rings", V2, F2, UV2, FUV2)
        nq = sum(1 for f in F2 if len(f) == 4)
        sw = info.get('strip_widths') or {}
        names = {3: '3-1', 4: '4-2', 5: '5-3', 6: '6-4'}
        strips = ", ".join(f"{v} {names[int(k)]}" for k, v in sorted(sw.items(), key=lambda kv: int(kv[0]))) or "no strips"
        msg = f"{len(Q)} -> {len(F2)} faces ({n} loops removed, {strips}, {nq} quads"
        if info.get('symmetry_planes'): msg += f", {len(info['symmetry_planes'])} symmetry plane(s)"
        msg += iso_msg + ")"
        if st.get('stopped_by_error'): msg += " - stopped at the error limit"
        elif len(F2) > tgt * 1.1:
            msg += (" - stopped early: the remaining loops are fenced by protected lines (features, seams, symmetry"
                    " pairs) or run into triangles; try Local transitions 'When stuck'/'Adaptive' or Symmetry 'None'")
        self.report({'INFO'}, msg)
        warn_modifiers(self, ob)
        return {'FINISHED'}


class RQM_OT_quad_caps(bpy.types.Operator):
    """Replace triangle fans, n-gons and high-valence poles on flat or gently curved caps with a patterned quad grid
    (poles move to the four cap corners, out of the flat face)"""
    bl_idname = "mesh.rqmesh_quad_caps"
    bl_label = "Quad Caps (fans, n-gons)"
    bl_options = {'REGISTER', 'UNDO'}

    cap_angle: FloatProperty(name="Cap flatness", default=12.0, min=1.0, max=60.0,
                             description="Grow the cap over faces within this angle of its normal")
    feature_angle: FloatProperty(name="Feature angle", default=30.0, min=1.0, max=90.0)

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import topofix as TF
        ob = ctx.active_object; V, F, _, _ = poly_data(ob.data, False)
        V2, F2, info = TF.fill_caps(V, F, self.feature_angle, self.cap_angle)
        done = [c for c in info if c['filled']]; odd = [c for c in info if not c['filled']]
        if not info:
            self.report({'INFO'}, "No fans, n-gons or high-valence poles found"); return {'CANCELLED'}
        new_object_from(ctx, ob, ob.name + "_caps", V2, F2)
        msg = f"{len(done)} cap(s) filled: " + ", ".join(f"{c['kind']} {c['n']}-loop -> {c['grid'][0]}x{c['grid'][1]}" for c in done)
        if odd: msg += f"; {len(odd)} skipped (odd boundary count needs a triangle)"
        self.report({'INFO'}, msg + " (UVs are not carried by this prototype)")
        warn_modifiers(self, ob)
        return {'FINISHED'}


class RQM_OT_fix_poles(bpy.types.Operator):
    """Clean up poles: cancel 3-5 pole pairs by dissolving edge loops through them (loop cuts give the faces back),
    then local edge rotations, diagonal collapses and doublet removal, guided by the valence each vertex SHOULD have
    (from the surface angle: 3 on cube-like corners, 4 on flat/smooth areas); finally a feature-preserving relax"""
    bl_idname = "mesh.rqmesh_fix_poles"
    bl_label = "Fix Poles"
    bl_options = {'REGISTER', 'UNDO'}

    feature_angle: FloatProperty(name="Feature angle", default=30.0, min=1.0, max=90.0)
    loops: BoolProperty(name="Allow loop dissolve + cut", default=False,
                        description="Cancel 3-5 pole pairs by dissolving the edge loop through them and cutting a new loop "
                                    "elsewhere (changes the layout; off for hand-made models)")
    keep_count: BoolProperty(name="Keep face count", default=True)
    lookahead: BoolProperty(name="Look-ahead (slower)", default=True)
    refill: BoolProperty(name="Refill pole clusters", default=True,
                         description="Replace a patch holding several poles by the minimal quad pattern its border "
                                     "allows (grid, or one 3-/5-pole); pairs of 3-5 dislocations cancel inside one patch. "
                                     "The border and everything outside stay as they are")
    relax: IntProperty(name="Relax iterations", default=10, min=0, max=100)
    keep_isolation: IntProperty(name="Guard isolation rings", default=2, min=0, max=8,
                                description="Never break the innermost N rings of quads around each opening (eye socket, "
                                            "mouth); edits may still reshape them (0 = off)")
    iso_selected: BoolProperty(name="Also around selected faces", default=False,
                               description="Guard the rings around each selected face region too (select a bump such as a "
                                           "nose or a closed mouth)")

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import topofix as TF
        from .core import structsimp as SS
        ob = ctx.active_object; V, F, _, _ = poly_data(ob.data, False)
        T = [[f[0], f[k], f[k + 1]] for f in F for k in range(1, len(f) - 1)]
        proj = SS.Projector(V, T, SS.feature_edges(V, F, self.feature_angle))
        pf = TF.PoleFixer(V, F, self.feature_angle, extra_feat=protected_edges(ob.data), proj=proj)
        if self.keep_isolation:
            pf.guard_isolation(self.keep_isolation, selected_regions(ob.data) if self.iso_selected else ())
        p0 = pf.poles()
        if self.loops: pf.cancel_by_chords(keep_count=self.keep_count)
        pf.improve(lookahead=self.lookahead)
        if self.refill:
            pf.refill_poles(); pf.improve(lookahead=self.lookahead)
        if self.relax: pf.relax(self.relax)
        V2, F2 = pf.result(); p1 = pf.poles()
        new_object_from(ctx, ob, ob.name + "_poles", V2, F2)
        self.report({'INFO'}, f"misplaced poles {p0} -> {p1}; ops: {pf.stats}. Isolated poles need loop-level edits "
                              "(see 'Select Misplaced Poles')")
        warn_modifiers(self, ob)
        return {'FINISHED'}


class RQM_OT_select_poles(bpy.types.Operator):
    """Select poles whose valence differs from what the surface needs (a 3-pole on a flat area, a 4-valent vertex on
    a cube-like corner...).  Enter Edit Mode afterwards to see them"""
    bl_idname = "mesh.rqmesh_select_poles"
    bl_label = "Select Misplaced Poles"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import topofix as TF
        ob = ctx.active_object; me = ob.data; V, F, _, _ = poly_data(me, False)
        pf = TF.PoleFixer(V, F, 30.0)
        sel = np.zeros(len(me.vertices), bool); kinds = {}
        for v, ideal in pf.ideal.items():
            val = pf.M.valence(v)
            if val != ideal:
                sel[v] = True; kinds[(val, ideal)] = kinds.get((val, ideal), 0) + 1
        me.vertices.foreach_set('select', sel); me.update()
        txt = ", ".join(f"{n}x valence {a} (should be {b})" for (a, b), n in sorted(kinds.items()))
        self.report({'INFO'}, f"{int(sel.sum())} misplaced poles selected: {txt or 'none'}")
        return {'FINISHED'}


class RQM_OT_select_isolation(bpy.types.Operator):
    """Select the isolation rings (closed rings of quads) around every opening and selected face region, and report how
    many each has and whether poles sit on the innermost ones.  Enter Edit Mode afterwards to see them"""
    bl_idname = "mesh.rqmesh_select_isolation"
    bl_label = "Select Isolation Rings"
    bl_options = {'REGISTER', 'UNDO'}

    rings: IntProperty(name="Rings", default=2, min=1, max=16, description="Innermost rings to select per opening / region")
    bumps: BoolProperty(name="Bumps on closed surfaces", default=False,
                        description="Also find nested rings around bumps (nose, ears) on closed parts")

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import isolation as ISO
        ob = ctx.active_object; me = ob.data; V, F, _, _ = poly_data(me, False)
        regions = selected_regions(me)
        band = ISO.protected_band(V, F, k=self.rings, regions=regions, families=self.bumps)
        rep = ISO.report(V, F, regions)
        sel = np.zeros(len(me.polygons), bool); sel[sorted(band['faces'])] = True
        me.polygons.foreach_set('select', sel)
        vs = np.zeros(len(me.vertices), bool); vs[sorted(band['verts'])] = True; me.vertices.foreach_set('select', vs)
        es = np.zeros(len(me.edges), bool)
        for i, e in enumerate(me.edges):
            es[i] = vs[e.vertices[0]] and vs[e.vertices[1]]
        me.edges.foreach_set('select', es); me.update()
        if not rep:
            self.report({'INFO'}, "No openings or selected regions (select faces around a bump to check it)")
            return {'FINISHED'}
        parts = []
        for r in rep:
            on = r['poles_on_inner_rings']
            tag = "clean" if not on else "poles on inner rings: " + ", ".join(f"{n}x{v}" for v, n in on.items())
            parts.append(f"{r['kind']} {r['size']}: {r['rings']} ring(s), {tag}")
        self.report({'INFO'}, "; ".join(parts))
        return {'FINISHED'}


class RQM_OT_recover_base(bpy.types.Operator):
    """Recover the control cage of a subdivided mesh (Catmull-Clark or simple subdivision of triangles, quads or
    n-gons, also when exported as triangles).  The cage is fitted so that it re-subdivides to this mesh"""
    bl_idname = "mesh.rqmesh_recover_base"
    bl_label = "Recover Base Cage"
    bl_options = {'REGISTER', 'UNDO'}

    levels: IntProperty(name="Levels", default=0, min=0, max=6, description="Levels to remove (0 = all that are found)")
    placement: EnumProperty(name="Result", default='CAGE', items=[
        ('CAGE', "Cage (editable)", "Exact control cage; with the modifier it looks like the input"),
        ('FIT', "Low-poly, best fit", "Same faces, vertices placed for the smallest surface error (game LOD)"),
        ('SURFACE', "Low-poly, on surface", "Same faces, vertices at the input points they correspond to")])
    add_modifier: BoolProperty(name="Add Subdivision modifier", default=True,
                               description="Cage only: add a Subdivision Surface modifier with the detected scheme and levels")
    keep_uv: BoolProperty(name="Keep UVs", default=True)
    edited: BoolProperty(name="Surface edited after subdividing", default=False,
                         description="Recover the cage topology even when the surface was sculpted or deformed after "
                                     "subdividing; the cage is fitted to the edited surface (fine detail is lost). "
                                     "Set Levels to the number of times you subdivided: a hand-made cage can look like "
                                     "one more level")
    partial: BoolProperty(name="Keep edited parts", default=False,
                          description="Peel only the parts that are still a subdivision grid. Geometry you added or cut "
                                      "after subdividing (extrude, inset, loop cut, knife, bevel) stays as it is, "
                                      "joined to the recovered faces by an extra vertex on each shared edge")
    tolerance: FloatProperty(name="Tolerance", default=0.075, min=0.0, max=2.0,
                             description="Largest fit residual (in edge lengths) accepted for a level. For a surface "
                                         "edited after subdividing, use 'Surface edited after subdividing' with Levels "
                                         "rather than raising this: a looser limit also accepts hand-made cages as "
                                         "levels; 0 = topology only")

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def _partial(self, ctx, ob, V, F, UV, FUV):
        from .core import partial as PA
        import time
        t0 = time.time()
        r = PA.unsubdivide_partial(V, F, levels=self.levels or 1)
        if not r['levels']:
            self.report({'WARNING'}, "No part of this mesh is a subdivision grid"); warn_modifiers(self, ob, recover=True)
            return {'CANCELLED'}
        cs = r['corner_src']
        FUV2 = [[FUV[g][k] for g, k in c] for c in cs] if FUV is not None else None
        nob = new_object_from(ctx, ob, ob.name + "_base", r['V'], r['F'], UV, FUV2, vmap=r['orig'],
                              face_src=[c[0][0] for c in cs])
        lv = r['levels']
        nob['rqmesh_recover_base'] = json.dumps(dict(levels=lv, partial=True))
        last = lv[-1]
        self.report({'INFO'}, f"{len(F)} -> {len(r['F'])} faces: {len(lv)} level(s), {last['peeled']} faces recovered, "
                              f"{last['kept']} kept as they were, {last['stitch_points']} join points, {last['scheme']}, "
                              f"residual {max(l['residual_rel'] for l in lv):.1e} edge, {time.time() - t0:.1f} s. No "
                              f"Subdivision modifier added: it would subdivide the kept parts again")
        warn_modifiers(self, ob, recover=True)
        return {'FINISHED'}

    def execute(self, ctx):
        from .core import subdiv as SD
        import time
        ob = ctx.active_object; V, F, UV, FUV = poly_data(ob.data, self.keep_uv)
        if self.partial:
            return self._partial(ctx, ob, V, F, UV, FUV)
        Q, how = SD.to_quads_if_subdivided(V, F)
        if Q is None and F and all(len(f) == 3 for f in F):   # 1-to-4 triangle subdivision (Loop, Blender Subdivide)
            from .core import subdiv_tri as STRI
            t0 = time.time(); rt = STRI.unsubdivide_tri(V, F, levels=self.levels or None, tol=None if self.edited else (self.tolerance or None))
            if rt['levels']:
                V2, F2 = rt['cage'] if self.placement == 'CAGE' else rt['surface']
                new_object_from(ctx, ob, ob.name + "_base", V2, F2)
                self.report({'INFO'}, f"{len(F)} -> {len(F2)} triangles, {len(rt['levels'])} level(s) of "
                                      f"{rt['levels'][0]['scheme']} triangle subdivision, {time.time() - t0:.1f} s")
                warn_modifiers(self, ob, recover=True)
                return {'FINISHED'}
        if Q is None:
            self.report({'WARNING'}, f"Not a subdivided mesh: {how}. If you changed the topology after subdividing "
                                     f"(extrude, loop cut, ...), try 'Keep edited parts'")
            warn_modifiers(self, ob, recover=True)
            return {'CANCELLED'}
        if how != 'quads' and FUV is not None:
            FUV = SD.quad_uv_from_tris(Q, F, FUV)
            if FUV is None: UV = None
        t0 = time.time()
        stops = []
        r = SD.unsubdivide(V, Q, levels=self.levels or None, FUV=FUV, UV=UV, tol=None if self.edited else (self.tolerance or None), log=stops.append)
        if not r['levels']:
            self.report({'WARNING'}, "Topology looks subdivided but the fit does not match: if the surface was "
                                     "edited after subdividing, enable 'Surface edited after subdividing'")
            warn_modifiers(self, ob, recover=True); return {'CANCELLED'}
        V2, F2 = r[{'CAGE': 'cage', 'FIT': 'fit', 'SURFACE': 'surface'}[self.placement]]
        uv = r['uv']
        nob = new_object_from(ctx, ob, ob.name + "_base", V2, F2, uv[0] if uv else None, uv[1] if uv else None,
                              vmap=r['orig'])
        lv = r['levels']; sch = lv[0]['scheme']
        ncr = 0
        if r.get('creases') and self.placement == 'CAGE':      # write the recovered creases (crease 1.0)
            me = nob.data; ca = me.attributes.get('crease_edge') or me.attributes.new('crease_edge', 'FLOAT', 'EDGE')
            want = {(min(a, b), max(a, b)) for a, b in r['creases']}
            vals = [1.0 if (min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1])) in want else 0.0
                    for e in me.edges]
            ca.data.foreach_set('value', vals); ncr = int(sum(vals))
        if self.placement == 'CAGE' and self.add_modifier:
            m = nob.modifiers.new("Subdivision", 'SUBSURF'); m.levels = m.render_levels = len(lv)
            m.subdivision_type = 'SIMPLE' if sch == 'simple' else 'CATMULL_CLARK'
            m.use_limit_surface = 'limit' in sch
            um = (lv[0].get('uv_mode') or '').split('+')[0]
            uvs = {'cc/linear': 'PRESERVE_BOUNDARIES', 'cc/smooth': 'PRESERVE_CORNERS', 'cc/all': 'SMOOTH_ALL',
                   'simple/linear': 'NONE'}.get(um)
            if uvs:
                try: m.uv_smooth = uvs
                except Exception: pass
        nob['rqmesh_recover_base'] = json.dumps(dict(levels=lv, paired=how))
        deg = {}
        for f in F2: deg[len(f)] = deg.get(len(f), 0) + 1
        res = [l['residual_rel'] for l in lv]
        if self.edited and not self.levels:
            # measured: with Levels set, 74 of 76 held-out edited cages were recovered; counting levels from the
            # topology alone was right in 36 of 76, because a hand-made cage can look like one more level
            if len(res) >= 2 and res[-1] > SD.LEVEL_TOL and res[-1] > 5 * res[-2]:
                self.report({'WARNING'}, f"The last level fits much worse ({res[-1]:.3f} vs {res[-2]:.3f}): it may be "
                                         f"your hand-made cage, not a subdivision. If you subdivided {len(res) - 1} "
                                         f"time(s), undo and set Levels to {len(res) - 1}")
            else:
                self.report({'WARNING'}, f"Peeled {len(res)} level(s) from the topology alone. If you subdivided a "
                                         f"different number of times, undo and set Levels to that number")
        hint = ""
        stops = [m for m in stops if m.startswith('stop:')]
        if not self.edited and not self.levels and stops:   # a hand-made cage stops here too (residual 0.09-0.2)
            hint = (f"; stopped before a loosely fitting level ({stops[-1].split('residual ')[-1].split(' (')[0]}): "
                    f"if you subdivided more times and edited afterwards, use 'Surface edited after subdividing' "
                    f"with Levels")
        self.report({'INFO'}, f"{len(F)} -> {len(F2)} faces ({', '.join(f'{v} {k}-gons' for k, v in sorted(deg.items()))}), "
                              f"{len(lv)} level(s), {sch}, residual {max(l['residual_rel'] for l in lv):.1e} edge, "
                              + (f"{ncr} crease edges, " if ncr else "")
                              + ("APPROXIMATE (edited after subdividing, or semi-sharp creases with Limit Surface), "
                                 if max(l['residual_rel'] for l in lv) > 1e-3 else "")
                              + f"{time.time() - t0:.1f} s" + ("" if how == 'quads' else f"; {how}") + hint)
        warn_modifiers(self, ob, recover=True)
        return {'FINISHED'}


class RQM_OT_analyze(bpy.types.Operator):
    """Measure the mesh's modelling structure (quads, regular vertices, clean edge loops) and recommend a method --
    edge loops, adaptive loops with transitions, or Decimate / remesh -- the way a hybrid sort inspects its input"""
    bl_idname = "mesh.rqmesh_analyze"
    bl_label = "Analyze Structure"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import analyze as AN
        ob = ctx.active_object; V, F, _, _ = poly_data(ob.data, False)
        sc = AN.structure_scores(V, F); m, why = AN.recommend(sc)
        ob['rqmesh_structure'] = json.dumps(dict(sc, method=m, faces=len(ob.data.polygons)))
        names = {'recover_base_first': "Recover Base Cage", 'recover_edited': "Recover Base Cage (surface edited)", 'edge_loops': "Simplify by Edge Rings (Hard surface)", 'adaptive_loops': "Simplify by Edge Rings (Organic / Adaptive)",
                 'decimate_or_remesh': "Blender Decimate, or Remesh (QuadWild backend)",
                 'quad_caps_first': "Quad Caps"}
        sub = f"subdivided x{sc['subdiv_levels']}, " if sc.get('subdiv_levels') else ""
        self.report({'INFO'}, sub + f"quads {sc['quad_share']:.0%}, regular {sc['regular_share']:.0%}, clean loops "
                              f"{sc['loop_share']:.0%} (length {sc['loop_len']:.2f}), regular blocks {sc['block_share']:.0%} "
                              f"-> {names[m]}: {why}")
        warn_modifiers(self, ob)
        return {'FINISHED'}


class RQM_OT_detect_symmetry(bpy.types.Operator):
    """Detect reflective symmetry planes (bounding-box and principal axes, refined) and per-part principal axes;
    switches on Blender's mesh symmetry for axis-aligned planes"""
    bl_idname = "mesh.rqmesh_detect_symmetry"
    bl_label = "Detect Symmetry"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        from .core import symmetry as SYM
        ob = ctx.active_object; me = ob.data; V, F, _, _ = poly_data(me, False)
        planes = SYM.detect_planes(V, F); parts = SYM.part_axes(V, F)
        ob['rqmesh_symmetry'] = json.dumps([dict(n=n.tolist(), d=d, score=sc) for n, d, sc in planes])
        txt = []
        for n, d, sc in planes:
            ax = int(np.argmax(np.abs(n)))
            if abs(n[ax]) > 0.999 and abs(d) < 1e-4 * float(np.linalg.norm(V.max(0) - V.min(0))):
                setattr(me, 'use_mirror_' + 'xyz'[ax], True)
            txt.append(f"{'XYZ'[ax] if abs(n[ax]) > 0.999 else np.round(n, 3).tolist()} ({sc:.0%})")
        self.report({'INFO'}, f"{len(planes)} symmetry plane(s): {', '.join(txt) or 'none'}; {len(parts)} part(s)")
        return {'FINISHED'}


# ----------------------------------------------------------------------------- UI
class RQM_OT_backend_check(bpy.types.Operator):
    """Find the QuadWild-BiMDF back-end and run a short self-test in a separate process"""
    bl_idname = "rqmesh.backend_check"; bl_label = "Check Back-end"

    def execute(self, ctx):
        prefs = ctx.preferences.addons[__package__].preferences
        d, src = BK.find_quadwild(bpy.path.abspath(prefs.qwlib_dir), __package__, HERE)
        if d is None:
            self.report({'WARNING'}, f"{src}; Remesh will use Blender's QuadriFlow"); return {'CANCELLED'}
        ok, msg = BK.self_test(d, os.path.join(HERE, 'backend', 'qw_driver.py'), prefs.python or sys.executable)
        self.report({'INFO'} if ok else {'ERROR'}, f"QuadWild-BiMDF ({src}): {msg}")
        return {'FINISHED'} if ok else {'CANCELLED'}


class RQM_OT_backend_install_file(bpy.types.Operator, ImportHelper):
    """Install a QuadWild-BiMDF back-end package (.zip with rqm_backend.json) that you downloaded yourself.
    Platform and every file's SHA-256 are checked, then a self-test runs"""
    bl_idname = "rqmesh.backend_install_file"; bl_label = "Install Back-end from File"
    filename_ext = ".zip"
    filter_glob: StringProperty(default="*.zip", options={'HIDDEN'})

    def execute(self, ctx):
        prefs = ctx.preferences.addons[__package__].preferences
        try:
            d, man = BK.install_from_zip(self.filepath, __package__)
        except Exception as e:
            self.report({'ERROR'}, f"Not installed: {e}"); return {'CANCELLED'}
        ok, msg = BK.self_test(d, os.path.join(HERE, 'backend', 'qw_driver.py'), prefs.python or sys.executable)
        self.report({'INFO'} if ok else {'WARNING'}, f"Installed {man.get('name')} {man.get('version')} "
                                                       f"({man.get('license')}): {msg}")
        return {'FINISHED'}


class RQM_OT_backend_download(bpy.types.Operator):
    """Download the QuadWild-BiMDF back-end package listed in the index you set, verify its SHA-256 and install it.
    Only in the self-hosted build, only with Online Access enabled, only after you confirm"""
    bl_idname = "rqmesh.backend_download"; bl_label = "Download Back-end"

    @classmethod
    def poll(cls, ctx):
        prefs = ctx.preferences.addons[__package__].preferences
        return BK.ALLOW_DOWNLOAD and bpy.app.online_access and bool(prefs.index_url.strip())

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_confirm(self, event, title="Download QuadWild-BiMDF (GPL-3.0)?",
                                                 message=ctx.preferences.addons[__package__].preferences.index_url)

    def execute(self, ctx):
        prefs = ctx.preferences.addons[__package__].preferences
        try:
            idx = BK.fetch_index(prefs.index_url.strip())
            asset = next(a for a in idx.get('assets', []) if a.get('platform') == BK.platform_tag())
            path = BK.download(asset['url'])
            d, man = BK.install_from_zip(path, __package__, expect_sha=asset['sha256'])
        except StopIteration:
            self.report({'ERROR'}, f"The index has no package for {BK.platform_tag()}"); return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Not installed: {e}"); return {'CANCELLED'}
        ok, msg = BK.self_test(d, os.path.join(HERE, 'backend', 'qw_driver.py'), prefs.python or sys.executable)
        self.report({'INFO'} if ok else {'WARNING'}, f"Installed {man.get('version')}: {msg}")
        return {'FINISHED'}


class RQM_OT_backend_remove(bpy.types.Operator):
    """Remove the back-end installed by this add-on (folders set by hand are left alone)"""
    bl_idname = "rqmesh.backend_remove"; bl_label = "Remove Installed Back-end"

    def execute(self, ctx):
        import shutil
        d = BK.user_backend_dir(__package__)
        if os.path.isdir(d): shutil.rmtree(d); self.report({'INFO'}, "Removed")
        else: self.report({'INFO'}, "Nothing installed")
        return {'FINISHED'}


class RQM_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    qwlib_dir: StringProperty(name="Use folder instead", subtype='DIR_PATH',
                              description="Optional: a folder that contains 'qwlib' (e.g. from QRemeshify)")
    index_url: StringProperty(name="Back-end index URL",
                           default="https://github.com/sandner-art/rqmesh-backend-quadwild/releases/latest/download/index.json",
                              description="Self-hosted build only: JSON index listing back-end packages and SHA-256")
    python: StringProperty(name="Python for back-end", default="", description="Empty = Blender's Python")
    timeout: IntProperty(name="Timeout (s)", default=600, min=10)

    def draw(self, ctx):
        L = self.layout
        box = L.box(); d, src = BK.find_quadwild(bpy.path.abspath(self.qwlib_dir), __package__, HERE)
        box.label(text=f"QuadWild-BiMDF back-end ({BK.platform_tag()}): " + (f"found - {src}" if d else "not installed"),
                  icon='CHECKMARK' if d else 'INFO')
        if not d:
            box.label(text="Optional. Without it, Remesh uses Blender's QuadriFlow; all structure-first tools work.")
        row = box.row(); row.operator(RQM_OT_backend_check.bl_idname, icon='PLAY')
        row.operator(RQM_OT_backend_install_file.bl_idname, icon='FILE_FOLDER')
        row.operator(RQM_OT_backend_remove.bl_idname, icon='TRASH')
        if BK.ALLOW_DOWNLOAD:
            r2 = box.row(); r2.prop(self, 'index_url')
            r3 = box.row(); r3.enabled = bpy.app.online_access; r3.operator(RQM_OT_backend_download.bl_idname, icon='IMPORT')
            if not bpy.app.online_access:
                box.label(text="Online Access is off (Preferences > System): download disabled", icon='LOCKED')
        box.label(text="Back-end license: GPL-3.0-or-later (source links in the package's SOURCE.txt)")
        c = L.column(); c.prop(self, 'qwlib_dir'); c.prop(self, 'python'); c.prop(self, 'timeout')


class RQM_PT_panel(bpy.types.Panel):
    bl_label = "RQMesh"; bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = "RQMesh"

    def draw(self, ctx):
        ob = ctx.active_object
        if ob is None or ob.type != 'MESH':
            self.layout.label(text="Select a mesh object", icon='INFO'); return
        try:
            analysis = json.loads(ob['rqmesh_structure']) if 'rqmesh_structure' in ob else None
        except (ValueError, TypeError):
            analysis = None
        WF.draw(self.layout, ob, analysis)


class RQM_PT_tools(bpy.types.Panel):
    bl_label = "All Tools"; bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = "RQMesh"
    bl_parent_id = "RQM_PT_panel"; bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        c = self.layout.column(align=True)
        c.label(text="Inspect")
        c.operator(RQM_OT_analyze.bl_idname, icon='VIEWZOOM')
        c.operator(RQM_OT_select_poles.bl_idname, icon='RESTRICT_SELECT_OFF')
        c.operator(RQM_OT_select_isolation.bl_idname, icon='MESH_CIRCLE')
        c.operator(RQM_OT_detect_symmetry.bl_idname, icon='MOD_MIRROR')
        c.operator(RQM_OT_detect_features.bl_idname, icon='EDGESEL')
        c.separator(); c.label(text="Structure-first (keeps your edge flow)")
        c.operator(RQM_OT_recover_base.bl_idname, icon='MOD_SUBSURF')
        c.operator(RQM_OT_quad_caps.bl_idname, icon='MESH_GRID')
        c.operator(RQM_OT_structure_simplify.bl_idname, icon='MOD_DECIM')
        c.operator(RQM_OT_fix_poles.bl_idname, icon='VERTEXSEL')
        c.separator(); c.label(text="New topology (sculpts, scans)")
        c.operator(RQM_OT_remesh_quadwild.bl_idname, icon='MOD_REMESH')
        c.operator(RQM_OT_deformation_field.bl_idname, icon='ORIENTATION_GIMBAL')


classes = (RQM_OT_detect_features, RQM_OT_deformation_field, RQM_OT_remesh_quadwild, RQM_OT_structure_simplify,
           RQM_OT_quad_caps, RQM_OT_fix_poles, RQM_OT_select_poles, RQM_OT_select_isolation, RQM_OT_detect_symmetry, RQM_OT_analyze, RQM_OT_recover_base,
           RQM_OT_backend_check, RQM_OT_backend_install_file, RQM_OT_backend_download, RQM_OT_backend_remove,
           RQM_Preferences, RQM_PT_panel, RQM_PT_tools)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
