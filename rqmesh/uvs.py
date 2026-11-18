# SPDX-License-Identifier: GPL-3.0-or-later
"""UVs from Structure: a UV map and panel data from the recovered subdivision cage, or a shape-aware unwrap of a hard-
surface model's parts (core uvlayout)."""
import numpy as np
import bpy
from .core import subdiv, uvlayout as UL, masks as MK

UV_NAME = "RQMesh UV"


def mesh_vf(me):
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
    return co.reshape(-1, 3), [list(p.vertices) for p in me.polygons]


def write_uv(me, uv, name=UV_NAME):
    lay = me.uv_layers.get(name) or me.uv_layers.new(name=name)
    lay.data.foreach_set('uv', np.concatenate([np.asarray(u, float) for u in uv]).ravel())
    me.uv_layers.active = lay; return lay


def set_seams(me, F, uv):
    cor = {(i, v): uv[i][k] for i, f in enumerate(F) for k, v in enumerate(f)}; ef = {}
    for i, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; ef.setdefault((min(a, b), max(a, b)), []).append(i)
    cut = {e for e, fs in ef.items() if len(fs) == 2 and any(np.linalg.norm(cor[(fs[0], v)] - cor[(fs[1], v)]) > 1e-9 for v in e)}
    for e in me.edges:
        e.use_seam = tuple(sorted(e.vertices)) in cut


def edit_all(ctx, ob, fn):
    for o in ctx.selected_objects:
        o.select_set(False)
    ob.select_set(True); ctx.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    try:
        fn()
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def demo_material(ob, attr):
    """A material that varies by panel or part: random colour per id, darker near panel edges."""
    mat = bpy.data.materials.new(f"RQMesh {attr}"); mat.use_nodes = True; nt = mat.node_tree
    bsdf = nt.nodes.get('Principled BSDF')
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = attr; at.attribute_type = 'GEOMETRY'
    wn = nt.nodes.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '1D'
    nt.links.new(at.outputs['Fac'], wn.inputs['W']); col = wn.outputs['Color']
    if attr == 'rqmesh_panel':                         # panel edges: min(u, 1-u, v, 1-v) from the panel coordinates
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'rqmesh_panel_uv'
        sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(uvn.outputs['UV'], sep.inputs[0])
        m = []
        for c in ('X', 'Y'):
            inv = nt.nodes.new('ShaderNodeMath'); inv.operation = 'SUBTRACT'; inv.inputs[0].default_value = 1.0
            nt.links.new(sep.outputs[c], inv.inputs[1])
            mn = nt.nodes.new('ShaderNodeMath'); mn.operation = 'MINIMUM'
            nt.links.new(sep.outputs[c], mn.inputs[0]); nt.links.new(inv.outputs[0], mn.inputs[1]); m.append(mn)
        both = nt.nodes.new('ShaderNodeMath'); both.operation = 'MINIMUM'
        nt.links.new(m[0].outputs[0], both.inputs[0]); nt.links.new(m[1].outputs[0], both.inputs[1])
        ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.inputs['From Max'].default_value = 0.08
        ramp.inputs['To Min'].default_value = 0.25; nt.links.new(both.outputs[0], ramp.inputs['Value'])
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
        mix.inputs['Factor'].default_value = 1.0
        nt.links.new(col, mix.inputs['A']); nt.links.new(ramp.outputs['Result'], mix.inputs['B']); col = mix.outputs['Result']
    nt.links.new(col, bsdf.inputs['Base Color'])
    ob.data.materials.append(mat)


class RQM_OT_structure_uv(bpy.types.Operator):
    bl_idname = "mesh.rqmesh_structure_uv"
    bl_label = "UVs from Structure"
    bl_description = ("A new UV map ('RQMesh UV') from the mesh's structure: seams from the recovered subdivision cage, "
                      "straight-loop or per-panel layouts, or a shape-aware unwrap of a hard-surface model's parts. "
                      "Also stores panel ids and panel coordinates for materials")
    bl_options = {'REGISTER', 'UNDO'}

    method: bpy.props.EnumProperty(name="Method", default='SEAMS', items=[
        ('SEAMS', "Cage seams", "Subdivided meshes: cut along the cage's loops from its poles, creases and n-gons, merge "
                                "the pieces while each stays nearly flat, then Blender's Unwrap"),
        ('TRIM', "Trim sheet", "Subdivided meshes: every strip of quads along the loops laid onto the band of a trim "
                               "sheet that fits its height, along u. Stores the band in 'rqmesh_trim_band'"),
        ('STRAIGHT', "Straight loops", "Subdivided meshes: every patch of the cage laid out as a straight grid, so "
                                       "the texture runs exactly along the loops (trim sheets, tiling). Distorts "
                                       "panels that aren't rectangles"),
        ('PANELS', "Panels", "Subdivided meshes: every cage face its own island, at its true size"),
        ('PARTS', "Parts (hard surface)", "Any mesh of rigid parts: cylinders unwrapped around their axis (seam on "
                                          "the inner side), caps and other pieces box-projected in their own frame")])
    levels: bpy.props.IntProperty(name="Levels", default=0, min=0, max=4,
                                  description="Subdivision levels to look through (0 = detect)")
    panel_data: bpy.props.BoolProperty(
        name="Store panel data", default=True,
        description="Subdivided meshes: face attribute 'rqmesh_panel' (the cage face) and UV map 'rqmesh_panel_uv' "
                    "(position inside the panel, quads 0-1), for materials and Geometry Nodes")
    demo: bpy.props.BoolProperty(name="Add a variation material", default=False,
                                 description="A material with a random colour per panel (or part) and darker panel edges")
    margin: bpy.props.FloatProperty(name="Island margin", default=0.005, min=0.0, max=0.1)
    merge: bpy.props.FloatProperty(
        name="Merge up to", default=0.5, min=0.0, max=6.3, subtype='ANGLE',
        description="Cage seams: merge pieces while the curvature inside each stays under this (0 = no merging)")
    bands: bpy.props.StringProperty(
        name="Trim bands", default="0.5 0.25 0.125 0.125",
        description="Trim sheet: heights of its bands from the top, as fractions of the sheet")
    group: bpy.props.BoolProperty(name="Join strips", default=True,
                                  description="Trim sheet: adjacent strips join to fill taller bands (halved the "
                                              "texel-density error on our test meshes)")
    unit: bpy.props.EnumProperty(name="Trim unit", default='strip', items=[
        ('strip', "Strips", "One row of cage quads along the loops per trim placement"),
        ('patch', "Patches", "A whole grid patch per trim placement")])

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return ctx.mode == 'OBJECT' and o is not None and o.type == 'MESH'

    def execute(self, ctx):
        ob = ctx.active_object; me = ob.data; V, F = mesh_vf(me)
        if self.method == 'PARTS':
            uv, seg, kinds = UL.part_uvs(V, F)
            a = me.attributes.get('rqmesh_piece') or me.attributes.new('rqmesh_piece', 'INT', 'FACE')
            a.data.foreach_set('value', seg.astype(np.int32))
            write_uv(me, uv); edit_all(ctx, ob, lambda: bpy.ops.uv.pack_islands(rotate=False, margin=self.margin))
            nc = sum(1 for k in kinds.values() if k == 'cylinder')
            msg = f"{len(kinds)} pieces ({nc} cylinders)"; attr = 'rqmesh_piece'
        else:
            if any(len(f) != 4 for f in F):
                self.report({'ERROR'}, "Needs an all-quad subdivided mesh (Catmull-Clark or Simple). Use Parts, or "
                                       "Recover Base Cage first"); return {'CANCELLED'}
            tr = []; r = subdiv.unsubdivide(V, F, levels=self.levels or None, trace=tr)
            if not r['levels']:
                self.report({'ERROR'}, "No subdivision found to look through. Use Parts, or set Levels"); return {'CANCELLED'}
            cV, cF = r['cage']
            if self.panel_data:
                pc, pid = UL.panels(tr, cF)
                a = me.attributes.get('rqmesh_panel') or me.attributes.new('rqmesh_panel', 'INT', 'FACE')
                a.data.foreach_set('value', pid.astype(np.int32)); write_uv(me, pc, 'rqmesh_panel_uv')
            if self.method == 'TRIM':
                try:
                    hs = [float(x) for x in self.bands.replace(',', ' ').split()]
                except ValueError:
                    self.report({'ERROR'}, "Trim bands: numbers separated by spaces, e.g. 0.5 0.25 0.25"); return {'CANCELLED'}
                rows, top = [], 1.0
                for h in hs:
                    rows.append((max(top - h, 0.0), top)); top -= h
                cu, uid, band, err = UL.trim_layout(cV, cF, rows, unit=self.unit, cuts=r['creases'], group=self.group)
                duv, pid = UL.push_corners(tr, cu); write_uv(me, duv)
                a = me.attributes.get('rqmesh_trim_band') or me.attributes.new('rqmesh_trim_band', 'INT', 'FACE')
                a.data.foreach_set('value', band[pid].astype(np.int32))
                if self.demo:
                    trim_material(ob, rows)
                self.report({'INFO'}, f"UV map '{UV_NAME}': {len(set(uid.tolist()))} strips on {len(rows)} bands; "
                                      f"texel density off by x{2 ** float(np.median(np.abs(err))):.2f} (median)")
                return {'FINISHED'}
            cu, ch = UL.cage_layout(cV, cF, 'panels' if self.method == 'PANELS' else 'patches', cuts=r['creases'])
            duv, _ = UL.push_corners(tr, cu)
            if self.method == 'SEAMS':
                if self.merge > 0:
                    ch = UL.merge_charts(cV, cF, ch, cuts=r['creases'], max_curv=self.merge)
                    _, pid = UL.panels(tr, cF); duv = [np.full((len(f), 2), float(ch[pid[i]])) for i, f in enumerate(F)]
                set_seams(me, F, duv); lay = me.uv_layers.get(UV_NAME) or me.uv_layers.new(name=UV_NAME)
                me.uv_layers.active = lay
                edit_all(ctx, ob, lambda: bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=self.margin))
            else:
                write_uv(me, duv); edit_all(ctx, ob, lambda: bpy.ops.uv.pack_islands(rotate=False, margin=self.margin))
            msg = f"{len(r['levels'])} levels, {len(cF)} panels, {len(set(ch.tolist()))} islands"; attr = 'rqmesh_panel'
        if me.uv_layers.get(UV_NAME):
            me.uv_layers.active = me.uv_layers[UV_NAME]
            for lay in me.uv_layers:
                lay.active_render = lay.name == UV_NAME
        if self.demo and (self.method == 'PARTS' or self.panel_data):
            demo_material(ob, attr)
        self.report({'INFO'}, f"UV map '{UV_NAME}': {msg}")
        return {'FINISHED'}


def trim_image(rows, n=512):
    """A procedural trim sheet: a different pattern on each band (plates with rivets, hazard stripes, ribs, bolts)."""
    v = (np.arange(n) + 0.5) / n; u = v.copy(); U, Vv = np.meshgrid(u, v)
    img = np.ones((n, n, 4)); img[..., :3] = 0.35
    for b, (v0, v1) in enumerate(rows):
        m = (Vv >= v0) & (Vv < v1); t = (Vv - v0) / max(v1 - v0, 1e-9); h = v1 - v0; k = b % 4
        x = U / h                                    # u in band heights: patterns keep their aspect
        if k == 0:
            c = np.where((np.abs(t - 0.5) > 0.44) | ((x % 2.0) < 0.03), 0.18, 0.55)[..., None] * np.array([0.75, 0.82, 0.9])
            r = np.hypot((x % 1.0) - 0.5, t - 0.86); c = np.where((r < 0.05)[..., None], np.array([0.85, 0.85, 0.8]), c)
        elif k == 1:
            c = np.where((((x + t) * 2.0) % 1.0 < 0.5)[..., None], np.array([0.95, 0.75, 0.1]), np.array([0.08, 0.08, 0.08]))
        elif k == 2:
            c = (0.35 + 0.25 * np.cos(x * 2 * np.pi * 3))[..., None] * np.array([0.8, 0.8, 0.85])
        else:
            r = np.hypot((x % 1.0) - 0.5, t - 0.5)
            c = np.where((r < 0.28)[..., None], np.array([0.7, 0.7, 0.68]), np.array([0.25, 0.27, 0.3]))
        img[..., :3] = np.where(m[..., None], c, img[..., :3])
    im = bpy.data.images.new("RQMesh trim", n, n); im.pixels.foreach_set(img.astype(np.float32).ravel()); im.pack()
    return im


def trim_material(ob, rows):
    mat = bpy.data.materials.new("RQMesh trim"); mat.use_nodes = True; nt = mat.node_tree
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = trim_image(rows)
    uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = UV_NAME; nt.links.new(uvn.outputs['UV'], tex.inputs['Vector'])
    nt.links.new(tex.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color']); ob.data.materials.append(mat)


def worn_material(ob):
    """Paint that varies smoothly (rqmesh_variance), darkens in cavities (rqmesh_cavity) and wears to metal on
    convex edges (rqmesh_wear, broken up by the variance); a slight tint per part when part ids exist."""
    me = ob.data; mat = bpy.data.materials.new("RQMesh worn paint"); mat.use_nodes = True; nt = mat.node_tree
    bsdf = nt.nodes['Principled BSDF']; N = nt.nodes.new; L = nt.links.new

    def attr(name):
        a = N('ShaderNodeAttribute'); a.attribute_name = name; a.attribute_type = 'GEOMETRY'; return a.outputs['Fac']

    def math(op, a, b):
        m = N('ShaderNodeMath'); m.operation = op
        for i, x in enumerate((a, b)):
            (L(x, m.inputs[i]) if not isinstance(x, (int, float)) else m.inputs[i].__setattr__('default_value', x))
        return m.outputs[0]

    def mix(fac, a, b, blend='MIX'):
        m = N('ShaderNodeMix'); m.data_type = 'RGBA'; m.blend_type = blend
        for sock, x in (('Factor', fac), ('A', a), ('B', b)):
            (L(x, m.inputs[sock]) if not isinstance(x, (int, float, tuple)) else m.inputs[sock].__setattr__('default_value', x))
        return m.outputs['Result']
    ids = next((n for n in ('rqmesh_part', 'rqmesh_piece') if n in me.attributes), None)   # parts; panels stay smooth
    base = (0.18, 0.32, 0.55, 1.0)
    if ids:                                          # a slightly different paint per panel or part
        wn = N('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '1D'; L(attr(ids), wn.inputs['W'])
        base = mix(0.12, base, wn.outputs['Color'])
    var = attr('rqmesh_variance')
    shade = math('ADD', math('MULTIPLY', var, 0.5), 0.75)
    paint = mix(1.0, base, _gray(N, L, shade), 'MULTIPLY')
    dirt = attr('rqmesh_cavity')
    if 'rqmesh_exposure' in me.attributes:           # hidden places collect dirt too; weather fades the paint
        dirt = math('MAXIMUM', dirt, math('MULTIPLY', math('SUBTRACT', 1.0, attr('rqmesh_exposure')), 1.4))
        paint_fade = attr('rqmesh_weather')
    cav = math('SUBTRACT', 1.0, math('MULTIPLY', math('MINIMUM', dirt, 1.0), 0.6))
    paint = mix(1.0, paint, _gray(N, L, cav), 'MULTIPLY')
    if 'rqmesh_exposure' in me.attributes:
        paint = mix(math('MULTIPLY', paint_fade, 0.45), paint, (0.82, 0.8, 0.74, 1.0))
    worn = math('ADD', attr('rqmesh_wear'), math('MULTIPLY', math('SUBTRACT', var, 0.5), 0.6))
    mr = N('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.45; mr.inputs['From Max'].default_value = 0.6
    L(worn, mr.inputs['Value']); w = mr.outputs['Result']
    L(mix(w, paint, (0.62, 0.62, 0.64, 1.0)), bsdf.inputs['Base Color']); L(w, bsdf.inputs['Metallic'])
    rough = N('ShaderNodeMapRange'); rough.inputs['To Min'].default_value = 0.55; rough.inputs['To Max'].default_value = 0.28
    L(w, rough.inputs['Value']); L(rough.outputs['Result'], bsdf.inputs['Roughness'])
    me.materials.clear(); me.materials.append(mat)


def _gray(N, L, x):
    c = N('ShaderNodeCombineColor'); [L(x, c.inputs[i]) for i in range(3)]; return c.outputs[0]


class RQM_OT_surface_masks(bpy.types.Operator):
    bl_idname = "mesh.rqmesh_surface_masks"
    bl_label = "Surface Masks"
    bl_description = ("Per-vertex masks for materials: 'rqmesh_wear' (convex edges, where paint wears off), "
                      "'rqmesh_cavity' (concave creases, where dirt collects) and 'rqmesh_variance' (smooth variation; on "
                      "a subdivided mesh it follows the cage's panels and loops). They work in EEVEE and Cycles")
    bl_options = {'REGISTER', 'UNDO'}
    wear: bpy.props.FloatProperty(name="Wear", default=8.0, min=0.0, max=50.0, subtype='PERCENTAGE',
                                  description="Share of the surface (by area) on the most convex edges that wears")
    cavity: bpy.props.FloatProperty(name="Cavity", default=8.0, min=0.0, max=50.0, subtype='PERCENTAGE',
                                    description="Share of the surface (by area) in the deepest creases that gets dirt")
    detail: bpy.props.FloatProperty(name="Variation detail", default=0.35, min=0.0, max=1.0,
                                    description="How much finer levels add to the variation (0 = only large, smooth changes)")
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0)
    use_exposure: bpy.props.BoolProperty(
        name="Exposure", default=True,
        description="Also 'rqmesh_exposure' (how open each point is: ambient occlusion, matching Cycles' AO bake at "
                    "0.87-0.97 correlation on our tests) and 'rqmesh_weather' (open and facing up, where sun and rain "
                    "fade paint); exposed edges wear more")
    radius: bpy.props.FloatProperty(name="Exposure radius", default=15.0, min=1.0, max=100.0, subtype='PERCENTAGE',
                                    description="How far around a point counts for its exposure, as a share of the model's size")
    material: bpy.props.BoolProperty(name="Add a worn-paint material", default=False,
                                     description="Paint that varies, darkens in cavities and wears to metal on edges")

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return ctx.mode == 'OBJECT' and o is not None and o.type == 'MESH'

    def execute(self, ctx):
        ob = ctx.active_object; me = ob.data; V, F = mesh_vf(me)
        w, c = MK.wear_cavity(V, F, self.wear / 100.0, self.cavity / 100.0)
        how = "diffused"
        tr = []
        if all(len(f) == 4 for f in F):
            r = subdiv.unsubdivide(V, F, trace=tr)
        if tr:
            var = MK.structure_noise(tr, len(V), falloff=self.detail, seed=self.seed); how = f"through {len(tr)} levels of the cage"
        else:
            var = MK.diffusion_noise(V, F, seed=self.seed)
        out = [('rqmesh_wear', w), ('rqmesh_cavity', c), ('rqmesh_variance', var)]
        if self.use_exposure:
            ex = MK.exposure(V, F, radius=self.radius / 100.0)
            nw = MK.vertex_normals(V, F) @ np.array(ob.matrix_world.to_3x3().inverted().transposed()).T   # world normals
            up = nw[:, 2] / np.maximum(np.linalg.norm(nw, axis=1), 1e-30)
            out = [('rqmesh_wear', w * (0.4 + 0.6 * ex)), ('rqmesh_cavity', c), ('rqmesh_variance', var),
                   ('rqmesh_exposure', ex), ('rqmesh_weather', np.clip(up, 0, 1) * ex)]
        for name, x in out:
            a = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
            a.data.foreach_set('value', np.asarray(x, np.float32))
        if self.material:
            worn_material(ob)
        self.report({'INFO'}, f"Masks rqmesh_wear, rqmesh_cavity, rqmesh_variance ({how})")
        return {'FINISHED'}
