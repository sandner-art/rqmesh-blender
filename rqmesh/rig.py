# SPDX-License-Identifier: GPL-3.0-or-later
"""Rig Hard Surface: bones, joint locks and rigid weights for a model of rigid parts, from its structure
(core rigid_rig).  Works on all selected mesh objects together; objects shown as Wire or Bounds are helpers: they mark
hidden joints and get no weights."""
import collections
import numpy as np
import bpy
from mathutils import Vector
from .core import rigid_rig as RR

PALETTE = [(0.85, 0.45, 0.2), (0.25, 0.55, 0.85), (0.35, 0.7, 0.35), (0.85, 0.75, 0.25), (0.65, 0.4, 0.75),
           (0.4, 0.75, 0.75), (0.8, 0.4, 0.5), (0.55, 0.55, 0.55), (0.6, 0.8, 0.3), (0.3, 0.4, 0.7)]
HELP = ["Select all the model's mesh objects, then Rig.",
        "Hinge: a cylinder pin through both parts.",
        "Slide: a rod entering a sleeve, or a panel",
        "  held between two rails.",
        "Ball: a sphere touching both parts.",
        "Piston: rod and sleeve, each pinned to a part.",
        "Hidden joint? Add a cylinder (or sphere)",
        "  there and set Viewport Display > Display As",
        "  to Wire: it marks the joint, gets no weights.",
        "Check the parts first: Only show the parts."]


def gather(objs):
    """World-space vertices and faces of all objects, the object of each face, and helper flags."""
    V, F, fobj, helper, offs = [], [], [], [], []
    off = 0
    for k, ob in enumerate(objs):
        me = ob.data; M = np.array(ob.matrix_world)
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
        V.append(co @ M[:3, :3].T + M[:3, 3])
        is_helper = ob.display_type in {'WIRE', 'BOUNDS'}
        for p in me.polygons:
            F.append([off + i for i in p.vertices]); fobj.append(k); helper.append(is_helper)
        offs.append(off); off += len(me.vertices)
    return np.vstack(V) if V else np.zeros((0, 3)), F, np.array(fobj), np.array(helper, bool), offs


def write_parts(objs, F, fobj, face_body, order):
    """Face attribute rqmesh_part (body index) and colour attribute rqmesh_parts on every object."""
    col_of = {b: PALETTE[k % len(PALETTE)] for k, b in enumerate(order)}
    for k, ob in enumerate(objs):
        me = ob.data; fb = face_body[fobj == k]
        a = me.attributes.get('rqmesh_part') or me.attributes.new('rqmesh_part', 'INT', 'FACE')
        a.data.foreach_set('value', fb.astype(np.int32))
        ca = me.color_attributes.get('rqmesh_parts') or me.color_attributes.new('rqmesh_parts', 'FLOAT_COLOR', 'CORNER')
        cols = np.zeros((len(me.loops), 4)); cols[:, 3] = 1
        for p in me.polygons:
            c = col_of.get(int(fb[p.index]), (0.2, 0.2, 0.2))
            cols[p.loop_start:p.loop_start + p.loop_total, :3] = c
        ca.data.foreach_set('color', cols.ravel()); me.color_attributes.active_color = ca


def build_rig(ctx, objs, r, F, fobj, offs, name, lock=True):
    diag = r['diag']; order = r['order']; J = r['joint']
    face_body = r['face_body']
    # a bone is named after the object that holds most of its body's faces
    most = {}
    for b in order:
        ks = fobj[face_body == b]
        most[b] = objs[int(np.bincount(ks).argmax())].name if len(ks) else 'part'
    data = bpy.data.armatures.new(name); arm = bpy.data.objects.new(name, data); ctx.collection.objects.link(arm)
    arm.show_in_front = True
    for o in ctx.selected_objects:
        o.select_set(False)
    arm.select_set(True); ctx.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = data.edit_bones; names, along = {}, {}
    for b in order:
        j = J[b]; piv = np.asarray(j['pivot'], float); cen = np.asarray(r['centroid'][b], float)
        d = cen - piv; L = max(0.06 * diag, 0.5 * float(np.linalg.norm(d)))
        bone = eb.new(most[b]); bone.head = Vector(piv)
        kind, ax = j['kind'], None if j['axis'] is None else np.asarray(j['axis'], float)
        if kind == 'root':
            bone.tail = Vector(piv + np.array([0, 0, L]))
        elif kind == 'hinge':
            dp = d - (d @ ax) * ax
            if np.linalg.norm(dp) > max(0.2 * np.linalg.norm(d), 1e-6):     # bone toward the part, X = hinge axis
                y = dp / np.linalg.norm(dp); bone.tail = Vector(piv + y * L); bone.align_roll(Vector(np.cross(ax, y)))
                along[b] = False
            else:                                                            # the part lies on its axis: turn about Y
                bone.tail = Vector(piv + ax * L); along[b] = True
        elif kind in ('slide', 'track'):
            bone.tail = Vector(piv + ax * L)
        else:                                                                # ball, free
            y = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([0, 0, 1.0])
            bone.tail = Vector(piv + y * L)
        names[b] = bone.name
    for b in order:
        if r['parent'].get(b) is not None:
            eb[names[b]].parent = eb[names[r['parent'][b]]]
    targets = {}
    for b in order:
        j = J[b]
        if j['kind'] == 'track':                                             # what each piston half points at
            t = eb.new(names[b] + '_target'); tp = np.asarray(j['target_point'], float)
            t.head = Vector(tp); t.tail = Vector(tp + np.asarray(j['axis'], float) * 0.04 * diag)
            t.parent = eb[names[j['target_parent']]]; t.use_deform = False; targets[b] = t.name
    bpy.ops.object.mode_set(mode='OBJECT')
    for b in order:
        j = J[b]; bone = data.bones[names[b]]; pb = arm.pose.bones[names[b]]
        bone['rqmesh_body'] = int(b); bone['rqmesh_kind'] = j['kind']; bone['rqmesh_evidence'] = j['evidence']
        if j['axis'] is not None:
            bone['rqmesh_axis'] = [float(x) for x in j['axis']]
        if j['kind'] == 'track':
            c = pb.constraints.new('DAMPED_TRACK'); c.target = arm; c.subtarget = targets[b]; c.track_axis = 'TRACK_Y'
        if not lock or j['kind'] == 'root':
            continue
        pb.lock_location = (True, True, True)
        if j['kind'] == 'hinge':
            pb.rotation_mode = 'XYZ'
            pb.lock_rotation = (True, False, True) if along[b] else (False, True, True)
        elif j['kind'] in ('slide', 'track'):
            pb.rotation_mode = 'XYZ'; pb.lock_rotation = (True, True, True)
            if j['kind'] == 'slide':
                pb.lock_location = (True, False, True)
        else:
            pb.rotation_mode = 'QUATERNION'
    # rigid weights: every vertex 100 % to its part's bone
    for k, ob in enumerate(objs):
        if ob.display_type in {'WIRE', 'BOUNDS'}:
            continue
        fb = face_body[fobj == k]; vb = {}
        for p in ob.data.polygons:
            if fb[p.index] >= 0:
                for v in p.vertices:
                    vb.setdefault(v, int(fb[p.index]))
        by = collections.defaultdict(list)
        for v, b in vb.items():
            by[b].append(v)
        for b, vs in by.items():
            g = ob.vertex_groups.get(names[b]) or ob.vertex_groups.new(name=names[b]); g.add(vs, 1.0, 'REPLACE')
        for m in [m for m in ob.modifiers if m.type == 'ARMATURE' and m.name == 'RQMesh Rig']:
            ob.modifiers.remove(m)
        md = ob.modifiers.new('RQMesh Rig', 'ARMATURE'); md.object = arm
        mw = ob.matrix_world.copy(); ob.parent = arm; ob.matrix_world = mw
    return arm, names


class RQM_OT_rig_hard_surface(bpy.types.Operator):
    bl_idname = "mesh.rqmesh_rig_hard_surface"
    bl_label = "Rig Hard Surface"
    bl_description = ("Bones, joints and 100 % weights for a model of rigid parts (machines, robots, doors), found from "
                      "its pins, balls, sleeves and rails. Select all the model's mesh objects first. Wire objects mark "
                      "hidden joints")
    bl_options = {'REGISTER', 'UNDO'}

    preview: bpy.props.BoolProperty(
        name="Only show the parts", default=False,
        description="Colour the parts it finds (colour attribute 'rqmesh_parts') and report the joints, without a rig")
    lock: bpy.props.BoolProperty(
        name="Lock to the joints", default=True,
        description="Each bone moves only as its joint can: a hinge turns about one axis, a slide moves along one, "
                    "a ball turns freely in place")
    crease_angle: bpy.props.FloatProperty(
        name="Split at creases", default=30.0, min=5.0, max=90.0,
        description="Welded parts are split at concave creases sharper than this (degrees)")
    contact: bpy.props.FloatProperty(
        name="Contact gap", default=1.0, min=0.1, max=5.0, subtype='PERCENTAGE',
        description="Parts closer than this (percent of the model's size) touch")
    detail: bpy.props.FloatProperty(
        name="Detail size", default=0.5, min=0.0, max=1.0,
        description="A piece touching a part without a joint between them, smaller than this times the part, is "
                    "fixed to it (panels, plates, greebles). 0 keeps every piece separate")

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'OBJECT' and any(o.type == 'MESH' for o in ctx.selected_objects)

    def execute(self, ctx):
        from . import SHAPE_MODIFIERS
        objs = [o for o in ctx.selected_objects if o.type == 'MESH']
        if ctx.active_object in objs:                       # the active object first: it names the rig
            objs.remove(ctx.active_object); objs.insert(0, ctx.active_object)
        V, F, fobj, helper, offs = gather(objs)
        if not F:
            self.report({'ERROR'}, "No faces in the selected objects"); return {'CANCELLED'}
        r = RR.analyze(V, F, helper=helper, crease_deg=self.crease_angle, contact=self.contact / 100.0, detail=self.detail)
        write_parts(objs, F, fobj, r['face_body'], r['order'])
        jk = r['joints']; label = dict(hinge=('hinge', 'hinges'), slide=('slide', 'slides'), ball=('ball', 'balls'),
                                       track=('piston half', 'piston halves'), free=('free joint', 'free joints'))
        kinds = ", ".join(f"{jk[k]} {label[k][jk[k] != 1]}" for k in label if jk.get(k))
        msg = f"{len(r['order'])} parts" + (f"; {kinds}" if kinds else "")
        mods = sorted({m.type.title() for o in objs for m in o.modifiers if m.type in SHAPE_MODIFIERS})
        if mods:
            msg += f". Unapplied {', '.join(mods)}: analysed without them"
        if self.preview:
            self.report({'INFO'}, msg + ". Parts shown in the 'rqmesh_parts' colour attribute"); return {'FINISHED'}
        arm, _ = build_rig(ctx, objs, r, F, fobj, offs, objs[0].name + "_rig", self.lock)
        if jk.get('free'):
            msg += ". Free joints have no pin, ball or rail: check them, or add a Wire helper there and run again"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class RQM_PT_rig(bpy.types.Panel):
    bl_label = "Rig Hard Surface"; bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = "RQMesh"
    bl_parent_id = "RQM_PT_panel"; bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        col = self.layout.column(align=True)
        col.operator(RQM_OT_rig_hard_surface.bl_idname, icon='ARMATURE_DATA')
        col.operator(RQM_OT_rig_hard_surface.bl_idname, text="Only Show the Parts", icon='GROUP_VERTEX').preview = True
        box = self.layout.box().column(align=True); box.scale_y = 0.8
        for line in HELP:
            box.label(text=line)
