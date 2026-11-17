# SPDX-License-Identifier: GPL-3.0-or-later
"""Guided workflow: what to run, in which order, for the kind of mesh Analyze Structure found.

Each plan is a list of steps (operator, button text, one-line reason, preset options).  The orders come from the
round-14 tool-order tests (RESULTS-round14.md in the research kit); docs/WORKFLOW.md explains them."""

GUIDE_URL = "https://github.com/sandner-art/rqmesh-blender/blob/main/docs/WORKFLOW.md"

KIND = {
    'recover_base_first': "Subdivided mesh",
    'recover_edited': "Subdivided, then edited?",
    'quad_caps_first': "Quads with fans / n-gons",
    'edge_loops': "Quad-modelled, clean loops",
    'adaptive_loops': "Quad-modelled, organic",
    'decimate_or_remesh': "No usable quad structure",
}

# (operator idname, button text, reason, preset properties)
PLANS = {
    'recover_base_first': [
        ("mesh.rqmesh_recover_base", "Recover Base Cage", "exact cage + a Subdivision modifier", {}),
        ("mesh.rqmesh_analyze", "Analyze the cage", "then set the modifier level you need", {}),
    ],
    'recover_edited': [
        ("mesh.rqmesh_recover_base", "Recover Base Cage (edited)", "set Levels to how often you subdivided",
         {'edited': True}),
        ("mesh.rqmesh_analyze", "Analyze the cage", "the cage gets its own next steps", {}),
    ],
    'quad_caps_first': [
        ("mesh.rqmesh_structure_simplify", "Simplify by Edge Rings", "only if you need fewer faces", {'preset': 'MIXED'}),
        ("mesh.rqmesh_quad_caps", "Quad Caps", "fans and n-gons become quad grids", {}),
    ],
    'edge_loops': [
        ("mesh.rqmesh_structure_simplify", "Simplify by Edge Rings", "removes whole loops, keeps sharp lines",
         {'preset': 'HARD'}),
        ("mesh.rqmesh_select_poles", "Check: Select Poles", "poles should match the design", {}),
    ],
    'adaptive_loops': [
        ("mesh.rqmesh_structure_simplify", "Simplify by Edge Rings", "keeps loops around eyes and mouth",
         {'preset': 'ORGANIC'}),
        ("mesh.rqmesh_fix_poles", "Fix Poles", "after simplifying, never before", {'keep_isolation': 2, 'refill': True}),
        ("mesh.rqmesh_select_isolation", "Check: Isolation Rings", "eyes and mouth still ringed?", {}),
    ],
    'decimate_or_remesh': [
        ("mesh.rqmesh_remesh_quadwild", "Remesh", "at the face count you need", {}),
    ],
}

# a second way when the analysis cannot be sure: (operator, button text, preset properties)
_KEEP = ("mesh.rqmesh_recover_base", "Subdivided, then extruded or cut? Recover, keep edits", {'partial': True})
ALTERNATIVE = {
    'recover_edited': ("mesh.rqmesh_structure_simplify", "Modelled as is? Simplify instead", {}),
    'edge_loops': _KEEP, 'adaptive_loops': _KEEP, 'quad_caps_first': _KEEP,
}

TIPS = {
    'edge_loops': "More faces instead? Subdivide (Ctrl+2). Remeshing a clean low-poly model gave 244 poles and 86 "
                  "folds in our tests, subdividing 102 poles and 2 folds. Want the smooth shape remeshed? "
                  "Subdivide twice first (64 poles, no folds).",
    'adaptive_loops': "More faces instead? Subdivide (Ctrl+2). Remeshing a clean low-poly model gave 244 poles and 86 "
                      "folds in our tests, subdividing 102 poles and 2 folds. Want the smooth shape remeshed? "
                      "Subdivide twice first (64 poles, no folds).",
    'quad_caps_first': "More faces instead? Subdivide, don't remesh. Caps after simplifying gave 8 poles, before 24.",
    'decimate_or_remesh': "Don't simplify a remesh result by edge rings (its loops spiral). For a triangle game LOD, "
                          "Blender's Decimate is more accurate.",
    'recover_base_first': "Need fewer faces? Use the cage at a lower modifier level: it is the artist's own mesh. On "
                          "Suzanne at a quarter of the faces it had 102 poles, simplifying 442.",
    'recover_edited': "Set Levels: with it 74 of 76 held-out edited cages came back, counting levels alone 36. "
                      "Some hand-made cages look subdivided too; if you modelled this mesh as it is, simplify it.",
}


def wrap(text, width=36):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width and line:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def draw(layout, ob, analysis):
    """analysis: dict stored by Analyze Structure (or None)."""
    box = layout.box()
    from . import SHAPE_MODIFIERS
    mods = [m.name for m in (ob.modifiers if ob is not None else []) if m.show_viewport and m.type in SHAPE_MODIFIERS]
    if mods:
        for k, t in enumerate(wrap(f"The tools use the base mesh, without {', '.join(mods)}. Apply it first to work "
                                   f"on the shape you see.")):
            box.label(text=t, icon='MODIFIER' if k == 0 else 'BLANK1')
    if analysis is None:
        box.label(text="Start here", icon='INFO')
        box.operator("mesh.rqmesh_analyze", text="1. Analyze Structure", icon='VIEWZOOM')
        for t in wrap("Finds what kind of mesh this is and lists the tools to run, in order."):
            box.label(text=t)
        box.operator("wm.url_open", text="Open the workflow guide", icon='HELP').url = GUIDE_URL
        return
    method = analysis.get('method')
    stale = analysis.get('faces') is not None and ob is not None and analysis['faces'] != len(ob.data.polygons)
    head = box.row()
    head.label(text=KIND.get(method, method or "?"), icon='CHECKMARK' if not stale else 'ERROR')
    head.operator("mesh.rqmesh_analyze", text="", icon='FILE_REFRESH')
    if stale:
        for t in wrap("The mesh changed since it was analyzed: analyze again."):
            box.label(text=t)
        return
    col = box.column(align=True)
    for i, (idname, text, why, props) in enumerate(PLANS.get(method, []), 1):
        op = col.operator(idname, text=f"{i}. {text}")
        for k, v in props.items():
            try: setattr(op, k, v)
            except (AttributeError, TypeError): pass
        sub = col.row(); sub.scale_y = 0.7; sub.label(text="    " + why)
    tip = TIPS.get(method)
    if tip:
        box.separator()
        for j, t in enumerate(wrap(tip)):
            box.label(text=t, icon='LIGHT' if j == 0 else 'BLANK1')
    alt = ALTERNATIVE.get(method)
    if alt:
        op = box.operator(alt[0], text=alt[1])
        for k, v in alt[2].items():
            try: setattr(op, k, v)
            except (AttributeError, TypeError): pass
    box.operator("wm.url_open", text="Workflow guide", icon='HELP').url = GUIDE_URL
