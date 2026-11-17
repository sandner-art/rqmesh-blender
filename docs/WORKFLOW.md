# Which tool, in which order

RQMesh works *with* the structure your mesh already has. **Start with Analyze Structure.** The *Workflow* box at the top of the RQMesh tab then lists the steps for your mesh in order, each as a button with the right options already set. When you change the mesh, it asks you to analyze again.

The rules below come from measured tests: research round 14, which is the tool-order and edit tests. Each rule says what we measured.

**The tools use the base mesh**, like Blender's own mesh tools: unapplied modifiers (Subdivision, Mirror, Bevel, Boolean, …) are ignored. With a Subdivision modifier on, Remesh works on the low-poly mesh, not the smooth shape you see. Apply the modifiers first to work on what you see; the Workflow box reminds you when there are some. Recover is for subdivision that was *applied*: while the modifier is still there, the base mesh already is the cage.

## What kind of mesh is it?

| Analyze says | What to do | Why |
|---|---|---|
| **Subdivided mesh** | **Recover Base Cage.** It adds a Subdivision modifier; pick the level you need. | You get the artist's own mesh back, exactly. At a quarter of the faces of a subdivided Suzanne, the recovered cage subdivided once had **102 poles**, against **442** from simplifying, at the same error, in 0.7 s instead of 9 s. |
| **Subdivided, then edited?** | If *you* subdivided it and then sculpted or deformed it: **Recover Base Cage** with *Surface edited after subdividing*. If you modelled the mesh as it is, simplify instead. | The cage topology survives edits that keep the connectivity. With *Levels* set to the number of times you subdivided, the cage came back in **131 of 133** edited test meshes and **74 of 76** held-out ones; it failed only under the heaviest sculpts on hard-surface panels. Without *Levels* it was right in 36 of 76, because some hand-made cages look subdivided too (5 of our 23 designer cages did) and Edited mode can peel them as one more level. So it's your choice and never automatic. |
| **Quad-modelled, clean loops** (hard surface) | **Simplify by Edge Rings** (Hard surface). | It removes whole loops and never merges two feature lines. |
| **Quad-modelled, organic** | **Simplify by Edge Rings** (Organic), then **Fix Poles**. | Fix Poles *after* simplifying: **159 poles**, against **433** when run before. |
| **Quads with fans / n-gons** | Simplify (only if you need fewer faces), **then Quad Caps**. | Caps rebuilt after simplifying: **8 poles**, against **24** when rebuilt before. |
| **No usable quad structure** (sculpt, scan, triangle soup) | **Remesh** at the face count you need. | A remesh result has spiralling loops. Simplifying it by edge rings gave 5,540 poles, so don't. |

## Common goals

**Fewer faces on a model you built in quads**

1. Analyze.
2. Simplify by Edge Rings.
3. Fix Poles (organic only).
4. Check with *Select Poles* or *Select Isolation Rings*.

To keep edge loops around eyes and mouth, leave *Isolation loops around: Openings* on (the default for Organic). At very strong reductions, use *Loops: 1*: in our tests, two loops at 25% cost about 30% more error.

**More faces on a low-poly model.** **Subdivide** (Ctrl+2, or a Subdivision modifier). Don't remesh a low-poly model directly. Quadrupling the 500-face Suzanne gave these results:

- remeshed directly: **244 poles, 86 folds**, mean error 0.34%;
- subdivided: 102 poles, 2 folds, 0.09%;
- subdivided twice, then remeshed: 64 poles, 0 folds, 0.09%, if you want fewer poles than the artist's layout.

Errors here are measured against the smooth (subdivided) shape. Remeshing the faceted low-poly shape itself is the remesher's weak spot: on Suzanne QuadWild left 13 folds at 3,000 quads and 78 at 1,000 (where it also overshot to 2,156 faces). Subdividing it flat first didn't help (44–91 folds).

**A sculpt or a scan.** Remesh at the face count you need.

- If the sculpt has no sharp marks, Remesh uses no feature lines on a dense mesh. On a voxel sculpt of Suzanne at about 8,000 quads, automatic features gave 211 poles and 31 folds; no features gave 90 poles and 1 fold at the same error.
- *Fix Poles* afterwards is optional: it took those 90 poles to 78, with no change in error.

**A subdivided model you then edited.** Use *Recover Base Cage* with *Surface edited after subdividing*. The cage follows large edits (bends, twists, sculpted forms). Detail finer than the cage stays only in the dense mesh, so bake it (normal or displacement map) before you swap meshes. How much is lost grows with the edit: on the fox model, the recovered cage subdivided again was on average 0.02% of the model's size away from a 45° bend and 0.2% from a 5% sculpt, with the worst spot up to 2%.

Without *Surface edited after subdividing*, Recover stops at the first level that fits looser than 0.075 edge lengths. On edited meshes it then returned the full cage in 127 of 209 tests and stopped one level early in 81; it added a false level once. The message after Recover says when it stopped early.

**Model on a quad grid, then go back (hard surface).** A hard-surface model full of n-gons and triangles can be turned into all quads and back again:

1. Add a Subdivision modifier set to **Simple**, level 1, and apply it. Every n-gon and triangle becomes quads and the shape doesn't change.
2. Model on the grid with edits that keep the connectivity: move, rotate, scale, proportional editing, bend, twist, sculpt, smooth.
3. **Recover Base Cage** with *Surface edited after subdividing* and **Levels 1**. Your original n-gons come back with the changes. On five models (a cylinder with 32-sided caps, a boolean cut, a bevelled cube, a slotted panel and a gear) that was **34 of 40** edits; without *Levels*, 30 of 40. The misses were the stronger edits (twist, grab and sculpt on the gear, a 45° bend on the bevelled cube, 3 smoothing passes on the cylinder), where even the right reading fits loosely.

Two limits:

- **Concave n-gons fold.** Booleans leave C-shaped n-gons around holes. The simple subdivision puts the n-gon's centre point at its average position, which is outside such a face, and the quads around it fold (all four of our boolean's n-gons: 18 folds). Cut such n-gons into convex pieces before gridding.
- **Changed the topology on the grid** (extrude, inset, loop cut, knife, bevel, deleted faces)? Plain Recover then finds nothing, but **Recover Base Cage with *Keep edited parts*** peels everything that is still a grid and keeps what you changed as it is. Each recovered face that meets kept geometry gets an extra vertex on the shared edge, so there are no holes or T-junctions. On five hard-surface models with one such edit each, a median 96% of the original faces came back (64–91% on a small 11-face panel, where each kept face is a big share); a loop cut costs more, because it cuts through every face along its loop. UVs, materials, vertex groups and the armature are kept. No Subdivision modifier is added, since it would subdivide the kept parts again.

**A rigged model.** Every tool keeps the rig: the Armature modifier, the parent and the vertex groups. Recover Base Cage (also with *Keep edited parts*) copies each vertex's weights exactly from the vertex it came from. Simplify, Fix Poles, Quad Caps and Remesh make new vertices, so they interpolate the weights from the closest point of the original surface, as Blender's *Data Transfer* does with *Nearest Face Interpolated*.

**Game LODs of a model that bends.** Joints need loops. On a skinned arm bent 120° and on a Suzanne with jaw and ear bones, we compared LODs of equal size with the posed high-poly mesh:

- *If you subdivided the model*, **Recover Base Cage** gives the best joints: at the elbow it followed the posed high-poly 5.5 times more closely than Decimate at ¼ of the faces (2.7 times at 1/16), although Decimate is more accurate at rest.
- *Otherwise* use **Simplify** with *Keep joint loops* (on by default; it only acts on a rigged mesh). It keeps the loops where bones blend: the elbow error was 1.7 to 7.4 times lower than without it, and lower than Decimate in 3 of 4 cases (two arms, ¼ and 1/16 of the faces), with all-quad loops. At 1/16 on the arm whose loops were dense at the elbow, Decimate was a little better (0.62% vs 0.71% of the arm's size). Set *Joint rotation* to how far your joints bend.
- Broad, smooth weights (a jaw, ears) don't need extra loops: every LOD's error was the same posed as at rest, and Decimate was the most accurate.

**A game LOD with triangles.** Blender's own **Decimate** is more accurate than any quad method. Use RQMesh when you need quads, edge loops, UV islands or symmetry kept.

## Rigging hard-surface models

**Rig Hard Surface** rigs a model made of rigid parts: machines, robots, lamps, doors, hatches. Select all the model's mesh objects and click it (RQMesh tab › *Rig Hard Surface*). The parts can be separate objects, separate pieces in one object, or welded into one mesh. You get:

- an armature with one bone per part, named after its object, at the joint it found;
- every object parented to it, with an Armature modifier and 100% weights, so rigid parts never bend;
- each bone locked to its joint: a hinge bone turns about one axis only, a slide bone moves along one axis only, a ball bone turns freely in place;
- pistons that follow: the rod and the sleeve each turn about their pin to point at the other's (Damped Track), so the piston extends when the parts move.

**How it finds the joints.** Model them like this, and the rig follows:

| Joint | What it looks for |
|---|---|
| Hinge | A cylinder pin through both parts (at least 2.5 times as long as its radius). Or a round turntable (a disc or a drum) under the part. Or two parts touching along one straight edge, like a door against its jamb. |
| Slide | A rod entering one end of a sleeve (two cylinders on one axis). Or a panel held between two parallel rails. |
| Ball | A sphere touching both parts, smaller than either. |
| Piston | A rod and a sleeve, each pinned to a different part. |

Small pieces that touch a part without any of these become part of it: panels, plates, bolts, greebles. The pins, balls and stubs go with the part nearer the base.

**Check before you rig.** *Only Show the Parts* colours the parts it found (the `rqmesh_parts` colour attribute; set *Viewport Shading › Color › Attribute*) and reports the joints. Two parts in one colour will move together. A *free joint* means it found two large parts touching but no pin, ball or rail: the bone turns freely at their contact.

**Pinning a hidden joint.** If a hinge has no visible pin (it is inside the part, or the parts just meet), add a helper:
1. Add a cylinder through the hinge, along its axis, touching both parts (for a ball joint, a sphere at its centre).
2. Set it to *Object Properties › Viewport Display › Display As › Wire*. Wire objects mark joints and get no weights.
3. Select it with the model and run *Rig Hard Surface*. Hide or delete the helper afterwards.

On our test lamp without pins the joints came out wrong; with three helper cylinders where the pins were, the rig was exact. The same held for the robot arm.

**Measured.** We tested 37 cases: a robot arm, a desk lamp, a press, an excavator with a hydraulic cylinder, an arm on a ball joint, two doors and an airlock, each bevelled, triangulated, low-poly, joined into one object or welded. Each rig was posed with the true joint values.
- 33 matched the true pose within 0.35% of the model's size (most to rounding).
- The misses were models without pins and welded moving machines:
  - the robot without pins got free joints near the right places (mean error 0.7%);
  - the lamp without pins got wrong ones;
  - the welded lamp and the welded excavator lost their pins in the weld.

Keep parts that move as separate objects or pieces, and add helpers where a hinge is hidden.

**Not yet:** joint limits (it sets no rotation ranges), telescopic square tubes, gears, chains, cables and flexible parts. The tool also stores `rqmesh_part` (a face attribute) and `rqmesh_parts` (colours) on your meshes; delete them if you don't need them.

## What not to do

- **Don't simplify a mesh that was subdivided and then edited.** Edge-ring simplification needs a clean surface. After sculpting or adding detail it stopped at 2 to 16 times the cage's face count, because the noise looks like hundreds of small sharp edges. Recover the cage instead.
- **Don't simplify a remesh result by edge rings.**
- **Don't run Fix Poles before simplifying.** Simplification creates its own poles.

## Tools in short

| Tool | Does | Key options |
|---|---|---|
| Analyze Structure | Measures quads, regular vertices, clean loops and subdivision levels, and picks the method | — |
| Recover Base Cage | Reverses Catmull-Clark / simple / Loop / midpoint subdivision, with creases and UVs | *Surface edited after subdividing*, *Levels* (0 = detect), *Result* (cage / fitted / on surface) |
| Simplify by Edge Rings | Removes whole rings of quads, with transitions where needed | *Shape* preset, *Keep ratio* or *Target faces*, *Isolation loops around*, *Symmetry* |
| Fix Poles | Rotations, collapses and pattern refills that reduce poles | *Guard isolation rings*, *Refill pole clusters* |
| Quad Caps | Rebuilds fans and n-gon caps as quad grids | *Cap flatness* |
| Remesh | QuadWild-BiMDF (optional back-end) or Blender's QuadriFlow | *Target quads*, *Features* |
| Select Poles / Select Isolation Rings | Selects what to check | — |
| Rig Hard Surface | Bones, joints and rigid weights for a model of rigid parts, from its pins, balls, sleeves and rails | *Only show the parts*, *Lock to the joints*, *Contact gap*, *Detail size*; Wire objects are helpers |
