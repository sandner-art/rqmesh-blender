# Changelog

## 0.1.0 — unreleased

First public version (formerly the internal "RQM Lab" add-on, research rounds 1–13). Extension id `rqmesh`; operator
ids `mesh.rqmesh_*`; panel *View3D › Sidebar › RQMesh*.

- Simplify by Edge Rings (presets, transitions, isolation loops around openings and selected regions).
- Fix Poles (refills, guarded isolation rings), Select Poles, Select Isolation Rings.
- Recover Base Cage (Catmull-Clark, simple, Loop, midpoint; creases and UVs), Analyze Structure.
- Quad Caps, Detect Features, Detect Symmetry, Deformation Field.
- Remesh with the optional QuadWild-BiMDF back-end; back-end manager (install from file; download in the self-hosted
  build); result becomes the active object.
- Stored attribute names renamed to `rqmesh_*` (`rqmesh_feature`, `rqmesh_field`, `rqmesh_strain`, …).
- Recover Base Cage: *Surface edited after subdividing* with *Levels*, and *Keep edited parts* for topology edits.
- Rigged meshes: every tool keeps the Armature modifier, the parent and the vertex groups. Recover copies the weights
  exactly; Simplify, Fix Poles, Quad Caps and Remesh (QuadWild and QuadriFlow) interpolate them on the source surface.
- Simplify: *Keep joint loops* and *Joint rotation*, so game LODs of a rigged mesh keep the loops at the joints.
