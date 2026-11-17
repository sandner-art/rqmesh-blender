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
