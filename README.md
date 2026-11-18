# RQMesh for Blender

Structure-aware quad simplification, pole fixing and retopology for Blender 4.2 and later. It's built on the
[rqmesh](https://github.com/sandner-art/rqmesh) library.

RQMesh works *with* the edge flow your mesh already has:

- **Simplify by Edge Rings.** Removes whole rings of quads, with clean 3-1 or gentle transitions. The result stays all quads, and the loops around eyes, mouths and other openings are kept (*Isolation loops around*). On a rigged mesh it keeps the loops where the bones blend (*Keep joint loops*), so game LODs still bend well.
- **Fix Poles.** Improves irregular vertices without breaking the rings that isolate eyes and mouths.
- **Recover Base Cage.** Turns a mesh that was subdivided and applied (Catmull-Clark, simple, Loop or midpoint) back into its exact original cage, with creases and UVs. It still works after you sculpt or deform the subdivided mesh: enable *Surface edited after subdividing* and set *Levels*. After extrudes, insets, loop cuts or other topology changes, *Keep edited parts* recovers everything that is still a grid and keeps your changes as they are.
- **Analyze Structure.** Tells you what kind of mesh it is. The *Workflow* box then lists the steps for your mesh in order, as buttons with the options set.
- **Remesh (QuadWild-BiMDF).** Full quad remeshing for sculpts and scans, with the optional back-end.
- **Also:** Quad Caps, Detect Features, Detect Symmetry, Select Poles, Select Isolation Rings, Deformation Field.
- **UVs from Structure.** UV maps from the recovered subdivision cage (merged seams along its loops, trim-sheet placement, straight-loop layouts, per-panel islands) or a shape-aware unwrap of a hard-surface model's parts. It also stores panel ids and coordinates, so materials can vary per panel or part.
- **Surface Masks.** Per-vertex wear, cavity and smooth colour-variation masks that follow the model's structure, for materials in EEVEE and Cycles, with an example worn-paint material.
- **Rig Hard Surface.** Rigs a model of rigid parts (machines, robots, doors) from its structure: hinges from pins, slides from rods in sleeves or panels in rails, ball joints from spheres, and pistons that follow their parts. One bone per part, locked to its joint, with 100% weights. A Wire cylinder marks a hidden hinge.
- **Rigged models.** Every tool keeps the Armature modifier, the parent and the vertex groups: Recover copies each weight exactly, the others interpolate the weights from the original surface.

Everything is in the 3D Viewport sidebar (N), on the **RQMesh** tab. Which tool to use, in which order, and why: [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Install

Each release ships two builds:

| Build | For | Downloads the back-end on request |
|---|---|---|
| `rqmesh-X.Y.Z-selfhosted.zip` | Install from this repository (recommended) | Yes, after you confirm, with Blender's Online Access on |
| `rqmesh-X.Y.Z-official.zip` | extensions.blender.org rules | No: install the back-end from a file |

**As a remote repository (gets updates):**

1. Go to *Edit › Preferences › Get Extensions*.
2. Open the *Repositories* dropdown (top right) and click **+ › Add Remote Repository**.
3. Enter this URL:
   ```
   https://github.com/sandner-art/rqmesh-blender/releases/latest/download/index.json
   ```
4. Search for **RQMesh** and click *Install*.

**From a file:** download a zip from [Releases](https://github.com/sandner-art/rqmesh-blender/releases), then use
*Get Extensions ›* dropdown *› Install from Disk*.

**The QuadWild-BiMDF back-end** (only needed for *Remesh*) is a separate GPL package for each platform, built in
[rqmesh-backend-quadwild](https://github.com/sandner-art/rqmesh-backend-quadwild). In the add-on preferences, either
click *Download Back-end* (self-hosted build), or download the package for your platform yourself and click
*Install Back-end from File*. Each install runs a short self-test.

## Licences

- **The add-on:** GPL-3.0-or-later ([LICENSE](LICENSE)), as Blender's extension platform requires.
- **`rqmesh/core/`:** a copy of the rqmesh library, Apache-2.0, with its licence in `rqmesh/core/LICENSE` and its version in `rqmesh/core/SOURCE`. Apache-2.0 code may be combined with GPL-3 code.
- **The back-end:** GPL-3.0, distributed separately, with full source in every back-end release.

## Development

```
python tools/sync_core.py ../rqmesh                # refresh rqmesh/core from a checkout of the library
python tools/build_flavours.py rqmesh dist         # build dist/rqmesh-X.Y.Z-{official,selfhosted}.zip
pip install "bpy==4.5.*"                           # Blender as a Python module (Python 3.11), or bpy 5.x on 3.13
python tests/smoke_bpy.py dist/rqmesh-0.1.0-selfhosted.zip
```

**Smoke test.** `tests/smoke_bpy.py` installs the built zip, enables it the way Blender does, and runs every operator on a subdivided monkey. If `RQM_QWLIB` points to a back-end folder, it also runs the QuadWild remesh.

**CI.** It runs the smoke test with Blender 4.5 LTS (Python 3.11) and Blender 5.3 (Python 3.13).

**Releases.** Bump `version` in `rqmesh/blender_manifest.toml`, then push the tag `vX.Y.Z`. The release workflow:

1. builds both zips and runs the smoke test;
2. validates the zips with Blender 4.5 LTS (`blender --command extension validate`);
3. generates `index.json` from the self-hosted zip;
4. publishes everything as a GitHub release.

Issues are welcome. A `.blend` file and the *Analyze Structure* report help most.
