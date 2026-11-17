# SPDX-License-Identifier: Apache-2.0
"""rqmesh: structure-aware quad remeshing, simplification and retopology.

Pure Python + NumPy (SciPy optional, used when present).  Modules:

  structsimp, transitions   simplify a quad mesh by removing whole edge rings; 3-1 and gentle strip transitions
  topofix                   fix poles: rotations, collapses, loop moves and pattern refills; guarded isolation rings
  isolation                 openings, isolating rings around eyes / mouths / bumps, protected edge loops
  subdiv, subdiv_tri        recover the base cage of Catmull-Clark / simple / Loop / midpoint subdivided meshes
  features, crease_split    sharp features and crease splitting
  symmetry                  mirror-plane detection and symmetric operations
  analyze                   structure report and recommended workflow for a mesh
  scalefield, defofield     cross fields: scale-matched curvature fields for sculpts, deformation fields
  repair                    welds, small holes, degenerate faces
  qw_driver, strip_bounds   drive the optional QuadWild-BiMDF back-end (GPL, separate package)
  meshio_min, compat, rosy  OBJ / .rosy I/O and NumPy-only stand-ins for SciPy pieces

Most modules also run as command-line tools:  python -m rqmesh.transitions --help
"""
__version__ = "0.1.0"
