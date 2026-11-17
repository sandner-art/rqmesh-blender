# SPDX-License-Identifier: GPL-3.0-or-later
"""Copy the rqmesh library (Apache-2.0) into the add-on as rqmesh/core.

usage: python tools/sync_core.py PATH_TO_RQMESH_CHECKOUT
The add-on vendors the library until it bundles the released PyPI wheel (needed for extensions.blender.org).
core/SOURCE records the library version and commit; core/LICENSE carries the Apache-2.0 text the licence requires."""
import os, sys, glob, shutil, subprocess

src_repo = os.path.abspath(sys.argv[1])
pkg = os.path.join(src_repo, 'src', 'rqmesh')
if not os.path.isfile(os.path.join(pkg, '__init__.py')): sys.exit(f"no rqmesh package in {pkg}")
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rqmesh', 'core')
SKIP = {'qw_driver.py', 'scalefield.py'}          # driver ships as backend/qw_driver.py; scalefield needs SciPy
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
for f in sorted(glob.glob(os.path.join(pkg, '*.py'))):
    if os.path.basename(f) not in SKIP: shutil.copy2(f, dst)
shutil.copy2(os.path.join(src_repo, 'LICENSE'), os.path.join(dst, 'LICENSE'))
ver = {}
exec(open(os.path.join(pkg, '__init__.py')).read().split('"""', 2)[-1], ver)
try:
    commit = subprocess.run(['git', '-C', src_repo, 'describe', '--tags', '--always', '--dirty'],
                            capture_output=True, text=True).stdout.strip()
except OSError:
    commit = ''
with open(os.path.join(dst, 'SOURCE'), 'w') as fh:
    fh.write(f"rqmesh {ver['__version__']} {commit or '(not a git checkout)'}\nhttps://github.com/sandner-art/rqmesh\n")
# the add-on's own copy of the back-end driver
shutil.copy2(os.path.join(pkg, 'qw_driver.py'), os.path.join(os.path.dirname(dst), 'backend', 'qw_driver.py'))
drv = os.path.join(os.path.dirname(dst), 'backend', 'qw_driver.py'); s = open(drv).read()
s = s.replace("sys.path.insert(0, HERE)\n", "sys.path.insert(0, HERE)\nsys.path.insert(0, os.path.join(os.path.dirname(HERE), 'core'))   # inside the extension: ../core\n", 1)
open(drv, 'w').write(s)
print('synced', len(os.listdir(dst)), 'files into', dst, '|', open(os.path.join(dst, 'SOURCE')).readline().strip())
