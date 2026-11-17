# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the RQMesh extension zips.
  official   : extensions.blender.org rules -- pure Python, no downloads (ALLOW_DOWNLOAD = False), no network permission
  selfhosted : for a self-hosted extension repository -- may download a verified back-end package (network permission),
               and per-platform zips may bundle the back-end under backend/bin/<platform>/qwlib
usage: python tools/build_flavours.py rqmesh dist [PLATFORM=QWLIB_PARENT ...]"""
import sys, os, shutil, zipfile, re, tempfile

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
bundles = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
ver = re.search(r'^version = "([^"]+)"', open(os.path.join(src, 'blender_manifest.toml')).read(), re.M).group(1)


def stage(flavour, plat=None, qw=None):
    d = tempfile.mkdtemp(prefix='rqmesh_build_'); dst = os.path.join(d, 'rqmesh')
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.*'))
    bk = os.path.join(dst, 'backends.py'); s = open(bk).read()
    s = s.replace('ALLOW_DOWNLOAD = False', f'ALLOW_DOWNLOAD = {flavour == "selfhosted"}'); open(bk, 'w').write(s)
    mf = os.path.join(dst, 'blender_manifest.toml'); m = open(mf).read()
    if flavour == 'selfhosted':
        m = m.replace('[permissions]\n', '[permissions]\nnetwork = "Optional: download the back-end package you confirm"\n')
    if plat:
        m = m.replace('[permissions]', f'platforms = ["{plat}"]\n\n[permissions]')
        shutil.copytree(os.path.join(qw, 'qwlib'), os.path.join(dst, 'backend', 'bin', plat, 'qwlib'),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    open(mf, 'w').write(m)
    name = f"rqmesh-{ver}-{flavour}" + (f"-{plat.replace('-', '_')}" if plat else "") + ".zip"
    zp = os.path.join(out, name)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, dn, fn in os.walk(dst):
            for f in fn:
                p = os.path.join(dp, f); z.write(p, os.path.relpath(p, dst))
    shutil.rmtree(d); print(zp, os.path.getsize(zp) // 1024, 'KB')


stage('official'); stage('selfhosted')
for plat, qw in bundles.items(): stage('selfhosted', plat, qw)
