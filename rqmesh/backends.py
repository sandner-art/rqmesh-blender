# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional native back-ends (QuadWild-BiMDF) for RQMesh.

Policy (extensions.blender.org terms: self-contained, no remote code; network only with the manifest permission and
Blender's "Allow Online Access"):
  * The structure-first and clean-up tools never need a back-end.
  * A back-end is found in this order: the folder set in the preferences, the extension's user directory (where
    "Install Back-end from File" puts it), a folder bundled inside a platform-specific build, the RQM_QWLIB variable.
  * "Install Back-end from File" takes a back-end package (.zip with rqm_backend.json): the platform must match, every
    file's SHA-256 must match the package manifest, and a self-test must pass before it is used.
  * Downloading is compiled out of the extensions.blender.org flavour (ALLOW_DOWNLOAD = False).  The self-hosted
    flavour may download a package only from the index URL the user sets, only when Online Access is on, only after
    the user confirms, and verifies the SHA-256 published in that index before unpacking.
  * Without a back-end, Remesh falls back to Blender's built-in QuadriFlow.
"""
import os, sys, json, hashlib, zipfile, shutil, subprocess, tempfile, platform, time

ALLOW_DOWNLOAD = False                     # set True only in the self-hosted build (see bench/build_flavours.py)
PACKAGE_SCHEMA = 1
LIBS = {'windows': ('lib_quadwild.dll', 'lib_quadpatches.dll'),
        'macos': ('liblib_quadwild.dylib', 'liblib_quadpatches.dylib'),
        'linux': ('liblib_quadwild.so', 'liblib_quadpatches.so')}


def platform_tag():
    """Blender's platform naming: linux-x64, macos-arm64, windows-x64 ..."""
    s = sys.platform
    os_ = 'linux' if s.startswith('linux') else 'macos' if s == 'darwin' else 'windows' if s.startswith('win') else s
    m = platform.machine().lower()
    return f"{os_}-{'arm64' if m in ('arm64', 'aarch64') else 'x64'}"


def _os():
    return platform_tag().split('-')[0]


def valid_qwlib_parent(folder):
    """True if folder/qwlib holds the wrapper and both native libraries for this OS."""
    if not folder: return False
    q = os.path.join(folder, 'qwlib')
    return (os.path.isfile(os.path.join(q, '__init__.py')) and
            all(os.path.isfile(os.path.join(q, n)) for n in LIBS.get(_os(), ())))


def user_backend_dir(package, create=False):
    try:
        import bpy
        return bpy.utils.extension_path_user(package, path=os.path.join('backends', 'quadwild_bimdf'), create=create)
    except Exception:                       # outside Blender (tests)
        d = os.path.join(tempfile.gettempdir(), 'rqm_user', 'backends', 'quadwild_bimdf')
        if create: os.makedirs(d, exist_ok=True)
        return d


def find_quadwild(pref_dir, package, here):
    """(folder, source) of a usable QuadWild-BiMDF back-end, or (None, reason)."""
    cands = [(pref_dir, 'preferences'), (user_backend_dir(package), 'installed package'),
             (os.path.join(here, 'backend', 'bin', platform_tag()), 'bundled with this build'),
             (os.environ.get('RQM_QWLIB', ''), 'RQM_QWLIB variable')]
    for d, src in cands:
        if d and valid_qwlib_parent(d): return d, src
    return None, f"no QuadWild-BiMDF back-end for {platform_tag()}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def read_package_manifest(zpath):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        mf = [n for n in names if n.endswith('rqm_backend.json')]
        if len(mf) != 1: raise ValueError("not an rqmesh back-end package (rqm_backend.json missing)")
        root = mf[0][:-len('rqm_backend.json')]
        return json.loads(z.read(mf[0]).decode('utf-8')), root, names


def install_from_zip(zpath, package, expect_sha=None):
    """Verify and unpack a back-end package into the extension user directory.  Returns (folder, manifest)."""
    if expect_sha and sha256(zpath) != expect_sha: raise ValueError("package checksum does not match the index")
    man, root, names = read_package_manifest(zpath)
    if man.get('schema') != PACKAGE_SCHEMA: raise ValueError("unsupported package schema")
    if man.get('platform') != platform_tag():
        raise ValueError(f"package is for {man.get('platform')}, this computer is {platform_tag()}")
    files = man.get('files', {})
    for rel in files:                                   # refuse anything that escapes the target folder
        if rel.startswith('/') or '..' in rel.replace('\\', '/').split('/'): raise ValueError(f"unsafe path {rel}")
    stage = tempfile.mkdtemp(prefix='rqm_be_')
    try:
        with zipfile.ZipFile(zpath) as z:
            for rel, digest in files.items():
                src = root + rel
                if src not in names: raise ValueError(f"package is missing {rel}")
                dst = os.path.join(stage, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(src) as fi, open(dst, 'wb') as fo: shutil.copyfileobj(fi, fo)
                if sha256(dst) != digest: raise ValueError(f"checksum mismatch for {rel}")
        if not valid_qwlib_parent(stage): raise ValueError("package does not contain a usable qwlib for this OS")
        target = user_backend_dir(package, create=True)
        if os.path.isdir(target): shutil.rmtree(target)
        shutil.copytree(stage, target)
        with open(os.path.join(target, 'rqm_backend.json'), 'w') as fh: json.dump(man, fh, indent=1)
        return target, man
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def self_test(folder, driver, python, timeout=120):
    """Remesh a small sphere in a separate process.  Returns (ok, message)."""
    tmp = tempfile.mkdtemp(prefix='rqm_st_'); inp, out = os.path.join(tmp, 'in.obj'), os.path.join(tmp, 'out.obj')
    n = 8; key = {}; V = []; F = []                     # welded 8x8-per-side box, triangulated

    def vid(p):
        k = tuple(round(c, 6) for c in p)
        if k not in key: key[k] = len(V); V.append(k)
        return key[k]
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            u, v = [(ax + 1) % 3, (ax + 2) % 3][::(1 if sgn > 0 else -1)]
            for i in range(n):
                for j in range(n):
                    q = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = [0.0, 0.0, 0.0]; p[ax] = sgn; p[u] = -1 + 2 * (i + di) / n; p[v] = -1 + 2 * (j + dj) / n
                        q.append(vid(p))
                    F += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    with open(inp, 'w') as fh:
        for p in V: fh.write("v %.6f %.6f %.6f\n" % p)
        for f in F: fh.write("f %d %d %d\n" % tuple(x + 1 for x in f))
    t0 = time.time()
    try:
        r = subprocess.run([python, '-I', driver, folder, inp, out, '--target', '200', '--features', 'none'],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "self-test timed out"
    ok = r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 100
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, (f"self-test passed in {time.time() - t0:.1f} s" if ok else "self-test failed: " + (r.stderr or r.stdout)[-200:])


def fetch_index(url, timeout=20):
    """Read the back-end index (JSON) -- only called when ALLOW_DOWNLOAD and Online Access are both on."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def download(url, timeout=120):
    import urllib.request
    fd, path = tempfile.mkstemp(suffix='.zip', prefix='rqm_dl_'); os.close(fd)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(path, 'wb') as fh:
        shutil.copyfileobj(resp, fh)
    return path


def make_package(qwlib_parent, out_zip, version, plat=None, license_txt=None, source_txt=None):
    """Build a back-end package from a folder containing qwlib/ (used by the release script)."""
    plat = plat or platform_tag(); files = {}
    for dp, dn, fn in os.walk(os.path.join(qwlib_parent, 'qwlib')):
        dn[:] = [d for d in dn if d != '__pycache__']
        for f in fn:
            if f.endswith('.pyc'): continue
            p = os.path.join(dp, f); files[os.path.relpath(p, qwlib_parent).replace(os.sep, '/')] = sha256(p)
    man = dict(schema=PACKAGE_SCHEMA, name='QuadWild-BiMDF', version=version, platform=plat, license='GPL-3.0-or-later',
               source='https://github.com/sandner-art/rqmesh-backend-quadwild', files=files)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in files: z.write(os.path.join(qwlib_parent, rel), 'quadwild_bimdf/' + rel)
        z.writestr('quadwild_bimdf/rqm_backend.json', json.dumps(man, indent=1))
        if license_txt: z.writestr('quadwild_bimdf/LICENSE.txt', license_txt)
        if source_txt: z.writestr('quadwild_bimdf/SOURCE.txt', source_txt)
    return out_zip, man
