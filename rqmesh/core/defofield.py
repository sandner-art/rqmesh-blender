# SPDX-License-Identifier: Apache-2.0
"""Deformation-aware cross fields (prototype for idea 3).

Given a rest mesh and one or more posed copies (same connectivity: armature poses, shape keys, simulation
frames), compute per-face principal strain directions and blend them into a smooth 4-RoSy field:

    min_u  sum_edges |u_f - r_fg u_g|^2  +  sum_f  w_f |u_f - g_f|^2       (u = power-4 complex field)

g_f is the strain direction (where strain is large) or a prior field (elsewhere), w_f grows with strain.
"""
import numpy as np
try:
    from .compat import solve_hermitian
except ImportError:
    from compat import solve_hermitian


def face_frames(M):
    e1 = M.V[M.T[:, 1]] - M.V[M.T[:, 0]]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(M.N, e1)
    return e1, e2


def strain_tensors(M, posed_list):
    """Sum over poses of the 2x2 Green strain E = (F^T F - I)/2 in each face's rest frame."""
    e1, e2 = face_frames(M)
    X0 = M.V[M.T[:, 0]]
    R1, R2 = M.V[M.T[:, 1]] - X0, M.V[M.T[:, 2]] - X0
    Dm = np.stack([np.stack([np.einsum('ij,ij->i', R1, e1), np.einsum('ij,ij->i', R2, e1)], 1),
                   np.stack([np.einsum('ij,ij->i', R1, e2), np.einsum('ij,ij->i', R2, e2)], 1)], 1)  # (F,2,2)
    Dinv = np.linalg.inv(Dm)
    E = np.zeros((len(M.T), 2, 2))
    for P in posed_list:
        d1, d2 = P[M.T[:, 1]] - P[M.T[:, 0]], P[M.T[:, 2]] - P[M.T[:, 0]]
        Ds = np.stack([d1, d2], 2)                          # (F,3,2)
        Fg = Ds @ Dinv                                      # (F,3,2)
        C = np.transpose(Fg, (0, 2, 1)) @ Fg
        Ei = 0.5 * (C - np.eye(2)[None])
        E += Ei @ Ei                                        # squared strain: sign-free (stretch or compression), PSD
    return E


def principal(E):
    w, v = np.linalg.eigh(E)                                # ascending
    return v[:, :, 1], np.sqrt(np.maximum(w[:, 1], 0)), np.sqrt(np.maximum(w[:, 0], 0))


def vec_to_power4(M, D):
    e1, e2 = face_frames(M)
    ang = np.arctan2(np.einsum('ij,ij->i', D, e2), np.einsum('ij,ij->i', D, e1))
    return np.exp(4j * ang)


def power4_to_vec(M, u):
    e1, e2 = face_frames(M)
    ang = np.angle(u) / 4.0
    return np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2


def smooth_field(M, guide, weight, lam=1.0, face_stiff=None):
    """Solve the Hermitian system for the power-4 field with soft per-face guidance."""
    e1, e2 = face_frames(M)
    m = M.interior
    f, g = M.EF[m, 0], M.EF[m, 1]
    d = M.edir[m]
    af = np.arctan2(np.einsum('ij,ij->i', d, e2[f]), np.einsum('ij,ij->i', d, e1[f]))
    ag = np.arctan2(np.einsum('ij,ij->i', d, e2[g]), np.einsum('ij,ij->i', d, e1[g]))
    r = np.exp(4j * (af - ag))                              # transport g -> f
    n = len(M.T)
    rows = np.concatenate([f, g, f, g]); cols = np.concatenate([f, g, g, f])
    # optional per-edge stiffness: singularities cost more where the field is stiff, so a stiffness that grows
    # with strain pushes poles out of high-strain regions (cone repulsion)
    we = np.ones(len(f)) if face_stiff is None else 1.0 + np.maximum(face_stiff[f], face_stiff[g])
    vals = np.concatenate([we, we, -we * r, -we * np.conj(r)]).astype(complex)
    wd = lam * weight.astype(complex)
    u = solve_hermitian(rows, cols, vals, wd, wd * guide, n)
    return u / np.maximum(np.abs(u), 1e-12)


def deformation_aware_field(M, posed_list, prior_dirs, s_ref=None, prior_w=0.05, lam=4.0, repel=0.0):
    E = strain_tensors(M, posed_list)
    dmax, smax, smin = principal(E)
    aniso = smax - smin                                     # isotropic strain carries no direction
    Dvec = dmax[:, 0:1] * face_frames(M)[0] + dmax[:, 1:2] * face_frames(M)[1]
    s_ref = s_ref or np.percentile(aniso, 90) + 1e-12
    w_def = np.clip(aniso / s_ref, 0, 1)
    g = w_def * vec_to_power4(M, Dvec) + (1 - w_def) * vec_to_power4(M, prior_dirs)
    w = w_def + prior_w
    u = smooth_field(M, g / np.maximum(np.abs(g), 1e-12), w, lam, face_stiff=repel * w_def if repel > 0 else None)
    return power4_to_vec(M, u), w_def
