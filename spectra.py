# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:54:38 2025

@author: HOL1BRG
"""

import numpy as np, scipy.sparse as ss, scipy.sparse.linalg as sla

def dedup_eigs(E, V, atol=1e-8):
    order = np.argsort(E); E, V = E[order], V[:,order]
    keep = [0]
    for i in range(1, len(E)):
        if abs(E[i]-E[keep[-1]]) > atol: keep.append(i)
    return E[keep], V[:,keep]

def eigs_slice(H, sigma, k, ncv=None, tol=1e-10, maxiter=None):
    N = H.shape[0]; k_eff = min(k, max(1, N-2))

    try:
        if N > 3*k_eff:
            w,v = sla.eigsh(H, k=k_eff, sigma=sigma, which='LM',
                            ncv=(ncv or min(N-1, 2*k_eff+1)),
                            tol=tol, maxiter=maxiter, return_eigenvectors=True)
        else:
            A = H.toarray() if ss.issparse(H) else H
            w,v = np.linalg.eigh(A); sel = np.argsort(np.abs(w-sigma))[:k_eff]
            w,v = w[sel], v[:,sel]
    except Exception:
        A = H.toarray() if ss.issparse(H) else H
        w,v = np.linalg.eigh(A); sel = np.argsort(np.abs(w-sigma))[:k_eff]
        w,v = w[sel], v[:,sel]
    return w,v

def eigs_in_window_sliced(H, Emin, Emax, sigmas, k_per, n_target):
    Es, Vs = [], []
    for s in sigmas:
        w,v = eigs_slice(H, sigma=s, k=k_per)
        mask = (w > Emin) & (w < Emax)
        if mask.any(): Es.append(w[mask]); Vs.append(v[:,mask])
    if not Es: return np.array([]), np.empty((H.shape[0],0))
    E = np.concatenate(Es); V = np.concatenate(Vs, axis=1)
    E,V = dedup_eigs(E,V,atol=1e-8)
    if len(E) > n_target:
        center = 0.5*(Emin+Emax); sel = np.argsort(np.abs(E-center))[:n_target]
        E,V = E[sel], V[:,sel]
    return E,V

def ipr(V): return np.sum(np.abs(V)**4, axis=0)

def edge_mask(XY, poly_corners, d_edge):
    # distance to boundary segments
    import numpy as np
    P0,P1,P3,P2 = poly_corners
    segs = [(P0,P1),(P1,P3),(P3,P2),(P2,P0)]
    dmin = np.full(len(XY), np.inf)
    for (a,b) in segs:
        ab = b-a; denom = np.dot(ab,ab)
        ap = XY - a; t = np.clip((ap @ ab)/ (denom if denom!=0 else 1.0), 0.0, 1.0)
        proj = a + t[:,None]*ab
        d = np.linalg.norm(XY - proj, axis=1)
        dmin = np.minimum(dmin, d)
    return dmin <= d_edge

def edge_weight(V, XY, poly_corners, N1, d_edge):
    mask_b = edge_mask(XY[:N1], poly_corners, d_edge)
    mask_t = edge_mask(XY[N1:], poly_corners, d_edge)
    mask   = np.r_[mask_b, mask_t]
    return np.sum(np.abs(V[mask,:])**2, axis=0)
