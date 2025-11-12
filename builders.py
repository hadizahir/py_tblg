# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:53:55 2025

@author: HOL1BRG
"""

import numpy as np, scipy.sparse as ss
from scipy.spatial import cKDTree
from .geometry import inside_rhombus

def tperp_SK(r_xy, acc, dperp, t, tp, E_in_t, QSIGMA, QPI):
    tp_eff = tp/t if E_in_t else tp
    t_eff  = 1.0 if E_in_t else t
    rp = float(np.sqrt(r_xy**2 + dperp**2))
    sin2w = (r_xy**2)/(dperp**2 + r_xy**2 + 1e-30)
    cos2w = (dperp**2)/(dperp**2 + r_xy**2 + 1e-30)
    VPPsigma = tp_eff*np.exp(QSIGMA*(1.0 - rp/dperp))
    VPPpi    = -t_eff*np.exp(QPI*(1.0 - rp/acc))
    return float(VPPsigma*cos2w + VPPpi*sin2w)

def build_flake_H_sparse(n_mult, a1_b, a2_b, A_b, B_b, a1_t, a2_t, A_t, B_t,
                         T1, T2, acc, dperp, t, tp, r_xy_cut, t_intra, QSIGMA, QPI, E_in_t):
    origin = np.array([0.0,0.0])

    # index bounds via solving for (i,j) that cover rhombus roughly
    def bounds(a1,a2):
        M = np.linalg.solve(np.column_stack([a1,a2]), np.column_stack([T1,T2]))
        corners = np.array([[0,0], n_mult*M[:,0], n_mult*M[:,1], n_mult*(M[:,0]+M[:,1])])
        i_min,j_min = np.floor(corners.min(0)-4).astype(int)
        i_max,j_max = np.ceil (corners.max(0)+4).astype(int)
        return i_min,i_max,j_min,j_max

    i_min_b,i_max_b,j_min_b,j_max_b = bounds(a1_b,a2_b)
    i_min_t,i_max_t,j_min_t,j_max_t = bounds(a1_t,a2_t)

    # coordinates
    bot, top = [], []
    for i in range(i_min_b, i_max_b+1):
        for j in range(j_min_b, j_max_b+1):
            for basis in (A_b, B_b):
                r = basis + i*a1_b + j*a2_b
                if inside_rhombus(r, origin, T1, T2, n_mult): bot.append(r)
    for i in range(i_min_t, i_max_t+1):
        for j in range(j_min_t, j_max_t+1):
            for basis in (A_t, B_t):
                r = basis + i*a1_t + j*a2_t
                if inside_rhombus(r, origin, T1, T2, n_mult): top.append(r)
    bot = np.array(bot); top = np.array(top)
    N1, N2 = len(bot), len(top)
    XY = np.vstack([bot, top])

    rows, cols, vals = [], [], []

    # intralayer NN (distance ~ acc)
    for XYl, off in ((bot,0),(top,N1)):
        tree = cKDTree(XYl)
        for i,j in tree.query_pairs(1.01*acc):
            rows += [off+i, off+j]; cols += [off+j, off+i]; vals += [t_intra, t_intra]

    # interlayer SK
    # Interlayer SK hoppings
    ttree = cKDTree(top)
    for i, rb in enumerate(bot):
        hits = ttree.query_ball_point(rb, r_xy_cut)
        for j in hits:
            dx, dy = (top[j] - rb)
            amp = tperp_SK(
                np.hypot(dx, dy),
                acc=acc, dperp=dperp, t=t, tp=tp, E_in_t=E_in_t,
                QSIGMA=QSIGMA, QPI=QPI
            )
            if abs(amp) > 1e-12:
                rows += [i, N1 + j]
                cols += [N1 + j, i]
                vals += [amp, amp]


    H = ss.coo_matrix((vals, (rows, cols)), shape=(N1+N2, N1+N2)).tocsr()

    return H, XY, N1
