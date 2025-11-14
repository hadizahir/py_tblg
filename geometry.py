# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:52:44 2025

@author: HOL1BRG
"""

import numpy as np, numpy.linalg as npl

def moire_vectors_primitive(a1,a2,m,r):
    if r % 3 != 0:
        U = np.array([[m, -r-m], [r+m, 2*m+r]], float)
    else:
        U = np.array([[m + r/3, r/3], [-r/3, m + 2*r/3]], float)
    A = np.column_stack([a1,a2])
    T1, T2 = (A@U)[:,0], (A@U)[:,1]
    return T1, T2, U

def expected_sites_comm_tBLG(m,r):
    N = m*m + m*r + r*r
    total = 4*N//3 if r % 3 == 0 else 4*N
    return int(total), int(total//2), int(N)

def rhombus_polygon(origin, T1, T2, n_mult):
    O = origin; return np.vstack([O, O+ n_mult*T1, O+n_mult*(T1+T2), O+n_mult*T2, O])

def barycentric_uv(r, origin, T1, T2):
    return npl.solve(np.column_stack([T1, T2]), r - origin)

def inside_rhombus(r, origin, T1, T2, n_mult):
    u,v = barycentric_uv(r, origin, T1, T2)
    return (0.0 <= u < n_mult) and (0.0 <= v < n_mult)

def rhombus_corners(origin, T1, T2, n_mult):
    """
    Return the four corners of the rhombus defined by origin, T1, T2, and n_mult.
    """
    P0 = origin
    P1 = origin + n_mult * T1
    P2 = origin + n_mult * T2
    P3 = origin + n_mult * (T1 + T2)
    return P0, P1, P3, P2


def edge_region_mask(XY, origin, T1, T2, n_mult, d_edge):
    """
    Identify atoms that lie within 'd_edge' distance from any of the four
    rhombus edges (used to crop geometric edge states).
    Returns a boolean mask (True = edge atom).
    """
    P0, P1, P3, P2 = rhombus_corners(origin, T1, T2, n_mult)
    segments = [(P0, P1), (P1, P3), (P3, P2), (P2, P0)]

    dmin = np.full(len(XY), np.inf)
    for (a, b) in segments:
        ab = b - a
        denom = np.dot(ab, ab)
        ap = XY - a
        # project point onto segment
        t = np.einsum('ij,j->i', ap, ab) / (denom if denom != 0.0 else 1.0)
        t = np.clip(t, 0.0, 1.0)
        proj = a + np.outer(t, ab)
        d = np.linalg.norm(XY - proj, axis=1)
        dmin = np.minimum(dmin, d)
    return dmin <= d_edge



def theta_comm_deg_from_mr(m: int, r: int) -> float:
    """
    Commensurate twist angle for (m,r):
      cos θ = (3m^2 + 3mr + r^2/2) / (3m^2 + 3mr + r^2)
    Returns θ in degrees.
    """
    import numpy as np
    num = 3*m*m + 3*m*r + 0.5*r*r
    den = 3*m*m + 3*m*r + r*r
    c = float(num / den)
    c = max(-1.0, min(1.0, c))
    print(float(np.degrees(np.arccos(c))))
    return float(np.degrees(np.arccos(c)))

