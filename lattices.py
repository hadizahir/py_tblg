# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:52:01 2025

@author: HOL1BRG
"""

import numpy as np, numpy.linalg as npl
import kwant

def rot2d(angle_deg: float) -> np.ndarray:
    th = np.deg2rad(angle_deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],[s, c]], float)

def graphene_primitives(acc: float):
    a0 = np.sqrt(3)*acc
    a1_b = a0*np.array([1.0, 0.0]); a2_b = a0*np.array([0.5, np.sqrt(3)/2])
    A_b  = np.array([0.0, 0.0]);    B_b  = np.array([0.0, acc])
    return a1_b, a2_b, A_b, B_b

def layer_lattices(a1_b, a2_b, A_b, B_b, theta_deg: float, registration: str):
    # bottom
    L1 = kwant.lattice.general([a1_b, a2_b], [A_b, B_b], name='L1', norbs=1)
    # top (rotated + registration)
    Delta = (B_b - A_b) if registration.upper()=="AB" else np.zeros(2)
    A_t0, B_t0 = np.array([0.0, A_b[1]])+Delta, np.array([0.0, 0.0])+Delta
    R = rot2d(theta_deg)
    a1_t, a2_t = R@a1_b, R@a2_b
    A_t,  B_t  = R@A_t0, R@B_t0
    L2 = kwant.lattice.general([a1_t, a2_t], [A_t, B_t], name='L2', norbs=1)
    return L1, L2, (a1_t, a2_t, A_t, B_t)
