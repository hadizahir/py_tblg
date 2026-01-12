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
    """
    Correct construction of top-layer honeycomb after rotation.
    The basis (A,B) must be rotated *together with* the Bravais frame.

    bottom:
        r = i a1_b + j a2_b + δ_b

    top:
        r' = R (i a1_b + j a2_b + δ_b + τ)

    where τ is AB/AA registration shift in *bottom-layer basis*.
    """

    import kwant
    from numpy import array
    from .lattices import rot2d

    # --------------------------
    # bottom layer: unchanged
    # --------------------------
    L1 = kwant.lattice.general([a1_b, a2_b], [A_b, B_b],
                               name='L1', norbs=1)
    A1, B1 = L1.sublattices

    # --------------------------
    # compute registration shift
    # in bottom-layer basis
    # --------------------------
    if registration.upper() == "AB":
        tau = (B_b - A_b)
    else:  # "AA"
        tau = np.zeros(2)

    # --------------------------
    # rotation matrix
    # --------------------------
    R = rot2d(theta_deg)

    # --------------------------
    # rotate Bravais lattice
    # --------------------------
    a1_t = R @ a1_b
    a2_t = R @ a2_b

    # --------------------------
    # rotate basis positions
    # δ_t = R(δ_b + τ)
    # --------------------------
    A_t = R @ (A_b + tau)
    B_t = R @ (B_b + tau)

    # --------------------------
    # define Kwant top layer
    # --------------------------
    L2 = kwant.lattice.general([a1_t, a2_t], [A_t, B_t],
                               name='L2', norbs=1)

    return L1, L2, (a1_t, a2_t, A_t, B_t)
