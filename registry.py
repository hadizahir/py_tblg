# -*- coding: utf-8 -*-
"""
registry.py — registry/order-parameter utilities for TBLG overlays.

Exports
-------
reciprocal_vectors, wrap_to_cell, nearest_top_displacements,
phi_order_parameter, rasterize_scalar,
wall_mask_from_phase, wall_length_from_mask,
get_registry_grid,
region_masks_from_phi,
psi_region_fractions,
contour_paths  (extract closed paths from boolean masks)
"""

import numpy as np, numpy.linalg as npl
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt

# ---------------- basic geometry / order parameter ----------------
def reciprocal_vectors(a1, a2):
    A = np.column_stack([a1, a2])
    B = 2*np.pi * npl.inv(A.T)
    return B[:, 0], B[:, 1]

def wrap_to_cell(delta, a1, a2):
    A = np.column_stack([a1, a2])
    uv = np.linalg.solve(A, delta)
    uv -= np.round(uv)
    return A @ uv

def nearest_top_displacements(XY_bottom, XY_top, a1_b, a2_b):
    idx = cKDTree(XY_top).query(XY_bottom, k=1)[1]
    raw = XY_top[idx] - XY_bottom
    wrapped = np.array([wrap_to_cell(d, a1_b, a2_b) for d in raw])
    return wrapped

def phi_order_parameter(deltas_wrapped, a1_b, a2_b):
    """Complex registry order parameter Φ; AA≈|Φ| max; walls≈large ∇(arg Φ)."""
    b1, b2 = reciprocal_vectors(a1_b, a2_b)
    Gs = np.stack([b1, b2, -(b1 + b2)], axis=0)  # 3-star
    dots = deltas_wrapped @ Gs.T
    Phi = np.exp(1j * dots).mean(axis=1)
    return Phi, np.abs(Phi), np.angle(Phi)

# ---------------- rasterization + wall mask ----------------
def rasterize_scalar(XY, values, dx):
    x, y = XY[:, 0], XY[:, 1]
    xmin, xmax = x.min() - dx, x.max() + dx
    ymin, ymax = y.min() - dx, y.max() + dx
    Nx = max(64, int(np.ceil((xmax - xmin) / dx)))
    Ny = max(64, int(np.ceil((ymax - ymin) / dx)))
    xedges = np.linspace(xmin, xmax, Nx + 1)
    yedges = np.linspace(ymin, ymax, Ny + 1)

    Hs, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=values)
    Hc, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    img = np.divide(Hs, Hc, out=np.zeros_like(Hs), where=(Hc > 0))
    xg = 0.5 * (xedges[:-1] + xedges[1:])
    yg = 0.5 * (yedges[:-1] + yedges[1:])
    return img, xg, yg, Hc > 0  # also return a validity mask

def wall_mask_from_phase(phase_img, sigma_px=1.2, q=0.85):
    """Walls from large gradient of phase (no filled dots, just mask)."""
    ph_s = gaussian_filter(phase_img, sigma=sigma_px, mode="nearest")
    gx = sobel(ph_s, axis=1, mode="nearest")
    gy = sobel(ph_s, axis=0, mode="nearest")
    grad = np.hypot(gx, gy)
    finite = np.isfinite(grad)
    thr = np.quantile(grad[finite], q) if finite.any() else 0.0
    wall_mask = (grad >= thr)
    return wall_mask, thr

def wall_length_from_mask(wall_mask, xg, yg):
    """Approximate wall length by counting mask edges in physical units."""
    if wall_mask is None or wall_mask.size == 0:
        return 0.0
    dy = float(np.abs(yg[1] - yg[0])) if len(yg) > 1 else 0.0
    dx = float(np.abs(xg[1] - xg[0])) if len(xg) > 1 else 0.0
    # 8-connected perimeter count
    from scipy.ndimage import binary_dilation
    rim = binary_dilation(wall_mask) ^ wall_mask
    # perimeter pixels ~ number of true contacts
    perim_px = np.count_nonzero(rim & wall_mask)
    # use average pixel size
    ds = 0.5 * (dx + dy)
    return perim_px * ds

# ---------------- one-shot registry grid ----------------
def get_registry_grid(XY_all, N1, a1_b, a2_b, dx_reg=1.0):
    """Return (mag_img, phase_img, wall_mask, xg, yg, valid)."""
    XYb = XY_all[:N1]
    XYt = XY_all[N1:]

    deltas = nearest_top_displacements(XYb, XYt, a1_b, a2_b)
    Phi, Mag, Phase = phi_order_parameter(deltas, a1_b, a2_b)

    mag_img, xg, yg, valid = rasterize_scalar(XYb, Mag, dx_reg)
    phs_img, _,  _, _      = rasterize_scalar(XYb, Phase, dx_reg)

    wall_mask, _ = wall_mask_from_phase(phs_img, sigma_px=1.2, q=0.85)
    return mag_img, phs_img, wall_mask, xg, yg, valid

# ---------------- AA / AB / BA masks from Φ ----------------
def region_masks_from_phi(mag_img, phs_img, wall_mask, valid,
                          aa_frac=0.80, m_sign_thr=0.20):
    """
    Build boolean masks on the same grid:
      AA:  |Φ| >= aa_frac * |Φ|_max
      Walls: from wall_mask
      BA:  cos(arg Φ) >= +m_sign_thr (excluding AA,Walls)
      AB:  cos(arg Φ) <= -m_sign_thr (excluding AA,Walls)
    """
    mag_s = gaussian_filter(mag_img, 1.0, mode="nearest")
    m = np.cos(phs_img)

    aa = (mag_s >= (aa_frac * np.nanmax(mag_s)))
    walls = (wall_mask & valid)
    core = valid & (~aa) & (~walls)

    ba = (m >= +m_sign_thr) & core
    ab = (m <= -m_sign_thr) & core

    # Anything leftover is ignored (no label)
    return aa, ab, ba, walls

# ---------------- fraction of |psi|^2 in regions ------------
def psi_region_fractions(P2_state, XY_all, N1, dx_reg, masks, layer="total"):
    """
    Rasterize |psi|^2 on the same grid and return fractions in AA/AB/BA/Walls.
    masks = (aa, ab, ba, walls, xg, yg, valid)
    """
    aa, ab, ba, walls, xg, yg, valid = masks
    if layer == "bottom":
        XY = XY_all[:N1]; P2 = P2_state[:N1]
    elif layer == "top":
        XY = XY_all[N1:]; P2 = P2_state[N1:]
    else:
        XY = XY_all;      P2 = P2_state

    # same binning as registry grid
    x, y = XY[:, 0], XY[:, 1]
    xmin, xmax = xg[0], xg[-1]
    ymin, ymax = yg[0], yg[-1]
    Nx, Ny = len(xg), len(yg)
    xedges = np.linspace(xmin, xmax, Nx + 1)
    yedges = np.linspace(ymin, ymax, Ny + 1)

    Zsum, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=P2)
    W = valid.astype(float)
    Z = np.divide(Zsum, W, out=np.zeros_like(Zsum), where=(W > 0))

    def _sum(mask):
        return float(np.nansum(Z[mask & valid]))

    tot = float(np.nansum(Z[valid])) + 1e-18
    f_aa   = _sum(aa)   / tot
    f_ab   = _sum(ab)   / tot
    f_ba   = _sum(ba)   / tot
    f_wall = _sum(walls)/ tot
    return dict(AA=f_aa, AB=f_ab, BA=f_ba, WALL=f_wall)

# ---------------- contour extraction (no external deps) ----
def contour_paths(mask, xg, yg):
    """
    Return list of closed path arrays (N_i, 2) for a boolean mask using Matplotlib.
    """
    if mask is None or mask.size == 0:
        return []
    fig = plt.figure(); ax = fig.add_subplot(111)
    CS = ax.contour(xg, yg, mask.astype(float), levels=[0.5])
    paths = []
    if hasattr(CS, "collections") and CS.collections:
        for col in CS.collections:
            for p in col.get_paths():
                v = p.vertices
                if v.shape[0] >= 3:
                    paths.append(v.copy())
    plt.close(fig)
    return paths
