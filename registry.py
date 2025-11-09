# -*- coding: utf-8 -*-
"""
registry.py — interlayer registry helpers (robust version)

Adds:
  • compute_registry_metrics_safe(..., return_grid: bool = False)
    -> returns (L_wall, wall_mask) or (L_wall, wall_mask, xg, yg)
"""

import numpy as np, numpy.linalg as npl
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, sobel

# ---------------- basic lattice / registry helpers ----------------
def reciprocal_vectors(a1, a2):
    A = np.column_stack([a1, a2])
    B = 2*np.pi * npl.inv(A.T)
    return B[:, 0], B[:, 1]

def G_star_three(b1, b2):
    return np.stack([b1, b2, -(b1 + b2)], axis=0)

def wrap_to_cell(delta, a1, a2):
    A = np.column_stack([a1, a2])
    uv = np.linalg.solve(A, delta)
    uv -= np.round(uv)
    return A @ uv

def nearest_top_displacements(XY_bottom, XY_top, a1_b, a2_b):
    idx = cKDTree(XY_top).query(XY_bottom, k=1)[1]
    raw = XY_top[idx] - XY_bottom
    wrapped = np.array([wrap_to_cell(d, a1_b, a2_b) for d in raw])
    return wrapped, np.ones(len(XY_bottom), dtype=bool)

def phi_order_parameter(deltas_wrapped, a1_b, a2_b):
    b1, b2 = reciprocal_vectors(a1_b, a2_b)
    Gs = G_star_three(b1, b2)                # shape (3,2)
    dots = deltas_wrapped @ Gs.T             # (Nb,3)
    Phi = np.exp(1j * dots).mean(axis=1)     # (Nb,)
    return Phi, np.abs(Phi), np.angle(Phi)

# ---------------- rasterization ----------------
def rasterize_scalar(XY, values, dx):
    x, y = XY[:, 0], XY[:, 1]
    xmin, xmax = x.min() - dx, x.max() + dx
    ymin, ymax = y.min() - dx, y.max() + dx
    Nx = max(64, int(np.ceil((xmax - xmin) / dx)))
    Ny = max(64, int(np.ceil((ymax - ymin) / dx)))
    xedges = np.linspace(xmin, xmin + Nx*dx, Nx + 1)
    yedges = np.linspace(ymin, ymin + Ny*dx, Ny + 1)
    Hs, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=values)
    Hc, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    img = np.divide(Hs, Hc, out=np.zeros_like(Hs), where=(Hc > 0))
    xg = 0.5 * (xedges[:-1] + xedges[1:])
    yg = 0.5 * (yedges[:-1] + yedges[1:])
    return img, xg, yg

# ---------------- wall detection & length ----------------
def wall_mask_from_phase(phase_img, sigma_px=1.0, q=0.90):
    """
    Take the phase image, smooth, compute |∇phase|, and threshold its top-q quantile
    to obtain a thin domain-wall mask.
    """
    ph_s = gaussian_filter(phase_img, sigma=sigma_px)
    gx = sobel(ph_s, axis=1, mode='nearest')
    gy = sobel(ph_s, axis=0, mode='nearest')
    grad = np.hypot(gx, gy)
    finite = np.isfinite(grad)
    thr = np.quantile(grad[finite], q) if finite.any() else 0.0
    return grad >= thr, grad, thr

def _contour_length_from_mask(mask, xg, yg):
    """
    Prefer skimage marching squares for accurate length; fallback to a cheap estimate.
    """
    try:
        from skimage import measure
        arr = np.asarray(mask, dtype=float)
        cs = measure.find_contours(arr, 0.5)
        if not cs:
            return 0.0
        # map pixel coordinates (row, col) -> (x, y)
        def interp(idx, coords):
            N = coords.size
            i = np.clip(idx, 0.0, N - 1.0)
            i0 = np.floor(i).astype(int)
            i1 = np.clip(i0 + 1, 0, N - 1)
            w = i - i0
            return (1.0 - w) * coords[i0] + w * coords[i1]
        total = 0.0
        for c in cs:
            rows, cols = c[:, 0], c[:, 1]
            xs = interp(cols, np.asarray(xg))
            ys = interp(rows, np.asarray(yg))
            d = np.hypot(np.diff(xs), np.diff(ys)).sum()
            total += float(d)
        return total
    except Exception:
        # fallback: count boundary pixels and multiply by mean grid spacing
        mask = np.asarray(mask, bool)
        # an edge pixel has at least one 4-neighbor with opposite value
        up    = np.roll(mask, -1, axis=0)
        down  = np.roll(mask,  1, axis=0)
        left  = np.roll(mask,  1, axis=1)
        right = np.roll(mask, -1, axis=1)
        boundary = mask & (~(up & down & left & right))
        # average physical pixel size
        dx = float(np.mean(np.diff(xg))) if len(xg) > 1 else 1.0
        dy = float(np.mean(np.diff(yg))) if len(yg) > 1 else 1.0
        # use perimeter ≈ count * (dx+dy)/2  (crude but stable)
        return float(boundary.sum()) * 0.5 * (dx + dy)

def wall_length_from_mask(mask, xg, yg):
    if mask is None or np.size(mask) == 0:
        return 0.0
    return _contour_length_from_mask(mask, xg, yg)

# ---------------- public API ----------------
def compute_registry_metrics_safe(
    XY_all, N1, psi, a1_b, a2_b, dx_reg=0.5, tau=0.40, return_grid=False
):
    """
    Build a domain-wall mask from the interlayer registry phase and measure wall length.

    Returns:
      (L_wall, wall_mask)                          if return_grid == False
      (L_wall, wall_mask, xg, yg)                  if return_grid == True
    """
    try:
        XY_bot = XY_all[:N1]
        XY_top = XY_all[N1:]

        deltas_wrapped, _ok = nearest_top_displacements(XY_bot, XY_top, a1_b, a2_b)
        Phi, Mag, Phase = phi_order_parameter(deltas_wrapped, a1_b, a2_b)

        # bottom-layer weight (psi used only if you later need it; kept for API parity)
        _Pb = np.abs(psi[:N1])**2

        mag_img, xg, yg = rasterize_scalar(XY_bot, Mag,   dx_reg)
        phs_img, _,  _  = rasterize_scalar(XY_bot, Phase, dx_reg)

        wall_mask, _grad, _thr = wall_mask_from_phase(phs_img, sigma_px=1.0, q=0.90)
        L_wall = wall_length_from_mask(wall_mask, xg, yg)

        if return_grid:
            return float(L_wall), wall_mask, xg, yg
        return float(L_wall), wall_mask
    except Exception:
        if return_grid:
            return np.nan, None, None, None
        return np.nan, None

def wall_overlap_all_states(V, N1, XY_all, dx_reg, wall_mask, layer="bottom"):
    """
    Compute wall-overlap for each eigenstate via coarse rasterization.
    overlap = sum_{pixels on wall} ρ(pixel) / sum_{pixels} ρ(pixel)
    """
    P2 = np.abs(V)**2
    if layer == "bottom":
        XY = XY_all[:N1]; P2L = P2[:N1, :]
    elif layer == "top":
        XY = XY_all[N1:]; P2L = P2[N1:, :]
    else:  # total
        XY = XY_all;     P2L = P2

    x, y = XY[:, 0], XY[:, 1]
    xmin, xmax = x.min() - dx_reg, x.max() + dx_reg
    ymin, ymax = y.min() - dx_reg, y.max() + dx_reg
    Nx = max(64, int(np.ceil((xmax - xmin) / dx_reg)))
    Ny = max(64, int(np.ceil((ymax - ymin) / dx_reg)))
    xedges = np.linspace(xmin, xmax, Nx + 1)
    yedges = np.linspace(ymin, ymax, Ny + 1)

    Hcnt, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    mask_valid = (Hcnt > 0)

    overlaps = np.zeros(P2L.shape[1], dtype=float)
    for s in range(P2L.shape[1]):
        Hsum, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=P2L[:, s])
        num = np.nansum(Hsum[wall_mask & mask_valid])
        den = np.nansum(Hsum[mask_valid]) + 1e-18
        overlaps[s] = num / den
    return overlaps
