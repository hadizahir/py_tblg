# -*- coding: utf-8 -*-
"""
wavefunctions.py — overlay |ψ|² with clean, closed contour lines for AA / AB / BA
and domain walls derived from a smooth registry “mass” field. No shapely or skimage.

Exports:
  • save_wavefunctions_npz(...)
  • save_wavefunction_overlay_png_clean(npz_path, state, a1_b, a2_b, ...)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import cKDTree


# ------------------- core save -------------------
def save_wavefunctions_npz(tag, n_mult, XY_all, N1, E, V, save_dir, sel_idx=None, max_states=12):
    os.makedirs(save_dir, exist_ok=True)
    M = V.shape[1]
    if M == 0:
        return None
    if sel_idx is None:
        center = 0.5 * (E.min() + E.max())
        sel_idx = np.argsort(np.abs(E - center))[:min(max_states, M)]
    sel_idx = np.asarray(sel_idx, dtype=int)
    out = {
        "XY": np.asarray(XY_all, dtype=np.float64),
        "N1": np.int64(N1),
        "E":  np.asarray(E[sel_idx], dtype=np.float64),
        "P":  np.abs(V[:, sel_idx])**2
    }
    fpath = os.path.join(save_dir, f"wfmaps_{tag}_n{n_mult:02d}.npz")
    np.savez_compressed(fpath, **out)
    print(f"[saved wfmaps] {fpath}  (states saved: {len(sel_idx)})")
    return fpath


# ------------------- registry field helpers -------------------
def _reciprocal_vectors(a1, a2):
    A = np.column_stack([a1, a2])
    B = 2 * np.pi * np.linalg.inv(A.T)
    return B[:, 0], B[:, 1]

def _G_star_three(b1, b2):
    return np.stack([b1, b2, -(b1 + b2)], axis=0)

def _wrap_to_cell(delta, a1, a2):
    A = np.column_stack([a1, a2])
    uv = np.linalg.solve(A, delta)
    uv -= np.round(uv)
    return A @ uv

def _nearest_top_displacements(XY_bottom, XY_top, a1_b, a2_b):
    idx = cKDTree(XY_top).query(XY_bottom, k=1)[1]
    raw = XY_top[idx] - XY_bottom
    wrapped = np.array([_wrap_to_cell(d, a1_b, a2_b) for d in raw])
    return wrapped

def _mass_phase_proxy(XY_bottom, XY_top, a1_b, a2_b):
    """Return smooth mass-like field m_b at bottom-layer sites from local registry."""
    deltas = _nearest_top_displacements(XY_bottom, XY_top, a1_b, a2_b)
    b1, b2 = _reciprocal_vectors(a1_b, a2_b)
    Gs = _G_star_three(b1, b2)
    dots = deltas @ Gs.T           # (N,3)
    m_site = np.cos(dots).sum(axis=1)  # (N,)
    return m_site

def _rasterize_scalar_points(XY, values, dx):
    x, y = XY[:, 0], XY[:, 1]
    xmin, xmax = x.min() - dx, x.max() + dx
    ymin, ymax = y.min() - dx, y.max() + dx
    Nx = max(64, int(np.ceil((xmax - xmin) / dx)))
    Ny = max(64, int(np.ceil((ymax - ymin) / dx)))
    xedges = np.linspace(xmin, xmax, Nx + 1)
    yedges = np.linspace(ymin, ymax, Ny + 1)
    Hsum, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=values)
    Hcnt, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    xg = 0.5 * (xedges[:-1] + xedges[1:])
    yg = 0.5 * (yedges[:-1] + yedges[1:])
    img = np.divide(Hsum, Hcnt, out=np.zeros_like(Hsum), where=(Hcnt > 0))
    return img, xg, yg


# ------------------- overlay plot -------------------
def save_wavefunction_overlay_png_clean(npz_path, state, a1_b, a2_b,
                                        dx_mass=1.2, smooth_sigma=1.2,
                                        dot_size=6, cmap="viridis",
                                        out_png=None):
    """
    Plot |ψ|² and overlay *closed contour lines* for AA / AB / BA regions
    and domain walls derived from a smooth registry “mass” field.
    No point clouds or grids are drawn for overlays.
    """
    data = np.load(npz_path)
    XY  = data["XY"]
    N1  = int(data["N1"])
    E   = data["E"]
    P   = data["P"][:, state]

    XYb, XYt = XY[:N1], XY[N1:]
    Pb,  Pt  = P[:N1],  P[N1:]

    # ---- ψ² heatmap first (below everything) ----
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    vmin = max(P.min() + 1e-18, 1e-10)
    vmax = P.max()
    sc1 = ax.scatter(XYb[:, 0], XYb[:, 1], c=Pb, s=dot_size, cmap=cmap,
                     norm=LogNorm(vmin=vmin, vmax=vmax), edgecolors="none", zorder=1)
    ax.scatter(XYt[:, 0], XYt[:, 1], c=Pt, s=dot_size, cmap=cmap,
               norm=sc1.norm, edgecolors="none", alpha=0.85, zorder=1.05)

    cbar = fig.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label(r"$|\psi|^2$")
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(rf"$|\psi|^2$ map  (E/t = {E[state]:.6f})")

    # ---- build mass proxy on a regular grid from bottom-layer registry ----
    m_site = _mass_phase_proxy(XYb, XYt, a1_b, a2_b)
    m_img, xg, yg = _rasterize_scalar_points(XYb, m_site, dx_mass)

    # smooth & gradient
    if smooth_sigma and smooth_sigma > 0:
        m_img = gaussian_filter(m_img, smooth_sigma / max(dx_mass, 1e-6), mode="nearest")
    grad = np.hypot(sobel(m_img, axis=0), sobel(m_img, axis=1))

    mmax = np.nanmax(np.abs(m_img)) + 1e-12

    # choose levels → AA = high +, AB/BA = high -, walls = grad high
    levels_AA = [0.80 * mmax]
    levels_AB = [-0.60 * mmax]
    levels_BA = [-0.60 * mmax]   # same isocontour; we just style it differently
    level_wall = [np.nanpercentile(grad, 85.0)]

    # ---- overlay only lines (no filled regions, no points) ----
    ax.contour(xg, yg, m_img, levels=levels_AA, colors='#e41a1c',
               linewidths=1.6, zorder=3, alpha=0.95)  # AA
    ax.contour(xg, yg, m_img, levels=levels_AB, colors='#377eb8',
               linewidths=1.2, zorder=3, alpha=0.95)  # AB
    ax.contour(xg, yg, -m_img, levels=[0.60 * mmax], colors='#4daf4a',
               linewidths=1.2, zorder=3, alpha=0.95)  # BA (mirror)
    ax.contour(xg, yg, grad,  levels=level_wall, colors='#00bcd4',
               linewidths=1.0, zorder=3, alpha=0.9)   # Walls

    # Legend with proxy lines (no scatter entries)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='#e41a1c', lw=1.6, label='AA'),
        Line2D([0], [0], color='#377eb8', lw=1.2, label='AB'),
        Line2D([0], [0], color='#4daf4a', lw=1.2, label='BA'),
        Line2D([0], [0], color='#00bcd4', lw=1.0, label='Walls (m≈0)'),
    ]
    ax.legend(handles=legend_lines, frameon=False, loc='upper left', fontsize=9)

    plt.tight_layout()
    if out_png is None:
        base = npz_path.rsplit(".", 1)[0]
        out_png = f"{base}_state{state:02d}_registry_overlay.png"
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved registry overlay] {out_png}")
    return out_png
