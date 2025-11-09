# -*- coding: utf-8 -*-
"""
wavefunctions.py — clean overlays for |ψ|² with AA/AB/BA/Walls as thin CLOSED lines.

Exports
-------
save_wavefunctions_npz
save_wavefunction_overlay_registry_png_clean
save_wavefunction_3d_surface_html_clean   (3D surface + line contours)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

from .registry import (
    get_registry_grid, region_masks_from_phi, psi_region_fractions, contour_paths
)

# ---------------- save eigenstates ----------------
def save_wavefunctions_npz(tag, n_mult, XY_all, N1, E, V, save_dir, sel_idx=None, max_states=12):
    os.makedirs(save_dir, exist_ok=True)
    M = V.shape[1]
    if M == 0:
        return None
    if sel_idx is None:
        center = 0.5 * (E.min() + E.max())
        sel_idx = np.argsort(np.abs(E - center))[:min(max_states, M)]
    sel_idx = np.asarray(sel_idx, dtype=int)
    out = {"XY": np.asarray(XY_all, float),
           "N1": int(N1),
           "E":  np.asarray(E[sel_idx], float),
           "P":  np.abs(V[:, sel_idx])**2}
    fpath = os.path.join(save_dir, f"wfmaps_{tag}_n{n_mult:02d}.npz")
    np.savez_compressed(fpath, **out)
    print(f"[saved wfmaps] {fpath}  (states saved: {len(sel_idx)})")
    return fpath

# ---------------- utilities ----------------
def _rasterize_weighted(XY, weights, xg, yg):
    x, y = XY[:, 0], XY[:, 1]
    Nx, Ny = len(xg), len(yg)
    xedges = np.linspace(xg[0], xg[-1], Nx + 1)
    yedges = np.linspace(yg[0], yg[-1], Ny + 1)
    Zsum, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=weights)
    C, _, _    = np.histogram2d(y, x, bins=[yedges, xedges])
    Z = np.divide(Zsum, C, out=np.full_like(Zsum, np.nan), where=(C > 0))
    return Z

# ---------------- 2D overlay: CLEAN ----------------
def save_wavefunction_overlay_registry_png_clean(
    XY_all, N1, Elist, P2_all, a1_b, a2_b, state=0,
    dx_reg=1.0, aa_frac=0.80, wall_q=0.85, m_sign_thr=0.20,
    out_png=None, layer="total"
):
    """
    Plot |ψ|² as a heatmap (binned) and overlay CLOSED contour lines for
    AA (red), AB (blue), BA (green), Walls (cyan). No filled dots.
    Also prints & returns region fractions.
    """
    # 1) registry grid and masks
    mag_img, phs_img, wall_mask, xg, yg, valid = get_registry_grid(XY_all, N1, a1_b, a2_b, dx_reg=dx_reg)
    aa, ab, ba, walls = region_masks_from_phi(mag_img, phs_img, wall_mask, valid,
                                              aa_frac=aa_frac, m_sign_thr=m_sign_thr)
    masks = (aa, ab, ba, walls, xg, yg, valid)

    # 2) choose layer + rasterize |ψ|² on same grid
    P2 = P2_all[:, state]
    if layer == "bottom":
        XY, P2s = XY_all[:N1], P2[:N1]
    elif layer == "top":
        XY, P2s = XY_all[N1:], P2[N1:]
    else:
        XY, P2s = XY_all, P2

    Z = _rasterize_weighted(XY, P2s, xg, yg)
    Zs = gaussian_filter(np.nan_to_num(Z, nan=0.0), sigma=1.0, mode="nearest")
    Zs[~valid] = np.nan

    # 3) plot
    fig, ax = plt.subplots(figsize=(7.3, 7))
    vmin = max(np.nanmin(Zs[np.isfinite(Zs)]) * 0.8, 1e-10)
    vmax = np.nanmax(Zs) if np.isfinite(np.nanmax(Zs)) else 1.0
    im = ax.pcolormesh(xg, yg, Zs, shading="auto",
                       norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin*10)),
                       cmap="viridis")

    # closed lines only
    for mask, color, label in [(aa, '#e41a1c', 'AA'),
                               (ab, '#377eb8', 'AB'),
                               (ba, '#4daf4a', 'BA'),
                               (walls, '#00cfd0', 'Walls (m≈0)')]:
        paths = contour_paths(mask, xg, yg)
        for k, pts in enumerate(paths):
            show = (k == 0)
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.3, label=(label if show else None))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    ax.set_title(f"|ψ|² map  (E/t = {Elist[state]:.6f})")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("|ψ|²")

    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()

    if out_png is None:
        out_png = "wf_overlay_registry.png"
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved overlay PNG] {out_png}")

    # 4) fractions
    fracs = psi_region_fractions(P2, XY_all, N1, dx_reg, masks, layer="total")
    print("[region fractions]", ", ".join(f"{k}={v:.3f}" for k, v in fracs.items()))
    return out_png, fracs, (aa, ab, ba, walls, xg, yg, valid)

# ---------------- 3D surface + line contours ----------------
def save_wavefunction_3d_surface_html_clean(
    XY_all, N1, Elist, P2_all, masks_tuple, state=0, out_html=None
):
    aa, ab, ba, walls, xg, yg, valid = masks_tuple
    P2 = P2_all[:, state]

    # bin to grid used by registry
    Z = _rasterize_weighted(XY_all, P2, xg, yg)
    Z[~valid] = np.nan
    X, Y = np.meshgrid(xg, yg)

    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Viridis",
                               colorbar=dict(title="|ψ|²"), showscale=True))
    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, b=0, t=40),
                      title=f"|ψ|² surface (state #{state}, E/t={Elist[state]:.6f})")

    def add_contours(mask, color, name):
        paths = contour_paths(mask, xg, yg)
        for k, pts in enumerate(paths):
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), np.nanmax(Z)*1.02),
                mode='lines', line=dict(color=color, width=4), name=(name if k == 0 else None),
                showlegend=(k == 0)
            ))

    add_contours(aa,    '#e41a1c', 'AA')
    add_contours(ab,    '#377eb8', 'AB')
    add_contours(ba,    '#4daf4a', 'BA')
    add_contours(walls, '#00cfd0', 'Walls (m≈0)')

    if out_html is None:
        out_html = "wf_surface_registry.html"
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"[saved 3D surface] {out_html}")
    return out_html
