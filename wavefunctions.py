# -*- coding: utf-8 -*-
"""
wavefunctions.py — |ψ|² heatmaps with CLEAN overlays:
thin hex-cell outlines (AA/AB/BA) + AA circles (clipped to flake).
No scatter points for walls/regions, so the heatmap remains visible.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point   # needs shapely

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

# ------------------- geometry helpers -------------------
def _hex_outline(center, T1, T2, scale=1.0):
    """
    Regular hex whose edges are parallel to ±T1, ±T2, ±(T1−T2).
    scale=1.0 means circumscribed radius ≈ |T1|/√3 (moire hex).
    """
    c = np.asarray(center, float)
    t1 = np.asarray(T1, float); t2 = np.asarray(T2, float)
    e1 = t1 / np.linalg.norm(t1); e2 = t2 / np.linalg.norm(t2)
    e3 = (t1 - t2); e3 /= np.linalg.norm(e3)
    # edge directions
    dirs = np.stack([ e1, e2, e3, -e1, -e2, -e3 ], axis=0)
    # radius ~ side length; scale tunes size
    R = scale * (np.linalg.norm(T1) / np.sqrt(3.0))
    return np.array([c + R * d for d in dirs])

def _enumerate_hex_centers(origin, T1, T2, n_mult):
    """
    Centers on the moiré Bravais lattice inside a generous bounding box
    that covers the flake rhombus (n_mult times larger cell).
    """
    o = np.asarray(origin, float)
    centers = []
    # search range: a bit larger than n_mult in each lattice direction
    R = int(2*n_mult + 6)
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            centers.append(o + i*T1 + j*T2)
    return np.vstack(centers)

def _clip_points_to_polygon(points, corners4, margin=0.0):
    poly = Polygon(corners4)
    if margin != 0.0:
        poly = poly.buffer(-margin)
    inside = [poly.contains(Point(p)) for p in points]
    if not inside:
        return np.empty((0,2))
    return points[np.array(inside, bool)]

# ------------------- 2D overlay: CLEAN -------------------
def save_wavefunction_overlay_png_clean(npz_path, state,
                                        T1, T2, origin, n_mult,
                                        clip_polygon,
                                        aa_radius_frac=0.22,
                                        line_alpha=0.6, line_width=1.2,
                                        dot_size=6, cmap="viridis", out_png=None):
    """
    Draw |ψ|² atom heatmap first (bottom+top, shared log scale),
    then overlay ONLY:
      • thin hex-cell outlines (AA/AB/BA colors)
      • AA circles (clipped to flake)
    No scatter points for walls/regions.
    """
    data = np.load(npz_path)
    XY = data["XY"]; N1 = int(data["N1"])
    E  = data["E"];  P2 = data["P"][:, state]
    XYb, XYt = XY[:N1], XY[N1:]
    Pb,  Pt  = P2[:N1], P2[N1:]

    # heatmap first
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    vmin = max(P2.min() + 1e-18, 1e-10); vmax = P2.max()
    sc1 = ax.scatter(XYb[:,0], XYb[:,1], c=Pb, s=dot_size, cmap=cmap,
                     norm=LogNorm(vmin=vmin, vmax=vmax), edgecolors="none", label="Bottom")
    ax.scatter(XYt[:,0], XYt[:,1], c=Pt, s=dot_size, cmap=cmap,
               norm=sc1.norm, edgecolors="none", alpha=0.85, label="Top")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    ax.set_title(f"|ψ|² map  (E/t = {E[state]:.6f})")
    cbar = fig.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label("|ψ|²")

    # build moiré hex grid & AA circles; clip to flake polygon
    centers = _enumerate_hex_centers(origin, T1, T2, n_mult)
    centers = _clip_points_to_polygon(centers, clip_polygon, margin=0.0)

    # Hex outlines: color-code by AA/AB/BA pattern on the moiré honeycomb
    # We assign types by (i+j) mod 3 in lattice index space. To recover (i,j),
    # project centers back to the T1/T2 basis.
    A = np.column_stack([T1, T2])
    uv = np.linalg.lstsq(A, (centers - origin).T, rcond=None)[0].T   # (Nc, 2)
    i = np.round(uv[:,0]).astype(int); j = np.round(uv[:,1]).astype(int)
    typ = (i + j) % 3  # 0→AA, 1→AB, 2→BA

    colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}  # AA, AB, BA
    labels = {0: 'AA',       1: 'AB',       2: 'BA'}

    # draw thin hexes
    for c, t in zip(centers, typ):
        H = _hex_outline(c, T1, T2, scale=1.0)  # scale tunes cell size
        ax.plot(np.r_[H[:,0], H[0,0]], np.r_[H[:,1], H[0,1]],
                color=colors[t], lw=line_width, alpha=line_alpha)

    # AA circles (only for AA-type centers), clipped
    if len(centers) > 0:
        R = aa_radius_frac * (np.linalg.norm(T1) / np.sqrt(3.0))
        poly = Polygon(clip_polygon)
        for c, t in zip(centers, typ):
            if t != 0:  # only AA
                continue
            if not poly.contains(Point(c)):
                continue
            circ = plt.Circle((c[0], c[1]), R, fill=False, ec=colors[0], lw=1.5, alpha=0.9)
            ax.add_patch(circ)

    # Legend (lines only, small)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0],[0], color=colors[0], lw=2, label="AA"),
        Line2D([0],[0], color=colors[1], lw=2, label="AB"),
        Line2D([0],[0], color=colors[2], lw=2, label="BA"),
    ]
    ax.legend(handles=legend_lines, frameon=False, fontsize=9, loc="upper left")

    plt.tight_layout()
    if out_png is None:
        base = npz_path.rsplit(".", 1)[0]
        out_png = f"{base}_state{state:02d}_moire_overlay.png"
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved overlay PNG] {out_png}")
    return out_png

# ------------------- 3D surface (clean overlay) -------------------
def save_wavefunction_3d_surface_html_clean(npz_path, state,
                                            T1, T2, origin, n_mult,
                                            clip_polygon,
                                            aa_radius_frac=0.22,
                                            line_alpha=0.85, line_width=4.0,
                                            zmode="weight", dx=0.75,
                                            smooth_sigma_A=1.5, clip_q=0.0,
                                            layer="total", out_html=None):
    data = np.load(npz_path)
    XY = data["XY"]; N1 = int(data["N1"])
    Elist = data["E"]; P2all = data["P"][:, state]; Ntot = XY.shape[0]

    if layer == "bottom":
        XYs, P2 = XY[:N1], P2all[:N1]
    elif layer == "top":
        XYs, P2 = XY[N1:], P2all[N1:]
    else:
        XYs, P2 = XY, P2all

    if clip_q and clip_q > 0:
        thr = np.quantile(P2, clip_q)
        XYs, P2 = XYs[P2 >= thr], P2[P2 >= thr]

    eps = 1e-18
    if zmode == "weight":
        zvals = P2 * Ntot; ztitle = "N·|ψ|²"
    elif zmode == "log":
        zvals = np.log10(P2 + eps); ztitle = "log10(|ψ|²)"
    else:
        zvals = P2; ztitle = "|ψ|²"

    # rasterize
    x, y = XYs[:,0], XYs[:,1]
    xmin, xmax = x.min() - dx, x.max() + dx
    ymin, ymax = y.min() - dx, y.max() + dx
    Nx = max(64, int(np.ceil((xmax - xmin) / dx)))
    Ny = max(64, int(np.ceil((ymax - ymin) / dx)))
    xedges = np.linspace(xmin, xmax, Nx + 1)
    yedges = np.linspace(ymin, ymax, Ny + 1)
    Zsum, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=zvals)
    C,    _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    xg = 0.5*(xedges[:-1]+xedges[1:])
    yg = 0.5*(yedges[:-1]+yedges[1:])
    mask = C > 0
    Z = np.zeros_like(Zsum); Z[mask] = Zsum[mask] / C[mask]; Z[~mask] = np.nan

    # optional smoothing
    if smooth_sigma_A and smooth_sigma_A > 0:
        from scipy.ndimage import gaussian_filter
        sig_px = smooth_sigma_A / dx
        Zs = gaussian_filter(np.nan_to_num(Z, nan=0.0), sig_px, mode="nearest")
        Ws = gaussian_filter(mask.astype(float), sig_px, mode="nearest")
        Z = np.divide(Zs, Ws, out=np.full_like(Zs, np.nan), where=(Ws > 1e-8))

    X, Y = np.meshgrid(xg, yg)
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Viridis",
                               colorbar=dict(title=ztitle), showscale=True))
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=40),
                      title=f"|ψ|² surface — {layer}, state #{state} (E/t={Elist[state]:.6f})")

    # add moiré hex edges and AA circles as 3D lines at z=max(Z)
    z_level = float(np.nanmax(Z) + 1e-6)
    centers = _enumerate_hex_centers(origin, T1, T2, n_mult)
    centers = _clip_points_to_polygon(centers, clip_polygon, margin=0.0)

    A = np.column_stack([T1, T2])
    uv = np.linalg.lstsq(A, (centers - origin).T, rcond=None)[0].T
    i = np.round(uv[:,0]).astype(int); j = np.round(uv[:,1]).astype(int)
    typ = (i + j) % 3
    colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}

    def add_line(xs, ys, color, name=None):
        zs = np.full_like(xs, z_level, dtype=float)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                   line=dict(color=color, width=line_width),
                                   showlegend=False if name is None else True,
                                   name=name))

    # hex outlines
    for c, t in zip(centers, typ):
        H = _hex_outline(c, T1, T2, scale=1.0)
        xs = np.r_[H[:,0], H[0,0]]; ys = np.r_[H[:,1], H[0,1]]
        add_line(xs, ys, colors[t])

    # AA circles
    if len(centers) > 0:
        R = aa_radius_frac * (np.linalg.norm(T1) / np.sqrt(3.0))
        poly = Polygon(clip_polygon)
        theta = np.linspace(0, 2*np.pi, 128)
        for c, t in zip(centers, typ):
            if t != 0:  # AA only
                continue
            if not poly.contains(Point(c)):
                continue
            xs = c[0] + R*np.cos(theta)
            ys = c[1] + R*np.sin(theta)
            add_line(xs, ys, colors[0], name="AA")

    if out_html is None:
        base = npz_path.rsplit(".", 1)[0]
        out_html = f"{base}_state{state:02d}_total_surface_clean.html"
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"[saved 3D surface] {out_html}")
    return out_html
