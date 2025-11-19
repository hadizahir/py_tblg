# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 20:26:21 2025

@author: HOL1BRG
"""

# py_tbl/plots.py
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------- helpers moved from your monolithic script ----------
def _rasterize_weighted(XY, w, dx):
    x, y = XY[:,0], XY[:,1]
    xmin, xmax = x.min()-dx, x.max()+dx
    ymin, ymax = y.min()-dx, y.max()+dx
    Nx = max(64, int(np.ceil((xmax-xmin)/dx)))
    Ny = max(64, int(np.ceil((ymax-ymin)/dx)))
    xedges = np.linspace(xmin, xmax, Nx+1)
    yedges = np.linspace(ymin, ymax, Ny+1)
    H, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=w)   # (y,x)
    C, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    xgrid = 0.5*(xedges[:-1]+xedges[1:])
    ygrid = 0.5*(yedges[:-1]+yedges[1:])
    return H, C, xgrid, ygrid

def _save_surface_html(X, Y, Z, out_html, ztitle):
    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=ztitle),
        contours=dict(z=dict(show=False))
    ))
    fig.update_layout(
        title=out_html.split(os.sep)[-1],
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title=ztitle, aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)

# ---------- public API expected by wavefunctions.py ----------
def batch_save_wavefunction_3d_html(npz_path, states=None, max_points=150_000):
    """Scatter 3D point clouds — minimal working version."""
    data  = np.load(npz_path)
    XY    = data["XY"]
    N1    = int(data["N1"])
    Elist = data["E"]
    P2    = data["P"]  # (N, S)
    S = P2.shape[1]
    if states is None:
        states = list(range(S))

    outs = []
    for s in states:
        Pb = P2[:N1, s]
        Pt = P2[N1:, s]
        XYb, XYt = XY[:N1], XY[N1:]

        # simple decimation
        def _decimate(XY, P, limit):
            n = XY.shape[0]
            if n <= limit: return XY, P
            idx = np.linspace(0, n-1, limit).astype(int)
            return XY[idx], P[idx]

        XYb_d, Pb_d = _decimate(XYb, Pb, max_points//2)
        XYt_d, Pt_d = _decimate(XYt, Pt, max_points//2)

        # log color
        eps = 1e-18
        zb = np.log10(Pb_d + eps); zt = np.log10(Pt_d + eps)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=XYb_d[:,0], y=XYb_d[:,1], z=zb,
            mode='markers',
            marker=dict(size=3, color=zb, colorscale='Hot', opacity=0.95),
            name='Bottom'))
        fig.add_trace(go.Scatter3d(
            x=XYt_d[:,0], y=XYt_d[:,1], z=zt,
            mode='markers',
            marker=dict(size=3, color=zt, colorscale='Hot', opacity=0.85),
            name='Top'))

        fig.update_layout(
            title=f"|ψ|² 3D map — state #{s} (E/t={Elist[s]:.6f})",
            scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="log10 |ψ|²", aspectmode="data"),
            legend=dict(orientation="h", y=1.02, x=0.02),
            margin=dict(l=0,r=0,b=0,t=40)
        )

        out_html = f"{npz_path.rsplit('.',1)[0]}_state{s:02d}.html"
        fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
        outs.append(out_html)
        print(f"[saved 3D HTML] {out_html}")
    return outs

def batch_save_wavefunction_3d_surface_html(npz_path, states=None,
                                            zmode="weight", dx=0.75, smooth_sigma_A=1.5,
                                            clip_q=0.0, layers=("total","bottom","top")):
    """Surface maps — minimal working version."""
    data  = np.load(npz_path)
    XY    = data["XY"]
    N1    = int(data["N1"])
    Elist = data["E"]
    P     = data["P"]             # (N, S)
    Ntot  = XY.shape[0]
    S     = P.shape[1]
    if states is None:
        states = list(range(S))

    outs = []
    for s in states:
        P2 = P[:, s]
        for layer in layers:
            if layer == "bottom":
                XYs = XY[:N1]; Ps = P2[:N1]
            elif layer == "top":
                XYs = XY[N1:]; Ps = P2[N1:]
            else:
                XYs = XY; Ps = P2

            if clip_q and clip_q > 0:
                thr = np.quantile(Ps, clip_q)
                keep = Ps >= thr
                XYs, Ps = XYs[keep], Ps[keep]

            # choose z
            eps = 1e-18
            if zmode.lower() == "weight":
                zvals = Ps * Ntot; ztitle = "N·|ψ|²"
            elif zmode.lower() == "log":
                zvals = np.log10(Ps + eps); ztitle = "log10(|ψ|²)"
            else:
                zvals = Ps; ztitle="|ψ|²"

            Zsum, C, xg, yg = _rasterize_weighted(XYs, zvals, dx)
            mask = C > 0
            Z = np.zeros_like(Zsum, dtype=float); Z[mask] = Zsum[mask] / C[mask]; Z[~mask] = np.nan

            if smooth_sigma_A and smooth_sigma_A > 0:
                sig_px = smooth_sigma_A / dx
                Z_work = Z.copy(); Z_work[~mask] = 0.0
                W = mask.astype(float)
                Z_s = gaussian_filter(Z_work, sig_px, mode="nearest")
                W_s = gaussian_filter(W,      sig_px, mode="nearest")
                Z = np.where(W_s > 1e-6, Z_s/W_s, np.nan)

            X, Y = np.meshgrid(xg, yg)
            base = npz_path.rsplit(".",1)[0]
            out_html = f"{base}_state{s:02d}_{layer}_surface.html"
            _save_surface_html(X, Y, Z, out_html, ztitle)
            outs.append(out_html)
            print(f"[saved 3D surface] {out_html}")
    return outs







def plot_pbc_lattice(XY, N1, title="", savepath=None, show=False):
    import matplotlib.pyplot as plt
    import numpy as np

    XY = np.asarray(XY, float)
    N = XY.shape[0]

    # --- automatic point size ---
    scale = np.sqrt(N / 1000)
    s = 1 / scale
    #s = max(s, 1.0)

    bot = XY[:N1]
    top = XY[N1:]

    fig, ax = plt.subplots(figsize=(25, 25))

    ax.scatter(bot[:, 0], bot[:, 1], s=s, label="bottom", alpha=0.8)
    ax.scatter(top[:, 0], top[:, 1], s=s, marker="x", label="top", alpha=0.8)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)

def plot_pbc_wavefunction_layers(XY, N1, psi, title="", savepath=None, show=False):
    import numpy as np
    import matplotlib.pyplot as plt

    XY = np.asarray(XY, float)
    psi = np.asarray(psi)
    N = XY.shape[0]

    # automatic point size scaling
    scale = np.sqrt(N / 1000)
    s = 1 / scale
    #s = max(s, 1.0)

    dens = np.abs(psi)**2
    dens_bot = dens[:N1]
    dens_top = dens[N1:]

    XY_bot = XY[:N1]
    XY_top = XY[N1:]

    vmin = 0.0
    vmax = dens.max() if dens.max() > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(50, 25), sharex=True, sharey=True)

    axL, axR = axes

    scL = axL.scatter(XY_bot[:, 0], XY_bot[:, 1],
                      c=dens_bot, s=s, cmap="viridis", vmin=vmin, vmax=vmax)
    axL.set_title("bottom layer")
    axL.set_aspect("equal", "box")

    scR = axR.scatter(XY_top[:, 0], XY_top[:, 1],
                      c=dens_top, s=s, cmap="viridis", vmin=vmin, vmax=vmax)
    axR.set_title("top layer")
    axR.set_aspect("equal", "box")

    fig.suptitle(title)
    fig.colorbar(scR, ax=axes.ravel().tolist(), shrink=0.85)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt

def plot_wavefunction_heatmap(XY, psi, 
                              nbins=400,
                              cmap="viridis",
                              title="",
                              savepath=None):
    """
    Pure 2D heatmap of |psi|^2 on the moiré cell.

    Parameters
    ----------
    XY : (N,2) array
        Real-space positions.
    psi : (N,) array (complex)
        Eigenvector components.
    nbins : int
        Number of histogram bins for the heatmap.
    cmap : str
        Color map.
    savepath : str or None
        If given, save the figure here.
    """

    x = XY[:, 0]
    y = XY[:, 1]
    w = np.abs(psi)**2

    # histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=w)


   

    # smoother heatmap
    H_smooth = gaussian_filter(H, sigma=3.0)   # <--- adjust sigma for smoothing
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(
        H_smooth.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\psi|^2$", fontsize=12)

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_wavefunction_heatmap_log(
    XY,
    psi,
    nbins=350,
    sigma=1.,
    cmap="viridis",
    title="",
    savepath_log=None,
    eps=1e-12,
):
    """
    Log-scale 2D heatmap of |psi|^2 to highlight weak domain-wall tails.

    Parameters
    ----------
    XY : (N, 2)
        Site positions (x, y) for all sites (both layers).
    psi : (N,)
        Eigenvector components for each site (complex or real).
    nbins : int
        Number of bins in each dimension for the 2D histogram.
    sigma : float
        Gaussian smoothing width in pixel units (2–3 works well).
    cmap : str
        Matplotlib colormap.
    title : str
        Figure title (optional).
    savepath_log : str or None
        Where to save the log-scale PNG. If None, just returns fig, ax.
    eps : float
        Small constant added before log10 to avoid log(0).
    """

    x = XY[:, 0]
    y = XY[:, 1]
    w = np.abs(psi)**2

    # 2D histogram of |psi|^2
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=w)

    # Smooth to make patterns visually clear
    H = gaussian_filter(H, sigma=sigma)

    # Log10 scale to reveal weak tails
    H_log = np.log10(H + eps)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(
        H_log.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title + " (log₁₀ scale)", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"log₁₀(|ψ|² + ε)", fontsize=12)

    if savepath_log is not None:
        fig.savefig(savepath_log, dpi=200, bbox_inches="tight")

    return fig, ax

def plot_interlayer_links(
    XY_all,
    N1,
    H_sparse,
    title="tBLG PBC interlayer hoppings",
    savepath=None,
    show=False,
    r_xy_cut=None,
    min_abs_t=None,
    max_links=None,
    s_sites=3.0,
):
    """
    Plot lattice with ONLY interlayer hopping links drawn.

    Parameters
    ----------
    XY_all : (N, 2) array
        Coordinates of all sites; bottom layer first, then top.
    N1 : int
        Number of bottom-layer sites.
    H_sparse : csr or coo matrix (N x N)
        PBC Hamiltonian at k=0 (Γ-point).
    title : str
        Plot title.
    savepath : str or None
        If given, save figure to this path.
    show : bool
        If True, call plt.show().
    r_xy_cut : float or None
        If not None, only draw interlayer links with in-plane distance <= r_xy_cut.
    min_abs_t : float or None
        If not None, only draw interlayer links with |t_perp| >= min_abs_t.
    max_links : int or None
        If not None and too many links, keep only the strongest |t_perp| for plotting.
    s_sites : float
        Marker size for sites.
    """

    H = H_sparse.tocoo()
    rows = H.row
    cols = H.col
    vals = H.data

    XY = XY_all
    N = XY.shape[0]

    # ---- collect interlayer segments ----
    segments = []
    strengths = []

    for i, j, tval in zip(rows, cols, vals):
        # avoid double counting; we only use i < j
        if i >= j:
            continue

        # check if this is interlayer
        bottom_top = (i < N1 and j >= N1)
        top_bottom = (i >= N1 and j < N1)

        if not (bottom_top or top_bottom):
            continue

        # reorder so ib is bottom, it is top
        if i < N1:
            ib, it = i, j
        else:
            ib, it = j, i

        rb = XY[ib]
        rt = XY[it]

        # in-plane distance
        dx, dy = rt[0] - rb[0], rt[1] - rb[1]
        dist = float(np.hypot(dx, dy))

        if r_xy_cut is not None and dist > r_xy_cut:
            continue

        if min_abs_t is not None and abs(tval) < min_abs_t:
            continue

        segments.append([rb, rt])
        strengths.append(abs(tval))

    if not segments:
        print("[plot_interlayer_links] No interlayer links passed the filters.")
        return

    segments = np.asarray(segments)
    strengths = np.asarray(strengths)

    # ---- optionally subsample strongest links if too many ----
    if max_links is not None and len(segments) > max_links:
        idx_sorted = np.argsort(strengths)[::-1]  # strongest first
        idx_keep = idx_sorted[:max_links]
        segments = segments[idx_keep]
        strengths = strengths[idx_keep]
        print(f"[plot_interlayer_links] Too many links, plotting only {max_links} strongest.")

    # ---- normalize strengths for coloring ----
    if strengths.max() > 0:
        strengths_norm = strengths / strengths.max()
    else:
        strengths_norm = strengths

    # ---- build figure ----
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot sites: bottom (blue), top (red, slightly transparent)
    XY_bottom = XY[:N1]
    XY_top    = XY[N1:]

    ax.scatter(
        XY_bottom[:, 0],
        XY_bottom[:, 1],
        s=s_sites,
        color="tab:blue",
        alpha=0.7,
        label="bottom layer",
    )
    ax.scatter(
        XY_top[:, 0],
        XY_top[:, 1],
        s=s_sites,
        color="tab:red",
        alpha=0.7,
        label="top layer",
    )

    # line collection for interlayer links
    lc = LineCollection(
        segments,
        cmap="viridis",
        array=strengths_norm,
        linewidths=0.5,
        alpha=0.9,
    )
    ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=8)

    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|t_\perp|$ (normalized)", fontsize=11)

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"[saved interlayer link plot] {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)
