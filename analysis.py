# -*- coding: utf-8 -*-
"""
Post-processing and scaling analysis for in-gap and wall-localized states.
"""

import os
import time
import numpy as np
import pandas as pd
from .spectra import eigs_in_window_sliced, ipr, edge_weight
from .registry import compute_registry_metrics_safe, wall_overlap_all_states
from .geometry import edge_region_mask
# analysis.py (or wherever you prefer)
import scipy.sparse as ss

from .config import Config
from .io_utils import load_config_yaml, ensure_dir


def _linfit(x, y, min_points=2):
    """Robust linear fit with NaN guards."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < min_points or np.allclose(x, x.mean()):
        return (np.array([np.nan, np.nan]), np.nan)
    xm, xs = np.mean(x), np.std(x) or 1.0
    X = np.vstack([(x - xm) / xs, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_scaled, b_scaled = coef
    a = a_scaled / xs
    b = b_scaled - a * xm
    yhat = a * x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-18)
    return (a, b), r2


def run_gap_scan(config, a1_b, a2_b, lattices, builders):
    """Perform n-scan for in-gap and wall-localized state statistics."""
    tb = config.tb
    build = config.build
    window = config.window
    paths = config.paths

    os.makedirs(paths.save_dir, exist_ok=True)
    hp_range = tb.interlayer_mode
    acc = tb.acc

    # choose parameters
    hp = build.hp_values[0]
    if tb.interlayer_mode == "baseline":
        QSIGMA, QPI = 7.42, 3.15
    else:
        QSIGMA, QPI = 3.0, 1.3
    r_xy_cut = hp * acc

    tag = f"r{build.r:02d}_m{build.m:02d}_theta{build.theta_target_deg:.2f}_hp{hp:.2f}"

    rows_meta, per_n_paths = [], []

    for n_mult in build.n_list:
        print(f"\n[scan] n={n_mult} ----------------------------")
        t0 = time.time()

        H, XY_all, N1 = builders.build_flake_H_sparse(
            n_mult, a1_b, a2_b,
            lattices.A_b, lattices.B_b,
            lattices.a1_t, lattices.a2_t,
            lattices.A_t, lattices.B_t,
            lattices.T1, lattices.T2,
            tb.acc, tb.dperp, tb.t, tb.tp,
            r_xy_cut, -1.0 if tb.E_in_t else -tb.t,
            QSIGMA, QPI, tb.E_in_t
        )

        origin = np.array([0.0, 0.0])
        T1, T2 = lattices.T1, lattices.T2

        # crop geometric edges
        d_edge = 10.5 * acc
        mask_edge_L1 = edge_region_mask(XY_all[:N1], origin, T1, T2, n_mult, d_edge)
        mask_edge_L2 = edge_region_mask(XY_all[N1:], origin, T1, T2, n_mult, d_edge)
        keep = ~np.r_[mask_edge_L1, mask_edge_L2]
        H = H.tocsr()[keep][:, keep]
        XY_all = XY_all[keep]
        N1 = int((~mask_edge_L1).sum())

        # solve for eigenstates within energy window
        E, V = eigs_in_window_sliced(
            H, *window.E_window, window.sigmas,
            window.k_per_slice, window.n_states_target
        )

        N_sites = H.shape[0]
        N_states_in_window = len(E)
        L_wall_A, N_wall_states = np.nan, 0

        if len(E) > 0:
            # pick central in-gap state
            E_mid = 0.5 * sum(window.E_window)
            pick = int(np.argmin(np.abs(E - E_mid)))
            psi_pick = V[:, pick]

            L_wall_A, wall_mask = compute_registry_metrics_safe(
                XY_all, N1, psi_pick, a1_b, a2_b, dx_reg=0.5
            )

            if wall_mask is not None:
                overlaps = wall_overlap_all_states(V, N1, XY_all, dx_reg=0.5, wall_mask=wall_mask)
                N_wall_states = int(np.sum(overlaps >= 0.4))

            IPR = ipr(V)
            EW = edge_weight(V, XY_all, origin, T1, T2, n_mult, N1, d_edge, edge_region_mask)

            # save per-n CSV
            df_n = pd.DataFrame({"n": n_mult, "E": E, "IPR": IPR, "EdgeWeight": EW})
            fpath = os.path.join(paths.save_dir, f"flakes_states_{tag}_n{n_mult:02d}.csv")
            df_n.to_csv(fpath, index=False, float_format="%.8e")
            per_n_paths.append(fpath)

        rows_meta.append([n_mult, N_sites, N_states_in_window,
                          L_wall_A if np.isfinite(L_wall_A) else np.nan,
                          N_wall_states])
        print(f"[n={n_mult}] done in {time.time()-t0:.1f}s | N={N_sites} Ngap={N_states_in_window} Nwall={N_wall_states}")

    # save meta
    meta = pd.DataFrame(rows_meta, columns=["n","N_sites","N_states_in_window","L_wall_A","N_wall_states"])
    meta_csv = os.path.join(paths.save_dir, f"flakes_meta_{tag}.csv")
    meta.to_csv(meta_csv, index=False)
    print(f"[saved] {meta_csv}")

    # merge per-n files
    if per_n_paths:
        states = pd.concat([pd.read_csv(p) for p in per_n_paths], ignore_index=True)
        states_csv = os.path.join(paths.save_dir, f"flakes_states_{tag}.csv")
        states.to_csv(states_csv, index=False)
        print(f"[saved] {states_csv}")

    # scaling fits
    if not meta.empty:
        nvals = meta["n"].values
        Lwall = meta["L_wall_A"].values
        Ngap  = meta["N_states_in_window"].values
        Nwall = meta["N_wall_states"].values

        (a1,b1), r2_1 = _linfit(nvals, Ngap)
        (a2,b2), r2_2 = _linfit(Lwall, Ngap)
        (c1,d1), r2_3 = _linfit(Lwall, Nwall)

        print("\n[scaling summaries]")
        print(f" N_gap ~ a*n + b      : a={a1:.3f}, b={b1:.3f}, R²={r2_1:.3f}")
        print(f" N_gap ~ a*L_wall + b : a={a2:.3f}, b={b2:.3f}, R²={r2_2:.3f}")
        print(f" N_wall ~ a*L_wall + b: a={c1:.3f}, b={d1:.3f}, R²={r2_3:.3f}")

    return meta_csv





