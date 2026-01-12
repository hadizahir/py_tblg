#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== USER SETTINGS =====================

# Directory where flakes_meta_*_PBC.csv and flakes_states_* live
save_dir = r"C:\Users\hol1brg\OneDrive - Bosch Group\DAILY\tBLG\bands_pbc"

# Target TB parameters
hp_target   = 0.90          # hp value to select
theta_target = 1.80         # target commensurate angle (deg)
theta_tol    = 0.05         # accept theta in [theta_target - tol, theta_target + tol]

# Energy window (in units of t)
Emin, Emax = 0.012, 0.015

# Which IPR statistic to plot: "IPR_mean" or "IPR_median"
ipr_key = "IPR_mean"

# ========================================================


def classify_SE(m, r):
    """
    Sub-lattice exchange (SE) class from 'r':
      SE-even: r % 3 == 0
      SE-odd : r % 3 != 0
    """
    return "SE-even" if (r % 3 == 0) else "SE-odd"


def collect_ipr_vs_size():
    """Aggregate IPR stats per approximant from meta+states files."""
    pattern_meta = os.path.join(save_dir, "flakes_meta_*_PBC.csv")
    meta_files = sorted(glob.glob(pattern_meta))

    if not meta_files:
        raise RuntimeError(f"No meta files found with pattern: {pattern_meta}")

    rows = []

    for meta_path in meta_files:
        meta = pd.read_csv(meta_path)
        if meta.empty:
            continue

        # Map lowercase -> original column names for robustness
        col = {c.lower(): c for c in meta.columns}

        # Basic parameters
        approx_index = int(meta[col["approx_index"]].iloc[0])
        m            = int(meta[col["m"]].iloc[0])
        r            = int(meta[col["r"]].iloc[0])
        N_sites      = int(meta[col["n_sites"]].iloc[0])
        hp           = float(meta[col["hp"]].iloc[0])

        # Angle column name (in your files it's theta_comm_deg)
        theta_col = col.get("theta_comm_deg") or col.get("theta")
        if theta_col is None:
            raise KeyError(
                f"No theta column found in {meta_path}. "
                f"Available columns: {list(meta.columns)}"
            )
        theta = float(meta[theta_col].iloc[0])

        # Filter by hp and angle
        if abs(hp - hp_target) > 1e-6:
            continue
        if abs(theta - theta_target) > theta_tol:
            continue

        # Tag (present in your files as 'tag')
        tag_col = col.get("tag")
        if tag_col is not None:
            tag = str(meta[tag_col].iloc[0])
        else:
            # Fallback: derive from filename
            base = os.path.basename(meta_path)
            tag = base.replace("flakes_meta_", "").replace(".csv", "")

        # Matching states file
        states_path = os.path.join(save_dir, f"flakes_states_{tag}.csv")
        if not os.path.exists(states_path):
            print(f"[warn] states file missing for tag={tag}: {states_path}")
            continue

        st = pd.read_csv(states_path)
        if st.empty or not {"E", "IPR"}.issubset(st.columns):
            print(f"[warn] bad or empty states for tag={tag}")
            continue

        # Energy window
        mask = (st["E"].values >= Emin) & (st["E"].values <= Emax)
        st_win = st[mask]
        if st_win.empty:
            print(f"[info] no states in [{Emin},{Emax}] for tag={tag}")
            continue

        ipr_vals = st_win["IPR"].values

        rows.append({
            "approx_index": approx_index,
            "tag": tag,
            "m": m,
            "r": r,
            "SE_class": classify_SE(m, r),
            "theta_comm_deg": theta,
            "hp": hp,
            "N_sites": N_sites,
            "N_states_in_window": len(st_win),
            "IPR_mean": float(np.mean(ipr_vals)),
            "IPR_median": float(np.median(ipr_vals)),
            "IPR_max": float(np.max(ipr_vals)),
            "IPR_min": float(np.min(ipr_vals)),
        })

    if not rows:
        raise RuntimeError(
            f"No approximant had states in [{Emin},{Emax}] "
            f"with hp≈{hp_target} and theta≈{theta_target}±{theta_tol}."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("N_sites").reset_index(drop=True)
    return df


def fit_alpha(sub, ycol="IPR_mean"):
    """
    Fit log(IPR) = a + b log(N_sites) and return alpha = -b.
    """
    x = np.log(sub["N_sites"].values.astype(float))
    y = np.log(sub[ycol].values.astype(float))
    # polyfit returns slope, intercept
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    return alpha, slope, intercept


def main():
    df = collect_ipr_vs_size()

    # ---------- Save summary CSV ----------
    out_csv = os.path.join(
        save_dir,
        f"ipr_vs_size_hp{hp_target:.2f}_theta{theta_target:.2f}"
        f"_E[{Emin:.3f},{Emax:.3f}].csv"
    )
    df.to_csv(out_csv, index=False, float_format="%.8e")
    print(f"[info] summary saved to {out_csv}")

    # ---------- Plot IPR vs N (log-log) ----------
    fig, ax = plt.subplots(figsize=(6, 5))

    for SE_class, marker in [("SE-even", "o"), ("SE-odd", "s")]:
        sub = df[df["SE_class"] == SE_class]
        if sub.empty:
            continue
        ax.plot(
            sub["N_sites"].values,
            sub[ipr_key].values,
            marker=marker,
            ls="-",
            label=SE_class,
        )

    # 1/N reference curve
    N_all = df["N_sites"].values.astype(float)
    ax.plot(
        N_all,
        1.0 / N_all,
        ls="--",
        label="1/N"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N_sites (log)")
    ax.set_ylabel(f"{ipr_key} in [{Emin:.3f}, {Emax:.3f}] (log)")
    ax.set_title(
        f"IPR vs approximant size (hp={hp_target:.2f}, θ≈{theta_target:.2f}°)"
    )
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()

    out_png = os.path.join(
        save_dir,
        f"ipr_vs_size_hp{hp_target:.2f}_theta{theta_target:.2f}"
        f"_E[{Emin:.3f},{Emax:.3f}].png"
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[info] plot saved to {out_png}")

    # ---------- Alpha (scaling exponent) calculation ----------
    print("\n=== Scaling exponents α (IPR ~ N^{-α}) ===")
    for cls in ["SE-even", "SE-odd"]:
        sub = df[df["SE_class"] == cls].sort_values("N_sites")
        if len(sub) < 2:
            print(f"{cls}: not enough points")
            continue

        alpha_mean, slope_mean, _ = fit_alpha(sub, "IPR_mean")
        alpha_med,  slope_med,  _ = fit_alpha(sub, "IPR_median")

        print(
            f"{cls}: "
            f"α_mean = {alpha_mean:.4f} (slope = {slope_mean:.4f}), "
            f"α_median = {alpha_med:.4f} (slope = {slope_med:.4f}), "
            f"n = {len(sub)}"
        )


if __name__ == "__main__":
    main()
