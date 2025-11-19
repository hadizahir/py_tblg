# -*- coding: utf-8 -*-
"""
pbc_ladder_analysis.py — post-processing for a ladder of PBC approximants.

- reads pbc_ladder_summary.csv
- reads each flakes_states_{tag}.csv
- builds a combined spectrum CSV (approximant index, m, r, N_sites, state_idx, E, IPR)
- computes IPR scaling near E_center ± dE
- makes two plots:
    * spectral flow (E vs approximant index)
    * IPR vs N_sites in a small energy window
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--save_dir", type=str, default="bands_pbc",
        help="Directory where run_pbc outputs live (pbc_ladder_summary.csv, flakes_*)."
    )
    ap.add_argument(
        "--dE", type=float, default=0.02,
        help="Energy window half-width around E_center for IPR scaling."
    )
    args = ap.parse_args()

    save_dir = args.save_dir
    dE = args.dE

    ladder_csv = os.path.join(save_dir, "pbc_ladder_summary.csv")
    if not os.path.isfile(ladder_csv):
        raise FileNotFoundError(f"Did not find ladder summary: {ladder_csv}")

    ladder = pd.read_csv(ladder_csv)
    print(f"[analysis] loaded ladder summary with {len(ladder)} approximants")

    # We assume all have same E_center (from run_pbc)
    if "E_center" in ladder.columns:
        E_center = float(ladder["E_center"].iloc[0])
    else:
        E_center = 0.0
    print(f"[analysis] using E_center = {E_center:.4f}, dE = {dE:.4f}")

    # --- build combined spectrum table ---
    all_rows = []

    for _, row in ladder.iterrows():
        tag = row["tag"]
        approx_index = int(row.get("approx_index", -1))
        m = int(row["m"])
        r = int(row["r"])
        N_sites = int(row["N_sites"])

        states_path = os.path.join(save_dir, f"flakes_states_{tag}.csv")
        if not os.path.isfile(states_path):
            print(f"[warn] missing states file for tag={tag}: {states_path}")
            continue

        df_states = pd.read_csv(states_path)
        # expected columns: idx, E, IPR
        if "idx" not in df_states.columns:
            df_states["idx"] = np.arange(len(df_states), dtype=int)

        for _, st in df_states.iterrows():
            all_rows.append({
                "approx_index": approx_index,
                "m": m,
                "r": r,
                "N_sites": N_sites,
                "state_idx": int(st["idx"]),
                "E": float(st["E"]),
                "IPR": float(st["IPR"]),
                "tag": tag,
            })

    if not all_rows:
        raise RuntimeError("No states loaded — check that flakes_states_* exist.")

    spectrum = pd.DataFrame(all_rows)
    spectrum_csv = os.path.join(save_dir, "pbc_ladder_spectrum.csv")
    spectrum.to_csv(spectrum_csv, index=False, float_format="%.8e")
    print(f"[analysis] combined spectrum saved to {spectrum_csv}")

    # --- spectral flow plot: E vs approximant index ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    for idx in sorted(spectrum["approx_index"].unique()):
        spec_i = spectrum[spectrum["approx_index"] == idx]
        ax1.scatter(
            np.full(len(spec_i), idx),
            spec_i["E"].values,
            s=5,
            alpha=0.5,
            label=f"approx {idx}" if idx == 0 else None,
        )

    ax1.axhline(E_center, color="k", lw=1, ls="--", alpha=0.5)
    ax1.set_xlabel("approximant index")
    ax1.set_ylabel("E (in units of t)")
    ax1.set_title("PBC ladder spectral flow (Γ-point)")
    plt.tight_layout()
    flow_png = os.path.join(save_dir, "pbc_ladder_spectral_flow.png")
    fig1.savefig(flow_png, dpi=300)
    plt.close(fig1)
    print(f"[analysis] spectral flow plot saved to {flow_png}")

    # --- IPR scaling in a window around E_center ---
    approx_results = []

    for idx in sorted(spectrum["approx_index"].unique()):
        spec_i = spectrum[spectrum["approx_index"] == idx]
        N_sites = int(spec_i["N_sites"].iloc[0])

        mask = np.abs(spec_i["E"].values - E_center) <= dE
        sub = spec_i[mask]
        if len(sub) == 0:
            print(f"[analysis] no states in window for approx_index={idx}")
            continue

        approx_results.append({
            "approx_index": idx,
            "N_sites": N_sites,
            "N_states_in_window": len(sub),
            "IPR_mean": sub["IPR"].mean(),
            "IPR_median": sub["IPR"].median(),
            "IPR_max": sub["IPR"].max(),
            "IPR_min": sub["IPR"].min(),
        })

    if not approx_results:
        print("[analysis] no approximant had states in the energy window, aborting IPR plot.")
        return

    ipr_df = pd.DataFrame(approx_results)
    ipr_csv = os.path.join(save_dir, "pbc_ladder_ipr_scaling.csv")
    ipr_df.to_csv(ipr_csv, index=False, float_format="%.8e")
    print(f"[analysis] IPR scaling summary saved to {ipr_csv}")

    # plot IPR_mean vs N_sites
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(
        ipr_df["N_sites"].values,
        ipr_df["IPR_mean"].values,
        marker="o",
        ls="-",
        label="mean IPR",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("N_sites (log)")
    ax2.set_ylabel("mean IPR in window (log)")
    ax2.set_title(f"IPR scaling near E={E_center:.3f} ± {dE:.3f}")
    ax2.legend()
    plt.tight_layout()
    ipr_png = os.path.join(save_dir, "pbc_ladder_ipr_scaling.png")
    fig2.savefig(ipr_png, dpi=300)
    plt.close(fig2)
    print(f"[analysis] IPR scaling plot saved to {ipr_png}")


if __name__ == "__main__":
    main()
