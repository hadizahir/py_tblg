# -*- coding: utf-8 -*-
"""
run_pbc.py — periodic-approximant (PBC) driver for tBLG (Γ-point via kwant)

- reads config (YAML → Config),
- optionally loops over a list of (m,r) approximants (build.approximants),
  otherwise uses single build.m, build.r,
- for each approximant:
    - builds ONE commensurate (m,r) supercell with PBC using kwant logic,
    - constructs the Γ-point Hamiltonian H(k=0),
    - diagonalizes near E_center (shift–invert with fallback),
    - saves E and IPR to CSV,
    - saves lattice + one wavefunction PNG.

Also writes a ladder summary CSV if multiple approximants are run.
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from .config import Config
from .io_utils import load_config_yaml, ensure_dir
from .geometry import theta_comm_deg_from_mr
from .kwant_bands import build_pbc_H_gamma_from_kwant
from .spectra import ipr
from .plots import (
    plot_pbc_lattice,
    plot_pbc_wavefunction_layers,
    plot_wavefunction_heatmap,   # <-- ADDED
    plot_wavefunction_heatmap_log   # <-- ADDED

)
from .io_utils import save_interlayer_hoppings_csv
from .plots import plot_interlayer_links

def run_one_approximant(
    cfg: Config,
    m: int,
    r: int,
    hp: float,
    approx_idx: int | None = None,
):
    """Run full PBC pipeline for a single (m,r) approximant."""
    # --- TB parameters ---
    acc   = cfg.tb.acc
    dperp = cfg.tb.dperp
    t     = cfg.tb.t
    tp    = cfg.tb.tp
    E_in_t        = cfg.tb.E_in_t
    registration  = cfg.tb.registration
    interlayer_mode = cfg.tb.interlayer_mode

    # --- window parameters ---
    k_eigs_target = cfg.window.n_states_target
    E_center = getattr(cfg.window, "E_center", 0.0)
    wf_index = getattr(cfg.window, "wf_index", 0)

    # commensurate twist angle
    theta_comm = theta_comm_deg_from_mr(m, r)

    print()
    print("=" * 70)
    print(f"[PBC] starting approximant (m={m}, r={r}), θ_comm={theta_comm:.5f}°")
    if approx_idx is not None:
        print(f"[PBC] approximant index in ladder: {approx_idx}")
    print("=" * 70)

    # ----------------- build Γ-point H via kwant -----------------
    H, XY_all, N1 = build_pbc_H_gamma_from_kwant(
        m=m,
        r=r,
        hp=hp,
        acc=acc,
        dperp=dperp,
        t=t,
        tp=tp,
        E_in_t=E_in_t,
        interlayer_mode=interlayer_mode,
        registration=registration,
    )
    N_sites = H.shape[0]

    # ----------------- connectivity diagnostic -----------------
    row_sums = np.abs(H).sum(axis=1).A.ravel()
    deg = (H != 0).sum(axis=1).A.ravel()
    print("N_sites =", H.shape[0])
    print("N1 (bottom) =", N1, "N2 (top) =", H.shape[0] - N1)
    print("\n=== CONNECTIVITY DIAGNOSTIC (Γ via kwant) ===")
    print(f"Min row_sum  = {row_sums.min():.6f}")
    print(f"Max row_sum  = {row_sums.max():.6f}")
    print(f"Min degree   = {deg.min()}")
    print(f"Max degree   = {deg.max()}")

    unique_deg, counts_deg = np.unique(deg, return_counts=True)
    print("\nDegree histogram (number of nonzero neighbors per site):")
    for u, c in zip(unique_deg, counts_deg):
        print(f"  {u:.0f} neighbors: {c} sites")
    print("=============================================\n")

    # ----------------- save results -----------------
    save_dir = ensure_dir(cfg.paths.save_dir)
    tag = f"r{r:02d}_m{m:02d}_theta{theta_comm:.2f}_hp{hp:.2f}_PBC"
    #-------------------save interlayer hoppings-------
    save_interlayer_hoppings_csv(
        XY_all,
        N1,
        H,
        save_dir,
        tag,
        r_xy_cut=None,
    )

    print(f"[saved interlayer hoppings entries)")

    # --- plot lattice with interlayer hoppings only ---
    interlayer_png = os.path.join(save_dir, f"interlayer_links_{tag}.png")
    plot_interlayer_links(
        XY_all,
        N1,
        H,
        title=f"Interlayer hoppings (m={m}, r={r}, θ={theta_comm:.2f}°)",
        savepath=interlayer_png,
        show=False,
        r_xy_cut=hp*acc,      # None or hp*acc if you want explicit cutoff
        min_abs_t=None,     # or e.g. 1e-3 to hide tiny hoppings
        max_links=50000,    # optional safety for huge systems
    )








    # lattice PNG
    lattice_png = os.path.join(save_dir, f"lattice_{tag}.png")
    plot_pbc_lattice(
        XY_all,
        N1,
        title=f"tBLG PBC lattice (m={m}, r={r}, θ={theta_comm:.2f}°)",
        savepath=lattice_png,
        show=False,
    )
    print(f"[saved] {lattice_png}")

    # ----------------- diagonalize central spectrum -----------------
    k_eigs = min(k_eigs_target, N_sites - 2)
    """"
    try:
        print(
            f"[PBC] eigsh: shift–invert with sigma={E_center:.4f}, "
            f"which='LM', k={k_eigs}"
        )
        E, V = eigsh(H, k=k_eigs, sigma=E_center, which="LM")
    except Exception as err:
        print(f"[PBC] shift–invert failed ({err}). Falling back to which='SM'.")
        E, V = eigsh(H, k=k_eigs, which="SM")

    E = np.real(E)
    order = np.argsort(E)
    E, V = E[order], V[:, order]

    print(
        f"[PBC] eigsh returned {len(E)} eigenvalues; "
        f"E_min={E[0]:.4f}, E_max={E[-1]:.4f}"
    )

    # ----------------- existing wf_index-based layer plot -----------------
    if wf_index < 0 or wf_index >= len(E):
        print(f"[PBC] requested wf_index={wf_index} out of range, using 0 instead.")
        wf_index = 0
    psi = V[:, wf_index]
    wf_title = (
        f"tBLG PBC wavefunction (state {wf_index}, "
        f"E={E[wf_index]:.4f}, m={m}, r={r}, θ={theta_comm:.2f}°)"
    )
    wf_png = os.path.join(save_dir, f"wf_state{wf_index}_{tag}.png")
    plot_pbc_wavefunction_layers(
        XY_all,
        N1,
        psi,
        title=wf_title,
        savepath=wf_png,
        show=False,
    )
    print(f"[saved] {wf_png}")

    # ============================================================
    # NEW: SE-even / SE-odd selection + heatmap plots + save ψ’s
    # ============================================================

    # helper: choose index closest to a target energy
    def _closest_index(E_arr, target_E):
        return int(np.argmin(np.abs(E_arr - target_E)))

    # classify SE-even vs SE-odd (Lopes dos Santos convention: r multiple of 3)
    is_SE_even = (r % 3 == 0)

    if is_SE_even:
        print("[PBC] SE-even approximant → selecting state near E ≈ -0.015 t")
        targets = [(-0.015, "SEeven_E-0p015")]
    else:
        print("[PBC] SE-odd approximant → selecting 3 states near "
              "E ≈ -0.002, -0.003, -0.004 t")
        targets = [
            (-0.002, "SEodd_E-0p002"),
            (-0.003, "SEodd_E-0p003"),
            (-0.004, "SEodd_E-0p004"),
        ]

    # directory for wavefunction heatmaps & data
    wf_dir = ensure_dir(os.path.join(save_dir, "wf"))
    sel_indices = []
    sel_energies = []
    sel_psi = []

    for target_E, label in targets:
        idx = _closest_index(E, target_E)
        sel_indices.append(idx)
        sel_energies.append(E[idx])
        sel_psi.append(V[:, idx].copy())

        heat_png_log = os.path.join(
            wf_dir,
            f"{label}_state{idx:03d}_{tag}_log.png"
        )
        title = (f"{label}: state {idx}, "
                 f"E={E[idx]:.6f}, m={m}, r={r}, θ={theta_comm:.2f}°")

        # Log-scale heatmap to highlight weak domain-wall tails
        plot_wavefunction_heatmap_log(
            XY_all,
            V[:, idx],
            title=title,
            savepath_log=heat_png_log,
        )
        print(f"[saved] log heatmap {heat_png_log}")


    # save selected wavefunctions for later processing
    sel_indices = np.asarray(sel_indices, dtype=int)
    sel_energies = np.asarray(sel_energies, dtype=float)
    # stack as (N_sites, N_selected)
    sel_psi_arr = np.stack(sel_psi, axis=1)

    wf_npz = os.path.join(wf_dir, f"wf_selected_{tag}.npz")
    np.savez(
        wf_npz,
        indices=sel_indices,
        energies=sel_energies,
        psi=sel_psi_arr,
    )
    print(f"[saved] selected wavefunctions to {wf_npz}")

    # ----------------- states CSV: energies + IPR -----------------
    out_states = os.path.join(save_dir, f"flakes_states_{tag}.csv")
    if len(E) > 0:
        df_states = pd.DataFrame({
            "idx": np.arange(len(E), dtype=int),
            "E": E.astype(float),
            "IPR": ipr(V).astype(float),
        })
        df_states.to_csv(out_states, index=False, float_format="%.8e")
        print(f"[saved] {out_states}")
    else:
        print("[info] No states found (unexpected for PBC).")
        df_states = None

    # meta CSV: basic run metadata for this approximant
    out_meta = os.path.join(save_dir, f"flakes_meta_{tag}.csv")
    meta_row = {
        "approx_index": -1 if approx_idx is None else int(approx_idx),
        "tag": tag,
        "N_sites": int(N_sites),
        "N_states": int(len(E)),
        "m": int(m),
        "r": int(r),
        "theta_comm_deg": float(theta_comm),
        "hp": float(hp),
        "acc": float(acc),
        "dperp": float(dperp),
        "t": float(t),
        "tp": float(tp),
        "E_in_t": bool(E_in_t),
        "interlayer_mode": interlayer_mode,
        "E_center": float(E_center),
        "E_min": float(E[0]) if len(E) > 0 else np.nan,
        "E_max": float(E[-1]) if len(E) > 0 else np.nan,
    }
    pd.DataFrame([meta_row]).to_csv(out_meta, index=False)
    print(f"[saved] {out_meta}")

    return meta_row
    """

def main():
    # ----------------- parse config -----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = ap.parse_args()
    cfg: Config = load_config_yaml(args.config) if args.config else Config()

    # --- decide approximant list ---
    approx_list = getattr(cfg.build, "approximants", None)

    if approx_list is None:
        # single (m,r) from build.m, build.r
        m_single = cfg.build.m
        r_single = cfg.build.r
        hp = cfg.build.hp_values[0]
        meta = run_one_approximant(cfg, m_single, r_single, hp, approx_idx=None)
        # also save a one-row ladder summary
        save_dir = ensure_dir(cfg.paths.save_dir)
        ladder_csv = os.path.join(save_dir, "pbc_ladder_summary.csv")
        pd.DataFrame([meta]).to_csv(ladder_csv, index=False)
        print(f"[PBC] ladder summary saved to {ladder_csv}")
    else:
        # list of approximants
        hp = cfg.build.hp_values[0]
        meta_rows = []
        for idx, entry in enumerate(approx_list):
            m = int(entry["m"])
            r = int(entry["r"])
            meta = run_one_approximant(cfg, m, r, hp, approx_idx=idx)
            meta_rows.append(meta)

        # write ladder summary
        save_dir = ensure_dir(cfg.paths.save_dir)
        ladder_csv = os.path.join(save_dir, "pbc_ladder_summary.csv")
        pd.DataFrame(meta_rows).to_csv(ladder_csv, index=False)
        print(f"\n[PBC] ladder summary saved to {ladder_csv}")


if __name__ == "__main__":
    main()
