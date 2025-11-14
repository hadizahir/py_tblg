# -*- coding: utf-8 -*-
"""
run_pbc.py — periodic-approximant (PBC) driver for tBLG (Γ-point via kwant)

- reads config (YAML → Config),
- builds ONE commensurate (m,r) supercell with PBC using kwant logic,
- constructs the Γ-point Hamiltonian H(k=0),
- diagonalizes near E_center (shift–invert with fallback),
- saves E and IPR to CSV.
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
from .plots import plot_pbc_lattice
from .plots import plot_pbc_lattice, plot_pbc_wavefunction_layers


def main():
    # ----------------- parse config -----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = ap.parse_args()
    cfg: Config = load_config_yaml(args.config) if args.config else Config()

    # -------- TB parameters (from pbc.yaml) --------
    acc   = cfg.tb.acc
    dperp = cfg.tb.dperp
    t     = cfg.tb.t
    tp    = cfg.tb.tp
    E_in_t        = cfg.tb.E_in_t
    registration  = cfg.tb.registration
    interlayer_mode = cfg.tb.interlayer_mode

    # -------- build parameters (from pbc.yaml) --------
    m  = cfg.build.m
    r  = cfg.build.r
    hp = cfg.build.hp_values[0]   # first hp in list

    # -------- window parameters (from pbc.yaml) --------
    k_eigs_target = cfg.window.n_states_target
    E_center = getattr(cfg.window, "E_center", 0.0)

    # commensurate twist angle (degrees)
    theta_comm = theta_comm_deg_from_mr(m, r)

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



    # ----------------- plot lattice (diagnostic) -----------------
    save_dir = ensure_dir(cfg.paths.save_dir)
    tag = f"r{r:02d}_m{m:02d}_theta{theta_comm:.2f}_hp{hp:.2f}_PBC"

    lattice_png = os.path.join(save_dir, f"lattice_{tag}.png")
    plot_pbc_lattice(
        XY_all,
        N1,
        title=f"tBLG PBC lattice (m={m}, r={r}, θ={theta_comm:.2f}°)",
        savepath=lattice_png,
        show=False,  # set True if you want an interactive window
    )
    print(f"[saved] {lattice_png}")



    # ----------------- diagonalize central spectrum -----------------
    k_eigs = min(k_eigs_target, N_sites - 2)

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

    tag = f"r{r:02d}_m{m:02d}_theta{theta_comm:.2f}_hp{hp:.2f}_PBC"
    print(f"[PBC] N_sites={N_sites}, states={len(E)}, tag={tag}")

    # ----------------- save results -----------------
    save_dir = ensure_dir(cfg.paths.save_dir)

    out_states = os.path.join(save_dir, f"flakes_states_{tag}.csv")
    if len(E) > 0:
        df = pd.DataFrame({
            "E": E.astype(float),
            "IPR": ipr(V).astype(float),
        })
        df.to_csv(out_states, index=False, float_format="%.8e")
        print(f"[saved] {out_states}")
    else:
        print("[info] No states found (unexpected for PBC).")

    out_meta = os.path.join(save_dir, f"flakes_meta_{tag}.csv")
    pd.DataFrame([{
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
    }]).to_csv(out_meta, index=False)
    print(f"[saved] {out_meta}")
    E = np.real(E)
    order = np.argsort(E)
    E, V = E[order], V[:, order]

    print(
        f"[PBC] eigsh returned {len(E)} eigenvalues; "
        f"E_min={E[0]:.4f}, E_max={E[-1]:.4f}"
    )

    tag = f"r{r:02d}_m{m:02d}_theta{theta_comm:.2f}_hp{hp:.2f}_PBC"
    print(f"[PBC] N_sites={N_sites}, states={len(E)}, tag={tag}")

    # ----------------- plot one wavefunction (diagnostic) -----------------
    save_dir = ensure_dir(cfg.paths.save_dir)

    # choose which eigenstate to plot:
    # default: the one closest to E_center (after sorting), i.e. index 0,
    # or allow override via cfg.window.wf_index if present.
    wf_index = getattr(cfg.window, "wf_index", 0)
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


if __name__ == "__main__":
    main()
