# -*- coding: utf-8 -*-
"""
laplacian_tb_overlap.py — project interlayer Laplacian modes onto low-energy
PBC TB eigenstates at Γ for a given tag.

Usage (inside py_tbl package):
    python -m py_tbl.laplacian_tb_overlap \
        --config py_tbl/configs/pbc.yaml \
        --tag r01_m18_theta1.79_hp0.90_PBC

Outputs in cfg.paths.save_dir:
    laplacian_tb_overlap_{tag}.csv
    laplacian_tb_overlap_heatmap_{tag}.png
    laplacian_tb_energy_vs_lambda_{tag}.png
    laplacian_tb_mode_strength_{tag}.png
"""

import os
import argparse
import numpy as np
import pandas as pd

import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from .io_utils import load_config_yaml, ensure_dir
from .config import Config
from .kwant_bands import build_pbc_H_gamma_from_kwant
from .run_pbc_analysis import load_interlayer_graph, parse_tag_m_r_hp
from .laplacian_plot_utils import (
    plot_overlap_heatmap,
    plot_energy_vs_lambda,
    plot_laplacian_mode_strength,
)


# ----------------------------------------------------------------------
# Laplacian side
# ----------------------------------------------------------------------
def compute_laplacian_modes_for_tag(save_dir, tag, k_lap=20):
    """
    Build interlayer graph for `tag` and return (lambda_lap, v_lap, nodes).

    lambda_lap : array of shape (k_lap,)
    v_lap      : array of shape (N_graph, k_lap)
    nodes      : list of node labels (length N_graph)
    """
    csv_path = os.path.join(save_dir, f"interlayer_hoppings_ij_tag_{tag}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    G = load_interlayer_graph(csv_path)
    nodes = list(G.nodes())
    N = len(nodes)
    print(f"[lap] graph nodes={N}, edges={G.number_of_edges()} for tag={tag}")

    # sparse adjacency with fixed node ordering
    A_arr = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    A_sparse = csr_matrix(A_arr, dtype=float)

    # unnormalized Laplacian L = D - A
    deg_w = np.array(A_sparse.sum(axis=1)).ravel()
    L_sparse = diags(deg_w) - A_sparse

    k = min(k_lap, max(N - 2, 1))
    print(f"[lap] computing lowest {k} Laplacian modes with shift-invert...")
    sigma = 1e-6  # small shift to avoid singularity at λ=0
    w_lap, v_lap = eigsh(L_sparse, k=k, sigma=sigma, which="LM")
    order = np.argsort(w_lap)
    return w_lap[order], v_lap[:, order], nodes


# ----------------------------------------------------------------------
# TB side
# ----------------------------------------------------------------------
def build_tb_gamma_for_tag(cfg: Config, tag: str):
    """Rebuild PBC H_gamma and XY_all for this tag."""
    m, r, hp = parse_tag_m_r_hp(tag)
    tb = cfg.tb
    print(f"[H_gamma] building for tag={tag}  (m={m}, r={r}, hp={hp})")
    H, XY_all, N1 = build_pbc_H_gamma_from_kwant(
        m=m,
        r=r,
        hp=hp,
        acc=tb.acc,
        dperp=tb.dperp,
        t=tb.t,
        tp=tb.tp,
        E_in_t=tb.E_in_t,
        interlayer_mode=tb.interlayer_mode,
        registration=tb.registration,
    )
    return H.tocsr(), XY_all, N1


def build_embedding(nodes, n_sites):
    """
    Map Laplacian graph nodes 'B_i'/'T_j' with global 1-based indices
    into TB site indices [0..n_sites-1].

    Returns:
        graph2site : array of shape (N_graph,) with site indices or -1 if unknown.
    """
    graph2site = np.full(len(nodes), -1, dtype=int)
    for g, node in enumerate(nodes):
        if "_" not in node:
            continue
        pref, idx_str = node.split("_", 1)
        try:
            site_idx = int(idx_str) - 1  # 1-based → 0-based
        except ValueError:
            continue
        if 0 <= site_idx < n_sites:
            graph2site[g] = site_idx
    return graph2site


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="py_tbl/configs/pbc.yaml")
    ap.add_argument("--tag", type=str, required=True,
                    help="tag like r01_m18_theta1.79_hp0.90_PBC")
    ap.add_argument("--k_lap", type=int, default=20,
                    help="number of lowest Laplacian modes to use")
    ap.add_argument("--k_tb", type=int, default=40,
                    help="number of TB eigenstates near E=0 (Γ-point) to use")
    args = ap.parse_args()

    cfg = load_config_yaml(args.config)
    save_dir = ensure_dir(cfg.paths.save_dir)

    tag = args.tag

    # 1) Laplacian modes
    w_lap, v_lap, nodes = compute_laplacian_modes_for_tag(
        save_dir, tag, k_lap=args.k_lap
    )

    # 2) TB Γ-point eigenstates near E=0 (low-energy TB states only)
    Hs, XY_all, N1 = build_tb_gamma_for_tag(cfg, tag)
    n_sites = Hs.shape[0]
    k_tb = min(args.k_tb, n_sites - 2)
    print(f"[TB] computing {k_tb} Γ-point eigenstates near E=0 (shift-invert)...")
    E_tb, psi_tb = eigsh(Hs, k=k_tb, sigma=0.0, which="LM")
    order_tb = np.argsort(E_tb)
    E_tb = E_tb[order_tb]
    psi_tb = psi_tb[:, order_tb]

    # 3) Embed Laplacian modes into TB space and compute overlaps
    graph2site = build_embedding(nodes, n_sites)
    valid_mask = graph2site >= 0

    Lm = v_lap.shape[1]
    overlaps_rows = []

    for ell in range(Lm):
        u = v_lap[:, ell]
        phi = np.zeros(n_sites, dtype=psi_tb.dtype)
        phi[graph2site[valid_mask]] = u[valid_mask]

        # normalize embedded mode
        nrm = np.linalg.norm(phi)
        if nrm > 0:
            phi /= nrm

        coeffs = psi_tb.conj().T @ phi   # shape (k_tb,)
        weights = np.abs(coeffs)**2
        lambda_l = float(w_lap[ell])

        for n in range(k_tb):
            overlaps_rows.append({
                "mode_idx": ell,
                "lambda_lap": lambda_l,
                "state_idx": n,
                "E_tb": float(E_tb[n]),
                "weight": float(weights[n]),
            })

    df = pd.DataFrame(overlaps_rows)
    out_csv = os.path.join(save_dir, f"laplacian_tb_overlap_{tag}.csv")
    df.to_csv(out_csv, index=False, float_format="%.8e")
    print(f"[saved] {out_csv}")

    # 4) Plots
    print("[plot] making overlap heatmap...")
    path_heat = plot_overlap_heatmap(df, tag, save_dir)
    print(f"[saved] {path_heat}")

    print("[plot] making E_tb vs lambda_lap scatter...")
    path_E_vs_l = plot_energy_vs_lambda(df, tag, save_dir)
    print(f"[saved] {path_E_vs_l}")

    print("[plot] making Laplacian mode strength plot...")
    path_strength = plot_laplacian_mode_strength(df, tag, save_dir)
    print(f"[saved] {path_strength}")


if __name__ == "__main__":
    main()
