# -*- coding: utf-8 -*-
"""
run_pbc_analysis.py — Graph-theoretic analysis of interlayer hoppings.

Reads:  interlayer_hoppings_ij_tag_{tag}.csv
Builds weighted bipartite graph.
Computes spectral, structural and centrality metrics.
Saves CSV + PNG plots, including real-space Laplacian modes.
"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from .config import Config
from .io_utils import load_config_yaml, ensure_dir
from .kwant_bands import build_pbc_H_gamma_from_kwant


# ======================================================================
#  LOAD INTERLAYER GRAPH
# ======================================================================
def load_interlayer_graph(csv_path):
    """
    Load an interlayer graph from a CSV file.

    Supports several column conventions:
    (a) 'ib', 'it', 't_perp'
    (b) 'i_bottom', 'i_top', 't_perp'
    (c) 'x_bottom', 'y_bottom', 'x_top', 'y_top', 't_perp'
    """
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    print("[analysis] CSV columns:", cols)

    colset = set(cols)

    # --- Case 1: integer indices ib / it ---
    if {"ib", "it", "t_perp"} <= colset:
        ib = df["ib"].astype(int).to_numpy()
        it = df["it"].astype(int).to_numpy()
        w = df["t_perp"].astype(float).to_numpy()

        G = nx.Graph()
        for b, t, wt in zip(ib, it, w):
            G.add_edge(f"B_{b}", f"T_{t}", weight=wt)
        return G

    # --- Case 2: integer indices i_bottom / i_top ---
    if {"i_bottom", "i_top", "t_perp"} <= colset:
        ib = df["i_bottom"].astype(int).to_numpy()
        it = df["i_top"].astype(int).to_numpy()
        w = df["t_perp"].astype(float).to_numpy()

        G = nx.Graph()
        for b, t, wt in zip(ib, it, w):
            G.add_edge(f"B_{b}", f"T_{t}", weight=wt)
        return G

    # --- Case 3: coordinate-based CSV ---
    if {"x_bottom", "y_bottom", "x_top", "y_top", "t_perp"} <= colset:
        bottom_coords = list(zip(df["x_bottom"], df["y_bottom"]))
        top_coords = list(zip(df["x_top"], df["y_top"]))
        w = df["t_perp"].astype(float).to_numpy()

        from collections import OrderedDict
        b_map = OrderedDict()
        t_map = OrderedDict()

        def get_b_id(coord):
            if coord not in b_map:
                b_map[coord] = len(b_map)
            return b_map[coord]

        def get_t_id(coord):
            if coord not in t_map:
                t_map[coord] = len(t_map)
            return t_map[coord]

        G = nx.Graph()
        for (xb, yb), (xt, yt), wt in zip(bottom_coords, top_coords, w):
            b_id = get_b_id((xb, yb))
            t_id = get_t_id((xt, yt))
            G.add_edge(f"B_{b_id}", f"T_{t_id}", weight=wt)
        return G

    raise ValueError(f"CSV column names mismatch — got {cols}")


# ======================================================================
#  GRAPH METRICS + LAPLACIAN MODES
# ======================================================================
def compute_graph_metrics(G, tag, save_dir, XY_all=None):
    """
    Compute and save graph metrics for interlayer adjacency graph.

    If XY_all is provided (shape (N_sites,2)), also project the lowest
    Laplacian eigenvectors onto real space and save scatter plots.
    """
    if G is None:
        print(f"[graph] ERROR: Graph is None for tag={tag}, skipping.")
        return

    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    N = G.number_of_nodes()
    M = G.number_of_edges()
    print(f"[graph] nodes={N}, edges={M}")

    # Node order used in adjacency / Laplacian
    nodes = list(G.nodes())

    # ---- degree metrics ----
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))

    deg_vals = np.fromiter(deg.values(), float)
    wdeg_vals = np.fromiter(wdeg.values(), float)

    # ---- betweenness (sampled) ----
    k_sample = min(500, N)
    print(f"[graph] computing approximate betweenness (k={k_sample})...")
    bc = nx.betweenness_centrality(
        G,
        k=k_sample,
        weight="weight",
        normalized=True,
        seed=0,
    )
    bc_vals = np.fromiter(bc.values(), float)

    # ---- adjacency & Laplacian spectra ----
    print("[graph] building sparse adjacency and Laplacian...")

    # Fix node ordering so rows/cols match `nodes`
    A_arr = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    A_sparse = csr_matrix(A_arr, dtype=float)

    # Unnormalized Laplacian L = D - A
    deg_w = np.array(A_sparse.sum(axis=1)).ravel()
    L_sparse = diags(deg_w) - A_sparse

    k_spec = min(80, max(N - 2, 1))

    print(f"[graph] computing top-{k_spec} adjacency eigenvalues...")
    w_adj, _ = eigsh(A_sparse, k=k_spec, which="LA")
    w_adj = np.sort(w_adj)[::-1]

    print(f"[graph] computing bottom-{k_spec} Laplacian eigenvalues (unnormalized)...")
    w_lap, v_lap = eigsh(
        L_sparse,
        k=k_spec,
        sigma=-0.0015,             # shift-invert mode
        which="LM",            # find eigenvalues with largest |1/(λ - σ)|
    )

    order = np.argsort(w_lap)
    w_lap = w_lap[order]
    v_lap = v_lap[:, order]

    # -------- save summary CSV --------
    summary = {
        "tag": [tag],
        "N_nodes": [N],
        "N_edges": [M],
        "deg_mean": [deg_vals.mean()],
        "deg_std": [deg_vals.std()],
        "wdeg_mean": [wdeg_vals.mean()],
        "wdeg_std": [wdeg_vals.std()],
        "bc_mean": [bc_vals.mean()],
        "bc_std": [bc_vals.std()],
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(save_dir, f"graph_summary_{tag}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    # -------- save spectra CSV --------
    spec_df = pd.DataFrame({
        "lambda_adj": w_adj,
        "lambda_lap": w_lap,
    })
    spec_path = os.path.join(save_dir, f"graph_spectra_{tag}.csv")
    spec_df.to_csv(spec_path, index=False)
    print(f"[saved] {spec_path}")

    # ================================================================
    # =======================   PLOTTING   ===========================
    # ================================================================
    # Degree histogram
    plt.figure(figsize=(6, 4))
    plt.hist(deg_vals, bins=40, color="steelblue", alpha=0.8)
    plt.xlabel("degree")
    plt.ylabel("count")
    plt.title(f"Degree histogram ({tag})")
    plt.tight_layout()
    path_deg = os.path.join(save_dir, f"graph_degree_hist_{tag}.png")
    plt.savefig(path_deg, dpi=160)
    plt.close()
    print(f"[saved] {path_deg}")

    # Weighted degree histogram
    plt.figure(figsize=(6, 4))
    plt.hist(wdeg_vals, bins=40, color="firebrick", alpha=0.8)
    plt.xlabel("weighted degree")
    plt.ylabel("count")
    plt.title(f"Weighted degree histogram ({tag})")
    plt.tight_layout()
    path_wdeg = os.path.join(save_dir, f"graph_wdegree_hist_{tag}.png")
    plt.savefig(path_wdeg, dpi=160)
    plt.close()
    print(f"[saved] {path_wdeg}")

    # Betweenness histogram
    plt.figure(figsize=(6, 4))
    plt.hist(bc_vals, bins=40, color="purple", alpha=0.8)
    plt.xlabel("betweenness centrality")
    plt.ylabel("count")
    plt.title(f"Betweenness centrality ({tag})")
    plt.tight_layout()
    path_bc = os.path.join(save_dir, f"graph_betweenness_hist_{tag}.png")
    plt.savefig(path_bc, dpi=160)
    plt.close()
    print(f"[saved] {path_bc}")

    # Adjacency spectrum
    plt.figure(figsize=(6, 4))
    plt.plot(w_adj, "o-", ms=4)
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.title(f"Adjacency spectrum ({tag})")
    plt.tight_layout()
    path_adj = os.path.join(save_dir, f"graph_adj_spectrum_{tag}.png")
    plt.savefig(path_adj, dpi=160)
    plt.close()
    print(f"[saved] {path_adj}")

    # Laplacian spectrum
    plt.figure(figsize=(6, 4))
    plt.plot(w_lap, "o-", ms=4, color="green")
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.title(f"Laplacian spectrum ({tag})")
    plt.tight_layout()
    path_lap = os.path.join(save_dir, f"graph_lap_spectrum_{tag}.png")
    plt.savefig(path_lap, dpi=160)
    plt.close()
    print(f"[saved] {path_lap}")

    # ================================================================
    #     REAL-SPACE PROJECTION OF LOW LAPLACIAN EIGENVECTORS
    # ================================================================
    if XY_all is not None:
        XY_all = np.asarray(XY_all, float)
        n_modes_plot = v_lap.shape[1]

        # Map graph nodes -> XY indices using global 1-based indices from CSV
        XY_nodes = np.zeros((N, 2), float)
        valid_mask = np.zeros(N, bool)

        for k, node in enumerate(nodes):
            # Expect node labels "B_i" or "T_j"
            if "_" not in node:
                continue
            pref, idx_str = node.split("_", 1)
            try:
                # CSV uses 1-based global indices → shift to 0-based for XY_all
                site_idx = int(idx_str) - 1
            except ValueError:
                continue

            if 0 <= site_idx < XY_all.shape[0]:
                XY_nodes[k] = XY_all[site_idx]
                valid_mask[k] = True

        for mode_idx in range(n_modes_plot):
            lam = w_lap[mode_idx]
            u = v_lap[:, mode_idx]

            amp2 = np.abs(u) ** 2
            if amp2.max() > 0:
                amp2 = amp2 / amp2.max()

            from matplotlib.colors import LogNorm

            # avoid log(0)
            vals = amp2[valid_mask]
            vals_nonzero = vals[vals > 0]
            if len(vals_nonzero) > 0:
                vmin = vals_nonzero.min()
                vmax = vals.max()
            else:
                vmin = 1e-12
                vmax = 1e-12

            sc = plt.scatter(
                XY_nodes[valid_mask, 0],
                XY_nodes[valid_mask, 1],
                c=vals,
                s=4,
                cmap="viridis",
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )

            plt.gca().set_aspect("equal", adjustable="box")
            plt.axis("off")
            plt.colorbar(sc, label="|u|²")
            plt.title(f"Laplacian mode #{mode_idx}, λ={lam:.3e}\n{tag}")
            plt.tight_layout()

            out_mode = os.path.join(
                save_dir,
                f"graph_lap_mode_realspace_k{mode_idx}_{tag}.png",
            )
            plt.savefig(out_mode, dpi=200)
            plt.close()
            print(f"[saved] {out_mode}")


# ======================================================================
#  MAIN
# ======================================================================
def parse_tag_m_r_hp(tag: str):
    """
    Parse tag of the form:
        r01_m18_theta1.79_hp0.90_PBC
    and return (m, r, hp) as (int, int, float).
    """
    parts = tag.split("_")
    # parts[0] = 'r01', parts[1] = 'm18', parts[2] = 'theta1.79', parts[3] = 'hp0.90', parts[4] = 'PBC'
    r = int(parts[0][1:])
    m = int(parts[1][1:])
    # strip 'hp'
    hp = float(parts[3][2:])
    return m, r, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="py_tbl/configs/pbc.yaml")
    args = ap.parse_args()

    cfg: Config = load_config_yaml(args.config)
    save_dir = ensure_dir(cfg.paths.save_dir)

    # cache geometry per (m,r) so we don't rebuild H every time
    geometry_cache = {}

    # find all interlayer hopping CSVs
    files = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith("interlayer_hoppings_ij_tag_") and f.endswith(".csv")
    ])

    if not files:
        print("No interlayer_hoppings_ij_tag_*.csv found.")
        return

    for fname in files:
        tag = fname.replace("interlayer_hoppings_ij_tag_", "").replace(".csv", "")
        print(f"\n==== Analyzing tag = {tag} ====")

        csv_path = os.path.join(save_dir, fname)
        try:
            G = load_interlayer_graph(csv_path)
        except Exception as err:
            print(f"[ERROR loading graph] {err}")
            continue

        # parse m,r,hp from tag
        try:
            m_tag, r_tag, hp_tag = parse_tag_m_r_hp(tag)
        except Exception as err:
            print(f"[WARNING] could not parse (m,r,hp) from tag '{tag}': {err}")
            XY_all = None
        else:
            key = (m_tag, r_tag)
            if key in geometry_cache:
                XY_all = geometry_cache[key]
            else:
                tb = cfg.tb
                print(f"[geometry] building PBC H, XY for (m,r,hp)=({m_tag},{r_tag},{hp_tag})...")
                try:
                    H_gamma, XY_all, N1 = build_pbc_H_gamma_from_kwant(
                        m=m_tag,
                        r=r_tag,
                        hp=hp_tag,
                        acc=tb.acc,
                        dperp=tb.dperp,
                        t=tb.t,
                        tp=tb.tp,
                        E_in_t=tb.E_in_t,
                        interlayer_mode=tb.interlayer_mode,
                        registration=tb.registration,
                    )
                    geometry_cache[key] = XY_all
                except Exception as err_geom:
                    print(f"[geometry] WARNING: failed to build PBC geometry: {err_geom}")
                    XY_all = None

        compute_graph_metrics(G, tag, save_dir, XY_all=XY_all)


if __name__ == "__main__":
    main()
