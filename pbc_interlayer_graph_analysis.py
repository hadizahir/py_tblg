#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pbc_interlayer_graph_analysis.py

Analyse the interlayer hopping network of a commensurate approximant.

Input  : CSV with one row per interlayer hopping:
           i, j, t_perp
         (column names configurable below)

Output :
  - distinct_t_perp.csv
        index, t_perp_distinct, frequency
    (distinct |t_perp| values, grouped within a tolerance and sorted by value)

  - degree_histogram.csv
        degree, count

  - graph_summary.txt
        basic stats (N_nodes, N_edges, avg_degree, clustering, components, ...)

  - laplacian_eigs.csv  (optional; if --n-eigs > 0)
        k, lambda_k
    (smallest Laplacian eigenvalues of the interlayer graph)

Usage:
    python pbc_interlayer_graph_analysis.py \
        --input interlayer_hoppings_r09_m161_theta1.80.csv \
        --outdir ./graph_analysis \
        --tol 1e-3 \
        --n-eigs 40
"""

import os
import argparse

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


# --------- Adjust these if your CSV uses different headers ---------
COL_I = "i"
COL_J = "j"
COL_T = "t_perp"
# -------------------------------------------------------------------


def load_interlayer_csv(path):
    df = pd.read_csv(path)
    for col in (COL_I, COL_J, COL_T):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. "
                             f"Available columns: {list(df.columns)}")
    return df


def compute_distinct_t(df, tol=1e-3):
    """
    Group |t_perp| values within tolerance 'tol' and return:
      distinct_vals, freqs
    where distinct_vals[i] is the representative |t| of cluster i,
    and freqs[i] is the number of occurrences.
    """
    tvals = np.abs(df[COL_T].values.astype(np.complex128))
    if tvals.size == 0:
        return np.array([]), np.array([])

    t_sorted = np.sort(tvals)
    distinct = []
    counts = []

    current_val = t_sorted[0]
    current_count = 1

    for x in t_sorted[1:]:
        if abs(x - current_val) <= tol:
            current_count += 1
        else:
            distinct.append(current_val)
            counts.append(current_count)
            current_val = x
            current_count = 1

    # last cluster
    distinct.append(current_val)
    counts.append(current_count)

    distinct = np.array(distinct)
    counts = np.array(counts, dtype=int)

    # sort by distinct value (ascending) and reset indices later
    order = np.argsort(distinct)
    distinct = distinct[order]
    counts = counts[order]

    return distinct, counts


def build_graph(df):
    """
    Build an undirected weighted graph G from interlayer hoppings.
    Edge weight = |t_perp|.
    Also return sparse adjacency matrix A (scipy coo) and number of nodes N.
    """
    i = df[COL_I].to_numpy(dtype=int)
    j = df[COL_J].to_numpy(dtype=int)
    w = np.abs(df[COL_T].to_numpy(dtype=np.complex128))

    if i.size == 0:
        raise ValueError("No interlayer hoppings in dataframe.")

    N = int(max(i.max(), j.max()) + 1)

    # sparse adjacency
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([w, w])

    A = coo_matrix((vals, (rows, cols)), shape=(N, N))

    # networkx graph
    G = nx.Graph()
    for a, b, weight in zip(i, j, w):
        if G.has_edge(a, b):
            # keep the max weight if there are duplicates
            G[a][b]["weight"] = max(G[a][b]["weight"], weight)
        else:
            G.add_edge(a, b, weight=weight)

    return G, A, N


def degree_histogram(A):
    """
    Degree histogram (unweighted degree = number of non-zero neighbors).
    Returns degrees, counts.
    """
    deg = (A != 0).sum(axis=1).A.ravel().astype(int)
    vals, counts = np.unique(deg, return_counts=True)
    return vals, counts


def laplacian_small_eigs(A, k=20):
    """
    Smallest k eigenvalues of the weighted graph Laplacian L = D - A.
    Uses sparse eigsh, so k should be small.
    """
    if k <= 0:
        return np.array([])

    # Degree matrix (weighted degree)
    deg_w = np.array(A.sum(axis=1)).ravel()
    L = diags(deg_w) - A.tocsr()

    k = min(k, L.shape[0] - 2)
    print(f"[laplacian] computing {k} smallest eigenvalues of size {L.shape[0]}...")

    vals, _ = eigsh(L, k=k, which="SM")
    vals = np.sort(np.real(vals))
    return vals


def write_summary(outdir, G, distinct_vals, freqs, deg_vals, deg_counts, lap_eigs):
    """
    Save a simple text summary of graph-level quantities.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_deg = np.mean([d for _, d in G.degree()])
    components = list(nx.connected_components(G))
    n_comp = len(components)
    comp_sizes = sorted([len(c) for c in components], reverse=True)

    try:
        clustering = nx.average_clustering(G)
    except Exception:
        clustering = float("nan")

    path = os.path.join(outdir, "graph_summary.txt")
    with open(path, "w") as f:
        f.write("Interlayer graph summary\n")
        f.write("========================\n\n")
        f.write(f"Nodes (sites): {n_nodes}\n")
        f.write(f"Edges (interlayer hops): {n_edges}\n")
        f.write(f"Average degree: {avg_deg:.4f}\n")
        f.write(f"Connected components: {n_comp}\n")
        f.write(f"Component sizes: {comp_sizes}\n")
        f.write(f"Average clustering (unweighted): {clustering:.6f}\n\n")

        f.write(f"Distinct |t_perp| (tol): {len(distinct_vals)}\n")
        if len(distinct_vals) > 0:
            f.write(f"  min |t| = {distinct_vals[0]:.6e}\n")
            f.write(f"  max |t| = {distinct_vals[-1]:.6e}\n\n")

        f.write("Degree histogram (degree: count):\n")
        for d, c in zip(deg_vals, deg_counts):
            f.write(f"  {int(d)}: {int(c)}\n")
        f.write("\n")

        if lap_eigs.size > 0:
            f.write("Smallest Laplacian eigenvalues:\n")
            for k, val in enumerate(lap_eigs):
                f.write(f"  λ_{k} = {val:.8e}\n")

    print(f"[summary] wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse interlayer hopping graph for a PBC tBLG approximant."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with interlayer hoppings (i, j, t_perp).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="graph_analysis",
        help="Directory to save analysis outputs.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Tolerance for grouping distinct |t_perp| values.",
    )
    parser.add_argument(
        "--n-eigs",
        type=int,
        default=20,
        help="Number of smallest Laplacian eigenvalues to compute (0 to skip).",
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[load] reading {args.input}")
    df = load_interlayer_csv(args.input)

    # ---- distinct t_perp values ----
    distinct_vals, freqs = compute_distinct_t(df, tol=args.tol)
    distinct_rows = []
    for idx, (tval, count) in enumerate(zip(distinct_vals, freqs)):
        distinct_rows.append([idx, tval, int(count)])

    df_distinct = pd.DataFrame(
        distinct_rows,
        columns=["index", "t_perp_distinct", "frequency"],
    )
    path_distinct = os.path.join(args.outdir, "distinct_t_perp.csv")
    df_distinct.to_csv(path_distinct, index=False, float_format="%.8e")
    print(f"[distinct] saved {len(df_distinct)} rows to {path_distinct}")

    # ---- build graph and adjacency ----
    G, A, N = build_graph(df)
    print(f"[graph] N_nodes={N}, N_edges={G.number_of_edges()}")

    # ---- degree histogram ----
    deg_vals, deg_counts = degree_histogram(A)
    df_deg = pd.DataFrame(
        {"degree": deg_vals.astype(int), "count": deg_counts.astype(int)}
    )
    path_deg = os.path.join(args.outdir, "degree_histogram.csv")
    df_deg.to_csv(path_deg, index=False)
    print(f"[degree] saved histogram to {path_deg}")

    # ---- Laplacian eigenvalues ----
    if args.n_eigs > 0:
        lap_eigs = laplacian_small_eigs(A.tocsr(), k=args.n_eigs)
        df_lap = pd.DataFrame(
            {"k": np.arange(len(lap_eigs), dtype=int), "lambda": lap_eigs}
        )
        path_lap = os.path.join(args.outdir, "laplacian_eigs.csv")
        df_lap.to_csv(path_lap, index=False, float_format="%.8e")
        print(f"[laplacian] saved {len(lap_eigs)} eigenvalues to {path_lap}")
    else:
        lap_eigs = np.array([])

    # ---- text summary ----
    write_summary(args.outdir, G, distinct_vals, freqs, deg_vals, deg_counts, lap_eigs)

    print("[done] interlayer graph analysis complete.")


if __name__ == "__main__":
    main()
