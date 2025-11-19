# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:55:38 2025

@author: HOL1BRG
"""

import os, json, yaml, pandas as pd
from .config import Config
from .geometry import theta_comm_deg_from_mr
import numpy as np
import scipy.sparse as ss
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True); return path

def save_states_csv(path, df):
    ensure_dir(os.path.dirname(path)); df.to_csv(path, index=False, float_format="%.8e")

def load_config_yaml(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # light mapper
    from dataclasses import asdict
    cfg = Config()
    # shallow update
    for section, values in raw.items():
        obj = getattr(cfg, section)
        for k,v in values.items(): setattr(obj, k, v)
    return cfg


def save_interlayer_hoppings_csv(
    XY_all,
    N1,
    H_sparse,
    save_dir,
    tag,
    r_xy_cut=None,
):
    
    import numpy as np
    import pandas as pd
    import os
    """
    Extract and save interlayer hoppings from the PBC Hamiltonian.

    Parameters
    ----------
    XY_all : (N,2)
        Coordinates of all sites, bottom first then top.
    N1 : int
        Number of bottom-layer sites.
    H_sparse : csr_matrix
        The Γ-point Hamiltonian returned by build_pbc_H_gamma_from_kwant.
    save_dir : str
        Where the CSV will be written.
    tag : str
        Identifying string, e.g. 'r01_m18_theta1.80_hp0.90_PBC'.
    r_xy_cut : float or None
        Optional constraint: only save hoppings whose geometric distance
        is below r_xy_cut (useful to exclude artifacts).
    """

    H = H_sparse.tocoo()
    rows = H.row
    cols = H.col
    vals = H.data

    XY = XY_all

    inter_rows = []
    for i, j, tval in zip(rows, cols, vals):

        # Only keep i < j to avoid duplicates
        if i >= j:
            continue

        # Determine whether this is interlayer
        bottom_top = (i < N1 and j >= N1)
        top_bottom = (i >= N1 and j < N1)

        if not (bottom_top or top_bottom):
            continue

        # reorder so i_bottom <-> j_top
        if i < N1:
            ib, it = i, j
        else:
            ib, it = j, i

        rb = XY[ib]
        rt = XY[it]

        dx, dy = rt - rb
        dist = float(np.hypot(dx, dy))

        # Optional distance filter
        if r_xy_cut is not None and dist > r_xy_cut:
            continue

       # inter_rows.append([rb[0], rb[1], rt[0], rt[1], dist, float(tval)]) # uncomment if you want to save the interlayer hopping based on coordinates
        inter_rows.append([ib, it, dist, float(tval)])

    # -------------------------------------------
    # Compute distinct t_perp values (tol=0.001)
    # -------------------------------------------
    tol = 2e-5
    inter_rows.sort(key=lambda x: x[-1])
    # extract hopping values
    tvals = [row[-1] for row in inter_rows]

    # sort
    tvals_sorted = sorted(tvals)

    # build distinct list
    distinct_vals = []
    freqs = []

    for t in tvals_sorted:
        if not distinct_vals:
            distinct_vals.append(t)
            freqs.append(1)
        else:
            # compare to last distinct value
            if abs(t - distinct_vals[-1]) < tol:
                freqs[-1] += 1
            else:
                distinct_vals.append(t)
                freqs.append(1)

    # save distinct values
    distinct_rows = []
    for idx, (tval, count) in enumerate(zip(distinct_vals, freqs)):
        distinct_rows.append([idx, tval, count])
    distinct_rows.sort(key=lambda x: x[1]),
    # --- REASSIGN INDEX (first column) ---
    for i, row in enumerate(distinct_rows):
        row[0] = i
    df_distinct = pd.DataFrame(
        distinct_rows,
        columns=["index", "t_perp_distinct", "count"]
    )

    out_csv_distinct = os.path.join(
        save_dir, f"interlayer_hoppings_distinct_{tag}.csv"
    )
    df_distinct.to_csv(out_csv_distinct, index=False, float_format="%.8e")

    print(f"[saved distinct interlayer hoppings] {out_csv_distinct}")
    print(f"[distinct count] {len(distinct_vals)} values with tol={tol}")







    if not inter_rows:
        print("[interlayer] No interlayer hoppings found.")
        return None

   # df = pd.DataFrame(
   #     inter_rows,
   #     columns=["x_bottom", "y_bottom", "x_top", "y_top", "distance", "t_perp"],
   # )


    df = pd.DataFrame(
    inter_rows,
    columns=["i_bottom", "i_top", "distance", "t_perp"],
    )


    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"interlayer_hoppings_ij_tag_{tag}.csv")
    df.to_csv(out_csv, index=False, float_format="%.8e")

    print(f"[saved interlayer hoppings] {out_csv} ({len(df)} entries)")

    return out_csv



def build_interlayer_adjacency_from_csv(
    config_path: str = "py_tbl/configs/pbc.yaml",
    weighted: bool = False,
):
    """
    Read the saved interlayer_hoppings_{tag}.csv from cfg.paths.save_dir,
    build the interlayer adjacency matrix, and save it as a sparse .npz file.

    Parameters
    ----------
    config_path : str
        Path to pbc.yaml (or any config with build.m, build.r, build.hp_values, paths.save_dir).
    weighted : bool
        If False: adjacency A[i,j] = 1 for each interlayer link.
        If True : adjacency A[i,j] = |t_perp(i,j)| (symmetric).

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Interlayer adjacency matrix (N_sites x N_sites).
    out_path : str
        File path where the adjacency matrix was saved (.npz).
    """

    # --- load config and reconstruct tag ---
    cfg: Config = load_config_yaml(config_path)
    save_dir = ensure_dir(cfg.paths.save_dir)

    m  = cfg.build.m
    r  = cfg.build.r
    hp = cfg.build.hp_values[0]
    theta_comm = theta_comm_deg_from_mr(m, r)

    tag = f"r{r:02d}_m{m:02d}_theta{theta_comm:.2f}_hp{hp:.2f}_PBC"

    csv_path = os.path.join(save_dir, f"interlayer_hoppings_ij_tag_{tag}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"interlayer hoppings CSV not found: {csv_path}")

    print(f"[adjacency] reading interlayer hoppings from {csv_path}")

    df = pd.read_csv(csv_path)

    # expect integer indices
    for col in ("i_bottom", "i_top", "t_perp"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

    i_b = df["i_bottom"].to_numpy(dtype=int)
    i_t = df["i_top"].to_numpy(dtype=int)
    t   = df["t_perp"].to_numpy(dtype=float)

    # infer total number of sites
    N_sites = int(max(i_b.max(), i_t.max()) + 1)

    if weighted:
        w = np.abs(t)
    else:
        w = np.ones_like(t, dtype=float)

    # build symmetric adjacency
    rows = np.concatenate([i_b, i_t])
    cols = np.concatenate([i_t, i_b])
    vals = np.concatenate([w,   w   ])

    A = ss.coo_matrix((vals, (rows, cols)), shape=(N_sites, N_sites)).tocsr()

    out_path = os.path.join(save_dir, f"interlayer_adjacency_{tag}.npz")
    ss.save_npz(out_path, A)

    print(f"[adjacency] built interlayer adjacency (N={N_sites}) and saved to {out_path}")

    return A, out_path

