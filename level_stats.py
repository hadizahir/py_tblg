# py_tbl/level_stats.py
# -*- coding: utf-8 -*-
"""
Level statistics helpers used by run.py

Provides:
  - derive_tag_from_meta(save_dir) -> str|None
  - post_compute_levelstats_from_states_csv(save_dir, tag=None, lmax=20.0, nL=30, fit_Lmin=2.0, fit_Lmax=10.0) -> str|None

What it does:
  Reads the states CSV your pipeline already writes (flakes_states_{tag}.csv),
  groups rows by 'n' (system size) if available, and for each group computes:
    * adjacent-gap ratio <r>  (unfolding-free)
    * number variance Σ²(L) and its slope χ on [fit_Lmin, fit_Lmax]
  Then it writes:
    * levelstats_{tag}.csv            (summary per n)
    * levelvar_{tag}_nXX.csv          (Σ²(L) vs L curve per n)
"""

from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd

# ------------------------ small internal helpers ------------------------ #

def _adj_gap_ratio(E: np.ndarray):
    """Adjacent-gap ratio <r> on sorted energies; unfolding-free."""
    E = np.sort(np.asarray(E, dtype=float))
    s = np.diff(E)
    if s.size < 2:
        return float("nan"), np.array([])
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.nanmean(r)), r

def _simple_unfold(E: np.ndarray):
    """Minimal unfolding: scale spacings to unit mean and integrate."""
    E = np.sort(np.asarray(E, dtype=float))
    s = np.diff(E)
    if s.size == 0:
        return np.array([])
    ms = s.mean()
    if not np.isfinite(ms) or ms <= 0:
        return np.array([])
    s1 = s / ms
    return np.concatenate([[0.0], np.cumsum(s1)])

def _number_variance(x: np.ndarray, L_values: np.ndarray, n_grid: int = 256):
    """Σ²(L) via sliding windows on unfolded positions x, using periodic wrap."""
    x = np.asarray(x, float)
    L_values = np.asarray(L_values, float)
    if x.size < 2:
        return np.full_like(L_values, np.nan, dtype=float)

    span = x[-1] - x[0]
    if not np.isfinite(span) or span <= 0:
        return np.full_like(L_values, np.nan, dtype=float)

    # periodic tiling
    x0 = x - x[0]
    x_ext = np.concatenate([x0, x0 + span, x0 + 2 * span])
    starts = np.linspace(span * 0.5, span * 1.5, int(n_grid), endpoint=False)

    out = []
    for L in L_values:
        counts = []
        for s0 in starts:
            s1 = s0 + L
            c = np.searchsorted(x_ext, s1, side="right") - np.searchsorted(x_ext, s0, side="right")
            counts.append(c)
        counts = np.asarray(counts, float)
        out.append(counts.var(ddof=1))
    return np.asarray(out, float)

def _fit_chi(L: np.ndarray, Sigma2: np.ndarray, Lmin: float, Lmax: float):
    """Linear fit Σ²(L) ≈ χ L + b in [Lmin, Lmax]. Returns (χ, b)."""
    L = np.asarray(L, float)
    S = np.asarray(Sigma2, float)
    mask = np.isfinite(L) & np.isfinite(S) & (L >= Lmin) & (L <= Lmax)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    A = np.vstack([L[mask], np.ones(mask.sum())]).T
    chi, b = np.linalg.lstsq(A, S[mask], rcond=None)[0]
    return float(chi), float(b)

def _compute_level_stats_on_E(
    E: np.ndarray,
    Lmax: float = 20.0,
    nL: int = 30,
    fit_Lmin: float = 2.0,
    fit_Lmax: float = 10.0,
):
    """Compute <r>, Σ²(L) and χ for a 1D energy list E."""
    r_mean, _ = _adj_gap_ratio(E)
    x = _simple_unfold(E)
    if x.size < 10:
        return {"r_mean": r_mean, "L_values": np.array([]), "Sigma2": np.array([]),
                "chi": float("nan"), "intercept": float("nan")}
    L_values = np.linspace(1.0, Lmax, int(nL))
    Sigma2 = _number_variance(x, L_values)
    chi, b = _fit_chi(L_values, Sigma2, Lmin=fit_Lmin, Lmax=fit_Lmax)
    return {"r_mean": r_mean, "L_values": L_values, "Sigma2": Sigma2, "chi": chi, "intercept": b}

# -------------------------- public API (imported by run.py) -------------------------- #

def derive_tag_from_meta(save_dir: str) -> str | None:
    """
    Try to infer the most recent tag from meta_{tag}.csv in save_dir.
    Returns tag string or None if not found.
    """
    metas = sorted(glob.glob(os.path.join(save_dir, "meta_*.csv")), key=os.path.getmtime)
    if not metas:
        return None
    base = os.path.basename(metas[-1])
    m = re.match(r"meta_(.+)\.csv$", base)
    return m.group(1) if m else None

def post_compute_levelstats_from_states_csv(
    save_dir: str,
    tag: str | None = None,
    lmax: float = 20.0,
    nL: int = 30,
    fit_Lmin: float = 2.0,
    fit_Lmax: float = 10.0,
) -> str | None:
    """
    Post-process the states CSV produced by your pipeline:
      - Chooses flakes_states_{tag}.csv (if tag None, picks latest by mtime)
      - Groups rows by 'n' (if present), else treats all rows as one group
      - Computes per-group:
          r_mean (adjacent-gap ratio),
          Sigma2(L) curve,
          chi (slope on [fit_Lmin, fit_Lmax])
      - Writes:
          levelstats_{tag}.csv
          levelvar_{tag}_nXX.csv  for each n
    Returns the path to the summary CSV or None.
    """
    # pick states CSV
    if tag is None:
        candidates = sorted(glob.glob(os.path.join(save_dir, "flakes_states_*.csv")), key=os.path.getmtime)
        if not candidates:
            print("[level-stats] No flakes_states_*.csv found; skipping.")
            return None
        states_csv = candidates[-1]
        base = os.path.basename(states_csv)
        m = re.match(r"flakes_states_(.+)\.csv$", base)
        tag = m.group(1) if m else "latest"
    else:
        states_csv = os.path.join(save_dir, f"flakes_states_{tag}.csv")
        if not os.path.exists(states_csv):
            print(f"[level-stats] states CSV not found for tag={tag}; skipping.")
            return None

    print(f"[level-stats] Using states file: {states_csv}")
    df = pd.read_csv(states_csv)
    if df.empty or "E" not in df.columns:
        print("[level-stats] States CSV has no energies; skipping.")
        return None

    # group by n if available
    if "n" in df.columns:
        groups = df.groupby("n")
    else:
        df = df.copy()
        df["n"] = -1
        groups = df.groupby("n")

    rows = []
    for n_mult, g in groups:
        E = g["E"].values.astype(float)
        if E.size < 5:
            continue
        stats = _compute_level_stats_on_E(E, Lmax=lmax, nL=nL, fit_Lmin=fit_Lmin, fit_Lmax=fit_Lmax)

        # save Σ²(L) curve for this n
        if stats["L_values"].size:
            curve = pd.DataFrame({"n": int(n_mult), "L": stats["L_values"], "Sigma2": stats["Sigma2"]})
            fn_curve = os.path.join(save_dir, f"levelvar_{tag}_n{int(n_mult):02d}.csv")
            curve.to_csv(fn_curve, index=False)
            print(f"[saved] {fn_curve}")

        rows.append({
            "n": int(n_mult),
            "N_in_window": int(E.size),
            "r_mean": float(stats["r_mean"]),
            "chi": float(stats["chi"]),
            "chi_fit_Lmin": float(fit_Lmin),
            "chi_fit_Lmax": float(fit_Lmax),
        })

    if not rows:
        print("[level-stats] No valid groups; nothing written.")
        return None

    out = pd.DataFrame(rows).sort_values("n")
    out_csv = os.path.join(save_dir, f"levelstats_{tag}.csv")
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    return out_csv
