# -*- coding: utf-8 -*-
"""
laplacian_plot_utils.py

Plotting utilities for Laplacian–TB overlap analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_overlap_heatmap(df: pd.DataFrame, tag: str, out_dir: str) -> str:
    """
    Heatmap: Laplacian mode index (ℓ) vs TB energy (E_n), color = |<ψ_n | φ_ℓ>|^2.
    Saves PNG and returns its path.
    """
    # pivot: rows=mode_idx, cols=state_idx, values=weight
    pivot = df.pivot_table(
        values="weight",
        index="mode_idx",
        columns="state_idx",
        aggfunc=np.sum,
    ).fillna(0.0)

    # TB energies sorted by state_idx
    E_tb = (
        df[["state_idx", "E_tb"]]
        .drop_duplicates("state_idx")
        .sort_values("state_idx")["E_tb"]
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[E_tb.min(), E_tb.max(), 0, pivot.shape[0]],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\langle \psi_n \,|\, \phi_\ell \rangle|^2$")

    ax.set_xlabel(r"TB energy $E_n$ (near secondary gap)")
    ax.set_ylabel(r"Laplacian mode index $\ell$")
    ax.set_title(f"Laplacian → TB overlap map ({tag})")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"laplacian_tb_overlap_heatmap_{tag}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_energy_vs_lambda(df: pd.DataFrame, tag: str, out_dir: str) -> str:
    """
    Scatter plot: for each TB state, show its dominant Laplacian λ (by weight).

    x-axis: TB energy E_n
    y-axis: Laplacian eigenvalue λ_ell of the most contributing mode for that state
    color: corresponding max weight
    """
    # Pick, for each TB state_idx, the row with max weight
    dfn = (
        df.sort_values("weight", ascending=False)
        .groupby("state_idx")
        .head(1)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(
        dfn["E_tb"],
        dfn["lambda_lap"],
        c=dfn["weight"],
        cmap="viridis",
        s=30,
        edgecolors="none",
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\max_\ell |\langle \psi_n \,|\, \phi_\ell \rangle|^2$")

    ax.set_xlabel(r"TB energy $E_n$")
    ax.set_ylabel(r"Laplacian eigenvalue $\lambda_\ell$")
    ax.set_title(f"Dominant Laplacian mode per TB state ({tag})")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"laplacian_tb_energy_vs_lambda_{tag}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_laplacian_mode_strength(df: pd.DataFrame, tag: str, out_dir: str) -> str:
    """
    Scatter plot: for each Laplacian mode ℓ, show λ_ℓ vs max_n |<ψ_n | φ_ℓ>|^2.

    x-axis: λ_ℓ
    y-axis: max weight over TB states
    """
    dfl = (
        df.sort_values("weight", ascending=False)
        .groupby("mode_idx")
        .head(1)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        dfl["lambda_lap"],
        dfl["weight"],
        s=30,
        edgecolors="none",
    )
    ax.set_xlabel(r"Laplacian eigenvalue $\lambda_\ell$")
    ax.set_ylabel(r"$\max_n |\langle \psi_n \,|\, \phi_\ell \rangle|^2$")
    ax.set_title(f"Laplacian mode importance ({tag})")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"laplacian_tb_mode_strength_{tag}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
