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
from py_tbl.analysis import build_interlayer_adjacency_from_csv
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

 
    # commensurate twist angle
    theta_comm = theta_comm_deg_from_mr(m, r)



    

    A, A_path = build_interlayer_adjacency_from_csv(
        config_path="py_tbl/configs/pbc.yaml",
        weighted=False,   # or True if you want |t_perp| as weights
    )



if __name__ == "__main__":
    main()
