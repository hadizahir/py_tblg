# -*- coding: utf-8 -*-
"""
run_pbc_analysis.py — post-processing of PBC outputs

This script:
  - loads the interlayer_hoppings_{tag}.csv files
  - builds adjacency matrices
  - saves adjacency .npz files

It assumes that run_pbc.py has ALREADY been executed and all CSV files exist.
"""

import os
import argparse

import numpy as np
import pandas as pd

from .config import Config
from .io_utils import load_config_yaml, ensure_dir
from .io_utils import build_interlayer_adjacency_from_csv


def run_analysis(cfg: Config):
    """
    Perform adjacency analysis using previously saved interlayer hopping CSV.
    """

    print("\n=== PBC adjacency analysis ===\n")

    # build adjacency matrix (weighted or unweighted)
    A, A_path = build_interlayer_adjacency_from_csv(
        config_path="py_tbl/configs/pbc.yaml",
        weighted=False,
    )

    print(f"[analysis] adjacency matrix saved to: {A_path}")
    print(f"[analysis] shape = {A.shape}")

    return A


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="py_tbl/configs/pbc.yaml",
        help="Path to PBC config used in the original run."
    )
    args = parser.parse_args()

    # load config
    cfg = load_config_yaml(args.config)

    # run analysis
    run_analysis(cfg)


if __name__ == "__main__":
    main()
