#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append conductance G(E) as a new column to flakes_states_{tag}_PBC.csv.

This uses EXACTLY the same flake + lead machinery as run_pbc_conductance.py:
    - build_tblg_flake
    - build_leads
    - Kwant S-matrix

This script handles files named like:
    flakes_states_r03_m54_theta1.79_hp0.90_PBC.csv
"""

import os
import numpy as np
import pandas as pd
import kwant

# --- import EXACT functions ---
from py_tbl.run_pbc_conductance import (
    build_tblg_flake,
    build_leads,
)
from py_tbl.io_utils import load_config_yaml
from py_tbl.lattices import graphene_primitives
from py_tbl.geometry import moire_vectors_primitive


# ============================================================
# USER INPUT
# ============================================================
config_path = "py_tbl/configs/pbc.yaml"

# Folder containing states files
states_dir = r"C:\Users\hol1brg\OneDrive - Bosch Group\DAILY\tBLG\bands_pbc"


# ============================================================
# Compute G(E)
# ============================================================
def compute_G(fsyst, E):
    try:
        sm = kwant.smatrix(fsyst, energy=E)
        n_modes_left = sm.num_propagating(0)    # lead index 0

        return sm.transmission(1, 0), n_modes_left
    except Exception:
        return 0.0


# ============================================================
# Load config
# ============================================================
cfg = load_config_yaml(config_path)

tb = cfg.tb
acc = tb.acc
dperp = tb.dperp
t = tb.t
tp = tb.tp
E_in_t = tb.E_in_t
registration = tb.registration
interlayer_mode = tb.interlayer_mode

hp = cfg.build.hp_values[0]
#=========================Process Bar========================
import sys
def progress_bar(iteration, total, bar_length=20):
    """
    Print an updating progress bar like:
      [#####-------] 43.2%
    """
    frac = iteration / total
    filled = int(bar_length * frac)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = frac * 100
    sys.stdout.write(f"\r    [{bar}] {percent:5.1f}%")
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")

# ============================================================
# Process each PBC state file
# ============================================================
for fn in sorted(os.listdir(states_dir)):

    if not fn.startswith("flakes_states_"):
        continue
    if not fn.endswith("_PBC.csv"):
        continue

    full = os.path.join(states_dir, fn)
    print(f"\n=== Processing: {fn} ===")

    # -------------------------------
    # Parse m, r from filename
    # -------------------------------
    # Example filename:
    #   flakes_states_r03_m54_theta1.79_hp0.90_PBC.csv

    parts = fn.split("_")
    r_str = parts[2]     # e.g., r03
    m_str = parts[3]     # e.g., m54

    r = int(r_str.replace("r", ""))
    m = int(m_str.replace("m", ""))
    if r not in [11]:#[1, 3,4,5,6,7,8,9,12,15]:
        continue
    print(f"  → detected m = {m}, r = {r}")

    # Load CSV
    df = pd.read_csv(full)

    # -------------------------------
    # Build single scattering flake
    # -------------------------------
    print("  Building scattering region...")

    # Graphene geometry for leads
    a1_b, a2_b, A_b, B_b = graphene_primitives(acc)
    T1, T2, U = moire_vectors_primitive(a1_b, a2_b, m, r)

    # Use default n_mult from config
    n_mult = cfg.build.n_mult_list[0]

    syst, L1, L2, a1_b, a2_b, a1_t, a2_t, bottom_sites, top_sites = build_tblg_flake(
        m=m,
        r=r,
        n_mult=n_mult,
        hp=hp,
        acc=acc,
        dperp=dperp,
        t=t,
        tp=tp,
        E_in_t=E_in_t,
        registration=registration,
        interlayer_mode=interlayer_mode,
        mu_center=0.0,
    )

    # Build leads
    A1, B1 = L1.sublattices
    A2, B2 = L2.sublattices

    left_lead, right_lead = build_leads(
        layer1=L1, layer2=L2,
        A_b=A1, B_b=B1,
        A_t=A2, B_t=B2,
        bottom_sites=bottom_sites,
        top_sites=top_sites,
        acc=acc,
        m=m,
        r=r,
        LeadConnectionType=11,
    )
    
    left_lead.eradicate_dangling()
    right_lead.eradicate_dangling()
    #syst.eradicate_dangling()
    

    syst.attach_lead(left_lead)
    syst.attach_lead(right_lead)

    fsyst = syst.finalized()
    import matplotlib.pyplot as plt


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define colors for sites: bottom layer = blue, top layer = red
    site_colors = []
    for site in fsyst.sites:
        if site.family == L1.sublattices[0] or site.family == L1.sublattices[1]:
            site_colors.append("blue")  # bottom layer
        else:
            site_colors.append("red")   # top layer

    # Plot the system with custom colors
    '''
    kwant.plot(
        fsyst,
        site_size=5.3,
        hop_lw=0.5,
        num_lead_cells=2,
        ax=ax,
        site_color=site_colors
    )
    

    # Save as PDF
    plt.savefig(f"tblg_used4conductance_m={m}_r={r}_leadtype={11}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    # Print confirmation
    print(f"[cond] System with m={m} and r={r} has been plotted and saved as 'tblg_layers.pdf'.")
    '''
    # -------------------------------
    # Compute G(E) for each row
    # -------------------------------
    print("  Computing G(E)...")
    G_list = []
    energies=np.linspace(-0.05, 0.05, 200)
    N_modes = []
    Trans_list = []

    i=1
    for E in energies:  #df["E"]:
        G, n_modes_left = compute_G(fsyst, float(E))
        G_list.append(G)
        N_modes.append(n_modes_left)
        Trans_list.append(G/n_modes_left)
        progress_bar(i, len(energies))
        i=i+1
    #df["G"] = G_list

    # -------------------------------
    # Save file
    # -------------------------------

    filename = f"tblg_conductance_m={m}_r={r}_leadtype={11}.csv"
    cond_dir = r"C:\Users\hol1brg\OneDrive - Bosch Group\DAILY\tBLG\conductance"
    # Full path
    out_path = os.path.join(cond_dir, filename)
    #df.to_csv(full, index=False, float_format="%.8e")
    df2 = pd.DataFrame({
        "E": energies,
        "G": G_list,
        "T": Trans_list,
        "N_modes": N_modes})

    df2.to_csv(out_path, index=False)
    print(f"  [updated] added G column to: {fn}")


print("\n---------------------------------------")
print("DONE — all flakes_states_*_PBC.csv files updated with conductance.")
print("---------------------------------------")
