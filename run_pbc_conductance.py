# -*- coding: utf-8 -*-
"""
run_pbc_conductance.py — conductance ladder for commensurate tBLG flakes.

For each (m, r) approximant:
  * build a finite twisted-bilayer scattering region: an open flake of
    n_mult × n_mult moiré supercells (no wrapping),
  * attach two semi-infinite monolayer graphene leads to the BOTTOM layer,
    with translational invariance along a1_b,
  * compute two-terminal Landauer conductance G(E_F),
  * save a ladder CSV for scaling analysis.

Transport direction: ALONG a1_b.
Contacts: planes PERPENDICULAR to a1_b at minimal / maximal i-tag (flake edges).
"""

import os
import argparse
import numpy as np
import numpy.linalg as npl
import kwant
from scipy.spatial import cKDTree
import pandas as pd

from .config import Config
from .io_utils import load_config_yaml, ensure_dir
from .lattices import graphene_primitives, layer_lattices
from .geometry import moire_vectors_primitive, theta_comm_deg_from_mr, inside_rhombus
from .builders import tperp_SK


# ----------------------------------------------------------------------
# tBLG scattering region: open-boundary flake of moiré cells
# ----------------------------------------------------------------------
def build_tblg_flake(
    m: int,
    r: int,
    n_mult: int,
    hp: float,
    acc: float,
    dperp: float,
    t: float,
    tp: float,
    E_in_t: bool,
    registration: str,
    interlayer_mode: str = "baseline",
    mu_center: float = 0.0,
):
    """
    Build a finite tBLG scattering region corresponding to an open flake of
    n_mult × n_mult moiré supercells (no PBC), without leads.

    Sites:
      * bottom layer = lattice "L1"
      * top layer    = lattice "L2"
    Only sites inside the rhombus spanned by (T1, T2) with 0 <= u,v < n_mult
    are kept, where (u,v) are moiré barycentric coordinates (handled by
    inside_rhombus).

    Returns
    -------
    syst : kwant.Builder
        Scattering region (no leads attached).
    L1, L2 : kwant.lattice.general
        Bottom and top layer lattices (both honeycomb).
    a1_b, a2_b : np.ndarray
        Graphene primitive vectors of the bottom layer.
    bottom_sites : list[kwant.Site]
        All bottom-layer sites in the scattering region.
    """
    # bottom graphene primitives
    a1_b, a2_b, A_b, B_b = graphene_primitives(acc)

    # layer lattices (bottom L1, top L2)
    theta_deg = theta_comm_deg_from_mr(m, r)
    L1, L2, (a1_t, a2_t, A_t, B_t) = layer_lattices(
        a1_b, a2_b, A_b, B_b,
        theta_deg=theta_deg,
        registration=registration,
    )
    A1, B1 = L1.sublattices
    A2, B2 = L2.sublattices

    # moiré primitive vectors and integer matrix U
    T1, T2, U = moire_vectors_primitive(a1_b, a2_b, m, r)
    origin = np.array([0.0, 0.0], float)

    # open-boundary flake: u, v ∈ [0, n_mult)
    def inside_flake(r_vec):
        return inside_rhombus(np.asarray(r_vec, float), origin, T1, T2, n_mult)

    # bounds in (i,j) that cover an n_mult × n_mult rhombus
    def seed_bounds_from_U(U_int, n_mult_local, margin=3):
        U_int = np.asarray(U_int, float)
        c0 = np.array([0, 0], float)
        c1 = n_mult_local * U_int[:, 0]
        c2 = n_mult_local * U_int[:, 1]
        c3 = n_mult_local * (U_int[:, 0] + U_int[:, 1])
        corners = np.stack([c0, c1, c2, c3], axis=0)
        i_min = int(np.floor(np.min(corners[:, 0])) - margin)
        i_max = int(np.ceil (np.max(corners[:, 0])) + margin)
        j_min = int(np.floor(np.min(corners[:, 1])) - margin)
        j_max = int(np.ceil (np.max(corners[:, 1])) + margin)
        return (i_min, i_max), (j_min, j_max)

    (i_min, i_max), (j_min, j_max) = seed_bounds_from_U(U, n_mult, margin=4)

    syst = kwant.Builder()
    t_intra = -1.0 if E_in_t else -t

    # --- bottom-layer sites inside the flake ---
    bottom_sites = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            for fam in (A1, B1):
                site = fam(i, j)
                if inside_flake(site.pos):
                    syst[site] = mu_center
                    bottom_sites.append(site)

    # --- top-layer sites inside the flake ---
    top_sites = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            for fam in (A2, B2):
                site = fam(i, j)
                if inside_flake(site.pos):
                    syst[site] = mu_center
                    top_sites.append(site)

    # Intralayer hoppings (nearest neighbours, standard honeycomb)
    syst[L1.neighbors(1)] = t_intra
    syst[L2.neighbors(1)] = t_intra

    # --- interlayer hoppings (Slater–Koster, with in-plane cutoff) ---
    if interlayer_mode == "baseline":
        QSIGMA, QPI = 7.42, 3.15
    else:
        QSIGMA, QPI = 3.0, 1.3

    r_xy_cut = hp * acc

    bottom_pos = np.array([np.asarray(s.pos, float) for s in bottom_sites])
    top_pos = np.array([np.asarray(s.pos, float) for s in top_sites])

    if tp != 0.0 and len(bottom_pos) > 0 and len(top_pos) > 0:
        tree_top = cKDTree(top_pos)

        for i_b, s_b in enumerate(bottom_sites):
            r_b = bottom_pos[i_b]
            neigh_inds = tree_top.query_ball_point(r_b, r_xy_cut)
            for j_t in neigh_inds:
                s_t = top_sites[j_t]
                r_t = top_pos[j_t]
                dx, dy = r_t - r_b
                r_xy = float(np.hypot(dx, dy))
                if r_xy < 0.0 or r_xy > r_xy_cut:
                    continue
                amp = tperp_SK(
                    r_xy,
                    acc=acc,
                    dperp=dperp,
                    t=t,
                    tp=tp,
                    E_in_t=E_in_t,
                    QSIGMA=QSIGMA,
                    QPI=QPI,
                )
                if amp != 0.0:
                    syst[s_b, s_t] = amp

    return syst, L1, L2, a1_b, a2_b, a1_t, a2_t, bottom_sites, top_sites


# ----------------------------------------------------------------------
# Build a monolayer lead from interface sites (bottom layer only)
# ----------------------------------------------------------------------
def make_interface_lead(L1, interface_sites, direction_vec, t_intra, mu_lead):
    """
    Create a monolayer graphene lead using the SAME lattice L1 as the bottom
    layer of the device. The cross section is defined by `interface_sites`.

    Kwant 1.x pattern:
      - lead = kwant.Builder(TranslationalSymmetry(direction_vec))
      - add same families/tags as interface sites with onsite = mu_lead
      - add NN hoppings: lead[L1.neighbors(1)] = t_intra
      - syst.attach_lead(lead)
    """
    direction_vec = np.asarray(direction_vec, float)
    sym = kwant.TranslationalSymmetry(direction_vec)
    lead = kwant.Builder(symmetry=sym)

    # add all interface sites as part of the lead unit cell
    for s in interface_sites:
        fam = s.family
        tag = s.tag
        lead[fam(*tag)] = mu_lead

    # intralayer hoppings in the lead (same as bottom layer)
    lead[L1.neighbors(1)] = t_intra

    return lead
# Option 2


def build_leads(layer1, layer2, A_b, B_b, A_t, B_t, bottom_sites, top_sites, acc, m, r, LeadConnectionType=11):
    symL = kwant.TranslationalSymmetry(layer1.vec((-1, 0)))
    symR = kwant.TranslationalSymmetry(layer1.vec((1, 0)))
    left_lead_from_layer1 = kwant.Builder(symL)
    right_lead_from_layer1 = kwant.Builder(symR)
    sym1 = kwant.TranslationalSymmetry(layer2.vec((1, 0)))
    left_lead_from_layer2 = kwant.Builder(sym1)
    right_lead_from_layer2 = kwant.Builder(sym1)
    i_vals = [s.tag[0] for s in bottom_sites]
    i_min = min(i_vals)
    i_max = max(i_vals)

    a1_b, a2_b, AtomA_b, AtomB_b = graphene_primitives(acc)
    # moiré primitive vectors and integer matrix U
    T1, T2, U = moire_vectors_primitive(a1_b, a2_b, m, r)
    # magnitude of a2_b
    a1_perp=np.array([a1_b[1], -a1_b[0]])
    print(f"a1_b.dot(a1_perp): {a1_b.dot(a1_perp)}")

   
    proj = np.dot(T2,a1_perp)/np.dot(a1_perp,a2_b)
    proj_T1 = np.dot(T1,a1_perp)/np.dot(a1_perp,a2_b)  # the solution of T=m*a1_b+n*a2_b for n (=proj_T1)
    right_bottom_corner = int(round(proj_T1))
    print(f"right_bottom_corner: {right_bottom_corner}")
    # projection of U2 onto a2_b
    print(f"T1: {T1}")
    print(f"T2: {T2}")
    print(f"a1_b: {a1_b}")
    print(f"a2_b: {a2_b}")

    costh= abs(np.dot(T2, a2_b)) / np.linalg.norm(T2)/np.linalg.norm(a2_b)
    print(f"costh: {costh/np.pi*180}")
    # round to nearest integer number of graphene-rows
    lead_width_j = int(round(proj))

    print("Lead width in j (rows):", lead_width_j)
    print(f"proj={proj}")

    if LeadConnectionType==11:  

        j_vals = [s.tag[1] for s in bottom_sites]
        j_min = min(j_vals)
        j_max = max(j_vals)
        i_vals = [s.tag[0] for s in bottom_sites]
        i_min = min(i_vals)
        i_max = max(i_vals)
        print(f"j_max: {j_max}")
        print(f"j_min: {j_min}")
        print(f"i_max: {i_max}")
        print(f"i_min: {i_min}")
    elif LeadConnectionType==12:
        j_vals = [s.tag[1] for s in bottom_sites]
        j_min = min(j_vals)
        j_max = max(j_vals)
    elif LeadConnectionType==21:
        left_lead, right_lead = left_lead_from_layer2.reversed(), left_lead_from_layer1
            
    else:
        raise NameError('Unknown type of lead connections. Please, choose 1 to 1 or 1 to 2 connextions')
   # if r%3==0 and r!=1:
    #    right_bottom_corner = right_bottom_corner+3

    for x in range(0,3):
        for y in range(j_min,j_min+np.abs(lead_width_j)+1):
            
            left_lead_from_layer1[A_b(x,y)]=-0.5
            left_lead_from_layer1[B_b(x,y)]=-0.5
            left_lead_from_layer2[A_t(x,y)]=-0.5
            left_lead_from_layer2[B_t(x,y)]=-0.5
    for x in range(0,3):
        for y in range(j_min+np.abs(right_bottom_corner),j_min+np.abs(right_bottom_corner) +np.abs(lead_width_j)):
            #print([x,y])
            right_lead_from_layer1[A_b(x,y)]=-0.5
            right_lead_from_layer1[B_b(x,y)]=-0.5
            right_lead_from_layer2[A_t(x,y)]=-0.5
            right_lead_from_layer2[B_t(x,y)]=-0.5

            
       
    left_lead_from_layer1[layer1.neighbors()]=-1 
    left_lead_from_layer2[layer2.neighbors()]=-1 
    right_lead_from_layer1[layer1.neighbors()]=-1 
    right_lead_from_layer2[layer2.neighbors()]=-1
    if LeadConnectionType==11:  
        left_lead, right_lead = left_lead_from_layer1, right_lead_from_layer1

    elif LeadConnectionType==12:
        left_lead, right_lead = left_lead_from_layer1, left_lead_from_layer2.reversed()
    elif LeadConnectionType==21:
        left_lead, right_lead = left_lead_from_layer2.reversed(), left_lead_from_layer1
            
    else:
        raise NameError('Unknown type of lead connections. Please, choose 1 to 1 or 1 to 2 connextions')
    
    return left_lead, right_lead

# ----------------------------------------------------------------------
# Conductance for a single approximant (flake geometry)
# ----------------------------------------------------------------------
def compute_conductance_for_approximant(
    cfg: Config,
    m: int,
    r: int,
    hp: float,
    approx_index: int | None = None,
    plot_first: bool = True,
):
    tb = cfg.tb
    acc = tb.acc
    dperp = tb.dperp
    t = tb.t
    tp = tb.tp
    E_in_t = tb.E_in_t
    registration = tb.registration
    interlayer_mode = tb.interlayer_mode

    # flake size (n_mult × n_mult)
    n_mult_list = getattr(cfg.build, "n_mult_list", [1])
    n_mult = int(n_mult_list[0])

    # window / lead knobs
    E_window = getattr(cfg.window, "E_window", (-0.02, 0.0))
    E_center = getattr(cfg.window, "E_center", 0.5 * (E_window[0] + E_window[1]))
    mu_lead = getattr(cfg.window, "mu_lead", -0.3)  # doping for metallic leads

    theta_deg = theta_comm_deg_from_mr(m, r)

    print()
    print("=" * 72)
    print(f"[cond] approximant (m={m}, r={r}), θ = {theta_deg:.6f}°, n_mult = {n_mult}")
    if approx_index is not None:
        print(f"[cond] ladder index = {approx_index}")
    print("=" * 72)

    # ---- build scattering region (flake) ----
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
    

    A1, B1 = L1.sublattices
    A2, B2 = L2.sublattices
    all_sites = list(syst.sites())
    N_sites = len(all_sites)
    N1 = len(bottom_sites)
    print(f"[cond] scattering region built (flake): N_sites = {N_sites}")
    print(f"[cond] bottom-layer sites (lead candidates): N1 = {N1}")

    if N1 == 0:
        raise RuntimeError("No bottom-layer sites found in scattering region.")


    left_lead, right_lead = build_leads(layer1=L1, layer2=L2, A_b=A1, B_b=B1, A_t=A2, B_t=B2, bottom_sites=bottom_sites, top_sites=top_sites,acc=acc, m=m, r=r, LeadConnectionType=11)
    left_lead.eradicate_dangling() 
    right_lead.eradicate_dangling() 
    syst.eradicate_dangling() 
    syst.attach_lead(left_lead)
    syst.attach_lead(right_lead)



    # ---- finalize and optional plotting ----
    fsyst = syst.finalized()
    print(f"[cond] finalized system: #dofs (device + one cell of each lead) = "
          f"{fsyst.graph.num_nodes}")

    if plot_first and (approx_index in (0, None)):
        try:
            print("[cond] plotting finalized system (flake + leads)...")
            site_size = 5.15
            hop_lw = 0.01

            


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


        except Exception as exc:
            print(f"[cond] kwant.plot failed: {exc!r}")

    # ---- Landauer conductance via S-matrix ----
    E_F = E_center
    print(f"[cond] computing S-matrix at Fermi energy E_F = {E_F:.6f} ...")
    smat = kwant.smatrix(fsyst, energy=E_F)

    T = smat.transmission(1, 0)  # from left (0) to right (1)
    G_e2_over_h = T              # spinless; ×2 for spin if needed

    print(f"[cond] transmission T = {T:.6f}  ->  G = {G_e2_over_h:.6f} (e^2/h, spinless)")

    return {
        "approx_index": approx_index if approx_index is not None else 0,
        "m": m,
        "r": r,
        "theta_deg": theta_deg,
        "n_mult": n_mult,
        "hp": hp,
        "N_sites": N_sites,
        "N_bottom": N1,
        "E_F": E_F,
        "mu_lead": mu_lead,
        "G_e2_over_h": G_e2_over_h,
    }


# ----------------------------------------------------------------------
# Ladder driver
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute tBLG conductance ladder for commensurate flake approximants."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (same style as run_pbc.py).",
    )
    args = parser.parse_args()

    cfg: Config = load_config_yaml(args.config) if args.config else Config()
    save_dir = ensure_dir(cfg.paths.save_dir)

    hp_values = getattr(cfg.build, "hp_values", [0.9])
    hp = float(hp_values[0])

    approx_list = getattr(cfg.build, "approximants", None)
    rows = []

    if approx_list is None:
        m_single = int(cfg.build.m)
        r_single = int(cfg.build.r)
        res = compute_conductance_for_approximant(
            cfg=cfg,
            m=m_single,
            r=r_single,
            hp=hp,
            approx_index=None,
            plot_first=True,
        )
        rows.append(res)
    else:
        for idx, entry in enumerate(approx_list):
            m = int(entry["m"])
            r = int(entry["r"])
            res = compute_conductance_for_approximant(
                cfg=cfg,
                m=m,
                r=r,
                hp=hp,
                approx_index=idx,
                plot_first=True,
            )
            rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(save_dir, "pbc_conductance_ladder.csv")
    df.to_csv(out_csv, index=False, float_format="%.8e")
    print(f"[cond] ladder CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
