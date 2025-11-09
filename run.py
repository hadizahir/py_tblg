# -*- coding: utf-8 -*-
"""
run.py — main driver. Calls clean registry-overlay plot (2D only).
"""

import os, argparse, numpy as np, pandas as pd
from .config import Config
from .io_utils import load_config_yaml, ensure_dir, save_states_csv
from .lattices import graphene_primitives, layer_lattices
from .geometry import moire_vectors_primitive, rhombus_polygon
from .builders import build_flake_H_sparse
from .spectra import eigs_in_window_sliced, ipr, edge_mask, edge_weight
from .wavefunctions import save_wavefunctions_npz, save_wavefunction_overlay_png_clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = ap.parse_args()
    cfg: Config = load_config_yaml(args.config) if args.config else Config()

    acc = cfg.tb.acc; dperp = cfg.tb.dperp; t = cfg.tb.t; tp = cfg.tb.tp
    t_intra = (-1.0 if cfg.tb.E_in_t else -t)
    a1_b, a2_b, A_b, B_b = graphene_primitives(acc)
    T1, T2, U = moire_vectors_primitive(a1_b, a2_b, cfg.build.m, cfg.build.r)
    save_dir = ensure_dir(cfg.paths.save_dir)

    for hp_range in cfg.build.hp_values:
        QSIGMA, QPI = (7.42, 3.15) if cfg.tb.interlayer_mode == "baseline" else (3.0, 1.3)
        r_xy_cut = hp_range * acc

        tag = f"r{cfg.build.r:02d}_m{cfg.build.m:02d}_theta{cfg.build.theta_target_deg:.2f}_hp{hp_range:.2f}"
        rows_meta, all_state_frames = [], []

        L1, L2, (a1_t, a2_t, A_t, B_t) = layer_lattices(
            a1_b, a2_b, A_b, B_b, cfg.build.theta_target_deg, cfg.tb.registration
        )

        for n_mult in cfg.build.n_list:
            H, XY_all, N1 = build_flake_H_sparse(
                n_mult,
                a1_b, a2_b, A_b, B_b, a1_t, a2_t, A_t, B_t,
                T1, T2,
                acc=acc, dperp=dperp, t=t, tp=tp,
                r_xy_cut=r_xy_cut, t_intra=t_intra,
                QSIGMA=QSIGMA, QPI=QPI, E_in_t=cfg.tb.E_in_t
            )

            # trim geometric edges
            poly = rhombus_polygon(np.array([0.0, 0.0]), T1, T2, n_mult)
            corners4 = poly[:-1]
            d_edge = 10.5 * acc
            mb = edge_mask(XY_all[:N1], corners4, d_edge)
            mt = edge_mask(XY_all[N1:], corners4, d_edge)
            keep = ~np.r_[mb, mt]
            H = H.tocsr()[keep][:, keep]
            XY_all = XY_all[keep]
            N1 = int((~mb).sum())

            # eigens
            E, V = eigs_in_window_sliced(
                H,
                cfg.window.E_window[0], cfg.window.E_window[1],
                cfg.window.sigmas, cfg.window.k_per_slice,
                cfg.window.n_states_target
            )

            N_sites = H.shape[0]
            N_in_gap = int(len(E))
            print(f"[n={n_mult}] N_sites={N_sites}, states_in_window={N_in_gap}")
            rows_meta.append([n_mult, N_sites, N_in_gap])

            if N_in_gap > 0:
                I = ipr(V)
                EW = edge_weight(V, XY_all, corners4, N1, 2.5 * acc)
                df = pd.DataFrame({"n": n_mult, "E": E, "IPR": I, "EdgeWeight": EW})
                fn_n = os.path.join(save_dir, f"flakes_states_{tag}_n{n_mult:02d}.csv")
                save_states_csv(fn_n, df)
                all_state_frames.append(df)

                # save NPZ and make the clean overlay for the first saved state
                npz_path = save_wavefunctions_npz(tag, n_mult, XY_all, N1, E, V, save_dir)
                if npz_path:
                    save_wavefunction_overlay_png_clean(
                        npz_path, state=0, a1_b=a1_b, a2_b=a2_b,
                        dx_mass=1.2, smooth_sigma=1.2, dot_size=5
                    )

        meta = pd.DataFrame(rows_meta, columns=["n", "N_sites", "N_states_in_window"])
        meta_csv = os.path.join(save_dir, f"flakes_meta_{tag}.csv")
        meta.to_csv(meta_csv, index=False)

        if all_state_frames:
            states = pd.concat(all_state_frames, ignore_index=True)
            states_csv = os.path.join(save_dir, f"flakes_states_{tag}.csv")
            states.to_csv(states_csv, index=False)
            print(f"[saved] {meta_csv}\n[saved] {states_csv}")
        else:
            print(f"[saved] {meta_csv} (no states csv)")

if __name__ == "__main__":
    main()
