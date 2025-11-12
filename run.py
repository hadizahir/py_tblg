# -*- coding: utf-8 -*-
"""
run.py — main driver with registry-aware overlays + numeric fractions (per-state sweep).
"""
from .level_stats import post_compute_levelstats_from_states_csv, derive_tag_from_meta
import os, argparse, numpy as np, pandas as pd
from .config import Config
from .io_utils import load_config_yaml, ensure_dir, save_states_csv
from .lattices import graphene_primitives, layer_lattices
from .geometry import moire_vectors_primitive, rhombus_polygon
from .builders import build_flake_H_sparse
from .spectra import eigs_in_window_sliced, ipr, edge_weight, edge_mask
from .wavefunctions import (
    save_wavefunctions_npz,
    save_wavefunction_overlay_registry_png_clean,
    save_wavefunction_3d_surface_html_clean,
)

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
        QSIGMA, QPI = (7.42, 3.15) if (cfg.tb.interlayer_mode == "baseline") else (3.0, 1.3)
        r_xy_cut = hp_range * acc
        tag = f"r{cfg.build.r:02d}_m{cfg.build.m:02d}_theta{cfg.build.theta_target_deg:.2f}_hp{hp_range:.2f}"

        rows_meta, all_state_frames, frac_rows = [], [], []

        # top lattice
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

            # geometric edge trim
            poly = rhombus_polygon(np.array([0.0, 0.0]), T1, T2, n_mult)
            corners4 = poly[:-1]
            d_edge = 0 * acc
            mask_b = edge_mask(XY_all[:N1], corners4, d_edge)
            mask_t = edge_mask(XY_all[N1:], corners4, d_edge)
            keep = ~np.r_[mask_b, mask_t]
            H = H.tocsr()[keep][:, keep]
            XY_all = XY_all[keep]
            N1 = int((~mask_b).sum())

            # eigs-in-window
            E, V = eigs_in_window_sliced(
                H, cfg.window.E_window[0], cfg.window.E_window[1],
                cfg.window.sigmas, cfg.window.k_per_slice, cfg.window.n_states_target
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

                # save minimal NPZ so we can reuse P and E easily
                npz_path = save_wavefunctions_npz(tag, n_mult, XY_all, N1, E, V, save_dir)

                # ---- Per-state sweep: overlays + fractions for every in-window state ----
                data = np.load(npz_path)
                XY = data["XY"]; N1np = int(data["N1"])
                Elist = data["E"]; P = data["P"]

                for s in range(len(Elist)):
                    base = os.path.splitext(os.path.basename(npz_path))[0]
                    png_out_top = os.path.join(save_dir, f"{base}_state{s:02d}_E{E[s]:02f}_registry_overlay_top.png")
                    png_out_bottom = os.path.join(save_dir, f"{base}_state{s:02d}_E{E[s]:02f}_registry_overlay_bottom.png")

                    # Clean 2D overlay (lines only) + fractions
                    png_path, fracs, masks_tuple = save_wavefunction_overlay_registry_png_clean(
                        XY, N1np, Elist, P, a1_b, a2_b,
                        state=s,
                        dx_reg=5,        # registry pixel (Å)
                        aa_frac=1,      # AA threshold (fraction of max |Φ|)
                        wall_q=0.95,       # walls quantile on ∇(phase) — kept for consistency
                        m_sign_thr=0.1,   # AB/BA split with cos(phase)
                        out_png=png_out_top,
                        layer="top"
                    )
                    png_path, fracs, masks_tuple = save_wavefunction_overlay_registry_png_clean(
                        XY, N1np, Elist, P, a1_b, a2_b,
                        state=s,
                        dx_reg=5,        # registry pixel (Å)
                        aa_frac=1,      # AA threshold (fraction of max |Φ|)
                        wall_q=0.95,       # walls quantile on ∇(phase) — kept for consistency
                        m_sign_thr=0.1,   # AB/BA split with cos(phase)
                        out_png=png_out_bottom,
                        layer="bottom"
                    )
                    frac_rows.append({
                        "n": n_mult, "state": s,
                        "E": float(Elist[s]),
                        "f_AA": fracs["AA"], "f_AB": fracs["AB"],
                        "f_BA": fracs["BA"], "f_WALL": fracs["WALL"]
                    })

                    # Optional: 3D surface with same closed contours
                    save_wavefunction_3d_surface_html_clean(
                        XY, N1np, Elist, P, masks_tuple, state=s,
                    )

        # ---- Save meta and fractions ----
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

        if frac_rows:
            frac_df = pd.DataFrame(frac_rows)
            frac_csv = os.path.join(save_dir, f"flakes_state_fractions_{tag}.csv")
            frac_df.to_csv(frac_csv, index=False)
            print(f"[saved] {frac_csv}")

if __name__ == "__main__":
    # Keep original behavior
    main()

    # Post: compute level statistics from the states CSV we just saved
    try:
        # Re-read config like your main() does (non-invasive)
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None)
        args, _ = parser.parse_known_args()
        cfg = load_config_yaml(args.config) if args.config else Config.defaults()
        save_dir = cfg.paths.save_dir

        # Infer the tag from the latest meta_*.csv (provided by level_stats.py)
        tag = derive_tag_from_meta(save_dir)

        # Do the stats (helpers live in level_stats.py)
        post_compute_levelstats_from_states_csv(
            save_dir=save_dir,
            tag=tag,        # None ⇒ auto-picks latest flakes_states_*.csv
            lmax=20.0,
            nL=30,
            fit_Lmin=2.0,
            fit_Lmax=10.0
        )
    except Exception as e:
        # Never break your main pipeline on post-processing
        print(f"[level-stats] skipped due to: {e}")

