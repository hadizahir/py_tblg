# -*- coding: utf-8 -*-
"""
kwant_bands.py — bandstructure for commensurate tBLG using kwant.

This is a functional refactor of your standalone kwant script,
reusing helpers from the py_tbl package where possible.
"""

def compute_tblg_bands_kwant(
    m,
    r,
    hp,
    acc,
    dperp,
    t,
    tp,
    E_in_t,
    interlayer_mode="baseline",
    registration="AB",
    k_counts=(50, 50, 50, 50),
    save_dir=None,
    csv_tag=None,
    do_plot=True,
):
    """
    Compute tBLG bandstructure for a commensurate (m,r) using the kwant-based
    supercell + tagged-hopping construction from your standalone script.

    Reuses:
      - graphene_primitives, layer_lattices        → py_tbl.lattices
      - moire_vectors_primitive, expected_sites... → py_tbl.geometry
      - tperp_SK                                   → py_tbl.builders

    Parameters
    ----------
    m, r : int
        Commensurate integers defining the twist.
    hp : float
        In-plane cutoff in units of acc (hp * acc is r_xy_cut).
    acc, dperp : float
        Carbon-carbon distance and interlayer spacing.
    t, tp : float
        Intralayer NN hopping and interlayer SK prefactor.
    E_in_t : bool
        If True, energies are returned in units of t and t_intra = -1.
    interlayer_mode : {"baseline","stronger"}
        Selects QSIGMA, QPI.
    registration : {"AB","AA"}
        Relative shift of top layer.
    k_counts : sequence of 4 ints
        Number of k points on each K–K′, K′–Γ, Γ–M, M–K segment.
    save_dir : str or None
        If not None, save a CSV with bands to this directory.
    csv_tag : str or None
        Extra tag to put into the CSV filename. If None, uses m,r,hp.
    do_plot : bool
        If True, make an interactive Plotly figure.

    Returns
    -------
    KPATH : (Nk,2) ndarray
        k-points along the high-symmetry path.
    E_all : (Nk, Nbands) ndarray
        Eigenvalues along the path (with interlayer).
    kept_sites : list of kwant sites
        Sites kept in the fundamental moiré cell.
    """
    import numpy as np, numpy.linalg as npl
    from math import acos, pi
    from scipy.spatial import cKDTree
    import kwant
    from collections import defaultdict
    import os
    import pandas as pd
    import plotly.graph_objects as go

    from .lattices import graphene_primitives
    from .geometry import moire_vectors_primitive, expected_sites_comm_tBLG
    from .builders import tperp_SK

    hp_range = float(hp)

    # --- basic TB knobs ---
    t_intra = -1.0 if E_in_t else -t
    if interlayer_mode == "baseline":
        hp_range, QSIGMA, QPI = hp_range, 7.42, 3.15
    else:
        hp_range, QSIGMA, QPI = 3.0, 3.0, 1.3
    r_xy_cut = hp_range * acc

    # energy window for plotting
    E_min, E_max = (-1, 1) if E_in_t else (-t, t)
    k_counts = list(k_counts)

    # --- helpers that are *not* already in py_tbl ---
    def Rtheta(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]])

    def kpath(points, counts):
        out = []
        for (kA, kB), n in zip(points, counts):
            ts = np.linspace(0, 1, n, endpoint=False)
            out.extend(kA*(1-t) + kB*t for t in ts)
        out.append(points[-1][1])
        return np.array(out)

    def seed_ranges_from_U(U, margin=2):
        """
        Given integer (or integer-like) U with [T1 T2] = [a1 a2] @ U,
        return integer index ranges that surely cover one moiré cell.
        """
        U = np.asarray(U, float)
        c0 = np.array([0, 0])
        c1 = U[:, 0]
        c2 = U[:, 1]
        c3 = c1 + c2
        corners = np.stack([c0, c1, c2, c3], axis=0)
        i_min = int(np.floor(np.min(corners[:, 0])) - margin)
        i_max = int(np.ceil (np.max(corners[:, 0])) + margin)
        j_min = int(np.floor(np.min(corners[:, 1])) - margin)
        j_max = int(np.ceil (np.max(corners[:, 1])) + margin)
        return (i_min, i_max), (j_min, j_max)

    # -------------------- graphene canonical lattice --------------------
    a1_b, a2_b, A_b, B_b = graphene_primitives(acc)

    L1 = kwant.lattice.general([a1_b, a2_b], [A_b, B_b], name="L1", norbs=1)
    A1, B1 = L1.sublattices

    # twist angle from (m,r)
    theta = acos((3*m*m + 3*m*r + 0.5*r*r)/(3*m*m + 3*m*r + r*r))

    # top layer (same construction as your script)
    eps = 0.0
    Delta = (B_b - A_b) if registration.upper()=="AB" else np.zeros(2)
    A_t0 = np.array([0.0, acc]) + Delta + eps
    B_t0 = np.array([0.0, 0.0]) + Delta + eps
    R = Rtheta(theta)
    a1_t, a2_t = R @ a1_b, R @ a2_b
    A_t,  B_t  = R @ A_t0, R @ B_t0
    L2 = kwant.lattice.general([a1_t, a2_t], [A_t, B_t], name="L2", norbs=1)
    A2, B2 = L2.sublattices

    BOTTOM = {A1, B1}
    is_bottom = lambda fam: fam in BOTTOM

    # moiré vectors from py_tbl.geometry
    T1, T2, U = moire_vectors_primitive(a1_b, a2_b, m, r)
    T = np.column_stack([T1, T2])
    Tinv = npl.inv(T)

    tot_exp, per_layer_exp, N_moire = expected_sites_comm_tBLG(m, r)
    print(f"[supercell] (m,r)=({m},{r}), θ={theta*180/pi:.2f}°, detU≈{round(abs(np.linalg.det(U)))} = N={N_moire}")
    print("NN distance (Å):", np.linalg.norm(B_b - A_b), " t_intra:", t_intra)

    # -------------------- build superset --------------------
    (RANGE_BOT_M, RANGE_BOT_N) = seed_ranges_from_U(U, margin=4)
    (RANGE_TOP_M, RANGE_TOP_N) = seed_ranges_from_U(U, margin=4)

    print("[auto-seed] bottom M,N =", RANGE_BOT_M, RANGE_BOT_N)
    print("[auto-seed] top    M,N =", RANGE_TOP_M, RANGE_TOP_N)

    TBL = kwant.Builder()
    for m_bott in range(*RANGE_BOT_M):
        for n_bott in range(*RANGE_BOT_N):
            TBL[A1(m_bott, n_bott)] = 0.0
            TBL[B1(m_bott, n_bott)] = 0.0

    for m_top_i in range(*RANGE_TOP_M):
        for n_top_i in range(*RANGE_TOP_N):
            TBL[A2(m_top_i, n_top_i)] = 0.0
            TBL[B2(m_top_i, n_top_i)] = 0.0

    # intralayer NN
    TBL[L1.neighbors(1)] = t_intra
    TBL[L2.neighbors(1)] = t_intra
    print("[superset] total hoppings:", sum(1 for _ in TBL.hoppings()))

    # -------------------- crop + TAG --------------------
    def crop_and_tag(TBL, T, Tinv, u0=1e-7*np.sqrt(2.0), v0=1e-7*np.sqrt(3.0),
                     tol=1e-12, match_tol=3e-6):
        T1, T2 = T[:,0], T[:,1]
        def uv(r): return Tinv @ np.asarray(r,float)
        def inside(r):
            u, v = Tinv @ np.asarray(r, float)
            return (u0 <= u) and (u < u0 + 1 - 1e-12) and (v0 <= v) and (v < v0 + 1 - 1e-12)

        kept = [s for s in TBL.sites() if inside(s.pos)]
        kept = tuple(kept); index = {s:i for i,s in enumerate(kept)}
        fam_sites = {fam:[s for s in kept if s.family is fam] for fam in {s.family for s in kept}}
        fam_pos   = {fam:np.array([np.asarray(s.pos,float) for s in fam_sites[fam]]) for fam in fam_sites}
        fam_tree  = {fam:cKDTree(fam_pos[fam]) for fam in fam_pos}

        onsite = np.zeros(len(kept))
        for s in kept: onsite[index[s]] = float(TBL[s])

        tagged=[]; seen=set()
        def add(i,j,amp,n1,n2):
            key=(i,j,n1,n2)
            if key in seen: return
            tagged.append((i,j,float(amp),int(n1),int(n2)))
            tagged.append((j,i,float(np.conjugate(amp)),-int(n1),-int(n2)))
            seen.add(key)

        def match(pos,fam):
            d,j = fam_tree[fam].query(np.asarray(pos,float),k=1)
            return (fam_sites[fam][j], d)

        for s1,s2 in TBL.hoppings():
            r1,r2 = np.asarray(s1.pos,float), np.asarray(s2.pos,float)
            in1,in2 = inside(r1), inside(r2)
            amp = TBL[s1,s2]

            if in1 and in2:
                add(index[s1], index[s2], amp, 0, 0)
                continue

            if in1 ^ in2:
                s_in, r_in = (s1,r1) if in1 else (s2,r2)
                s_out, r_out = (s2,r2) if in1 else (s1,r1)

                # compute wrap from OUTSIDE endpoint
                u_out, v_out = uv(r_out)
                du = int(np.floor(u_out - u0 + 1e-12))
                dv = int(np.floor(v_out - v0 + 1e-12))
                r_wrap = r_out - du*T1 - dv*T2

                s_match,_ = match(r_wrap, s_out.family)
                if s_match is None: continue
                i_in, j_m = index[s_in], index[s_match]
                if in1: add(i_in, j_m, amp, int(du), int(dv))
                else:   add(j_m, i_in, np.conjugate(amp), -int(du), -int(dv))

        return kept, onsite, tagged, T1, T2

    kept_sites, onsite, tagged_hops, T1, T2 = crop_and_tag(TBL, T, Tinv)

    # -------------------- add interlayer (tagged) --------------------
    def add_interlayer_tags(kept_sites, tagged_hops, T1, T2, r_xy_cut):
        pos = np.array([np.asarray(s.pos,float) for s in kept_sites])
        idx_bot = [i for i,s in enumerate(kept_sites) if is_bottom(s.family)]
        idx_top = [i for i,s in enumerate(kept_sites) if not is_bottom(s.family)]
        if not idx_bot or not idx_top: return
        tree_top = cKDTree(pos[idx_top])
        Tmat = np.column_stack([T1,T2]); Tinv_loc = np.linalg.inv(Tmat)
        def ab_of(svec):
            n = Tinv_loc @ svec
            return (int(round(n[0])), int(round(n[1])))

        seen=set(); addc=0
        for ib in idx_bot:
            r_b = pos[ib]
            for a in (-1,0,1):
                for b in (-1,0,1):
                    svec = a*T1 + b*T2
                    for jloc in tree_top.query_ball_point(r_b - svec, r_xy_cut):
                        it = idx_top[jloc]
                        r_img = pos[it] + svec
                        r_xy = np.linalg.norm(r_img - r_b)
                        amp = tperp_SK(
                            r_xy,
                            acc=acc, dperp=dperp, t=t, tp=tp,
                            E_in_t=E_in_t, QSIGMA=QSIGMA, QPI=QPI
                        )
                        if abs(amp) < 1e-12: continue
                        n1,n2 = ab_of(svec)
                        key=(ib,it,n1,n2)
                        if key in seen: continue
                        tagged_hops.append((ib, it, float(amp), n1, n2))
                        tagged_hops.append((it, ib, float(np.conjugate(amp)), -n1, -n2))
                        seen.add(key); addc += 1
        print(f"[interlayer] tagged {addc} hoppings (cut={r_xy_cut:.2f} Å).")

    add_interlayer_tags(kept_sites, tagged_hops, T1, T2, r_xy_cut)

    # -------------------- diagnostics --------------------
    built_total = len(kept_sites)
    built_bottom = sum(1 for s in kept_sites if is_bottom(s.family))
    print(f"[sites] built={built_total} (expected both layers={tot_exp}) split {built_bottom}/{built_total-built_bottom}")
    intra = [(i,j,a,n1,n2) for (i,j,a,n1,n2) in tagged_hops
             if is_bottom(kept_sites[i].family) == is_bottom(kept_sites[j].family)]
    n_intra_shift = sum(1 for (_,_,_,n1,n2) in intra if (n1,n2)!=(0,0))
    print(f"[intralayer] total tagged={len(intra)}  with nonzero shifts={n_intra_shift}  (should be > 0)")

    # -------------------- build H(k) --------------------
    by_shift_all = defaultdict(list)
    for i,j,amp,n1,n2 in tagged_hops:
        by_shift_all[(n1,n2)].append((i,j,amp))

    def is_inter(i,j):
        fi,fj = kept_sites[i].family, kept_sites[j].family
        return (fi in BOTTOM) ^ (fj in BOTTOM)

    N = len(kept_sites)
    def Hk(groups,k,onsite):
        H = np.zeros((N,N),dtype=np.complex128); np.fill_diagonal(H, onsite)
        for (n1,n2), hops in groups.items():
            phase = np.exp(1j * (k @ (n1*T1 + n2*T2)))
            for i,j,amp in hops: H[i,j] += amp*phase
        return 0.5*(H + H.conj().T)

    # -------------------- k-path & bands --------------------
    Tmat  = np.column_stack([T1,T2]); BinvT = np.linalg.inv(Tmat).T
    b1, b2 = 2*np.pi*BinvT[:,0], 2*np.pi*BinvT[:,1]
    G  = np.array([0.0,0.0]); K = (b1 + 2*b2)/3; Kp = (2*b1 + b2)/3; M = (b1 + b2)/2
    KPATH = kpath([(K,Kp),(Kp,G),(G,M),(M,K)], k_counts); Nk=len(KPATH)

    E_all = np.empty((Nk,N))
    for i,k in enumerate(KPATH):
        E_all[i] = np.linalg.eigvalsh(Hk(by_shift_all,k,onsite))

    # -------------------- optional plot --------------------
    if do_plot:
        def mask(E):
            X = E.copy(); X[(X<E_min)|(X>E_max)] = np.nan; return X
        Ew_all = mask(E_all)

        x = np.arange(Nk)
        ticks = [0, k_counts[0], sum(k_counts[:2]), sum(k_counts[:3]), sum(k_counts)]
        ticktext = ["K","K′","Γ","M","K"]
        fig = go.Figure()

        def add_traces(Ey,name,visible,width):
            for n in range(Ey.shape[1]):
                fig.add_trace(go.Scatter(
                    x=x,y=Ey[:,n],mode='lines',line=dict(width=width),
                    name=name if n==0 else None,legendgroup=name,showlegend=(n==0),
                    visible=visible,
                    hovertemplate="k=%{x}<br>E/t=%{y:.4f}<extra>"+name+"</extra>"
                ))
        add_traces(Ew_all,"with interlayer",True,1.2)

        for tpos in ticks:
            fig.add_vline(x=tpos,line_width=1,line_dash="dot",
                          line_color="rgba(100,100,100,0.35)")

        fig.update_layout(
            title=f"tBLG bands (K–K′–Γ–M–K), θ={theta*180/pi:.2f}°, "
                  f"window [{E_min:.2f},{E_max:.2f}] — mode: {interlayer_mode}",
            xaxis=dict(tickmode='array',tickvals=ticks,ticktext=ticktext,
                       range=[0,ticks[-1]]),
            yaxis=dict(title="Energy (in units of t)" if E_in_t else "Energy (eV)",
                       range=[E_min,E_max]),
            hovermode="x unified",
            legend=dict(orientation="h",x=0.02,y=1.06),
        )
        fig.show()

    # -------------------- optional CSV export --------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        def save_bands_to_csv(kpath, evals_path, filename="bands.csv", header=True):
            if kpath.ndim > 1:
                dk = np.sqrt(np.sum(np.diff(kpath, axis=0)**2, axis=1))
                kdist = np.concatenate([[0], np.cumsum(dk)])
            else:
                kdist = kpath
            N_k, N_b = evals_path.shape
            data = np.column_stack([kdist, evals_path])
            columns = ["k"] + [f"band_{i+1}" for i in range(N_b)] if header else None
            filepath = os.path.join(save_dir, filename)
            pd.DataFrame(data, columns=columns).to_csv(filepath, index=False,
                                                       float_format="%.8f")
            print(f"Saved {N_k} k-points × {N_b} bands to {filepath}")
            return filepath

        if csv_tag is None:
            csv_tag = f"r{r:02d}_m{m:02d}_hp{hp_range:.2f}"
        fname = f"bands_{csv_tag}.csv"
        save_bands_to_csv(KPATH, E_all, filename=fname)

    return KPATH, E_all, kept_sites


# kwant_bands.py

def build_pbc_H_gamma_from_kwant(
    m,
    r,
    hp,
    acc,
    dperp,
    t,
    tp,
    E_in_t,
    interlayer_mode="baseline",
    registration="AB",
):
    """
    Build the Γ-point (k=0) PBC Hamiltonian using the same kwant construction
    as the bandstructure script.

    Returns
    -------
    H_sparse : csr_matrix (N x N, real)
        Γ-point Hamiltonian with PBC.
    XY : (N,2) ndarray
        Site positions in the moiré cell.
    N1 : int
        Number of bottom-layer sites.
    """
    import numpy as np, numpy.linalg as npl
    from math import acos, pi
    from scipy.spatial import cKDTree
    import kwant
    from collections import defaultdict
    import scipy.sparse as ss

    from .lattices import graphene_primitives
    from .geometry import moire_vectors_primitive, expected_sites_comm_tBLG
    from .builders import tperp_SK

    # ---- TB knobs (same as in your kwant script) ----
    t_intra = -1.0 if E_in_t else -t
    hp_range = float(hp)

    if interlayer_mode == "baseline":
        hp_range, QSIGMA, QPI = hp_range, 7.42, 3.15
    else:
        hp_range, QSIGMA, QPI = 3.0, 3.0, 1.3

    r_xy_cut = hp_range * acc

    # -------------------- graphene + moiré --------------------
    a1_b, a2_b, A_b, B_b = graphene_primitives(acc)

    def Rtheta(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]])

    theta = acos((3*m*m + 3*m*r + 0.5*r*r)/(3*m*m + 3*m*r + r*r))

    eps = 0.0
    Delta = (B_b - A_b) if registration.upper() == "AB" else np.zeros(2)
    A_t0 = np.array([0.0, acc]) + Delta + eps
    B_t0 = np.array([0.0, 0.0]) + Delta + eps

    R = Rtheta(theta)
    a1_t, a2_t = R @ a1_b, R @ a2_b
    A_t,  B_t  = R @ A_t0, R @ B_t0

    L1 = kwant.lattice.general([a1_b, a2_b], [A_b, B_b], name="L1", norbs=1)
    A1, B1 = L1.sublattices
    L2 = kwant.lattice.general([a1_t, a2_t], [A_t, B_t], name="L2", norbs=1)
    A2, B2 = L2.sublattices

    BOTTOM = {A1, B1}
    is_bottom = lambda fam: fam in BOTTOM

    T1, T2, U = moire_vectors_primitive(a1_b, a2_b, m, r)
    T = np.column_stack([T1, T2])
    Tinv = npl.inv(T)

    tot_exp, per_layer_exp, N_moire = expected_sites_comm_tBLG(m, r)
    print(f"[PBC/kwant] (m,r)=({m},{r}), θ={theta*180/pi:.2f}°, expected N={N_moire}")

    # -------------------- seed ranges --------------------
# --- NEW: per-layer seed ranges from geometry, not from U ---
    def seed_ranges_layer(a1, a2, T1, T2, margin=4):
        """
        Find index ranges (i,j) in the (a1,a2) basis that surely cover
        the rhombus spanned by T1,T2 (with n_mult=1), padded by `margin`.
        """
        M = np.linalg.solve(np.column_stack([a1, a2]),
                            np.column_stack([T1, T2]))
        # corners in (i,j) index space
        corners = np.array([
            [0.0, 0.0],
            M[:, 0],
            M[:, 1],
            M[:, 0] + M[:, 1],
        ])
        i_min, j_min = np.floor(corners.min(0) - margin).astype(int)
        i_max, j_max = np.ceil (corners.max(0) + margin).astype(int)
        return (i_min, i_max+1), (j_min, j_max+1)  # half-open ranges

    (RANGE_BOT_M, RANGE_BOT_N) = seed_ranges_layer(a1_b, a2_b, T1, T2, margin=4)
    (RANGE_TOP_M, RANGE_TOP_N) = seed_ranges_layer(a1_t, a2_t, T1, T2, margin=4)

    print("[PBC/kwant] auto-seed bottom M,N =", RANGE_BOT_M, RANGE_BOT_N)
    print("[PBC/kwant] auto-seed top    M,N =", RANGE_TOP_M, RANGE_TOP_N)

    # -------------------- superset Builder --------------------
    TBL = kwant.Builder()

    for m_bott in range(*RANGE_BOT_M):
        for n_bott in range(*RANGE_BOT_N):
            TBL[A1(m_bott, n_bott)] = 0.0
            TBL[B1(m_bott, n_bott)] = 0.0

    for m_top_i in range(*RANGE_TOP_M):
        for n_top_i in range(*RANGE_TOP_N):
            TBL[A2(m_top_i, n_top_i)] = 0.0
            TBL[B2(m_top_i, n_top_i)] = 0.0

    TBL[L1.neighbors(1)] = t_intra
    TBL[L2.neighbors(1)] = t_intra
    print("[PBC/kwant] superset total hoppings:", sum(1 for _ in TBL.hoppings()))

    # -------------------- crop + tag intralayer --------------------
    def crop_and_tag(TBL, T, Tinv,
                     u0=1e-7 * np.sqrt(2.0),
                     v0=1e-7 * np.sqrt(3.0),
                     match_tol=3e-6):
        T1_loc, T2_loc = T[:, 0], T[:, 1]

        def uv(r):
            return Tinv @ np.asarray(r, float)

        def inside(r):
            u, v = uv(r)
            return (u0 <= u) and (u < u0 + 1 - 1e-12) and (v0 <= v) and (v < v0 + 1 - 1e-12)

        kept = [s for s in TBL.sites() if inside(s.pos)]
        kept = tuple(kept)
        index = {s: i for i, s in enumerate(kept)}

        fams = {s.family for s in kept}
        fam_sites = {fam: [s for s in kept if s.family is fam] for fam in fams}
        fam_pos   = {fam: np.array([np.asarray(s.pos, float) for s in fam_sites[fam]])
                     for fam in fams}
        fam_tree  = {fam: cKDTree(fam_pos[fam]) for fam in fams}

        onsite = np.zeros(len(kept), float)
        for s in kept:
            onsite[index[s]] = float(TBL[s])

        tagged = []
        seen = set()

        def add(i, j, amp, n1, n2):
            key = (i, j, n1, n2)
            if key in seen:
                return
            tagged.append((i, j, float(amp), int(n1), int(n2)))
            tagged.append((j, i, float(np.conjugate(amp)), -int(n1), -int(n2)))
            seen.add(key)

        def match(pos, fam):
            d, j = fam_tree[fam].query(np.asarray(pos, float), k=1)
            return fam_sites[fam][j], d

        for s1, s2 in TBL.hoppings():
            r1 = np.asarray(s1.pos, float)
            r2 = np.asarray(s2.pos, float)
            in1 = inside(r1)
            in2 = inside(r2)
            amp = TBL[s1, s2]

            if in1 and in2:
                add(index[s1], index[s2], amp, 0, 0)
                continue

            if in1 ^ in2:
                s_in, r_in = (s1, r1) if in1 else (s2, r2)
                s_out, r_out = (s2, r2) if in1 else (s1, r1)

                u_out, v_out = uv(r_out)
                du = int(np.floor(u_out - u0 + 1e-12))
                dv = int(np.floor(v_out - v0 + 1e-12))
                r_wrap = r_out - du * T1_loc - dv * T2_loc

                s_match, d = match(r_wrap, s_out.family)
                if d > match_tol:
                    continue

                i_in = index[s_in]
                j_m  = index[s_match]
                if in1:
                    add(i_in, j_m, amp, du, dv)
                else:
                    add(j_m, i_in, np.conjugate(amp), -du, -dv)

        return kept, onsite, tagged, T1_loc, T2_loc

    kept_sites, onsite, tagged_hops, T1_loc, T2_loc = crop_and_tag(TBL, T, Tinv)

    # -------------------- add interlayer SK tags --------------------
    def add_interlayer_tags(kept_sites, tagged_hops, T1, T2, r_xy_cut):
        pos = np.array([np.asarray(s.pos, float) for s in kept_sites])
        idx_bot = [i for i, s in enumerate(kept_sites) if is_bottom(s.family)]
        idx_top = [i for i, s in enumerate(kept_sites) if not is_bottom(s.family)]
        if not idx_bot or not idx_top:
            return

        tree_top = cKDTree(pos[idx_top])
        Tmat = np.column_stack([T1, T2])
        Tinv_local = np.linalg.inv(Tmat)

        def ab_of(svec):
            n = Tinv_local @ svec
            return (int(round(n[0])), int(round(n[1])))

        seen = set()
        addc = 0

        def add(i, j, amp, n1, n2):
            nonlocal addc
            key = (i, j, n1, n2)
            if key in seen:
                return
            tagged_hops.append((i, j, float(amp), n1, n2))
            tagged_hops.append((j, i, float(np.conjugate(amp)), -n1, -n2))
            seen.add(key)
            addc += 1

        for ib in idx_bot:
            r_b = pos[ib]
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    svec = a * T1 + b * T2
                    for jloc in tree_top.query_ball_point(r_b - svec, r_xy_cut):
                        it = idx_top[jloc]
                        r_img = pos[it] + svec
                        r_xy = np.linalg.norm(r_img - r_b)
                        amp = tperp_SK(
                            r_xy,
                            acc=acc, dperp=dperp, t=t, tp=tp,
                            E_in_t=E_in_t, QSIGMA=QSIGMA, QPI=QPI
                        )
                        if abs(amp) < 1e-12:
                            continue
                        n1, n2 = ab_of(svec)
                        add(ib, it, amp, n1, n2)

        print(f"[PBC/kwant] interlayer tagged {addc} hoppings (cut={r_xy_cut:.2f} Å).")

    if tp != 0.0 and r_xy_cut > 0.0:
        add_interlayer_tags(kept_sites, tagged_hops, T1_loc, T2_loc, r_xy_cut)

    # -------------------- build H(k=0) --------------------
    # -------------------- build sparse H(k=0) directly --------------------

    N = len(kept_sites)

    rows = []
    cols = []
    vals = []

    # diagonal (onsite)
    for i in range(N):
        rows.append(i)
        cols.append(i)
        vals.append(float(onsite[i]))

    # off-diagonal hoppings at k=0 (all phases = 1)
    # tagged_hops already contains both i→j and j→i entries
    for i, j, amp, n1, n2 in tagged_hops:
        rows.append(i)
        cols.append(j)
        vals.append(float(np.real(amp)))  # Hamiltonian is Hermitian, keep real part

    H_sparse = ss.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    # positions & N1
    XY = np.array([np.asarray(s.pos, float) for s in kept_sites])
    N1 = sum(1 for s in kept_sites if is_bottom(s.family))

    return H_sparse, XY, N1

