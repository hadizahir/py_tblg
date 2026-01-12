# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:53:55 2025

@author: HOL1BRG
"""
from collections import defaultdict
import numpy as np, scipy.sparse as ss
from scipy.spatial import cKDTree
from .geometry import inside_rhombus
import kwant  
# --- PBC helpers (safe to use elsewhere if needed) ---
def _tile_shifts(T1, T2):
    """
    3×3 stencil of lattice shifts u*T1 + v*T2 to realize minimum-image PBC.
    """
    shifts = []
    for u in (-1, 0, 1):
        for v in (-1, 0, 1):
            shifts.append(u * T1 + v * T2)
    return np.array(shifts, float)

def build_approximant_H_sparse_pbc(
    a1_b, a2_b, A_b, B_b, a1_t, a2_t, A_t, B_t,
    T1, T2,
    acc, dperp, t, tp,
    r_xy_cut, t_intra, QSIGMA, QPI, E_in_t
):
    """
    Periodic (PBC) Hamiltonian on a single (T1,T2) rhombic supercell.

    Intralayer:
      - build unit-cell sites for bottom/top via inside_rhombus (n_mult=1),
      - tile the cell 3×3 with shifts u*T1+v*T2,
      - use KDTree to find NN within ~acc,
      - deduplicate on base indices.

    Interlayer:
      - tile top layer 3×3,
      - for each bottom site, find top images within r_xy_cut,
      - for each base pair keep the closest image, compute SK amplitude.

    Returns
    -------
    H : csr_matrix (N x N)
    XY : (N,2) float array
    N1 : int  number of bottom-layer sites
    """
    origin = np.array([0.0, 0.0])

    # --- 1) Unit-cell sites (no giant superset) ---
    def _layer_sites(a1, a2, A, B):
        # identical to your original version: build a box in (a1,a2) indices,
        # then keep only those inside the T1,T2 rhombus with n_mult=1.
        M = np.linalg.solve(np.column_stack([a1, a2]), np.column_stack([T1, T2]))
        corners = np.array([[0, 0], M[:, 0], M[:, 1], (M[:, 0] + M[:, 1])])
        i_min, j_min = np.floor(corners.min(0) - 2).astype(int)
        i_max, j_max = np.ceil (corners.max(0) + 2).astype(int)

        pts = []
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for basis in (A, B):
                    r = basis + i * a1 + j * a2
                    if inside_rhombus(r, origin, T1, T2, 1):
                        pts.append(r)
        return np.array(pts, float)

    bot = _layer_sites(a1_b, a2_b, A_b, B_b)
    top = _layer_sites(a1_t, a2_t, A_t, B_t)
    N1, N2 = len(bot), len(top)
    XY = np.vstack([bot, top])

    # --- helper: intralayer via 3×3 tiling on torus ---
    def intralayer_edges(XY_layer, offset):
        """
        XY_layer: (N_layer, 2) positions for one layer in unit cell
        offset: integer offset to add to base indices (0 for bottom, N1 for top)
        Returns lists rows, cols, vals for intralayer hoppings.
        """
        N_layer = XY_layer.shape[0]
        shifts = _tile_shifts(T1, T2)  # 3×3 array of shape (9,2)
        nS = len(shifts)

        # tile positions
        imgs = np.vstack([XY_layer + s for s in shifts])  # (nS*N_layer, 2)
        base_idx = np.repeat(np.arange(N_layer), nS)

        tree = cKDTree(imgs)
        rows_l, cols_l, vals_l = [], [], []
        seen = set()

        for a, b in tree.query_pairs(1.01 * acc):
            ia = base_idx[a]
            ib = base_idx[b]
            if ia == ib:
                continue
            i = offset + ia
            j = offset + ib
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            rows_l += [i, j]
            cols_l += [j, i]
            vals_l += [t_intra, t_intra]

        return rows_l, cols_l, vals_l

    rows, cols, vals = [], [], []

    # --- 2) Intralayer bottom & top ---
    r_b, c_b, v_b = intralayer_edges(bot, 0)
    rows += r_b; cols += c_b; vals += v_b

    r_t, c_t, v_t = intralayer_edges(top, N1)
    rows += r_t; cols += c_t; vals += v_t

    # --- 3) Interlayer via 3×3 tiling (optional; skip if tp=0) ---
    if tp != 0.0 and r_xy_cut > 0.0:
        shifts = _tile_shifts(T1, T2)
        nS = len(shifts)
        # tile top
        top_imgs = np.vstack([top + s for s in shifts])        # (nS*N2, 2)
        top_base = np.repeat(np.arange(N2), nS)

        tree_top = cKDTree(top_imgs)
        best_bt = {}  # (ib, jt) -> (r_xy_min, amp)

        for ib, r_b in enumerate(bot):
            hits = tree_top.query_ball_point(r_b, r_xy_cut)
            if not hits:
                continue
            for h in hits:
                jt = top_base[h]
                dr = top_imgs[h] - r_b
                r_xy = float(np.hypot(dr[0], dr[1]))
                if r_xy <= 0.0 or r_xy > r_xy_cut:
                    continue
                amp = tperp_SK(
                    r_xy,
                    acc=acc, dperp=dperp, t=t, tp=tp,
                    E_in_t=E_in_t, QSIGMA=QSIGMA, QPI=QPI
                )
                if abs(amp) < 1e-12:
                    continue
                key = (ib, jt)
                if key not in best_bt or r_xy < best_bt[key][0]:
                    best_bt[key] = (r_xy, amp)

        for (ib, jt), (rmin, amp) in best_bt.items():
            i = ib
            j = N1 + jt
            rows += [i, j]
            cols += [j, i]
            vals += [amp, amp]

    # --- 4) Assemble H ---
    N = N1 + N2
    H = ss.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return H, XY, N1

def build_approximant_H_sparse_pbc_kwant(
    a1_b, a2_b, A_b, B_b,
    a1_t, a2_t, A_t, B_t,
    T1, T2, U,
    acc, dperp, t, tp,
    r_xy_cut, t_intra, QSIGMA, QPI, E_in_t
):
    """
    Build the commensurate supercell with PBC using the same logic
    as your kwant bandstructure script, and return the Γ-point
    Hamiltonian H (sparse), positions XY, and N1 (number of bottom sites).

    Parameters
    ----------
    a1_b, a2_b, A_b, B_b : bottom layer primitive vectors and sublattice shifts
    a1_t, a2_t, A_t, B_t : top layer primitive vectors and sublattice shifts
    T1, T2               : moiré supercell primitive vectors
    U                    : 2x2 integer matrix from moire_vectors_primitive
    acc, dperp, t, tp    : TB parameters
    r_xy_cut             : in-plane cutoff for interlayer
    t_intra              : intralayer hopping (usually -1 if E_in_t=True)
    QSIGMA, QPI, E_in_t  : SK parameters & scaling mode

    Returns
    -------
    H : csr_matrix (N x N)
        Γ-point Hamiltonian (PBC)
    XY : (N,2) float array
        Positions of sites (kept in kwant order; N1 is count of bottom sites)
    N1 : int
        Number of bottom-layer sites among the N sites
    """
    # ---- 0. basic geometry / helpers ----
    T = np.column_stack([T1, T2])
    Tinv = np.linalg.inv(T)

    # kwant lattices for bottom & top
    L1 = kwant.lattice.general([a1_b, a2_b], [A_b, B_b], name='L1', norbs=1)
    A1, B1 = L1.sublattices

    L2 = kwant.lattice.general([a1_t, a2_t], [A_t, B_t], name='L2', norbs=1)
    A2, B2 = L2.sublattices

    BOTTOM = {A1, B1}
    def is_bottom(fam):
        return fam in BOTTOM

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

    # ---- 1. Build superset in kwant.Builder (both layers) ----
    TBL = kwant.Builder()

    # bottom superset
    for m_bott in range(*RANGE_BOT_M):
        for n_bott in range(*RANGE_BOT_N):
            TBL[A1(m_bott, n_bott)] = 0.0
            TBL[B1(m_bott, n_bott)] = 0.0

    # top superset
    for m_top_i in range(*RANGE_TOP_M):
        for n_top_i in range(*RANGE_TOP_N):
            TBL[A2(m_top_i, n_top_i)] = 0.0
            TBL[B2(m_top_i, n_top_i)] = 0.0

    # intralayer hoppings (graphene NN)
    TBL[L1.neighbors(1)] = t_intra
    TBL[L2.neighbors(1)] = t_intra

    # ---- 2. crop + tag intralayer hoppings (same as crop_and_tag) ----
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

        # keep sites inside
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
                # one inside, one outside
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

    # ---- 3. add interlayer tagged hoppings (same logic as add_interlayer_tags) ----
    def add_interlayer_tags(kept_sites, tagged_hops, T1, T2, r_xy_cut, tperp_func):
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
                        amp = tperp_func(np.linalg.norm(r_img - r_b))
                        if amp == 0:
                            continue
                        n1, n2 = ab_of(svec)
                        add(ib, it, amp, n1, n2)

    # SK wrapper using existing tperp_SK
    def tperp_local(r_xy):
        return tperp_SK(
            r_xy,
            acc=acc, dperp=dperp, t=t, tp=tp,
            E_in_t=E_in_t, QSIGMA=QSIGMA, QPI=QPI
        )

    if tp != 0.0 and r_xy_cut > 0.0:
        add_interlayer_tags(kept_sites, tagged_hops, T1_loc, T2_loc, r_xy_cut, tperp_local)

    # ---- 4. Build H(k=0) from tagged_hops (same as Hk with k=0) ----
    N = len(kept_sites)
    by_shift_all = defaultdict(list)
    for i, j, amp, n1, n2 in tagged_hops:
        by_shift_all[(n1, n2)].append((i, j, amp))

    H_dense = np.zeros((N, N), dtype=np.complex128)
    np.fill_diagonal(H_dense, onsite)

    # at k=0, all phases = 1
    for (n1, n2), hops in by_shift_all.items():
        for i, j, amp in hops:
            H_dense[i, j] += amp

    # Hermitian symmetrize
    H_dense = 0.5 * (H_dense + H_dense.conj().T)

    # positions & N1
    XY = np.array([np.asarray(s.pos, float) for s in kept_sites])
    N1 = sum(1 for s in kept_sites if is_bottom(s.family))

    # convert to real CSR for eigsh
    H_real = H_dense.real
    H_sparse = ss.csr_matrix(H_real)

    return H_sparse, XY, N1

def tperp_SK(r_xy, acc, dperp, t, tp, E_in_t, QSIGMA, QPI):
    tp_eff = tp/t if E_in_t else tp
    t_eff  = 1.0 if E_in_t else t
    rp = float(np.sqrt(r_xy**2 + dperp**2))
    sin2w = (r_xy**2)/(dperp**2 + r_xy**2 + 1e-30)
    cos2w = (dperp**2)/(dperp**2 + r_xy**2 + 1e-30)
    VPPsigma = tp_eff*np.exp(QSIGMA*(1.0 - rp/dperp))
    VPPpi    = -t_eff*np.exp(QPI*(1.0 - rp/acc))
    return float(VPPsigma*cos2w + VPPpi*sin2w)

def build_flake_H_sparse(n_mult, a1_b, a2_b, A_b, B_b, a1_t, a2_t, A_t, B_t,
                         T1, T2, acc, dperp, t, tp, r_xy_cut, t_intra, QSIGMA, QPI, E_in_t):
    origin = np.array([0.0,0.0])

    # index bounds via solving for (i,j) that cover rhombus roughly
    def bounds(a1,a2):
        M = np.linalg.solve(np.column_stack([a1,a2]), np.column_stack([T1,T2]))
        corners = np.array([[0,0], n_mult*M[:,0], n_mult*M[:,1], n_mult*(M[:,0]+M[:,1])])
        i_min,j_min = np.floor(corners.min(0)-4).astype(int)
        i_max,j_max = np.ceil (corners.max(0)+4).astype(int)
        return i_min,i_max,j_min,j_max

    i_min_b,i_max_b,j_min_b,j_max_b = bounds(a1_b,a2_b)
    i_min_t,i_max_t,j_min_t,j_max_t = bounds(a1_t,a2_t)

    # coordinates
    bot, top = [], []
    for i in range(i_min_b, i_max_b+1):
        for j in range(j_min_b, j_max_b+1):
            for basis in (A_b, B_b):
                r = basis + i*a1_b + j*a2_b
                if inside_rhombus(r, origin, T1, T2, n_mult): bot.append(r)
    for i in range(i_min_t, i_max_t+1):
        for j in range(j_min_t, j_max_t+1):
            for basis in (A_t, B_t):
                r = basis + i*a1_t + j*a2_t
                if inside_rhombus(r, origin, T1, T2, n_mult): top.append(r)
    bot = np.array(bot); top = np.array(top)
    N1, N2 = len(bot), len(top)
    XY = np.vstack([bot, top])

    rows, cols, vals = [], [], []

    # intralayer NN (distance ~ acc)
    for XYl, off in ((bot,0),(top,N1)):
        tree = cKDTree(XYl)
        for i,j in tree.query_pairs(1.01*acc):
            rows += [off+i, off+j]; cols += [off+j, off+i]; vals += [t_intra, t_intra]

    # interlayer SK
    ttree = cKDTree(top)
    for i, rb in enumerate(bot):
        hits = ttree.query_ball_point(rb, r_xy_cut)
        for j in hits:
            dx, dy = (top[j] - rb)
            amp = tperp_SK(
                np.hypot(dx, dy),
                acc=acc, dperp=dperp, t=t, tp=tp, E_in_t=E_in_t,
                QSIGMA=QSIGMA, QPI=QPI
            )
            if abs(amp) > 1e-12:
                rows += [i, N1 + j]
                cols += [N1 + j, i]
                vals += [amp, amp]

    H = ss.coo_matrix((vals, (rows, cols)), shape=(N1+N2, N1+N2)).tocsr()
    return H, XY, N1
