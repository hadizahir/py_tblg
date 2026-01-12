
import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# FIXED DIRECTORY (your folder)
# ---------------------------------------------------------
BASE_DIR = Path(r"C:\Users\hol1brg\OneDrive - Bosch Group\DAILY\tBLG\conductance")

# Subfolders to create
SE_EVEN_DIR = BASE_DIR / "SE even"
SE_ODD_DIR  = BASE_DIR / "SE odd"

SE_EVEN_DIR.mkdir(exist_ok=True)
SE_ODD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Filename pattern
# ---------------------------------------------------------
FILENAME_RE = re.compile(
    r"tblg_conductance_m=(?P<m>-?\d+)_r=(?P<r>-?\d+)_leadtype=\d+\.csv$"
)

def parse_m_r(file_name):
    match = FILENAME_RE.search(file_name)
    if not match:
        return None
    return int(match.group("m")), int(match.group("r"))

# ---------------------------------------------------------
# Commensurate Moiré supercell size (dimensionless proxy)
# ---------------------------------------------------------
def moire_cell_size(m, r):
    S = 3*m*m + 3*m*r + r*r
    if r % 3 == 0:
        return (4 * S) / 3.0   # SE even
    else:
        return 4 * S           # SE odd

# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():

    # Gather all matching CSVs
    files = [p for p in BASE_DIR.glob("tblg_conductance_m=*_*_leadtype=11.csv")
             if FILENAME_RE.search(p.name)]

    if not files:
        print("No matching files found in:", BASE_DIR)
        return

    # Storage: energy -> list of (MoireSize, T) for each family
    se_even_data = {}
    se_odd_data  = {}

    # Also write per-energy CSVs into SE even / SE odd subfolders
    # (same as your previous requirement)
    se_even_rows = {}
    se_odd_rows  = {}

    for path in files:
        parsed = parse_m_r(path.name)
        if parsed is None:
            continue

        m, r = parsed
        df = pd.read_csv(path)
        # Sanity: expect columns E, G, T, N_modes
        df.columns = [c.strip() for c in df.columns]
        if not {"E","G","T","N_modes"}.issubset(df.columns):
            print(f"Skipping {path} (missing expected columns)")
            continue

        moire = moire_cell_size(m, r)
        is_even = (r % 3 == 0)

        for _, row in df.iterrows():
            E = float(row["E"])
            T = float(row["T"])

            if is_even:
                se_even_data.setdefault(E, []).append((moire, T))
                se_even_rows.setdefault(E, []).append((moire, T))
            else:
                se_odd_data.setdefault(E, []).append((moire, T))
                se_odd_rows.setdefault(E, []).append((moire, T))

    # Write CSVs with exact labels and spaces in names
    for E, data in se_even_rows.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        df_out = pd.DataFrame(data_sorted, columns=["MoireSize", "T"])
        out_path = SE_EVEN_DIR / f"T_vs_Moire_SE even_E={E:.6f}.csv"
        df_out.to_csv(out_path, index=False)
        print("Wrote", out_path)

    for E, data in se_odd_rows.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        df_out = pd.DataFrame(data_sorted, columns=["MoireSize", "T"])
        out_path = SE_ODD_DIR / f"T_vs_Moire_SE odd_E={E:.6f}.csv"
        df_out.to_csv(out_path, index=False)
        print("Wrote", out_path)

    # -----------------------------------------------------
    # Plotting: one panel for SE even, one for SE odd
    # -----------------------------------------------------
    plt.figure(figsize=(14, 6))
    ax_even = plt.subplot(1, 2, 1)
    ax_odd  = plt.subplot(1, 2, 2)

    # Choose colormaps for distinct energies
    # We’ll sort energies to have consistent color ordering
    energies_even = sorted(se_even_data.keys())
    energies_odd  = sorted(se_odd_data.keys())

    # Plot SE even
    if energies_even:
        cmap_even = plt.cm.tab20
        for i, E in enumerate(energies_even):
            pairs = se_even_data[E]
            if not pairs:
                continue
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            moire_vals = [p[0] for p in pairs_sorted]
            T_vals     = [p[1] for p in pairs_sorted]
            color = cmap_even(i % cmap_even.N)
            ax_even.plot(moire_vals, T_vals, label=f"E={E:.6f}", color=color, linewidth=1.8)
        ax_even.set_title("SE even: T vs Moiré size")
        ax_even.set_xlabel("Moiré size (dimensionless, commensuration proxy)")
        ax_even.set_ylabel("T")
        ax_even.grid(True, alpha=0.3)
        # Place legend outside if too many energies
        ax_even.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    else:
        ax_even.text(0.5, 0.5, "No SE even data", ha="center", va="center", transform=ax_even.transAxes)

    # Plot SE odd
    if energies_odd:
        cmap_odd = plt.cm.tab20b
        for i, E in enumerate(energies_odd):
            pairs = se_odd_data[E]
            if not pairs:
                continue
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            moire_vals = [p[0] for p in pairs_sorted]
            T_vals     = [p[1] for p in pairs_sorted]
            color = cmap_odd(i % cmap_odd.N)
            ax_odd.plot(moire_vals, T_vals, label=f"E={E:.6f}", color=color, linewidth=1.8)
        ax_odd.set_title("SE odd: T vs Moiré size")
        ax_odd.set_xlabel("Moiré size (dimensionless, commensuration proxy)")
        ax_odd.set_ylabel("T")
        ax_odd.grid(True, alpha=0.3)
        ax_odd.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    else:
        ax_odd.text(0.5, 0.5, "No SE odd data", ha="center", va="center", transform=ax_odd.transAxes)

    plt.tight_layout()
    fig_path = BASE_DIR / "T_vs_MoireSize_SE_panels.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print("Saved plot:", fig_path)

if __name__ == "__main__":
    main()
