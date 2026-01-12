
import os
import re
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# FIXED DIRECTORY (your folder)
# ---------------------------------------------------------
BASE_DIR = Path(r"C:\Users\hol1brg\OneDrive - Bosch Group\DAILY\tBLG\conductance")

# Subfolders to create
SE_EVEN_DIR = BASE_DIR / "SE even"
SE_ODD_DIR = BASE_DIR / "SE odd"

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
# Commensurate Moiré supercell size
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
    files = [f for f in BASE_DIR.glob("tblg_conductance_m=*_*_leadtype=11.csv")]

    if not files:
        print("No matching files found.")
        return

    SE_even_data = {}
    SE_odd_data = {}

    for path in files:
        parsed = parse_m_r(path.name)
        if parsed is None:
            continue

        m, r = parsed
        df = pd.read_csv(path)

        moire = moire_cell_size(m, r)
        is_even = (r % 3 == 0)

        for _, row in df.iterrows():
            E = float(row["E"])
            T = float(row["T"])

            if is_even:
                SE_even_data.setdefault(E, []).append((moire, T))
            else:
                SE_odd_data.setdefault(E, []).append((moire, T))

    # Write out SE even files
    for E, data in SE_even_data.items():
        data_sorted = sorted(data)
        df_out = pd.DataFrame(data_sorted, columns=["MoireSize", "T_M"])
        out_path = SE_EVEN_DIR / f"T_vs_Moire_SE even_E={E:.6f}.csv"
        df_out.to_csv(out_path, index=False)
        print("Wrote", out_path)

    # Write out SE odd files
    for E, data in SE_odd_data.items():
        data_sorted = sorted(data)
        df_out = pd.DataFrame(data_sorted, columns=["MoireSize", "T_M"])
        out_path = SE_ODD_DIR / f"T_vs_Moire_SE odd_E={E:.6f}.csv"
        df_out.to_csv(out_path, index=False)
        print("Wrote", out_path)


if __name__ == "__main__":
    main()
