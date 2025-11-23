
import os
from PIL import Image, ImageDraw, ImageFont

# === 1. Generate Architecture Diagram ===
def create_architecture_diagram(output_path="docs/architecture.png"):
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 36)
        font_subtitle = ImageFont.truetype("arial.ttf", 28)
        font_text = ImageFont.truetype("arial.ttf", 22)
    except:
        font_title = font_subtitle = font_text = ImageFont.load_default()

    draw.text((width//2 - 200, 20), "Architectural Overview", font=font_title, fill="black")

    boxes = [
        ("Geometry & Lattice", ["geometry.py", "lattices.py"], "Compute moiré vectors\nTwist angle"),
        ("Hamiltonian Construction", ["builders.py", "kwant_bands.py"], "Sparse Hamiltonians\nBandstructure"),
        ("Analysis", ["analysis.py", "run_pbc.py"], "Scaling studies\nPBC eigenstates"),
        ("Graph & Connectivity", ["io_utils.py"], "Interlayer adjacency\nLaplacian modes"),
        ("Visualization & Post-Processing", ["wavefunctions.py", "plots.py", "registry.py", "spectra.py"], "Wavefunction plots\nRegistry maps\nSpectral metrics")
    ]

    coords = [(50, 100), (650, 100), (650, 300), (50, 300), (50, 500)]
    box_w, box_h = 500, 150

    for (title, files, desc), (x, y) in zip(boxes, coords):
        draw.rectangle([x, y, x+box_w, y+box_h], outline="black", width=3)
        draw.text((x+10, y+10), title, font=font_subtitle, fill="black")
        draw.text((x+10, y+50), ", ".join(files), font=font_text, fill="darkblue")
        draw.text((x+10, y+90), desc, font=font_text, fill="gray")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"[INFO] Architecture diagram saved to {output_path}")

# === 2. Update README.md ===
def update_readme(readme_path="README.md", image_path="docs/architecture.png"):
    if not os.path.isfile(readme_path):
        print(f"[ERROR] README.md not found at {readme_path}")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "## 🏗 Architectural Overview" in content:
        content = content.split("## 🏗 Architectural Overview")[0]

    new_section = f"""
---

## 🏗 Architectural Overview

The following diagram illustrates the modular architecture of **py_tblg**:

{image_path}

### Layers:
1. **Geometry & Lattice**: `geometry.py`, `lattices.py`
2. **Hamiltonian Construction**: `builders.py`, `kwant_bands.py`
3. **Analysis**: `analysis.py`, `run_pbc.py`
4. **Graph & Connectivity**: `io_utils.py`
5. **Visualization & Post-Processing**: `wavefunctions.py`, `plots.py`, `registry.py`, `spectra.py`

---

### 📂 File Responsibilities

| File                | Responsibilities                                      | Key Functions                                      |
|---------------------|--------------------------------------------------------|----------------------------------------------------|
| `geometry.py`       | Moiré vectors, rhombus geometry, twist angle          | `moire_vectors_primitive`, `theta_comm_deg_from_mr` |
| `lattices.py`       | Graphene primitives, Kwant lattice setup              | `graphene_primitives`, `layer_lattices`           |
| `builders.py`       | Sparse Hamiltonians for flakes and PBC                | `build_flake_H_sparse`, `build_approximant_H_sparse_pbc` |
| `kwant_bands.py`    | Kwant-based bandstructure and Γ-point Hamiltonian     | `compute_tblg_bands_kwant`, `build_pbc_H_gamma_from_kwant` |
| `analysis.py`       | Scaling studies, gap analysis                         | `run_gap_scan`                                     |
| `run_pbc.py`        | PBC eigenstate computation, wavefunction saving       | `run_one_approximant`                              |
| `run_pbc_analysis.py`| Graph-theoretic analysis, Laplacian modes            | `compute_graph_metrics`                            |
| `io_utils.py`       | Interlayer adjacency, CSV export                      | `save_interlayer_hoppings_csv`, `build_interlayer_adjacency_from_csv` |
| `wavefunctions.py`  | Wavefunction overlays and 3D plots                    | `save_wavefunction_overlay_registry_png_clean`     |
| `plots.py`          | Heatmaps, lattice plots, interlayer link visualization| `plot_wavefunction_heatmap`, `plot_interlayer_links` |
| `registry.py`       | Moiré registry maps, AA/AB/BA/Walls masks             | `get_registry_grid`, `region_masks_from_phi`       |
| `spectra.py`        | Eigenvalue solvers, IPR, edge metrics                 | `eigs_in_window_sliced`, `ipr`                    |

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/yourusername/py_tblg.git
cd py_tblg
pip install -r requirements.txt
