pip install -U numpy scipy kwant matplotlib plotly pandas pyyaml
# either:
python -m py_tbl.run --config py_tbl/configs/default.yaml

# or, without YAML (uses defaults):
python -m py_tbl.run


# run post processing 

python -m py_tbl.pbc_ladder_analysis --save_dir bands_pbc --dE 0.02

