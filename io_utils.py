# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:55:38 2025

@author: HOL1BRG
"""

import os, json, yaml, pandas as pd
from .config import Config

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True); return path

def save_states_csv(path, df):
    ensure_dir(os.path.dirname(path)); df.to_csv(path, index=False, float_format="%.8e")

def load_config_yaml(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # light mapper
    from dataclasses import asdict
    cfg = Config()
    # shallow update
    for section, values in raw.items():
        obj = getattr(cfg, section)
        for k,v in values.items(): setattr(obj, k, v)
    return cfg
