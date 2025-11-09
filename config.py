# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:46:02 2025

@author: HOL1BRG
"""

# py_tbl/config.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TBParams:
    acc: float = 1.42
    dperp: float = 3.35
    t: float = 2.79
    tp: float = 0.35
    E_in_t: bool = True
    interlayer_mode: str = "baseline"  # "baseline" or "stronger"
    registration: str = "AB"

@dataclass
class BuildKnobs:
    m: int = 18
    r: int = 1
    theta_target_deg: float = 1.8
    hp_values: List[float] = field(default_factory=lambda: [0.9])
    n_list: List[int] = field(default_factory=lambda: [10,11,12,13,14,15,16,17,18,19])

@dataclass
class WindowKnobs:
    E_window: Tuple[float, float] = (-0.022, -0.006)
    sigmas: List[float] = field(default_factory=lambda: [-0.018, -0.015, -0.011, -0.007])
    k_per_slice: int = 30
    n_states_target: int = 100

@dataclass
class Paths:
    save_dir: str = r"C:\Users\hol1brg\OneDrive - Bosch Group\DAILY\tBLG\bands"

@dataclass
class Config:
    tb: TBParams = field(default_factory=TBParams)
    build: BuildKnobs = field(default_factory=BuildKnobs)
    window: WindowKnobs = field(default_factory=WindowKnobs)
    paths: Paths = field(default_factory=Paths)
