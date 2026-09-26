# -*- coding: utf-8 -*-
"""해석 한 종류 = 함수 하나. 시뮬레이터를 만들고 스윕까지 돌려 곡선 표를 돌려준다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Device
from .simulator import MosfetSimulator


def run_idvg(device: Device, drain_v: float = 0.05, start_v: float = 0.0,
             stop_v: float = 2.0, step_v: float = 0.1) -> pd.DataFrame:
    sim = MosfetSimulator(device, name="idvg")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()
    return sim.sweep_idvg(drain_v, start_v, stop_v, step_v)


def run_idvd(device: Device, gate_v: float = 2.0, start_v: float = 0.0,
             stop_v: float = 2.0, step_v: float = 0.1) -> pd.DataFrame:
    """[과제] run_idvg를 참고해 Id-Vd 해석을 완성하라."""

    # Creat and build Simulator, solve equilibrium, enable transport and 
    # sweep through vd from starv_v to stop_v with step step_v to calculcate Id
    sim = MosfetSimulator(device, name="idvd")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()
    return sim.sweep_idvd(gate_v, start_v, stop_v, step_v)


def run_cv(device: Device, start_v: float = -1.0, stop_v: float = 2.0,
           step_v: float = 0.1) -> pd.DataFrame:
    """[과제] C-V는 전류가 필요 없다 — enable_transport()를 부르지 않는 이유를 생각해 보라."""
    
    # Creat and build Simulator, solve equilibrium, DO NOT enable transport and 
    # sweep through vg from starv_v to stop_v with step step_v to calculate Cgg
    sim = MosfetSimulator(device, name="cv")
    sim.build()
    sim.solve_equilibrium()
    return sim.sweep_cv(start_v, stop_v, step_v)


def save_csv(path: str | Path, curve: pd.DataFrame) -> None:
    curve.to_csv(path, index=False)
