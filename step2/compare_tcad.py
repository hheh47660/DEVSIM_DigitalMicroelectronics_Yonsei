# -*- coding: utf-8 -*-
"""[2단계 대표 예제] config.yaml의 compare: 항목에 적힌 소자들을 비교한다.

소자 옵션(config)만 바꾸면 진짜 TCAD 곡선이 한 그래프에 겹쳐 나온다.
1단계 compare.py와 코드 모양이 같다 — 공식 모델이 진짜 물리로 바뀌었을 뿐이다.
실행: STEP2_COMPARE.bat 더블클릭 (소자당 수십 초, 전체 몇 분)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import plotly.express as px
import yaml

from mosfet_tool.config import load_config
from mosfet_tool.workflows import run_idvg, run_idvd, run_cv

ROOT = Path(__file__).resolve().parent

base_device, sweeps = load_config(ROOT / "config.yaml")
with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
    variants = yaml.safe_load(handle).get("compare", {})

curves = []
for name, overrides in variants.items():
    # 기본 소자에서 compare:에 적힌 옵션만 덮어쓴 새 소자를 만든다
    device = dataclasses.replace(base_device, **overrides)
    print(f"\n===== {name} 시뮬레이션 =====", flush=True)
    
    # I execute run_idvg, runvd and run_cv and add curves 
    # to a single array 
    curve_idvg = run_idvg(device, **sweeps.get("idvg", {}))
    curve_idvd = run_idvd(device, **sweeps.get("idvd", {}))
    curve_cv = run_cv(device, **sweeps.get("cv", {}))
    
    curve_idvg["device"] = name
    curve_idvd["device"] = name
    curve_cv["device"] = name
    
    curves.append(curve_idvg)
    curves.append(curve_idvd)
    curves.append(curve_cv)

result = pd.concat(curves)
result.to_csv("compare_tcad.csv", index=False)
print("\n소자별 최대 전류 [A/um]:")
print(result.groupby("device")["Id_A_per_um"].max())

print("saved: compare_tcad.csv")

fig = px.line(result, x="Vg_V", y="Id_A_per_um", color="device", markers=True,
              title="TCAD Id-Vg 비교 (config.yaml의 compare: 항목)",
              labels={"Vg_V": "Vg (V)", "Id_A_per_um": "Id (A/µm)", "device": "소자"})
fig.show()


# Create a Figure also for idvd and cv
fig = px.line(result, x="Vd_V", y="Id_A_per_um", color="device", markers=True,
              title="TCAD Id-Vd 비교 (config.yaml의 compare: 항목)",
              labels={"Vd_V": "Vd (V)", "Id_A_per_um": "Id (A/µm)", "device": "소자"})
fig.show()

fig = px.line(result, x="Vg_V", y="Cgg_F_per_um", color="device", markers=True,
              title="TCAD Cgg-Vg 비교 (config.yaml의 compare: 항목)",
              labels={"Vg_V": "Vg (V)", "Cgg_F_per_um": "Cgg (F/µm)", "device": "소자"})
fig.show()