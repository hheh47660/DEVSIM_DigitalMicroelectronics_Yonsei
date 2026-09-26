# -*- coding: utf-8 -*-
"""[모델 비교] 같은 소자를 square-law 모델과 실제 TCAD(DEVSIM) 모델로 각각 계산해서
   한 그래프에 겹쳐 그린다.

STEP1(compare_step1.py)의 SimpleMosfet과 STEP2(compare_tcad.py)의 DEVSIM 워크플로를
같은 스크립트에서 함께 사용한다. 두 모델 모두 sweep_idvg()를 호출한다.

전제 조건: SimpleMosfet에 sweep_idvg(vd, voltages) 메서드가 있어야 한다.
           없다면 이 파일 하단의 주석 처리된 코드를 simulator_practice.py에 추가할 것.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from step1.simulator_practice import SimpleMosfet            # 1단계: square-law 모델
from step2.mosfet_tool.config import load_config              # 2단계: 실제 소자 설정
from step2.mosfet_tool.workflows import run_idvg               # 2단계: DEVSIM TCAD 워크플로

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# 1) TCAD 실제 물리 모델 — config.yaml의 기본 소자(base device) 사용
# ------------------------------------------------------------------
base_device, sweeps = load_config(ROOT / "step2/config.yaml")
idvg_settings = sweeps.get("idvg", {})   # drain_v, start_v, stop_v, step_v

start_v = 0.0
stop_v = 2.0
drain_v = 0.05
step_v = 0.01

print("===== TCAD (DEVSIM) 시뮬레이션 =====", flush=True)
curve_tcad = run_idvg(base_device, start_v=start_v, stop_v=stop_v, drain_v=drain_v, step_v=step_v)
curve_tcad["model"] = "TCAD (DEVSIM)"

# ------------------------------------------------------------------
# 2) Square-law 모델 — "같은 소자"를 근사하기 위해 TCAD 곡선에서
#    추출한 Vth를 사용한다 (extract_vth.py 참고: tox=10nm 소자는 Vth ≈ 0.63 V)
# ------------------------------------------------------------------
VTH_MATCHED = 0.6295   # TODO: 다른 base_device를 쓴다면 이 값을 다시 추출해서 갱신할 것
K_MATCHED = 1.4e-4 


voltages = np.arange(start_v, stop_v + 1e-9, step_v)

print("===== Square-law (SimpleMosfet) =====", flush=True)
simple_mosfet = SimpleMosfet(vth=VTH_MATCHED, k=K_MATCHED)
curve_simple = simple_mosfet.sweep_idvg(vd=drain_v, voltages=voltages)
curve_simple["model"] = "Square-law (SimpleMosfet)"

# ------------------------------------------------------------------
# 3) 합쳐서 하나의 그래프에 겹쳐 그리기
# ------------------------------------------------------------------
result = pd.concat([
    curve_tcad[["Vg_V", "Id_A_per_um", "model"]],
    curve_simple[["Vg_V", "Id_A_per_um", "model"]],
])
result.to_csv("compare_models.csv", index=False)
print("saved: compare_models.csv")

fig = px.line(result, x="Vg_V", y="Id_A_per_um", color="model", markers=True,
              title=f"Square-law vs TCAD (Vd = {drain_v} V, Vth_matched = {VTH_MATCHED} V)",
              labels={"Vg_V": "Vg (V)", "Id_A_per_um": "Id (A/µm)", "model": "모델"})
fig.show()
