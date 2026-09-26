# -*- coding: utf-8 -*-
"""[1단계 대표 예제] 인스턴스 3개를 만들어 한 그래프에서 비교한다.

같은 클래스라도 인스턴스마다 값이 다르면 곡선이 달라진다 — 이게 클래스를 쓰는 이유다.
실제 공정에도 문턱 전압이 다른 LVT / RVT / HVT 트랜지스터가 따로 존재한다.
실행: STEP1_COMPARE.bat 더블클릭
"""

import numpy as np
import pandas as pd
import plotly.express as px

from simulator_practice import SimpleMosfet   # 1단계 클래스를 import해서 재사용

voltages = np.arange(0.0, 2.0 + 1e-9, 0.05)

# 문턱 전압이 다른 소자 3종 (Low / Regular / High Vth)
devices = {
    "LVT (Vth 0.4 V)": SimpleMosfet(vth=0.4),
    "RVT (Vth 0.6 V)": SimpleMosfet(vth=0.6),
    "HVT (Vth 0.8 V)": SimpleMosfet(vth=0.8),
}

curves = []
for name, mosfet in devices.items():          # 인스턴스 3개를 같은 루프로 돌린다
    # Changed to sweep_idvd
    curve = mosfet.sweep_idvd(vg=2.0, voltages=voltages)
    
    curve["device"] = name                    # 어느 소자의 결과인지 열로 표시
    curves.append(curve)

result = pd.concat(curves)                    # 표 3개를 하나로 합친다
result.to_csv("compare_step1.csv", index=False)
print(result.groupby("device")["Id_A_per_um"].max())  # 소자별 최대 전류 요약
print("saved: compare_step1.csv")

# Changed units
fig = px.line(result, x="Vd_V", y="Id_A_per_um", color="device", markers=True,
              title="LVT / RVT / HVT 비교 (Vg = 2 V)",
              labels={"Vd_V": "Vd (V)", "Id_A_per_um": "Id (A/µm)", "device": "소자"})
fig.show()   # 곡선 3개가 색깔별로 한 그래프에 겹쳐 나온다
