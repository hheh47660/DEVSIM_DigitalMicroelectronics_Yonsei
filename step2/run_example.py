# -*- coding: utf-8 -*-
"""시뮬레이터 클래스를 직접 불러와 쓰는 최소 예시.

mosfet.py(CLI)를 거치지 않고 MosfetSimulator를 바로 다룬다.
클래스 호출 순서를 눈으로 확인하고 싶을 때 실행해 볼 것.
"""

from mosfet_tool.config import Device
from mosfet_tool.simulator import MosfetSimulator
from mosfet_tool.workflows import run_cv, run_idvd
import plotly.express as px

device = Device()  # 기본값 사용. Device(gate_length_um=0.5)처럼 바꿀 수 있다.

sim = MosfetSimulator(device, name="example")


#curve = run_cv(device, start_v=0.0, stop_v=2.0, step_v=0.1)
curve = run_cv(device, start_v=-3.0, stop_v=2.0, step_v=0.1)

fig = px.line(curve, x="Vg_V", y="Cgg_F_per_um", markers=True,
              title="TCAD Cgg-Vg (DEFAULT)",
              labels={"Vg_V": "Vg (V)", "Cgg_F_per_um": "Cgg (F/µm)", "device": "Default"})
# fig = px.line(curve, x="Vd_V", y="Id_A_per_um", markers=True,
#               title="TCAD Id-Vd (DEFAULT) Vg=2.0V",
#               labels={"Vd_V": "Vd (V)", "Id_A_per_um": "Id (A/µm)", "device": "Default"})

fig.show()

print("\n== Id-Vg (Vd = 0.05 V) ==")
print(curve.to_string(index=False))
