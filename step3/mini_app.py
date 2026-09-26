# -*- coding: utf-8 -*-
"""[3단계 예제] 간이 시뮬레이터(1단계)를 GUI 툴로 포장한다.

Device 1 / Device 2 를 나란히 설정하고 Run 버튼을 누르면 두 곡선이 비교된다.
DEMO와 같은 방식이다: 버튼을 눌렀을 때만 계산하고, 결과는 st.session_state에
저장해 두어 재실행(슬라이더 조작 등) 후에도 그래프가 사라지지 않는다.
실행: STEP3_MINIAPP.bat 더블클릭
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# 1단계에서 만든 클래스를 재사용한다 — step1 폴더를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "step1"))
from simulator_practice import SimpleMosfet

st.set_page_config(page_title="Mini MOSFET Compare", page_icon="🔬", layout="wide")
st.title("Mini MOSFET Compare")
st.caption("square-law 간이 모델 · 값을 설정한 뒤 Run simulation을 누르세요")

col1, col2 = st.columns(2)


def device_panel(column, label, default_vth):
    """한쪽 컬럼에 소자 설정 패널을 만들고 SimpleMosfet 인스턴스를 돌려준다."""
    with column:
        st.subheader(label)
        vth = st.slider("Vth (V)", 0.2, 1.0, default_vth, 0.05, key=f"{label}-vth")
        k_ua = st.slider("k (µA/V²)", 50, 500, 200, 10, key=f"{label}-k")
    return SimpleMosfet(vth=vth, k=k_ua * 1.0e-6)


device1 = device_panel(col1, "Device 1", 0.4)
device2 = device_panel(col2, "Device 2", 0.8)

# 버튼을 눌렀을 때만 계산한다 — TCAD처럼 계산이 느려도 쓸 수 있는 방식(DEMO와 동일).
run_clicked = st.button("Run simulation", type="primary")

if run_clicked:
    voltages = np.arange(0.0, 2.0 + 1e-9, 0.05)
    curves = []
    for name, mosfet in {"Device 1": device1, "Device 2": device2}.items():
        curve = mosfet.sweep_idvg(vd=2.0, voltages=voltages)
        curve["device"] = name
        curves.append(curve)
    # 결과를 session_state에 저장 — 다음 재실행에서도 그래프가 사라지지 않는다
    st.session_state["result"] = pd.concat(curves)

result = st.session_state.get("result")
if result is not None:
    fig = px.line(result, x="Vg_V", y="Id_A_per_um", color="device",
                  labels={"Vg_V": "Vg (V)", "Id_A_per_um": "Id (A/µm)", "device": ""})
    fig.update_layout(height=420, margin=dict(l=30, r=20, t=20, b=30))
    st.plotly_chart(fig, width="stretch")
