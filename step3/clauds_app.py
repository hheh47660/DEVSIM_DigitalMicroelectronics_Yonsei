# -*- coding: utf-8 -*-
"""[3단계 완성본] DEVSIM 2D MOSFET GUI 데모.

Device 1 / Device 2 를 사이드바에서 나란히 설정하고, 사이드바의 Analysis 패널에서
Sweep 종류(Id-Vg / Id-Vd / C-V)를 고른 뒤 Run simulation을 누르면 두 소자의 단면
구조와 곡선(선형 + 로그), 그리고 결과 표가 함께 표시된다.
DEVSIM 계산은 소자당 수십 초가 걸릴 수 있으므로, 버튼을 눌렀을 때만 계산하고
결과는 st.session_state에 저장해 재실행(슬라이더 조작 등) 후에도 유지한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# step2 폴더의 mosfet_tool 패키지를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "step2"))
from mosfet_tool.config import Device          # noqa: E402  (필드명은 config.yaml과 동일하다고 가정)
from mosfet_tool.simulator import MosfetSimulator  # noqa: E402

st.set_page_config(page_title="DEVSIM 2D MOSFET", page_icon="🔬", layout="wide")

st.title("DEVSIM 2D MOSFET — 완성본 데모")
st.caption("여러분이 3단계에서 직접 만들게 될 툴의 완성 예시입니다. Id–Vg / Id–Vd / C–V + 소자 비교")

# ------------------------------------------------------------------
# 사이드바: 소자 파라미터 입력
# ------------------------------------------------------------------

DEFAULTS = dict(
    gate_length_um=1.0,
    source_length_um=0.5,
    drain_length_um=0.5,
    oxide_thickness_nm=10.0,
    silicon_thickness_um=0.5,
    junction_depth_um=0.1,
    log_body_doping=16.0,
    log_sd_doping=19.0,
    temperature_k=300.0,
)


def device_sidebar_panel(sidebar, label: str, expanded: bool) -> Device:
    """사이드바에 소자 파라미터 입력 패널을 그리고 Device 객체를 돌려준다."""
    with sidebar.expander(label, expanded=expanded):
        gate_length = st.number_input(
            "Gate length (µm)", 0.05, 5.0, DEFAULTS["gate_length_um"], 0.05,
            key=f"{label}-gate")
        source_length = st.number_input(
            "Source length (µm)", 0.05, 5.0, DEFAULTS["source_length_um"], 0.05,
            key=f"{label}-source")
        drain_length = st.number_input(
            "Drain length (µm)", 0.05, 5.0, DEFAULTS["drain_length_um"], 0.05,
            key=f"{label}-drain")
        oxide_thickness = st.number_input(
            "Oxide thickness (nm)", 1.0, 50.0, DEFAULTS["oxide_thickness_nm"], 0.5,
            key=f"{label}-tox")
        silicon_thickness = st.number_input(
            "Silicon thickness (µm)", 0.1, 5.0, DEFAULTS["silicon_thickness_um"], 0.05,
            key=f"{label}-tsi")
        junction_depth = st.number_input(
            "Junction depth (µm)", 0.02, 1.0, DEFAULTS["junction_depth_um"], 0.01,
            key=f"{label}-xj")
        log_body = st.slider(
            "log10 body doping (cm⁻³)", 14.0, 19.0, DEFAULTS["log_body_doping"], 0.25,
            key=f"{label}-Na")
        log_sd = st.slider(
            "log10 S/D doping (cm⁻³)", 17.0, 21.0, DEFAULTS["log_sd_doping"], 0.25,
            key=f"{label}-Nd")
        temperature = st.number_input(
            "Temperature (K)", 200.0, 450.0, DEFAULTS["temperature_k"], 5.0,
            key=f"{label}-T")

    return Device(
        gate_length_um=gate_length,
        source_length_um=source_length,
        drain_length_um=drain_length,
        oxide_thickness_nm=oxide_thickness,
        silicon_thickness_um=silicon_thickness,
        junction_depth_um=junction_depth,
        body_doping_cm3=10.0 ** log_body,
        sd_doping_cm3=10.0 ** log_sd,
        temperature_k=temperature,
    )


sb = st.sidebar

device1 = device_sidebar_panel(sb, "Device 1", expanded=False)
compare_enabled = sb.checkbox("Device 2와 비교", value=True)
device2 = device_sidebar_panel(sb, "Device 2", expanded=compare_enabled) if compare_enabled else None

# ------------------------------------------------------------------
# 사이드바: Analysis 패널 (Sweep 종류 선택 + Run 버튼)
# ------------------------------------------------------------------

VG_START, VG_STOP, VG_STEP = 0.0, 2.0, 0.05
VD_START, VD_STOP, VD_STEP = 0.0, 2.0, 0.05

with sb:
    st.header("Analysis")
    sweep_type = st.selectbox("Sweep", ["Id–Vg", "Id–Vd", "C–V"], key="sweep_type")

    fixed_bias = None
    if sweep_type == "Id–Vg":
        fixed_bias = st.number_input("Fixed Vd (V)", 0.0, 5.0, 0.05, 0.05, key="idvg_vd_fixed")
    elif sweep_type == "Id–Vd":
        fixed_bias = st.number_input("Fixed Vg (V)", 0.0, 5.0, 1.0, 0.05, key="idvd_vg_fixed")
    # C-V는 Vg만 스윕하므로 추가 입력이 필요 없다 (source/drain은 0V로 고정)

    run_clicked = st.button("Run simulation", type="primary")


# ------------------------------------------------------------------
# 소자 단면 구조 그림 (실제 build() 없이 기하 정보만으로 빠르게 그린다)
# ------------------------------------------------------------------

def structure_figure(device: Device) -> go.Figure:
    """Device 파라미터로부터 게이트/산화막/소스-드레인 단면도를 그린다."""
    x_gate_left = device.source_length_um
    x_gate_right = x_gate_left + device.gate_length_um
    x_right = x_gate_right + device.drain_length_um
    y_oxide_top = -device.oxide_thickness_nm * 1.0e-3   # nm -> µm, 표면 위쪽은 음수
    y_bottom = device.silicon_thickness_um
    xj = device.junction_depth_um
    gate_h = max(0.03, device.oxide_thickness_nm * 1.0e-3 * 2)

    fig = go.Figure()

    # 실리콘 바디 (연한 파란색)
    fig.add_shape(type="rect", x0=0, x1=x_right, y0=0, y1=y_bottom,
                  fillcolor="#a9d6e5", line=dict(width=1, color="#1f6f8b"), layer="below")
    # 소스 / 드레인 (표면 근처 얕은 빨간 영역)
    fig.add_shape(type="rect", x0=0, x1=x_gate_left, y0=0, y1=xj,
                  fillcolor="#e05263", line=dict(width=0))
    fig.add_shape(type="rect", x0=x_gate_right, x1=x_right, y0=0, y1=xj,
                  fillcolor="#e05263", line=dict(width=0))
    # 산화막 (얇은 주황색 띠)
    fig.add_shape(type="rect", x0=x_gate_left, x1=x_gate_right, y0=y_oxide_top, y1=0,
                  fillcolor="#f6c85f", line=dict(width=0))
    # 게이트 (짙은 회색)
    fig.add_shape(type="rect", x0=x_gate_left, x1=x_gate_right,
                  y0=y_oxide_top - gate_h, y1=y_oxide_top,
                  fillcolor="#333333", line=dict(width=0))

    fig.update_xaxes(title="x (µm)", range=[0, max(2.0, x_right)])
    # y=0을 위쪽에, 실리콘 깊이가 아래로 갈수록 커지도록 축을 뒤집는다
    fig.update_yaxes(title="y (µm)", range=[y_bottom + 0.05, y_oxide_top - gate_h - 0.05])
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=20, b=40), showlegend=False)
    return fig


struct_col1, struct_col2 = st.columns(2)
with struct_col1:
    st.subheader("Device 1 구조")
    st.plotly_chart(structure_figure(device1), width="stretch", key="struct_1")
if device2 is not None:
    with struct_col2:
        st.subheader("Device 2 구조")
        st.plotly_chart(structure_figure(device2), width="stretch", key="struct_2")


# ------------------------------------------------------------------
# 시뮬레이션 실행 (DEVSIM, 버튼을 눌렀을 때만 계산 — 느린 계산이므로)
# ------------------------------------------------------------------

def run_idvg_curve(device: Device, label: str, vd_fixed: float) -> pd.DataFrame:
    sim = MosfetSimulator(device, name=f"gui_{label.replace(' ', '_')}_idvg")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()
    curve = sim.sweep_idvg(vd=vd_fixed, start=VG_START, stop=VG_STOP, step=VG_STEP)
    curve["device"] = label
    return curve


def run_idvd_curve(device: Device, label: str, vg_fixed: float) -> pd.DataFrame:
    sim = MosfetSimulator(device, name=f"gui_{label.replace(' ', '_')}_idvd")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()
    curve = sim.sweep_idvd(vg=vg_fixed, start=VD_START, stop=VD_STOP, step=VD_STEP)
    curve["device"] = label
    return curve


def run_cv_curve(device: Device, label: str) -> pd.DataFrame:
    sim = MosfetSimulator(device, name=f"gui_{label.replace(' ', '_')}_cv")
    sim.build()
    sim.solve_equilibrium()   # C-V는 enable_transport() 없이 평형 상태에서 계산
    curve = sim.sweep_cv(start=VG_START, stop=VG_STOP, step=VG_STEP)
    curve["device"] = label
    return curve


if run_clicked:
    with st.spinner("DEVSIM 시뮬레이션 실행 중... (소자당 수십 초 소요)"):
        if sweep_type == "Id–Vg":
            curves = [run_idvg_curve(device1, "Device 1", fixed_bias)]
            if device2 is not None:
                curves.append(run_idvg_curve(device2, "Device 2", fixed_bias))
        elif sweep_type == "Id–Vd":
            curves = [run_idvd_curve(device1, "Device 1", fixed_bias)]
            if device2 is not None:
                curves.append(run_idvd_curve(device2, "Device 2", fixed_bias))
        else:  # C–V
            curves = [run_cv_curve(device1, "Device 1")]
            if device2 is not None:
                curves.append(run_cv_curve(device2, "Device 2"))
    st.session_state["result"] = pd.concat(curves).reset_index(drop=True)
    st.session_state["result_type"] = sweep_type

result = st.session_state.get("result")
result_type = st.session_state.get("result_type")

# ------------------------------------------------------------------
# 결과: 그래프 + 데이터 표
# ------------------------------------------------------------------

st.header(result_type if result_type else "결과")

if result is not None:
    if result_type == "Id–Vg":
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            fig_lin = px.line(result, x="Vg_V", y="Id_A_per_um", color="device",
                               labels={"Vg_V": "Vg (V)", "Id_A_per_um": "Id (A/µm)", "device": ""})
            fig_lin.update_layout(height=420, margin=dict(l=30, r=20, t=20, b=30))
            st.plotly_chart(fig_lin, width="stretch", key="chart_a")
        with plot_col2:
            log_df = result.copy()
            log_df["Id_abs"] = log_df["Id_A_per_um"].abs()
            fig_log = px.line(log_df, x="Vg_V", y="Id_abs", color="device", log_y=True,
                               labels={"Vg_V": "Vg (V)", "Id_abs": "|Id| (A/µm)", "device": ""})
            fig_log.update_layout(height=420, margin=dict(l=30, r=20, t=20, b=30))
            st.plotly_chart(fig_log, width="stretch", key="chart_b")

    elif result_type == "Id–Vd":
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            fig_lin = px.line(result, x="Vd_V", y="Id_A_per_um", color="device",
                               labels={"Vd_V": "Vd (V)", "Id_A_per_um": "Id (A/µm)", "device": ""})
            fig_lin.update_layout(height=420, margin=dict(l=30, r=20, t=20, b=30))
            st.plotly_chart(fig_lin, width="stretch", key="chart_a")
        with plot_col2:
            log_df = result.copy()
            log_df["Id_abs"] = log_df["Id_A_per_um"].abs()
            fig_log = px.line(log_df, x="Vd_V", y="Id_abs", color="device", log_y=True,
                               labels={"Vd_V": "Vd (V)", "Id_abs": "|Id| (A/µm)", "device": ""})
            fig_log.update_layout(height=420, margin=dict(l=30, r=20, t=20, b=30))
            st.plotly_chart(fig_log, width="stretch", key="chart_b")

    else:  # C–V
        fig_cv = px.line(result, x="Vg_V", y="Cgg_F_per_um", color="device",
                          labels={"Vg_V": "Vg (V)", "Cgg_F_per_um": "Cgg (F/µm)", "device": ""})
        fig_cv.update_layout(height=420, margin=dict(l=30, r=20, t=20, b=30))
        st.plotly_chart(fig_cv, width="stretch", key="chart_a")

    st.subheader("Data")
    st.dataframe(result, width="stretch", key="results_table")
    st.download_button(
        "Download CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"{result_type.replace('–', '-')}_result.csv",
        mime="text/csv",
        key="download_csv",
    )

else:
    st.info("왼쪽에서 소자와 Sweep 종류를 설정하고 'Run simulation'을 눌러주세요.")