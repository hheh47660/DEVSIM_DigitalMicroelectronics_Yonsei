# -*- coding: utf-8 -*-
"""[1단계] 파이썬 객체지향으로 만드는 간이 MOSFET 시뮬레이터.

교과서 square-law 공식으로 전류를 계산하는 클래스다. DEVSIM은 아직 안 쓴다.
여기서 익힌 뼈대(클래스 + numpy + pandas + plotly)가 2·3단계에서 그대로 확장된다.
실행: STEP1_PRACTICE.bat 더블클릭
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1) numpy: 숫자 배열을 한 번에 다루기 ----------------------------------
voltages = np.arange(0.0, 2.0 + 1e-9, 0.25)   # 0 ~ 2 V, 0.25 V 간격 배열
print("전압 배열   :", voltages)
print("배열 연산   :", voltages * 2)           # 반복문 없이 모든 원소에 한 번에 적용
print("일부만 선택 :", voltages[voltages > 1.0])


# --- 2) 클래스와 메서드: 교과서 square-law nMOS 모델 ------------------------
class SimpleMosfet:
    """긴 채널 근사의 교과서 모델. 진짜 시뮬레이터(2단계)와 같은 뼈대다.

    전류 단위는 A/µm (transistor width W = 1 µm 기준) — 2단계 DEVSIM 결과와 같은 단위다.
    """

    def __init__(self, vth=0.6, k=1.4e-4):
        self.vth = vth  # 문턱 전압 [V]
        self.k = k      # 이득 계수 µn·Cox·W/L, transistor width W=1µm 기준 [A/V²]
        # 기본값 1.4e-4는 2단계 DEVSIM 소자(tox 10nm, L 1µm, µn 400)로 계산한 값

    def current(self, vg, vd):
        """한 바이어스 점의 드레인 전류 [A/µm]."""
        vov = vg - self.vth
        if vov <= 0:
            return 0.0                            # 차단 영역
        if vd < vov:
            return self.k * (vov - vd / 2) * vd   # 선형 영역
        return 0.5 * self.k * vov * vov           # 포화 영역

    def sweep_idvg(self, vd, voltages):
        """전압 배열을 받아 (Vg, Id) 표를 돌려준다."""
        ids = [self.current(vg, vd) for vg in voltages]
        return pd.DataFrame({"Vg_V": voltages, "Id_A_per_um": ids})

    # [과제] sweep_idvd(self, vg, voltages)를 직접 추가해 보자.
    def sweep_idvd(self, vg, voltages):
        # I run self.current going through all voltages passed and fixing vg
        ids = [self.current(vg, vd) for vd in voltages]
        return pd.DataFrame({"Vd_V": voltages, "Id_A_per_um": ids})


def multiple_vth_plot(vths, vg=2.0):

    # Create an empty figure to which we can add curves
    fig = go.Figure()
    fig.update_layout(
        title=f'Id-Vd (square-law model, Vg = {vg} V)',
        xaxis_title='Vd (V)',
        yaxis_title='Id (A/µm)',
    )

    # We itereate over vths values, sweep thorugh voltages values and plot
    for vth in vths:
        mosfet = SimpleMosfet(vth=vth, k=1.4e-4)
        curve_vd = mosfet.sweep_idvd(vg=vg, voltages=voltages)

        fig.add_trace(go.Scatter(
            x=curve_vd["Vd_V"],
            y=curve_vd["Id_A_per_um"],
            mode='lines+markers',
            name=f'Vth {vth}V'
        ))
    fig.show()




if __name__ == "__main__":   # import될 때는 실행되지 않는 부분 (compare.py가 import한다)
    mosfet = SimpleMosfet(vth=0.6, k=1.4e-4)      # 인스턴스 생성 (틀 → 붕어빵)
    print(f"\n단일 점: Vg=1.0 V, Vd=2.0 V  ->  Id = {mosfet.current(1.0, 2.0):.3e} A/um")

    # --- 3) pandas: 결과를 표(DataFrame)로 정리하고 저장하기 -----------------

    # mod: added curve_vd and changed curve -> curve_vg to distingue
    curve_vg = mosfet.sweep_idvg(vd=0.05, voltages=voltages)

    curve_vd_medium_vg = mosfet.sweep_idvd(vg=2.0, voltages=voltages)

    print("\n== Id-Vg (Vd = 0.05 V) ==")
    print(curve_vg.to_string(index=False))
    print("\n최대 전류  :", curve_vg["Id_A_per_um"].max(), "A/um")
    curve_vg.to_csv("practice_idvg.csv", index=False)
    print("saved: practice_idvg.csv")

    # --- 4) plotly: 표를 그래프로 --------------------------------------------
    fig = px.line(curve_vg, x="Vg_V", y="Id_A_per_um", markers=True,
                  title="Id-Vg (square-law model, Vd = 0.05 V)",
                  labels={"Vg_V": "Vg (V)", "Id_A_per_um": "Id (A/µm)"})
    fig.show()   # 기본 브라우저에 열린다


    # mod: showing curve_vd as well
    # fig = px.line(curve_vd_medium_vg, x="Vd_V", y="Id_A_per_um", markers=True,
    #               title="Id-Vd (square-law model, Vg = 2 V)",
    #               labels={"Vd_V": "Vd (V)", "Id_A_per_um": "Id (A/µm)"})


    # fig.show()   # 기본 브라우저에 열린다

    
    # Delegated Vth plotting to multiple_vth_plot()
    # I'm using as vths [0.4V, 0.6V, 0.8V]
    # Vg will be 2.0V for all simulated devices

    #multiple_vth_plot(vg=2.0, vths=[0.4, 0.6, 0.8])



