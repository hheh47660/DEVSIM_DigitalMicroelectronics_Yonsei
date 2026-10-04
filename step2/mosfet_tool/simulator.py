# -*- coding: utf-8 -*-
"""DEVSIM 2-D nMOSFET 라이트 시뮬레이터 (수업용).

원본 프로젝트의 simulator.py에서 I-V / C-V 계산에 꼭 필요한 기능만 남겼다.
물리 방정식 자체는 이 파일에 없다 — devsim.python_packages의 헬퍼가
문자열 수식으로 조립한다 (MANUAL.html 6절 참고).
"""

from __future__ import annotations

import math
from pathlib import Path

import devsim
import numpy as np
import pandas as pd
from devsim.python_packages.model_create import CreateSolution
from devsim.python_packages.simple_physics import (
    CreateOxideContact,
    CreateOxidePotentialOnly,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconOxideInterface,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    GetContactBiasName,
    SetOxideParameters,
    SetSiliconParameters,
    GetContactNodeModelName,
    CreateContactNodeModel,
    CreateContactNodeModelDerivative,
    InEdgeModelList, 
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
)

from .config import Device

UM = 1.0e-4  # 1 µm in cm (DEVSIM 내부 단위는 cm)
NM = 1.0e-7  # 1 nm in cm

# Project 1 : Boltzmann Constant
KB_EV = 8.617333262145e-5

# Project 1 : Add gate work function table for different materials
PHI_M_TABLE = {
    "n+ poly-Si": 4.05, "Al": 4.10, "Ta": 4.25, "Ti": 4.33,
    "TaN": 4.45, "W": 4.60, "TiN": 4.65, "Mo": 4.70,
    "Ni": 5.10, "p+ poly-Si": 5.15, "Pt": 5.30,
}

def voltage_points(start: float, stop: float, step: float) -> np.ndarray:
    """start에서 stop까지 step 간격의 전압 배열을 만든다 (practice.py의 np.arange 참고)."""
    count = round(abs(stop - start) / step)
    signed_step = step if stop >= start else -step
    return np.round(start + signed_step * np.arange(count + 1), 9)


class MosfetSimulator:
    """2-D planar nMOS drift-diffusion solver.

    호출 순서: build() → solve_equilibrium() → [enable_transport()] → sweep_*()
    C-V만 계산할 때는 enable_transport()를 건너뛴다.
    """

    MU_N = 400.0  # electron mobility [cm^2/V·s]
    MU_P = 200.0  # hole mobility [cm^2/V·s]
    DX_CHANNEL = 25.0 * NM
    DY_OXIDE = 2.5 * NM
    DY_JUNCTION = 10.0 * NM
    DY_BULK = 50.0 * NM
    RAMP_STEP_V = 0.1

    BODY_TAP_WIDTH_UM = 0.1  # Project 1 : Width of the p+ body tap region in µm

    def __init__(self, device: Device, name: str = "mos_light"):
        self.dev = device
        self.name = name
        self.mesh = f"{name}_mesh"

        # Project 1 : Add body p+ region
        self.x_body_right = self.BODY_TAP_WIDTH_UM * UM
        self.x_gate_left = self.x_body_right + device.source_length_um * UM
        self.x_gate_right = self.x_gate_left + device.gate_length_um * UM
        self.x_right = self.x_gate_right + device.drain_length_um * UM
        self.y_oxide_top = -device.oxide_thickness_nm * NM
        self.y_junction = device.junction_depth_um * UM
        self.y_bottom = device.silicon_thickness_um * UM       
        self.bias = {"gate": 0.0, "source": 0.0, "drain": 0.0, "body": 0.0}

        # Project 1 : Add gate_metal
        self.y_gate_top = self.y_oxide_top - 10.0 * NM
        self.silicon_gate_metal_name = device.silicon_gate_metal_name

    # ------------------------------------------------------------------
    # 1) 구조 만들기
    # ------------------------------------------------------------------
    def build(self) -> None:
        """메시 → 영역/접점 → 도핑 → Poisson 방정식 등록까지 수행한다."""
        self._clear_session()
        self._build_mesh()
        self._build_doping()

        # Project 1 : Save device design
        design_file = Path(__file__).resolve().parents[2] / "project1" / "part1_2026848368.devsim"
        devsim.write_devices(file=str(design_file), device=self.name, type="devsim")

        self._build_physics()

    def _clear_session(self) -> None:
        # DEVSIM의 solve는 프로세스 안의 모든 device를 함께 진행시키므로,
        # 새 시뮬레이션을 시작하기 전에 이전 device를 지운다.
        for device in tuple(devsim.get_device_list()):
            devsim.delete_device(device=device)
        for mesh in tuple(devsim.get_mesh_list()):
            devsim.delete_mesh(mesh=mesh)

    def _build_mesh(self) -> None:
        pad = 1.0e-8
        x_min, x_max = -pad, self.x_right + pad
        y_min, y_max = self.y_oxide_top - pad, self.y_bottom + pad

        devsim.create_2d_mesh(mesh=self.mesh)
        for pos in (x_min, 0.0, self.x_body_right, self.x_gate_left, self.x_gate_right, self.x_right, x_max):
            devsim.add_2d_mesh_line(mesh=self.mesh, dir="x", pos=pos, ps=self.DX_CHANNEL)
        for pos, spacing in (
            (y_min, self.DY_BULK),
            (self.y_gate_top, self.DY_OXIDE), # Project 1: Add gate_material
            (self.y_oxide_top, self.DY_OXIDE),
            (0.0, min(self.DY_OXIDE, self.DY_JUNCTION)),
            (self.y_junction, self.DY_JUNCTION),
            (self.y_bottom, self.DY_BULK),
            (y_max, self.DY_BULK),
        ):
            devsim.add_2d_mesh_line(mesh=self.mesh, dir="y", pos=pos, ps=spacing)

        devsim.add_2d_region(mesh=self.mesh, material="Air", region="air")
        devsim.add_2d_region(mesh=self.mesh, material="Silicon", region="bulk",
                             xl=0.0, xh=self.x_right, yl=self.y_bottom, yh=0.0)
        devsim.add_2d_region(mesh=self.mesh, material="Oxide", region="oxide",
                             xl=self.x_gate_left, xh=self.x_gate_right,
                             yl=0.0, yh=self.y_oxide_top)

        # Project 1 : Add gate_material
        devsim.add_2d_region(mesh=self.mesh, material=self.silicon_gate_metal_name, 
                             region="gate_metal", xl=self.x_gate_left, xh=self.x_gate_right,
                             yl=self.y_oxide_top, yh=self.y_gate_top)

        devsim.add_2d_contact(mesh=self.mesh, name="gate", region="oxide", material="metal",
                              xl=self.x_gate_left, xh=self.x_gate_right,
                              yl=self.y_oxide_top, yh=self.y_oxide_top)
        

        # Project 1 : Body contact now is in the top-left corner
        devsim.add_2d_contact(mesh=self.mesh, name="body", region="bulk", material="metal",
                              xl=x_min, xh=self.x_body_right, yl=0.0, yh=0.0)
        devsim.add_2d_contact(mesh=self.mesh, name="source", region="bulk", material="metal",
                              xl=self.x_body_right, xh=self.x_gate_left, yl=0.0, yh=0.0)
        devsim.add_2d_contact(mesh=self.mesh, name="drain", region="bulk", material="metal",
                              xl=self.x_gate_right, xh=x_max, yl=0.0, yh=0.0)

        
        devsim.add_2d_interface(mesh=self.mesh, name="bulk_oxide",
                                region0="bulk", region1="oxide")
        devsim.finalize_mesh(mesh=self.mesh)
        devsim.create_device(mesh=self.mesh, device=self.name)

    def _build_doping(self) -> None:
        # n+ 소스/드레인 도핑을 erfc()로 부드럽게 감소시킨다 (수업에서 다룸).
        nd, na = self.dev.sd_doping_cm3, self.dev.body_doping_cm3
        decay_x = 0.5 * self.DX_CHANNEL
        decay_y = 0.5 * self.DY_JUNCTION
        devsim.node_model(
            device=self.name, region="bulk", name="SourceDoping",
            equation=(f"0.25*{nd:.6e}*erfc((x-{self.x_gate_left:.6e})/{decay_x:.6e})"
                      f"*erfc((y-{self.y_junction:.6e})/{decay_y:.6e})"),
        )
        devsim.node_model(
            device=self.name, region="bulk", name="DrainDoping",
            equation=(f"0.25*{nd:.6e}*erfc(-(x-{self.x_gate_right:.6e})/{decay_x:.6e})"
                      f"*erfc((y-{self.y_junction:.6e})/{decay_y:.6e})"),
        )

        # Project 1 : Add p+ body tap doping
        tap_doping = self.dev.body_tap_doping_cm3  
        devsim.node_model(
            device=self.name, region="bulk", name="BodyTapDoping",
            equation=(f"0.25*{tap_doping:.6e}*erfc((x-{self.x_body_right:.6e})/{decay_x:.6e})"
                    f"*erfc((y-{self.y_junction:.6e})/{decay_y:.6e})"),
        )
        devsim.node_model(
            device=self.name, region="bulk", name="NetDoping",
            equation=f"SourceDoping + DrainDoping - {na:.6e} - BodyTapDoping",
        )
    def _build_physics(self) -> None:
        # Project 1 : Add 128-bit extended precision for convergece
        devsim.set_parameter(name="extended_precision", value=True)
        devsim.set_parameter(name="extended_solver", value=True)
        devsim.set_parameter(name="extended_model", value=True)


        for region in ("bulk", "oxide"):
            CreateSolution(self.name, region, "Potential")
        SetSiliconParameters(self.name, "bulk", self.dev.temperature_k)
        # Project 1 : Note, you don't need to change these constants
        devsim.set_parameter(device=self.name, region="bulk", name="mu_n", value=self.MU_N)
        devsim.set_parameter(device=self.name, region="bulk", name="mu_p", value=self.MU_P)

        # Project 1: Add temperature dependent concentretion 
        ni_T = self._intrinsic_concentration(self.dev.temperature_k)
        for param_name in ("n_i", "n1", "p1"):
            devsim.set_parameter(device=self.name, region="bulk", name=param_name, value=ni_T)


        CreateSiliconPotentialOnly(self.name, "bulk")
        SetOxideParameters(self.name, "oxide", self.dev.temperature_k)
        CreateOxidePotentialOnly(self.name, "oxide", "log_damp")
        
        
        
        # Project 1: gate work function fix (replaces plain CreateOxideContact)
        self._create_gate_contact_with_workfunction()
        devsim.set_parameter(device=self.name, name=GetContactBiasName("gate"), value=0.0)
        
        for contact in ("source", "drain", "body"):
            CreateSiliconPotentialOnlyContact(self.name, "bulk", contact)
            devsim.set_parameter(device=self.name, name=GetContactBiasName(contact), value=0.0)
        CreateSiliconOxideInterface(self.name, "bulk_oxide")

    # ------------------------------------------------------------------
    # 2) 풀기
    # ------------------------------------------------------------------
    def solve_equilibrium(self) -> None:
        """모든 접점 0 V에서 Poisson을 수렴시킨다 (다음 단계의 초기값)."""
        devsim.solve(type="dc", absolute_error=1.0e-13, relative_error=1.0e-10,
                     maximum_iterations=80)

    def enable_transport(self) -> None:
        """전자/정공을 미지수로 추가하고 drift-diffusion 방정식을 켠다."""
        CreateSolution(self.name, "bulk", "Electrons")
        CreateSolution(self.name, "bulk", "Holes")
        devsim.set_node_values(device=self.name, region="bulk", name="Electrons",
                               init_from="IntrinsicElectrons")
        devsim.set_node_values(device=self.name, region="bulk", name="Holes",
                               init_from="IntrinsicHoles")
        CreateSiliconDriftDiffusion(self.name, "bulk", "mu_n", "mu_p")
        for contact in ("source", "drain", "body"):
            CreateSiliconDriftDiffusionAtContact(self.name, "bulk", contact)
        self._solve()

    def _solve(self) -> None:
        devsim.solve(type="dc", absolute_error=1.0e30, relative_error=1.0e-6,
                     maximum_iterations=80)

    def set_bias(self, contact: str, volts: float) -> None:
        """전압을 RAMP_STEP_V 간격으로 나눠 올리며 매 스텝 다시 푼다."""
        start = self.bias[contact]
        steps = max(1, math.ceil(abs(volts - start) / self.RAMP_STEP_V - 1.0e-9))
        for i in range(1, steps + 1):
            value = start + (volts - start) * i / steps
            devsim.set_parameter(device=self.name,
                                 name=GetContactBiasName(contact), value=value)
            self._solve()
        self.bias[contact] = volts

    # ------------------------------------------------------------------
    # 3) 물리량 추출
    # ------------------------------------------------------------------
    def drain_current(self) -> float:
        """드레인 접점의 전자+정공 전류 [A/µm] (2-D 해는 A/cm로 나옴)."""
        electron = devsim.get_contact_current(device=self.name, contact="drain",
                                              equation="ElectronContinuityEquation")
        hole = devsim.get_contact_current(device=self.name, contact="drain",
                                          equation="HoleContinuityEquation")
        return (electron + hole) * UM

    def gate_charge(self) -> float:
        """게이트 접점의 전하 [C/cm]."""
        return devsim.get_contact_charge(device=self.name, contact="gate",
                                         equation="PotentialEquation")

    # ------------------------------------------------------------------
    # 4) 스윕
    # ------------------------------------------------------------------
    def sweep_idvg(self, vd: float, start: float, stop: float, step: float) -> pd.DataFrame:
        """드레인 전압을 고정하고 Vg를 스윕하여 (Vg, Id) 표를 돌려준다."""
        self.set_bias("drain", vd)
        vgs = voltage_points(start, stop, step)
        ids = []
        for vg in vgs:
            self.set_bias("gate", float(vg))
            ids.append(self.drain_current())
        return pd.DataFrame({"Vg_V": vgs, "Id_A_per_um": ids})

    def sweep_idvd(self, vg: float, start: float, stop: float, step: float) -> pd.DataFrame:
        """[과제] 게이트 전압을 고정하고 Vd를 스윕하여 (Vd, Id) 표를 돌려준다.

        sweep_idvg와 거의 같다 — 어느 접점이 고정되고 어느 접점이 변하는가?
        """

        # We first need to set Gate Voltage to Vg, we go through Vd values 
        # from start to stop and for each step we set Drain Voltage to Vd and 
        # get Current value Id and finally return

        self.set_bias("gate", vg) 
        vds = voltage_points(start, stop, step)
        ids = []
        for vd in vds:
            self.set_bias("drain", float(vd)) 
            ids.append(self.drain_current())
        return pd.DataFrame({"Vd_V": vds, "Id_A_per_um": ids})

    def sweep_cv(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """[과제] Vg를 스윕하며 게이트 전하의 기울기 dQg/dVg로 (Vg, Cgg) 표를 돌려준다.

        quasi-static C-V는 enable_transport() 없이 평형 상태에서 계산한다.
        힌트: 전압마다 gate_charge()를 모은 뒤 np.gradient(전하, 전압)로
        기울기를 구하고, * UM 으로 F/µm 단위로 바꾼다.
        """
        
        # We go through Vg values from start to stop and for each step
        # we set Gate Voltage to Vg and get Charge value Qg
        # After that we calculate gradient to find Capacitance Cgg,
        # fix measurement unit multplying by UM and finally return

        vgs = voltage_points(start, stop, step)
        charges = []
        for vg in vgs:
            self.set_bias("gate", float(vg)) 
            charges.append(self.gate_charge())
        

        capacitances = np.gradient(charges, vgs) * UM
        return pd.DataFrame({"Vg_V": vgs, "Cgg_F_per_um": capacitances})


    # Project 1 : Calculate concentration based on temperature
    def _intrinsic_concentration(self, T: float) -> float:
        Eg0, alpha, beta = 1.166, 4.73e-4, 636.0 
        Eg_T = Eg0 - alpha * T**2 / (T + beta)   
        Eg_300 = Eg0 - alpha * 300.0**2 / (300.0 + beta)
        ni_300 = 1.0e10
        return ni_300 * (T / 300.0) ** 1.5 * math.exp(
                -Eg_T / (2 * KB_EV * T) + Eg_300 / (2 * KB_EV * 300.0)
            )
    # Project 1 : Create gate contact with workfunction
    def _create_gate_contact_with_workfunction(self) -> None:
        """Replaces CreateOxideContact for the gate only, adding the Vfb
        offset so silicon_gate_metal_name actually affects Vth."""
        vfb = self._work_function_offset()
        contact_bias_name = GetContactBiasName("gate")
        contact_model_name = GetContactNodeModelName("gate")

        eq = "Potential - ({0} - {1:.6e})".format(contact_bias_name, vfb)
        CreateContactNodeModel(self.name, "gate", contact_model_name, eq)
        CreateContactNodeModelDerivative(self.name, "gate", contact_model_name, eq, "Potential")

        if not InEdgeModelList(self.name, "oxide", "contactcharge_edge"):
            CreateEdgeModel(self.name, "oxide", "contactcharge_edge", "Permittivity*ElectricField")
            CreateEdgeModelDerivatives(self.name, "oxide", "contactcharge_edge",
                                        "Permittivity*ElectricField", "Potential")

        devsim.contact_equation(
            device=self.name, contact="gate", name="PotentialEquation",
            node_model=contact_model_name, edge_charge_model="contactcharge_edge")

    # Project 1: gate work function fix
    def _work_function_offset(self) -> float:
        """Vfb = Phi_M - Phi_i [V]. Phi_i = chi_Si + Eg/2 is the reference
        work function implicit in DEVSIM's intrinsic-level potential
        convention (the body's own -phi_F offset is already handled by
        CreateSiliconPotentialOnlyContact, so it must NOT be added again here)."""
        chi_si = 4.05   # Si electron affinity [eV]
        Eg = 1.12       # Si bandgap at 300K [eV]
        phi_i = chi_si + Eg / 2
        return PHI_M_TABLE[self.silicon_gate_metal_name] - phi_i