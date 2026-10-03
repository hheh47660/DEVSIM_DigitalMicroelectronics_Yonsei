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
)

from .config import Device

UM = 1.0e-4  # 1 µm in cm (DEVSIM 내부 단위는 cm)
NM = 1.0e-7  # 1 nm in cm

# Project 1 : Boltzmann Constant
KB_EV = 8.617333262145e-5

# Project 2: relative permittivity of each allowed capacitor dielectric
# (from the assignment's material table)
DIELECTRIC_EPS_R_TABLE = {
    "SiO2": 3.9,
    "Al2O3": 9.0,
    "HfO2": 20.0,
    "ZrO2": 35.0,
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

    # Project 2: storage-capacitor pillar width and mesh resolution.
    # 0.1 um matches the assignment's own 2-D estimate example
    # ("C ~ eps*eps0*Lfacing/t per um, x W = 0.1 um").
    CAP_PILLAR_WIDTH_UM = 0.1
    DX_CAP = 2.5 * NM  # fine x-spacing, comparable to the 3-10nm dielectric thickness range

    def __init__(self, device: Device, name: str = "mos_light"):
        self.dev = device
        self.name = name
        self.mesh = f"{name}_mesh"
        self.x_gate_left = device.source_length_um * UM
        self.x_gate_right = self.x_gate_left + device.gate_length_um * UM
        self.x_right = self.x_gate_right + device.drain_length_um * UM
        self.y_oxide_top = -device.oxide_thickness_nm * NM
        self.y_junction = device.junction_depth_um * UM
        self.y_bottom = device.silicon_thickness_um * UM        
        self.bias = {"gate": 0.0, "source": 0.0, "drain": 0.0, "body": 0.0}

        # Project 1 : Add gate_metal
        self.y_gate_top = self.y_oxide_top - 10.0 * NM
        self.silicon_gate_metal_name = device.silicon_gate_metal_name

        # Project 2: storage capacitor geometry (1T1C cell).
        # The pillar sits centered over the source region, so it must fit
        # entirely between x=0 and x=x_gate_left: if source_length_um is
        # too small for the chosen cap_height_um/cap_dielectric_thickness_nm,
        # the capacitor will overlap the gate oxide region.
        tdiel_cm = device.cap_dielectric_thickness_nm * NM
        self.y_cap_top = -device.cap_height_um * UM  # negative y = above the silicon surface
        self.x_pillar_center = self.x_gate_left / 2.0
        self.x_pillar_left = self.x_pillar_center - 0.5 * self.CAP_PILLAR_WIDTH_UM * UM
        self.x_pillar_right = self.x_pillar_center + 0.5 * self.CAP_PILLAR_WIDTH_UM * UM
        self.x_hk_l_left = self.x_pillar_left - tdiel_cm
        self.x_hk_r_right = self.x_pillar_right + tdiel_cm
        self.cap_dielectric_material = device.cap_dielectric_material

    # ------------------------------------------------------------------
    # 1) 구조 만들기
    # ------------------------------------------------------------------
    def build(self) -> None:
        """메시 → 영역/접점 → 도핑 → Poisson 방정식 등록까지 수행한다."""
        self._clear_session()
        self._build_mesh()
        self._build_doping()

        # Project 1 : Save device design
        design_file = Path(__file__).resolve().parents[2] / "project1" / "part2_2026848368.devsim"
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
        # Project 2: added the four x-boundaries of the storage capacitor
        # (hk_l outer face, pillar left/right faces, hk_r outer face),
        # inserted between the device's left edge and the gate.
        for pos in (x_min, 0.0,
                    self.x_hk_l_left, self.x_pillar_left,
                    self.x_pillar_right, self.x_hk_r_right,
                    self.x_gate_left, self.x_gate_right, self.x_right, x_max):
            devsim.add_2d_mesh_line(mesh=self.mesh, dir="x", pos=pos, ps=self.DX_CAP)
        for pos, spacing in (
            (y_min, self.DY_BULK),
            (self.y_gate_top, self.DY_OXIDE), # Project 1: Add gate_material
            # Project 2: top boundary of the storage capacitor (pillar + hk slabs)
            (self.y_cap_top, self.DY_OXIDE),
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

        # Project 2: storage capacitor — equation-free conductive pillar
        # (region name starts with "metal", so the TA physics leaves it alone;
        # it only gives geometric shape between the two dielectric slabs).
        devsim.add_2d_region(mesh=self.mesh, material="metal", region="metal_pillar",
                             xl=self.x_pillar_left, xh=self.x_pillar_right,
                             yl=self.y_cap_top, yh=0.0)

        # Project 2: storage capacitor dielectric, left and right slabs
        # (region names must start with "hk"; material = dielectric name
        # from the table, e.g. "HfO2").
        devsim.add_2d_region(mesh=self.mesh, material=self.cap_dielectric_material,
                             region="hk_l", xl=self.x_hk_l_left, xh=self.x_pillar_left,
                             yl=self.y_cap_top, yh=0.0)
        devsim.add_2d_region(mesh=self.mesh, material=self.cap_dielectric_material,
                             region="hk_r", xl=self.x_pillar_right, xh=self.x_hk_r_right,
                             yl=self.y_cap_top, yh=0.0)

        devsim.add_2d_contact(mesh=self.mesh, name="gate", region="oxide", material="metal",
                              xl=self.x_gate_left, xh=self.x_gate_right,
                              yl=self.y_oxide_top, yh=self.y_oxide_top)
        devsim.add_2d_contact(mesh=self.mesh, name="source", region="bulk", material="metal",
                              xl=x_min, xh=self.x_gate_left, yl=0.0, yh=0.0)
        devsim.add_2d_contact(mesh=self.mesh, name="drain", region="bulk", material="metal",
                              xl=self.x_gate_right, xh=x_max, yl=0.0, yh=0.0)
        devsim.add_2d_contact(mesh=self.mesh, name="body", region="bulk", material="metal",
                              xl=x_min, xh=x_max, yl=self.y_bottom, yh=self.y_bottom)

        # Project 2: storage node contacts — inner faces of the dielectric
        # slabs, facing the equation-free metal pillar (same pattern as
        # "gate" on "oxide": the contact sits on the region that carries
        # equations, not on the equation-free metal region).
        devsim.add_2d_contact(mesh=self.mesh, name="storage_l", region="hk_l", material="metal",
                              xl=self.x_pillar_left, xh=self.x_pillar_left,
                              yl=self.y_cap_top, yh=0.0)
        devsim.add_2d_contact(mesh=self.mesh, name="storage_r", region="hk_r", material="metal",
                              xl=self.x_pillar_right, xh=self.x_pillar_right,
                              yl=self.y_cap_top, yh=0.0)

        # Project 2: plate contacts — outer faces of the dielectric slabs
        # (held at VDD/2 during read/retention); dQ/dV of these gives C_STORE.
        devsim.add_2d_contact(mesh=self.mesh, name="plate_l", region="hk_l", material="metal",
                              xl=self.x_hk_l_left, xh=self.x_hk_l_left,
                              yl=self.y_cap_top, yh=0.0)
        devsim.add_2d_contact(mesh=self.mesh, name="plate_r", region="hk_r", material="metal",
                              xl=self.x_hk_r_right, xh=self.x_hk_r_right,
                              yl=self.y_cap_top, yh=0.0)

        devsim.add_2d_interface(mesh=self.mesh, name="bulk_oxide",
                                region0="bulk", region1="oxide")

        # Project 2: interfaces where the dielectric slabs touch silicon
        # at the surface (required: "an interface wherever bulk/oxide/hk
        # regions touch"). No interface is added between metal_pillar and
        # bulk/hk, since the pillar is equation-free.
        devsim.add_2d_interface(mesh=self.mesh, name="bulk_hk_l",
                                region0="bulk", region1="hk_l")
        devsim.add_2d_interface(mesh=self.mesh, name="bulk_hk_r",
                                region0="bulk", region1="hk_r")

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
        devsim.node_model(
            device=self.name, region="bulk", name="NetDoping",
            equation=f"SourceDoping + DrainDoping - {na:.6e}",
        )

    def _build_physics(self) -> None:
        # Project 1 : Add 128-bit extended precision for convergece
        devsim.set_parameter(name="extended_precision", value=True)
        devsim.set_parameter(name="extended_solver", value=True)
        devsim.set_parameter(name="extended_model", value=True)

        for region in ("bulk", "oxide"):
            CreateSolution(self.name, region, "Potential")
        SetSiliconParameters(self.name, "bulk", self.dev.temperature_k)
        devsim.set_parameter(device=self.name, region="bulk", name="mu_n", value=self.MU_N)
        devsim.set_parameter(device=self.name, region="bulk", name="mu_p", value=self.MU_P)

        # Project 1: Add concentretion temperature dependent
        ni_T = self._intrinsic_concentration(self.dev.temperature_k)
        for param_name in ("n_i", "n1", "p1"):
            devsim.set_parameter(device=self.name, region="bulk", name=param_name, value=ni_T)

        CreateSiliconPotentialOnly(self.name, "bulk")
        SetOxideParameters(self.name, "oxide", self.dev.temperature_k)
        CreateOxidePotentialOnly(self.name, "oxide", "log_damp")
        CreateOxideContact(self.name, "oxide", "gate")
        devsim.set_parameter(device=self.name, name=GetContactBiasName("gate"), value=0.0)

        # Project 2: storage-capacitor dielectric physics (hk_l / hk_r).
        # CreateOxidePotentialOnly is generic — it only needs "Permittivity"
        # already set on the region, so we replicate SetOxideParameters
        # ourselves using the chosen material's relative permittivity
        # instead of the library's hardcoded eps_ox=3.9 (SiO2 only).
        eps_r = DIELECTRIC_EPS_R_TABLE[self.cap_dielectric_material]
        EPS_0 = 8.85e-14  # F/cm, same constant simple_physics.py uses internally
        for region in ("hk_l", "hk_r"):
            devsim.set_parameter(device=self.name, region=region, name="Permittivity",
                                 value=eps_r * EPS_0)
            devsim.set_parameter(device=self.name, region=region, name="ElectronCharge",
                                 value=1.6e-19)
            CreateOxidePotentialOnly(self.name, region, "log_damp")

        # Project 2: storage node and plate contacts. CreateOxideContact
        # just ties Potential to an external bias on the contact's own
        # region, so it works unchanged here — same mechanism as "gate" on
        # "oxide", called twice per region (storage_* and plate_* both sit
        # on hk_l / hk_r, exactly like source/drain/body all sit on "bulk").
        CreateOxideContact(self.name, "hk_l", "storage_l")
        CreateOxideContact(self.name, "hk_l", "plate_l")
        CreateOxideContact(self.name, "hk_r", "storage_r")
        CreateOxideContact(self.name, "hk_r", "plate_r")
        for contact in ("storage_l", "plate_l", "storage_r", "plate_r"):
            devsim.set_parameter(device=self.name, name=GetContactBiasName(contact), value=0.0)

        for contact in ("source", "drain", "body"):
            CreateSiliconPotentialOnlyContact(self.name, "bulk", contact)
            devsim.set_parameter(device=self.name, name=GetContactBiasName(contact), value=0.0)
        CreateSiliconOxideInterface(self.name, "bulk_oxide")

        # Project 2: continuous-potential interfaces where the dielectric
        # slabs touch silicon — CreateSiliconOxideInterface is also generic
        # despite the name, it works for any two regions by name.
        CreateSiliconOxideInterface(self.name, "bulk_hk_l")
        CreateSiliconOxideInterface(self.name, "bulk_hk_r")

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
        Eg0, alpha, beta = 1.166, 4.73e-4, 636.0  # Costanti di Varshni per il Si
        Eg_T = Eg0 - alpha * T**2 / (T + beta)     # Bandgap Eg(T) a temperatura T
        Eg_300 = Eg0 - alpha * 300.0**2 / (300.0 + beta)
        ni_300 = 1.0e10
        return ni_300 * (T / 300.0) ** 1.5 * math.exp(
                -Eg_T / (2 * KB_EV * T) + Eg_300 / (2 * KB_EV * 300.0)
            )