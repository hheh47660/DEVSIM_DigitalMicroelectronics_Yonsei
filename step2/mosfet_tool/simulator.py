# -*- coding: utf-8 -*-
"""DEVSIM 2-D nMOSFET lightweight simulator (for coursework).

Kept only what's needed for I-V / C-V calculations from the original
project's simulator.py. The physics equations themselves aren't in this
file -- devsim.python_packages helpers assemble them as string expressions
(see MANUAL.html section 6).
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
    # needed by _create_gate_contact_with_workfunction()
    GetContactNodeModelName,
    CreateContactNodeModel,
    CreateContactNodeModelDerivative,
    InEdgeModelList,
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
)

from .config import Device

UM = 1.0e-4  # 1 um in cm (DEVSIM's internal unit is cm)
NM = 1.0e-7  # 1 nm in cm

# Project 1 : Boltzmann Constant
KB_EV = 8.617333262145e-5

# Project 1: gate metal work function table (eV), from the assignment
PHI_M_TABLE = {
    "n+ poly-Si": 4.05, "Al": 4.10, "Ta": 4.25, "Ti": 4.33,
    "TaN": 4.45, "W": 4.60, "TiN": 4.65, "Mo": 4.70,
    "Ni": 5.10, "p+ poly-Si": 5.15, "Pt": 5.30,
}

# Project 2: relative permittivity of each allowed capacitor dielectric
# (from the assignment's material table)
DIELECTRIC_EPS_R_TABLE = {
    "SiO2": 3.9,
    "Al2O3": 9.0,
    "HfO2": 20.0,
    "ZrO2": 35.0,
}

def voltage_points(start: float, stop: float, step: float) -> np.ndarray:
    """Build a voltage array from start to stop in steps of step (see np.arange in practice.py)."""
    count = round(abs(stop - start) / step)
    signed_step = step if stop >= start else -step
    return np.round(start + signed_step * np.arange(count + 1), 9)


class MosfetSimulator:
    """2-D planar nMOS drift-diffusion solver.

    Call order: build() -> solve_equilibrium() -> [enable_transport()] -> sweep_*()
    For C-V only, skip enable_transport().
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
        self.bias = {"gate": 0.0, "source": 0.0, "drain": 0.0, "body": 0.0,
            "storage_l": 0.0, "storage_r": 0.0, "plate_l": 0.0, "plate_r": 0.0}

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
    # 1) Build the structure
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Mesh -> regions/contacts -> doping -> register the Poisson equations."""
        self._clear_session()
        self._build_mesh()
        self._build_doping()

        # Project 1 : Save device design
        design_file = Path(__file__).resolve().parents[2] / "project1" / "part2_2026848368.devsim"
        devsim.write_devices(file=str(design_file), device=self.name, type="devsim")

        self._build_physics()

    def _clear_session(self) -> None:
        # DEVSIM's solve advances every device in the process together, so
        # clear out any previous device before starting a new simulation.
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
        # Smoothly roll off the n+ source/drain doping with erfc() (covered in class).
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
        # Project 1: gate work function fix (replaces plain CreateOxideContact)
        self._create_gate_contact_with_workfunction()
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
    # 2) Solve
    # ------------------------------------------------------------------
    def solve_equilibrium(self) -> None:
        """Converge Poisson with all contacts at 0V (initial value for the next stage)."""
        devsim.solve(type="dc", absolute_error=1.0e-13, relative_error=1.0e-10,
                     maximum_iterations=80)

    def enable_transport(self) -> None:
        """Add electrons/holes as unknowns and turn on the drift-diffusion equations."""
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
        """Ramp the voltage up in RAMP_STEP_V steps, re-solving at each step."""
        start = self.bias[contact]
        steps = max(1, math.ceil(abs(volts - start) / self.RAMP_STEP_V - 1.0e-9))
        for i in range(1, steps + 1):
            value = start + (volts - start) * i / steps
            devsim.set_parameter(device=self.name,
                                 name=GetContactBiasName(contact), value=value)
            self._solve()
        self.bias[contact] = volts

    # ------------------------------------------------------------------
    # 3) Extract quantities
    # ------------------------------------------------------------------
    def drain_current(self) -> float:
        """Electron+hole current at the drain contact [A/um] (the 2-D solution comes out in A/cm)."""
        electron = devsim.get_contact_current(device=self.name, contact="drain",
                                              equation="ElectronContinuityEquation")
        hole = devsim.get_contact_current(device=self.name, contact="drain",
                                          equation="HoleContinuityEquation")
        return (electron + hole) * UM

    def gate_charge(self) -> float:
        """Charge at the gate contact [C/cm]."""
        return devsim.get_contact_charge(device=self.name, contact="gate",
                                         equation="PotentialEquation")

    def plate_charge(self) -> float:
        """Charge at the storage-capacitor plate contacts [C/cm]."""
        ql = devsim.get_contact_charge(device=self.name, contact="plate_l",
                                       equation="PotentialEquation")
        qr = devsim.get_contact_charge(device=self.name, contact="plate_r",
                                       equation="PotentialEquation")
        return ql + qr

    def plate_capacitance(self, vdd: float) -> float:
        """C_STORE from the storage-capacitor plate contacts [F/um] (dQ/dV)."""
        self.set_bias("plate_l", vdd / 2.0)
        self.set_bias("plate_r", vdd / 2.0)
        q1 = self.plate_charge()
        self.set_bias("plate_l", vdd / 2.0 + 0.01)
        self.set_bias("plate_r", vdd / 2.0 + 0.01)
        q2 = self.plate_charge()
        return (q2 - q1) * UM / 0.01
         
    # ------------------------------------------------------------------
    # 4) Sweeps
    # ------------------------------------------------------------------
    def sweep_idvg(self, vd: float, start: float, stop: float, step: float) -> pd.DataFrame:
        """Fix the drain voltage and sweep Vg, returning a (Vg, Id) table."""
        self.set_bias("drain", vd)
        vgs = voltage_points(start, stop, step)
        ids = []
        for vg in vgs:
            self.set_bias("gate", float(vg))
            ids.append(self.drain_current())
        return pd.DataFrame({"Vg_V": vgs, "Id_A_per_um": ids})

    def sweep_idvd(self, vg: float, start: float, stop: float, step: float) -> pd.DataFrame:
        """[Assignment] Fix the gate voltage and sweep Vd, returning a (Vd, Id) table.

        Nearly identical to sweep_idvg -- which contact is fixed and which one varies?
        """

        # Fix the gate voltage to vg, then sweep Vd from start to stop,
        # setting the drain voltage at each step and recording Id.

        self.set_bias("gate", vg) 
        vds = voltage_points(start, stop, step)
        ids = []
        for vd in vds:
            self.set_bias("drain", float(vd)) 
            ids.append(self.drain_current())
        return pd.DataFrame({"Vd_V": vds, "Id_A_per_um": ids})

    def sweep_cv(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """[Assignment] Sweep Vg and return a (Vg, Cgg) table from the slope dQg/dVg.

        Quasi-static C-V is computed at equilibrium, without enable_transport().
        Hint: collect gate_charge() at each voltage, take np.gradient(charge, voltage)
        for the slope, then *UM to convert to F/um.
        """

        # Sweep Vg from start to stop, recording the gate charge Qg at each
        # step, then take the gradient to get Cgg and convert units with *UM.

        vgs = voltage_points(start, stop, step)
        charges = []
        for vg in vgs:
            self.set_bias("gate", float(vg)) 
            charges.append(self.gate_charge())
        

        capacitances = np.gradient(charges, vgs) * UM
        return pd.DataFrame({"Vg_V": vgs, "Cgg_F_per_um": capacitances})

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

    # Project 1 : Calculate concentration based on temperature
    def _intrinsic_concentration(self, T: float) -> float:
        Eg0, alpha, beta = 1.166, 4.73e-4, 636.0  # Si Varshni constants
        Eg_T = Eg0 - alpha * T**2 / (T + beta)     # Bandgap Eg(T) at temperature T
        Eg_300 = Eg0 - alpha * 300.0**2 / (300.0 + beta)
        ni_300 = 1.0e10
        return ni_300 * (T / 300.0) ** 1.5 * math.exp(
                -Eg_T / (2 * KB_EV * T) + Eg_300 / (2 * KB_EV * 300.0)
            )