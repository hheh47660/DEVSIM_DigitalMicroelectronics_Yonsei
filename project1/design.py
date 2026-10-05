from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from step2.mosfet_tool.config import Device
from step2.mosfet_tool.simulator import MosfetSimulator


import dataclasses
import numpy as np
import pandas as pd


# =====================================================================
# 1. FUNZIONI INDIVIDUALI (Accettano l'istanza 'sim' come primo argomento)
# =====================================================================

def extract_vth(sim, vd: float = 0.05, target_id: float = 1e-7, df: pd.DataFrame = None) -> float:
    """[1/8] Vth [V]: Tensione Vg per cui Id = 1e-7 A/um a Vd = 0.05 V."""
    if df is None:
        df = sim.sweep_idvg(vd=vd, start=-1.5, stop=2.0, step=0.02)
    vth = float(np.interp(target_id, df["Id_A_per_um"], df["Vg_V"]))
    return vth


def extract_ion(sim) -> float:
    """[2/8] Ion [A/um]: Corrente di Drain a Vg = 2.0 V, Vd = 2.0 V, 300 K."""
    sim.set_bias("drain", 2.0)
    sim.set_bias("gate", 2.0)
    return float(sim.drain_current())


def extract_ioff(sim) -> float:
    """[3/8] Ioff [A/um]: Corrente di Drain a Vg = 0.0 V, Vd = 2.0 V, 300 K."""
    sim.set_bias("drain", 2.0)
    sim.set_bias("gate", 0.0)
    return float(sim.drain_current())


def extract_ss(sim, vd: float = 0.05, df: pd.DataFrame = None) -> float:
    """[4/8] SS [mV/dec]: Pendenza della corrente tra 1e-10 e 1e-8 A/um."""
    if df is None:
        df = sim.sweep_idvg(vd=vd, start=0.0, stop=2.0, step=0.02)
    mask = (df["Id_A_per_um"] >= 1e-10) & (df["Id_A_per_um"] <= 1e-8)
    if np.sum(mask) < 2:
        return 999.0
    slope, _ = np.polyfit(df.loc[mask, "Vg_V"], np.log10(df.loc[mask, "Id_A_per_um"]), 1)
    return float((1.0 / slope) * 1000.0)


def extract_dibl(sim, vth_low: float = None, vth_high: float = None) -> float:
    """[5/8] DIBL [mV/V]: (Vth(Vd=0.05) - Vth(Vd=2.0)) / 1.95 V * 1000."""
    if vth_low is None:
        vth_low = extract_vth(sim, vd=0.05)
    if vth_high is None:
        vth_high = extract_vth(sim, vd=2.0)
    return float((vth_low - vth_high) / 1.95 * 1000.0)


def extract_body_effect(sim, vth_zero: float = None) -> float:
    """[6/8] Body Effect [V]: Vth(VB=-0.5V) - Vth(VB=0V) a Vd = 0.05 V."""
    if vth_zero is None:
        vth_zero = extract_vth(sim, vd=0.05)
    sim.set_bias("body", -0.5)
    df_body = sim.sweep_idvg(vd=0.05, start=0.0, stop=2.0, step=0.02)
    vth_body = float(np.interp(1e-7, df_body["Id_A_per_um"], df_body["Vg_V"]))
    sim.set_bias("body", 0.0)  # Ripristina VB a 0 V
    return float(vth_body - vth_zero)


def extract_eox(sim) -> float:
    """[7/8] Eox [MV/cm]: VDD / tox."""
    tox_cm = sim.dev.oxide_thickness_nm * 1e-7
    eox_v_per_cm = 2.0 / tox_cm
    return float(eox_v_per_cm / 1e6)


def extract_ioff_hightemp(sim) -> float:
    """[8/8] Ioff a 125 °C (398 K) [A/um]."""
    # Importiamo MosfetSimulator dinamicamente o dal modulo del simulatore
    MosfetSimulatorClass = sim.__class__
    hot_device = dataclasses.replace(sim.dev, temperature_k=398.0)
    sim_hot = MosfetSimulatorClass(hot_device, name=f"{sim.name}_hot")
    sim_hot.build()
    sim_hot.solve_equilibrium()
    sim_hot.enable_transport()
    return float(extract_ioff(sim_hot))


# =====================================================================
# 2. FUNZIONE MASTER (Restituisce il dizionario con tutti i parametri)
# =====================================================================


def main() -> None:
    device = Device(
        gate_length_um=0.65,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti",
        
        cap_height_um=0.8,
        cap_dielectric_thickness_nm=5.0,
        cap_dielectric_material="ZrO2"
    )

    sim = MosfetSimulator(device, name="Project1")
    sim.build()

    # sim.solve_equilibrium()
    # sim.enable_transport()

    # metrics = extract_all_metrics(sim)
    # for metric_name, metric_info in metrics.items():
    #     value = metric_info["value"]
    #     unit = metric_info["unit"]
    #     required = metric_info["required"]
    #     passed = "PASS" if metric_info["pass"] else "FAIL"
    #     print(f"{metric_name}: {value:.4e} {unit} (Required: {required}) -> {passed}")

    #vth = extract_vth(sim, vd=0.05)
    #ion = extract_ion(sim)
    #print(f"Vth: {vth:.4e} V")
    #print(f"Ion: {ion:.4e} A/um")



if __name__ == "__main__":
    main()
