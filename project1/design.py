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
        df = sim.sweep_idvg(vd=vd, start=0.0, stop=2.0, step=0.02)
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
        df = sim.sweep_idvg(vd=vd, start=-1.5, stop=2.0, step=0.02)
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

def extract_all_metrics(sim, verbosity: bool = False) -> dict:
    """
    Esegue gli sweep minimi necessari su un'istanza di MosfetSimulator e
    restituisce un dizionario contenente tutte le 8 metriche e gli esiti PASS/FAIL.
    """
    print("Estrazione delle 8 metriche in corso...", flush=True)

    # 1. Sweep Vd = 0.05 V (300 K) -> Vth_low e SS
    df_low = sim.sweep_idvg(vd=0.05, start=0.0, stop=2.0, step=0.02)
    vth = extract_vth(sim, vd=0.05, df=df_low)
    ss = extract_ss(sim, vd=0.05, df=df_low)

    # 2. Sweep Vd = 2.0 V (300 K) -> Vth_high, Ion e Ioff
    df_high = sim.sweep_idvg(vd=2.0, start=0.0, stop=2.0, step=0.02)
    vth_high = extract_vth(sim, vd=2.0, df=df_high)
    ion = extract_ion(sim)
    ioff = extract_ioff(sim)

    # 3. Metriche derivate
    dibl = extract_dibl(sim, vth_low=vth, vth_high=vth_high)
    body = extract_body_effect(sim, vth_zero=vth)
    eox = extract_eox(sim)

    # 4. Misurazione ad alta temperatura (398 K)
    print("Simulazione ad alta temperatura (125 °C)...", flush=True)
    ioff_hot = extract_ioff_hightemp(sim)

    # Costruzione del dizionario finale
    if verbosity: 
        return {
            "Vth": {
                "value": vth,
                "unit": "V",
                "required": "0.45 ± 0.05 V",
                "pass": 0.40 <= vth <= 0.50
            },
            "Ion": {
                "value": ion,
                "unit": "A/um",
                "required": ">= 450 uA/um",
                "pass": ion >= 450e-6
            },
            "Ioff": {
                "value": ioff,
                "unit": "A/um",
                "required": "<= 1 pA/um",
                "pass": ioff <= 1e-12
            },
            "SS": {
                "value": ss,
                "unit": "mV/dec",
                "required": "<= 75 mV/dec",
                "pass": ss <= 75.0
            },
            "Ioff_125C": {
                "value": ioff_hot,
                "unit": "A/um",
                "required": "<= 100 pA/um",
                "pass": ioff_hot <= 100e-12
            },
            "DIBL": {
                "value": dibl,
                "unit": "mV/V",
                "required": "<= 30 mV/V",
                "pass": dibl <= 30.0
            },
            "Body_effect": {
                "value": body,
                "unit": "V",
                "required": "<= 0.08 V",
                "pass": body <= 0.08
            },
            "Eox": {
                "value": eox,
                "unit": "MV/cm",
                "required": "<= 5 MV/cm",
                "pass": eox <= 5.0
            }
        }
    else:
        return {
            "Vth": vth,
            "Ion": ion,
            "Ioff": ioff,
            "SS": ss,
            "Ioff_125C": ioff_hot,
            "DIBL": dibl,
            "Body_effect": body,
            "Eox": eox
        }

def test_physics_SS():
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
        silicon_gate_metal_name="Ti"
    )

    sim = MosfetSimulator(device, name="300K")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    ss300 = extract_ss(sim)

    device = Device(
        gate_length_um=0.65,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=398.0,
        silicon_gate_metal_name="Ti"
    )

    sim = MosfetSimulator(device, name="398K")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    ss398 = extract_ss(sim)
    print(f"SS at 300 K: {ss300:.4e} mV/dec")
    print(f"SS at 398 K: {ss398:.4e} mV/dec")
    print(f"Ratio SS(398K)/SS(300K): {ss398/ss300:.4f} and should be 1.327")


def test_physics_workfunction():
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
        silicon_gate_metal_name="Al"
    )

    sim = MosfetSimulator(device, name="AlluminumGate")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    Al_vth = extract_vth(sim)
    Al_ss = extract_ss(sim)

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
        silicon_gate_metal_name="Ti"
    )

    sim = MosfetSimulator(device, name="TitaniumGate")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    Ti_vth = extract_vth(sim)
    Ti_ss = extract_ss(sim)
    

    print(f"SS for Aluminum Gate: {Al_ss:.4e} mV/dec")
    print(f"SS for Titanium Gate: {Ti_ss:.4e} mV/dec")
    print(f"Ratio SS(Ti)/SS(Al): {Ti_ss/Al_ss:.4f} and should be ~1.0 (no significant change)")

    print(f"Vth for Aluminum Gate: {Al_vth:.4e} V")
    print(f"Vth for Titanium Gate: {Ti_vth:.4e} V")
    print(f"Difference in Vth: {Ti_vth - Al_vth:.4e} V and should be ~0.23 V (due to work function difference)")

def test_gate_length_effect():
    
    #study gate_length 
    device1 = Device(
        gate_length_um=0.3,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    device2 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    device3 = Device(
        gate_length_um=0.60,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    # sim = MosfetSimulator(device1, name="gate_length_0.3um")
    # sim.build()
    # sim.solve_equilibrium()
    # sim.enable_transport()

    # res1 = extract_all_metrics(sim)

    # sim = MosfetSimulator(device2, name="gate_length_0.45um")
    # sim.build()
    # sim.solve_equilibrium()
    # sim.enable_transport()

    # res2 = extract_all_metrics(sim)

    sim = MosfetSimulator(device3, name="gate_length_0.60um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res3 = extract_all_metrics(sim)

    # print("Results for gate_length_0.3um:", res1)
    # print("\n")
    # print("Results for gate_length_0.45um:", res2)
    # print("\n")
    print("Results for gate_length_0.60um:", res3)
    
def test_gate_oxide_thickness_effect():
    
    #study gate_length 
    device1 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=4.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    device2 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.5,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    device3 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=7.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    sim = MosfetSimulator(device1, name="oxide_thickness_4.0nm")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res1 = extract_all_metrics(sim)

    sim = MosfetSimulator(device2, name="oxide_thickness_5.5nm")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res2 = extract_all_metrics(sim)

    sim = MosfetSimulator(device3, name="oxide_thickness_7.0nm")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res3 = extract_all_metrics(sim)

    print("Results for oxide_thickness_4.0nm:", res1)
    print("\n")
    print("Results for oxide_thickness_5.5nm",  res2)
    print("\n")
    print("Results for oxide_thickness_7.0nm:", res3)

def test_silicon_thickness_um_effect():
    
    
    device1 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.3,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )

    device2 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=1.2,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    device3 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=2.0,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    sim = MosfetSimulator(device1, name="silicon_thickness_0.3um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res1 = extract_all_metrics(sim)

    sim = MosfetSimulator(device2, name="silicon_thickness_1.2um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res2 = extract_all_metrics(sim)

    sim = MosfetSimulator(device3, name="silicon_thickness_2.0um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res3 = extract_all_metrics(sim)

    print("Results for silicon_thickness_0.3um:", res1)
    print("\n")
    print("Results for silicon_thickness_1.2um:",  res2)
    print("\n")
    print("Results for silicon_thickness_2.0um:", res3)


def test_source_lenght_um_effect():
    
    
    device1 = Device(
        gate_length_um=0.45,
        source_length_um=0.2,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )

    device2 = Device(
        gate_length_um=0.45,
        source_length_um=0.6,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    device3 = Device(
        gate_length_um=0.45,
        source_length_um=1.0,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    sim = MosfetSimulator(device1, name="source_length_0.2um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res1 = extract_all_metrics(sim)

    sim = MosfetSimulator(device2, name="source_length_0.6um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res2 = extract_all_metrics(sim)

    sim = MosfetSimulator(device3, name="source_length_1.0um")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res3 = extract_all_metrics(sim)

    print("Results for source_length_0.2um:", res1)
    print("\n")
    print("Results for source_length_0.6um:",  res2)
    print("\n")
    print("Results for source_length_1.0um:", res3)


def test_body_doping_cm3_effect():
    
    device1 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e14,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )

    device2 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=5.0e15,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    device3 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e17,
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    sim = MosfetSimulator(device1, name="body_doping_1.0e14cm3")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res1 = extract_all_metrics(sim)

    sim = MosfetSimulator(device2, name="body_doping_5.0e15cm3")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res2 = extract_all_metrics(sim)

    sim = MosfetSimulator(device3, name="body_doping_1.0e17cm3")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res3 = extract_all_metrics(sim)

    print("Results for body_doping_1.0e14cm3:", res1)
    print("\n")
    print("Results for body_doping_5.0e15cm3:",  res2)
    print("\n")
    print("Results for body_doping_1.0e17cm3:", res3)

def test_source_doping_cm3_effect():
    
    device1 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e18,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )

    device2 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=5.0e19,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    device3 = Device(
        gate_length_um=0.45,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e21,
        temperature_k=300.0,
        silicon_gate_metal_name="Ti"
    )
    
    sim = MosfetSimulator(device1, name="source_doping_1.0e18cm3")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res1 = extract_all_metrics(sim)

    sim = MosfetSimulator(device2, name="body_doping_5.0e19cm3")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res2 = extract_all_metrics(sim)

    sim = MosfetSimulator(device3, name="body_doping_1.0e21cm3")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    res3 = extract_all_metrics(sim)

    print("Results for body_doping_1.0e18cm3:", res1)
    print("\n")
    print("Results for body_doping_5.0e19cm3:",  res2)
    print("\n")
    print("Results for body_doping_1.0e21cm3:", res3)

def main() -> None:

    # This is default
    device = Device(
        gate_length_um=0.60,             # Lunghezza canale
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=4.5,          # Ridotto a 4.5 nm per abbassare SS e alzare Ion
        silicon_thickness_um=0.5,
        junction_depth_um=0.06,          # Ridotto da 0.1 a 0.06 um per abbattere Ioff_125C
        body_doping_cm3=2e16,          # Calibrato per Vth ~ 0.45 V
        sd_doping_cm3=1.0e20,
        temperature_k=300.0,
        silicon_gate_metal_name="W"    # TiN per Vth stabile
    )
    # gate_length_um in range > 0.3um - Done!
    # gate_oxide_thickness_nm in range > 4nm - Done!
    # junction_depth_um in range  0.02um - 0.25um - Done! -> Ho eliminato la funzione per sbaglio
    # silicon_thickness_um in range 0.3um - 2.0um - Done!
    # source/drain_length_um in range 0.2um - 1.0um - Done!
    # body_doping_cm3 in range 1e14 - 1e17 - Done! 
    # sd_doping_cm3 in range 1e18 - 1e21 - Not Done 

    
    


    # device = Device(
    #     gate_length_um=0.65,
    #     source_length_um=0.45,
    #     drain_length_um=0.5,
    #     oxide_thickness_nm=5.0,
    #     silicon_thickness_um=0.5,
    #     junction_depth_um=0.1,
    #     body_doping_cm3=1.0e16,
    #     sd_doping_cm3=1.0e20,
    #     temperature_k=300.0,
    #     silicon_gate_metal_name="Ti"
    # )

    sim = MosfetSimulator(device, name="NMOS")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    print(extract_all_metrics(sim, verbosity=True))


    # curve = sim.sweep_idvg(vd=0.05, start=-1.0, stop=1.5, step=0.01)
    # print(curve[(curve["Id_A_per_um"] >= 1e-10) & (curve["Id_A_per_um"] <= 1e-8)])
    # print("Corrente a Vg=-1.0 (bordo sweep):", curve.iloc[0]["Id_A_per_um"])

    #Verify physics is working good


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

