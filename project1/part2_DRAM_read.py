import numpy as np
from pathlib import Path
import sys


# Sostituisci "TuaClasseSimulatore" con l'effettiva classe o funzione di setup
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from step2.mosfet_tool.simulator import MosfetSimulator
from step2.mosfet_tool.config import Device

import pandas as pd
import plotly.express as px
import yaml


def run_dram_read_transient(c_store_ff, initial_v_cell):
    """
    Esegue la simulazione di lettura DRAM per 1 ns con step di 10 ps.
    """
    # 1. Inizializzazione dei parametri secondo Project1_Assignment.pdf
    t_max_ps = 1000          # Durata totale: 1 ns[cite: 43, 52]
    dt_ps = 10               # Timestep: 10 ps
    c_bl_f = 100e-15         # Capacità Bitline: 100 fF[cite: 43, 52]
    c_store_f = c_store_ff * 1e-15  # Capacità della cella in Farad
    
    # Condizioni iniziali
    v_bl = 1.0               
    v_cell = initial_v_cell  # 0.0V per il dato "0", o il livello del dato "1"
    v_wl = 2.5               
    v_body = -0.5            

    device = Device(
        gate_length_um=0.65,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=398,  # Imposta la temperatura a 125 °C (398 K) per la simulazione
        silicon_gate_metal_name="Ti",
        
        cap_height_um=0.8,
        cap_dielectric_thickness_nm=5.0,
        cap_dielectric_material="ZrO2"
    )

    sim = MosfetSimulator(device, name="Project1")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    sim.set_bias("drain", v_bl)
    sim.set_bias("source", v_cell)
    sim.set_bias("gate", v_wl)
    sim.set_bias("body", v_body)

    time_steps = []
    i_cell_data = []
    v_bl_data = []

    for t in range(0, t_max_ps + dt_ps, dt_ps):
        
        sim._solve()  # Risolvi il DC per l'istante t[cite: 39, 44, 53]
        i_cell = sim.drain_current()  # Corrente della cella in A[cite: 39, 44, 53] 
        delta_q = i_cell * (dt_ps * 1e-12)  # Corrente in A, dt in s

        v_cell += delta_q / c_store_f
        sim.set_bias("source", v_cell)  # Aggiorna la tensione della cella per il prossimo passo
        
        v_bl -= delta_q / c_bl_f
        sim.set_bias("drain", v_bl)  # Aggiorna la tensione della bitline per il prossimo passo

        time_steps.append(t)
        i_cell_data.append(i_cell * 1e6)  # Converti in uA
        v_bl_data.append(v_bl)



    # 3. Valutazione finale al tempo 1 ns
    v_bl_final = v_bl_data[-1]
    read_margin_v = abs(v_bl_final - 1.0)
    read_margin_mv = read_margin_v * 1000

    print(f"--- Risultati a {t_max_ps} ps ---")
    print(f"V_BL finale: {v_bl_final:.4f} V")
    print(f"Margine di lettura (|V_BL - 1V|): {read_margin_mv:.2f} mV")

    # Verifica se il margine è >= 40mV[cite: 43, 52]
    if read_margin_mv >= 40.0:
        print("ESITO POSITIVO: Il margine di lettura ha superato la soglia dei 40 mV.")
    else:
        print("ESITO NEGATIVO: Il margine di lettura è insufficiente (< 40 mV).")

    # 4. Generazione dei grafici (I(t) e V_BL(t))
    plot_results_matplotlib(time_steps, i_cell_data, v_bl_data)

    retention_time = calculate_time_retention(sim)  # Calcola il tempo di retention della cella DRAM
    print(f"Tempo di retention stimato: {retention_time:.4e} millisecondi")
    if retention_time > 64.0:
        print("ESITO POSITIVO: Il tempo di retention ha superato la soglia dei 64 ms.")
    else:
        print("ESITO NEGATIVO: Il tempo di retention è insufficiente (< 64 ms).")

def plot_results_streamlit(time_steps, i_cell_data, v_bl_data):
    """
    Genera i grafici per I_cell(t) e V_BL(t). senza matplolib, usando plotly.
    """
    # Grafico della corrente I_cell(t)
    fig_i = px.line(x=time_steps, y=i_cell_data, labels={'x': 'Tempo (ps)', 'y': 'I_cell (uA)'}, title='Corrente I_cell(t)')
    fig_i.update_layout(yaxis=dict(title='I_cell (uA)'), xaxis=dict(title='Tempo (ps)'))
    fig_i.show()

    # Grafico della tensione V_BL(t)
    fig_v = px.line(x=time_steps, y=v_bl_data, labels={'x': 'Tempo (ps)', 'y': 'V_BL (V)'}, title='Tensione V_BL(t)')
    fig_v.update_layout(yaxis=dict(title='V_BL (V)'), xaxis=dict(title='Tempo (ps)'))
    fig_v.show()
    
def plot_results_matplotlib(time_steps, i_cell_data, v_bl_data):
    """
    Genera i grafici per I_cell(t) e V_BL(t) usando matplotlib.
    """
    import matplotlib.pyplot as plt

    # Grafico della corrente I_cell(t)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, i_cell_data, label='I_cell (uA)', color='blue')
    plt.xlabel('Tempo (ps)')
    plt.ylabel('I_cell (uA)')
    plt.title('Corrente I_cell(t)')
    plt.grid(True)

    # Grafico della tensione V_BL(t)
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, v_bl_data, label='V_BL (V)', color='orange')
    plt.xlabel('Tempo (ps)')
    plt.ylabel('V_BL (V)')
    plt.title('Tensione V_BL(t)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# need to get time retention  = C_store * Delta_V_allowed / I_leakage with
# Delta_V_allowed = 1V - 40mV * (C_store + C_BL) / C_store
# The condition would be WL = 0V, BL = 1V, Cell = 0V, Body = -0.5V, T = 125C (398K) and then measure I_leakage
# make the function that does this and returns the time retention in seconds, given C_store in fF and I_leakage in A
def calculate_time_retention(sim):
    """
    Calcola il tempo di retention della cella DRAM in secondi.
    """
   
    i_leakage_a = sim.drain_current()  # Corrente di perdita in A
    c_store_f = sim.plate_capacitance(2.0)  # Capacità della cella in Farad
    delta_v_allowed = 1.0 - 0.04 * (c_store_f + 100e-15) / c_store_f  # Assumendo C_BL = 100 fF
    time_retention_s = c_store_f * delta_v_allowed / i_leakage_a
    return time_retention_s * 1000  # Converti in millisecondi


if __name__ == "__main__":
    # Esempio di esecuzione: Inserisci il valore C_STORE reale (in fF) estratto dal tuo design
    # La temperatura deve essere impostata a 125 gradi (398 K) nel tuo config.py o simulator.py[cite: 44, 53]
    C_STORE_ESTRATTO = 12.0 # fF (Valore di esempio)
    
    print("Avvio simulazione lettura dato '0' (V_cell = 0V)...")
    run_dram_read_transient(c_store_ff=C_STORE_ESTRATTO, initial_v_cell=2.0)