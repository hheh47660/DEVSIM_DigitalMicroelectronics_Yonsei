import numpy as np
from pathlib import Path
import sys

# Set the correct path to your module
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from step2.mosfet_tool.simulator import MosfetSimulator
from step2.mosfet_tool.config import Device

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Cell geometric constant
W_UM = 0.1 # Pass TR width (100 nm = 0.1 um), fixed from project


def run_dram_read_transient(c_store_ff, initial_v_cell):
    """
    Executes the DRAM read simulation for 1 ns with 10 ps steps.
    """
    t_max_ps = 1000          # Total duration: 1 ns
    dt_ps = 10               # Timestep: 10 ps
    c_bl_f = 100e-15         # Bitline capacitance: 100 fF
    
    # c_store_ff is now already an absolute value
    c_store_f = c_store_ff * 1e-15  
    
    # Initial conditions
    v_bl = 1.0               
    v_cell = initial_v_cell  
    v_wl = 2.5               
    v_body = -0.5            

    sim.set_bias("drain", v_bl)
    sim.set_bias("source", v_cell)
    sim.set_bias("gate", v_wl)
    sim.set_bias("body", v_body)

    time_steps = []
    i_cell_data = []
    v_bl_data = []

    for t in range(0, t_max_ps, dt_ps):
        
        sim._solve()  # Solve DC for time t
        
        # Extract the current in A/um and multiply it by the width W[cite: 43, 52]
        i_cell_per_um = sim.drain_current()
        i_cell_abs = i_cell_per_um * W_UM  # Absolute cell current (A)
        
        # Calculate delta_q with the absolute current
        delta_q = i_cell_abs * (dt_ps * 1e-12)

        v_cell += delta_q / c_store_f
        sim.set_bias("source", v_cell)
        sim.set_bias("storage_l", v_cell)
        sim.set_bias("storage_r", v_cell)
        
        v_bl -= delta_q / c_bl_f
        sim.set_bias("drain", v_bl)

        time_steps.append(t)
        i_cell_data.append(i_cell_abs * 1e6)  # Convert to uA for the plot
        v_bl_data.append(v_bl)

    # Final evaluation at time 1 ns
    v_bl_final = v_bl_data[-1]
    read_margin_v = abs(v_bl_final - 1.0)
    read_margin_mv = read_margin_v * 1000

    print(f"\n--- Results at {t_max_ps} ps ---")
    print(f"Final V_BL: {v_bl_final:.4f} V")
    print(f"Read margin (|V_BL - 1V|): {read_margin_mv:.2f} mV")

    # Check if the margin is >= 40mV[cite: 43, 52]
    if read_margin_mv >= 40.0:
        print("PASS: The read margin exceeded the 40 mV threshold.")
    else:
        print("FAIL: The read margin is insufficient (< 40 mV).")

    plot_results_matplotlib(time_steps, i_cell_data, v_bl_data)

    retention_time = calculate_time_retention(sim)
    print(f"\nEstimated retention time: {retention_time:.4e} milliseconds")
    if retention_time > 64.0:
        print("PASS: The retention time exceeded the 64 ms threshold.")
    else:
        print("FAIL: The retention time is insufficient (< 64 ms).")


def plot_results_streamlit(time_steps, i_cell_data, v_bl_data):
    fig_i = px.line(x=time_steps, y=i_cell_data, labels={'x': 'Time (ps)', 'y': 'I_cell (uA)'}, title='Current I_cell(t)')
    fig_i.update_layout(yaxis=dict(title='I_cell (uA)'), xaxis=dict(title='Time (ps)'))
    fig_i.show()

    fig_v = px.line(x=time_steps, y=v_bl_data, labels={'x': 'Time (ps)', 'y': 'V_BL (V)'}, title='Voltage V_BL(t)')
    fig_v.update_layout(yaxis=dict(title='V_BL (V)'), xaxis=dict(title='Time (ps)'))
    fig_v.show()
    

def plot_results_matplotlib(time_steps, i_cell_data, v_bl_data):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, i_cell_data, label='I_cell (uA)', color='blue')
    plt.xlabel('Time (ps)')
    plt.ylabel('I_cell (uA)')
    plt.title('Current I_cell(t)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, v_bl_data, label='V_BL (V)', color='orange')
    plt.xlabel('Time (ps)')
    plt.ylabel('V_BL (V)')
    plt.title('Voltage V_BL(t)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def calculate_time_retention(sim):
    """
    Calculates the DRAM cell retention time in milliseconds.
    """
    sim.set_bias("drain", 1.0)
    sim.set_bias("source", 0.0)
    sim.set_bias("storage_l", 0.0)
    sim.set_bias("storage_r", 0.0)
    sim.set_bias("gate", 0.0)
    sim.set_bias("body", -0.5)
   
    # Convert the leakage current to an absolute value
    i_leakage_per_um = sim.drain_current()
    i_leakage_abs = abs(i_leakage_per_um) * W_UM  # Use abs() to avoid negative retention times
    
    # Convert the cell capacitance to an absolute value
    c_store_per_um = sim.plate_capacitance(2.0)
    c_store_abs_f = c_store_per_um * W_UM
    
    c_bl_f = 100e-15
    delta_v_allowed = 1.0 - 0.04 * (c_store_abs_f + c_bl_f) / c_store_abs_f  
    
    time_retention_s = (c_store_abs_f * delta_v_allowed) / i_leakage_abs
    return time_retention_s * 1000  # Convert to milliseconds


if __name__ == "__main__":
    device_cfg = Device(
        gate_length_um=0.30,  # is fixed
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=398,  # Set the temperature to 125 °C (398 K)[cite: 44, 53]
        silicon_gate_metal_name="Ti",
        cap_height_um=0.8,
        cap_dielectric_thickness_nm=5.0,
        cap_dielectric_material="ZrO2"
    )

    sim = MosfetSimulator(device_cfg, name="Project1")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    # Retrieve the absolute C_STORE in fF[cite: 43, 52]
    c_store_abs_ff = sim.plate_capacitance(2.0) * W_UM * 1e15
    print(f"Extracted storage capacitance: {c_store_abs_ff:.2f} fF")

    print("\nStarting read simulation for data '1' (V_cell = 2.0V)...")
    run_dram_read_transient(c_store_ff=c_store_abs_ff, initial_v_cell=2.0)
    
    # Retention is already called at the end of run_dram_read_transient, 
    # so I removed the separate call.