from pathlib import Path
import sys

# Set the correct path to your module
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from step2.mosfet_tool.simulator import MosfetSimulator
from step2.mosfet_tool.config import Device

import plotly.express as px
import matplotlib.pyplot as plt
import devsim

# Cell geometric constant
W_UM = 0.1       # Pass TR width (100 nm = 0.1 um), fixed from project
C_BL_F = 100e-15  # Bitline capacitance, given (absolute Farads)


def wait_and_read(sim, time_ps: float, timestep_ps: float,
                     c_store_only: bool = False, c_store_f: float = None):

    time_steps = []
    i_cell_data = []
    v_bl_data = []

    v_cell = sim.bias["source"]    
    v_bl = sim.bias["drain"]

    if c_store_f is None:
        c_store_f = sim.plate_capacitance(2.0)

    for t in range(0, time_ps, timestep_ps):

        # Extract the current in A/um and scale by the absolute pass-TR width
        i_cell_abs = sim.drain_current() * W_UM  # Absolute cell current (A)

        delta_q = i_cell_abs * (timestep_ps * 1e-12)

        v_cell += delta_q / c_store_f
        if not c_store_only:
            v_bl -= delta_q / C_BL_F

        time_steps.append(t)
        i_cell_data.append(i_cell_abs * 1e6)  # Convert to uA for the plot
        v_bl_data.append(v_bl)

        sim.set_bias("source", v_cell)
        sim.set_bias("storage_l", v_cell)
        sim.set_bias("storage_r", v_cell)
        sim.set_bias("drain", v_bl)


    return (time_steps, i_cell_data, v_bl_data)
    

def read_check(sim):
    # Initial conditions
    v_bl = 1.0
    v_cell = 0.0
    v_wl = 2.5
    v_body = -0.5

    c_store_f = sim.plate_capacitance(2.0)

    sim.set_bias("drain", v_bl)
    sim.set_bias("source", v_cell)
    sim.set_bias("storage_l", v_cell)
    sim.set_bias("storage_r", v_cell)
    sim.set_bias("plate_l", v_wl / 2.0)
    sim.set_bias("plate_r", v_wl / 2.0)
    sim.set_bias("gate", v_wl)
    sim.set_bias("body", v_body)

    t_0, i_0, v_0 = wait_and_read(sim, 1000, 10)

    read_0_after_1ns = v_0[-1]
    read_0_margin_mv = abs(read_0_after_1ns - 1.0) * 1000

    # Start to charge the capacitor
    sim.set_bias("drain", 2.0)
    _, _, tmp = wait_and_read(sim, 5 * 1000, 10, c_store_only = True, c_store_f = c_store_f)

    write_1_after_5ns = tmp[-1]

    # Now read the logical 1
    sim.set_bias("drain", v_bl)
    t_1, i_1, v_1 = wait_and_read(sim, 1000, 10, c_store_f = c_store_f)

    read_1_after_1ns = v_1[-1]
    read_1_margin_mv = abs(read_1_after_1ns - 1.0) * 1000

    retention_time = calculate_time_retention(sim, c_store_abs_f=c_store_f)

    print(f"\n--- Results ---")
    print(f"Final V_BL after read 0 for 1ns: {read_0_after_1ns:.4f} V")
    print(f"Read 0 margin (|V_BL - 1V|): {read_0_margin_mv:.2f} mV")
    print("PASS: The read margin exceeded the 40 mV threshold."
          if read_0_margin_mv >= 40.0
          else "FAIL: The read margin is insufficient (< 40 mV).")

    print(f"Final V_BL after read 1 for 1ns: {read_1_after_1ns:.4f} V")
    print(f"Read 1 margin (|V_BL - 1V|): {read_1_margin_mv:.2f} mV")
    print("PASS: The read margin exceeded the 40 mV threshold."
          if read_1_margin_mv >= 40.0
          else "FAIL: The read margin is insufficient (< 40 mV).")

    print(f"\nEstimated retention time: {retention_time:.4e} milliseconds")
    print("PASS: The retention time exceeded the 64 ms threshold."
          if retention_time > 64.0
          else "FAIL: The retention time is insufficient (< 64 ms).")

    plot_results_matplotlib(t_0, i_0, v_0)
    plot_results_matplotlib(t_1, i_1, v_1)




def plot_results_streamlit(time_steps, i_cell_data, v_bl_data):
    fig_i = px.line(x=time_steps, y=i_cell_data,
                    labels={'x': 'Time (ps)', 'y': 'I_cell (uA)'}, title='Current I_cell(t)')
    fig_i.update_layout(yaxis=dict(title='I_cell (uA)'), xaxis=dict(title='Time (ps)'))
    fig_i.show()

    fig_v = px.line(x=time_steps, y=v_bl_data,
                    labels={'x': 'Time (ps)', 'y': 'V_BL (V)'}, title='Voltage V_BL(t)')
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


def calculate_time_retention(sim, c_store_abs_f=None):
    """
    Calculates the DRAM cell retention time in milliseconds.
    If c_store_abs_f is not given, it is measured with plate_capacitance()
    (costs two extra solves); pass it in when you already know it.
    """
    sim.set_bias("drain", 1.0)
    sim.set_bias("source", 0.0)
    sim.set_bias("storage_l", 0.0)
    sim.set_bias("storage_r", 0.0)
    sim.set_bias("plate_l", 0.0)
    sim.set_bias("plate_r", 0.0)
    sim.set_bias("gate", 0.0)
    sim.set_bias("body", -0.5)

    i_leakage_per_um = sim.drain_current()
    i_leakage_abs = abs(i_leakage_per_um) * W_UM  # abs() avoids negative retention times

    if c_store_abs_f is None:
        c_store_per_um = sim.plate_capacitance(2.0)
        c_store_abs_f = c_store_per_um * W_UM

    delta_v_allowed = 1.0 - 0.04 * (c_store_abs_f + C_BL_F) / c_store_abs_f
    time_retention_s = (c_store_abs_f * delta_v_allowed) / i_leakage_abs
    return time_retention_s * 1000  # Convert to milliseconds




def quick_leakage_capacitance_check(sim):
    sim.build(); sim.solve_equilibrium(); sim.enable_transport()
    sim.set_bias("drain", 1.0); sim.set_bias("source", 0.0)
    sim.set_bias("storage_l", 0.0); sim.set_bias("storage_r", 0.0)
    sim.set_bias("gate", 0.0); sim.set_bias("body", -0.5)
    i_leak = abs(sim.drain_current()) * W_UM
    capacacitance = sim.plate_capacitance(2.0)
    print(f"I_leak = {i_leak:.4e} A")
    print(f"Plate_capacitance = {capacacitance:.4e} [Unit]")
    return (i_leak, capacacitance)


if __name__ == "__main__":
    device_cfg = Device(
        gate_length_um=0.30,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,       # tenuto come v1 (ottimo per retention)
        sd_doping_cm3=1.0e19,
        temperature_k=398,
        silicon_gate_metal_name="TiN", # tenuto come v1, NON tornare a TaN
        cap_height_um=1.5,             # massimo consentito (era 1.2 in v1)
        cap_dielectric_thickness_nm=3.0,
        cap_dielectric_material="ZrO2")

    sim = MosfetSimulator(device_cfg, name="Project1")
    sim.build()
    sim.solve_equilibrium()
    sim.enable_transport()

    # Extended precision is only needed for the hard initial equilibrium
    # solve; leaving it on for every one of the transient's ~100 solves
    # makes each one far slower than necessary. These are global DEVSIM
    # parameters (no device= argument), so one call here is enough.
    devsim.set_parameter(name="extended_precision", value=False)
    devsim.set_parameter(name="extended_solver", value=False)
    devsim.set_parameter(name="extended_model", value=False)

    read_check(sim)

    # retention_time   = calculate_time_retention(sim)
    # print(f"\nEstimated retention time: {retention_time:.4e} milliseconds")
    # print("PASS: The retention time exceeded the 64 ms threshold."
    # if retention_time > 64.0
    #     else "FAIL: The retention time is insufficient (< 64 ms).")
    
    