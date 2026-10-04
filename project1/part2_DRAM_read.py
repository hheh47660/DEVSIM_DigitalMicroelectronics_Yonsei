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


def run_dram_read_transient(sim, c_store_ff, initial_v_cell):
    """
    Executes the DRAM read simulation for 1 ns with 10 ps steps.
    """
    t_max_ps = 1000  # Total duration: 1 ns
    dt_ps = 10        # Timestep: 10 ps

    c_store_f = c_store_ff * 1e-15  # c_store_ff is already absolute, just fF -> F

    # Initial conditions
    v_bl = 1.0
    v_cell = initial_v_cell
    v_wl = 2.5
    v_body = -0.5

    

    sim.set_bias("drain", v_bl)
    sim.set_bias("source", v_cell)
    sim.set_bias("storage_l", v_cell)
    sim.set_bias("storage_r", v_cell)
    sim.set_bias("plate_l", v_wl / 2.0)
    sim.set_bias("plate_r", v_wl / 2.0)
    sim.set_bias("gate", v_wl)
    sim.set_bias("body", v_body)

    time_steps = []
    i_cell_data = []
    v_bl_data = []

    for t in range(0, t_max_ps, dt_ps):
        sim.set_bias("source", v_cell)
        sim.set_bias("storage_l", v_cell)
        sim.set_bias("storage_r", v_cell)
        sim.set_bias("drain", v_bl)

        # Extract the current in A/um and scale by the absolute pass-TR width
        i_cell_per_um = sim.drain_current()
        i_cell_abs = i_cell_per_um * W_UM  # Absolute cell current (A)

        delta_q = i_cell_abs * (dt_ps * 1e-12)

        v_cell += delta_q / c_store_f
        v_bl -= delta_q / C_BL_F

        time_steps.append(t)
        i_cell_data.append(i_cell_abs * 1e6)  # Convert to uA for the plot
        v_bl_data.append(v_bl)

    # Final evaluation at time 1 ns
    v_bl_final = v_bl_data[-1]
    read_margin_mv = abs(v_bl_final - 1.0) * 1000

    # Reuse the already-known c_store_f instead of re-measuring it (saves
    # the two extra solves plate_capacitance() would otherwise cost here).
    retention_time = calculate_time_retention(sim, c_store_abs_f=c_store_f)

    print(f"\n--- Results at {t_max_ps} ps ---")
    print(f"Final V_BL: {v_bl_final:.4f} V")
    print(f"Read margin (|V_BL - 1V|): {read_margin_mv:.2f} mV")
    print("PASS: The read margin exceeded the 40 mV threshold."
          if read_margin_mv >= 40.0
          else "FAIL: The read margin is insufficient (< 40 mV).")

    print(f"\nEstimated retention time: {retention_time:.4e} milliseconds")
    print("PASS: The retention time exceeded the 64 ms threshold."
          if retention_time > 64.0
          else "FAIL: The retention time is insufficient (< 64 ms).")

    plot_results_matplotlib(time_steps, i_cell_data, v_bl_data)


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
    sim.set_bias("plate_l", 1.0)
    sim.set_bias("plate_r", 1.0)
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


def run_dram_write_pulse(sim, c_store_abs_f=None, initial_v_cell=0.0,
                         v_bl=2.0, v_wl=2.5, v_body=-0.5):
    """
    Simulates the 5 ns write pulse (BL=2V, WL=2.5V; the cell voltage reached
    at the end of the pulse is the actual written '1' level). Per the
    assignment, only C_STORE charges during the write -- unlike the read,
    BL is actively driven to 2V by the write driver, so it is NOT treated
    as a floating capacitor that drains into the cell: v_bl stays fixed.
    Returns (written_v_cell, time_steps, v_cell_data).
    """
    t_max_ps = 5000  # 5 ns
    dt_ps = 10         # same 10 ps stepping as the read

    v_cell = initial_v_cell

    sim.set_bias("drain", v_bl)
    sim.set_bias("source", v_cell)
    sim.set_bias("storage_l", v_cell)
    sim.set_bias("storage_r", v_cell)
    sim.set_bias("gate", v_wl)
    sim.set_bias("body", v_body)
    sim.set_bias("plate_l", 1.0)  # plate rail stays at VDD/2 = 1.0V throughout
    sim.set_bias("plate_r", 1.0)

    if c_store_abs_f is None:
        c_store_per_um = sim.plate_capacitance(2.0)
        c_store_abs_f = c_store_per_um * W_UM
        # plate_capacitance() perturbs the plate bias to measure dQ/dV; restore it
        sim.set_bias("plate_l", 1.0)
        sim.set_bias("plate_r", 1.0)

    time_steps = []
    v_cell_data = []

    for t in range(0, t_max_ps, dt_ps):
        sim.set_bias("source", v_cell)
        sim.set_bias("storage_l", v_cell)
        sim.set_bias("storage_r", v_cell)
        # drain stays fixed at v_bl -- unlike the read, BL is not drained here

        i_cell_per_um = sim.drain_current()
        i_cell_abs = i_cell_per_um * W_UM
        delta_q = i_cell_abs * (dt_ps * 1e-12)

        v_cell += delta_q / c_store_abs_f

        time_steps.append(t)
        v_cell_data.append(v_cell)

    written_v_cell = v_cell_data[-1]
    print(f"\n--- Write pulse results at {t_max_ps} ps ---")
    print(f"Written '1' level: {written_v_cell:.4f} V "
          f"(ideal target was {v_bl:.2f} V -- the pass transistor's Vth and "
          f"body effect keep it below that)")

    return written_v_cell, time_steps, v_cell_data


def plot_write_pulse(time_steps, v_cell_data):
    plt.figure(figsize=(6, 5))
    plt.plot(time_steps, v_cell_data, color='green')
    plt.xlabel('Time (ps)')
    plt.ylabel('V_cell (V)')
    plt.title('Write pulse: V_cell(t)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_impuse(sim):
    # 1. Write the "1" level
    written_v1, write_ts, write_vcell = run_dram_write_pulse(sim, c_store_abs_f=c_store_abs_ff * 1e-15)
    plot_write_pulse(write_ts, write_vcell)

    # 2. Read the "0" (already fatto)
    run_dram_read_transient(sim, c_store_ff=c_store_abs_ff, initial_v_cell=0.0)

    # 3. Read the "1" (usando il livello reale appena scritto, non 2.0V ideale)
    run_dram_read_transient(sim, c_store_ff=c_store_abs_ff, initial_v_cell=written_v1)

if __name__ == "__main__":
    device_cfg = Device(
        gate_length_um=0.30,  # is fixed
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=6.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.1,
        body_doping_cm3=1.0e16,
        sd_doping_cm3=1.0e20,
        temperature_k=398,  # Set the temperature to 125 degC (398 K)
        silicon_gate_metal_name="Ti",
        cap_height_um=0.8,
        cap_dielectric_thickness_nm=5.0,
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

    # Retrieve the absolute C_STORE in fF
    c_store_abs_ff = sim.plate_capacitance(2.0) * W_UM * 1e15
    print(f"Extracted storage capacitance: {c_store_abs_ff:.2f} fF")

    run_dram_read_transient(sim, c_store_ff=c_store_abs_ff, initial_v_cell=2.0)

    # Retention is already computed and printed inside run_dram_read_transient.