import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import matplotlib.patches as patches

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from step2.mosfet_tool.simulator import MosfetSimulator
from step2.mosfet_tool.config import Device


def draw_mosfet_schema(sim, output_file="mosfet_schema.png"):
    # Conversione unità: DEVSIM usa cm, convertiamo in µm per la rappresentazione
    UM = 1e4 
    
    # 1. Estrazione delle dimensioni chiave dal simulatore
    x_right = sim.x_right * UM
    y_bottom = sim.y_bottom * UM
    y_junc = sim.y_junction * UM
    
    x_g_left = sim.x_gate_left * UM
    x_g_right = sim.x_gate_right * UM
    
    t_ox = sim.dev.oxide_thickness_nm / 1000.0  # nm -> µm
    y_gate_top = (-sim.y_gate_top) * UM
    
    x_p_left = sim.x_pillar_left * UM
    x_p_right = sim.x_pillar_right * UM
    x_hk_l = sim.x_hk_l_left * UM
    x_hk_r = sim.x_hk_r_right * UM
    cap_h = sim.dev.cap_height_um
    
    # Creazione della figura
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')  # Nasconde gli assi graduati per un diagramma pulito

    # -------------------------------------------------------------------------
    # COLORI (Coerenti con le slide del corso)
    # -------------------------------------------------------------------------
    c_bulk = '#c6dbe1'         # Azzurro p-substrate
    c_sd = '#f4b3b3'           # Rosa n+ S/D
    c_p_tap = '#e2c2f0'        # Lilla p+ tap
    c_oxide = '#fce289'        # Giallo oxide
    c_gate_metal = '#9ea7b0'   # Grigio metallo gate
    c_pillar = '#3f474e'       # Grigio scuro pillar
    c_hk = '#b8e3c4'           # Verde chiaro dielettrico hk
    c_contact = '#434b52'      # Grigio scuro contatti

    # -------------------------------------------------------------------------
    # 2. DISEGNO DELLE AREE COLORATE (Con etichette per la legenda)
    # -------------------------------------------------------------------------
    # Bulk Silicon Base
    ax.add_patch(patches.Rectangle((0, -y_bottom), x_right, y_bottom, facecolor=c_bulk, edgecolor='#7f9db9', lw=1, label='Silicon Bulk (p-type)'))
    
    # p+ body tap region (sinistra)
    p_tap_w = 0.1
    ax.add_patch(patches.Rectangle((0, -y_junc), p_tap_w, y_junc, facecolor=c_p_tap, edgecolor='none', label='p+ Body Tap'))
    
    # n+ Source & Drain regions
    ax.add_patch(patches.Rectangle((p_tap_w + 0.02, -y_junc), x_g_left - (p_tap_w + 0.02), y_junc, facecolor=c_sd, edgecolor='none', label='n+ Source/Drain'))
    ax.add_patch(patches.Rectangle((x_g_right, -y_junc), x_right - x_g_right, y_junc, facecolor=c_sd, edgecolor='none'))
    
    # Gate Oxide & Gate Metal
    ax.add_patch(patches.Rectangle((x_g_left, 0), x_g_right - x_g_left, t_ox, facecolor=c_oxide, edgecolor='none', label='Gate Oxide (SiO2)'))
    ax.add_patch(patches.Rectangle((x_g_left, t_ox), x_g_right - x_g_left, y_gate_top - t_ox, facecolor=c_gate_metal, edgecolor='none', label=f'Gate Metal ({sim.silicon_gate_metal_name})'))
    
    # Capacitor: Metal Pillar + ZrO2 Slabs
    ax.add_patch(patches.Rectangle((x_p_left, 0), x_p_right - x_p_left, cap_h, facecolor=c_pillar, edgecolor='none', label='Metal Pillar'))
    ax.add_patch(patches.Rectangle((x_hk_l, 0), x_p_left - x_hk_l, cap_h, facecolor=c_hk, edgecolor='none', label=f'Dielectric ({sim.cap_dielectric_material})'))
    ax.add_patch(patches.Rectangle((x_p_right, 0), x_hk_r - x_p_right, cap_h, facecolor=c_hk, edgecolor='none'))

    # -------------------------------------------------------------------------
    # 3. CONTATTI E METALLIZZAZIONI
    # -------------------------------------------------------------------------
    c_h = 0.04
    # Contact Body, Source, Drain
    ax.add_patch(patches.Rectangle((0.01, 0), 0.06, c_h, facecolor=c_contact, label='Metal Contacts'))
    ax.add_patch(patches.Rectangle((0.12, 0), 0.12, c_h, facecolor=c_contact))
    ax.add_patch(patches.Rectangle((x_g_right + 0.1, 0), 0.12, c_h, facecolor=c_contact))

    # Limiti del grafico
    ax.set_xlim(-0.05, x_right + 0.05)
    ax.set_ylim(-y_bottom - 0.05, cap_h + 0.1)

    # -------------------------------------------------------------------------
    # 4. AGGIUNTA DELLA LEGENDA
    # -------------------------------------------------------------------------
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    # Mostra il grafico a schermo al termine della simulazione
    plt.show()

# Esempio d'uso:

def main():
    device = Device(
        gate_length_um=0.30,
        source_length_um=0.45,
        drain_length_um=0.5,
        oxide_thickness_nm=5.0,
        silicon_thickness_um=0.5,
        junction_depth_um=0.08,          # <-- PORTATO A 0.08 per stabilizzare la reverse bias (V_body = -0.5V)
        body_doping_cm3=7.0e16,
        sd_doping_cm3=1.0e19,
        temperature_k=398,
        silicon_gate_metal_name="TiN",
        cap_height_um=0.8,
        cap_dielectric_thickness_nm=4.5,
        cap_dielectric_material="ZrO2"
    )
    sim = MosfetSimulator(device, name="Project1")
    sim.build()
    draw_mosfet_schema(sim)

if __name__ == "__main__":
    main()