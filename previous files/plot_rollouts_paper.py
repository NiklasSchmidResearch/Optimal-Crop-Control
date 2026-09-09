import sys
import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import pickle as pkl
import os
import math
from dataclasses import dataclass


# 1. DEFINE THE CLASS
@dataclass
class RolloutResult:
    penalty: float
    deficit_cost: float
    total_leaching: float
    violated: int


# 2. ABSOLUTE PATH DIAGNOSIS
target_dir = r"C:\Users\nikla\Documents\GitHub\Optimal-Crop-Control"
file_name = "3_rollout_objects.pkl"
full_path = os.path.join(target_dir, file_name)

print(f"--- PATH DIAGNOSIS ---")
print(f"Script is looking for: {full_path}")
print(f"Does the file exist at this path? {os.path.exists(full_path)}")
print(f"File size at this path: {os.path.getsize(full_path)} bytes")

# 3. LOAD THE FILE
rollout_parameters = None
if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
    with open(full_path, "rb") as f:
        try:
            rollout_parameters = pkl.load(f)
            print("Successfully loaded rollout objects.")
        except Exception as e:
            print(f"Error loading pickle: {e}")
else:
    print(f"CRITICAL: The file at {full_path} is still reporting 0 bytes.")

# Only proceed if data loaded successfully
if rollout_parameters is not None:
    # Safely match the key near 0.201267
    target_key = 0.20126710688876742
    if target_key in rollout_parameters:
        rollout_data = rollout_parameters[target_key]
    else:
        # Fallback in case exact float representation varies slightly
        closest_key = min(rollout_parameters.keys(), key=lambda k: abs(k - 0.201267))
        rollout_data = rollout_parameters[closest_key]

    num_trajectories_plotted = 400
    rollout_data = rollout_data[:num_trajectories_plotted]

    # --- IMPORT CONSTANTS AND FUNCTIONS FROM MODULES ---
    from forward_models import SOIL_DEPTH_MM
    from main import (
        N_SUFFICIENT_YIELD_G_GRAIN_PER_M2,
        MAX_BIOMASS_G_DRY_MASS_PER_M2,
        NUM_DAYS_TOTAL,
        NNI_TO_RELATIVE_YIELD_CORRELATION
    )

    # Limit is 10 mg N / L (g N / m3 is mathematically equivalent to mg N / L)
    LEACHING_LIMIT = 10.0

    soil_depth_m = SOIL_DEPTH_MM / 1000.0

    # --- SETUP PLOTS (2 SUBPLOTS SIDE-BY-SIDE WITH TIKZ AESTHETIC) ---
    import matplotlib.pyplot as plt

    # --- ADJUST THIS VARIABLE TO CHANGE FONT SIZES ---
    FONT_SIZE = 14  # Standard LaTeX document font size (try 10, 11, or 12)

    # Apply global configurations to match native LaTeX / pgfplots styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",

        # Font size assignments
        "font.size": FONT_SIZE,  # Base font size
        "axes.titlesize": FONT_SIZE + 1,  # Slightly larger for subplot titles
        "axes.labelsize": FONT_SIZE,  # Axis labels (Time, Concentration, etc.)
        "xtick.labelsize": FONT_SIZE+5,  # Axis tick numbers
        "ytick.labelsize": FONT_SIZE+5,  # Axis tick numbers
        "legend.fontsize": FONT_SIZE,  # Legend text

        "figure.autolayout": True,
        "axes.edgecolor": "#262626",
        "axes.linewidth": 0.8,
        "grid.color": "#E6E6E6",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "text.antialiased": True
    })

    # Generate a 1x2 grid focusing only on the 1st and 3rd metrics
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4))

    soil_depth_m = SOIL_DEPTH_MM / 1000.0
    highlighted_violating_trajectory = False

    # --- DATA PROCESSING & PLOTTING ---
    for idx, traj in enumerate(rollout_data):
        get_val = lambda obj, attr: getattr(obj, attr) if hasattr(obj, attr) else obj[attr]

        leaching_hourly = get_val(traj, 'nitrate_leaching_gN_per_m3_per_day')
        plant_N_hourly = get_val(traj, 'plant_accumulated_N_gN_per_m3')

        hourly_leaching_conc = leaching_hourly
        hourly_accumulated_N_m2 = plant_N_hourly * soil_depth_m
        hours = np.arange(1, len(hourly_leaching_conc) + 1)

        is_violating = np.cumsum(hourly_leaching_conc > LEACHING_LIMIT)[-1] > 0

        # Style mapping to match exact visual weight requirements
        if is_violating and not highlighted_violating_trajectory:
            color = 'red'
            lw = 1.8
            alpha_val = 1.0
            z_order = 5
            highlighted_violating_trajectory = True
        else:
            color = 'lightcoral' if is_violating else 'gray'
            lw = 0.4
            # Background spaghetti lines use high transparency for visual clarity
            alpha_val = 0.55 if is_violating else 0.45
            z_order = 1

        ax1.plot(hours, hourly_leaching_conc, color=color, alpha=alpha_val, linewidth=lw, zorder=z_order)
        ax3.plot(hours, hourly_accumulated_N_m2, color=color, alpha=alpha_val, linewidth=lw, zorder=z_order)

    # --- FORMAT PLOT 1: Hourly Leaching over Time ---
    ax1.axhline(y=LEACHING_LIMIT, color='red', linestyle='--', linewidth=1.5, label=f'Limit ({LEACHING_LIMIT} mg/L)',
                zorder=6)
    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Nitrate Concentration (mg N / L)')
    # Limit y axis
    ax1.set_ylim(0, 35)
    ax1.grid(True)

    # Sharp-edged rectangular legend matching pgfplots standard box
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#262626', framealpha=1.0)

    # --- FORMAT PLOT 2 (Original Plot 3): Accumulated N Uptake over Time ---
    ax3.set_xlabel('Time (Hours)')
    ax3.set_ylabel('Accumulated Plant N (g N / m²)')
    ax3.set_ylim(0, 30)
    ax3.grid(True)

    # --- TIKZ AXIS FRAME REPLICATION ---
    for ax in [ax1, ax3]:
        # Mimics the clear enclosed box with internal-facing tick markers
        ax.tick_params(direction='in', top=True, right=True, colors='#262626', labelsize=10)
        ax.set_facecolor('white')

    # --- TIKZ AXIS FRAME REPLICATION ---
    for ax in [ax1, ax3]:
        # Changed labelsize=10 to labelsize=FONT_SIZE - 1
        ax.tick_params(direction='in', top=True, right=True, colors='#262626', labelsize=FONT_SIZE - 1)
        ax.set_facecolor('white')

    # Save directly to a vector PDF format for clean LaTeX integration
    output_filename = "trajectory_plots.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)

    print(f"\n--- SIMULATION AVERAGES ---")
    print(f"Total evaluated realizations: {len(rollout_data)}")
    print(f"Vector PDF exported successfully as '{output_filename}' with full-resolution data.")