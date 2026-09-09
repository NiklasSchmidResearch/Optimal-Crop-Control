import sys
import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import pickle as pkl
import os
from dataclasses import dataclass


# 1. DEFINE THE CLASS
@dataclass
class RolloutResult:
    penalty: float
    deficit_cost: float
    total_leaching: float
    violated: int


# Custom Unpickler to solve PyCharm Console / Namespace issues
class LooseUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if name == 'RolloutResult':
            return RolloutResult
        return super().find_class(module, name)


def safe_get_val(obj, attr):
    if hasattr(obj, attr): return getattr(obj, attr)
    try:
        return obj[attr]
    except:
        return None


def plot_results_for_one_controller(file_name, target_idx=-1, output_file="trajectory_plots.pdf", dry_run=False):
    # 2. ABSOLUTE PATH DIAGNOSIS
    target_dir = r"C:\Users\nikla\Documents\GitHub\Optimal-Crop-Control"
    full_path = os.path.join(target_dir, file_name)

    print(f"--- PATH DIAGNOSIS ---")
    print(f"Script is looking for: {full_path}")
    print(f"Does the file exist at this path? {os.path.exists(full_path)}")
    if os.path.exists(full_path):
        print(f"File size at this path: {os.path.getsize(full_path)} bytes")

    # 3. LOAD THE FILE USING LOOSE UNPICKLER
    rollout_parameters = None
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        with open(full_path, "rb") as f:
            try:
                rollout_parameters = LooseUnpickler(f).load()
            except Exception as e:
                print(f"Error loading pickle: {e}")
                return [], [], [], []

    if rollout_parameters is None:
        print(f"Failed to load rollout parameters for {file_name}")
        return [], [], [], []

    target_key = 0.20126710688876742
    if target_key in rollout_parameters:
        rollout_data = rollout_parameters[target_key]
    else:
        closest_key = min(rollout_parameters.keys(), key=lambda k: abs(k - 0.201267))
        rollout_data = rollout_parameters[closest_key]

    num_trajectories_plotted = 400
    rollout_data = rollout_data[:num_trajectories_plotted]

    from forward_models import SOIL_DEPTH_MM
    from main import (
        INORGANIC_N_MODEL_DT_DAYS,
        LEACHING_AVERAGE_WINDOW_DAYS,
        LEACHING_CONCENTRATION_LIMIT_MG_PER_LITER
    )

    LEACHING_LIMIT = 10.0
    soil_depth_m = SOIL_DEPTH_MM / 1000.0

    if not dry_run:
        import matplotlib.pyplot as plt

        FONT_SIZE = 14
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 1,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE,
            "figure.autolayout": True,
            "axes.edgecolor": "#262626",
            "axes.linewidth": 0.8,
            "grid.color": "#E6E6E6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6
        })

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4))
        print(f"[{file_name}] Highlighting trajectory index: {target_idx}")

    number_of_trajectories_violating = 0

    constraint_values_list = []
    plant_N_hourly_list = []
    hours_list = []
    is_violating_list = []

    # --- PLOTTING & EVALUATION PASS ---
    for idx, traj in enumerate(rollout_data):
        leaching_hourly = safe_get_val(traj, 'nitrate_leaching_gN_per_m3_per_day')
        plant_N_hourly_val = safe_get_val(traj, 'plant_accumulated_N_gN_per_m3')
        leakage_hourly = safe_get_val(traj, 'leakage_rate_mm_per_day')
        amm_leaching_hourly = safe_get_val(traj, 'ammonium_leaching_gN_per_m3_per_day')

        if leaching_hourly is None or plant_N_hourly_val is None:
            continue

        # Calculate violation status and constraint values
        constraint_values = np.zeros_like(leaching_hourly)
        is_violating = False

        if leakage_hourly is not None and amm_leaching_hourly is not None:
            for i in range(0, len(leakage_hourly), LEACHING_AVERAGE_WINDOW_DAYS):
                total_L = np.sum(leakage_hourly[i:(i + LEACHING_AVERAGE_WINDOW_DAYS)])
                low, high = int(i / INORGANIC_N_MODEL_DT_DAYS), int(
                    (i + LEACHING_AVERAGE_WINDOW_DAYS) / INORGANIC_N_MODEL_DT_DAYS)
                val = (np.sum(amm_leaching_hourly[low:high]) + np.sum(
                    leaching_hourly[low:high])) * INORGANIC_N_MODEL_DT_DAYS / total_L if total_L > 0 else 0
                constraint_values[low:high] = val * 1000.0
                if val * 1000.0 > LEACHING_LIMIT:
                    is_violating = True

        constraint_values_list.append(constraint_values)
        is_violating_list.append(is_violating)

        if is_violating:
            number_of_trajectories_violating += 1

        hours = np.arange(1, len(leaching_hourly) + 1)

        if not dry_run:
            # STYLING: Highlight target, standard red for violators, gray for others
            if idx == target_idx:
                color, lw, alpha, zorder = 'red', 3, 1.0, 5
                mean_violation = np.mean(constraint_values)
                print(f"Avg violation over season for target index {idx}: {mean_violation:.4f}")
            elif is_violating:
                color, lw, alpha, zorder = 'lightcoral', 0.4, 0.4, 1
            else:
                color, lw, alpha, zorder = 'gray', 0.4, 0.7, 1

            ax1.plot(hours, constraint_values, color=color, alpha=alpha, linewidth=lw, zorder=zorder)
            ax3.plot(hours, plant_N_hourly_val * soil_depth_m, color=color, alpha=alpha, linewidth=lw, zorder=zorder)

        plant_N_hourly_list.append(plant_N_hourly_val * soil_depth_m)
        hours_list.append(hours)

    if not dry_run:
        ax1.axhline(y=LEACHING_LIMIT, color='red', linestyle='--', linewidth=1.5,
                    label=f'Limit ({LEACHING_LIMIT} mg/L)', zorder=6)
        ax1.set(xlabel='Time (Hours)', ylabel='Nitrogen Concentration (mg N / L)', yscale='log', xlim=(0,2591))
        ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#262626', framealpha=1.0)

        ax3.set(xlabel='Time (Hours)', ylabel='Accumulated Plant N (g N / m²)', ylim=(0, 30), xlim=(0,2591))

        for ax in [ax1, ax3]:
            ax.grid(True)
            ax.set_facecolor('white')
            ax.tick_params(direction='in', top=True, right=True, colors='#262626', labelsize=FONT_SIZE - 1)

        # Save the file
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {output_file}")

        # Display window and hold execution until user manually closes it
        plt.show()
        plt.close(fig)

    # check number of trajectories where avg of constraint values is above constraint:
    counter = sum(1 for cv in constraint_values_list if np.mean(cv) >= LEACHING_LIMIT)

    if not dry_run:
        print("Number of trajectories violated: ", number_of_trajectories_violating)
        print("Leaching Limit over full season averaged violated by trajectories:" + str(counter))

    return constraint_values_list, plant_N_hourly_list, hours_list, is_violating_list


# ==========================================
# EXECUTION SCRIPT
# ==========================================

print("--- Running initial data assessment (Dry Run) ---")
# 1. Run a 'dry pass' to extract profiles without generating UI plots yet
cv_3, _, _, is_viol_3 = plot_results_for_one_controller("3_rollout_objects.pkl", dry_run=True)
cv_36, _, _, is_viol_36 = plot_results_for_one_controller("36_rollout_objects.pkl", dry_run=True)

# 2. Find a specific trajectory index matching the combined criteria:
#    Violates constraint window(s) locally but its full seasonal average stays below 10 mg/L
LEACHING_LIMIT = 10.0
found_target_idx = -1

for idx in range(min(len(is_viol_3), len(is_viol_36))):
    # Check conditions for 3-day controller
    violates_locally_3 = is_viol_3[idx]
    avg_safe_3 = np.mean(cv_3[idx]) < LEACHING_LIMIT

    # Check conditions for 36-day controller
    violates_locally_36 = is_viol_36[idx]
    avg_safe_36 = np.mean(cv_36[idx]) < LEACHING_LIMIT

    if violates_locally_3 and avg_safe_3 and violates_locally_36 and avg_safe_36:
        found_target_idx = idx
        break

if found_target_idx != -1:
    print(
        f"\nSUCCESS: Found trajectory index {found_target_idx} which violates local windows but remains safe on average for BOTH controllers.")
else:
    print(
        "\nWARNING: Could not find a single trajectory index meeting criteria for both. Defaulting highlight index to -1.")

# 3. Generate the actual plots highlighting the identified trajectory for both controllers
if found_target_idx != -1:
    print("\n--- Generating plots for 3-day controller (Close this window to see the next plot) ---")
    constraint_values_list_3, plant_N_hourly_list_3, hours_list_3, _ = plot_results_for_one_controller(
        file_name="3_rollout_objects.pkl",
        target_idx=found_target_idx,
        output_file="trajectory_plots_3.pdf",
        dry_run=False
    )

    print("\n--- Generating plots for 36-day controller ---")
    constraint_values_list_36, plant_N_hourly_list_36, hours_list_36, _ = plot_results_for_one_controller(
        file_name="36_rollout_objects.pkl",
        target_idx=found_target_idx,
        output_file="trajectory_plots_36.pdf",
        dry_run=False
    )