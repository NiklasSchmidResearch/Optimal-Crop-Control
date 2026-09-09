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



def plot_results_for_one_controller(file_name):


    # 2. ABSOLUTE PATH DIAGNOSIS
    target_dir = r"C:\Users\nikla\Documents\GitHub\Optimal-Crop-Control"
    full_path = os.path.join(target_dir, file_name)

    # 3. LOAD THE FILE
    rollout_parameters = None
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        with open(full_path, "rb") as f:
            try:
                rollout_parameters = pkl.load(f)
            except Exception as e:
                print(f"Error loading pickle: {e}")

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


    def safe_get_val(obj, attr):
        if hasattr(obj, attr): return getattr(obj, attr)
        try:
            return obj[attr]
        except:
            return None


    # --- SEARCH PASS: Identify candidates for highlight ---
    candidate_indices = []
    for idx, traj in enumerate(rollout_data):
        leakage = safe_get_val(traj, 'leakage_rate_mm_per_day')
        if leakage is None: continue

        viol_count = 0
        leaching = safe_get_val(traj, 'nitrate_leaching_gN_per_m3_per_day')
        amm_leaching = safe_get_val(traj, 'ammonium_leaching_gN_per_m3_per_day')

        for i in range(0, len(leakage), LEACHING_AVERAGE_WINDOW_DAYS):
            total_leakage = np.sum(leakage[i:(i + LEACHING_AVERAGE_WINDOW_DAYS)])
            if total_leakage > 0:
                idx_low = int(i / INORGANIC_N_MODEL_DT_DAYS)
                idx_high = int((i + LEACHING_AVERAGE_WINDOW_DAYS) / INORGANIC_N_MODEL_DT_DAYS)
                val = (np.sum(amm_leaching[idx_low:idx_high]) + np.sum(
                    leaching[idx_low:idx_high])) * INORGANIC_N_MODEL_DT_DAYS / total_leakage
                if val * 1000.0 > LEACHING_CONCENTRATION_LIMIT_MG_PER_LITER:
                    viol_count += 1

        if viol_count == 1:
            candidate_indices.append(idx)

    print(f"Trajectories violating exactly once: {candidate_indices}")

    # --- SELECTION ---
    # Change the index below to any number from candidate_indices to pick a different one
    target_idx = candidate_indices[3] if candidate_indices else -1
    print(f"Highlighting trajectory index: {target_idx}")


    number_of_trajectories_violating = 0

    constraint_values_list = []
    plant_N_hourly_list = []
    hours_list = []
    # --- PLOTTING PASS ---
    for idx, traj in enumerate(rollout_data):
        leaching_hourly = safe_get_val(traj, 'nitrate_leaching_gN_per_m3_per_day')
        plant_N_hourly = safe_get_val(traj, 'plant_accumulated_N_gN_per_m3')
        leakage_hourly = safe_get_val(traj, 'leakage_rate_mm_per_day')
        amm_leaching_hourly = safe_get_val(traj, 'ammonium_leaching_gN_per_m3_per_day')

        if leaching_hourly is None or plant_N_hourly is None: continue

        # Calculate violation status and constraint values
        constraint_values = np.zeros_like(leaching_hourly)
        constraint_values_list.append(constraint_values)
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

        if is_violating:
            number_of_trajectories_violating += 1

        # STYLING:
        # Highlight target, standard red for violators, gray for others
        if idx == target_idx:
            color, lw, alpha, zorder = 'red', 3, 1.0, 5
            mean_violation = np.mean(constraint_values)
            print("avg violation over season for trarget index:"+str(mean_violation))
        elif is_violating:
            color, lw, alpha, zorder = 'lightcoral', 0.4, 0.4, 1
        else:
            color, lw, alpha, zorder = 'gray', 0.4, 0.7, 1

        hours = np.arange(1, len(leaching_hourly) + 1)
        ax1.plot(hours, constraint_values, color=color, alpha=alpha, linewidth=lw, zorder=zorder)
        ax3.plot(hours, plant_N_hourly * soil_depth_m, color=color, alpha=alpha, linewidth=lw, zorder=zorder)
        plant_N_hourly_list.append(plant_N_hourly * soil_depth_m)
        hours_list.append(hours)

    ax1.axhline(y=LEACHING_LIMIT, color='red', linestyle='--', linewidth=1.5, label=f'Limit ({LEACHING_LIMIT} mg/L)',
                zorder=6)
    ax1.set(xlabel='Time (Hours)', ylabel='Nitrogen Concentration (mg N / L)', yscale='log')
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#262626', framealpha=1.0)

    ax3.set(xlabel='Time (Hours)', ylabel='Accumulated Plant N (g N / m²)', ylim=(0, 30))

    for ax in [ax1, ax3]:
        ax.grid(True)
        ax.set_facecolor('white')
        ax.tick_params(direction='in', top=True, right=True, colors='#262626', labelsize=FONT_SIZE - 1)

    plt.savefig("trajectory_plots.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Plot saved as trajectory_plots.pdf")


    # check number of trajectories where avg of constraint values is above constraint:

    counter = 0
    for constraint_values in constraint_values_list:
        mean_violation = np.mean(constraint_values)

        if mean_violation >= LEACHING_LIMIT:
            counter +=1


    print("Number of trajectories violated: ", number_of_trajectories_violating)
    print("Leaching Limit over full season averaged violated by trajectories:" + str(counter))
    return constraint_values_list, plant_N_hourly_list, hours_list


constraint_values_list_3, plant_N_hourly_list_3, hours_list_3 = plot_results_for_one_controller(file_name = "3_rollout_objects.pkl")
constraint_values_list_36, plant_N_hourly_list_36, hours_list_36 = plot_results_for_one_controller(file_name = "36_rollout_objects.pkl")

# Find a trajectory index where the 36 day controller is not violating (constraint_value above 10 mg/L) but the 3 day controller is.
# Generate plots like those above where not a trajectory is highlighted that violates for a single time-window, but this particular trajectory that we found.