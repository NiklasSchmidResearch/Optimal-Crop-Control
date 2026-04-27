import sys
import matplotlib

# FORCE WINDOWS BACKEND - Must be before importing pyplot
# This fixes the 'backend_macosx' error on Windows
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pickle as pkl
from dataclasses import dataclass

# IMPORT CONSTANTS
from main import N_SUFFICIENT_YIELD_G_GRAIN_PER_M2, CORN_PRICE_USD_PER_G_GRAIN

# DEFINE THE CLASS
@dataclass
class StochasticParameters:
    rainfalls: npt.NDArray[np.float64]
    initial_moisture: float
    initial_ammonium: float
    initial_nitrate: float

# NAMESPACE BRIDGE (Prevents the Pickle AttributeError)
import __main__
__main__.StochasticParameters = StochasticParameters

rollouts_by_control_period = {}
dtypes_dict = {"control_period": np.int32, "parameters_idx": np.int32, "violated": bool}

# LOAD ROLLOUT DATA
for period in [3, 6, 9, 36]:
    filename = f"{period}_days_rollouts.csv"
    try:
        with open(filename, "r") as f:
            rollouts_by_control_period[period] = pd.read_csv(f, dtype=dtypes_dict)
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")

# LOAD ROLLOUT PARAMETERS
with open("rollout_parameters.pkl", "rb") as f:
    rollout_parameters = pkl.load(f)

# DATA PROCESSING
for period_days, rollouts_df in rollouts_by_control_period.items():
    rollouts_df.set_index(["control_period", "penalty", "parameters_idx"], inplace=True)

if not rollouts_by_control_period:
    print("No CSV data found. Exiting.")
    sys.exit()

full_df = pd.concat(list(rollouts_by_control_period.values()))

# CALCULATIONS
full_df["relative_yield"] = 1.0 - (full_df["deficit_cost"] / (N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN))

# Map rain flags
rainy_flags = [(p.rainfalls[0] > 0 or p.rainfalls[1] > 0 or p.rainfalls[2] > 0) for p in rollout_parameters]
full_df["rainy_start"] = rainy_flags * (len(full_df) // len(rollout_parameters))

means = full_df.groupby(["control_period", "penalty"]).mean()

# PLOT 1: Pareto Fronts
plt.figure('Rollout Pareto Fronts', figsize=(8, 5))
for cntrl_period in [3, 6, 9, 36]:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        plt.plot(subset["violated"],
                 subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN, label=f"Every {cntrl_period} days")

plt.xlabel("Probability of violation")
plt.ylabel("Relative Yield Value")
plt.title("Rollout Pareto Fronts")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("rollout_pareto_fronts.png", dpi=300)

# PLOT 2: P(violation) vs fine
plt.figure('violation_vs_fine', figsize=(8, 5))
for cntrl_period in [3, 6, 9, 36]:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        plt.plot(subset.index, subset["violated"], label=f"Every {cntrl_period} days")

plt.ylabel("Probability of violation")
plt.xlabel("Fine")
plt.title("Probability of violation vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("rollout_violation_vs_fine.png", dpi=300)

# PLOT 3: Probability that total N leaching is greater than some value over different costs
plt.figure('n_leaching_vs_cost', figsize=(8, 5))
# Use linspace to generate a range of N leaching values, then sum the number of times in the rollouts that the total N leaching exceeds those values for each cost and control period
n_leaching_thresholds = np.linspace(0, 20, 20)  # Example thresholds from 0 to 100 kg/ha
for cntrl_period in [3, 6, 9, 36]:
    if cntrl_period in full_df.index.get_level_values(0):
        subset = full_df.loc[cntrl_period]
        probabilities = []
        for threshold in n_leaching_thresholds:
            prob = (subset["total_leaching_gN_per_m2"] > threshold).mean()
            probabilities.append(prob)
        plt.plot(n_leaching_thresholds, probabilities, label=f"Every {cntrl_period} days")
plt.xlabel("N leaching threshold (kg/ha)")
plt.ylabel("Probability of exceeding threshold")
plt.title("Probability of exceeding N leaching thresholds vs cost")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 8)
plt.ylim(0, 0.1)

# PLOT 4: Profit with fines against fines
plt.figure('profit_with_fines', figsize=(8, 5))
for cntrl_period in [3, 6, 9, 36]:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        profit_with_fines = subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN - subset.index
        plt.plot(subset.index, profit_with_fines, label=f"Every {cntrl_period} days")
plt.xlabel("Fine")
plt.ylabel("Profit with fines")
plt.title("Profit with fines vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# PLOT 5: Profit without fines against fines
plt.figure('profit_without_fines', figsize=(8, 5))
for cntrl_period in [3, 6, 9, 36]:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        profit_without_fines = subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN
        plt.plot(subset.index, profit_without_fines, label=f"Every {cntrl_period} days")
plt.xlabel("Fine")
plt.ylabel("Profit without fines")
plt.title("Profit without fines vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# PLOT 6: Rollout scatter plot of N leaching against relative yield for different control periods; 2x2, depending on whether there is rain during the first forecast and second forecast for the respective control frequency
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
for i, cntrl_period in enumerate([3, 6, 9, 36]):
    if cntrl_period in full_df.index.get_level_values(0):
        subset = full_df.loc[cntrl_period]
        ax = axes[i // 2, i % 2]
        ax.scatter(subset["total_leaching_gN_per_m2"], subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN, alpha=0.5)
        ax.set_title(f"Every {cntrl_period} days")
        ax.set_xlabel("Total N leaching (gN/m²)")
        ax.set_ylabel("Relative Yield Value")
        ax.grid(True, linestyle='--', alpha=0.6)

# Plot 7 P(violation) vs relative yield for non-violating rollouts only (same as 1. but the profit is only averaged over non-violating rollouts (lineplot)
# 1. Calculate the overall means to get the probability of violation (X-axis)
means = full_df.groupby(["control_period", "penalty"]).mean()

# 2. Calculate the means ONLY for non-violating trajectories (Y-axis)
# We filter the full_df first, then group by the same indices
success_means = full_df[~full_df["violated"]].groupby(["control_period", "penalty"]).mean()

plt.figure('Rollout Pareto Fronts (Success-Only)', figsize=(8, 5))
colors = {3: 'tab:blue', 6: 'tab:orange', 9: 'tab:green', 36: 'tab:red'}

for cntrl_period in [3, 6, 9, 36]:
    if (cntrl_period in means.index.get_level_values(0) and
            cntrl_period in success_means.index.get_level_values(0)):
        # Select the data for this control period
        m_subset = means.loc[cntrl_period]
        s_subset = success_means.loc[cntrl_period]

        # Align by penalty index to ensure X and Y points correspond
        common_penalties = m_subset.index.intersection(s_subset.index)

        # X: Probability of violation (from original dataset)
        x_vals = m_subset.loc[common_penalties, "violated"]

        # Y: Profit/Yield (only from successful runs)
        y_vals = (s_subset.loc[
                      common_penalties, "relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN)

        # Sort by X-axis to ensure the line draws correctly left-to-right
        sort_order = np.argsort(x_vals)
        plt.plot(x_vals.values[sort_order], y_vals.values[sort_order],
                 marker='o', label=f"Every {cntrl_period} days",
                 color=colors.get(cntrl_period))

plt.xlabel("Probability of violation")
plt.ylabel("Mean Profit of Non-Violating Runs (USD/m2)")
plt.title("Pareto Fronts: Yield of Successful Trajectories")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("rollout_pareto_fronts_success_only.png", dpi=300)
plt.show()


plt.show(block=True)
# Plots to generate:
# 1. P(violation) vs relative yield
# 2. P(violation) vs fine
# 3. Probability that total N leaching is greater than some value over different costs
# 4. Profit with fines against fines
# 5. Profit without fines against fines
# 6. Rollout scatter plot of N leaching against relative yield for different control periods; 2x2, depending on whether there is rain during the first forecast and second forcecast for the respective control frequency
# 7. P(violation) vs relative yield for non-violating rollouts only (same as 1. but the profit is only averaged over non-violating rollouts


