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

period_list = [3,6,36]

# LOAD ROLLOUT DATA
for period in period_list:
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
for cntrl_period in period_list:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        plt.plot(subset["violated"],
                 subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN, label=f"Every {cntrl_period} days", linewidth=5)

plt.xlabel("Probability of violation")
plt.ylabel("Crop Yield Value (USD/m$^2$)")
plt.xlim(0.4, 0.65)
plt.ylim(0.2,0.27)
# plt.title("Rollout Pareto Fronts")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("rollout_pareto_fronts.png", dpi=300)

# PLOT 2: P(violation) vs fine
plt.figure('violation_vs_fine', figsize=(8, 5))
for cntrl_period in period_list:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        plt.plot(subset.index, subset["violated"], label=f"Every {cntrl_period} days", linewidth=5)

plt.ylabel("Probability of violation")
plt.xlabel("Fine")
# plt.title("Probability of violation vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("rollout_violation_vs_fine.png", dpi=300)

# PLOT 3: Probability that total N leaching is greater than some value over different costs
plt.figure('n_leaching_vs_cost', figsize=(8, 5))
# Use linspace to generate a range of N leaching values, then sum the number of times in the rollouts that the total N leaching exceeds those values for each cost and control period
n_leaching_thresholds = np.linspace(0, 20, 20)  # Example thresholds from 0 to 100 kg/ha
for cntrl_period in period_list:
    if cntrl_period in full_df.index.get_level_values(0):
        subset = full_df.loc[cntrl_period]
        probabilities = []
        for threshold in n_leaching_thresholds:
            prob = (subset["total_leaching_gN_per_m2"] > threshold).mean()
            probabilities.append(prob)
        plt.plot(n_leaching_thresholds, probabilities, label=f"Every {cntrl_period} days", linewidth=5)
plt.xlabel("N leaching threshold (kg/ha)")
plt.ylabel("Probability of exceeding threshold")
# plt.title("Probability of exceeding N leaching thresholds vs cost")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 8)
plt.ylim(0, 0.1)

# PLOT 4: Profit with fines against fines
plt.figure('profit_with_fines', figsize=(8, 5))
for cntrl_period in period_list:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        profit_with_fines = subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN - subset.index
        plt.plot(subset.index, profit_with_fines, label=f"Every {cntrl_period} days", linewidth=5)
plt.xlabel("Fine")
plt.ylabel("Profit with fines")
# plt.title("Profit with fines vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# PLOT 5: Profit without fines against fines
plt.figure('profit_without_fines', figsize=(8, 5))
for cntrl_period in period_list:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        profit_without_fines = subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN
        plt.plot(subset.index, profit_without_fines, label=f"Every {cntrl_period} days", linewidth=5)
plt.xlabel("Fine")
plt.ylabel("Profit without fines")
# plt.title("Profit without fines vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
















# PLOT 6: Rollout scatter plot of N leaching against relative yield for different control periods; 2x2, depending on whether there is rain during the first forecast and second forecast for the respective control frequency
# Setup and Filter Data
fig, axs = plt.subplots(2, 2, figsize=(8, 5))

# Find the penalty value closest to 0.2
penalties = full_df.index.get_level_values(1).unique()
closest_penalty = penalties[np.argmin(np.abs(penalties - 0.2))]

# Extract data for the closest penalty and drop that level from the index
df = full_df.xs(closest_penalty, level=1).copy()

# Ensure index levels have names so we can reset and use them as columns safely
# Forcefully rename the remaining 2 index levels before resetting
df.index.names = ['cntrl_period', 'rollout_index']
df = df.reset_index()

# Vectorize the Yield Calculation
df['plot_y'] = df['relative_yield'] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN


# Fast Calculation for Rain Conditions
def check_second_rain(row):
    # Extract weather data using the rollout index
    rainfall_data = rollout_parameters[row['rollout_index']].rainfalls
    cp = row['cntrl_period']
    rain_slice = rainfall_data[cp: cp + 3]

    # Note: I used np.any() to check if it rained on *any* of the 3 days.
    # Your original code used `[0] and [1] and [2]`, which requires rain on *all* 3 days.
    # Change to np.all(rain_slice > 0) if you strictly need rain on every single day.
    return np.any(rain_slice > 0)


df['rainy_second_three_days'] = df.apply(check_second_rain, axis=1)

# Ensure the first rain column is boolean to match
df['rainy_start'] = df['rainy_start'].astype(bool)

# Group Data by Subplot Conditions and Plot in Bulk
# We can map the booleans directly to subplot matrix coordinates!
# not rain1 (0) -> col 0 | rain1 (1) -> col 1
# not rain2 (0) -> row 0 | rain2 (1) -> row 1
# Add initial soil moisture to the groupby for filtering; it is in the rollout parameters, so we need to map it using the rollout index
df['initial_moisture'] = df['rollout_index'].apply(lambda idx: rollout_parameters[idx].initial_moisture)


grouped = df.groupby(['rainy_second_three_days', 'rainy_start', 'cntrl_period'])




# sum the amount of rain during the first control_period days and filter for data where this sum is lower than a threshold (e.g., 5 mm) to ensure we are only looking at rollouts where there was not a lot of rain during the first control period, which would confound the results
# Filter out data from rollouts with high initial soil moisture >0.1
#grouped = [(name, group[group['initial_moisture'] <= 0.09]) for name, group in grouped] --- I filtered these before
# --- And plotted like this:
# for ax in axs.flat:
#     ax.set_ylabel("Total N leaching (gN/m2)")
#     ax.set_xlabel("Relative Yield Value")
#     ax.grid(True, linestyle='--', alpha=0.6)
#
#     # Optional: Add deduplicated legends to each subplot
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     if by_label:
#         ax.legend(by_label.values(), by_label.keys())
#
# plt.tight_layout()
# plt.savefig("rollout_scatter_leaching_vs_yield.png", dpi=300)

# Instead I want to sort all those points that had high humidity to the case where rain happened the first days:


# --- PLOT 6: REVISED CATEGORIZATION --- (no moisture differentiation!!!)
# Color mapping provided
colors = {3: 'tab:blue', 6: 'tab:orange', 9: 'tab:red', 36: 'tab:green'}
import matplotlib.pyplot as plt


moisture_limit = 0.1 # 0.1

# 1. Map row indices for control periods
unique_periods = sorted(df['cntrl_period'].unique())
period_to_row = {period: i for i, period in enumerate(unique_periods)}

# 2. Setup the 3x2 grid
# Use sharex=True and hspace to bring plots closer together
# 1. Create the figure and axes
fig, axs = plt.subplots(3, 2, figsize=(8, 5), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.05, 'wspace': 0.1})

# 2. Assign the figure title
#fig.suptitle(f"2x3, Moisture: {moisture_limit}")

# 3. Iterate through unique periods to populate rows
for period in unique_periods:
    row_idx = period_to_row[period]
    period_df = df[df['cntrl_period'] == period]

    for rollout_idx in period_df['rollout_index'].unique():
        sub_group = period_df[period_df['rollout_index'] == rollout_idx]

        rain1 = sub_group['rainy_start'].iloc[0]
        rain2 = sub_group['rainy_second_three_days'].iloc[0]
        moist = sub_group['initial_moisture'].iloc[0] > moisture_limit

        col_idx = 1 if (rain1 or moist) else 0
        color = colors.get(sub_group['cntrl_period'].iloc[0])

        axs[row_idx, col_idx].scatter(
            sub_group['plot_y'],
            sub_group['total_leaching_gN_per_m2'],
            alpha=0.5,
            edgecolors='none',
            s=70,
            c=color
        )

# Formatting
for i, period in enumerate(unique_periods):
    # Y-axis labels for the left column only
    axs[i, 0].set_ylabel(f"Every {period} days\nLeaching (gN/m$^2$)", fontsize = 'small')

axs[0, 0].set_title("Dry Start", pad=10)
axs[0, 1].set_title("Wet Start", pad=10)

for i, ax in enumerate(axs.flat):
    ax.grid(True, linestyle='--', alpha=0.4)

    # Logic: Only set xlabel for the bottom row (indices 4 and 5 in a 3x2 grid)
    if i >= 4:
        ax.set_xlabel("Profit (USD/m$^2$)", fontsize = 'small')
    else:
        # Explicitly remove xlabel for upper plots
        ax.set_xlabel("")

# Final adjustment to prevent labels from overlapping
plt.subplots_adjust(bottom=0.1)
plt.show()






# --- PLOT 6: REVISED CATEGORIZATION ---

# Find the penalty value closest to 0.2
penalties = full_df.index.get_level_values(1).unique()
closest_penalty = penalties[np.argmin(np.abs(penalties - 0.2))]

# Extract data and reset index
df = full_df.xs(closest_penalty, level=1).copy()
df.index.names = ['cntrl_period', 'rollout_index']
df = df.reset_index()

# Vectorize the Yield Calculation
df['plot_y'] = df['relative_yield'] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN

# 1. Map initial moisture from parameters
df['initial_moisture'] = df['rollout_index'].apply(lambda idx: rollout_parameters[idx].initial_moisture)


# 2. Define the second rain check
def check_second_rain(row):
    rainfall_data = rollout_parameters[row['rollout_index']].rainfalls
    cp = row['cntrl_period']
    rain_slice = rainfall_data[cp: cp + 3]
    return np.any(rain_slice > 0)


df['rainy_second_three_days'] = df.apply(check_second_rain, axis=1)

# 3. MERGE LOGIC: Treat high initial moisture as if it rained in the first period
# This creates the "High Risk" category you described
df['high_moisture_flag'] = df['initial_moisture'] > 0.1
df['combined_early_risk'] = df['rainy_start'].astype(bool) | df['high_moisture_flag']

# Grouping using the combined risk flag instead of just rainy_start
grouped = df.groupby(['rainy_second_three_days', 'combined_early_risk', 'cntrl_period'])

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(8, 5))

for (rain2, risk1, cntrl_period), group in grouped:
    # row = rain in 2nd period, col = (rain in 1st period OR high initial moisture)
    row, col = int(rain2), int(risk1)

    axs[row, col].scatter(
        group['plot_y'],
        group['total_leaching_gN_per_m2'],
        label=f"Every {cntrl_period} days",
        alpha=0.6,
        edgecolors='none',
        s=80
    )

# Formatting the specific subplot logic
axs[0, 0].set_title("Dry Start, No Late Rain")
axs[0, 1].set_title("Wet Start, No Late Rain")
axs[1, 0].set_title("Dry Start, Late Rain")
axs[1, 1].set_title("Wet Start, Late Rain")

for ax in axs.flat:
    ax.set_ylabel("Total N leaching (gN/m2)")
    ax.set_xlabel("Profit without fines (USD/m2)")
    ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()









# --- PLOT 6: REVISED CATEGORIZATION WITH PRE-PERIOD RAIN ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PLOT 6: DYNAMIC TRANSITION WINDOWS ---

# 1. Setup Data
penalties = full_df.index.get_level_values(1).unique()
closest_penalty = penalties[np.argmin(np.abs(penalties - 0.2))]

df = full_df.xs(closest_penalty, level=1).copy()
df.index.names = ['cntrl_period', 'rollout_index']
df = df.reset_index()

# Constants
LATE_RAIN_THRESHOLD = 0.4
df['plot_y'] = df['relative_yield'] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN
df['initial_moisture'] = df['rollout_index'].apply(lambda idx: rollout_parameters[idx].initial_moisture)


# 2. Define the Dynamic Logic
def analyze_weather_windows(row):
    cp = row['cntrl_period']

    # EXEMPTION: Keep the 36-day controller in the top plots (Dry Transition)
    # because its second period hasn't logically "started" in a way comparable to the others.
    if cp == 36:
        return False

    p = rollout_parameters[row['rollout_index']]

    # DYNAMIC WINDOW: 3 days before the start of the second control period
    # cp=3 -> [0:3], cp=6 -> [3:6], cp=9 -> [6:9]
    start_idx = max(0, cp - 3)
    end_idx = cp

    pre_second_period_rain = p.rainfalls[start_idx: end_idx].sum()
    high_moisture_surrogate = pre_second_period_rain > LATE_RAIN_THRESHOLD

    # Check for actual rain in the first 3 days of the second control period
    actual_second_period_rain = np.any(p.rainfalls[cp: cp + 3] > 0)

    return high_moisture_surrogate or actual_second_period_rain


# 3. Apply Categories
df['rainy_second_risk'] = df.apply(analyze_weather_windows, axis=1)

# Early Risk: Initial moisture > 0.1 OR rain in the first 3 days of the sim
df['high_moisture_start'] = df['initial_moisture'] > 0.1
df['combined_early_risk'] = df['rainy_start'].astype(bool) | df['high_moisture_start']

# 4. Plotting
fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)

grouped = df.groupby(['rainy_second_risk', 'combined_early_risk', 'cntrl_period'])

for (rain2, risk1, cntrl_period), group in grouped:
    row, col = int(rain2), int(risk1)

    axs[row, col].scatter(
        group['plot_y'],
        group['total_leaching_gN_per_m2'],
        label=f"Every {cntrl_period} days",
        alpha=0.6,
        s=80,
        edgecolors='none'
    )

# Formatting
axs[0, 0].set_title("Dry Start, No Late Rain")
axs[0, 1].set_title("Wet Start, No Late Rain")
axs[1, 0].set_title("Dry Start, Late Rain")
axs[1, 1].set_title("Wet Start, Late Rain")

for i, ax in enumerate(axs.flat):
    ax.set_ylabel("Total N leaching (gN/m2)")
    ax.set_xlabel("Profit without fines (USD/m2)")
    ax.grid(True, linestyle='--', alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if by_label:
        # Check if this is the top-right plot (index 1 in the flat list)
        if i == 1:
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper left',
                bbox_to_anchor=(0.0, 0.6),  # Adjust 0.85 lower or higher as needed
                fontsize='small'
            )
        else:
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='small')

plt.tight_layout()
plt.savefig("rollout_scatter_dynamic_windows.png", dpi=300)
plt.show()






















# PLOT 7: P(violation) vs relative yield for non-violating rollouts only (same as 1. but the profit is only averaged over non-violating rollouts (lineplot)
means = full_df.groupby(["control_period", "penalty"]).mean()
success_means = full_df[~full_df["violated"]].groupby(["control_period", "penalty"]).mean()

plt.figure('Rollout Pareto Fronts (Success-Only)', figsize=(8, 5))
colors = {3: 'tab:blue', 6: 'tab:orange', 9: 'tab:green', 36: 'tab:red'}

for cntrl_period in period_list:
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
                 color=colors.get(cntrl_period), linewidth=5)

plt.xlabel("Probability of violation")
plt.ylabel("Mean Profit of Non-Violating Runs (USD/m2)")
# plt.title("Pareto Fronts: Yield of Successful Trajectories")
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

# import pandas as pd
# import numpy as np
# import numpy.typing as npt
# import matplotlib.pyplot as plt
# import pickle as pkl
# from dataclasses import dataclass, field, asdict
#
# from main import N_SUFFICIENT_YIELD_G_GRAIN_PER_M2, CORN_PRICE_USD_PER_G_GRAIN
#
# @dataclass
# class StochasticParameters:
#     rainfalls: npt.NDArray[np.float64]
#     initial_moisture: float
#     initial_ammonium: float
#     initial_nitrate: float
#
# rollouts_by_control_period = {}
# dtypes_dict = {"control_period": np.int32, "parameters_idx": np.int32, "violated": np.bool}
# with open("3_days_rollouts.csv", "r") as f:
#     rollouts_by_control_period[3] = pd.read_csv(f, dtype=dtypes_dict)
# with open("9_days_rollouts.csv", "r") as f:
#     rollouts_by_control_period[9] = pd.read_csv(f, dtype=dtypes_dict)
# with open("36_days_rollouts.csv", "r") as f:
#     rollouts_by_control_period[36] = pd.read_csv(f, dtype=dtypes_dict)
# with open("6_days_rollouts.csv", "r") as f:
#     rollouts_by_control_period[6] = pd.read_csv(f, dtype=dtypes_dict)
# rollouts_by_control_period[3]
#
# with open("rollout_parameters.pkl", "rb") as f:
#     rollout_parameters = pkl.load(f)
# rollout_parameters
#
# for period_days, rollouts_df in rollouts_by_control_period.items():
#     # rollouts_df["rainfall_idx"] = list(range(400)) * 3
#     # rollouts_df["control_period"] = [period_days] * 1200
#     rollouts_df.set_index(["control_period", "penalty", "parameters_idx"], inplace=True)
#
#
# full_df = pd.concat(list(rollouts_by_control_period.values()))
# full_df["relative_yield"] = 1.0 - (full_df["deficit_cost"]  / N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 / CORN_PRICE_USD_PER_G_GRAIN)
# full_df["rainy_start"] = [(stochastic_parameters.rainfalls[0] > 0 or stochastic_parameters.rainfalls[1] > 0 or stochastic_parameters.rainfalls[2] > 0) for stochastic_parameters in rollout_parameters] * int(len(full_df) / len(rollout_parameters))
# full_df
#
# means = full_df.groupby(["control_period", "penalty"]).mean()
#
# plt.plot(full_df.groupby(["control_period", "penalty"]).mean().loc[3]["violated"], full_df.groupby(["control_period", "penalty"]).mean().loc[3]["relative_yield"])
#
# for cntrl_period in [3, 6, 9, 36]:
#     plt.plot(means.loc[cntrl_period]["violated"], means.loc[cntrl_period]["relative_yield"], label="Every "+str(cntrl_period))
# plt.xlabel("Probability of violation")
# plt.ylabel("Relative Yield")
# plt.title("Rollout pareto fronts")
# plt.legend()
# plt.savefig("rollout_pareto_fronts.png", transparent=False, dpi=300, bbox_inches='tight')
# plt.xlim(0.3, 0.75)
# plt.ylim(0.6, 1.0)
#
#
# full_df.groupby(["control_period", "penalty"]).std()
#
#
# for name, group in full_df.groupby(['control_period', 'penalty']):
#     print(name)
#
# leaching_by_penalty = [group['total_leaching_gN_per_m2'].values for name, group in full_df.groupby(['control_period', 'penalty'])]
#
# # Get the penalty labels for the x-axis
# penalty_labels = [str(name[0]) + "_days_" + str(np.round(name[1], 1)) + "_penal" for name, group in full_df.groupby(['control_period', 'penalty'])]
#
# # Create the violin plot
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.violinplot(leaching_by_penalty)
#
# # Set x-axis labels
# ax.set_xticks(range(1, len(penalty_labels) + 1))
# ax.set_xticklabels(penalty_labels)
# ax.tick_params(axis='x', labelrotation=90)
#
# ax.set_ylabel('Total Leaching (gN/m²)')
# ax.set_xlabel('Penalty')
# ax.set_title('Distribution of Total Leaching For Violating Trajectories by Penalty')
# ax.set_ylim([0,2])
#
# plt.show()
#
#
# fine_USD = full_df.index.get_level_values(1).unique()[1]
# fig, ax = plt.subplots(figsize=(10, 6))
# for control_period_days in full_df.index.get_level_values(0).unique():
#     selected_rollouts = full_df.loc[(control_period_days, fine_USD)]
#     ax.scatter(selected_rollouts[selected_rollouts["rainy_start"]]["relative_yield"], selected_rollouts[selected_rollouts["rainy_start"]]["total_leaching_gN_per_m2"], label="every " + str(control_period_days) + ", rainy start")
#     ax.scatter(selected_rollouts[~selected_rollouts["rainy_start"]]["relative_yield"], selected_rollouts[~selected_rollouts["rainy_start"]]["total_leaching_gN_per_m2"], label="every " + str(control_period_days) + ", clear start")
# ax.set_ylabel("Total N leaching (g/m^2)")
# ax.set_xlabel("Relative yield")
# ax.set_title("Relative yield vs N leaching, for $" + str(round(fine_USD, 2)) +" fine")
# ax.legend()
# fig.savefig("leaching_vs_yield_across_frequencies.png", transparent=False, dpi=300, bbox_inches='tight')