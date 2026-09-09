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

        subset = means.loc[cntrl_period].copy()

        # Calculate revenue to match the plot logic
        subset["revenue"] = (subset["relative_yield"] *
                             N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 *
                             CORN_PRICE_USD_PER_G_GRAIN)

        # Clean export for PGFPlots
        filename = f"pareto_period_{cntrl_period}.csv"
        subset[["violated", "revenue"]].to_csv(filename, index=False)

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
        filename = f"violation_period_{cntrl_period}.csv"
        subset[["violated"]].to_csv(filename, index=True, index_label="idx")

plt.ylabel("Probability of violation")
plt.xlabel("Fine")
# plt.title("Probability of violation vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("rollout_violation_vs_fine.png", dpi=300)

# PLOT 5: Profit without fines against fines
plt.figure('profit_without_fines', figsize=(8, 5))
for cntrl_period in period_list:
    if cntrl_period in means.index.get_level_values(0):
        subset = means.loc[cntrl_period]
        profit_without_fines = subset["relative_yield"] * N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN
        plt.plot(subset.index, profit_without_fines, label=f"Every {cntrl_period} days", linewidth=5)

        profit_without_fines = (subset["relative_yield"] *
                                N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 *
                                CORN_PRICE_USD_PER_G_GRAIN)

        # Save to CSV including the index
        filename = f"profit_period_{cntrl_period}.csv"
        export_df = pd.DataFrame({"profit": profit_without_fines}, index=subset.index)
        export_df.to_csv(filename, index=True, index_label="idx")
plt.xlabel("Fine")
plt.ylabel("Profit without fines")
# plt.title("Profit without fines vs fine")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
















# PLOT 6: Rollout scatter plot of N leaching against relative yield for different control periods; 2x2, depending on whether there is rain during the first forecast and second forecast for the respective control frequency
# Setup and Filter Data

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

import pandas as pd

# Define your logic-based rows and columns
period_to_row = {3: 0, 6: 1, 36: 2}

for period in unique_periods:
    row_idx = period_to_row[period]
    period_df = df[df['cntrl_period'] == period]

    for col_idx in [0, 1]:
        # Apply the logic: col_idx 1 if (rain1 or moist)
        if col_idx == 1:
            mask = (period_df['rainy_start'] == True) | (period_df['initial_moisture'] > moisture_limit)
        else:
            mask = ~((period_df['rainy_start'] == True) | (period_df['initial_moisture'] > moisture_limit))

        subset = period_df[mask]

        if not subset.empty:
            filename = f"scatter_r{row_idx}_c{col_idx}.csv"
            # Export x (plot_y) and y (total_leaching)
            subset[['plot_y', 'total_leaching_gN_per_m2']].to_csv(filename, index=False)






plt.show(block=True)