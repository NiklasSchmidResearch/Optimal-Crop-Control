import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Tuple, List
from multiprocessing import Pool

# Note: This script requires 'forward_models.py' to be in the same directory
from forward_models import (
    FIELD_CAPACITY,
    SOIL_DEPTH_MM,
    generate_rain,
    hydrology_model,
    soil_organic_model
)

# --- Configuration & Constants ---
N_YEARS = 100
TRANSIENT_PERIOD_DAYS = 365 * N_YEARS
GROWING_SEASON_DAYS = 36
YEARLY_RESIDUE_N_G_PER_M2 = 8.0
YEARLY_RESIDUE_C_G_PER_M2 = YEARLY_RESIDUE_N_G_PER_M2 * 55.0

# Initialize residue arrays
nitrogen_residue_add_gN_per_m2_per_day = np.zeros(TRANSIENT_PERIOD_DAYS)
carbon_residue_add_gC_per_m2_per_day = np.zeros(TRANSIENT_PERIOD_DAYS)

for i in range(0, TRANSIENT_PERIOD_DAYS, 365):
    nitrogen_residue_add_gN_per_m2_per_day[i] = YEARLY_RESIDUE_N_G_PER_M2
    carbon_residue_add_gC_per_m2_per_day[i] = YEARLY_RESIDUE_C_G_PER_M2

# --- Model Execution ---
rainfall_mm = generate_rain(TRANSIENT_PERIOD_DAYS)

# Hydrology Model
(soil_moisture,
 infiltration_rate_mm_per_day,
 evaporation_rate_mm_per_day,
 transpiration_rate_mm_per_day,
 leakage_rate_mm_per_day,
 moisture_effect_on_decomposition_factor,
 moisture_effect_on_nitrification_factor) = hydrology_model(0.22, rainfall_mm)

# Soil Organic Model
(litter_carbon_gC_per_m3,
 litter_nitrogen_gN_per_m3,
 microbial_carbon_gC_per_m3,
 humus_carbon_gC_per_m3,
 net_flux_to_mineral_nitrogen_gN_per_m3_per_day) = soil_organic_model(
    1200.0, 54.55, 50, 8500,
    moisture_effect_on_decomposition_factor,
    nitrogen_residue_add_gN_per_m2_per_day / (SOIL_DEPTH_MM / 1000.),
    carbon_residue_add_gC_per_m2_per_day / (SOIL_DEPTH_MM / 1000.)
)

# --- Visualization: Long Term Trends ---
x = np.linspace(0, N_YEARS, TRANSIENT_PERIOD_DAYS)
planting_days = np.linspace(365 - GROWING_SEASON_DAYS, TRANSIENT_PERIOD_DAYS - GROWING_SEASON_DAYS, N_YEARS,
                            dtype=np.float64) / 365.

# Plot Soil Moisture (Kept as original)
plt.figure()
plt.plot(x, soil_moisture * 100.)
plt.scatter(planting_days, soil_moisture[365 - GROWING_SEASON_DAYS::365] * 100., color="orange", zorder=100,
            label="Start of growing season")
plt.legend()
plt.xlabel("Time (years)")
plt.ylabel("Soil moisture (%)")
plt.title("Soil moisture over 100 years of agricultural conditions")


# Function to plot carbon/nitrogen states
def plot_state(data, label, title, units):
    plt.figure()
    plt.plot(x, data)
    planting_vals = data[365 - GROWING_SEASON_DAYS::365]
    plt.scatter(planting_days, planting_vals, color="orange", zorder=100, label="Start of growing season")

    # Calculate the mean of the later period for the steady state reference
    steady_state_mean = planting_vals[50:].mean()

    # Updated: use axhline for full period and color="red"
    plt.axhline(steady_state_mean, color="red", linestyle="--", linewidth=2.0, label="Chosen initial condition")

    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel(f"{label} ({units})")
    plt.title(title)
    return steady_state_mean


init_litter_c = plot_state(litter_carbon_gC_per_m3, "Litter Carbon", "Litter carbon over 100 years", "gC/m$^3$")
init_litter_n = plot_state(litter_nitrogen_gN_per_m3, "Litter Nitrogen", "Litter nitrogen over 100 years", "gN/m$^3$")
init_microbe_c = plot_state(microbial_carbon_gC_per_m3, "Microbial Carbon", "Microbial carbon over 100 years",
                            "gC/m$^3$")
init_humus_c = plot_state(humus_carbon_gC_per_m3, "Humus Carbon", "Humus carbon over 100 years", "gC/m$^3$")

print(f"Initial states:\n"
      f"  Litter C: {init_litter_c:.0f} gC/m^3\n"
      f"  Litter N: {init_litter_n:.1f} gN/m^3\n"
      f"  Microbial C: {init_microbe_c:.0f} gC/m^3\n"
      f"  Humus C: {init_humus_c:.0f} gC/m^3")

# --- Mineral Nitrogen Flux ---
plt.figure()
plt.plot(x, net_flux_to_mineral_nitrogen_gN_per_m3_per_day)
plt.ylim(0, net_flux_to_mineral_nitrogen_gN_per_m3_per_day.max() * 1.1)
plt.xlabel("Time (years)")
plt.ylabel("Net flux to mineral N (gN/m$^3$/day)")
plt.title("Flux to mineral nitrogen over 100 years of agricultural conditions")

# --- Growing Season Volatility ---
days_x = np.linspace(0, 35, 36)
fig, ax = plt.subplots()
states = {
    "Litter C": litter_carbon_gC_per_m3,
    "Litter N": litter_nitrogen_gN_per_m3,
    "Microbial C": microbial_carbon_gC_per_m3,
    "Humus C": humus_carbon_gC_per_m3,
    "Soil moisture": soil_moisture
}

for label, data in states.items():
    segment = data[TRANSIENT_PERIOD_DAYS - GROWING_SEASON_DAYS:]
    ax.plot(days_x, segment / np.max(segment), label=label)

ax.set_ylim(0, 1.05)
ax.legend(loc="lower left")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Normalized magnitude")
ax.set_title("Relative volatility of Porporato states over a growing season")

# --- Analysis of Initial Moisture & Leakage ---
initial_moistures = np.linspace(0.0, 1.0, 100)
resulting_moistures = np.ndarray(initial_moistures.shape)
leakages = np.ndarray(initial_moistures.shape)

for i in range(len(initial_moistures)):
    out = hydrology_model(initial_moistures[i], np.array([0.0, 0.0]))
    resulting_moistures[i] = out[0][1]
    leakages[i] = out[4][0]

plt.figure()
plt.plot(initial_moistures, resulting_moistures)
plt.title("Resulting Moisture vs Initial")

plt.figure()
plt.plot(initial_moistures, leakages)
plt.title("Leakage vs Initial Moisture")

# --- Histogram Analysis ---
MAX_MOISTURE = 0.8
MOISTURE_GRID_SIZE = 0.05
bins = np.arange(0, MAX_MOISTURE + MOISTURE_GRID_SIZE, MOISTURE_GRID_SIZE)

plt.figure()
moisture_counts, _, _ = plt.hist(soil_moisture, bins=bins)
plt.title("Soil Moisture Distribution")

# Statistics
non_leaky_idx = int(FIELD_CAPACITY / MOISTURE_GRID_SIZE)
non_leaky_moisture_counts = moisture_counts[:non_leaky_idx]
print("\nNormalized non-leaky moisture counts:")
print(non_leaky_moisture_counts / np.sum(non_leaky_moisture_counts))

final_flux_sum = np.sum(net_flux_to_mineral_nitrogen_gN_per_m3_per_day[TRANSIENT_PERIOD_DAYS - 36:])
print(f"\nSum of net flux (last 36 days): {final_flux_sum:.4f}")

plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forward_models import (
    FIELD_CAPACITY, SOIL_DEPTH_MM, generate_rain,
    hydrology_model, soil_organic_model
)

# --- Configuration ---
N_YEARS = 100
TRANSIENT_PERIOD_DAYS = 365 * N_YEARS
GROWING_SEASON_DAYS = 36
YEARLY_RESIDUE_N_G_PER_M2 = 8.0
YEARLY_RESIDUE_C_G_PER_M2 = YEARLY_RESIDUE_N_G_PER_M2 * 55.0

# Residue setup
nitrogen_add = np.zeros(TRANSIENT_PERIOD_DAYS)
carbon_add = np.zeros(TRANSIENT_PERIOD_DAYS)
for i in range(0, TRANSIENT_PERIOD_DAYS, 365):
    nitrogen_add[i] = YEARLY_RESIDUE_N_G_PER_M2
    carbon_add[i] = YEARLY_RESIDUE_C_G_PER_M2

# --- Run Models ---
rainfall = generate_rain(TRANSIENT_PERIOD_DAYS)
(soil_moisture, _, _, _, _, moisture_decomp, _) = hydrology_model(0.22, rainfall)
(litter_c, litter_n, microbe_c, humus_c, net_flux) = soil_organic_model(
    1200.0, 54.55, 50, 8500, moisture_decomp,
    nitrogen_add / (SOIL_DEPTH_MM / 1000.),
    carbon_add / (SOIL_DEPTH_MM / 1000.)
)

# Time arrays
x_years = np.linspace(0, N_YEARS, TRANSIENT_PERIOD_DAYS)
days_x = np.linspace(0, GROWING_SEASON_DAYS-1, GROWING_SEASON_DAYS)

# Calculate steady state means for the red lines
idx_start = 365 - GROWING_SEASON_DAYS
planting_litter_c = litter_c[idx_start::365]
planting_litter_n = litter_n[idx_start::365]
planting_microbe_c = microbe_c[idx_start::365]
planting_humus_c = humus_c[idx_start::365]

means = {
    "litter_c": planting_litter_c[50:].mean(),
    "litter_n": planting_litter_n[50:].mean(),
    "microbe_c": planting_microbe_c[50:].mean(),
    "humus_c": planting_humus_c[50:].mean()
}
















import numpy as np
import pandas as pd
from forward_models import (
    FIELD_CAPACITY, SOIL_DEPTH_MM, generate_rain,
    hydrology_model, soil_organic_model
)

# --- Standardized Year Calculation ---
# Day 0 is 0.0 years. Day 364 is the end of year 1.
# We calculate year as: (day_index / 365)
all_days = np.arange(TRANSIENT_PERIOD_DAYS)
all_years = all_days / 365.0

# Define the exact indices for the planting points
planting_indices = np.arange(365 - GROWING_SEASON_DAYS, TRANSIENT_PERIOD_DAYS, 365)

# --- Export Data ---

# Create base downsampled indices (every 10th day to keep LaTeX compiling fast)
base_indices = np.arange(0, TRANSIENT_PERIOD_DAYS, 10)

# CRITICAL FIX: Merge downsampled indices with exact planting indices
# np.unique sorts them and removes duplicates, ensuring lines connect perfectly to the dots
trend_indices = np.unique(np.concatenate((base_indices, planting_indices)))

# 1. Long term trends (Plots 1-5)
# Use trend_indices for ALL columns to avoid Pandas length mismatch errors
df_trends = pd.DataFrame({
    'year': all_years[trend_indices],
    'moisture': soil_moisture[trend_indices] * 100,
    'litterC': litter_c[trend_indices],
    'litterN': litter_n[trend_indices],
    'microbeC': microbe_c[trend_indices],
    'humusC': humus_c[trend_indices]
})
df_trends.to_csv('long_term_trends.txt', sep='\t', index=False)

# 2. Planting Points (The orange dots)
# We use the EXACT same years and indices as the plots
df_dots = pd.DataFrame({
    'year': all_years[planting_indices],
    'moisture': soil_moisture[planting_indices] * 100,
    'litterC': litter_c[planting_indices],
    'litterN': litter_n[planting_indices],
    'microbeC': microbe_c[planting_indices],
    'humusC': humus_c[planting_indices]
})
df_dots.to_csv('planting_points.txt', sep='\t', index=False)

# 3. Volatility (The 6th plot in your 2x3 grid)
seg_start = TRANSIENT_PERIOD_DAYS - GROWING_SEASON_DAYS
df_vol = pd.DataFrame({
    'day': np.arange(GROWING_SEASON_DAYS),
    'litterC': litter_c[seg_start:] / np.max(litter_c[seg_start:]),
    'litterN': litter_n[seg_start:] / np.max(litter_n[seg_start:]),
    'microbeC': microbe_c[seg_start:] / np.max(microbe_c[seg_start:]),
    'humusC': humus_c[seg_start:] / np.max(humus_c[seg_start:]),
    'moisture': soil_moisture[seg_start:] / np.max(soil_moisture[seg_start:])
})
df_vol.to_csv('growing_season_volatility.txt', sep='\t', index=False)

print("Export alignment fixed. Trend lines now perfectly intersect planting points.")