import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# FORCE WINDOWS BACKEND
matplotlib.use('TkAgg')

# IMPORT CONSTANTS (from main.py as in original script)
try:
    from main import N_SUFFICIENT_YIELD_G_GRAIN_PER_M2, CORN_PRICE_USD_PER_G_GRAIN
    MAX_PROFIT = N_SUFFICIENT_YIELD_G_GRAIN_PER_M2 * CORN_PRICE_USD_PER_G_GRAIN
except ImportError:
    # Fallback default estimate if main.py is not in the execution path
    MAX_PROFIT = 0.27

# ==========================================
# CONFIGURATION & COLUMN NAMES
# ==========================================
FILENAME = "rainfall_sweep_data.csv"

COL_CONTROL_PERIOD = "control_period"
COL_PENALTY = "penalty"
COL_RAINFALL = "rainfall_rate"
COL_DEFICIT_COST = "deficit_cost"
COL_VIOLATED = "violated"

TARGET_PERIOD = 3
TARGET_PENALTY = 0.20126710688876742
Z_SCORE = 1.96  # 95% Confidence Interval multiplier

# ==========================================
# LOAD AND FILTER DATA
# ==========================================
try:
    df = pd.read_csv(FILENAME)
except FileNotFoundError:
    print(f"Error: {FILENAME} not found. Please check the file path.")
    sys.exit(1)

df = df[df[COL_CONTROL_PERIOD] == TARGET_PERIOD]
df = df[np.isclose(df[COL_PENALTY], TARGET_PENALTY)]

if df.empty:
    print("Warning: No data found for the specified control period and penalty.")
    sys.exit(1)

# ==========================================
# CALCULATE PROFIT METRICS
# ==========================================
# Profit without fines = Max Profit - Deficit Cost
df["profit_without_fines"] = MAX_PROFIT - df[COL_DEFICIT_COST]

# Profit with fines = Profit without fines - Penalty (if violated)
df["profit_with_fines"] = df["profit_without_fines"] - (df[COL_VIOLATED] * df[COL_PENALTY])

# Safety Probability = 1 - Violation indicator
df["safety_probability"] = 1.0 - df[COL_VIOLATED].astype(float)

# ==========================================
# GROUP BY RAINFALL RATE
# ==========================================
grouped = df.groupby(COL_RAINFALL)
means = grouped.mean()
sems = grouped.sem()
margin_of_error = sems * Z_SCORE
rainfall_rates = means.index

# ==========================================
# PLOTTING
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle(
    f"Controller Robustness vs. Rainfall Rate\n"
    f"(3-Day Control Period, Penalty: {TARGET_PENALTY:.4f} | 95% CI)",
    fontsize=14,
    y=0.98,
)

# --- Subplot 1: Profit Without Fines ---
ax1.plot(rainfall_rates, means["profit_without_fines"], label="Profit (No Fines)", color='tab:blue', linewidth=2)
ax1.fill_between(
    rainfall_rates,
    means["profit_without_fines"] - margin_of_error["profit_without_fines"],
    means["profit_without_fines"] + margin_of_error["profit_without_fines"],
    color='tab:blue',
    alpha=0.2,
)
ax1.set_ylabel("Profit (USD/m$^2$)")
ax1.legend(loc="lower left")
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Subplot 2: Profit With Fines ---
ax2.plot(rainfall_rates, means["profit_with_fines"], label="Profit (With Fines)", color='tab:red', linewidth=2, linestyle='--')
ax2.fill_between(
    rainfall_rates,
    means["profit_with_fines"] - margin_of_error["profit_with_fines"],
    means["profit_with_fines"] + margin_of_error["profit_with_fines"],
    color='tab:red',
    alpha=0.2,
)
ax2.set_ylabel("Profit w/ Fines (USD/m$^2$)")
ax2.legend(loc="lower left")
ax2.grid(True, linestyle='--', alpha=0.6)

# --- Subplot 3: Safety Probability ---
ax3.plot(rainfall_rates, means["safety_probability"], label="Safety Probability", color='tab:green', linewidth=2)
ax3.fill_between(
    rainfall_rates,
    np.clip(means["safety_probability"] - margin_of_error["safety_probability"], 0, 1),
    np.clip(means["safety_probability"] + margin_of_error["safety_probability"], 0, 1),
    color='tab:green',
    alpha=0.2,
)
ax3.set_xlabel("Rainfall Rate")
ax3.set_ylabel("Probability of Safety")
ax3.set_ylim(-0.05, 1.05)
ax3.legend(loc="lower left")
ax3.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("rainfall_robustness_sweep_profit.png", dpi=300)
plt.show(block=True)