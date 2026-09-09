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
    MAX_PROFIT = 0.27

# ==========================================
# CONFIGURATION
# ==========================================
FILE_ORIGINAL = "3_days_rollouts.csv"
FILE_LINEAR = "linear_growth_data.csv"

TARGET_PERIOD = 3
TARGET_PENALTY = 0.20126710688876742
Z_SCORE = 1.96  # 95% Confidence Interval multiplier


def load_and_process(filename):
    """Loads dataset, filters for target controller, and calculates profit metrics & 95% CIs."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Check filename/path.")
        sys.exit(1)

    if "control_period" in df.columns:
        df = df[np.isclose(df["control_period"], TARGET_PERIOD)]
    if "penalty" in df.columns:
        df = df[np.isclose(df["penalty"], TARGET_PENALTY)]

    if df.empty:
        print(f"Warning: No matching rows found in '{filename}'.")
        sys.exit(1)

    # Calculate Profit Metrics
    df["profit_without_fines"] = MAX_PROFIT - df["deficit_cost"]
    df["profit_with_fines"] = df["profit_without_fines"] - (df["violated"] * TARGET_PENALTY)
    df["safety_probability"] = 1.0 - df["violated"].astype(float)

    metrics = ["profit_without_fines", "profit_with_fines", "safety_probability"]

    means = df[metrics].mean()
    sems = df[metrics].sem()
    cis = sems * Z_SCORE

    return means, cis


# Load and compute statistics for both models
orig_means, orig_cis = load_and_process(FILE_ORIGINAL)
lin_means, lin_cis = load_and_process(FILE_LINEAR)

# ==========================================
# PLOTTING
# ==========================================
model_labels = ["Original Model", "Linear Growth"]
colors = ["#1f77b4", "#ff7f0e"]

fig, axs = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle(
    f"Controller Comparison: Original vs. Linear Growth Model\n"
    f"(3-Day Control Period, Penalty: {TARGET_PENALTY:.4f} | 95% CI)",
    fontsize=13,
    y=0.98,
)

metrics_info = [
    ("profit_without_fines", "Profit w/o Penalty (USD/m$^2$)", axs[0]),
    ("profit_with_fines", "Profit w/ Penalty (USD/m$^2$)", axs[1]),
    ("safety_probability", "Safety Probability", axs[2]),
]

for metric_key, title, ax in metrics_info:
    means_val = [orig_means[metric_key], lin_means[metric_key]]
    cis_val = [orig_cis[metric_key], lin_cis[metric_key]]

    bars = ax.bar(
        model_labels,
        means_val,
        yerr=cis_val,
        capsize=6,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel(title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    if metric_key == "safety_probability":
        ax.set_ylim(0, 1.05)

    # Add numeric labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig("model_comparison_profit_bar_chart.png", dpi=300)
plt.show(block=True)