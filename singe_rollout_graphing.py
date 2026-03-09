import pickle
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import main
from forward_models import SOIL_DEPTH_MM, INORGANIC_N_MODEL_DT_DAYS
from dataclasses import dataclass, field
import inspect
from typing import Dict, List
import numpy as np

CONTROL_TIMESTEP_DAYS = 3
NUM_CONTROL_STEPS = int(36 / CONTROL_TIMESTEP_DAYS)
@dataclass
class RolloutGraphData:
    n_accumulation_trajectories_by_penalty: Dict[float, npt.NDArray[np.float64]]
    fertilizer_adds_by_penalty: Dict[float, npt.NDArray[np.float64]]
    soil_condition_risks: List[float]
with open(str(CONTROL_TIMESTEP_DAYS) + "_days_rainfall_317_graph.pkl", "rb") as f:
    rollout_graph_data = pickle.load(f)

# Modern color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
line_styles = ['--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

penalties = list(rollout_graph_data.n_accumulation_trajectories_by_penalty.keys())
fig, (ax, lower_ax) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], figsize=[10, 6.4], layout='constrained', dpi=150)

ax.set_title("Plant nitrogen stocks over the growing season", fontsize=14, fontweight='bold', pad=15)
x_series = np.arange(len(rollout_graph_data.n_accumulation_trajectories_by_penalty[penalties[0]])) * INORGANIC_N_MODEL_DT_DAYS

# Draw colored blocks FIRST (behind everything else)
cmap = mpl.colormaps['binary']
n_crit = main.biomass_g_per_m2_to_critical_nitrogen_gN_per_m2(main.day_to_biomass_g_per_m2(x_series)) / (main.SOIL_DEPTH_MM / 1000.)
y_max = np.max(n_crit)
for trajectory in rollout_graph_data.n_accumulation_trajectories_by_penalty.values():
    y_max = max(y_max, np.max(trajectory))
y_max *= 1.1
x_max = NUM_CONTROL_STEPS * CONTROL_TIMESTEP_DAYS

for i in range(NUM_CONTROL_STEPS):
    day = i * CONTROL_TIMESTEP_DAYS
    ax.fill([day, day, day + CONTROL_TIMESTEP_DAYS, day + CONTROL_TIMESTEP_DAYS], [0, y_max, y_max, 0], color=cmap(rollout_graph_data.soil_condition_risks[i]), zorder=0, linewidth=0)

# Now draw lines on top with white outlines for visibility
ax.plot(x_series, n_crit, label="Critical nitrogen level", color=colors[0], linewidth=4, alpha=1, zorder=2, solid_capstyle='round', path_effects=[path_effects.Stroke(linewidth=6.5, foreground='white', alpha=0.8), path_effects.Normal()])
trajectories_by_penalty = {}
for idx, (leaching_penalty_USD, trajectory) in enumerate(rollout_graph_data.n_accumulation_trajectories_by_penalty.items()):
    ax.plot(x_series,
            trajectory,
            label="${:04.2f} penalty".format(leaching_penalty_USD),
            color=colors[1 + idx],
            # linestyle=line_styles[idx],
            marker=markers[idx],
            markersize=6,
            markevery=0.1 + 0.03 * idx,
            linewidth=4,
            alpha=1,
            zorder=2,
            solid_capstyle='round',
            path_effects=[path_effects.Stroke(linewidth=6.5, foreground='white', alpha=0.8), path_effects.Normal()])

ax.vlines(np.linspace(0, x_max, NUM_CONTROL_STEPS, endpoint=False), 0, y_max, color='#E07A5F', alpha=0.3, linestyles="dashed", linewidth=1.5, zorder=1)

ax.set_ylabel("Total plant N accumulation (gN/$\\text{m}^3$)", fontsize=12)
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)

cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)
cbar.set_label('Probability of violation', fontsize=10)

ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=10)
ax.grid(True, alpha=0.4, zorder=1)

import numpy as np

# Define bar width parameters
num_series = len(rollout_graph_data.fertilizer_adds_by_penalty)
total_group_width = 0.8  # The total width used by all bars at a single time point
bar_width = total_group_width / num_series

# Lower subplot with modern bar chart styling
for idx, fertilizer_adds in enumerate(rollout_graph_data.fertilizer_adds_by_penalty.values()):
    # Calculate original x positions (same as before)
    x_base = np.linspace(0, x_max, len(fertilizer_adds) + 1)[:-1] + 0.3

    # Calculate offsets to place bars side-by-side within each time step
    # This centers the group of bars over the original x coordinate
    offset = (idx - num_series / 2) * bar_width + bar_width / 2
    x_pos = x_base + offset

    lower_ax.bar(x_pos, fertilizer_adds,
                 width=bar_width,
                 color=colors[idx + 1],
                 alpha=0.7,
                 edgecolor='white',  # Adds a clean separation between bars
                 linewidth=0.5)

lower_ax.set_title("Added fertilizer", fontsize=12, fontweight='bold', pad=10)
lower_ax.set_xlabel("Day", fontsize=12)
lower_ax.set_ylabel(r"Added N (gN/$\mathrm{m}^3$)",
                    fontsize=12)  # Note: Changed \text to \mathrm for better compatibility

# Add vertical guides for control steps
for x in np.linspace(0, x_max, NUM_CONTROL_STEPS, endpoint=False):
    lower_ax.axvline(x, color='#E07A5F', alpha=0.3, linestyle="dashed", linewidth=1.5)

lower_ax.grid(True, alpha=0.4, axis='y')  # Grid on Y-axis is usually cleaner for bars

# Handle log scale for bar heights
# Note: In log scales, bars start from the base of the axis (y_min)
lower_ax.set_yscale('log')
y_min = 1e-1  # Set slightly above 0 for log scale visibility
y_max_val = max([max(adds) for adds in
                 rollout_graph_data.fertilizer_adds_by_penalty.values()]) if rollout_graph_data.fertilizer_adds_by_penalty.values() else 1
lower_ax.set_ylim(y_min, y_max_val * 1.5)

plt.show()
fig.savefig("mixed_violating_trajectory.png", transparent=True, dpi=300, bbox_inches='tight')