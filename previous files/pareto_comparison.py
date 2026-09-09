import pickle
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from main import MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS, PROBABILITY_OF_CLEAR_FORECAST
# Note: These constants are imported from a local module 'main' in the source
# from main import MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS, PROBABILITY_OF_CLEAR_FORECAST

# Setup Weights for Pareto calculations
weights_for_pareto = np.zeros([2, MOISTURE_NUM_STEPS, AMMONIUM_NUM_STEPS, NITRATE_NUM_STEPS], np.float64)
weights_for_pareto[:, 2, 0, 0] = 1
weights_for_pareto[0] *= PROBABILITY_OF_CLEAR_FORECAST
weights_for_pareto[1] *= (1 - PROBABILITY_OF_CLEAR_FORECAST)
weights_for_pareto /= np.sum(weights_for_pareto)

# Define Leaching Penalties
leaching_penalties_USD_per_m2 = np.linspace(0, 1.0734245700734262 / 2.0, 25)

@dataclass
class ParetoOutcomes:
    expected_profit_by_initial_state: npt.NDArray[np.float64]
    probability_of_violation_by_initial_state: npt.NDArray[np.float64]
    average_profits: npt.NDArray[np.float64]
    average_probability_of_violation: npt.NDArray[np.float64]
    average_pre_fine_profits: npt.NDArray[np.float64]

    def __init__(self, expected_profit_by_initial_state, probability_of_violation_by_initial_state):
        self.expected_profit_by_initial_state = expected_profit_by_initial_state
        self.probability_of_violation_by_initial_state = probability_of_violation_by_initial_state
        self.average_profits = np.average(expected_profit_by_initial_state, axis=(1, 2, 3, 4), weights=weights_for_pareto)
        self.average_probability_of_violation = np.average(probability_of_violation_by_initial_state, axis=(1, 2, 3, 4), weights=weights_for_pareto)
        self.average_pre_fine_profits = self.average_profits + (self.average_probability_of_violation * leaching_penalties_USD_per_m2)

# Load data from pickle files
pareto_outcomes_by_num_control_steps = {}

files = [
    (1, "36_pareto_sweep_expected_profit.pickle", "36_pareto_sweep_probability_of_violation.pickle"),
    (4, "9_pareto_sweep_expected_profit.pickle", "9_pareto_sweep_probability_of_violation.pickle"),
    (6, "6_pareto_sweep_expected_profit.pickle", "6_pareto_sweep_probability_of_violation.pickle"),
    (12, "3_pareto_sweep_expected_profit.pickle", "3_pareto_sweep_probability_of_violation.pickle")
]

for steps, profit_file, violation_file in files:
    with open(profit_file, "rb") as f_prof, open(violation_file, "rb") as f_viol:
        prof_data = pickle.load(f_prof)
        viol_data = pickle.load(f_viol)
        pareto_outcomes_by_num_control_steps[steps] = ParetoOutcomes(prof_data, viol_data)

# Visualization
CHOSEN_FINE_IDX = 22
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
line_styles = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

for idx, (key, value) in enumerate(pareto_outcomes_by_num_control_steps.items()):
    label = "single application" if key == 1 else "every {:d} days".format(int(36 / key))
    mask = value.average_pre_fine_profits > 0.18
    ax.plot(value.average_probability_of_violation[mask],
            value.average_pre_fine_profits[mask],
            label=label,
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2.5,
            color=colors[idx % len(colors)],
            alpha=0.85)

# Highlight chosen operating point
ax.scatter(pareto_outcomes_by_num_control_steps[12].average_probability_of_violation[CHOSEN_FINE_IDX],
           pareto_outcomes_by_num_control_steps[12].average_pre_fine_profits[CHOSEN_FINE_IDX],
           marker="*", s=400, color='#F4A259', edgecolors='#2D3142', linewidths=2, zorder=5, label='Chosen operating point')

ax.set_title("Profit vs Probability of Violation Pareto Fronts", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Profit (USD/$\text{m}^2$)")
ax.set_xlabel("Probability of N leaching limit violation")
ax.legend(loc="lower right", frameon=True, fontsize=12)
ax.grid(True, alpha=0.5, lw=1.0)
plt.tight_layout()
plt.show()

# Print Analysis
print("Maximum profit without leaching concerns")
for key, value in pareto_outcomes_by_num_control_steps.items():
    print(f"x{key}: \t{value.average_profits[0]}")