# Hog Data Tool

A Python project for analyzing and visualizing HOG grip training session data.  
Includes session statistics, power curve fitting, and regime-based rolling averages.

---

## Setup

1. **Environment**  
   Copy `env.shared` to `.env` and update paths and settings:  

   ```text
   INPUT_DATA_PATH=./data/hog_data.csv
   OUTPUT_DATA_PATH=./data/outputs/
   WEIGHT_UNIT=lbs

2. **Install dependencies**
    simply call `poetry install` and poetry will install the dependencies to your venv.

---

## Running the project

Run the main data pipeline `make run`

This will

1. Load the CSV from `INPUT_DATA_PATH`.

2. Create plots for all grippers:
    - Power curves (plot_power_curve)
    - Inverted power curves (plot_inverted_power_curve)
    - Session gaps (plot_session_gap)
    - Session frequency (plot_session_frequency)
    - Rolling average weight in regimes (plot_rolling_average_weight_in_regimes)

3.  Save outputs to `OUTPUT_DATA_PATH`.

---

## Loading data

Data can be loaded directly in Python:

```
from hog_data_tool import StructuredHogData, EnvConfig, get_env_config

config = get_env_config()
data = StructuredHogData.from_csv(config.input_data_path)
```

---

## Plotting

You can generate plots programmatically:

```
# Plot per gripper
data.create_plot_for_all_grippers(plot_method=plot_power_curve, output_path=config.output_data_path / "power_curve")

# Plot shared data (all right hands by default)
data.create_shared_gripper_plot(plot_method=plot_session_gap, output_path=config.output_data_path / "session_gap")

```

You can create your own plotting functions, note all plotting functions must follow the SessionPlotMethod or SharedSessionPlotMethod signature.
"""

---

## Curve fitting

- Basic hyperbolic curve fitting is included (`fit_power_curve_with_hyperbolic_decay`).
- Piecewise curve fitting will be added in future updates.