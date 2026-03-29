from hog_data_tool.env_config import EnvConfig, get_env_config
from hog_data_tool.hog_data.hog_data_model import StructuredHogData
from hog_data_tool.visualisations.visualisation import (
    plot_piecewise_power_curve,
    plot_rolling_average_weight_in_regimes,
    plot_session_frequency,
    plot_session_gap,
)


def load_structured_data(config: EnvConfig) -> StructuredHogData:
    """Load StructuredHogData from config:
    main CSV and, if set, other data from alt_data_folder (file or directory).
    Grippers are read from the 'gripper' column."""
    alt_path = config.alt_data_folder
    if alt_path is not None and str(alt_path).strip() and alt_path.exists():
        return StructuredHogData.from_csvs(
            config.input_data_path,
            other_path=alt_path,
        )
    return StructuredHogData.from_csv(config.input_data_path)


def main() -> None:
    config = get_env_config()
    data = load_structured_data(config)

    data.create_plot_for_all_grippers(
        plot_method=plot_piecewise_power_curve,
        output_path=config.output_data_path / "piecewise_power_curve",
        min_sessions=0,  # curve fits: always try to produce hyperbolic fit
    )

    data.create_shared_gripper_plot(
        plot_method=plot_session_gap, output_path=config.output_data_path / "session_gap"
    )

    data.create_shared_gripper_plot(
        plot_method=plot_session_frequency,
        output_path=config.output_data_path / "session_frequency",
    )

    data.create_plot_for_all_grippers(
        plot_method=plot_rolling_average_weight_in_regimes,
        output_path=config.output_data_path / "rolling_average_weight_in_regimes",
        min_sessions=30,  # progress charts: skip low-session grippers
    )


if __name__ == "__main__":
    main()
