from hog_data_tool.env_config import EnvConfig, get_env_config
from hog_data_tool.hog_data.hog_data_model import StructuredHogData
from hog_data_tool.visualisations.visualisation import (
    plot_inverted_power_curve,
    plot_power_curve,
    plot_session_frequency,
    plot_session_gap,
)


def main(config: EnvConfig) -> None:
    data = StructuredHogData.from_csv(config.input_data_path)
    data.create_plot_for_all_grippers(
        plot_method=plot_power_curve, output_path=config.output_data_path / "power_curve"
    )

    data.create_plot_for_all_grippers(
        plot_method=plot_inverted_power_curve,
        output_path=config.output_data_path / "inverted_power_curvexs",
    )

    data.create_shared_gripper_plot(
        plot_method=plot_session_gap, output_path=config.output_data_path / "session_gap"
    )

    data.create_shared_gripper_plot(
        plot_method=plot_session_frequency,
        output_path=config.output_data_path / "session_frequency",
    )


if __name__ == "__main__":
    config = get_env_config()
    main(config)
