from hog_data_tool.env_config import EnvConfig, get_env_config
from hog_data_tool.hog_data.hog_data_model import StructuredHogData


def main(config: EnvConfig) -> None:
    data = StructuredHogData.from_csv(config.input_data_path)
    data.plot_power_curves(output_dir=config.output_data_path)


if __name__ == "__main__":
    config = get_env_config()
    main(config)
