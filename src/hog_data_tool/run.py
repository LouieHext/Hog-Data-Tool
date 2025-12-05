
from hog_data_tool.env_config import EnvConfig, get_env_config
from hog_data_tool.hog_data.reader import HogData

def main(config: EnvConfig) -> None:
    reader = HogData.from_csv(config.input_data_path)

    

  




if __name__ == "__main__":
    config = get_env_config()
    main()
