import yaml


def load_config(config_path: str = "config.yaml", logger: object = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded from %s", config_path)
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Invalid YAML syntax in %s: %s", config_path, e)
        raise
