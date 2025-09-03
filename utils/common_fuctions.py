import os
import pandas as pd
from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
import yaml
import sys

logger = get_logger(__name__)


def read_yaml(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path: {file_path}")

        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the yaml file")
            return config

    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException("FAILED to read YAML file", sys)


def load_data(path):
    try:
        logger.info("Loading Data...")
        return pd.read_csv(path)
    except Exception as e:
        logger.error("Error while loading data")
        raise CustomException("FAILED to load data", sys)
