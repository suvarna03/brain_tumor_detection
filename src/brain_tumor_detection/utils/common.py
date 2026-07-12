from brain_tumor_detection.logger import logger
from brain_tumor_detection.exception import CustomException
from pathlib import Path
import sys
import yaml
import json
import joblib
from box import ConfigBox
def create_directory(path: Path) -> None:
    try:
        path.mkdir(
        parents=True,
        exist_ok=True
        )
        logger.info(f"Created directory at {path}")
    except Exception as e:
        raise CustomException(e,sys)

def read_yaml(path_to_yaml: Path):

    try:
        with open(path_to_yaml, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(
                f"YAML file loaded successfully from: {path_to_yaml}"
                )
        return ConfigBox(config)

    except Exception as e:
        raise CustomException(e, sys)

def save_json(path: Path, data: dict):
    try:
        create_directory(path.parent)
        with open(path,'w') as json_file:
            json.dump(data,
                      json_file,
                       indent=4)
        logger.info(
            f"Saved json file at {path}"
            )
    except Exception as e:
        raise CustomException(e,sys)
    
def load_json(path: Path) -> dict:
    try:
        with open(path,'r') as json_file:
            data = json.load(
                      json_file )
            logger.info(
                f"Json file loaded from {path}"
                )
        return data
    
    except Exception as e:
        raise CustomException(e,sys)
    
def save_bin(path: Path, data):
    try:
        create_directory(path.parent)

        joblib.dump(
            value=data,
            filename=path
        )

        logger.info(
            f"Saved binary file at {path}"
        )

    except Exception as e:
        raise CustomException(e, sys)
    
def load_bin(path: Path):

    try:

        data = joblib.load(path)

        logger.info(
            f"Binary file loaded from {path}"
        )

        return data

    except Exception as e:
        raise CustomException(e, sys)