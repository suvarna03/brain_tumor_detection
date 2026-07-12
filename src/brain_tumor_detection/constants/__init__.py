from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent

CONFIG_FILE_PATH = ROOT_PATH / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_PATH / "config" / "params.yaml"
SCHEMA_FILE_PATH = ROOT_PATH / "config" / "schema.yaml"

LOG_DIR = ROOT_PATH / "logs"
