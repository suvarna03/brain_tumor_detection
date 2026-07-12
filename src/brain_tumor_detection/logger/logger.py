import logging
from datetime import datetime
from brain_tumor_detection.constants import LOG_DIR

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y')}.log"
logs_path = LOG_DIR
logs_path.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = logs_path / LOG_FILE

logging.basicConfig(filename= str(LOG_FILE_PATH),
                    format= "%(asctime)s | %(levelname)s | %(module)s | %(message)s",
                    filemode= "a",
                    level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully")
