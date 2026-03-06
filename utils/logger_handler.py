import os
import logging
from utils.path_tool import get_abs_path
from datetime import datetime



# the root path of the project for logs
LOG_ROOT_PATH = get_abs_path("logs")
os.makedirs(LOG_ROOT_PATH, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
        name: str = "agent", 
        log_file=None, 
        level=logging.DEBUG, 
        console_level=logging.INFO
        ) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger  # Avoid adding multiple handlers to the same logger

    # File handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT_PATH, f"{name}-{datetime.now().strftime('%Y-%m-%d')}.log")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(DEFAULT_LOG_FORMAT)
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    return logger


# get a default logger for the project
logger = get_logger()

if __name__ == "__main__":
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")       
