# src/utils/logger.py


import logging
from pathlib import Path
from datetime import datetime

# from src.paths import RUNS_DIR


def get_logger(name: str, run_name: str = None):
    """

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__)
    run_name : str
        Optional run identifier (e.g., exp01)

    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (print to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (save log to file) lưu file sẽ để sau

    # if run_name is None:
    #     run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # log_dir = RUNS_DIR / run_name
    # log_dir.mkdir(parents=True, exist_ok=True)

    # file_handler = logging.FileHandler(log_dir / "training.log")
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger