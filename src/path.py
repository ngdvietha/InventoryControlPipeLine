from pathlib import Path

# Root directory của project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "final"

# Config
CONFIG_DIR = ROOT_DIR / "configs"

# Results
RESULTS_DIR = ROOT_DIR / "results"

# Reports
REPORTS_DIR = ROOT_DIR / "reports"


# Notebooks
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"