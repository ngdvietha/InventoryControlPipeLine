# src/utils/io.py

import json
import joblib
import pandas as pd
from pathlib import Path


# =========================
# Model Saving / Loading
# =========================

def save_model(model, path: Path):
    """
    Save trained model to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    """
    Load trained model from disk.
    """
    return joblib.load(path)


# =========================
# JSON Saving / Loading
# =========================

def save_json(data: dict, path: Path):
    """
    Save dictionary to JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path):
    """
    Load dictionary from JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


# =========================
# CSV Saving
# =========================

def save_csv(df: pd.DataFrame, path: Path):
    """
    Save DataFrame to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =========================
# Parquet Saving / Loading
# =========================

def save_parquet(df: pd.DataFrame, path: Path):
    """
    Save DataFrame to Parquet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path):
    """
    Load DataFrame from Parquet.
    """
    return pd.read_parquet(path)

