
# src/data/make_dataset.py

import pandas as pd
from pathlib import Path
from src.path import RAW_DATA_DIR, PROCESSED_DATA_DIR


def melt_sales_to_long(
    sales_wide: pd.DataFrame,
    id_cols: list[str],
    var_name: str = "d",
    value_name: str = "demand",
) -> pd.DataFrame:
    """
    Convert wide sales data to long format.
    """

    missing = [c for c in id_cols if c not in sales_wide.columns]
    if missing:
        raise ValueError(f"Missing ID columns in sales_wide: {missing}")

    sales_long = sales_wide.melt(
        id_vars=id_cols,
        var_name=var_name,
        value_name=value_name,
    )

    return sales_long


def add_dim_columns(
    sales_long: pd.DataFrame,
    dim: pd.DataFrame,
    join_key: str | list[str],
    dim_cols: list[str],
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge dim columns into sales_long.
    
    Parameters:
    -----------
    sales_long : pd.DataFrame
        Sales data in long format
        
    dim : pd.DataFrame
        Dimension table to merge
        
    join_key : str or list[str]
        Column name(s) to join on. Can be a single string or list of strings for multi-key joins.
        
    dim_cols : list[str]
        Dimension columns to add from dim table
        
    how : str
        Type of merge (default: 'left')
    
    Returns:
    --------
    pd.DataFrame
        Sales data merged with dimension columns
    """
    
    # Convert join_key to list if it's a string
    if isinstance(join_key, str):
        join_key_list = [join_key]
    else:
        join_key_list = list(join_key)

    needed = set(join_key_list + dim_cols)

    if not needed.issubset(dim.columns):
        raise ValueError(f"dim must contain columns {needed}")

    cal = dim[join_key_list + dim_cols].drop_duplicates(join_key_list)

    merged = sales_long.merge(
        cal,
        on=join_key_list,
        how=how,
        validate="m:1",
    )

    return merged

