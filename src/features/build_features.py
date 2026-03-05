# src/features/build_features.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def calculate_sbc_features(
    group: pd.DataFrame,
    demand_col: str = 'demand',
    date_col: str = None,
    adi_method: str = 'ratio'   # 'ratio' hoặc 'gap'
) -> pd.Series:
    """
    Calculate SBC features (ADI and CV2) for demand classification.

    Parameters
    ----------
    group : pd.DataFrame
        Sales data of a single item
    demand_col : str
        Column name of demand
    date_col : str
        Column name of date (required if adi_method='gap')
    adi_method : str
        'ratio' → ADI = T / nz
        'gap'   → ADI = mean inter-demand interval (calendar-based)

    Returns
    -------
    pd.Series
        ADI and CV2
    """

    demand = group[demand_col].to_numpy()
    nz_mask = demand > 0
    nz = nz_mask.sum()

    # =============================
    # ======== ADI PART ===========
    # =============================

    if nz == 0:
        adi = np.inf

    else:
        if adi_method == 'ratio':
            # Code #2 logic
            T = len(demand)
            adi = T / nz

        elif adi_method == 'gap':
            # Code #1 logic (calendar-based)
            if date_col is None:
                raise ValueError("date_col must be provided when adi_method='gap'")

            g = group.sort_values(date_col)

            nz_dates = g.loc[g[demand_col] > 0, date_col]

            if len(nz_dates) <= 1:
                adi = 1.0
            else:
                gaps = nz_dates.diff().dt.days.dropna()
                adi = gaps.mean()

        else:
            raise ValueError("adi_method must be 'ratio' or 'gap'")

    # =============================
    # ======== CV2 PART ===========
    # =============================

    if nz == 0:
        cv2 = 0.0
    else:
        sizes = demand[nz_mask]
        mean_size = sizes.mean()

        if mean_size == 0 or len(sizes) == 1:
            cv2 = 0.0
        else:
            cv = sizes.std(ddof=0) / mean_size
            cv2 = cv ** 2

    return pd.Series({'ADI': adi, 'CV2': cv2})


def classify_demand_type(row):
    """
    Classify demand type based on ADI and CV2 values.
    
    Categories:
    -----------
    - smooth: ADI <= 1.32 and CV2 <= 0.49
    - erratic: ADI <= 1.32 and CV2 > 0.49
    - intermittent: ADI > 1.32 and CV2 <= 0.49
    - lumpy: ADI > 1.32 and CV2 > 0.49
    """
    adi = row['ADI']
    cv2 = row['CV2']
    if adi <= 1.32 and cv2 <= 0.49:
        return 'smooth'
    elif adi <= 1.32 and cv2 > 0.49:
        return 'erratic'
    elif adi > 1.32 and cv2 <= 0.49:
        return 'intermittent'
    else:
        return 'lumpy'


def create_sales_by_item_with_classification(
        sales_with_prices: pd.DataFrame,
        product_colID:str,
        demand_col:str,
        totalsales_col:str,
        adi_method: str = 'ratio',
        date_col: str = None) -> pd.DataFrame:
    """
    Aggregate sales by item and classify demand type using SBC method.
    
    Parameters:
    -----------
    sales_with_prices : pd.DataFrame
        Sales data with price and demand information
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: item_id, sales_amount, ADI, CV2, demand_type
        sorted by sales_amount in descending order
    """
    # Calculate total sales by item
    sales_by_item = (
        sales_with_prices
        .groupby(product_colID, as_index=False)[totalsales_col]
        .sum()
        .sort_values(totalsales_col, ascending=False)
    )
    
    # Calculate SBC features for each item
    item_stats = (
        sales_with_prices
        .groupby(product_colID, as_index=False)
        .apply(lambda df: calculate_sbc_features(df, demand_col=demand_col, adi_method=adi_method, date_col=date_col))
        .reset_index(drop=True)
    )
    
    # Classify demand type
    item_stats['demand_type'] = item_stats.apply(classify_demand_type, axis=1)
    
    # Merge classification into sales_by_item
    sales_by_item = sales_by_item.merge(
        item_stats[[product_colID, 'ADI', 'CV2', 'demand_type']],
        on=product_colID,
        how='left'
    )
    
    return sales_by_item


def classify_abc_by_kmeans(
        df: pd.DataFrame,
        sales_col: str = 'sales_amount',
        n_clusters: int = 3,
        random_state: int = 42) -> pd.DataFrame:
    """
    Classify items into ABC classes using K-means clustering on log-transformed sales.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing sales data
        
    sales_col : str
        Name of the column containing sales amounts (default: 'sales_amount')
        
    n_clusters : int
        Number of clusters for K-means (default: 3)
        
    random_state : int
        Random state for reproducibility (default: 42)
    
    Returns:
    --------
    pd.DataFrame
        Input dataframe with added columns:
        - 'cluster_{n_clusters}': numeric cluster labels
        - 'ABC_class': ABC classification (A=highest, C=lowest)
    """


    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Log transform sales to reduce skew
    X_for_clust = np.log1p(result[sales_col].values).reshape(-1, 1)
    
    # Fit K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_for_clust)
    
    # Get cluster centers in original scale
    centers_log = kmeans.cluster_centers_.ravel()
    centers_orig = np.expm1(centers_log)
    
    # Sort clusters by center value (smallest to largest)
    order = np.argsort(centers_orig)
    
    # Map clusters to ABC classes (A=largest, B=middle, C=smallest)
    # For k=3: C=smallest, B=middle, A=largest
    if n_clusters == 3:
        cluster_to_class = {
            order[0]: 'C',
            order[1]: 'B',
            order[2]: 'A'
        }
    else:
        # For other k values, use A, B, C, ... mapping
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:n_clusters]
        # Reverse to assign A to highest cluster
        cluster_to_class = {order[i]: class_labels[n_clusters - 1 - i] for i in range(n_clusters)}
    
    # Add cluster and ABC class columns
    result[f'cluster_{n_clusters}'] = cluster_labels
    result['ABC_class'] = [cluster_to_class[c] for c in cluster_labels]
    
    return result


def classify_product_lifecycle(
        daily_sales: pd.DataFrame,
        item_col: str = 'item_id',
        date_col: str = 'd',
        sales_col: str = 'sales_amount') -> pd.DataFrame:
    """
    Classify products into lifecycle stages based on year-over-year sales growth.
    
    Takes daily sales data, aggregates to annual sales, and classifies products
    into lifecycle stages based on year-over-year growth rates.
    
    Stages:
    -------
    - Introduction: First year with sales > 0
    - Phase out: Previous year had zero sales or growth rate undefined
    - Decline: pct_change <= Q20
    - Decaying maturity: Q20 < pct_change <= -5%
    - Stable maturity: -5% < pct_change <= 5%
    - Sustained maturity: 5% < pct_change <= Q80
    - Growth: pct_change > Q80
    
    Parameters:
    -----------
    daily_sales : pd.DataFrame
        DataFrame with daily sales data containing columns for item_id, date, and sales_amount
        
    item_col : str
        Name of the item/product ID column (default: 'item_id')
        
    date_col : str
        Name of the date column (default: 'd')
        
    sales_col : str
        Name of the sales amount column (default: 'sales_amount')
    
    Returns:
    --------
    pd.DataFrame
        Annual sales dataframe with added columns:
        - 'pct_change': Year-over-year percentage change in sales
        - 'last_annual_sales': Previous year's sales
        - 'first_sales_year': First year with sales > 0
        - 'life_cycle_stage': Lifecycle stage classification
    """

    # Create a copy to avoid modifying the original
    data = daily_sales.copy()
    
    # Extract year from date column
    data['year'] = pd.to_datetime(data[date_col]).dt.year
    
    # Aggregate daily sales to annual sales by item and year
    result = (data
              .groupby([item_col, 'year'], as_index=False)[sales_col]
              .sum()
              .rename(columns={sales_col: 'annual_sales'}))
    
    # Ensure proper sorting
    result = result.sort_values([item_col, 'year'])
    
    # Use annual_sales column for calculations
    sales_col_agg = 'annual_sales'
    year_col = 'year'
    
    # Calculate year-over-year percentage change
    result['pct_change'] = (
        result
        .groupby(item_col)[sales_col_agg]
        .pct_change() * 100
    )
    
    # Get previous year's sales
    result['last_annual_sales'] = (
        result
        .groupby(item_col)[sales_col_agg]
        .shift(1)
    )
    
    # Find first year with sales > 0 for each item
    first_year_map = (
        result.loc[result[sales_col_agg] > 0]
        .groupby(item_col)[year_col]
        .min()
    )
    result['first_sales_year'] = result[item_col].map(first_year_map)
    
    # Calculate quantile thresholds
    pct = (
        result['pct_change']
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    
    quints = pct.quantile([0.2, 0.4, 0.6, 0.8])
    Q20 = quints.loc[0.2]
    Q40 = quints.loc[0.4]
    Q60 = quints.loc[0.6]
    Q80 = quints.loc[0.8]
    
    # Define classification function
    def _classify_stage(row):
        sales = row[sales_col_agg]
        pct = row['pct_change']
        year = row[year_col]
        first_year = row['first_sales_year']
        last_sales = row['last_annual_sales']
        
        # Introduction: first year with sales
        if year == first_year:
            return 'Introduction'
        
        # Phase out: previous year had no sales or growth rate undefined
        if last_sales == 0 or pd.isna(pct):
            return 'Phase out'
        
        # Classify by growth quantiles
        if pct <= Q20:
            return 'Decline'
        elif Q20 < pct <= -5:
            return 'Decaying maturity'
        elif -5 < pct <= 5:
            return 'Stable maturity'
        elif 5 < pct <= Q80:
            return 'Sustained maturity'
        elif pct > Q80:
            return 'Growth'
        else:
            return 'Unknown'
    
    # Apply classification
    result['life_cycle_stage'] = result.apply(_classify_stage, axis=1)
    
    return result


def get_current_lifecycle_stage(
        annual_sales: pd.DataFrame,
        year_col: str = 'year',
        item_col: str = 'item_id',
        as_of_year: int = None) -> pd.DataFrame:
    """
    Extract the most recent lifecycle stage for each product as of a given year.
    
    Parameters:
    -----------
    annual_sales : pd.DataFrame
        Annual sales dataframe with lifecycle classifications (output from classify_product_lifecycle)
        
    year_col : str
        Name of the year column (default: 'year')
        
    item_col : str
        Name of the item/product ID column (default: 'item_id')
        
    as_of_year : int
        Get the most recent stage on or before this year. 
        If None, uses the maximum year in the data (default: None)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per product, showing their most recent lifecycle stage
        as of the specified year
    """
    # Create a copy to avoid modifying the original
    data = annual_sales.copy()
    
    # If as_of_year not specified, use the max year in data
    if as_of_year is None:
        as_of_year = data[year_col].max()
    
    # Filter to data on or before the specified year
    data = data[data[year_col] <= as_of_year]
    
    # Sort by item and year to ensure proper ordering
    data = data.sort_values([item_col, year_col])
    
    # Get the last (most recent) row for each item
    result = data.groupby(item_col).tail(1).reset_index(drop=True)
    
    return result
