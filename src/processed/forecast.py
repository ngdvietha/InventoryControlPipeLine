import pandas as pd
import numpy as np

def moving_average_forecast(series, window, forecast_horizon=1):
    """
    Moving Average Forecast
    
    Parameters:
    - series: array-like (list, numpy array, pandas Series)
    - window: số kỳ dùng để tính trung bình (ví dụ: 3, 5, 7)
    - forecast_horizon: số bước forecast tương lai
    
    Returns:
    - forecasts: list giá trị forecast cho các bước tương lai
    """
    
    series = list(series)
    forecasts = []
    
    temp_series = series.copy()
    
    for _ in range(forecast_horizon):
        if len(temp_series) < window:
            raise ValueError("Not enough data points for the specified window")
        
        ma = np.mean(temp_series[-window:])
        forecasts.append(ma)
        
        # append forecast vào series để forecast step tiếp theo (recursive)
        temp_series.append(ma)
    
    return forecasts


def simple_exponential_smoothing(series, alpha, forecast_horizon=1):
    """
    Simple Exponential Smoothing (SES)
    
    Parameters:
    - series: list / array (data lịch sử)
    - alpha: smoothing factor (0 < alpha < 1)
    - forecast_horizon: số bước dự báo tương lai
    
    Returns:
    - fitted_values: list giá trị fitted (in-sample)
    - forecasts: list giá trị forecast (out-of-sample)
    """
    
    series = list(series)
    
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")
    
    # Khởi tạo: forecast đầu tiên = giá trị đầu tiên
    fitted = [series[0]]
    
    # Tính fitted values
    for t in range(1, len(series)):
        ft = alpha * series[t-1] + (1 - alpha) * fitted[-1]
        fitted.append(ft)
    
    # Forecast tương lai (SES sẽ flat)
    last_forecast = alpha * series[-1] + (1 - alpha) * fitted[-1]
    forecasts = [last_forecast] * forecast_horizon
    
    return fitted, forecasts