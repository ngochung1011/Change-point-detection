import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load VN30 stock market data."""
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df.dropna()
    return df

def normalize_data(df):
    """Apply Z-score normalization."""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    return df_scaled, scaler

if __name__ == "__main__":
    data = load_data("vn30_data.csv")
    normalized_data, scaler = normalize_data(data)
    normalized_data.to_csv("processed_vn30_data.csv")
    print("Data preprocessing completed.")
