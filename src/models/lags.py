import pandas as pd


def create_df_with_lags(df: pd.DataFrame, num_lags=3):
    for lag in range(1, num_lags + 1):
        df[f"energy_price_lag_{lag}"] = df["energy_price"].shift(lag)
    df.dropna(inplace=True)
    return df
