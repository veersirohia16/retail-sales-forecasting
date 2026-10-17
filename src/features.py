"""
features.py
-----------
Feature engineering for the machine-learning models.

Rule followed everywhere: a feature for week t may only use information that
would really be known when the forecast is made (the "forecast origin").
Because our test horizon is at most 39 weeks, any lag of 51 weeks or more
always refers to weeks that are already in the past -> no data leakage.
Store-level statistics are computed from the training period only.
"""

import numpy as np
import pandas as pd

from config import SEASON_LENGTH

# Columns given to the models
FEATURE_COLS = [
    "Store", "Type_code", "Size",
    "WeekOfYear", "Month", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown_Total", "MarkDown_Recorded",
    "lag_52", "lag_52_smooth", "store_mean", "store_growth", "lag_52_x_growth",
]
TARGET = "Weekly_Sales"


def add_lag_features(df):
    """
    Seasonal lag features, computed per store:
      lag_52         sales in the same week one year earlier
      lag_52_smooth  average of weeks t-51, t-52, t-53 (smooths out holidays
                     that shift by a week from one year to the next)
    """
    df = df.sort_values(["Store", "Date"]).copy()
    g = df.groupby("Store")["Weekly_Sales"]
    df["lag_52"] = g.shift(SEASON_LENGTH)
    # mean of whichever of the three weeks exist (the very first lagged week
    # has no t-53 value, so it averages t-51 and t-52 only)
    df["lag_52_smooth"] = pd.concat(
        [g.shift(SEASON_LENGTH - 1), g.shift(SEASON_LENGTH), g.shift(SEASON_LENGTH + 1)], axis=1
    ).mean(axis=1, skipna=True)
    df.loc[df["lag_52"].isna(), "lag_52_smooth"] = np.nan
    return df


def add_store_stats(df, history_end):
    """
    Per-store level and growth, measured only on data up to `history_end`:
      store_mean    average weekly sales over the whole history
      store_growth  last 52 weeks' sales / the 52 weeks before that
    lag_52_x_growth = last year's sales scaled by the store's recent growth.
    """
    hist = df[df["Date"] <= history_end].sort_values(["Store", "Date"])

    def yoy_growth(s):
        # compare the most recent k weeks with the SAME k weeks one year
        # earlier (k = 52 when two full years are available)
        v = s.to_numpy()
        k = min(SEASON_LENGTH, len(v) - SEASON_LENGTH)
        return v[-k:].sum() / v[-k - SEASON_LENGTH:-SEASON_LENGTH].sum()

    stats = hist.groupby("Store")["Weekly_Sales"].agg(
        store_mean="mean", store_growth=yoy_growth).reset_index()
    out = df.drop(columns=["store_mean", "store_growth"], errors="ignore").merge(stats, on="Store", how="left")
    out["lag_52_x_growth"] = out["lag_52"] * out["store_growth"]
    return out


def encode(df):
    """Store type A/B/C -> 0/1/2 (tree models need numbers)."""
    df = df.copy()
    df["Type_code"] = df["Type"].map({"A": 0, "B": 1, "C": 2})
    return df


def build_matrix(df, history_end):
    """Full feature table for every store-week, given the forecast origin."""
    return encode(add_store_stats(add_lag_features(df), history_end))
