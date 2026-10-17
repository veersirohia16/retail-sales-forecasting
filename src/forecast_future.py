"""
forecast_future.py
------------------
Step 4 of the pipeline: use the winning model to forecast the 13 weeks AFTER
the data ends (2 Nov 2012 - 25 Jan 2013), i.e. the next holiday season,
for every store.

The model is refit on ALL 143 weeks so it uses the most recent information.
features.csv already contains temperature, fuel price, markdowns etc. for
these future weeks; CPI and unemployment are blank for 2013, so the last
known value per store is carried forward (a common, conservative choice).
There are no actual sales for these weeks, so this is a business output,
not an accuracy test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import COLORS, FUTURE_WEEKS, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SERIES
from features import FEATURE_COLS, TARGET, build_matrix
from plot_style import apply_style, millions, save
from train_evaluate import make_gbm, make_random_forest, make_ridge


def future_frame(df):
    """Build store-week rows for the 13 future weeks, with external features."""
    features = pd.read_csv(RAW_DIR / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    last = df["Date"].max()
    future_dates = pd.date_range(last + pd.Timedelta(weeks=1), periods=FUTURE_WEEKS, freq="7D")

    fut = features[features["Date"].isin(future_dates)].merge(stores, on="Store")
    # carry the last known CPI / unemployment forward per store
    fut = fut.sort_values(["Store", "Date"])
    last_known = df.sort_values("Date").groupby("Store")[["CPI", "Unemployment"]].last()
    for col in ["CPI", "Unemployment"]:
        fut[col] = fut[col].fillna(fut["Store"].map(last_known[col]))

    md = [f"MarkDown{i}" for i in range(1, 6)]
    fut["MarkDown_Recorded"] = fut[md].notna().any(axis=1).astype(int)
    fut[md] = fut[md].fillna(0)
    fut["MarkDown_Total"] = fut[md].sum(axis=1)
    fut["IsHoliday"] = fut["IsHoliday"].astype(int)
    fut["Year"] = fut["Date"].dt.year
    fut["Month"] = fut["Date"].dt.month
    fut["WeekOfYear"] = fut["Date"].dt.isocalendar().week.astype(int)
    fut["Weekly_Sales"] = np.nan  # unknown -> this is what we forecast
    return fut[df.columns]


def run(best_name="Random forest", best_params=None, df=None):
    print("[4/4] Forecasting the next 13 weeks ...")
    apply_style()
    if df is None:
        df = pd.read_csv(PROCESSED_DIR / "store_weekly.csv", parse_dates=["Date"])

    fut = future_frame(df)
    full = pd.concat([df, fut], ignore_index=True)
    feats = build_matrix(full, history_end=df["Date"].max())

    train_rows = feats[feats["Weekly_Sales"].notna()].dropna(subset=["lag_52"])
    future_rows = feats[feats["Weekly_Sales"].isna()].sort_values(["Store", "Date"])

    makers = {"Random forest": make_random_forest, "Ridge regression": make_ridge,
              "Gradient boosting": lambda: make_gbm(**(best_params or {}))}
    model_name = best_name if best_name in makers else "Random forest"
    model = makers[model_name]().fit(train_rows[FEATURE_COLS], train_rows[TARGET])
    future_rows = future_rows.assign(Forecast=model.predict(future_rows[FEATURE_COLS]))

    out = future_rows[["Store", "Type", "Date", "IsHoliday", "lag_52", "Forecast"]].rename(
        columns={"lag_52": "Sales_same_week_last_year"})
    out.to_csv(RESULTS_DIR / "future_forecast_by_store.csv", index=False)
    chain = out.groupby("Date")[["Forecast", "Sales_same_week_last_year"]].sum()
    chain.to_csv(RESULTS_DIR / "future_forecast_chain_total.csv")

    # chart: last 40 weeks of history + 13-week forecast
    hist = df.groupby("Date")["Weekly_Sales"].sum()
    hist = hist[hist.index >= "2012-01-01"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist.index, hist.values, color=COLORS["text"], lw=1.6, label="Actual sales")
    ax.plot(chain.index, chain["Forecast"], color=SERIES[0], lw=2, marker="o", ms=4,
            label=f"Forecast ({model_name.lower()})")
    ax.plot(chain.index, chain["Sales_same_week_last_year"], color=SERIES[1], lw=1.2, ls="--",
            label="Same week last year")
    peak = chain["Forecast"].idxmax()
    ax.annotate(f"peak week {peak:%d %b %Y}\n${chain['Forecast'].max() / 1e6:.1f}M",
                (peak, chain["Forecast"].max()), xytext=(-110, -10), textcoords="offset points",
                fontsize=8, color=COLORS["text_muted"])
    ax.set_title("Holiday-season forecast for all 45 stores (Nov 2012 - Jan 2013)", loc="left")
    millions(ax)
    ax.legend(loc="upper left")
    save(fig, "12_future_forecast.png")

    print(f"   model used: {model_name}; forecast total for 13 weeks: "
          f"${chain['Forecast'].sum() / 1e6:,.1f}M "
          f"(same weeks last year: ${chain['Sales_same_week_last_year'].sum() / 1e6:,.1f}M)")
    return out, chain


if __name__ == "__main__":
    run()
