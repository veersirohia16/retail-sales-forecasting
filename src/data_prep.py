"""
data_prep.py
------------
Step 1 of the pipeline: load the three raw Kaggle files, check data quality,
clean them, and merge them into ONE tidy table with one row per
(store, week).

Raw files (Walmart Recruiting - Store Sales Forecasting, Kaggle):
    train.csv    : Store, Dept, Date, Weekly_Sales, IsHoliday   (421,570 rows)
    features.csv : Store, Date, Temperature, Fuel_Price, MarkDown1-5,
                   CPI, Unemployment, IsHoliday                 (8,190 rows)
    stores.csv   : Store, Type (A/B/C), Size (sq. ft.)          (45 rows)

Output:
    data/processed/store_weekly.csv
"""

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


def load_raw():
    """Read the three raw CSV files and parse the Date columns."""
    train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(RAW_DIR / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    return train, features, stores


def quality_report(train, features, stores):
    """
    Collect simple data-quality facts (shape, missing values, duplicates,
    negative sales). The table is saved so it can be quoted in the report.
    """
    rows = [
        ("train.csv rows", len(train)),
        ("train.csv columns", train.shape[1]),
        ("Stores", train["Store"].nunique()),
        ("Departments", train["Dept"].nunique()),
        ("Weeks", train["Date"].nunique()),
        ("First week", train["Date"].min().date()),
        ("Last week", train["Date"].max().date()),
        ("Duplicate rows in train.csv", int(train.duplicated().sum())),
        ("Missing values in train.csv", int(train.isna().sum().sum())),
        ("Negative Weekly_Sales rows (returns)", int((train["Weekly_Sales"] < 0).sum())),
        ("features.csv rows", len(features)),
    ]
    # Missing-value percentage for each column of features.csv
    for col, pct in (features.isna().mean() * 100).items():
        if pct > 0:
            rows.append((f"features.csv missing % - {col}", round(pct, 1)))
    rows.append(("stores.csv rows", len(stores)))

    report = pd.DataFrame(rows, columns=["Check", "Value"])
    report.to_csv(RESULTS_DIR / "data_quality_report.csv", index=False)
    return report


def build_store_weekly(train, features, stores):
    """
    Turn department-level sales into store-level weekly sales and attach the
    external features and store information.

    Why store level?  Store managers plan staffing, inventory and cash flow
    per store per week, so that is the level the business forecasts at.
    """
    # 1) Department -> store aggregation.
    #    Negative department values are product returns; they are genuine
    #    accounting entries, so we keep them inside the store total.
    sales = (
        train.groupby(["Store", "Date"], as_index=False)
        .agg(Weekly_Sales=("Weekly_Sales", "sum"), IsHoliday=("IsHoliday", "max"))
    )

    # 2) Attach economic / weather / promotion features (same store & week).
    #    features.csv has its own IsHoliday column, which duplicates ours.
    df = sales.merge(features.drop(columns="IsHoliday"), on=["Store", "Date"], how="left")

    # 3) Attach store type and size.
    df = df.merge(stores, on="Store", how="left")

    # 4) Missing markdowns: Walmart only started recording promotional
    #    markdowns in Nov 2011, and a blank afterwards means "no markdown that
    #    week". Both cases are filled with 0, and a flag keeps the information.
    df["MarkDown_Recorded"] = df[MARKDOWN_COLS].notna().any(axis=1).astype(int)
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)
    df["MarkDown_Total"] = df[MARKDOWN_COLS].sum(axis=1)

    # 5) Calendar features.
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # 6) Sanity checks: every store must have every week exactly once.
    n_weeks = df["Date"].nunique()
    assert df.groupby("Store").size().eq(n_weeks).all(), "missing store-weeks"
    assert not df.duplicated(["Store", "Date"]).any(), "duplicate store-weeks"
    assert df[["Temperature", "Fuel_Price", "CPI", "Unemployment"]].notna().all().all()

    return df


def department_totals(train):
    """Total sales per department (used for one EDA chart)."""
    return (
        train.groupby("Dept")["Weekly_Sales"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def run():
    print("[1/4] Loading and cleaning data ...")
    train, features, stores = load_raw()
    report = quality_report(train, features, stores)
    print(report.to_string(index=False))

    df = build_store_weekly(train, features, stores)
    df.to_csv(PROCESSED_DIR / "store_weekly.csv", index=False)
    department_totals(train).to_csv(PROCESSED_DIR / "department_totals.csv", index=False)
    print(f"   processed dataset: {df.shape[0]} rows x {df.shape[1]} columns "
          f"-> data/processed/store_weekly.csv")
    return df, train


if __name__ == "__main__":
    run()
