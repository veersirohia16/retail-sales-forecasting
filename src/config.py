"""
config.py
---------
Central configuration for the Retail Sales Forecasting project.

Every other module imports its paths, split dates and plotting colours from
here, so changing an experiment setting only requires editing this one file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folder layout (all paths are relative to the project root)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"              # original Kaggle CSV files
PROCESSED_DIR = ROOT_DIR / "data" / "processed"  # cleaned, merged dataset
FIG_DIR = ROOT_DIR / "outputs" / "figures"       # every chart used in the report
RESULTS_DIR = ROOT_DIR / "outputs" / "results"   # metrics tables, forecasts

for _d in (PROCESSED_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Time-based train / test split
# ---------------------------------------------------------------------------
# The data runs from 2010-02-05 to 2012-10-26 (143 weekly observations).
# We keep exactly two years (104 weeks) for training and hold out the
# remaining 39 weeks of 2012 as an unseen test period. A random split is NOT
# used, because it would leak future information into training.
TRAIN_END = "2012-01-27"
TEST_START = "2012-02-03"

# Inside the training period, the last 13 weeks (the 2011 holiday season)
# are used as a validation set for hyper-parameter tuning.
VALID_START = "2011-11-04"

# Forecast horizon for the "future" forecast made after the data ends
FUTURE_WEEKS = 13

# Weekly data -> one year = 52 weeks (Walmart weeks end on a Friday, so a
# 52-week lag lands on the same holiday week of the previous year).
SEASON_LENGTH = 52

# Holiday weeks are weighted 5x in the WMAE metric (the official metric of
# the original Kaggle competition).
HOLIDAY_WEIGHT = 5

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Plot styling (colour-blind-safe categorical palette, fixed order)
# ---------------------------------------------------------------------------
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#6250d6",
    "grey": "#8a8984",
    "text": "#0b0b0b",
    "text_muted": "#52514e",
    "grid": "#e6e5e0",
}
SERIES = [COLORS["blue"], COLORS["orange"], COLORS["aqua"]]
