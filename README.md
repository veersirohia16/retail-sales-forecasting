# Retail Sales Forecasting for a 45-Store Chain

**Author:** Veer Sirohia, B.Tech Data Science, Manipal University Jaipur
**Type:** Minor project (data science)

Forecasts weekly sales for each of 45 Walmart stores using two years of history,
compares six models (baselines, Holt-Winters, Ridge, Random Forest, Gradient
Boosting) on 39 unseen weeks, and produces a 13-week holiday-season forecast.

## Result in one line
A random forest with engineered seasonal features cut weighted error (WMAE) by
**12.4%** versus the seasonal-naive baseline, with a store-level MAPE of **4.8%**
and a chain-level MAPE of **2.6%** on Feb - Oct 2012.

## Dataset
Walmart Recruiting - Store Sales Forecasting (Kaggle):
<https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data>
Copy used in this project was taken from the public GitHub repository
<https://github.com/ShiviBhatt/E-Commerce_Sales_Forecasting> (folder `data/`).

| File | Rows | Contents |
|---|---|---|
| `data/raw/train.csv` | 421,570 | Store, Dept, Date, Weekly_Sales, IsHoliday |
| `data/raw/features.csv` | 8,190 | Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment |
| `data/raw/stores.csv` | 45 | Store type (A/B/C) and size |

The competition's `test.csv` has no sales values, so it is not used; the
project creates its own time-based test period from `train.csv`.

## How to run
```bash
pip install -r requirements.txt
python main.py
```
Runs in under a minute. Every chart is saved in `outputs/figures/`, every
table in `outputs/results/`, and the cleaned dataset in `data/processed/`.
Each step can also be run on its own, e.g. `cd src && python eda.py`.

## Folder structure
```
Retail_Sales_Forecasting/
├── main.py                  runs the full pipeline
├── requirements.txt
├── data/
│   ├── raw/                 original Kaggle CSVs
│   └── processed/           store_weekly.csv (cleaned, merged), department_totals.csv
├── src/
│   ├── config.py            paths, split dates, colours
│   ├── data_prep.py         load, quality checks, cleaning, merging
│   ├── eda.py               exploratory analysis (charts 01-06)
│   ├── features.py          leakage-safe feature engineering
│   ├── holt_winters.py      Holt-Winters implemented from scratch
│   ├── train_evaluate.py    6 models, tuning, metrics (charts 07-11)
│   ├── forecast_future.py   13-week forecast (chart 12)
│   └── plot_style.py        shared chart style
└── outputs/
    ├── figures/             12 PNG charts
    └── results/             metrics, predictions, forecasts (CSV)
```

## Key output files
- `outputs/results/model_comparison.csv` - MAE, RMSE, MAPE, WMAE per model
- `outputs/results/test_predictions.csv` - every model's prediction for every store-week in the test period
- `outputs/results/per_store_error.csv` - error of the best model per store
- `outputs/results/future_forecast_by_store.csv` - Nov 2012 - Jan 2013 forecast per store
