"""
train_evaluate.py
-----------------
Step 3 of the pipeline: train six forecasting models on 104 weeks of data
(Feb 2010 - Jan 2012) and compare them on the 39 unseen weeks that follow
(Feb 2012 - Oct 2012), for all 45 stores.

Models
  Baselines  1. Naive            - repeat the last observed week
             2. Seasonal naive   - repeat the same week from last year
  Statistical 3. Holt-Winters    - additive triple exponential smoothing
  Machine    4. Ridge regression - regularised linear model
  learning   5. Random forest    - bagged decision trees
             6. Gradient boosting (HistGradientBoostingRegressor), tuned

Metrics (lower is better)
  MAE, RMSE, MAPE and WMAE (holiday weeks weighted 5x, the official
  Kaggle competition metric).
"""

import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (COLORS, HOLIDAY_WEIGHT, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR,
                    SEASON_LENGTH, SERIES, TEST_START, TRAIN_END, VALID_START)
from features import FEATURE_COLS, TARGET, build_matrix
from holt_winters import HoltWintersAdditive
from plot_style import apply_style, millions, save, thousands


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wmae(y, y_hat, holiday):
    """Weighted MAE: holiday weeks count HOLIDAY_WEIGHT times more."""
    w = np.where(holiday == 1, HOLIDAY_WEIGHT, 1)
    return np.sum(w * np.abs(y - y_hat)) / np.sum(w)


def all_metrics(y, y_hat, holiday):
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    return {
        "MAE": np.mean(np.abs(y - y_hat)),
        "RMSE": np.sqrt(np.mean((y - y_hat) ** 2)),
        "MAPE_%": np.mean(np.abs((y - y_hat) / y)) * 100,
        "WMAE": wmae(y, y_hat, np.asarray(holiday)),
    }


# ---------------------------------------------------------------------------
# Baseline & statistical models (fit separately for every store)
# ---------------------------------------------------------------------------
def naive_forecast(train, test):
    last = train.sort_values("Date").groupby("Store")["Weekly_Sales"].last()
    return test["Store"].map(last).to_numpy()


def seasonal_naive_forecast(full_feats, test):
    # lag_52 already holds "same week last year" for every row
    return test.merge(full_feats[["Store", "Date", "lag_52"]], on=["Store", "Date"])["lag_52"].to_numpy()


def holt_winters_forecast(train, test):
    preds, params = [], []
    for store, g in train.groupby("Store"):
        horizon = (test["Store"] == store).sum()
        hw = HoltWintersAdditive(SEASON_LENGTH).fit(g.sort_values("Date")["Weekly_Sales"].to_numpy())
        preds.append(pd.DataFrame({
            "Store": store,
            "Date": np.sort(test.loc[test["Store"] == store, "Date"].unique()),
            "pred": hw.forecast(horizon),
        }))
        params.append({"Store": store, **hw.params_})
    p = pd.concat(preds)
    pd.DataFrame(params).to_csv(RESULTS_DIR / "holt_winters_params.csv", index=False)
    return test.merge(p, on=["Store", "Date"], how="left")["pred"].to_numpy()


# ---------------------------------------------------------------------------
# Machine-learning models (one global model across all stores)
# ---------------------------------------------------------------------------
def make_ridge():
    numeric = [c for c in FEATURE_COLS if c not in ("Store", "Type_code")]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type_code"]),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def make_random_forest():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.6,
                                 n_jobs=-1, random_state=RANDOM_STATE)


def make_gbm(**params):
    return HistGradientBoostingRegressor(random_state=RANDOM_STATE, loss="absolute_error",
                                         categorical_features=None, **params)


def tune_gbm(df):
    """
    Small grid search with a TIME-BASED validation split:
      fit on Feb 2011 - Oct 2011, validate on Nov 2011 - Jan 2012
    (the validation weeks include Thanksgiving and Christmas, the hardest
    part of the year). Nothing from the test period is touched.
    """
    origin = pd.Timestamp(VALID_START) - pd.Timedelta(weeks=1)
    feats = build_matrix(df, history_end=origin)
    rows = feats.dropna(subset=["lag_52"])
    fit = rows[rows["Date"] <= origin]
    val = rows[(rows["Date"] >= VALID_START) & (rows["Date"] <= TRAIN_END)]

    grid = {"learning_rate": [0.03, 0.1], "max_iter": [300, 600],
            "max_leaf_nodes": [15, 31], "min_samples_leaf": [10, 20]}
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = make_gbm(**params).fit(fit[FEATURE_COLS], fit[TARGET])
        score = wmae(val[TARGET].to_numpy(), model.predict(val[FEATURE_COLS]), val["IsHoliday"].to_numpy())
        results.append({**params, "valid_WMAE": score})
    res = pd.DataFrame(results).sort_values("valid_WMAE")
    res.to_csv(RESULTS_DIR / "gbm_tuning_results.csv", index=False)
    best = res.iloc[0].drop("valid_WMAE").to_dict()
    best = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best.items()}
    print(f"   best gradient boosting params (validation WMAE {res.iloc[0]['valid_WMAE']:,.0f}): {best}")
    return best


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_model_comparison(metrics):
    m = metrics.sort_values("WMAE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    best = m["WMAE"].min()
    colors = [COLORS["blue"] if v == best else COLORS["grey"] for v in m["WMAE"]]
    ax.barh(m["Model"], m["WMAE"], color=colors, height=0.55)
    for i, (v, mape) in enumerate(zip(m["WMAE"], m["MAPE_%"])):
        ax.text(v + best * 0.03, i, f"${v / 1e3:,.1f}K   (MAPE {mape:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, m["WMAE"].max() * 1.45)
    ax.set_title("Test-period error by model (WMAE, lower is better)", loc="left")
    thousands(ax, axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, "07_model_comparison.png")


def plot_chain_forecast(train, preds, best_name):
    hist = train.groupby("Date")["Weekly_Sales"].sum()
    hist = hist[hist.index >= "2011-06-01"]
    cols = list(dict.fromkeys(["Actual", best_name, "Seasonal naive", "Holt-Winters"]))
    agg = preds.groupby("Date")[cols].sum()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(hist.index, hist.values, color=COLORS["grey"], lw=1.5, label="Training data")
    ax.plot(agg.index, agg["Actual"], color=COLORS["text"], lw=1.8, label="Actual (test)")
    ax.plot(agg.index, agg[best_name], color=SERIES[0], lw=1.8, label=best_name)
    for name, color, ls in (("Seasonal naive", SERIES[1], "--"), ("Holt-Winters", SERIES[2], ":")):
        if name != best_name:  # avoid drawing the same line twice
            ax.plot(agg.index, agg[name], color=color, lw=1.4, ls=ls, label=name)
    ax.axvline(pd.Timestamp(TEST_START), color=COLORS["text_muted"], lw=0.8, ls="--")
    ax.text(pd.Timestamp(TEST_START), ax.get_ylim()[0], "  test period starts",
            fontsize=8, color=COLORS["text_muted"], va="bottom")
    ax.set_title("Chain-wide forecast vs actual, Feb - Oct 2012 (sum of 45 store forecasts)", loc="left")
    millions(ax)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "08_chain_forecast_vs_actual.png")


def plot_example_stores(train, preds, best_name, stores):
    fig, axes = plt.subplots(1, len(stores), figsize=(12, 3.4), sharey=False)
    for ax, store in zip(axes, stores):
        h = train[(train["Store"] == store) & (train["Date"] >= "2011-09-01")]
        p = preds[preds["Store"] == store]
        ax.plot(h["Date"], h["Weekly_Sales"], color=COLORS["grey"], lw=1.3)
        ax.plot(p["Date"], p["Actual"], color=COLORS["text"], lw=1.5, label="Actual")
        ax.plot(p["Date"], p[best_name], color=SERIES[0], lw=1.5, label=best_name)
        mape = np.mean(np.abs(p["Actual"] - p[best_name]) / p["Actual"]) * 100
        typ = train.loc[train["Store"] == store, "Type"].iloc[0]
        ax.set_title(f"Store {store} (Type {typ}) - MAPE {mape:.1f}%", loc="left", fontsize=10)
        thousands(ax)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].legend(loc="upper left")
    save(fig, "09_example_store_forecasts.png")


def plot_feature_importance(imp, model_name):
    imp = imp.head(10).sort_values("importance")  # top 10 features
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color=COLORS["blue"], height=0.6,
            error_kw=dict(ecolor=COLORS["text_muted"], lw=0.8))
    ax.set_title(f"Top 10 drivers of the {model_name.lower()} forecast (permutation importance)", loc="left")
    ax.set_xlabel("Increase in test MAE when the feature is shuffled")
    thousands(ax, axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, "10_feature_importance.png")


def plot_per_store_error(per_store):
    s = per_store.sort_values("MAPE_%")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = [SERIES[{"A": 0, "B": 1, "C": 2}[t]] for t in s["Type"]]
    ax.bar(s["Store"].astype(str), s["MAPE_%"], color=colors, width=0.7)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f"Type {t}") for c, t in zip(SERIES, "ABC")]
    ax.axhline(s["MAPE_%"].median(), color=COLORS["text_muted"], lw=0.8, ls="--")
    ax.text(-0.5, s["MAPE_%"].median() + 0.15, f"median {s['MAPE_%'].median():.1f}%",
            fontsize=8, color=COLORS["text_muted"], ha="left", va="bottom")
    ax.set_title("Forecast error per store, best model (MAPE, test period)", loc="left")
    ax.set_xlabel("Store (sorted by error)")
    ax.set_ylabel("MAPE %")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 1))
    save(fig, "11_per_store_error.png")


# ---------------------------------------------------------------------------
def run(df=None):
    print("[3/4] Training and evaluating models ...")
    apply_style()
    if df is None:
        df = pd.read_csv(PROCESSED_DIR / "store_weekly.csv", parse_dates=["Date"])

    train = df[df["Date"] <= TRAIN_END].copy()
    test = df[df["Date"] >= TEST_START].copy().sort_values(["Store", "Date"]).reset_index(drop=True)
    print(f"   train: {train['Date'].min().date()} -> {train['Date'].max().date()} "
          f"({train['Date'].nunique()} weeks, {len(train)} rows)")
    print(f"   test : {test['Date'].min().date()} -> {test['Date'].max().date()} "
          f"({test['Date'].nunique()} weeks, {len(test)} rows)")

    # Feature table with the forecast origin = last training week
    feats = build_matrix(df, history_end=TRAIN_END)
    ml_train = feats[(feats["Date"] <= TRAIN_END)].dropna(subset=["lag_52"])
    ml_test = feats[feats["Date"] >= TEST_START].sort_values(["Store", "Date"]).reset_index(drop=True)
    assert (ml_test[["Store", "Date"]].to_numpy() == test[["Store", "Date"]].to_numpy()).all()
    print(f"   ML training rows (weeks with a 52-week lag available): {len(ml_train)}")

    preds = test[["Store", "Date", "Type", "IsHoliday"]].copy()
    preds["Actual"] = test["Weekly_Sales"].to_numpy()
    timings = {}

    def timed(name, fn):
        t0 = time.time()
        preds[name] = fn()
        timings[name] = time.time() - t0

    timed("Naive", lambda: naive_forecast(train, test))
    timed("Seasonal naive", lambda: seasonal_naive_forecast(feats, test))
    timed("Holt-Winters", lambda: holt_winters_forecast(train, test))

    X_tr, y_tr, X_te = ml_train[FEATURE_COLS], ml_train[TARGET], ml_test[FEATURE_COLS]
    ridge = make_ridge()
    timed("Ridge regression", lambda: ridge.fit(X_tr, y_tr).predict(X_te))
    rf = make_random_forest()
    timed("Random forest", lambda: rf.fit(X_tr, y_tr).predict(X_te))
    best_params = tune_gbm(df)
    gbm = make_gbm(**best_params)
    timed("Gradient boosting", lambda: gbm.fit(X_tr, y_tr).predict(X_te))

    model_names = ["Naive", "Seasonal naive", "Holt-Winters",
                   "Ridge regression", "Random forest", "Gradient boosting"]
    metrics = pd.DataFrame([
        {"Model": m, **all_metrics(preds["Actual"], preds[m], preds["IsHoliday"]),
         "Fit_seconds": timings[m]} for m in model_names
    ])
    sn = metrics.set_index("Model").loc["Seasonal naive", "WMAE"]
    metrics["WMAE_vs_seasonal_naive_%"] = (metrics["WMAE"] / sn - 1) * 100
    metrics = metrics.sort_values("WMAE").reset_index(drop=True)
    best_name = metrics.loc[0, "Model"]
    metrics.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    preds.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    print(metrics.round(2).to_string(index=False))
    print(f"   best model: {best_name}")

    # Chain-level accuracy (business view: total sales across all stores)
    chain = preds.groupby("Date")[["Actual"] + model_names].sum()
    chain_err = pd.DataFrame({
        "Model": model_names,
        "Chain_MAPE_%": [np.mean(np.abs(chain[m] - chain["Actual"]) / chain["Actual"]) * 100 for m in model_names],
        "Chain_total_error_%": [(chain[m].sum() / chain["Actual"].sum() - 1) * 100 for m in model_names],
    }).sort_values("Chain_MAPE_%")
    chain_err.to_csv(RESULTS_DIR / "chain_level_error.csv", index=False)
    print(chain_err.round(2).to_string(index=False))

    # Per-store error of the best model
    per_store = preds.groupby(["Store", "Type"]).apply(
        lambda g: pd.Series(all_metrics(g["Actual"], g[best_name], g["IsHoliday"])),
        include_groups=False).reset_index()
    per_store.to_csv(RESULTS_DIR / "per_store_error.csv", index=False)

    # Permutation importance for the best machine-learning model (on test data)
    ml_models = {"Ridge regression": ridge, "Random forest": rf, "Gradient boosting": gbm}
    imp_name = next(m for m in metrics["Model"] if m in ml_models)
    pi = permutation_importance(ml_models[imp_name], X_te, ml_test[TARGET], scoring="neg_mean_absolute_error",
                                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # Charts
    plot_model_comparison(metrics)
    plot_chain_forecast(train, preds, best_name)
    best_store, worst_store = per_store.sort_values("MAPE_%")["Store"].iloc[[0, -1]]
    median_store = per_store.sort_values("MAPE_%")["Store"].iloc[len(per_store) // 2]
    plot_example_stores(train, preds, best_name, [int(best_store), int(median_store), int(worst_store)])
    plot_feature_importance(imp, imp_name)
    plot_per_store_error(per_store)

    return {"metrics": metrics, "best_name": best_name, "best_params": best_params,
            "per_store": per_store, "importance": imp, "importance_model": imp_name, "chain": chain_err}


if __name__ == "__main__":
    run()
