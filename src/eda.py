"""
eda.py
------
Step 2 of the pipeline: Exploratory Data Analysis (EDA).

Each function answers one business question with one chart, and the key
numbers are written to outputs/results/eda_summary.csv for the report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import COLORS, PROCESSED_DIR, RESULTS_DIR, SERIES
from plot_style import apply_style, millions, save, thousands

HOLIDAY_NAMES = {  # holiday weeks as defined by the dataset
    "Super Bowl": ["2010-02-12", "2011-02-11", "2012-02-10"],
    "Labor Day": ["2010-09-10", "2011-09-09", "2012-09-07"],
    "Thanksgiving": ["2010-11-26", "2011-11-25"],
    "Christmas": ["2010-12-31", "2011-12-30"],
}


def chain_trend(df):
    """Q1: How do total weekly sales across all 45 stores move over time?"""
    total = df.groupby("Date")["Weekly_Sales"].sum()
    hol = df.groupby("Date")["IsHoliday"].max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total.index, total.values, color=COLORS["blue"], lw=1.8, label="Total weekly sales")
    ax.scatter(total.index[hol == 1], total[hol == 1], s=36, color=COLORS["orange"],
               edgecolor="white", linewidth=1.2, zorder=3, label="Holiday week")
    # Label the two biggest peaks directly (Thanksgiving & pre-Christmas)
    for d in total.nlargest(4).index[:2]:
        ax.annotate(d.strftime("%d %b %Y"), (d, total[d]), xytext=(6, -4),
                    textcoords="offset points", fontsize=8, color=COLORS["text_muted"])
    ax.set_title("Total weekly sales, all 45 stores (Feb 2010 - Oct 2012)", loc="left")
    ax.set_ylabel("Weekly sales")
    millions(ax)
    ax.legend(loc="upper left")
    save(fig, "01_chain_weekly_sales.png")
    return total


def yearly_seasonality(df):
    """Q2: Is there a repeating yearly pattern? Overlay 2010, 2011, 2012."""
    total = df.groupby(["Year", "WeekOfYear"])["Weekly_Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    for color, (year, g) in zip(SERIES, total.groupby("Year")):
        ax.plot(g["WeekOfYear"], g["Weekly_Sales"], color=color, lw=1.8, label=str(year))
    ax.set_title("Same seasonal shape every year: weekly sales by week of year", loc="left")
    ax.set_xlabel("ISO week of year")
    ax.set_ylabel("Total weekly sales")
    ax.set_xticks(range(1, 53, 4))
    millions(ax)
    ax.legend(title="Year", loc="upper left")
    save(fig, "02_yearly_seasonality.png")


def store_type_and_size(df):
    """Q3: Do store type and store size explain sales level?"""
    per_store = df.groupby(["Store", "Type", "Size"])["Weekly_Sales"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # (a) box plot of average weekly sales per store, by type
    types = ["A", "B", "C"]
    data = [per_store.loc[per_store["Type"] == t, "Weekly_Sales"] for t in types]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.5,
                         medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], SERIES):
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
    axes[0].set_xticks([1, 2, 3], [f"Type {t}\n(n={len(d)})" for t, d in zip(types, data)])
    axes[0].set_title("Average weekly sales per store, by type", loc="left")
    thousands(axes[0])

    # (b) scatter of store size vs average weekly sales
    for c, t in zip(SERIES, types):
        s = per_store[per_store["Type"] == t]
        axes[1].scatter(s["Size"], s["Weekly_Sales"], s=42, color=c, edgecolor="white",
                        linewidth=1.2, label=f"Type {t}")
    r = per_store["Size"].corr(per_store["Weekly_Sales"])
    axes[1].set_title(f"Bigger stores sell more (r = {r:.2f})", loc="left")
    axes[1].set_xlabel("Store size (sq. ft.)")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K"))
    thousands(axes[1])
    axes[1].legend(loc="upper left")
    save(fig, "03_store_type_and_size.png")
    return per_store, r


def holiday_effect(df):
    """Q4: How much higher are sales in holiday weeks? (store-week average)"""
    avg = df.groupby("IsHoliday")["Weekly_Sales"].mean()
    # Separate the four named holidays (dates differ every year)
    rows = []
    for name, dates in HOLIDAY_NAMES.items():
        m = df["Date"].isin(pd.to_datetime(dates))
        rows.append((name, df.loc[m, "Weekly_Sales"].mean()))
    named = pd.DataFrame(rows, columns=["Holiday", "Avg"])
    named = pd.concat([pd.DataFrame([("Normal week", avg[0])], columns=["Holiday", "Avg"]), named])

    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [COLORS["grey"]] + [COLORS["blue"]] * 4
    bars = ax.barh(named["Holiday"], named["Avg"], color=colors, height=0.55)
    for b, v in zip(bars, named["Avg"]):
        lift = (v / avg[0] - 1) * 100
        label = f"${v / 1e3:,.0f}K" + ("" if b == bars[0] else f"  ({lift:+.0f}%)")
        ax.text(v + 8000, b.get_y() + b.get_height() / 2, label, va="center",
                fontsize=9, color=COLORS["text"])
    ax.invert_yaxis()
    ax.set_xlim(0, named["Avg"].max() * 1.28)
    ax.set_title("Average sales per store in holiday weeks vs normal weeks", loc="left")
    thousands(ax, axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, "04_holiday_effect.png")
    return avg, named


def external_factor_correlation(df):
    """
    Q5: Do temperature, fuel price, CPI, unemployment or markdowns move with
    sales? Correlation is computed WITHIN each store (sales and features are
    de-meaned per store) so differences in store size do not dominate.
    """
    cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "MarkDown_Total"]
    demeaned = df[["Weekly_Sales"] + cols] - df.groupby("Store")[["Weekly_Sales"] + cols].transform("mean")
    corr = demeaned[cols].corrwith(demeaned["Weekly_Sales"]).sort_values()

    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [COLORS["orange"] if v < 0 else COLORS["blue"] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, height=0.55)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=COLORS["text"])
    ax.axvline(0, color=COLORS["text_muted"], lw=0.8)
    lim = max(0.12, corr.abs().max() * 1.6)
    ax.set_xlim(-lim, lim)
    ax.set_title("External factors barely move weekly sales (within-store correlation)", loc="left")
    ax.set_xlabel("Pearson correlation with weekly sales")
    ax.grid(axis="y", visible=False)
    save(fig, "05_external_factor_correlation.png")
    return corr


def top_departments():
    """Q6: Which departments drive revenue? (top 10 by total sales)"""
    dept = pd.read_csv(PROCESSED_DIR / "department_totals.csv").head(10)
    share = dept["Weekly_Sales"].sum()  # top-10 combined, used for label only

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(dept["Dept"].astype(str), dept["Weekly_Sales"], color=COLORS["blue"], height=0.6)
    ax.invert_yaxis()
    ax.set_title("Top 10 departments by total sales (Feb 2010 - Oct 2012)", loc="left")
    ax.set_ylabel("Department number")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1e6:,.0f}M"))
    ax.grid(axis="y", visible=False)
    save(fig, "06_top_departments.png")
    return dept, share


def run(df=None):
    print("[2/4] Exploratory data analysis ...")
    apply_style()
    if df is None:
        df = pd.read_csv(PROCESSED_DIR / "store_weekly.csv", parse_dates=["Date"])

    total = chain_trend(df)
    yearly_seasonality(df)
    per_store, size_r = store_type_and_size(df)
    avg, named = holiday_effect(df)
    corr = external_factor_correlation(df)
    dept, top10_total = top_departments()
    all_dept_total = pd.read_csv(PROCESSED_DIR / "department_totals.csv")["Weekly_Sales"].sum()

    # Year-over-year growth using the weeks present in all three years (Feb-Oct)
    common = df[df["Date"].dt.month.between(2, 10)]
    by_year = common.groupby("Year")["Weekly_Sales"].sum()

    summary = pd.DataFrame([
        ("Average total weekly sales (all stores)", total.mean()),
        ("Highest week", f"{total.idxmax().date()} (${total.max() / 1e6:.1f}M)"),
        ("Lowest week", f"{total.idxmin().date()} (${total.min() / 1e6:.1f}M)"),
        ("Avg store-week sales, normal weeks", avg[0]),
        ("Avg store-week sales, holiday weeks", avg[1]),
        ("Holiday lift (all holiday weeks)", f"{(avg[1] / avg[0] - 1) * 100:.1f}%"),
        ("Thanksgiving lift", f"{(named.iloc[3]['Avg'] / avg[0] - 1) * 100:.1f}%"),
        ("Correlation store size vs avg sales", round(size_r, 3)),
        ("Type A avg weekly sales per store", per_store.query("Type=='A'")["Weekly_Sales"].mean()),
        ("Type B avg weekly sales per store", per_store.query("Type=='B'")["Weekly_Sales"].mean()),
        ("Type C avg weekly sales per store", per_store.query("Type=='C'")["Weekly_Sales"].mean()),
        ("Top-10 departments share of revenue", f"{top10_total / all_dept_total * 100:.1f}%"),
        ("Feb-Oct sales growth 2010 -> 2011", f"{(by_year[2011] / by_year[2010] - 1) * 100:.1f}%"),
        ("Feb-Oct sales growth 2011 -> 2012", f"{(by_year[2012] / by_year[2011] - 1) * 100:.1f}%"),
    ] + [(f"Within-store corr: {k}", round(v, 3)) for k, v in corr.items()],
        columns=["Metric", "Value"])
    summary.to_csv(RESULTS_DIR / "eda_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
