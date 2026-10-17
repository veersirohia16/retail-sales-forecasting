"""
main.py
-------
Runs the whole Retail Sales Forecasting pipeline end to end:

    1. data_prep.py       load, check, clean and merge the raw data
    2. eda.py             exploratory analysis + 6 charts
    3. train_evaluate.py  train 6 models, compare them on 39 unseen weeks
    4. forecast_future.py forecast the next 13 weeks with the best model

Usage (from the project folder):
    pip install -r requirements.txt
    python main.py

All charts are written to outputs/figures/ and all tables to outputs/results/.
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

import data_prep        # noqa: E402
import eda              # noqa: E402
import forecast_future  # noqa: E402
import train_evaluate   # noqa: E402


def main():
    start = time.time()
    df, _ = data_prep.run()
    eda.run(df)
    result = train_evaluate.run(df)
    forecast_future.run(result["best_name"], result["best_params"], df)
    print(f"\nDone in {time.time() - start:.0f} s. Best model on the test period: {result['best_name']}")


if __name__ == "__main__":
    main()
