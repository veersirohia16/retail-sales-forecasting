"""
holt_winters.py
---------------
A from-scratch implementation of additive Holt-Winters exponential smoothing
(triple exponential smoothing), the classic statistical method for series
with a trend and a repeating season.

Writing it ourselves (instead of importing statsmodels) keeps the project
dependency-light and shows exactly how the method works.

Model (additive), season length m:
    level     l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
    trend     b_t = beta  * (l_t - l_{t-1}) + (1 - beta)  * b_{t-1}
    season    s_t = gamma * (y_t - l_t)     + (1 - gamma) * s_{t-m}
    forecast  y_hat_{t+h} = l_t + h * b_t + s_{t+h-m*k}   (k = ceil(h/m))

alpha, beta, gamma (all between 0 and 1) are chosen to minimise the sum of
squared one-step-ahead errors on the training data.
"""

import numpy as np
from scipy.optimize import minimize


class HoltWintersAdditive:
    def __init__(self, season_length=52):
        self.m = season_length
        self.params_ = None

    # ------------------------------------------------------------------
    def _initial_state(self, y):
        """Initial level, trend and seasonal indices from the first 2 seasons."""
        m = self.m
        season1, season2 = y[:m], y[m:2 * m]
        level = season1.mean()
        trend = (season2.mean() - season1.mean()) / m
        seasonals = season1 - level
        return level, trend, seasonals.copy()

    def _smooth(self, y, alpha, beta, gamma):
        """Run the recursions; return one-step errors and the final state."""
        m = self.m
        level, trend, seasonals = self._initial_state(y)
        errors = np.empty(len(y))
        for t in range(len(y)):
            s = seasonals[t % m]
            prediction = level + trend + s           # one-step-ahead forecast
            errors[t] = y[t] - prediction
            new_level = alpha * (y[t] - s) + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            seasonals[t % m] = gamma * (y[t] - new_level) + (1 - gamma) * s
            level = new_level
        return errors, level, trend, seasonals

    # ------------------------------------------------------------------
    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if len(y) < 2 * self.m:
            raise ValueError("Holt-Winters needs at least two full seasons of data")
        self.y_ = y

        def sse(p):
            errors, *_ = self._smooth(y, *p)
            # skip the first season: those errors only reflect initialisation
            return np.sum(errors[self.m:] ** 2)

        best = None
        # a few starting points avoid getting stuck in a poor local minimum
        for start in ([0.3, 0.05, 0.3], [0.1, 0.01, 0.1], [0.6, 0.1, 0.6]):
            res = minimize(sse, start, method="L-BFGS-B", bounds=[(0, 1)] * 3)
            if best is None or res.fun < best.fun:
                best = res
        self.params_ = dict(zip(["alpha", "beta", "gamma"], best.x))
        _, self.level_, self.trend_, self.seasonals_ = self._smooth(y, *best.x)
        return self

    def forecast(self, horizon):
        """Forecast the next `horizon` steps after the end of the training data."""
        n = len(self.y_)
        steps = np.arange(1, horizon + 1)
        season_idx = (n + steps - 1) % self.m
        return self.level_ + steps * self.trend_ + self.seasonals_[season_idx]
