"""Forecasting engine: lag-feature Ridge regression (log returns) vs a naive baseline."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAGS = [1, 2, 3, 7, 14, 28]
USE = ["d7", "dev30", "sin", "cos"]     # simple, stable feature set (mean-reversion + season)


def clean_series(df: pd.DataFrame) -> pd.Series:
    """Daily, gap-filled price series from a (date, modal_price) frame."""
    s = (df.assign(date=pd.to_datetime(df["date"]))
           .groupby("date")["modal_price"].mean().sort_index())
    s = s.asfreq("D").interpolate(limit_direction="both")
    return s[s > 0]


def _features(logp: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=logp.index)
    for k in LAGS:
        f[f"d{k}"] = logp - logp.shift(k)          # change over last k days
    f["vol7"] = logp.diff().rolling(7).std()
    f["dev30"] = logp - logp.rolling(30).mean()     # distance from 30d mean
    doy = logp.index.dayofyear
    f["sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def _fit(logp: pd.Series):
    X = _features(logp)[USE]
    y = logp.shift(-1) - logp                       # next-day log return
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    model = make_pipeline(StandardScaler(), Ridge(alpha=500.0))
    model.fit(data.drop(columns="y"), data["y"])
    resid = data["y"] - model.predict(data.drop(columns="y"))
    sigma = float(resid.std())
    model.step_cap_ = 2.0 * sigma                   # never move more than 2 sigma/day
    return model, sigma


def _roll(model, logp: pd.Series, horizon: int) -> np.ndarray:
    """Recursive multi-day forecast of log prices."""
    hist = logp.copy()
    out = []
    for _ in range(horizon):
        nxt_date = hist.index[-1] + pd.Timedelta(days=1)
        ext = pd.concat([hist, pd.Series([hist.iloc[-1]], index=[nxt_date])])
        x = _features(ext)[USE].iloc[[-2]]               # features known at last day
        step = float(np.clip(model.predict(x)[0], -model.step_cap_, model.step_cap_))
        hist = pd.concat([hist, pd.Series([hist.iloc[-1] + step], index=[nxt_date])])
        out.append(hist.iloc[-1])
    return np.array(out)


@dataclass
class Result:
    history: pd.Series
    forecast: pd.DataFrame      # columns: price, low, high (index = future dates)
    mae: float                  # model, Rs/quintal, on holdout
    mape: float                 # model, %
    naive_mae: float
    naive_mape: float


def run(series: pd.Series, horizon: int = 14, holdout: int = 14) -> Result:
    logp = np.log(series)
    # --- honest backtest: 5 past start dates, train only on data before each,
    #     forecast the next `holdout` days and compare with what really happened
    errs, n_errs, n_pct, m_pct = [], [], [], []
    for off in (30, 60, 90, 120, 150):
        if len(series) - off < 120:
            continue
        train, test = logp.iloc[:-off], series.iloc[-off:-off + holdout]
        m_bt, _ = _fit(train)
        pred = np.exp(_roll(m_bt, train, holdout))
        errs.append(np.mean(np.abs(pred - test.values)))
        m_pct.append(np.mean(np.abs(pred - test.values) / test.values) * 100)
        naive = np.full(holdout, series.iloc[-off - 1])
        n_errs.append(np.mean(np.abs(naive - test.values)))
        n_pct.append(np.mean(np.abs(naive - test.values) / test.values) * 100)
    mae, mape = float(np.mean(errs)), float(np.mean(m_pct))
    n_mae, n_mape = float(np.mean(n_errs)), float(np.mean(n_pct))

    # --- final model on all data
    model, sigma = _fit(logp)
    path = _roll(model, logp, horizon)
    steps = np.arange(1, horizon + 1)
    band = 1.28 * sigma * np.sqrt(steps)            # ~80% interval
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fc = pd.DataFrame({"price": np.exp(path),
                       "low": np.exp(path - band),
                       "high": np.exp(path + band)}, index=idx).round(0)
    return Result(series, fc, mae, mape, n_mae, n_mape)


def advise(current: float, fc: pd.DataFrame, threshold: float = 0.04):
    """Return (decision, best_day, expected_gain_pct) using the next 7 days."""
    window = fc.iloc[:7]
    best_day = window["price"].idxmax()
    gain = (window["price"].max() - current) / current
    # only advise HOLD if even the pessimistic band is not far below today
    risk_ok = window.loc[best_day, "low"] > current * 0.93
    if gain >= threshold and risk_ok:
        return "HOLD", best_day, gain * 100
    if gain >= threshold:
        return "UNSURE", best_day, gain * 100
    return "SELL", fc.index[0], gain * 100