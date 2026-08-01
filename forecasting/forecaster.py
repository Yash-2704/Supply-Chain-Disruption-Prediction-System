"""Modeling logic: Prophet (primary, with intervals) + Holt-Winters (sanity check).

Both fitters raise ForecastError on failure so the orchestrator can record a
'model_error' and continue. Horizons are expressed in WEEKS (series are weekly).
"""
import contextlib
import io
import logging
import os
import threading

import pandas as pd

logger = logging.getLogger(__name__)

# Keep Prophet/Stan chatter out of our logs.
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

# Horizons in WEEKS (series are resampled weekly). Named once, derived here.
HORIZONS = {"2w": 2, "1m": 4, "3m": 13, "6m": 26}

# Minimum weekly points to forecast. ~12 weeks (~3 months) is a defensible floor
# for a short-horizon, non-seasonal trend: enough for Prophet to estimate a trend
# with changepoints without fabricating structure. Below this -> insufficient_data.
MIN_DATA_POINTS = 12

# If Prophet and Holt-Winters point forecasts differ by more than this at a
# horizon, the trend is model-sensitive -> flag that horizon 'unstable_disagreement'.
DISAGREEMENT_THRESHOLD_PCT = 25.0

_MAX_STEPS = max(HORIZONS.values())


class ForecastError(Exception):
    """Raised when a model fails to fit/predict on a given series."""


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Silence Prophet/Stan console spew during fit."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def run_prophet(series_df: pd.DataFrame, horizons: dict = HORIZONS) -> dict:
    """Fit Prophet; return {horizon_key: {'point','lower','upper'}}. Raises
    ForecastError on any failure."""
    try:
        from prophet import Prophet
        df = series_df.rename(columns={"ds": "ds", "y": "y"})[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        n = len(df)
        # Short, weekly-aggregated series: trend only, no sub-weekly/yearly seasonality.
        model = Prophet(weekly_seasonality=False, daily_seasonality=False,
                        yearly_seasonality=False, interval_width=0.80)
        with _suppress_stdout_stderr():
            model.fit(df)
            future = model.make_future_dataframe(periods=_MAX_STEPS, freq="W")
            fc = model.predict(future)
        out = {}
        for key, weeks in horizons.items():
            row = fc.iloc[n + weeks - 1]  # future rows start at index n (= +1 week)
            out[key] = {"point": float(row["yhat"]),
                        "lower": float(row["yhat_lower"]),
                        "upper": float(row["yhat_upper"])}
        return out
    except Exception as exc:
        raise ForecastError(f"Prophet failed: {exc}") from exc


def run_holt_winters(series_df: pd.DataFrame, horizons: dict = HORIZONS) -> dict:
    """Fit Holt's linear-trend exponential smoothing; return {horizon_key: point}.
    Point estimate only (no calibrated interval). Raises ForecastError on failure."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = pd.to_numeric(series_df["y"], errors="coerce").dropna().reset_index(drop=True)
        if len(y) < 2:
            raise ValueError("need >= 2 points for Holt-Winters")
        model = ExponentialSmoothing(y, trend="add", seasonal=None,
                                     initialization_method="estimated")
        with _suppress_stdout_stderr():
            fit = model.fit()
            fc = fit.forecast(_MAX_STEPS)
        fc = fc.reset_index(drop=True)
        return {key: float(fc.iloc[weeks - 1]) for key, weeks in horizons.items()}
    except Exception as exc:
        raise ForecastError(f"Holt-Winters failed: {exc}") from exc


def compute_disagreement(prophet_point: float, hw_point: float) -> float:
    """Relative % difference between the two point forecasts (0 = identical)."""
    denom = max(abs(prophet_point), abs(hw_point), 1e-9)
    return abs(prophet_point - hw_point) / denom * 100.0


# --- Chronos: pretrained zero-shot forecaster (Apache-2.0, CPU-capable) ------
# Added as an OPTIONAL third model. chronos-bolt-small (~48M params) is a small,
# free, commercially-usable download; it needs no training and handles the short
# ~weekly series gracefully. It is loaded lazily and cached, and — unlike Prophet/
# Holt-Winters — a Chronos failure is non-fatal to a forecast run (the orchestrator
# treats it as "absent" and still records Prophet+HW). Its point is stored for the
# Phase-3 three-way benchmark; disagreement_pct stays Prophet-vs-HW for continuity.
CHRONOS_MODEL = os.environ.get("CHRONOS_MODEL", "amazon/chronos-bolt-small")

_chronos_pipe = None
_chronos_lock = threading.Lock()
_chronos_failed = False


def _get_chronos_pipeline():
    """Lazily load + cache the Chronos pipeline once. Returns None (logged once)
    if chronos/torch or the weights are unavailable."""
    global _chronos_pipe, _chronos_failed
    if _chronos_pipe is not None or _chronos_failed:
        return _chronos_pipe
    with _chronos_lock:
        if _chronos_pipe is not None or _chronos_failed:
            return _chronos_pipe
        try:
            from chronos import BaseChronosPipeline
            _chronos_pipe = BaseChronosPipeline.from_pretrained(
                CHRONOS_MODEL, device_map="cpu")
        except Exception as exc:
            _chronos_failed = True
            logger.warning("Chronos unavailable (%s) — third forecaster skipped.", exc)
            return None
    return _chronos_pipe


def chronos_available() -> bool:
    return _get_chronos_pipeline() is not None


def run_chronos(series_df: pd.DataFrame, horizons: dict = HORIZONS) -> dict:
    """Zero-shot Chronos forecast; return {horizon_key: point} using the median
    (0.5) quantile at each horizon. Raises ForecastError on any failure (the
    orchestrator catches it and proceeds without Chronos)."""
    try:
        import torch
        pipe = _get_chronos_pipeline()
        if pipe is None:
            raise RuntimeError("Chronos pipeline could not be loaded")
        y = pd.to_numeric(series_df["y"], errors="coerce").dropna().tolist()
        if len(y) < 2:
            raise ValueError("need >= 2 points for Chronos")
        context = torch.tensor(y, dtype=torch.float32)
        # predict_quantiles(inputs, ...) -> (quantiles[B, H, Q], mean[B, H]);
        # request the 0.5 (median) quantile explicitly and read it back.
        quantiles, _mean = pipe.predict_quantiles(
            context, prediction_length=_MAX_STEPS, quantile_levels=[0.1, 0.5, 0.9])
        median = quantiles[0, :, 1]  # batch 0, all horizons, quantile index 1 = 0.5
        return {key: float(median[weeks - 1]) for key, weeks in horizons.items()}
    except Exception as exc:
        raise ForecastError(f"Chronos failed: {exc}") from exc
