"""Error measures for width/depth predictions, shared by retrain_experiment.py and the notebook.

Every prediction is compared to its target as a ratio r = predicted / target (log error e = ln r).
  median_abs_err_%   median |r - 1|                       "half the predictions are closer than this"
  within_25_%        share with 0.75 <= r <= 1.25
  within_2x_%        share with 0.5 <= r <= 2
  bias_%             median(r) - 1                        systematic over (+) / under (-) prediction
  log_r2             1 - sum(e^2) / sum((ln target - mean)^2)   the paper's metric
  rms_factor_%       exp(sqrt(mean(e^2))) - 1             old "typical error"; sensitive to outliers
"""
import numpy as np
import pandas as pd

METRICS = {
    "median_abs_err_%": "Median absolute error (%)",
    "within_25_%": "Within ±25% (% of sites)",
    "within_2x_%": "Within a factor of 2 (% of sites)",
    "bias_%": "Median bias (%)",
    "log_r2": "log R²",
    "rms_factor_%": "RMS error factor (%)",
}


def error_metrics(log_err, log_true):
    e = np.asarray(log_err, float)
    t = np.asarray(log_true, float)
    ok = np.isfinite(e) & np.isfinite(t)
    e, t = e[ok], t[ok]
    r = np.exp(e)
    sst = np.sum((t - t.mean()) ** 2)
    return pd.Series({
        "sites": len(e),
        "median_abs_err_%": 100 * np.median(np.abs(r - 1)),
        "within_25_%": 100 * np.mean((r >= 0.75) & (r <= 1.25)),
        "within_2x_%": 100 * np.mean((r >= 0.5) & (r <= 2)),
        "bias_%": 100 * (np.median(r) - 1),
        "log_r2": 1 - np.sum(e ** 2) / sst if sst > 0 else np.nan,
        "rms_factor_%": 100 * (np.exp(np.sqrt(np.mean(e ** 2))) - 1),
    })


def summarize(preds, by, err="log_err", true="log_true"):
    """Metrics per group and variant, computed per seed and then averaged across seeds."""
    per_seed = (preds.groupby(by + ["variant", "seed"], observed=True)
                .apply(lambda g: error_metrics(g[err], g[true]), include_groups=False))
    return per_seed.groupby(by + ["variant"], observed=True).mean().round(3).reset_index()
