"""Bankfull-width targets for the retrain experiment, built the same way for every site.

For each site in retrain_sites.csv (with a COMID):
  q2      : 2-year flood = median of USGS annual peak discharges (>= 10 peaks)
  tw_all  : width at q2 from a log-log fit to ALL USGS width measurements since 2000
  tw_hs   : same, using only the site's HYDRoSWOT records (2000-2015), i.e. the status-quo data
Each fit needs >= 5 measurements (after trimming gross outliers), a standard error of log-width at q2
<= 0.2, and q2 <= 3x the largest measured flow; otherwise the target is NaN. This replaces the paper's
R² >= 0.6 filter: R² is low for confined channels whose width barely changes with flow, even when
their width is precisely known. ext_* = q2 / largest measured flow.

Simplifications vs RiverShapeML: median annual peak instead of a fitted GEV/GP/LP3 distribution, and
plain least squares instead of AHGestimation.

Usage: python src/build_targets.py [max_se] [max_ext] [out_csv]   (defaults 0.2, 3.0, data/derived/retrain_targets.csv)
Output: data/derived/retrain_targets.csv
"""
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data/raw", ROOT / "data/derived"
MIN_MEAS, MIN_PEAKS = 5, 10
MAX_SE = 0.2   # max standard error of log-width at q2 (~ +/-20%)
MAX_EXT = 3.0  # max q2 / largest measured flow


def pages(subdir, columns=None):
    return pd.concat((pd.read_parquet(p, columns=columns) for p in sorted((RAW / subdir).glob("page_*.parquet"))),
                     ignore_index=True)


def fit_at(q, w, q_eval):
    """log-log least squares after one pass of outlier trimming (|resid| > 3 robust sd).
    Returns width at q_eval (NaN unless precise enough), standard error of log-width at q_eval,
    r2, n used, and q_eval / max measured q."""
    lq, lw = np.log(q), np.log(w)
    if len(lq) >= MIN_MEAS and np.ptp(lq) > 0:
        b, a = np.polyfit(lq, lw, 1)
        res = lw - (a + b * lq)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        keep = np.abs(res - np.median(res)) <= 3 * mad if mad > 0 else np.ones(len(lq), bool)
        lq, lw = lq[keep], lw[keep]
    n = len(lq)
    if n < MIN_MEAS or np.ptp(lq) == 0:
        return np.nan, np.nan, np.nan, n, np.nan
    b, a = np.polyfit(lq, lw, 1)
    res = lw - (a + b * lq)
    s = np.sqrt(np.sum(res ** 2) / (n - 2))
    x0 = np.log(q_eval)
    se = s * np.sqrt(1 / n + (x0 - lq.mean()) ** 2 / np.sum((lq - lq.mean()) ** 2))
    sst = np.sum((lw - lw.mean()) ** 2)
    r2 = 1 - np.sum(res ** 2) / sst if sst > 0 else np.nan
    ext = q_eval / np.exp(lq.max())
    ok = (se <= MAX_SE) and (ext <= MAX_EXT)
    return (np.exp(a + b * x0) if ok else np.nan), se, r2, n, ext


def main(max_se=None, max_ext=None, out=None):
    global MAX_SE, MAX_EXT
    MAX_SE, MAX_EXT = float(max_se or MAX_SE), float(max_ext or MAX_EXT)
    sites = pd.read_csv(DER / "retrain_sites.csv", dtype={"site_no": str}).dropna(subset=["comid"])
    sites = sites.drop_duplicates("site_no")
    ids = set("USGS-" + sites.site_no)

    pk = pages("peaks")
    pk = pk[pk.monitoring_location_id.isin(ids)]
    pk["value"] = pd.to_numeric(pk.value, errors="coerce")
    pk = pk[pk.value > 0]
    peaks = pk.groupby(pk.monitoring_location_id.str[5:]).value.agg(q2="median", n_peaks="size")
    peaks.loc[peaks.n_peaks < MIN_PEAKS, "q2"] = np.nan

    cm = pages("channel_meas", ["monitoring_location_id", "time", "channel_flow", "channel_width", "measurement_type"])
    cm = cm[cm.monitoring_location_id.isin(ids)]
    for c in ["channel_flow", "channel_width"]:
        cm[c] = pd.to_numeric(cm[c], errors="coerce")
    cm = cm[(cm.channel_flow > 0) & (cm.channel_width > 0) & (cm.measurement_type != "Ice") & (cm.time.str[:4] >= "2000")]
    cm["site_no"] = cm.monitoring_location_id.str[5:]
    by_site = dict(tuple(cm.groupby("site_no")))

    hs = pd.read_parquet(RAW / "repo/RiverShapeML/data/adcp.parquet")
    hs = hs[(hs.meas_q_va > 0) & (hs.stream_wdth_va > 0) & hs.site_no.isin(sites.site_no)]
    hs = hs[pd.to_datetime(hs.site_visit_start_dt).between("2000-01-01", "2015-12-31")]
    hs_by_site = dict(tuple(hs.groupby("site_no")))

    rows = []
    for s in sites.itertuples():
        q2 = peaks.q2.get(s.site_no, np.nan)
        row = {"site_no": s.site_no, "group": s.group, "comid": int(s.comid), "q2": q2,
               "n_peaks": peaks.n_peaks.get(s.site_no, 0)}
        if np.isfinite(q2):
            d = by_site.get(s.site_no)
            if d is not None:
                row["tw_all"], row["se_all"], row["r2_all"], row["n_all"], row["ext_all"] = fit_at(d.channel_flow.values, d.channel_width.values, q2)
            h = hs_by_site.get(s.site_no)
            if h is not None:
                row["tw_hs"], row["se_hs"], row["r2_hs"], row["n_hs"], row["ext_hs"] = fit_at(h.meas_q_va.values, h.stream_wdth_va.values, q2)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out or DER / "retrain_targets.csv", index=False)
    print(f"filter: se <= {MAX_SE}, q2/maxQ <= {MAX_EXT}")

    print(f"sites with COMID: {len(t)}  | with q2: {t.q2.notna().sum()}")
    print(t.groupby("group").agg(tw_all=("tw_all", "count"), tw_hs=("tw_hs", "count")))
    both = t.dropna(subset=["tw_all", "tw_hs"])
    print(f"existing sites with both targets: {len(both)}; tw_all/tw_hs median {np.median(both.tw_all / both.tw_hs):.3f}, "
          f"IQR [{(both.tw_all / both.tw_hs).quantile(.25):.2f}, {(both.tw_all / both.tw_hs).quantile(.75):.2f}]")
    print("extrapolation q2/maxQ, median: all", round(t.ext_all.median(), 2), "| hs", round(t.ext_hs.median(), 2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
