"""Noise floor: how well can the measurements themselves pin down a site's target (y)?

Each collection of measurements at a site is treated as an equally valid look at the reach:
  H = measurements in HYDRoSWOT, N = USGS measurements not in HYDRoSWOT (2000+).
For each collection with >= 5 measurements, fit log(W) vs log(Q) (and depth = area/width) and
evaluate at the site's in-channel and bankfull flows. The disagreement d = ln(y_N / y_H) measures
target uncertainty: if each estimate has log-error sigma, sd(d) = sqrt(2) * sigma.

Compared against RiverShapeML's held-out error implied by the paper's log R²:
  sigma_model = sqrt(1 - R²) * sd(ln y) across training targets.
(The published values at gaged reaches equal the targets, so in-sample error can't be read off.)

Usage: python src/noise_floor.py   (uses sites cached by method_agreement.py)
Output: data/derived/noise_floor_sites.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from method_agreement import CACHE, DER, MIN_FIT, REPO, clean, label_hydroswot

PAPER_TEST_R2 = {("w", "bf"): 0.76, ("w", "in"): 0.66, ("d", "bf"): 0.80, ("d", "in"): 0.77}
COVER = 1.5  # a collection "covers" a flow if it lies within [min/1.5, max*1.5] of its measured flows


def y_at(d, col, qs):
    b, a = np.polyfit(np.log(d.channel_flow), np.log(d[col]), 1)
    lo, hi = d.channel_flow.min() / COVER, d.channel_flow.max() * COVER
    return np.exp(a + b * np.log(qs)), (qs >= lo) & (qs <= hi)


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)


def pct(s):
    return 100 * (np.exp(s) - 1)


def main():
    tw = pd.read_parquet(REPO / "nwis_width_pred_tar_up.parquet").assign(siteID=lambda x: x.siteID.astype(str))
    td = pd.read_parquet(REPO / "nwis_depth_pred_tar_up.parquet")
    tgt = tw.set_index("siteID")
    hs = pd.read_parquet(REPO / "adcp.parquet")
    hs = hs[(hs.meas_q_va > 0) & (hs.stream_wdth_va > 0)].copy()
    hs["date"] = pd.to_datetime(hs.site_visit_start_dt).dt.normalize()

    rows = []
    for p in sorted(CACHE.glob("*.parquet")):
        site = p.stem
        if site not in tgt.index:
            continue
        us = clean(pd.read_parquet(p))
        h = hs[hs.site_no == site]
        if us.empty or h.empty:
            continue
        us = label_hydroswot(us, h)
        qs = np.array([tgt.at[site, "in_ff"], tgt.at[site, "bf_ff"]])
        row = {"site_no": site}
        for col, tag in [("channel_width", "w"), ("depth", "d")]:
            d = us[us[col] > 0]
            H, N = d[d.in_hs], d[~d.in_hs]
            if len(H) < MIN_FIT or len(N) < MIN_FIT:
                continue
            yh, ch = y_at(H, col, qs)
            yn, cn = y_at(N, col, qs)
            for k, cond in enumerate(["in", "bf"]):
                row[f"{tag}_{cond}_logdiff"] = np.log(yn[k] / yh[k])
                row[f"{tag}_{cond}_both_cover"] = bool(ch[k] and cn[k])
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(DER / "noise_floor_sites.csv", index=False)

    print(f"sites: {len(res)}\n")
    print(f"{'target':<22}{'n':>5}{'noise floor':>13}{'n covered':>11}{'floor covered':>15}{'model error':>13}")
    for tag, name, t, col in [("w", "width", tw, "TW"), ("d", "depth", td, "Y")]:
        t = t[t.R2 > 0.6]
        for cond, label in [("in", "in-channel"), ("bf", "bankfull")]:
            dcol = f"{tag}_{cond}_logdiff"
            if dcol not in res:
                continue
            dd = res[dcol].dropna()
            cov = res.loc[res[f"{tag}_{cond}_both_cover"].fillna(False).astype(bool), dcol]
            sig_model = np.sqrt(1 - PAPER_TEST_R2[(tag, cond)]) * np.log(t[f"{col}_{cond}"]).std()
            print(f"{label + ' ' + name:<22}{len(dd):>5}{pct(robust_sigma(dd)):>12.0f}%{len(cov):>11}"
                  f"{pct(robust_sigma(cov)) if len(cov) > 5 else float('nan'):>14.0f}%{pct(sig_model):>12.0f}%")
    print("\nnoise floor = typical error of one collection's y (robust, from H vs N disagreement)")
    print("model error = RiverShapeML held-out error implied by the paper's log R² (typical, 1 sd)")


if __name__ == "__main__":
    main()
