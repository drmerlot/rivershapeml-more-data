"""Bankfull width and depth targets for every site, built the same way for all sites.

For each site in retrain_sites.csv (with a COMID):
  q2      : 2-year flood = median of USGS annual peak discharges (>= 10 peaks). On the annual-maximum
            basis the 2-year flood is the median annual flood; the sample median estimates it.
  tw_all  : bankfull width at q2 from all USGS channel measurements since 2000
  td_all  : bankfull mean depth (area / width) at q2, same measurements
  tw_hs   : bankfull width at q2 from the site's HYDRoSWOT records only (2000-2015): the status-quo data
  td_hs   : bankfull mean depth at q2 from the USGS records that match a HYDRoSWOT record (same visit: within
            1 day, Q within 3%); HYDRoSWOT itself has only maximum depth, so mean depth (area / width) for
            those same visits comes from the USGS collection
            (no depth version: RiverShapeML's HYDRoSWOT copy has no cross-section area)

Fitting (--fit ahg, default): AHGestimation (Johnson et al., NOAA OWP), the package RiverShapeML used, via
src/ahg_fit.R: mad_filter(), qva_filter() (continuity), then ahg_estimate(), which fits width, depth and
velocity jointly so they conserve mass (V = Q / area). --fit ols: the earlier independent log-log fits.

RiverShapeML's sanity rules, applied before fitting: positive values; width < 4 km; depth < 65 m;
measured flow no larger than 1.1x the gage's largest recorded annual peak. Sites whose fitted depth
decreases with flow (depth exponent < 0) are dropped, as in the paper.

Site selection (both fits): >= 5 measurements after filtering, standard error of the log target at q2
<= 0.2 (~ +/-22%) from the residuals around the fitted curve, and q2 <= 3x the largest measured flow.
This replaces the paper's R² >= 0.6 rule (R² is recorded for comparison as r2_*).

Units: measurements are converted to SI for fitting; target columns are written in feet (tw_*, td_*)
and q2 in ft3/s, as downstream scripts expect.

Usage: python src/build_targets.py [--fit ahg|ols] [--max-se 0.2] [--max-ext 3] [--out CSV] [--reuse-fits]
Output: data/derived/retrain_targets.csv (ahg) or retrain_targets_ols.csv (ols), unless --out is given
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data/raw", ROOT / "data/derived"
MIN_MEAS, MIN_PEAKS = 5, 10
MAX_SE = 0.2   # max standard error of the log target at q2
MAX_EXT = 3.0  # max q2 / largest measured flow
FT, CFS = 0.3048, 0.0283168  # m per ft, m3/s per ft3/s


def pages(subdir, columns=None):
    return pd.concat((pd.read_parquet(p, columns=columns) for p in sorted((RAW / subdir).glob("page_*.parquet"))),
                     ignore_index=True)


def curve_stats(q, v, coef, exp, q_eval):
    """Target at q_eval from the curve v = coef * q^exp, plus its precision. Residuals around the curve are
    trimmed at 3 robust SD; SE of the log target at q_eval uses the linear-regression formula in log space.
    Returns (target or NaN if not precise enough, se, r2, n, q_eval / max q)."""
    lq, lv = np.log(q), np.log(v)
    res = lv - (np.log(coef) + exp * lq)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    keep = np.abs(res - np.median(res)) <= 3 * mad if mad > 0 else np.ones(len(lq), bool)
    lq, lv, res = lq[keep], lv[keep], res[keep]
    n = len(lq)
    if n < MIN_MEAS or np.ptp(lq) == 0:
        return np.nan, np.nan, np.nan, n, np.nan
    s = np.sqrt(np.sum(res ** 2) / (n - 2))
    x0 = np.log(q_eval)
    se = s * np.sqrt(1 / n + (x0 - lq.mean()) ** 2 / np.sum((lq - lq.mean()) ** 2))
    sst = np.sum((lv - lv.mean()) ** 2)
    r2 = 1 - np.sum(res ** 2) / sst if sst > 0 else np.nan
    ext = q_eval / np.exp(lq.max())
    ok = (se <= MAX_SE) and (ext <= MAX_EXT)
    return (coef * q_eval ** exp if ok else np.nan), se, r2, n, ext


def ols_curve(q, v):
    """Independent log-log least squares after one pass of 3-MAD trimming (the --fit ols method)."""
    lq, lv = np.log(q), np.log(v)
    if len(lq) < MIN_MEAS or np.ptp(lq) == 0:
        return np.nan, np.nan
    b, a = np.polyfit(lq, lv, 1)
    res = lv - (a + b * lq)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    keep = np.abs(res - np.median(res)) <= 3 * mad if mad > 0 else np.ones(len(lq), bool)
    if keep.sum() < MIN_MEAS or np.ptp(lq[keep]) == 0:
        return np.nan, np.nan
    b, a = np.polyfit(lq[keep], lv[keep], 1)
    return np.exp(a), b


def load_measurements(sites, max_peak_cms):
    """All USGS measurements since 2000 and HYDRoSWOT records 2000-2015, in SI, with the sanity rules."""
    ids = set("USGS-" + sites.site_no)
    cm = pages("channel_meas", ["monitoring_location_id", "time", "channel_flow", "channel_width", "channel_area",
                                "measurement_type"])
    cm = cm[cm.monitoring_location_id.isin(ids)]
    for c in ["channel_flow", "channel_width", "channel_area"]:
        cm[c] = pd.to_numeric(cm[c], errors="coerce")
    cm = cm[(cm.measurement_type != "Ice") & (cm.time.str[:4] >= "2000")]
    cm = pd.DataFrame({"site_no": cm.monitoring_location_id.str[5:], "set": "all",
                       "date": pd.to_datetime(cm.time, format="ISO8601", utc=True).dt.tz_localize(None).dt.normalize(),
                       "Q": cm.channel_flow * CFS, "TW": cm.channel_width * FT,
                       "Y": cm.channel_area / cm.channel_width * FT,
                       "V": cm.channel_flow / cm.channel_area * FT})  # (ft3/s)/(ft2) = ft/s -> m/s

    hs = pd.read_parquet(RAW / "repo/RiverShapeML/data/adcp.parquet")
    hs = hs[hs.site_no.isin(sites.site_no) & pd.to_datetime(hs.site_visit_start_dt).between("2000-01-01", "2015-12-31")]
    hs = pd.DataFrame({"site_no": hs.site_no, "date": pd.to_datetime(hs.site_visit_start_dt).dt.normalize(),
                       "Q": hs.meas_q_va * CFS, "TW": hs.stream_wdth_va * FT})

    # USGS records matching a HYDRoSWOT record (same visit), for the HYDRoSWOT-only depth target
    m = hs.reset_index(names="hs_id").merge(cm.reset_index(names="cm_id")[["site_no", "cm_id", "date", "Q"]],
                                            on="site_no", suffixes=("", "_cm"))
    m = m[((m.date - m.date_cm).abs() <= pd.Timedelta(days=1)) & ((m.Q_cm - m.Q).abs() <= 0.03 * m.Q)]
    m = m.assign(dq=(m.Q_cm - m.Q).abs()).sort_values("dq").drop_duplicates("hs_id").drop_duplicates("cm_id")
    hsm = cm.loc[m.cm_id].assign(set="hsm")
    print(f"HYDRoSWOT records matched to a USGS record: {len(m):,} of {len(hs):,}", flush=True)

    hs = hs.assign(set="hs", Y=np.nan, V=np.nan)
    d = pd.concat([cm, hs, hsm], ignore_index=True).drop(columns="date")
    n0 = len(d)
    ok = (d.Q > 0) & (d.TW > 0) & (d.TW < 4000)
    ok &= d.Y.isna() | ((d.Y > 0) & (d.Y < 65) & (d.V > 0))
    ok &= d.Q <= 1.1 * d.site_no.map(max_peak_cms)
    d = d[ok.fillna(False)]
    print(f"measurements: {n0:,} -> {len(d):,} after sanity rules "
          f"(all: {(d.set == 'all').sum():,}, HYDRoSWOT: {(d.set == 'hs').sum():,})", flush=True)
    return d


def main(fit="ahg", max_se=MAX_SE, max_ext=MAX_EXT, out=None, reuse_fits=False):
    global MAX_SE, MAX_EXT
    MAX_SE, MAX_EXT = float(max_se), float(max_ext)
    out = Path(out) if out else DER / ("retrain_targets.csv" if fit == "ahg" else "retrain_targets_ols.csv")
    sites = pd.read_csv(DER / "retrain_sites.csv", dtype={"site_no": str}).dropna(subset=["comid"])
    sites = sites.drop_duplicates("site_no")

    pk = pages("peaks")
    pk = pk[pk.monitoring_location_id.isin(set("USGS-" + sites.site_no))]
    pk["value"] = pd.to_numeric(pk.value, errors="coerce")
    pk = pk[pk.value > 0]
    peaks = pk.groupby(pk.monitoring_location_id.str[5:]).value.agg(q2="median", n_peaks="size", max_peak="max")
    peaks.loc[peaks.n_peaks < MIN_PEAKS, "q2"] = np.nan

    d = load_measurements(sites, peaks.max_peak * CFS)
    d = d[d.site_no.isin(peaks.index[peaks.q2.notna()])]

    if fit == "ahg":
        fits_path = DER / "ahg_fits.parquet"
        if not (reuse_fits and fits_path.exists()):
            d.to_parquet(DER / "ahg_input.parquet", index=False)
            env = {**os.environ, "R_LIBS_USER": str(ROOT / "r_lib")}
            subprocess.run(["Rscript", str(ROOT / "src/ahg_fit.R")], check=True, env=env)
        fits = pd.read_parquet(fits_path).set_index(["site_no", "set"])
    by = dict(tuple(d.groupby(["site_no", "set"])))

    rows = []
    for s in sites.itertuples():
        q2 = peaks.q2.get(s.site_no, np.nan)
        row = {"site_no": s.site_no, "group": s.group, "comid": int(s.comid), "q2": q2,
               "n_peaks": peaks.n_peaks.get(s.site_no, 0)}
        if np.isfinite(q2):
            q2_si = q2 * CFS
            for st, specs in [("all", [("TW", "tw_all", ("se_all", "r2_all", "n_all", "ext_all")),
                                       ("Y", "td_all", ("se_d_all", "r2_d_all", "n_d_all", "ext_d_all"))]),
                              ("hs", [("TW", "tw_hs", ("se_hs", "r2_hs", "n_hs", "ext_hs"))]),
                              ("hsm", [("Y", "td_hs", ("se_d_hs", "r2_d_hs", "n_d_hs", "ext_d_hs"))])]:
                g = by.get((s.site_no, st))
                if g is None:
                    continue
                if fit == "ahg":
                    if (s.site_no, st) not in fits.index or pd.notna(fits.loc[(s.site_no, st), "error"]):
                        continue
                    f = fits.loc[(s.site_no, st)]
                    coefs = {"TW": (f.TW_coef, f.TW_exp), "Y": (f.Y_coef, f.Y_exp)}
                    row[f"viable_{st}"] = f.viable
                else:
                    coefs = {v: ols_curve(g.Q.values, g[v].values) for v in ["TW", "Y"] if g[v].notna().any()}
                if st in ("all", "hsm") and "Y" in coefs and coefs["Y"][1] < 0:
                    row["dropped_inverse_depth" if st == "all" else "dropped_inverse_depth_hs"] = True  # depth falls with flow: drop
                    continue
                for v, col, (c_se, c_r2, c_n, c_ext) in specs:
                    c, e = coefs.get(v, (np.nan, np.nan))
                    if not (np.isfinite(c) and np.isfinite(e) and c > 0):
                        continue
                    gv = g[g[v] > 0]
                    target, row[c_se], row[c_r2], row[c_n], row[c_ext] = curve_stats(gv.Q.values, gv[v].values, c, e, q2_si)
                    row[col] = target / FT  # m -> ft
                    row[f"{col}_exp"] = e
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)

    print(f"fit: {fit} | filter: se <= {MAX_SE}, q2/maxQ <= {MAX_EXT} | wrote {out.name}")
    print(f"sites with COMID: {len(t)}  | with q2: {t.q2.notna().sum()} | dropped (depth falls with flow): "
          f"{int(t.get('dropped_inverse_depth', pd.Series(dtype=bool)).fillna(False).sum())}")
    print(t.groupby("group").agg(tw_all=("tw_all", "count"), tw_hs=("tw_hs", "count"), td_all=("td_all", "count"),
                                 td_hs=("td_hs", "count")))
    both = t.dropna(subset=["tw_all", "tw_hs"])
    print(f"existing sites with both width targets: {len(both)}; tw_all/tw_hs median {np.median(both.tw_all / both.tw_hs):.3f}")
    print("extrapolation q2/maxQ, median: all", round(t.ext_all.median(), 2), "| hs", round(t.ext_hs.median(), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=["ahg", "ols"], default="ahg")
    ap.add_argument("--max-se", type=float, default=MAX_SE)
    ap.add_argument("--max-ext", type=float, default=MAX_EXT)
    ap.add_argument("--out")
    ap.add_argument("--reuse-fits", action="store_true", help="use existing data/derived/ahg_fits.parquet")
    a = ap.parse_args()
    main(a.fit, a.max_se, a.max_ext, a.out, a.reuse_fits)
