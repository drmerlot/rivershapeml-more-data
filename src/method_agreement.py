"""Steps 3-4: are USGS measurements that are NOT in HYDRoSWOT consistent with the ones that are,
and does adding them make the training targets (y) better determined?

Neither set is treated as truth. For a random sample of RiverShapeML training sites:
  1. Pull every USGS channel measurement for the site (cached per site).
  2. Match HYDRoSWOT records to USGS records (within 1 day, discharge within 3%) to split the
     USGS measurements into H (in HYDRoSWOT) and N (not in HYDRoSWOT).
  3. Consistency, symmetric: fit log-log width-vs-flow (and depth = area/width) to each set and
     compare the two curves at the flows where both sets have data (overlap range).
  4. Target certainty: bootstrap the fit with H only and with H+N, and evaluate at the site's
     in-channel and bankfull flows from the training targets. Report how far y moves and how
     much its 90% interval narrows.

Simplification: plain log-log least squares, not the paper's AHGestimation ensemble with
mass conservation, so absolute y values differ from the published targets.

Usage: python src/method_agreement.py [n_sites] [seed] [--cached]  (--cached: only already-downloaded sites)
Outputs: data/derived/method_agreement_sites.csv, method_agreement_types.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from keys import api_key

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/repo/RiverShapeML/data"
CACHE = ROOT / "data/raw/channel_meas_by_site"
DER = ROOT / "data/derived"
API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/channel-measurements/items"
PROPS = ("monitoring_location_id,time,channel_flow,channel_width,channel_area,measurement_type,channel_name,"
         "channel_location_distance,channel_location_direction,channel_material,channel_stability")
MIN_FIT = 5  # paper's minimum measurements per site
N_BOOT = 300
RNG = np.random.default_rng(0)


def fetch(site, session):
    path = CACHE / f"{site}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = f"{API}?f=json&limit=10000&skipGeometry=true&properties={PROPS}&monitoring_location_id=USGS-{site}"
    for attempt in range(5):
        r = session.get(url, timeout=120)
        if r.status_code == 200:
            break
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    df = pd.DataFrame([f["properties"] for f in r.json()["features"]])
    df.to_parquet(path)
    time.sleep(1)
    return df


def clean(df):
    if df.empty:
        return df
    df = df.copy()
    for c in ["channel_flow", "channel_width", "channel_area"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.channel_flow > 0) & (df.channel_width > 0)]
    df["date"] = pd.to_datetime(df.time, utc=True, format="ISO8601").dt.tz_convert(None).dt.normalize()
    df["depth"] = df.channel_area / df.channel_width
    return df[(df.date >= "2000-01-01") & (df.measurement_type != "Ice")]


def label_hydroswot(us, hs):
    """Mark USGS rows that correspond to a HYDRoSWOT record (within 1 day, since USGS times are
    UTC and HYDRoSWOT's are local; Q within 3%)."""
    us = us.copy()
    us["in_hs"] = False
    for _, h in hs.iterrows():
        m = ((us.date - h.date).abs() <= pd.Timedelta(days=1)) & \
            ((us.channel_flow - h.meas_q_va).abs() <= 0.03 * h.meas_q_va)
        if m.any():
            us.loc[(us.channel_flow - h.meas_q_va).abs().where(m).idxmin(), "in_hs"] = True
    return us


def fit(q, v):
    b, a = np.polyfit(np.log(q), np.log(v), 1)
    return a, b


def consistency(h, n, col):
    """% difference between the N curve and the H curve, at log-spaced flows in the overlap range."""
    lo, hi = max(h.channel_flow.min(), n.channel_flow.min()), min(h.channel_flow.max(), n.channel_flow.max())
    if hi <= lo:
        return None
    ah, bh = fit(h.channel_flow, h[col])
    an, bn = fit(n.channel_flow, n[col])
    q = np.exp(np.linspace(np.log(lo), np.log(hi), 20))
    return float(np.median(100 * (np.exp((an + bn * np.log(q)) - (ah + bh * np.log(q))) - 1)))


def boot_y(d, col, qs):
    """Bootstrap y at flows qs. Returns median and 90% interval width (as % of median) per flow."""
    lq, lv = np.log(d.channel_flow.values), np.log(d[col].values)
    ys = np.empty((N_BOOT, len(qs)))
    for i in range(N_BOOT):
        idx = RNG.integers(0, len(lq), len(lq))
        if np.ptp(lq[idx]) == 0:
            ys[i] = np.nan
            continue
        b, a = np.polyfit(lq[idx], lv[idx], 1)
        ys[i] = np.exp(a + b * np.log(qs))
    med = np.nanmedian(ys, axis=0)
    width = 100 * (np.nanpercentile(ys, 95, axis=0) - np.nanpercentile(ys, 5, axis=0)) / med
    return med, width


def analyze_site(us, q_in, q_bf):
    out = {"n_H": int(us.in_hs.sum()), "n_N": int((~us.in_hs).sum()),
           "q_in": q_in, "q_bf": q_bf,
           "maxQ_H": us[us.in_hs].channel_flow.max(), "maxQ_all": us.channel_flow.max(),
           "minQ_H": us[us.in_hs].channel_flow.min(), "minQ_all": us.channel_flow.min()}
    qs = np.array([q_in, q_bf])
    for col, tag in [("channel_width", "w"), ("depth", "d")]:
        d = us[us[col] > 0]
        h, n = d[d.in_hs], d[~d.in_hs]
        if len(h) < MIN_FIT:
            continue
        if len(n) >= MIN_FIT:
            out[f"{tag}_consistency_pct"] = consistency(h, n, col)
        mh, wh = boot_y(h, col, qs)
        ma, wa = boot_y(d, col, qs)
        for k, cond in enumerate(["in", "bf"]):
            out[f"{tag}_{cond}_shift_pct"] = 100 * (ma[k] / mh[k] - 1)
            out[f"{tag}_{cond}_ci_H"] = wh[k]
            out[f"{tag}_{cond}_ci_all"] = wa[k]
    return out


def q(s, p=(.1, .25, .5, .75, .9)):
    return s.dropna().quantile(list(p)).round(1).to_dict()


def report(res, ty):
    print("Measurement types, in vs not in HYDRoSWOT:\n", ty.to_string())
    print(f"\nsites: {len(res)} | measurements per site, median: H={res.n_H.median():.0f}, N={res.n_N.median():.0f}")
    print("in-channel flow below lowest H flow:", round((res.q_in < res.minQ_H).mean(), 2),
          "| below lowest of all flows:", round((res.q_in < res.minQ_all).mean(), 2))
    print("bankfull flow above highest H flow:", round((res.q_bf > res.maxQ_H).mean(), 2),
          "| above highest of all flows:", round((res.q_bf > res.maxQ_all).mean(), 2))
    for tag, name in [("w", "WIDTH"), ("d", "DEPTH")]:
        print(f"\n== {name}")
        print("consistency (N curve vs H curve over shared flows, %):", q(res[f"{tag}_consistency_pct"]),
              "| within +/-10%:", round((res[f"{tag}_consistency_pct"].dropna().abs() <= 10).mean(), 2))
        for cond in ["in", "bf"]:
            s = res[f"{tag}_{cond}_shift_pct"]
            ci_h, ci_a = res[f"{tag}_{cond}_ci_H"], res[f"{tag}_{cond}_ci_all"]
            print(f"  [{cond}] y shift adding N (%): {q(s)} | |shift|>20%: {round((s.dropna().abs() > 20).mean(), 2)}")
            print(f"  [{cond}] 90% interval width, % of y (median): H only {ci_h.median():.0f} -> H+N {ci_a.median():.0f}"
                  f" | sites where it narrows: {round((ci_a < ci_h).mean(), 2)}")


def main(n_sites=300, seed=1, cached_only=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    tw = pd.read_parquet(REPO / "nwis_width_pred_tar_up.parquet")
    tw["siteID"] = tw.siteID.astype(str)
    sample = tw.sample(n_sites, random_state=seed).set_index("siteID")
    hs = pd.read_parquet(REPO / "adcp.parquet")
    hs = hs[hs.site_no.isin(sample.index) & (hs.meas_q_va > 0) & (hs.stream_wdth_va > 0)].copy()
    hs["date"] = pd.to_datetime(hs.site_visit_start_dt).dt.normalize()

    session = requests.Session()
    if api_key():
        session.headers["X-Api-Key"] = api_key()

    rows, types = [], []
    for site, t in sample.iterrows():
        if cached_only and not (CACHE / f"{site}.parquet").exists():
            continue
        us = clean(fetch(site, session))
        h = hs[hs.site_no == site]
        if us.empty or h.empty:
            continue
        us = label_hydroswot(us, h)
        types.append(us.groupby(["in_hs", "measurement_type"]).size().rename("n").reset_index())
        rows.append({"site_no": site, **analyze_site(us, t.in_ff, t.bf_ff)})

    res = pd.DataFrame(rows)
    res.to_csv(DER / "method_agreement_sites.csv", index=False)
    ty = pd.concat(types).groupby(["in_hs", "measurement_type"]).n.sum().unstack(0).fillna(0).astype(int)
    ty.to_csv(DER / "method_agreement_types.csv")
    report(res, ty.sort_values(True, ascending=False))


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(*nums, cached_only="--cached" in sys.argv)
