"""NWM v2.1 retrospective 2-year flood at the retrain sites' reaches.

For each COMID in retrain_targets.csv that is in the USGS NWM-at-gages file (feature_id = COMID):
annual maximum hourly flow per water year (WY1980-WY2020), then the median = 2-year flood.
Same recipe as the gage-based q2 (median annual peak), so differences come from the model, not the method.

Usage: python src/nwm_q2.py
Output: data/derived/nwm_q2.csv (comid, nwm_q2 [cfs], n_years) and a gage-vs-NWM comparison
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data/derived"
NC = ROOT / "data/raw/nwm_gages/nwmv21_nwis.nc"
CMS_TO_CFS = 35.3147
BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]


def main():
    t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    ds = xr.open_dataset(NC)
    fids = pd.Index(ds.feature_id.values)
    comids = sorted(set(t.comid.astype(int)) & set(fids))
    print(f"retrain reaches: {t.comid.nunique()}, in NWM-at-gages file: {len(comids)}", flush=True)

    time = pd.DatetimeIndex(ds.time.values)
    wy = np.where(time.month >= 10, time.year + 1, time.year)
    full = (wy >= 1980) & (wy <= 2020)
    wy = wy[full]
    rows = []
    for i, c in enumerate(comids):
        q = ds.streamflow.isel(feature_id=fids.get_loc(c)).values[full]
        amax = pd.Series(q).groupby(wy).max().dropna()
        rows.append({"comid": c, "nwm_q2": float(amax.median()) * CMS_TO_CFS, "n_years": len(amax)})
        if i % 1000 == 0:
            print(f"  {i}/{len(comids)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DER / "nwm_q2.csv", index=False)

    x = pd.read_csv(DER / "retrain_features.csv", dtype={"site_no": str})[["site_no", "totdasqkm"]]
    m = t.merge(out, on="comid").merge(x, on="site_no").dropna(subset=["q2"])
    m = m[m.nwm_q2 > 0]
    m["ratio"] = m.nwm_q2 / m.q2
    m["da_bin"] = pd.cut(m.totdasqkm, BINS, labels=LABELS)
    lr = np.log(m.ratio)
    print(f"\nsites with gage and NWM q2: {len(m)}; NWM/gage median {m.ratio.median():.2f}; "
          f"log R² {np.corrcoef(np.log(m.q2), np.log(m.nwm_q2))[0, 1] ** 2:.2f}; typical factor {np.exp(lr.std()):.2f}")
    print(m.groupby("da_bin", observed=True).ratio.agg(
        sites="size", median="median", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9)).round(2).to_string())


if __name__ == "__main__":
    main()
