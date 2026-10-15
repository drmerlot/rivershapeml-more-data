"""Reach features (x) for the retrain experiment, from public sources, keyed by COMID.

Approximates the 15 features of RiverShapeML's deployed bankfull-width model:
  direct      StreamOrde, ArbolateSu, TotDASqKM, slope, roughness  <- E2NHDPlusV2 / NWM v2.1 attributes
              RckDepWs, BFIWs, WtDepWs                             <- EPA StreamCat (watershed)
  stand-ins   Topo_pc_*      (elevation, lat/long, climate)  -> StreamCat elev, precip, temp, runoff; gage lat/lon
              Watershed_pc_0 (HLR, WetIndexWs, PctWdWet2011Ws) -> StreamCat wetindex, pctwdwet2011
              bf_ff (NWM 2-yr flood)                           -> NWM v2.1 2-yr flood (log_nwm_q2, from nwm_q2.py)
                                                                  or the gage 2-yr flood (log_q2); the experiment uses one
              qsb_tavg (GLDAS subsurface runoff)               -> StreamCat runoff, bfi
  missing     Sinuosity, rp_100
Plus a few soil and land-cover fields (clay, sand, permeability, impervious) as general context.

Usage: python src/build_features.py   (after build_targets.py and nwm_q2.py)
Output: data/derived/retrain_features.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from keys import api_key

ROOT = Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data/raw", ROOT / "data/derived"
STREAMCAT = "https://api.epa.gov/StreamCat/streams/metrics"
SC_METRICS = ["bfi", "rckdep", "wtdep", "wetindex", "pctwdwet2011", "precip8110", "tmean8110", "elev",
              "runoff", "clay", "sand", "perm", "om", "pctimp2011"]
ENHD_COLS = ["comid", "streamorde", "arbolatesu", "totdasqkm", "slope", "roughness", "lengthkm", "slopelenkm"]
SC_CACHE = DER / "streamcat_cache.csv"


def streamcat(comids):
    have = pd.read_csv(SC_CACHE) if SC_CACHE.exists() else pd.DataFrame(columns=["comid"])
    todo = sorted(set(comids) - set(have.comid))
    rows = []
    for i in range(0, len(todo), 200):
        chunk = ",".join(str(c) for c in todo[i:i + 200])
        r = requests.post(STREAMCAT, params={"api_key": api_key()} if api_key() else {}, timeout=300,
                          data={"name": ",".join(SC_METRICS), "areaOfInterest": "watershed", "comid": chunk})
        r.raise_for_status()
        rows += r.json()["items"]
        print(f"  streamcat {i + 200}/{len(todo)}", flush=True)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(SC_CACHE, index=False)
    return out


def main():
    t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    comids = t.comid.astype(int).unique()

    enhd = pd.read_parquet(RAW / "enhd/enhd_nhdplusatts.parquet", columns=ENHD_COLS)
    enhd = enhd[enhd.comid.isin(comids)]
    sc = streamcat(comids)
    sc = sc[["comid"] + [c for c in sc.columns if c.endswith("ws") and not c.endswith("rp100ws")
                         and c.removesuffix("ws") in SC_METRICS]]

    ml = pd.concat(pd.read_parquet(p, columns=["id", "lon", "lat"])
                   for p in sorted((RAW / "monitoring_locations").glob("page_*.parquet")))
    ml = ml.assign(site_no=ml.id.str[5:]).drop(columns="id").drop_duplicates("site_no")

    x = (t[["site_no", "comid", "q2"]]
         .merge(ml, on="site_no", how="left")
         .merge(enhd, on="comid", how="left")
         .merge(sc, on="comid", how="left"))
    x["log_q2"] = np.log(x.q2)
    # NWM v2.1 2-year flood (what a deployed model would get at ungaged reaches); from nwm_q2.py
    nwm = pd.read_csv(DER / "nwm_q2.csv")
    x = x.merge(nwm[["comid", "nwm_q2"]], on="comid", how="left")
    x["log_nwm_q2"] = np.log(x.nwm_q2.where(x.nwm_q2 > 0))
    x = x.drop(columns=["nwm_q2"])
    x["log_totdasqkm"] = np.log(x.totdasqkm.clip(lower=0.01))
    x["log_arbolatesu"] = np.log(x.arbolatesu.clip(lower=0.01))
    x["log_slope"] = np.log(x.slope.clip(lower=1e-5))
    x = x.drop(columns=["q2"])
    x.to_csv(DER / "retrain_features.csv", index=False)
    print(f"features for {len(x)} sites, {x.shape[1] - 2} columns")
    print("missing share:", x.drop(columns=["site_no", "comid"]).isna().mean().round(3).sort_values(ascending=False).head(8).to_dict())


if __name__ == "__main__":
    main()
