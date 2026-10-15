"""Site list for the retrain experiment, with NHDPlusV2 COMIDs.

Sites: RiverShapeML width-training sites (group "existing", COMID from the targets file) plus new
candidates from new_rows.py (>= 5 width measurements since 2000, >= 10 yr daily Q, not in training;
group "new", COMID from NLDI). Cached, so reruns only look up missing sites.

Usage: python src/site_comids.py   (after new_rows.py)
Output: data/derived/retrain_sites.csv  (site_no, group, comid)
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data/derived"
REPO = ROOT / "data/raw/repo/RiverShapeML/data"
OUT = DER / "retrain_sites.csv"
NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{}?f=json"


def nldi_comid(site):
    for _ in range(3):
        try:
            r = requests.get(NLDI.format(site), timeout=60)
            if r.status_code == 404:
                return site, None
            if r.ok:
                return site, int(r.json()["features"][0]["properties"]["comid"])
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
            pass
    return site, None


def main():
    tw = pd.read_parquet(REPO / "nwis_width_pred_tar_up.parquet")
    existing = pd.DataFrame({"site_no": tw.siteID.astype(str), "group": "existing", "comid": tw.COMID})
    s = pd.read_csv(DER / "new_rows_sites.csv", dtype={"site_no": str})
    new = s[s.conus & ~s.in_training & (s.n_meas >= 5) & (s.flow_years >= 10)][["site_no"]].assign(group="new")
    have = pd.read_csv(OUT, dtype={"site_no": str}) if OUT.exists() else pd.DataFrame(columns=["site_no", "comid"])
    known = have[have.group == "new"].dropna(subset=["comid"]) if "group" in have else have.iloc[0:0]
    todo = sorted(set(new.site_no) - set(known.site_no))
    print(f"existing: {len(existing)}, new: {len(new)}, new needing NLDI lookup: {len(todo)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        found = pd.DataFrame(list(ex.map(nldi_comid, todo)), columns=["site_no", "comid"])
    new = new.merge(pd.concat([known[["site_no", "comid"]], found]), on="site_no", how="left")
    out = pd.concat([existing, new], ignore_index=True)
    out["comid"] = pd.to_numeric(out.comid, errors="coerce").astype("Int64")
    out.to_csv(OUT, index=False)
    print(out.groupby("group").comid.agg(["size", "count"]))


if __name__ == "__main__":
    main()
