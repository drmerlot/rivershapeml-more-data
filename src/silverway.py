"""The opening example: USGS 08156675, Shoal Ck at Silverway Dr, Austin TX.

Prints the NHDPlus reach (via NLDI), its RiverShapeML values, how many width measurements USGS
has at the gage, and how many of those RiverShapeML's HYDRoSWOT copy contains.

Usage: python src/silverway.py
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from keys import api_key

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SITE = "08156675"
NLDI = "https://api.water.usgs.gov/nldi/linked-data"
API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/channel-measurements/items"


def main():
    site = requests.get(f"{NLDI}/nwissite/USGS-{SITE}?f=json", timeout=60).json()["features"][0]["properties"]
    comid = int(site["comid"])
    reach = gpd.read_file(f"{NLDI}/comid/{comid}?f=json").to_crs(32614)
    print(f"{site['name']}: COMID {comid}, reach length {reach.length.iloc[0] / 1000:.1f} km")

    r = pd.read_parquet(RAW / "rivershapeml/ml_channel_dimension.parquet").set_index("hf_id").loc[comid]
    print(f"RiverShapeML bankfull: width {r['tw_bf(m)']:.1f} m, depth {r['y_bf(m)']:.2f} m")

    cache = RAW / f"usgs_channel-measurements_{SITE}.json"
    if not cache.exists():
        headers = {"X-Api-Key": api_key()} if api_key() else {}
        cache.write_text(requests.get(f"{API}?f=json&limit=1000&monitoring_location_id=USGS-{SITE}",
                                      headers=headers, timeout=120).text)
    cm = pd.DataFrame([f["properties"] for f in json.loads(cache.read_text())["features"]])
    w = pd.to_numeric(cm.channel_width, errors="coerce")
    q = pd.to_numeric(cm.channel_flow, errors="coerce")
    ok = cm[(w > 0) & (q > 0)]
    print(f"USGS width measurements: {len(ok)} ({ok.measurement_type.value_counts().to_dict()})")

    hs = pd.read_parquet(RAW / "repo/RiverShapeML/data/adcp.parquet")
    hs = hs[(hs.site_no == SITE) & (hs.stream_wdth_va > 0)]
    print(f"In RiverShapeML's HYDRoSWOT copy: {len(hs)} ({', '.join(hs.site_visit_start_dt.astype(str).str[:10])})")
    tw = pd.read_parquet(RAW / "repo/RiverShapeML/data/nwis_width_pred_tar_up.parquet")
    print(f"In width training targets: {SITE in set(tw.siteID.astype(str))}")


if __name__ == "__main__":
    main()
