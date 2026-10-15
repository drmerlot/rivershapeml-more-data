"""Page through a national USGS OGC API collection and save selected fields.

Usage: python src/fetch_usgs_ogc.py [channel-measurements|monitoring-locations|time-series-metadata]
Writes data/raw/<collection>/page_NNNN.parquet; resumable via cursor.json.
"""
import base64
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from keys import api_key

ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = {
    "channel-measurements": ("channel_meas", "monitoring_location_id,time,channel_flow,channel_width,"
                             "channel_area,measurement_type,channel_name", ""),
    "monitoring-locations": ("monitoring_locations", "id,state_code,site_type_code,drainage_area,"
                             "contributing_drainage_area,monitoring_location_name", "&site_type_code=ST"),
    # daily mean discharge series: which sites have a flow record for flood frequency
    "time-series-metadata": ("ts_meta", "monitoring_location_id,parameter_code,statistic_id,begin,end,primary",
                             "&parameter_code=00060&statistic_id=00003"),
}
COLL = sys.argv[1] if len(sys.argv) > 1 else "channel-measurements"
SUBDIR, PROPS, EXTRA = COLLECTIONS[COLL]
OUT = ROOT / "data/raw" / SUBDIR
OUT.mkdir(parents=True, exist_ok=True)
BASE = f"https://api.waterdata.usgs.gov/ogcapi/v0/collections/{COLL}/items"
STATE = OUT / "cursor.json"
PAGE = 20000


def bump_cursor(url, step):
    """The cursor is a base64-encoded record UUID; advance it by `step`."""
    cur = re.search(r"cursor=([^&]+)", url).group(1)
    uid = base64.b64decode(cur + "=" * (-len(cur) % 4)).decode()
    v = f"{int(uid.replace('-', ''), 16) + step:032x}"
    new = base64.b64encode(f"{v[:8]}-{v[8:12]}-{v[12:16]}-{v[16:20]}-{v[20:]}".encode()).decode().rstrip("=")
    return url.replace(cur, new)


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {"page": 0, "url": None}
    url = st["url"] or f"{BASE}?f=json&limit={PAGE}&skipGeometry=true&properties={PROPS}{EXTRA}"
    s = requests.Session()
    # Anonymous requests get rate-limited (HTTP 403/429); a free key from
    # https://api.waterdata.usgs.gov/signup/ raises the limit.
    if api_key():
        s.headers["X-Api-Key"] = api_key()
    skip = 0
    while url:
        for attempt in range(30):
            r = s.get(url, timeout=300)
            if r.status_code == 200:
                break
            limit = int(re.search(r"limit=(\d+)", url).group(1))
            if r.status_code == 500 and limit > 1:
                # some records make the server fail any page that contains them: shrink the page
                url = re.sub(r"limit=\d+", f"limit={max(1, limit // 5)}", url)
                continue
            if r.status_code == 500 and "cursor=" in url:
                # even one record fails: step the cursor (a record UUID) past it; may drop ~1 neighbor
                skip = skip * 2 or 2**100
                url = bump_cursor(url, skip)
                print("skipping bad record(s), step", hex(skip), flush=True)
                continue
            print("status", r.status_code, "retrying", flush=True)
            time.sleep(30 * (attempt + 1))
        r.raise_for_status()
        skip = 0
        d = r.json()
        df = pd.DataFrame([f["properties"] for f in d["features"]])
        df.to_parquet(OUT / f"page_{st['page']:04d}.parquet")
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
        if url:
            url = re.sub(r"limit=\d+", f"limit={PAGE}", url)
        st = {"page": st["page"] + 1, "url": url}
        STATE.write_text(json.dumps(st))
        print(st["page"], len(df), r.headers.get("x-ratelimit-remaining"), flush=True)


if __name__ == "__main__":
    main()
