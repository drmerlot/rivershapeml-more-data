"""How many training rows could the full USGS channel-measurement record add to RiverShapeML?

A site counts as a candidate if it is in CONUS, not already a width training site, has
>= MIN_MEAS width measurements since 2000 (Q > 0, W > 0, no ice), and has >= MIN_YEARS of daily
mean discharge (needed for flood frequency). The R2 > 0.6 filter can't be applied without fitting
each site, so the share passing it is taken from the existing targets.

Usage: python src/new_rows.py
Inputs: data/raw/channel_meas/ (fetch_usgs_ogc.py channel-measurements),
        data/raw/ts_meta/ (time-series-metadata), data/raw/monitoring_locations/
Output: data/derived/new_rows_sites.csv and a printed summary
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data/raw", ROOT / "data/derived"
REPO = RAW / "repo/RiverShapeML/data"
MIN_MEAS, MIN_YEARS = 5, 10
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}
BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]


def pages(subdir, columns=None):
    """Concatenate a download folder's page_*.parquet files (skipping cursor.json)."""
    return pd.concat((pd.read_parquet(p, columns=columns) for p in sorted((RAW / subdir).glob("page_*.parquet"))),
                     ignore_index=True)


def width_counts():
    cols = ["monitoring_location_id", "time", "channel_flow", "channel_width", "measurement_type"]
    cm = pages("channel_meas", cols)
    print(f"USGS channel measurements: {len(cm):,} rows")
    cm = cm[cm.monitoring_location_id.str.startswith("USGS-")]
    for c in ["channel_flow", "channel_width"]:
        cm[c] = pd.to_numeric(cm[c], errors="coerce")
    cm = cm[(cm.channel_flow > 0) & (cm.channel_width > 0) & (cm.measurement_type != "Ice")]
    cm = cm[cm.time.str[:4] >= "2000"]
    print(f"  with Q > 0, W > 0, no ice, since 2000: {len(cm):,}")
    return cm.groupby(cm.monitoring_location_id.str[5:]).size().rename("n_meas")


def main():
    n = width_counts()
    ml = pages("monitoring_locations")
    ml = ml[ml.id.str.startswith("USGS-")].assign(site=lambda d: d.id.str[5:]).set_index("site")
    ts = pages("ts_meta")
    ts = ts[ts.monitoring_location_id.str.startswith("USGS-")]
    yrs = ((pd.to_datetime(ts.end, format="ISO8601") - pd.to_datetime(ts.begin, format="ISO8601")).dt.days / 365.25
           ).groupby(ts.monitoring_location_id.str[5:]).max()
    tw = pd.read_parquet(REPO / "nwis_width_pred_tar_up.parquet")
    train = set(tw.siteID.astype(str))
    train_r2 = set(tw[tw.R2 > 0.6].siteID.astype(str))
    pass_rate = len(train_r2) / len(train)

    s = pd.DataFrame({"n_meas": n})
    s["conus"] = s.index.isin(ml.index[~ml.state_code.isin(NON_CONUS)])
    s["da_km2"] = pd.to_numeric(ml.drainage_area, errors="coerce").reindex(s.index) * 2.58999
    s["flow_years"] = yrs.reindex(s.index).fillna(0)
    s["in_training"] = s.index.isin(train)
    cand = s[s.conus & ~s.in_training & (s.n_meas >= MIN_MEAS)]
    cand_q = cand[cand.flow_years >= MIN_YEARS]
    s.to_csv(DER / "new_rows_sites.csv", index_label="site_no")

    print(f"\nCONUS sites with >= {MIN_MEAS} width measurements since 2000: {(s.conus & (s.n_meas >= MIN_MEAS)).sum():,}")
    print(f"  not in training: {len(cand):,};  with >= {MIN_YEARS} yr daily Q: {len(cand_q):,}")
    print(f"  expected to pass R2 > 0.6 at the existing rate ({pass_rate:.0%}): ~{pass_rate * len(cand_q):,.0f}")
    print(f"current training rows with R2 > 0.6: {len(train_r2):,}")
    da = pd.to_numeric(ml.drainage_area, errors="coerce") * 2.58999
    now = pd.cut(da.reindex(list(train_r2)), BINS, labels=LABELS).value_counts().sort_index()
    add = (pass_rate * pd.cut(cand_q.da_km2, BINS, labels=LABELS).value_counts().sort_index()).round()
    print(pd.DataFrame({"training_now": now, "est_additions": add, "est_after": now + add,
                        "factor": ((now + add) / now).round(1)}).to_string())
    print(f"no drainage area: now {da.reindex(list(train_r2)).isna().sum()}, candidates {cand_q.da_km2.isna().sum()}")
    no_q = cand[cand.flow_years < MIN_YEARS]
    print(f"\ncandidates lacking a long daily record (NWM-based flood frequency?): {len(no_q):,}")
    print(f"Silverway 08156675: {s.loc['08156675'].to_dict() if '08156675' in s.index else 'n/a'}")


if __name__ == "__main__":
    main()
