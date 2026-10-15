"""One table comparing four bankfull products for every reach, for the notebook and the memo.

  ours        : retrained with the added USGS sites (predict_product.py), with 90% intervals
  deployed    : RiverShapeML's deployed models (predict_baseline.py)
  hydroshare  : the 2024 HydroShare release (duplicate reach IDs collapsed to their median)
  bieger      : Bieger et al. (2015) regional curves applied per reach by USGS (doi:10.5066/F7765D7V)
Plus drainage area, stream order, river name (E2NHDPlus) and coordinates (RiverShapeML inputs).

Usage: python src/build_product_compare.py   (after predict_product.py)
Output: data/derived/product_compare.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER, RAW = ROOT / "data/derived", ROOT / "data/raw"
BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]


def main():
    ours = pd.read_parquet(DER / "product_ours.parquet").set_index("FEATUREID")
    ours = ours[[c for c in ours.columns if c.startswith(("tw_", "y_"))]].add_prefix("ours_")
    dep = pd.read_parquet(DER / "product_baseline.parquet").set_index("FEATUREID").add_prefix("deployed_")
    hs = (pd.read_parquet(RAW / "rivershapeml/ml_channel_dimension.parquet").groupby("hf_id")[["tw_bf(m)", "y_bf(m)"]]
          .median().rename(columns={"tw_bf(m)": "hydroshare_tw_bf_m", "y_bf(m)": "hydroshare_y_bf_m"}))
    bg = pd.read_csv(RAW / "bieger/BANKFULL_CONUS.txt").replace(-9999, np.nan).set_index("COMID")
    bg = bg[["BANKFULL_WIDTH", "BANKFULL_DEPTH"]].rename(columns={"BANKFULL_WIDTH": "bieger_tw_bf_m",
                                                                  "BANKFULL_DEPTH": "bieger_y_bf_m"})
    att = (pd.read_parquet(RAW / "enhd/enhd_nhdplusatts.parquet", columns=["comid", "totdasqkm", "streamorde", "gnis_name"])
           .set_index("comid"))
    xy = pd.read_parquet(RAW / "predict-riverML/data/ml_inputs.parquet", columns=["FEATUREID", "lat", "long"]).set_index("FEATUREID")

    d = ours.join([dep, hs, bg, att, xy], how="left")
    d.index.name = "comid"
    d["da_bin"] = pd.cut(d.totdasqkm, BINS, labels=LABELS).astype(str)
    d = d[d.totdasqkm > 0]
    d.reset_index().to_parquet(DER / "product_compare.parquet", index=False)

    print(f"{len(d):,} reaches; coverage:",
          {c: f"{d[c].notna().mean():.1%}" for c in ["deployed_tw_bf_m", "hydroshare_tw_bf_m", "bieger_tw_bf_m", "lat"]})
    for v in ["tw_bf_m", "y_bf_m"]:
        cols = {k: f"{k}_{v}" for k in ["ours", "deployed", "hydroshare", "bieger"]}
        med = pd.DataFrame({k: d.groupby("da_bin")[c].median() for k, c in cols.items()}).reindex(LABELS).round(2)
        print(f"\nmedian {v} by drainage area (km²)\n{med.to_string()}")
        for k in ["ours", "deployed", "hydroshare"]:
            r = (d[cols[k]] / d[cols["bieger"]]).replace([np.inf, -np.inf], np.nan).dropna()
            print(f"  {k} / bieger: median {r.median():.2f}  (IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f})")


if __name__ == "__main__":
    main()
