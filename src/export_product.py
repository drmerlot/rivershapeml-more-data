"""Release file for the re-predicted product: "ML-derived channel width and depth, expanded-set retrain".

One row per NHDPlusV2 reach (COMID): bankfull width and depth from the expanded-set retrain, with split-conformal
90% prediction intervals (5th and 95th percentiles). Values in meters, rounded to the millimeter. Reaches with a
zero or missing drainage area have no interval (NaN).

Usage: python src/export_product.py
Output: data/derived/release/rivershapeml_expanded_set_v1.parquet  (attached to the GitHub release, not tracked)
"""
from pathlib import Path

import pandas as pd

DER = Path(__file__).resolve().parents[1] / "data/derived"
COLS = ["tw_bf_m", "tw_bf_m_p05", "tw_bf_m_p95", "y_bf_m", "y_bf_m_p05", "y_bf_m_p95"]


def main():
    d = pd.read_parquet(DER / "product_ours.parquet")
    out = pd.DataFrame({"comid": d.FEATUREID.astype("int64")})
    for c in COLS:
        out[c] = d[c].round(3).astype("float32")
    (DER / "release").mkdir(exist_ok=True)
    path = DER / "release/rivershapeml_expanded_set_v1.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {path.name}: {len(out):,} reaches; without intervals: {out.tw_bf_m_p05.isna().sum():,}")


if __name__ == "__main__":
    main()
