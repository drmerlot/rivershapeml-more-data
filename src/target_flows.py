"""Where do the training targets sit on the flow axis?

From RiverShapeML's width targets (R2 > 0.6): the in-channel flow (100% AEP) relative to the bankfull
flow (50% AEP), and both relative to the largest measured flow at each site.

Usage: python src/target_flows.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/repo/RiverShapeML/data"


def main():
    t = pd.read_parquet(REPO / "nwis_width_pred_tar_up.parquet")
    t = t[t.R2 > 0.6]
    qs = [.1, .25, .5, .75, .9]
    print(f"training sites: {len(t)}")
    print("in-channel flow / bankfull flow:", (t.in_ff / t.bf_ff).quantile(qs).round(3).to_dict())
    print("in-channel flow / max measured flow:", (t.in_ff / t.meas_q_va).quantile(qs).round(2).to_dict())
    print("bankfull flow / max measured flow:", (t.bf_ff / t.meas_q_va).quantile(qs).round(2).to_dict())


if __name__ == "__main__":
    main()
