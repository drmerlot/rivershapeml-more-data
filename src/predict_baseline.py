"""Baseline product: RiverShapeML's currently deployed models (NOAA-OWP/predict-riverML) run on every reach.

Usage: python src/predict_baseline.py
Output: data/derived/product_baseline.parquet  (FEATUREID, tw_bf_m, y_bf_m)
"""
from pathlib import Path

import pandas as pd

from riverml_pipeline import ID, load_features, load_model, predict

DER = Path(__file__).resolve().parents[1] / "data/derived"


def main():
    out = None
    for target, col in [("TW_bf", "tw_bf_m"), ("Y_bf", "y_bf_m")]:
        X, ids = load_features(target)
        model = load_model(target)
        preds = pd.concat(pd.Series(predict(model, X.iloc[i:i + 250_000]), index=X.index[i:i + 250_000])
                          for i in range(0, len(X), 250_000))
        df = pd.DataFrame({ID: ids, col: preds.sort_index().to_numpy()})
        out = df if out is None else out.merge(df, on=ID)
        print(target, df[col].describe().round(2).to_dict(), flush=True)
    out.to_parquet(DER / "product_baseline.parquet", index=False)


if __name__ == "__main__":
    main()
