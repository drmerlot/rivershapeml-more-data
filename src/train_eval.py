"""Train / validation / test diagnostics for the final method (B2: all sites with targets, RiverShapeML's
30 deployment features, MODEL_PARAMS settings), for bankfull width and depth.

Split: 70 / 15 / 15 by reach (COMID), so no reach appears in two sets. The model is fit on train only, with
the full fixed number of trees from RiverShapeML's settings (no early stopping); train, validation and test
RMSE (on log1p(meters), the model's own loss scale) are recorded after every boosting round.

Usage: python src/train_eval.py [--seed 0]
Output: data/derived/training/<target>/{curves.parquet, predictions.parquet, metrics.csv, setup.json}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import error_metrics
from retrain_riverml import FT, model
from riverml_pipeline import load_features

DER = Path(__file__).resolve().parents[1] / "data/derived"
TARGETS = {"TW_bf": "tw_all", "Y_bf": "td_all"}


def split_reaches(comids, seed):
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(comids))
    n = len(u)
    part = dict(zip(u[: int(.70 * n)], ["train"] * n)) | dict(zip(u[int(.70 * n): int(.85 * n)], ["valid"] * n)) \
        | dict(zip(u[int(.85 * n):], ["test"] * n))
    return pd.Series(comids).map(part).to_numpy()


def main(seed=0):
    t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    for target, ycol in TARGETS.items():
        out = DER / "training" / target
        out.mkdir(parents=True, exist_ok=True)
        d = t.dropna(subset=[ycol])
        X, ids = load_features(target, d.comid.astype(int).unique())
        X.insert(0, "comid", ids)
        d = d.merge(X, on="comid", how="inner").reset_index(drop=True)
        feats = [c for c in X.columns if c != "comid"]
        d["split"] = split_reaches(d.comid.to_numpy(), seed)
        y = np.log1p(d[ycol] * FT)
        sets = {s: (d[d.split == s][feats], y[d.split == s]) for s in ["train", "valid", "test"]}

        m = model(target, seed)
        m.set_params(eval_metric="rmse")
        m.fit(*sets["train"], eval_set=[sets["train"], sets["valid"], sets["test"]], verbose=False)
        ev = m.evals_result()
        curves = pd.DataFrame({name: ev[f"validation_{i}"]["rmse"] for i, name in enumerate(["train", "valid", "test"])})
        curves.index.name = "round"
        curves.reset_index().to_parquet(out / "curves.parquet", index=False)

        pred_m = np.abs(np.expm1(m.predict(d[feats])))
        obs_m = d[ycol] * FT
        p = pd.DataFrame({"site_no": d.site_no, "comid": d.comid, "group": d.group, "split": d.split,
                          "observed_m": obs_m, "predicted_m": pred_m})
        p.to_parquet(out / "predictions.parquet", index=False)
        met = pd.DataFrame({s: error_metrics(np.log(g.predicted_m / g.observed_m), np.log(g.observed_m))
                            for s, g in p.groupby("split")}).T.loc[["train", "valid", "test"]]
        met.to_csv(out / "metrics.csv")
        best = int(curves.valid.idxmin())
        (out / "setup.json").write_text(json.dumps({
            "target": target, "target_column": ycol, "seed": seed, "n_features": len(feats), "features": feats,
            "params": {k: v for k, v in m.get_params().items() if v is not None and k in
                       ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "random_state")},
            "sites": d.split.value_counts().to_dict(), "reaches": d.groupby("split").comid.nunique().to_dict(),
            "groups_by_split": d.groupby(["split", "group"]).size().unstack(fill_value=0).to_dict("index"),
            "best_valid_round": best, "valid_rmse_best": float(curves.valid.min()),
            "valid_rmse_final": float(curves.valid.iloc[-1]), "train_rmse_final": float(curves.train.iloc[-1])}, indent=2))
        print(f"{target}: sites {d.split.value_counts().to_dict()} | valid RMSE best {curves.valid.min():.4f} at round {best} "
              f"vs final {curves.valid.iloc[-1]:.4f} | train final {curves.train.iloc[-1]:.4f}")
        print(met.round(3).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
