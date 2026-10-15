"""Regularization sweep with early stopping for the expanded-set retrain (width and depth).

Same data and 70/15/15 reach split as train_eval.py (seed 0). For each setting of tree depth, minimum child
weight and L2 penalty, train XGBoost at the deployed learning rate (0.01) with early stopping on the validation set
(stop after 300 rounds without improvement, at most 10,000 trees). The setting with the lowest validation RMSE is
then evaluated once on the test set, next to the deployed configuration (depth 9, min_child_weight 1, lambda 1,
fixed 9,765 / 11,000 trees) trained on the same split.

Usage: python src/tune_regularization.py
Output: data/derived/tuning/<target>/{grid.csv, comparison.csv, best.json}
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from metrics import error_metrics
from retrain_riverml import FT, model as deployed_model
from riverml_pipeline import load_features
from train_eval import TARGETS, split_reaches

DER = Path(__file__).resolve().parents[1] / "data/derived"
GRID = {"max_depth": [3, 4, 6, 9], "min_child_weight": [1, 5, 20, 50], "reg_lambda": [1, 10]}
BASE = dict(learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, n_estimators=10_000,
            early_stopping_rounds=300, eval_metric="rmse", tree_method="hist", n_jobs=-1, random_state=0)


def rmse(m, X, y):
    return float(np.sqrt(np.mean((m.predict(X) - y) ** 2)))


def main(seed=0):
    t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    for target, ycol in TARGETS.items():
        out = DER / "tuning" / target
        out.mkdir(parents=True, exist_ok=True)
        d = t.dropna(subset=[ycol])
        X, ids = load_features(target, d.comid.astype(int).unique())
        X.insert(0, "comid", ids)
        d = d.merge(X, on="comid", how="inner").reset_index(drop=True)
        feats = [c for c in X.columns if c != "comid"]
        d["split"] = split_reaches(d.comid.to_numpy(), seed)
        y = np.log1p(d[ycol] * FT)
        S = {s: (d[d.split == s][feats], y[d.split == s]) for s in ["train", "valid", "test"]}

        rows = []
        for depth, mcw, lam in itertools.product(*GRID.values()):
            m = XGBRegressor(**BASE, max_depth=depth, min_child_weight=mcw, reg_lambda=lam)
            m.fit(*S["train"], eval_set=[S["valid"]], verbose=False)
            rows.append({"max_depth": depth, "min_child_weight": mcw, "reg_lambda": lam,
                         "n_trees": m.best_iteration + 1, "train_rmse": rmse(m, *S["train"]),
                         "valid_rmse": rmse(m, *S["valid"])})
            print(f"{target} depth={depth} mcw={mcw} lambda={lam}: trees={rows[-1]['n_trees']}, "
                  f"valid={rows[-1]['valid_rmse']:.4f}, train={rows[-1]['train_rmse']:.4f}", flush=True)
        grid = pd.DataFrame(rows).sort_values("valid_rmse")
        grid.to_csv(out / "grid.csv", index=False)
        best = grid.iloc[0].to_dict()

        # final check on the test set: best setting vs the deployed configuration, same training data
        tuned = XGBRegressor(**{**BASE, "early_stopping_rounds": None, "n_estimators": int(best["n_trees"])},
                             max_depth=int(best["max_depth"]), min_child_weight=best["min_child_weight"],
                             reg_lambda=best["reg_lambda"]).fit(*S["train"])
        deployed = deployed_model(target, seed).fit(*S["train"])
        comp = {}
        for name, m in [("deployed settings", deployed), ("tuned (early stopping)", tuned)]:
            r = {"valid_rmse": rmse(m, *S["valid"]), "test_rmse": rmse(m, *S["test"]), "train_rmse": rmse(m, *S["train"]),
                 "n_trees": m.get_params()["n_estimators"]}
            Xt, yt = S["test"]
            pred_m = np.abs(np.expm1(m.predict(Xt)))
            true_m = np.expm1(yt.to_numpy())
            r.update(error_metrics(np.log(pred_m / true_m), np.log(true_m)).to_dict())
            comp[name] = r
        comp = pd.DataFrame(comp).T
        comp.to_csv(out / "comparison.csv")
        (out / "best.json").write_text(json.dumps({k: (int(v) if k in ("max_depth", "n_trees") else v)
                                                   for k, v in best.items()}, indent=2))
        print(f"\n== {target}: best by validation\n{grid.head(5).round(4).to_string(index=False)}")
        print(f"\n== {target}: test set (checked once)\n{comp.round(4).to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
