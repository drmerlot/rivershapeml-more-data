"""Retrain experiment on RiverShapeML's own setup: their 30 deployment features (from ml_inputs.parquet,
engineered exactly as their pipeline does) and their XGBoost hyperparameters, with our targets.

Variants (all scored on the same held-out sites, folds grouped by COMID):
  A   status quo      : existing sites, HYDRoSWOT-only targets        (width only)
  B1  better targets  : existing sites, all-measurement targets
  B2  + new sites     : B1 plus new sites
Targets are converted to meters and modeled as log1p(m), as their models are. Errors are ln(pred / target).

Usage: python src/retrain_riverml.py --target TW_bf|Y_bf [--seeds 3] [--name RUN]
Output: data/derived/runs/<name>/{config.json, predictions.parquet, summary.txt}  (same layout as retrain_experiment.py)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from metrics import summarize
from riverml_pipeline import THEIR_PARAMS, load_features

DER = Path(__file__).resolve().parents[1] / "data/derived"
RAW = Path(__file__).resolve().parents[1] / "data/raw"
FT = 0.3048
N_FOLDS = 5
BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]
TARGET_COLS = {"TW_bf": ("tw_all", "tw_hs"), "Y_bf": ("td_all", None)}
pd.set_option("display.width", 200)


def model(target, seed):
    return XGBRegressor(**{**THEIR_PARAMS[target], "random_state": seed}, tree_method="hist", n_jobs=-1)


def main(target="TW_bf", n_seeds=3, name=None):
    name = name or f"riverml_{target}"
    run_dir = DER / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    y_all, y_hs = TARGET_COLS[target]
    variants = {"B1_better_targets": (y_all, {"existing"}), "B2_plus_new_sites": (y_all, {"existing", "new"})}
    if y_hs:
        variants = {"A_status_quo": (y_hs, {"existing"}), **variants}

    t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    da = pd.read_parquet(RAW / "enhd/enhd_nhdplusatts.parquet", columns=["comid", "totdasqkm"])
    X, ids = load_features(target, t.comid.astype(int).unique())
    X.insert(0, "comid", ids)
    feats = [c for c in X.columns if c != "comid"]
    d = t.merge(da, on="comid", how="left").merge(X, on="comid", how="inner").reset_index(drop=True)
    d["da_bin"] = pd.cut(d.totdasqkm, BINS, labels=LABELS)
    for c in [y_all, y_hs]:
        if c:
            d[c + "_m"] = d[c] * FT
    (run_dir / "config.json").write_text(json.dumps(
        {"target": target, "features": feats, "model": "XGBRegressor", "params": THEIR_PARAMS[target],
         "seeds": list(range(n_seeds)), "n_folds": N_FOLDS, "targets": "retrain_targets.csv",
         "variants": {k: [v[0], sorted(v[1])] for k, v in variants.items()}}, indent=2))
    print(f"{name}: {len(d)} sites with features; evaluable {d[y_all].notna().sum()}", flush=True)

    preds = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        perm = {c: i for i, c in enumerate(rng.permutation(d.comid.unique()))}
        for fold, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(d, groups=d.comid.map(perm))):
            train, test = d.iloc[tr], d.iloc[te]
            test = test[test[y_all].notna()]
            for vname, (ycol, use) in variants.items():
                trn = train[train.group.isin(use) & train[ycol].notna()]
                m = model(target, seed).fit(trn[feats], np.log1p(trn[ycol + "_m"]))
                lp = np.log(np.abs(np.expm1(m.predict(test[feats]))))
                rec = {"site_no": test.site_no, "group": test.group, "da_bin": test.da_bin, "seed": seed, "fold": fold,
                       "variant": vname, "n_train": len(trn), "log_true": np.log(test[y_all + "_m"]),
                       "log_err": lp - np.log(test[y_all + "_m"])}
                if y_hs:
                    rec["log_true_hs"] = np.log(test[y_hs + "_m"])
                    rec["log_err_hs"] = lp - np.log(test[y_hs + "_m"])
                preds.append(pd.DataFrame(rec))
            print(f"  seed {seed} fold {fold} done", flush=True)
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(run_dir / "predictions.parquet", index=False)

    lines = []

    def log(*a):
        lines.append(" ".join(str(x) for x in a))
        print(*a)

    def table(by):
        return summarize(p, by).set_index(by + ["variant"] if by else ["variant"])

    log(f"run: {name}  (their features + their XGBoost settings; target {target})")
    log("training rows per variant (mean):", p.groupby("variant").n_train.mean().round(0).to_dict())
    log("\n== All held-out sites"); log(table([]).to_string())
    log("\n== By site group"); log(table(["group"]).to_string())
    log("\n== By drainage area (km²)"); log(table(["da_bin"]).to_string())
    if y_hs:
        q = p.dropna(subset=["log_err_hs"]).assign(log_err=lambda x: x.log_err_hs, log_true=lambda x: x.log_true_hs)
        log("\n== Fairness check: scored against HYDRoSWOT-only targets (existing sites)")
        log(summarize(q, []).set_index("variant").to_string())
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["TW_bf", "Y_bf"], default="TW_bf")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--name")
    a = ap.parse_args()
    main(a.target, a.seeds, a.name)
