"""Does adding the extra USGS measurements improve a bankfull-width model?

Stand-in model (not RiverShapeML itself): gradient-boosted trees on log(width) with the public features
from build_features.py. Three training sets, all evaluated on the same held-out sites:
  A   status quo     : existing training sites, HYDRoSWOT-only targets
  B1  better targets : existing training sites, all-measurement targets
  B2  + new sites    : B1 plus new sites (all-measurement targets)
Evaluation targets are the all-measurement targets (the best available estimate) for held-out
existing AND new sites; a second score uses HYDRoSWOT-only targets so B1/B2 aren't favored by
sharing the evaluation target's definition. Folds are grouped by COMID so no reach is in both train and test; the
split is repeated with several seeds.

Usage: python src/retrain_experiment.py [--flow nwm|gage] [--targets CSV] [--name RUN]
  --flow nwm  (default) uses the NWM 2-yr flood as the flow feature, as a deployed model would get;
  --flow gage uses the gage's 2-yr flood (optimistic: not available at ungaged reaches).
Output: data/derived/runs/<name>/{config.json, predictions.parquet, summary.txt}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data/derived"
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]
VARIANTS = {"A_status_quo": ("tw_hs", {"existing"}),
            "B1_better_targets": ("tw_all", {"existing"}),
            "B2_plus_new_sites": ("tw_all", {"existing", "new"})}


def model(seed):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


pd.set_option("display.width", 200)


def main(flow="nwm", targets_csv=DER / "retrain_targets.csv", name=None):
    name = name or f"{flow}_{Path(targets_csv).stem}"
    run_dir = DER / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(targets_csv, dtype={"site_no": str})
    x = pd.read_csv(DER / "retrain_features.csv", dtype={"site_no": str})
    d = t.merge(x.drop(columns="comid"), on="site_no").reset_index(drop=True)
    drop = {"site_no", "comid", "log_nwm_q2" if flow == "gage" else "log_q2"}
    feats = [c for c in x.columns if c not in drop]
    (run_dir / "config.json").write_text(json.dumps(
        {"flow": flow, "targets": str(Path(targets_csv).name), "features": feats, "seeds": SEEDS,
         "n_folds": N_FOLDS, "model": model(0).get_params(), "variants": {k: [v[0], sorted(v[1])] for k, v in VARIANTS.items()}},
        indent=2, default=str))
    d["da_bin"] = pd.cut(d.totdasqkm, BINS, labels=LABELS)
    evaluable = d.tw_all.notna()

    preds = []
    for seed in SEEDS:
        # shuffle reach groups so each seed gives a different grouped split
        rng = np.random.default_rng(seed)
        perm = {c: i for i, c in enumerate(rng.permutation(d.comid.unique()))}
        groups = d.comid.map(perm)
        for fold, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(d, groups=groups)):
            train, test = d.iloc[tr], d.iloc[te]
            test = test[test.tw_all.notna()]
            for name, (ycol, use) in VARIANTS.items():
                trn = train[train.group.isin(use) & train[ycol].notna()]
                m = model(seed).fit(trn[feats], np.log(trn[ycol]))
                preds.append(pd.DataFrame({"site_no": test.site_no, "group": test.group, "da_bin": test.da_bin,
                                           "seed": seed, "fold": fold, "variant": name, "n_train": len(trn),
                                           "log_true": np.log(test.tw_all), "log_true_hs": np.log(test.tw_hs),
                                           "log_err": m.predict(test[feats]) - np.log(test.tw_all),
                                           "log_err_hs": m.predict(test[feats]) - np.log(test.tw_hs)}))
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(run_dir / "predictions.parquet", index=False)

    def table(by):
        return summarize(p, by).set_index(by + ["variant"] if by else ["variant"])

    lines = []

    def log(*a):
        """Print and keep a copy for the run's summary.txt."""
        lines.append(" ".join(str(x) for x in a))
        print(*a)

    log(f"run: {name}")
    log(f"evaluable sites: {evaluable.sum()} (existing {(evaluable & (d.group == 'existing')).sum()}, "
          f"new {(evaluable & (d.group == 'new')).sum()}); features: {len(feats)}")
    log("metrics: see src/metrics.py (ratio = predicted / target)")
    log("training rows per variant (mean):", p.groupby("variant").n_train.mean().round(0).to_dict())
    log("\n== All held-out sites"); log(table([]).to_string())
    log("\n== By site group"); log(table(["group"]).to_string())
    log("\n== By drainage area (km²)"); log(table(["da_bin"]).to_string())

    # Fairness check: judge all variants by the status-quo kind of target (HYDRoSWOT-only), on the
    # existing sites that have one, so B1/B2 get no advantage from sharing the evaluation target's definition.
    p["log_err"], p["log_true"] = p.log_err_hs, p.log_true_hs
    p.dropna(subset=["log_err"], inplace=True)
    log("\n== Fairness check: scored against HYDRoSWOT-only targets (existing sites)")
    log(table([]).to_string())
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=["nwm", "gage"], default="nwm")
    ap.add_argument("--targets", default=DER / "retrain_targets.csv")
    ap.add_argument("--name")
    a = ap.parse_args()
    main(a.flow, a.targets, a.name)
