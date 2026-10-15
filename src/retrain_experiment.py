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

Usage: python src/retrain_experiment.py [targets_csv]
Output: data/derived/retrain_predictions.csv and a printed summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

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


def main(targets_csv=DER / "retrain_targets.csv"):
    t = pd.read_csv(targets_csv, dtype={"site_no": str})
    x = pd.read_csv(DER / "retrain_features.csv", dtype={"site_no": str})
    d = t.merge(x.drop(columns="comid"), on="site_no").reset_index(drop=True)
    feats = [c for c in x.columns if c not in ("site_no", "comid")]
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
                                           "log_err": m.predict(test[feats]) - np.log(test.tw_all),
                                           "log_err_hs": m.predict(test[feats]) - np.log(test.tw_hs)}))
    p = pd.concat(preds, ignore_index=True)
    p.to_csv(DER / "retrain_predictions.csv", index=False)

    def summarize(g):
        rmse = np.sqrt((g.log_err ** 2).mean())
        return pd.Series({"sites": g.site_no.nunique(), "typical_err_pct": 100 * (np.exp(rmse) - 1),
                          "median_abs_err_pct": 100 * (np.exp(g.log_err.abs().median()) - 1),
                          "bias_pct": 100 * (np.exp(g.log_err.median()) - 1)})

    def table(by):
        # metric per seed, then mean and spread across seeds
        per_seed = p.groupby(by + ["variant", "seed"], observed=True).apply(summarize, include_groups=False)
        m = per_seed.groupby(by + ["variant"], observed=True).agg(["mean", "std"])
        out = pd.DataFrame({"sites": m[("sites", "mean")].round(0),
                            "typical_err_%": m[("typical_err_pct", "mean")].round(1),
                            "±sd": m[("typical_err_pct", "std")].round(1),
                            "median_abs_err_%": m[("median_abs_err_pct", "mean")].round(1),
                            "bias_%": m[("bias_pct", "mean")].round(1)})
        return out

    print(f"evaluable sites: {evaluable.sum()} (existing {(evaluable & (d.group == 'existing')).sum()}, "
          f"new {(evaluable & (d.group == 'new')).sum()}); features: {len(feats)}")
    print("training rows per variant (mean):", p.groupby("variant").n_train.mean().round(0).to_dict())
    print("\n== All held-out sites"); print(table([]).to_string())
    print("\n== By site group"); print(table(["group"]).to_string())
    print("\n== By drainage area (km²)"); print(table(["da_bin"]).to_string())

    # Fairness check: judge all variants by the status-quo kind of target (HYDRoSWOT-only), on the
    # existing sites that have one, so B1/B2 get no advantage from sharing the evaluation target's definition.
    p["log_err"] = p.log_err_hs
    p.dropna(subset=["log_err"], inplace=True)
    print("\n== Fairness check: scored against HYDRoSWOT-only targets (existing sites)")
    print(table([]).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:2])
