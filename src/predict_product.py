"""Re-predict the product with the added training sites, with uncertainty, reasonableness checks, and a
comparison to the deployed RiverShapeML model and the 2024 HydroShare release.

1. Final models: their 30 features + their XGBoost settings, trained on ALL sites with targets (variant B2),
   for bankfull width (TW_bf) and depth (Y_bf); predict every reach in ml_inputs.parquet.
2. Uncertainty: 90% prediction intervals from held-out errors by drainage-area class (split-conformal, by class):
   interval = prediction x [P5, P95] of held-out ratio target/prediction in that class, from the
   retrain_riverml B2 runs. Coverage is checked on unseen sites (bounds set on half the sites, tested on the other).
   Assumes ungaged reaches carry the same uncertainty as gaged reaches of similar size.
3. Reasonableness checks on ours and the baseline: ranges, width/depth ratio, downstream consistency on
   the same river (levelpath), width ~ drainage-area scaling, extremes.
4. Comparison: quantiles overall / by drainage area / by stream order; reach-level ratios.

Usage: python src/predict_product.py   (after predict_baseline.py and retrain_riverml.py for both targets)
Output: data/derived/product_ours.parquet, data/derived/product_checks.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_riverml import FT, model
from riverml_pipeline import ID, load_features

ROOT = Path(__file__).resolve().parents[1]
DER, RAW = ROOT / "data/derived", ROOT / "data/raw"
BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]
pd.set_option("display.width", 200)


def conformal_bounds(target):
    """Per drainage-area class: P5/P95 of ln(target/prediction) for B2 held-out predictions."""
    p = pd.read_parquet(DER / f"runs/riverml_{target}/predictions.parquet")
    p = p[p.variant == "B2_plus_new_sites"]
    r = -p.log_err  # ln(target / prediction)
    b = r.groupby(p.da_bin.astype(str)).quantile([0.05, 0.95]).unstack()
    b.columns = ["lo", "hi"]
    # coverage check on unseen sites: set the bounds on a random half of the sites, test on the other half
    # (and vice versa), over 20 random splits
    cov = []
    sites = p.site_no.unique()
    for rep in range(20):
        half = set(np.random.default_rng(rep).choice(sites, len(sites) // 2, replace=False))
        for cal_mask in (p.site_no.isin(half), ~p.site_no.isin(half)):
            cal, ev = p[cal_mask], p[~cal_mask]
            bb = (-cal.log_err).groupby(cal.da_bin.astype(str)).quantile([0.05, 0.95]).unstack()
            lo, hi = ev.da_bin.astype(str).map(bb[0.05]), ev.da_bin.astype(str).map(bb[0.95])
            cov.append(((-ev.log_err >= lo) & (-ev.log_err <= hi)).groupby(ev.da_bin.astype(str)).mean())
    b["coverage_check"] = pd.concat(cov, axis=1).mean(axis=1)
    return b


def train_and_predict(target, ycol):
    t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str}).dropna(subset=[ycol])
    t = t.groupby("comid", as_index=False)[ycol].median()  # one row per reach
    Xs, ids = load_features(target, t.comid.astype(int))
    y = np.log1p(t.set_index("comid")[ycol].reindex(ids).to_numpy() * FT)
    m = model(target, 0).fit(Xs, y)
    X, all_ids = load_features(target)
    pred = np.concatenate([np.abs(np.expm1(m.predict(X.iloc[i:i + 250_000]))) for i in range(0, len(X), 250_000)])
    print(f"{target}: trained on {len(Xs)} reaches; predicted {len(pred)}", flush=True)
    return pd.Series(pred, index=all_ids)


def downstream_drops(w, enhd):
    """Share of reaches whose width is < 50% of the reach directly upstream on the same levelpath."""
    e = enhd[enhd.comid.isin(w.index) & enhd.tocomid.isin(w.index)]
    e = e[e.levelpathi == e.tocomid.map(enhd.set_index("comid").levelpathi)]
    ratio = w.reindex(e.tocomid).to_numpy() / w.reindex(e.comid).to_numpy()
    return float(np.nanmean(ratio < 0.5)), len(e)


def main():
    out = pd.DataFrame({"tw_bf_m": train_and_predict("TW_bf", "tw_all"), "y_bf_m": train_and_predict("Y_bf", "td_all")})
    out.index.name = ID
    enhd = pd.read_parquet(RAW / "enhd/enhd_nhdplusatts.parquet",
                           columns=["comid", "tocomid", "levelpathi", "totdasqkm", "streamorde"])
    attrs = enhd.set_index("comid").reindex(out.index)
    out["da_bin"] = pd.cut(attrs.totdasqkm, BINS, labels=LABELS).astype(str)
    out["streamorde"] = attrs.streamorde
    lines = []

    def log(*a):
        lines.append(" ".join(str(x) for x in a))
        print(*a, flush=True)

    for target, col in [("TW_bf", "tw_bf_m"), ("Y_bf", "y_bf_m")]:
        b = conformal_bounds(target)
        log(f"\n== 90% prediction-interval factors by drainage area, {target} (ln-ratio P5/P95 -> multiplicative)")
        log(pd.DataFrame({"lower_x": np.exp(b.lo), "upper_x": np.exp(b.hi), "coverage_check": b.coverage_check}).round(3).to_string())
        out[col + "_p05"] = out[col] * np.exp(out.da_bin.map(b.lo))
        out[col + "_p95"] = out[col] * np.exp(out.da_bin.map(b.hi))
    out.reset_index().to_parquet(DER / "product_ours.parquet", index=False)

    base = pd.read_parquet(DER / "product_baseline.parquet").set_index(ID)
    pub = (pd.read_parquet(RAW / "rivershapeml/ml_channel_dimension.parquet")
           .groupby("hf_id")[["tw_bf(m)", "y_bf(m)"]].median()
           .rename(columns={"tw_bf(m)": "tw_bf_m", "y_bf(m)": "y_bf_m"}))
    products = {"ours": out, "deployed": base, "hydroshare_2024": pub}

    log("\n== Reasonableness checks")
    for name, p in products.items():
        w, dpt = p.tw_bf_m, p.y_bf_m
        wd = w / dpt
        drop, n = downstream_drops(w.dropna(), enhd)
        log(f"{name:>16}: width<1m {np.mean(w < 1):.2%} | width>1km {np.mean(w > 1000):.3%} | depth<0.1m {np.mean(dpt < 0.1):.2%} | "
            f"depth>20m {np.mean(dpt > 20):.3%} | w/d<2 {np.mean(wd < 2):.2%} | w/d>100 {np.mean(wd > 100):.2%} | "
            f"downstream halving {drop:.2%} of {n:,} links")
    log("\nwidth ~ drainage area scaling (log-log slope, all reaches with DA > 1 km²):")
    da = enhd.set_index("comid").totdasqkm
    for name, p in products.items():
        d = pd.DataFrame({"w": p.tw_bf_m, "a": da.reindex(p.index)}).dropna()
        d = d[(d.a > 1) & (d.w > 0)]
        b1, b0 = np.polyfit(np.log(d.a), np.log(d.w), 1)
        log(f"{name:>16}: W = {np.exp(b0):.2f} * A^{b1:.3f}")

    log("\n== Distributions (quantiles, m)")
    for col in ["tw_bf_m", "y_bf_m"]:
        q = pd.DataFrame({n: p[col].quantile([.05, .25, .5, .75, .95]) for n, p in products.items()}).T.round(2)
        log(f"-- {col}\n{q.to_string()}")
        g = pd.DataFrame({n: p[col].groupby(out.da_bin.reindex(p.index)).median() for n, p in products.items()}).round(2)
        log(f"-- {col} median by drainage area\n{g.reindex(LABELS).to_string()}")
        g = pd.DataFrame({n: p[col].groupby(out.streamorde.reindex(p.index)).median() for n, p in products.items()}).round(2)
        log(f"-- {col} median by stream order\n{g.to_string()}")
        for other in ["deployed", "hydroshare_2024"]:
            r = (out[col] / products[other][col].reindex(out.index)).dropna()
            log(f"   ours / {other}: median {r.median():.3f}, IQR [{r.quantile(.25):.2f}, {r.quantile(.75):.2f}], "
                f"log R² {np.corrcoef(np.log(out[col].reindex(r.index)), np.log(products[other][col].reindex(r.index)))[0, 1] ** 2:.3f}")
    for name, p in products.items():
        log(f"Silverway (5781223) {name}: width {p.tw_bf_m.get(5781223, np.nan):.1f} m, depth {p.y_bf_m.get(5781223, np.nan):.2f} m")
    (DER / "product_checks.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
