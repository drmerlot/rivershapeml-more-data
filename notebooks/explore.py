import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="RiverShapeML data explorer")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    alt.data_transformers.disable_max_rows()
    ROOT = Path(__file__).resolve().parents[1]
    DER, RAW = ROOT / "data/derived", ROOT / "data/raw"
    # categorical slots 1-3 (validated all-pairs for scatter); text stays in neutral ink
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    VARIANT_COLORS = alt.Scale(domain=["A_status_quo", "B1_better_targets", "B2_plus_new_sites"], range=[C1, C2, C3])
    BINS, LABELS = [0, 10, 100, 1000, 1e4, 1e7], ["<10", "10-100", "100-1k", "1k-10k", ">10k"]
    return (
        BINS,
        C1,
        C2,
        DER,
        LABELS,
        RAW,
        VARIANT_COLORS,
        alt,
        json,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md("""
    # RiverShapeML: more training data?

    Explore the targets, the NWM vs gage floods, and the retrain runs. Everything reads
    `data/derived/` (run `./get_data.sh` and `./run_all.sh` first).

    ### Names used here

    | Name | Definition |
    |:--|:--|
    | **Expanded training set** | RiverShapeML's training gages plus additional USGS gages, with targets built from all USGS channel measurements since 2000 (every measurement method, not only ADCP) |
    | **Current model** | RiverShapeML as deployed by NOAA OWP ([NOAA-OWP/predict-riverML](https://github.com/NOAA-OWP/predict-riverML)), run on its published inputs |
    | **2024 release** | RiverShapeML outputs published on HydroShare in 2024 (an earlier model generation; the version history between it and the current model isn't public) |
    | **Expanded-set retrain** | the current model's features and settings, retrained on the expanded training set |
    | **ML-derived channel width and depth, expanded-set retrain** | the expanded-set retrain's predictions: bankfull width and depth with 90% prediction intervals for ~2.65M NHDPlus reaches. "ML-derived channel width and depth" is the RiverShapeML product's published name (HydroShare); the current model and the 2024 release are versions of it |
    | **Bieger regional curve** | Bieger et al. (2015) field-based regional equations for bankfull width and depth, applied per reach by USGS |
    """)
    return


@app.cell
def _(DER, mo):
    mo.vstack([mo.md("### Process overview"),
               mo.Html((DER.parents[1] / "docs/process_overview.svg").read_text()
                       .replace('width="680" height="708"', 'width="100%" style="max-width:680px;height:auto"'))])
    return


@app.cell
def _(DER, json, mo):
    import sys as _sys2
    _sys2.path.insert(0, str(DER.parents[1] / "src"))
    import yaml as _yaml
    import data_lineage as dl
    lineage = _yaml.safe_load((DER.parents[1] / "docs/data_lineage.yaml").read_text())
    _cp = DER.parents[1] / "docs/data_lineage_counts.json"
    lineage_counts = json.loads(_cp.read_text()) if _cp.exists() else {}
    mo.vstack([
        mo.md("""
    ### Data lineage

    Every table in the final pipeline, where it came from (source, query, version, download date) and the math that
    connects them. Gray = table (row count), dashed = table from another chapter, teal = operation. Generated from
    `docs/data_lineage.yaml`; `python src/data_lineage.py` checks the spec against the code and files.
    """),
        mo.mermaid(dl.mermaid(lineage, lineage_counts)),
    ])
    return dl, lineage, lineage_counts


@app.cell
def _(lineage, mo):
    chapter_pick = mo.ui.dropdown({f"{c}. {t}": c for c, t in lineage["chapters"].items()},
                                  value=f"C. {lineage['chapters']['C']}", label="Chapter")
    chapter_pick
    return (chapter_pick,)


@app.cell
def _(chapter_pick, dl, lineage, lineage_counts, mo):
    _c = chapter_pick.value
    _T, _O = lineage["tables"], lineage["operations"]
    _tables = {f"{t['title']}": mo.md(dl.table_card(k, t, lineage_counts.get(k))) for k, t in _T.items() if t["chapter"] == _c}
    _ops = {f"{o['title']}": mo.md(dl.op_card(k, o, _T)) for k, o in _O.items() if o["chapter"] == _c}
    mo.vstack([mo.mermaid(dl.mermaid(lineage, lineage_counts, _c)),
               mo.md("**Tables**"), mo.accordion(_tables),
               *([mo.md("**Operations**"), mo.accordion(_ops)] if _ops else [])])
    return


@app.cell
def _(DER, RAW, mo, np, pd):
    _site, _comid = "08158200", 5781217
    _sl = pd.read_csv(DER / "retrain_sites.csv", dtype={"site_no": str}).query("site_no == @_site").iloc[0]
    _ai = pd.read_parquet(DER / "ahg_input.parquet", filters=[("site_no", "==", _site)])
    _fit = pd.read_parquet(DER / "ahg_fits.parquet").query("site_no == @_site and set == 'all'").iloc[0]
    _t = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str}).query("site_no == @_site").iloc[0]
    _runs = {t: pd.read_parquet(DER / f"runs/riverml_{t}/predictions.parquet").query("site_no == @_site")
             for t in ["TW_bf", "Y_bf"]}
    _err = {t: (100 * (np.exp(r.groupby("variant").log_err.mean()) - 1)).round(0).to_dict() for t, r in _runs.items()}
    _p = pd.read_parquet(DER / "product_compare.parquet").query("comid == @_comid").iloc[0]
    _n_raw = sum(len(pd.read_parquet(f, columns=["monitoring_location_id"]).query("monitoring_location_id == 'USGS-08158200'"))
                 for f in sorted((RAW / "channel_meas").glob("page_*.parquet")))
    _rows = [
        ("A", "USGS channel measurements", f"{_n_raw} records for USGS-08158200 (all years)"),
        ("B", "Site list", f"group = {_sl.group}, reach (COMID) = {int(_sl.comid)}"),
        ("C", "AHG input", f"{(_ai.set == 'all').sum()} measurements since 2000 after sanity rules (+ {(_ai.set == 'hs').sum()} HYDRoSWOT, 2000-2015)"),
        ("C", "AHG fits", f"W = {_fit.TW_coef:.3f}·Q^{_fit.TW_exp:.3f}, D = {_fit.Y_coef:.3f}·Q^{_fit.Y_exp:.3f}, V = {_fit.V_coef:.3f}·Q^{_fit.V_exp:.3f} "
                          f"(m, m³/s; {int(_fit.n_used)} of {int(_fit.n_in)} used; exponents sum {_fit.TW_exp + _fit.Y_exp + _fit.V_exp:.3f})"),
        ("C", "Targets", f"Q₂ = {_t.q2:,.0f} ft³/s → bankfull width {_t.tw_all * 0.3048:.2f} m, depth {_t.td_all * 0.3048:.2f} m "
                         f"(SE {_t.se_all:.3f}; Q₂ / max Q = {_t.ext_all:.2f})"),
        ("E", "Held-out predictions", f"mean error, width: {_err['TW_bf']}; depth: {_err['Y_bf']} (%)"),
        ("F", "Expanded-set retrain", f"width {_p.ours_tw_bf_m:.1f} m (90% PI {_p.ours_tw_bf_m_p05:.1f}–{_p.ours_tw_bf_m_p95:.1f}), "
                                      f"depth {_p.ours_y_bf_m:.2f} m (90% PI {_p.ours_y_bf_m_p05:.2f}–{_p.ours_y_bf_m_p95:.2f})"),
        ("F", "Comparison", f"current model {_p.deployed_tw_bf_m:.1f} m / {_p.deployed_y_bf_m:.2f} m · 2024 release "
                            f"{_p.hydroshare_tw_bf_m:.1f} m / {_p.hydroshare_y_bf_m:.2f} m · Bieger {_p.bieger_tw_bf_m:.1f} m / {_p.bieger_y_bf_m:.2f} m"),
    ]
    mo.accordion({"Follow one site through every table: Walnut Ck at Dessau Rd, Austin (USGS 08158200)":
                  pd.DataFrame(_rows, columns=["chapter", "table", "Walnut Creek"])})
    return


@app.cell
def _(BINS, DER, LABELS, RAW, pd):
    targets = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    _da = pd.read_parquet(RAW / "enhd/enhd_nhdplusatts.parquet", columns=["comid", "totdasqkm"])
    sites = targets.merge(_da, on="comid", how="left")
    sites["da_bin"] = pd.cut(sites.totdasqkm, BINS, labels=LABELS)
    return sites, targets


@app.cell
def _(mo):
    mo.md("""
    ## 1. The opening example: Shoal Ck at Silverway Dr (USGS 08156675)
    """)
    return


@app.cell
def _(C1, C2, RAW, alt, json, pd):
    _cm = pd.DataFrame([f["properties"] for f in json.loads(
        (RAW / "usgs_channel-measurements_08156675.json").read_text())["features"]])
    _cm["flow_cfs"] = pd.to_numeric(_cm.channel_flow, errors="coerce")
    _cm["width_ft"] = pd.to_numeric(_cm.channel_width, errors="coerce")
    _cm = _cm[(_cm.flow_cfs > 0) & (_cm.width_ft > 0)]
    _hs = pd.read_parquet(RAW / "repo/RiverShapeML/data/adcp.parquet")
    _hs = _hs[_hs.site_no == "08156675"]
    _cm["in HYDRoSWOT"] = _cm.flow_cfs.round(0).isin(_hs.meas_q_va.round(0)) & _cm.measurement_type.eq("MannedMovingBoat")
    _cm["date"] = _cm.time.str[:10]
    _cm = _cm[["date", "flow_cfs", "width_ft", "measurement_type", "in HYDRoSWOT"]]
    _pts = alt.Chart(_cm).mark_circle(size=70, opacity=0.85, stroke="white", strokeWidth=1).encode(
        x=alt.X("flow_cfs:Q", scale=alt.Scale(type="log"), title="Discharge (cfs)"),
        y=alt.Y("width_ft:Q", scale=alt.Scale(type="log"), title="Width (ft)"),
        color=alt.Color("in HYDRoSWOT:N", scale=alt.Scale(domain=[False, True], range=[C1, C2]),
                        legend=alt.Legend(title="In HYDRoSWOT")),
        tooltip=["date", "flow_cfs", "width_ft", "measurement_type"])
    _q2 = alt.Chart(pd.DataFrame({"q": [2700], "label": ["2-yr flood (gage)"]})).mark_rule(
        strokeDash=[4, 3], color="#52514e").encode(x="q:Q", tooltip=["label", "q"])
    (_pts + _q2).properties(width=620, height=320,
                            title="68 USGS width measurements; RiverShapeML's training data has 2")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The model: f(x) = y

    **One row = one USGS gage site** (its NHDPlus reach). Two separate models: bankfull **width** and bankfull **depth**.

    **y (target), built from field measurements.** For each site: take all USGS channel measurements since 2000
    (flow, width, mean depth = cross-section area ÷ width, velocity = flow ÷ area), apply RiverShapeML's sanity rules
    (width < 4 km, depth < 65 m, flow within the gage's recorded range), and fit width, depth and velocity against
    flow **jointly with AHGestimation** (Johnson et al., NOAA OWP; the package RiverShapeML used), so the three power
    laws conserve mass. Sites where depth falls as flow rises are dropped. Then read the curves at the site's
    **2-year flood**. The 2-year flood is the flow with a 50% chance of being exceeded
    in any year; on the annual-maximum basis that is, by definition, the **median of the annual peak flows**, so we
    estimate it as the median of the gage's USGS annual peaks (≥ 10 years). Converted to meters and
    modeled as log(1 + y). A site is kept only if that value is precise (standard error ≤ ~±20%) and the 2-year flood
    is within 3× the largest measured flow.

    **x (features): RiverShapeML's own 30 deployment inputs**, engineered exactly as in
    NOAA-OWP/predict-riverML, available for every reach:

    | Group | Features |
    |:--|:--|
    | Flow and stream power | NWM 2-year flood (log), flow per unit drainage area, stream power (NWM max flow × slope), subsurface runoff and its ratio to flood flow |
    | Size and network | drainage area (log), arbolate sum, stream order, flowline type (3 indicators) |
    | Slope and form | slope, slope², sinuosity, sinuosity ÷ slope, elevation drop, Manning's roughness |
    | Watershed, soils, geology | baseflow index, depth to water table, depth to bedrock, rock resistance and erodibility indices |
    | Vegetation | leaf area index, vegetation and bank stability indices |
    | Location | projected x / y, distance and angle from the CONUS center |
    | Prior estimate | `suggested_width_m` (a width estimate included in the deployment inputs) |

    **f (model): XGBoost regression with RiverShapeML's deployed settings**, unchanged: 9,765 trees (width) or
    11,000 (depth), learning rate 0.01, depth 9, 80% row and 70% column sampling. Squared error on log(1 + y).

    **Data setup.** Sites = RiverShapeML's existing training gages plus new USGS gages that pass the same rules.
    Split **by reach** into 70% train / 15% validation / 15% test, so no reach appears in two sets. The model is fit
    on train only; validation and test are never seen during fitting.
    """)
    return


@app.cell
def _(DER, json, mo, pd):
    _rows = []
    for _t, _name in [("TW_bf", "width"), ("Y_bf", "depth")]:
        _s = json.loads((DER / f"training/{_t}/setup.json").read_text())
        for _split in ["train", "valid", "test"]:
            _g = _s["groups_by_split"].get(_split, {})
            _rows.append({"model": _name, "split": _split, "sites": _s["sites"][_split],
                          "existing sites": _g.get("existing", 0), "new sites": _g.get("new", 0)})
    mo.vstack([mo.md("**Sites per split**"), pd.DataFrame(_rows)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Major assumptions

    | # | Assumption | Checked? |
    |:--|:--|:--|
    | 1 | **"Bankfull" = width/depth at the 2-year flood** (RiverShapeML's definition, kept for comparability). | Definition. Bieger's field-surveyed curves run ~1.5–2× lower. |
    | 2 | **2-year flood estimated as the median of a gage's annual peaks** (≥ 10 years). On the annual-maximum basis the 2-year flood *is* the median annual flood (as in the UK Flood Estimation Handbook's QMED); the median is a simple estimate of it, where RiverShapeML fits flood distributions. | Within ~3% (median) of RiverShapeML's fitted values at ~5,000 shared sites; peaks span the full record, so changes in flood regime are averaged in. |
    | 3 | **One power law per site** for each of width, depth and velocity vs flow (fitted jointly with AHGestimation), readable up to 3× the largest measured flow. | Earlier plain-fit results held with 2× and 5× limits; AHG and plain-fit targets compared below. |
    | 4 | **Measurements near a gage represent its reach.** | Spots within ~200 ft differ (wading ~33% shallower than bridge), treated as real variability. |
    | 5 | **All measurement methods are equally valid.** | Yes: same deployment, ADCP vs other meters agree within 0–2%. |
    | 6 | **Channels are stable since 2000.** | Not tested. |
    | 7 | **The precision filter keeps trustworthy targets.** | Conclusions stable across three filter settings. |
    | 8 | **RiverShapeML's deployment inputs are correct and identical** for gaged and ungaged reaches. | Reproduced the deployed outputs exactly; input accuracy not checked. |
    | 9 | **Each gage is linked to the right reach** (USGS NLDI). | Not checked site by site; unlinked gages dropped. |
    | 10 | **RiverShapeML's model settings suit the larger training set** (not retuned). | Learning curves below: converges, no validation overfitting. |
    | 11 | **Splitting by reach prevents leakage.** | By design; adjacent reaches could still be correlated. |
    | 12 | **Held-out gaged sites represent the reaches we predict.** | **No: the main limitation.** Gages favor perennial, larger streams; 44% of reaches are < 5 km² with almost no training sites. |
    | 13 | **Ungaged reaches carry the same uncertainty as gaged reaches of similar size**, so prediction intervals transfer. | ~90% coverage on held-out gaged sites (site split; ~88% for < 10 km²); unverifiable for ungaged reaches. |
    """)
    return


@app.cell
def _(C1, C2, C3, DER, alt, json, mo, pd):
    _charts = []
    for _t, _name in [("TW_bf", "Bankfull width"), ("Y_bf", "Bankfull depth")]:
        _c = pd.read_parquet(DER / f"training/{_t}/curves.parquet")
        _c = _c[(_c["round"] % 10 == 0) | (_c["round"] < 50)]
        _long = _c.melt("round", var_name="set", value_name="rmse")
        _best = json.loads((DER / f"training/{_t}/setup.json").read_text())["best_valid_round"]
        _lines = alt.Chart(_long).mark_line(strokeWidth=2).encode(
            x=alt.X("round:Q", scale=alt.Scale(type="symlog"), title="Boosting round (trees added)"),
            y=alt.Y("rmse:Q", title="RMSE of log(1 + y)"),
            color=alt.Color("set:N", scale=alt.Scale(domain=["train", "valid", "test"], range=[C1, C2, C3]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["set", "round", alt.Tooltip("rmse:Q", format=".4f")])
        _rule = alt.Chart(pd.DataFrame({"round": [_best]})).mark_rule(strokeDash=[4, 3], color="#52514e").encode(
            x="round:Q", tooltip=[alt.Tooltip("round:Q", title="best validation round")])
        _charts.append((_lines + _rule).properties(width=380, height=260, title=f"{_name}: learning curves"))
    mo.vstack([alt.hconcat(*_charts),
               mo.md("Dashed line = round with the lowest validation error. Validation and test flatten after "
                     "~500–1,000 rounds and stay flat: no overfitting on held-out data. Train error keeps falling "
                     "toward zero: the model memorizes its training sites, so training accuracy is never quoted.")])
    return


@app.cell
def _(C1, C2, DER, alt, mo, pd):
    _charts, _tables = [], []
    for _t, _name in [("TW_bf", "width"), ("Y_bf", "depth")]:
        _p = pd.read_parquet(DER / f"training/{_t}/predictions.parquet")
        _test = _p[_p.split == "test"][["site_no", "group", "observed_m", "predicted_m"]].round(3)
        _lo, _hi = _test[["observed_m", "predicted_m"]].min().min(), _test[["observed_m", "predicted_m"]].max().max()
        _pts = alt.Chart(_test).mark_circle(size=22, opacity=0.6).encode(
            x=alt.X("observed_m:Q", scale=alt.Scale(type="log"), title=f"Target {_name} (m)"),
            y=alt.Y("predicted_m:Q", scale=alt.Scale(type="log"), title=f"Predicted {_name} (m)"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["existing", "new"], range=[C1, C2]),
                            legend=alt.Legend(title="Site", orient="top")),
            tooltip=["site_no", "group", "observed_m", "predicted_m"])
        _one = alt.Chart(pd.DataFrame({"v": [_lo, _hi]})).mark_line(color="#52514e", strokeDash=[4, 3]).encode(x="v:Q", y="v:Q")
        _charts.append((_pts + _one).properties(width=380, height=320, title=f"Test set: {_name} (dashed = perfect)"))
        _m = pd.read_csv(DER / f"training/{_t}/metrics.csv", index_col=0).round(3)
        _m.insert(0, "model", _name)
        _tables.append(_m.reset_index(names="split"))
    mo.vstack([alt.hconcat(*_charts), mo.md("**Error measures by split** (see `src/metrics.py`)"), pd.concat(_tables, ignore_index=True)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Target filter explorer

    A site gets a bankfull-width target if the fitted width at its 2-year flood is precise enough
    (standard error of log-width) and the 2-year flood isn't too far past the largest measured flow.
    """)
    return


@app.cell
def _(mo):
    se_slider = mo.ui.slider(0.05, 0.5, step=0.05, value=0.2, label="Max SE of log-width", show_value=True)
    ext_slider = mo.ui.slider(1.0, 10.0, step=0.5, value=3.0, label="Max 2-yr flood / largest measured flow", show_value=True)
    mo.hstack([se_slider, ext_slider], justify="start", gap=2)
    return ext_slider, se_slider


@app.cell
def _(C1, C2, alt, ext_slider, mo, pd, se_slider, targets):
    _t = targets.dropna(subset=["se_all", "ext_all"]).copy()
    _t["passes"] = (_t.se_all <= se_slider.value) & (_t.ext_all <= ext_slider.value)
    _t["passes R² ≥ 0.6"] = _t.r2_all >= 0.6
    _counts = pd.crosstab(_t.group, _t.passes).rename(columns={True: "pass", False: "fail"})
    _vs_r2 = pd.crosstab(_t["passes R² ≥ 0.6"], _t.passes.rename("passes precision filter"))
    _plot = _t.sample(min(4000, len(_t)), random_state=0)[["site_no", "group", "n_all", "r2_all", "se_all", "ext_all", "passes"]].round(3)
    _chart = alt.Chart(_plot).mark_circle(size=18, opacity=0.6).encode(
        x=alt.X("ext_all:Q", scale=alt.Scale(type="log"), title="2-yr flood / largest measured flow"),
        y=alt.Y("se_all:Q", scale=alt.Scale(type="log"), title="SE of log-width at 2-yr flood"),
        color=alt.Color("passes:N", scale=alt.Scale(domain=[True, False], range=[C1, C2]), legend=alt.Legend(title="Passes")),
        tooltip=["site_no", "group", "n_all", "r2_all", "se_all", "ext_all"]).properties(width=420, height=300)
    mo.hstack([_chart, mo.vstack([mo.md("**Sites by group**"), _counts, mo.md("**vs the paper's R² filter**"), _vs_r2])])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Training runs

    Each run trains **A** (existing sites, HYDRoSWOT-only targets), **B1** (existing sites, all-measurement
    targets) and **B2** (B1 + new sites), scored on the same held-out sites. **B1 vs B2 is the clean
    comparison** (same targets; only the new sites differ).

    Each prediction is compared to its target as a ratio, *predicted ÷ target*:
    **median absolute error** = median |ratio − 1| · **within ±25%** = share with ratio 0.75–1.25 ·
    **within 2×** = share with ratio 0.5–2 · **median bias** = median ratio − 1 · **log R²** = the paper's metric.
    Definitions live in `src/metrics.py`.
    """)
    return


@app.cell
def _(DER, mo):
    run_names = sorted(p.name for p in (DER / "runs").iterdir() if (p / "predictions.parquet").exists())
    mo.stop(not run_names, mo.md("_No finished runs yet: run `python src/retrain_experiment.py`, then re-run this cell._"))
    run_pick = mo.ui.dropdown(run_names, value="riverml_TW_bf" if "riverml_TW_bf" in run_names else run_names[0], label="Run")
    variant_pick = mo.ui.multiselect(["A_status_quo", "B1_better_targets", "B2_plus_new_sites"],
                                     value=["A_status_quo", "B1_better_targets", "B2_plus_new_sites"], label="Variants")
    mo.hstack([run_pick, variant_pick], justify="start", gap=2)
    return run_names, run_pick, variant_pick


@app.cell
def _(DER, json, np, pd):
    import sys as _sys
    _sys.path.insert(0, str(DER.parents[1] / "src"))
    from metrics import METRICS, summarize

    def load_preds(run_dir):
        """Predictions for a run; older runs lack log_true, so rebuild it from the run's targets file."""
        p = pd.read_parquet(run_dir / "predictions.parquet")
        if "log_true" not in p:
            cfg = json.loads((run_dir / "config.json").read_text())
            t = pd.read_csv(DER / cfg["targets"], dtype={"site_no": str}).drop_duplicates("site_no")
            p = p.merge(t[["site_no", "tw_all"]], on="site_no", how="left").assign(log_true=lambda d: np.log(d.tw_all))
        return p
    return METRICS, load_preds, summarize


@app.cell
def _(DER, json, load_preds, mo, run_pick):
    run_dir = DER / "runs" / run_pick.value
    preds = load_preds(run_dir)
    _cfg = json.loads((run_dir / "config.json").read_text())
    mo.accordion({"Run config": mo.md(f"```json\n{json.dumps(_cfg, indent=2)[:3000]}\n```")})
    return (preds,)


@app.cell
def _(LABELS, METRICS, VARIANT_COLORS, alt, mo, preds, summarize, variant_pick):
    _p = preds[preds.variant.isin(variant_pick.value)]
    _by_da = summarize(_p, ["da_bin"])
    _tip = ["da_bin", "variant", "sites"] + list(METRICS)

    def _panel(metric, title, zero_rule=False):
        bars = alt.Chart(_by_da).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("da_bin:N", sort=LABELS, title="Drainage area (km²)"),
            xOffset=alt.XOffset("variant:N"),
            y=alt.Y(f"{metric}:Q", title=None),
            color=alt.Color("variant:N", scale=VARIANT_COLORS, legend=alt.Legend(title="Variant", orient="top")),
            tooltip=_tip).properties(width=300, height=200, title=title)
        if zero_rule:
            bars = bars + alt.Chart().mark_rule(color="#52514e").encode(y=alt.datum(0))
        return bars

    _grid = alt.vconcat(
        alt.hconcat(_panel("median_abs_err_%", "Median absolute error (%) · lower is better"),
                    _panel("bias_%", "Median bias (%) · closer to 0 is better", zero_rule=True)),
        alt.hconcat(_panel("within_25_%", "Within ±25% (% of sites) · higher is better"),
                    _panel("within_2x_%", "Within a factor of 2 (% of sites) · higher is better")),
        alt.hconcat(_panel("log_r2", "log R² · higher is better"),
                    _panel("rms_factor_%", "RMS error factor (%) · old 'typical error'")),
    ).resolve_scale(color="shared")
    mo.vstack([_grid,
               mo.md("**All held-out sites**"), summarize(_p, []),
               mo.md("**By site group**"), summarize(_p, ["group"]),
               mo.md("**By drainage area**"), _by_da])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Compare runs
    """)
    return


@app.cell
def _(METRICS, mo):
    metric_pick = mo.ui.dropdown({v: k for k, v in METRICS.items()}, value="Median absolute error (%)", label="Measure")
    metric_pick
    return (metric_pick,)


@app.cell
def _(DER, load_preds, metric_pick, pd, run_names, summarize):
    _rows = []
    for _r in run_names:
        _p = load_preds(DER / "runs" / _r)
        for _label, _sub in [("all", _p), ("<10 km²", _p[_p.da_bin == "<10"]),
                             ("10-100 km²", _p[_p.da_bin == "10-100"]), ("new sites", _p[_p.group == "new"])]:
            _s = summarize(_sub, [])
            _s.insert(0, "subset", _label)
            _s.insert(0, "run", _r)
            _rows.append(_s)
    pd.concat(_rows).pivot_table(index=["run", "subset"], columns="variant", values=metric_pick.value).round(2)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Site explorer
    """)
    return


@app.cell
def _(DER, RAW, pd, targets):
    # all usable width measurements since 2000 for the retrain sites (cached after the first build)
    _cache = DER / "retrain_measurements.parquet"
    if _cache.exists():
        meas = pd.read_parquet(_cache)
    else:
        _ids = set("USGS-" + targets.site_no)
        _cols = ["monitoring_location_id", "time", "channel_flow", "channel_width", "measurement_type"]
        _cm = pd.concat(pd.read_parquet(p, columns=_cols) for p in sorted((RAW / "channel_meas").glob("page_*.parquet")))
        _cm = _cm[_cm.monitoring_location_id.isin(_ids)]
        for _c in ["channel_flow", "channel_width"]:
            _cm[_c] = pd.to_numeric(_cm[_c], errors="coerce")
        _cm = _cm[(_cm.channel_flow > 0) & (_cm.channel_width > 0) & (_cm.time.str[:4] >= "2000")]
        meas = _cm.assign(site_no=_cm.monitoring_location_id.str[5:]).drop(columns="monitoring_location_id")
        meas.to_parquet(_cache, index=False)
    hydroswot = pd.read_parquet(RAW / "repo/RiverShapeML/data/adcp.parquet",
                                columns=["site_no", "site_visit_start_dt", "meas_q_va", "stream_wdth_va"])
    return hydroswot, meas


@app.cell
def _(mo, sites):
    _opts = {f"{r.site_no} · {r.group} · {r.totdasqkm:.0f} km²": r.site_no
             for r in sites.dropna(subset=["tw_all"]).sort_values("totdasqkm").itertuples()}
    site_pick = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="Site (sorted by drainage area)", searchable=True)
    site_pick
    return (site_pick,)


@app.cell
def _(C1, C2, alt, hydroswot, meas, mo, np, pd, preds, site_pick, sites):
    _sid = site_pick.value
    _row = sites[sites.site_no == _sid].iloc[0]
    _m = meas[meas.site_no == _sid].assign(source="other USGS")
    _h = hydroswot[(hydroswot.site_no == _sid) & (hydroswot.stream_wdth_va > 0)]
    _h = pd.DataFrame({"channel_flow": _h.meas_q_va, "channel_width": _h.stream_wdth_va,
                       "time": _h.site_visit_start_dt.astype(str), "source": "HYDRoSWOT"})
    _all = pd.concat([_m, _h], ignore_index=True)[["time", "channel_flow", "channel_width", "measurement_type", "source"]]
    _pts = alt.Chart(_all).mark_circle(size=40, opacity=0.75).encode(
        x=alt.X("channel_flow:Q", scale=alt.Scale(type="log"), title="Discharge (cfs)"),
        y=alt.Y("channel_width:Q", scale=alt.Scale(type="log"), title="Width (ft)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=["other USGS", "HYDRoSWOT"], range=[C1, C2])),
        tooltip=["time", "channel_flow", "channel_width", "measurement_type", "source"])
    _rules = pd.DataFrame({"q": [_row.q2], "label": ["2-yr flood (gage)"]}).dropna()
    _r = alt.Chart(_rules).mark_rule(strokeDash=[4, 3]).encode(
        x="q:Q", strokeDash=alt.StrokeDash("label:N", legend=alt.Legend(title=None)), tooltip=["label", "q"])
    _t = alt.Chart(pd.DataFrame({"q": [_row.q2], "w": [_row.tw_all]}).dropna()).mark_point(
        shape="diamond", size=140, filled=True, color="#0b0b0b").encode(x="q:Q", y="w:Q")
    _p = preds[preds.site_no == _sid].groupby("variant").log_err.median()
    _info = pd.DataFrame({"value": {
        "group": _row.group, "drainage area (km²)": round(_row.totdasqkm, 1),
        "target width, all data (ft)": round(_row.tw_all, 1) if np.isfinite(_row.tw_all) else None,
        "target width, HYDRoSWOT only (ft)": round(_row.tw_hs, 1) if np.isfinite(_row.tw_hs) else None,
        **{f"{k} error (%)": round(100 * (np.exp(v) - 1), 1) for k, v in _p.items()}}})
    mo.hstack([(_pts + _r + _t).properties(width=440, height=320, title=f"USGS {_sid} (◆ = bankfull target)"), _info])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. The re-predicted product

    Four versions of bankfull width and depth for ~2.6M reaches:
    **ours** (the expanded-set retrain, with 90% prediction intervals) · **deployed** (RiverShapeML's current models,
    NOAA-OWP/predict-riverML) · **HydroShare 2024** (the published release) · **Bieger et al. (2015)** regional curves
    applied per reach by USGS (a field-survey-based reference with no machine learning; dashed line).

    Note the definitions differ: the ML products target water-surface width at the 2-year flood at gage sections;
    Bieger's curves come from field-surveyed bankfull (top of bank) at stable reaches.
    """)
    return


@app.cell
def _(DER, np, pd):
    prod = pd.read_parquet(DER / "product_compare.parquet")
    _edges = np.logspace(-1, 6.5, 31)
    prod["da_log_bin"] = pd.cut(prod.totdasqkm, _edges)
    prod["da_mid"] = prod.da_log_bin.map(lambda iv: float(np.sqrt(iv.left * iv.right)) if pd.notna(iv) else np.nan).astype(float)
    PRODUCTS = {"deployed": "Deployed (current)", "hydroshare": "HydroShare 2024", "ours": "Ours (+ new sites)"}
    return PRODUCTS, prod


@app.cell
def _(mo):
    dim_pick = mo.ui.dropdown({"Bankfull width (m)": "tw_bf_m", "Bankfull depth (m)": "y_bf_m"},
                              value="Bankfull width (m)", label="Quantity")
    dim_pick
    return (dim_pick,)


@app.cell
def _(C1, C2, C3, PRODUCTS, alt, dim_pick, mo, pd, prod):
    _v = dim_pick.value
    _g = prod.groupby("da_mid", observed=True)
    _lines = pd.concat([pd.DataFrame({"da": _g[f"{k}_{_v}"].median().index, "value": _g[f"{k}_{_v}"].median().values,
                                      "product": name, "reaches": _g.size().values}) for k, name in PRODUCTS.items()])
    _band = pd.DataFrame({"da": _g[f"ours_{_v}_p05"].median().index, "lo": _g[f"ours_{_v}_p05"].median().values,
                          "hi": _g[f"ours_{_v}_p95"].median().values})
    _bieger = pd.DataFrame({"da": _g[f"bieger_{_v}"].median().index, "value": _g[f"bieger_{_v}"].median().values,
                            "product": "Bieger et al. 2015 (regional curve)"})
    _x = alt.X("da:Q", scale=alt.Scale(type="log"), title="Drainage area (km²)")
    _y = alt.Y("value:Q", scale=alt.Scale(type="log"), title=dim_pick.selected_key)
    _colors = alt.Scale(domain=list(PRODUCTS.values()), range=[C1, C2, C3])
    _band_ch = alt.Chart(_band).mark_area(opacity=0.18, color=C3).encode(x=_x, y="lo:Q", y2="hi:Q")
    _line_ch = alt.Chart(_lines).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30)).encode(
        x=_x, y=_y, color=alt.Color("product:N", scale=_colors, legend=alt.Legend(title=None, orient="top")),
        tooltip=["product", alt.Tooltip("da:Q", format=",.1f"), alt.Tooltip("value:Q", format=".2f"), "reaches"])
    _bieger_ch = alt.Chart(_bieger).mark_line(strokeDash=[5, 4], color="#52514e", strokeWidth=2).encode(
        x=_x, y=_y, tooltip=["product", alt.Tooltip("value:Q", format=".2f")])
    mo.vstack([
        (_band_ch + _line_ch + _bieger_ch).properties(
            width=640, height=380, title="Median by drainage area · shaded = typical 90% prediction interval of ours · dashed = Bieger"),
    ])
    return


@app.cell
def _(C1, C2, C3, PRODUCTS, alt, dim_pick, np, pd, prod):
    _v = dim_pick.value
    _edges = np.linspace(-1, 3.5, 46)
    _rows = []
    for _k, _name in {**PRODUCTS, "bieger": "Bieger et al. 2015 (regional curve)"}.items():
        _vals = np.log10(prod[f"{_k}_{_v}"].where(lambda s: s > 0).dropna())
        _h, _ = np.histogram(_vals, _edges)
        _rows.append(pd.DataFrame({"value": 10 ** ((_edges[:-1] + _edges[1:]) / 2), "share_%": 100 * _h / _h.sum(), "product": _name}))
    _h = pd.concat(_rows)
    alt.Chart(_h).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("value:Q", scale=alt.Scale(type="log"), title=dim_pick.selected_key),
        y=alt.Y("share_%:Q", title="Share of reaches (%)"),
        color=alt.Color("product:N", scale=alt.Scale(domain=list(PRODUCTS.values()) + ["Bieger et al. 2015 (regional curve)"],
                                                     range=[C1, C2, C3, "#52514e"]), legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.condition(alt.datum.product == "Bieger et al. 2015 (regional curve)", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=["product", alt.Tooltip("value:Q", format=".2f"), alt.Tooltip("share_%:Q", format=".2f")],
    ).properties(width=640, height=260, title="Distribution across all reaches")
    return


@app.cell
def _(mo):
    map_pick = mo.ui.dropdown({"Ours vs deployed": ("ours", "deployed"), "Ours vs Bieger": ("ours", "bieger"),
                               "Deployed vs Bieger": ("deployed", "bieger"), "Deployed vs HydroShare 2024": ("deployed", "hydroshare")},
                              value="Ours vs deployed", label="Map: ratio")
    map_pick
    return (map_pick,)


@app.cell
def _(alt, dim_pick, map_pick, np, prod):
    _a, _b = map_pick.value
    _v = dim_pick.value
    _s = prod.sample(15000, random_state=0)
    _s = _s.assign(log2_ratio=np.log2(_s[f"{_a}_{_v}"] / _s[f"{_b}_{_v}"]).clip(-2, 2).round(2),
                   ratio=(_s[f"{_a}_{_v}"] / _s[f"{_b}_{_v}"]).round(2)).dropna(subset=["log2_ratio", "lat"])
    _s = _s[["comid", "totdasqkm", "lat", "long", "log2_ratio", "ratio"]].astype(float).round(
        {"lat": 2, "long": 2, "totdasqkm": 1})
    _s["comid"] = _s.comid.astype(int)
    alt.Chart(_s).mark_circle(size=6, opacity=0.8).encode(
        longitude="long:Q", latitude="lat:Q",
        color=alt.Color("log2_ratio:Q", scale=alt.Scale(domain=[-2, 0, 2], range=["#2a78d6", "#f0efec", "#e34948"]),
                        legend=alt.Legend(title=f"log₂({_a} ÷ {_b})", orient="right")),
        tooltip=["comid", alt.Tooltip("totdasqkm:Q", format=",.1f"), "ratio"],
    ).project("albersUsa").properties(width=720, height=440,
                                      title=f"{map_pick.selected_key} ({dim_pick.selected_key}), 15k random reaches · blue = first is smaller, red = larger")
    return


@app.cell
def _(DER, PRODUCTS, dim_pick, mo, pd, prod):
    _v = dim_pick.value
    _cols = {name: f"{k}_{_v}" for k, name in {**PRODUCTS, "bieger": "Bieger 2015"}.items()}
    _by_order = pd.DataFrame({name: prod.groupby("streamorde")[c].median() for name, c in _cols.items()}).astype(float).round(2)
    _by_order.insert(0, "reaches", prod.groupby("streamorde").size())
    mo.vstack([mo.md(f"**Median {dim_pick.selected_key} by stream order**"), _by_order.reset_index(),
               mo.accordion({"Reasonableness checks and prediction-interval calibration (product_checks.txt)":
                             mo.md("```\n" + (DER / "product_checks.txt").read_text() + "\n```")})])
    return


@app.cell
def _(mo):
    comid_in = mo.ui.text(value="5781223", label="Look up a reach (COMID)")
    comid_in
    return (comid_in,)


@app.cell
def _(comid_in, mo, pd, prod):
    _r = prod[prod.comid == int(comid_in.value)] if comid_in.value.strip().isdigit() else prod.iloc[0:0]
    mo.stop(_r.empty, mo.md("_No reach with that COMID._"))
    _r = _r.iloc[0]
    mo.vstack([mo.md(f"**{_r.gnis_name or '(unnamed)'}** · COMID {_r.comid} · {_r.totdasqkm:,.1f} km² · order {_r.streamorde:.0f}"),
               pd.DataFrame({"width (m)": [_r.ours_tw_bf_m, _r.deployed_tw_bf_m, _r.hydroshare_tw_bf_m, _r.bieger_tw_bf_m],
                             "depth (m)": [_r.ours_y_bf_m, _r.deployed_y_bf_m, _r.hydroshare_y_bf_m, _r.bieger_y_bf_m]},
                            index=["ours", "deployed", "HydroShare 2024", "Bieger 2015"]).round(2),
               mo.md(f"Ours, 90% prediction interval: width {_r.ours_tw_bf_m_p05:.1f}–{_r.ours_tw_bf_m_p95:.1f} m · "
                     f"depth {_r.ours_y_bf_m_p05:.2f}–{_r.ours_y_bf_m_p95:.2f} m")])
    return


if __name__ == "__main__":
    app.run()
