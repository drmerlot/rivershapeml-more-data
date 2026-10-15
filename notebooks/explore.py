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
    """)
    return


@app.cell
def _(BINS, DER, LABELS, pd):
    targets = pd.read_csv(DER / "retrain_targets.csv", dtype={"site_no": str})
    features = pd.read_csv(DER / "retrain_features.csv", dtype={"site_no": str})
    nwm_q2 = pd.read_csv(DER / "nwm_q2.csv")
    sites = targets.merge(features.drop(columns="comid"), on="site_no", how="left").merge(nwm_q2, on="comid", how="left")
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
    mo.md("""
    ## 2. Target filter explorer

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
    _chart = alt.Chart(_t.sample(min(4000, len(_t)), random_state=0)).mark_circle(size=18, opacity=0.6).encode(
        x=alt.X("ext_all:Q", scale=alt.Scale(type="log"), title="2-yr flood / largest measured flow"),
        y=alt.Y("se_all:Q", scale=alt.Scale(type="log"), title="SE of log-width at 2-yr flood"),
        color=alt.Color("passes:N", scale=alt.Scale(domain=[True, False], range=[C1, C2]), legend=alt.Legend(title="Passes")),
        tooltip=["site_no", "group", "n_all", "r2_all", "se_all", "ext_all"]).properties(width=420, height=300)
    mo.hstack([_chart, mo.vstack([mo.md("**Sites by group**"), _counts, mo.md("**vs the paper's R² filter**"), _vs_r2])])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. NWM vs gage 2-year flood
    """)
    return


@app.cell
def _(LABELS, mo):
    da_pick = mo.ui.multiselect(options=LABELS, value=LABELS, label="Drainage area (km²)")
    da_pick
    return (da_pick,)


@app.cell
def _(C1, alt, da_pick, mo, np, pd, sites):
    _s = sites.dropna(subset=["q2", "nwm_q2"])
    _s = _s[(_s.nwm_q2 > 0) & _s.da_bin.astype(str).isin(da_pick.value)]
    _s = _s.assign(ratio=_s.nwm_q2 / _s.q2)
    _lim = [max(1, _s[["q2", "nwm_q2"]].min().min()), _s[["q2", "nwm_q2"]].max().max()]
    _pts = alt.Chart(_s).mark_circle(size=16, opacity=0.5, color=C1).encode(
        x=alt.X("q2:Q", scale=alt.Scale(type="log"), title="Gage 2-yr flood (cfs)"),
        y=alt.Y("nwm_q2:Q", scale=alt.Scale(type="log"), title="NWM v2.1 2-yr flood (cfs)"),
        tooltip=["site_no", "da_bin", "q2", "nwm_q2", "ratio"])
    _one = alt.Chart(pd.DataFrame({"x": _lim, "y": _lim})).mark_line(color="#52514e", strokeDash=[4, 3]).encode(x="x", y="y")
    _tbl = _s.groupby("da_bin", observed=True).ratio.agg(
        sites="size", median="median", p10=lambda r: r.quantile(.1), p90=lambda r: r.quantile(.9)).round(2)
    mo.hstack([(_pts + _one).properties(width=420, height=360, title="Dashed line = perfect agreement"),
               mo.vstack([mo.md("**NWM ÷ gage, by drainage area**"), _tbl,
                          mo.md(f"log R² = {np.corrcoef(np.log(_s.q2), np.log(_s.nwm_q2))[0, 1] ** 2:.2f}")])])
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
    run_pick = mo.ui.dropdown(run_names, value="nwm_default" if "nwm_default" in run_names else run_names[0], label="Run")
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
    _all = pd.concat([_m, _h], ignore_index=True)
    _pts = alt.Chart(_all).mark_circle(size=40, opacity=0.75).encode(
        x=alt.X("channel_flow:Q", scale=alt.Scale(type="log"), title="Discharge (cfs)"),
        y=alt.Y("channel_width:Q", scale=alt.Scale(type="log"), title="Width (ft)"),
        color=alt.Color("source:N", scale=alt.Scale(domain=["other USGS", "HYDRoSWOT"], range=[C1, C2])),
        tooltip=["time", "channel_flow", "channel_width", "measurement_type", "source"])
    _rules = pd.DataFrame({"q": [_row.q2, _row.nwm_q2], "label": ["2-yr flood (gage)", "2-yr flood (NWM)"]}).dropna()
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


if __name__ == "__main__":
    app.run()
