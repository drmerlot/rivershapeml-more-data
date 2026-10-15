"""Why do non-HYDRoSWOT depths run shallower? Separate instrument from location.

For each site, fit one log-log depth-vs-flow curve to ALL measurements, then express every
measurement as a % residual from it. Compare residuals between groups within each site
(site-level median per group, then the median across sites):
  - same deployment, different instrument: HYDRoSWOT (ADCP) vs other, within Bridge and within Wading
  - same instrument and deployment, different place: non-HYDRoSWOT wading by distance from the gage
  - wading vs bridge sections at the same flows (non-HYDRoSWOT)

Usage: python src/depth_gap.py   (uses sites cached by method_agreement.py)
"""
import numpy as np
import pandas as pd

from method_agreement import CACHE, DER, REPO, clean, label_hydroswot


def deployment(t):
    if t.startswith("Bridge"):
        return "Bridge"
    if t in ("MannedMovingBoat", "Boat", "RemoteControlledBoat"):
        return "Boat"
    return t


def paired(df, group_col, a, b, label):
    """Within-site difference of median residual, group a minus group b, across sites."""
    g = df.groupby(["site_no", group_col]).resid.median().unstack()
    if a not in g or b not in g:
        return
    d = (g[a] - g[b]).dropna()
    print(f"{label:<55} sites={len(d):>4}  median diff={d.median():+6.1f}%  IQR=[{d.quantile(.25):+.1f}, {d.quantile(.75):+.1f}]")


def wade_vs_bridge_same_flow(o):
    """Per site: fit log(depth) = c0 + c1*log(Q) + c2*[wading] to wading and bridge measurements
    within their shared flow range; exp(c2) - 1 is the wading-vs-bridge depth difference."""
    diffs = []
    for _, g in o.groupby("site_no"):
        w, b = g[g.deploy == "Wading"], g[g.deploy == "Bridge"]
        if len(w) < 3 or len(b) < 3:
            continue
        lo = max(w.channel_flow.min(), b.channel_flow.min())
        hi = min(w.channel_flow.max(), b.channel_flow.max())
        w, b = w[w.channel_flow.between(lo, hi)], b[b.channel_flow.between(lo, hi)]
        if len(w) < 2 or len(b) < 2:
            continue
        x = np.log(np.r_[w.channel_flow, b.channel_flow])
        y = np.log(np.r_[w.depth, b.depth])
        is_wade = np.r_[np.ones(len(w)), np.zeros(len(b))]
        c = np.linalg.lstsq(np.c_[np.ones_like(x), x, is_wade], y, rcond=None)[0]
        diffs.append(100 * (np.exp(c[2]) - 1))
    d = pd.Series(diffs)
    print(f"{'wading minus bridge, same flows':<55} sites={len(d):>4}  median diff={d.median():+6.1f}%  "
          f"IQR=[{d.quantile(.25):+.1f}, {d.quantile(.75):+.1f}]  wading shallower at {(d < 0).mean():.0%} of sites")


def main():
    hs = pd.read_parquet(REPO / "adcp.parquet")
    hs = hs[(hs.meas_q_va > 0) & (hs.stream_wdth_va > 0)].copy()
    hs["date"] = pd.to_datetime(hs.site_visit_start_dt).dt.normalize()
    frames = []
    for p in sorted(CACHE.glob("*.parquet")):
        site = p.stem
        us = clean(pd.read_parquet(p))
        h = hs[hs.site_no == site]
        if us.empty or h.empty or "channel_location_distance" not in us:
            continue
        us = label_hydroswot(us, h)
        us = us[us.depth > 0]
        if len(us) < 10:
            continue
        b, a = np.polyfit(np.log(us.channel_flow), np.log(us.depth), 1)
        us["resid"] = 100 * (np.exp(np.log(us.depth) - (a + b * np.log(us.channel_flow))) - 1)
        us["site_no"] = site
        frames.append(us)
    df = pd.concat(frames)
    df["deploy"] = df.measurement_type.fillna("").map(deployment)
    df["src"] = np.where(df.in_hs, "HYDRoSWOT", "other")
    dist = pd.to_numeric(df.channel_location_distance, errors="coerce")
    df["dist_bin"] = pd.cut(dist, [-1, 50, 200, 1e6], labels=["<=50 ft", "50-200 ft", ">200 ft"])
    print(f"sites: {df.site_no.nunique()}, measurements: {len(df)}; distance recorded: {dist.notna().mean():.0%}\n")

    print("Depth residual vs each site's pooled curve (+ = deeper than curve)")
    print("-- overall")
    paired(df, "src", "other", "HYDRoSWOT", "other minus HYDRoSWOT (all deployments)")
    print("-- same deployment, different instrument (HYDRoSWOT = ADCP)")
    for dep in ["Bridge", "Wading", "Boat", "Cableway"]:
        paired(df[df.deploy == dep], "src", "other", "HYDRoSWOT", f"{dep}: other minus HYDRoSWOT")
    print("-- non-HYDRoSWOT only: different deployment")
    o = df[~df.in_hs]
    paired(o, "deploy", "Wading", "Bridge", "Wading minus Bridge")
    paired(o, "deploy", "Wading", "Cableway", "Wading minus Cableway")
    print("-- non-HYDRoSWOT wading only: distance from gage")
    w = o[o.deploy == "Wading"]
    paired(w, "dist_bin", "50-200 ft", "<=50 ft", "50-200 ft minus <=50 ft")
    paired(w, "dist_bin", ">200 ft", "<=50 ft", ">200 ft minus <=50 ft")
    print("-- non-HYDRoSWOT wading vs bridge at the SAME flows (shared slope over the overlap range)")
    wade_vs_bridge_same_flow(o)
    print("\n-- flow overlap check: median flow (cfs) by group")
    print(df.groupby(["deploy", "src"]).channel_flow.median().unstack().round(1).to_string())
    df[["site_no", "time", "channel_flow", "depth", "resid", "deploy", "src", "dist_bin",
        "channel_location_direction", "channel_material"]].to_csv(DER / "depth_gap_measurements.csv", index=False)


if __name__ == "__main__":
    main()
