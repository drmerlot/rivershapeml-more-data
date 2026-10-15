# More training data for RiverShapeML?

Code behind a short note ([memo](memo/memo_v1.md)) on how many more USGS channel measurements are available
to [RiverShapeML](https://github.com/lynker-spatial/RiverShapeML), and whether they're compatible with the
HYDRoSWOT data it trains on. Everything uses public data and runs on a laptop.

## Built on RiverShapeML

This work stands on RiverShapeML by Arash Modaresi Rad and colleagues (Modaresi Rad et al., 2024,
*JGR: Machine Learning and Computation*). It was only possible because the model's deployment pipeline,
[NOAA-OWP/predict-riverML](https://github.com/NOAA-OWP/predict-riverML), is public: its inputs for every reach,
feature engineering and trained models let me reproduce the deployed bankfull width and depth for all 2.65M
reaches exactly (matching the published output statistics), and then run every experiment here on the model's own
features and settings. The training data, targets and code in
[lynker-spatial/RiverShapeML](https://github.com/lynker-spatial/RiverShapeML) made the rest possible.

## Findings

| Question | Result | Script |
|---|---|---|
| Example: Shoal Ck at Silverway Dr, Austin (USGS 08156675) | 68 USGS width measurements; 2 in HYDRoSWOT, so not in training | `silverway.py` |
| How many more training sites? | 4,375 → ~7,800 (+~80%); < 10 km² 25 → ~240 (~9×); 10–100 km² 567 → ~1,500 | `new_rows.py` |
| Where do targets sit? | In-channel flow is a median ~6% of bankfull flow, i.e. near low flow | `target_flows.py` |
| Do extra measurements reduce extrapolation? | In-channel flow below all measured flows: 37% → 3% of sites; bankfull above: 32% → 10% | `method_agreement.py` |
| Are targets better determined? | 90% interval on in-channel width ±21% → ±8%; bankfull ±25% → ±19% (median) | `method_agreement.py` |
| Same deployment, different instrument? | Depths agree: bridge +0.1%, boat +2.4%, cableway +1.0% (median, within-site) | `depth_gap.py` |
| Different place, same flow? | Wading sections ~33% shallower than bridge sections (162 sites; 90% of them) | `depth_gap.py` |
| Measurement noise vs model error | Targets from independent collections differ by 8–15% (width) and 14–19% (depth); model held-out error ≈ 50–64% and 30–33% | `noise_floor.py` |

Consistency, certainty, noise floor and depth results use a random sample of 300 width-training sites (seed 1).

### Retrain experiment (bankfull width)

A stand-in model (gradient-boosted trees, 27 public features approximating the deployed model's 15) trained three ways
and scored on the same held-out sites (5-fold CV grouped by reach, 5 repeats):
**A** existing sites + HYDRoSWOT-only targets; **B1** existing sites + all-measurement targets; **B2** B1 + new sites.

| Held-out sites (typical error; bias) | B1 | B2 |
|---|---|---|
| < 10 km² | 88%; +28% | **56%; ~0%** |
| 10–100 km² | 43% | 39% |
| New-type sites | 49% | 42% |
| Existing-type sites | 37.0% | 36.4% |

B1 → B2 (same target definition; only new sites added) is the clean comparison. Scored against HYDRoSWOT-only targets,
A, B1 and B2 tie on existing sites (37.8 / 38.1 / 37.7%), so "better targets" alone shows no gain on familiar sites.
Results hold with stricter (SE ≤ 0.1, ≤ 2×) and looser (SE ≤ 0.3, ≤ 5×) target filters: < 10 km² goes 97 → 62% and 83 → 53%.
Targets use a precision filter (SE of log-width at the 2-yr flood ≤ 0.2, 2-yr flood ≤ 3× max measured flow) instead of
R² ≥ 0.6, which discards ~3,300 sites whose width is known to a few percent. Scripts: `site_comids.py`, `build_targets.py`,
`build_features.py`, `retrain_experiment.py`.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "YOUR_KEY" > .env   # free key: https://api.waterdata.usgs.gov/signup/ (needed for the ~1 hr USGS download)
./get_data.sh            # RiverShapeML repo @29c1209, HydroShare outputs, USGS Water Data APIs
./run_all.sh
```

## Method notes and caveats

- **"In HYDRoSWOT"** means a USGS measurement matches a record in RiverShapeML's HYDRoSWOT copy (same site, within one
  day, discharge within 3%). 97% of HYDRoSWOT records since 2000 at the sampled sites match.
- **New-site counts:** CONUS sites not in the width targets with ≥ 5 width measurements since 2000 and ≥ 10 years of daily
  discharge. The R² > 0.6 filter is applied at the existing pass rate (75%), since new sites weren't fitted.
- **Curves** are plain log-log least-squares fits, not the AHGestimation ensemble used for the published targets.
  Comparisons are relative.
- **Model error** is approximated from the paper's held-out log R² (σ ≈ √(1−R²)·sd(ln y)), not computed per site.
  Published values at gaged reaches equal the training targets, so they can't be used to check the model.
- **The noise floor** reflects variation within a few hundred feet of each gage, not along a whole reach.
- About 14 records the USGS API failed to serve were skipped during the national download.

## Layout

- `src/fetch_usgs_ogc.py`: pages through national USGS OGC API collections (resumable, handles bad records)
- `src/method_agreement.py`: per-site downloads, HYDRoSWOT matching, consistency and bootstrap target certainty
- `src/noise_floor.py`, `src/depth_gap.py`, `src/new_rows.py`, `src/target_flows.py`, `src/silverway.py`: analyses above
- `src/keys.py`: reads the API key from `.env` or `API_USGS_PAT`
- `data/`: downloads and outputs (not tracked)

## Sources

- Modaresi Rad et al. (2024), *JGR: Machine Learning and Computation*, doi:10.1029/2024JH000173
- RiverShapeML: github.com/lynker-spatial/RiverShapeML; outputs on HydroShare, doi:10.4211/hs.d147fcf554a54b2aaa4f146f85da0e03
- HYDRoSWOT: Canova et al. (2016), doi:10.5066/F7D798H6
- USGS Water Data APIs: api.waterdata.usgs.gov; NLDI: api.water.usgs.gov/nldi

*Andrew Sommerlot, 2026*
