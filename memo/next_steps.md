# Next steps

Follow-ups the analysis points to, roughly in order of expected value.

## Most promising

1. **Recover gages that don't link to a reach.** ~1,300 of 4,501 new candidate gages (~30%, slightly more for
   small streams, more often in the West) aren't linked to an NHDPlus reach in NLDI, so they were dropped. Two ways
   to recover them: USGS's improved gage-to-reach crosswalk (possibly exposed in NLDI as a "reference gages" source;
   to verify), or snapping gage coordinates to the nearest reach and accepting the match only if drainage areas
   agree within ~25%. Could add several hundred sites.


2. **In-channel targets and models.** Repeat the full process (targets, retrain, re-predict, compare) for
   in-channel width and depth. The in-channel flow (RiverShapeML's 100%-annual-exceedance flood, a median ~6% of the
   2-year flood) sits where wading measurements are densest, so the added data should help most there. The current
   deployed model is bankfull-only, but OWP flood mapping uses in-channel values.

## Other candidates

- **Sites without a long daily flow record** (~7,075 more candidates, mostly small streams: 985 under 10 km²
  and 2,159 at 10–100 km² of those with a known drainage area): usable if NWM-based flood frequencies are
  acceptable for their targets; RiverShapeML's training code has an NWM flood-frequency option.
- **Split-flow measurements:** sum flow, width and area per site and time where USGS records one row per
  channel, and drop exact duplicate rows; then rerun targets and models.
- **Basin-blocked cross-validation** (e.g., grouped by HUC-4) as a robustness check on the A vs B2 result.
- **Monotone constraints** in XGBoost (width and depth never decrease with drainage area or flow) to stop the
  upward bend below ~5 km², where 44% of reaches sit with almost no training data. Or blend toward a regional
  curve below the smallest training size.
- **A precision-based target filter** instead of R² ≥ 0.6, which drops ~3,300 sites whose bankfull width is known
  to within a few percent (confined channels whose width barely changes with flow).
- **Early stopping and regularization (measured):** a sweep with early stopping on validation (tree depth,
  min_child_weight, L2) picks depth 6, min_child_weight 5, ~2,450 trees for both width and depth. On the held-out test
  set it matches the deployed settings (width log R² 0.838 vs 0.835; depth 0.761 vs 0.761) with ~4x fewer trees
  and 3-6x less memorization of training sites. Practical recommendation; settings are not the accuracy bottleneck.
- **Match the flood record to the measurement period.** Annual peaks use each gage's full record (e.g. 1975-2026)
  while measurements start in 2000; where flood regimes changed (urbanization, dams), restricting peaks to the same
  period would keep the curve and the flow it's read at from the same era.
- **Reach-specific prediction intervals** (e.g., quantile models) instead of one interval per size class.
- **Measurement location as variability:** use the spread between measurement spots at a gage as an estimate of
  within-reach variability, or a per-site target uncertainty for weighting.
- **Questions rather than steps:** whether the HydroShare 2024 release should be updated to the current model (and
  its 932 repeated reach IDs (66,341 rows; perfect-square repeat counts suggest a many-to-many merge)); which bankfull definition downstream users need (2-year-flood water width vs
  field top-of-bank, ~1.5–2× apart).
- **Observation (not a step):** in an exploratory check at ~7,400 gaged reaches (not part of this pipeline),
  the NWM v2.1 2-year flood ran ~20% below the gage value overall and ~50% below for drainage areas < 10 km²
  (10th-90th percentile ratio 0.03-2.2). It is a key input to the current model, so small-stream inputs are
  weakest exactly where training data is thinnest.
