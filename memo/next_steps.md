# Next steps (for the memo)

Running list of follow-ups the analysis points to. **Confirmed** = agreed for the memo. **Candidates** = raised
during the analysis; keep, merge or cut when we draft.

## Confirmed

1. **Recover gages that don't link to a reach.** ~1,300 of 4,501 new candidate gages (~30%, slightly more for
   small streams, more often in the West) aren't linked to an NHDPlus reach in NLDI, so they were dropped. Two ways
   to recover them: USGS's improved gage-to-reach crosswalk (possibly exposed in NLDI as a "reference gages" source;
   to verify), or snapping gage coordinates to the nearest reach and accepting the match only if drainage areas
   agree within ~25%. Could add several hundred sites.

## Candidates

- **Sites without a long daily flow record** (~7,075 more candidates): usable if NWM-based flood frequencies are
  acceptable for their targets; the research repo already has `nwm_*` target variants.
- **Monotone constraints** in XGBoost (width and depth never decrease with drainage area or flow) to stop the
  upward bend below ~5 km², where 44% of reaches sit with almost no training data. Or blend toward a regional
  curve below the smallest training size.
- **A precision-based target filter** instead of R² ≥ 0.6, which drops ~3,300 sites whose bankfull width is known
  to within a few percent (confined channels whose width barely changes with flow).
- **Lighter model settings:** validation error flattens after ~500–1,000 trees vs 9,765 / 11,000 deployed; similar
  accuracy for far less compute, and less memorization of training sites.
- **In-channel (1-year flood) targets** from the same measurements; the in-channel flow sits where wading data is
  densest.
- **Reach-specific prediction intervals** (e.g., quantile models) instead of one interval per size class.
- **Measurement location as variability:** use the spread between measurement spots at a gage as an estimate of
  within-reach variability, or a per-site target uncertainty for weighting.
- **Questions rather than steps:** whether the HydroShare 2024 release should be updated to the current model (and
  its ~65k duplicated reach IDs); which bankfull definition downstream users need (2-year-flood water width vs
  field top-of-bank, ~1.5–2× apart).
