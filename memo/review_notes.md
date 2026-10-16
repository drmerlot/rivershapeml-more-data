# Review notes: process double-check (2026-10-15)

An independent, read-only audit of the pipeline, with the key items re-verified by hand. **No pipeline code
was changed.** The memo text was corrected where the audit showed a claim was overstated (section A). The
code-level items (section B) wait for your approval.

## A. Already corrected in the memos (text only)

1. **"Half to two-thirds of the gain comes from better targets" was wrong. Removed.** At existing sites, A
   looks worse only when it is scored on the all-measurement targets, which it never trained on. Scored on
   the HYDRoSWOT targets (existing sites, n = 3,446):
   - width: A 20.4% vs B2 19.6%
   - depth: A 18.2% vs B2 18.2%

   The real gain is at sites new to training:
   - width: 31.6% → 25.9%
   - depth: 37.8% → 25.3%

   The memos now say this: the gain comes from the new sites, and at existing sites the two models match.
   This is a simpler and more defensible message.
2. **"Reproduction check" renamed to "same-target check"** and restricted to existing sites. The old
   numbers (20.8 vs 20.0) included 212 new sites, and the name clashed with the reproduction of the deployed
   model.
3. **Bieger width ratio is now 1.80**, up from a stale 1.84.
4. **"Prediction intervals are honest…" softened.** Below 10 km² the intervals rest on about 110
   calibration sites, and gaged small streams aren't typical headwaters.

## B. Code-level findings: need your decision

| # | Finding | Effect | Suggested fix | Cost |
|---|---|---|---|---|
| 1 | **Split-channel measurements are treated as separate points.** At split-flow sites USGS stores one row per channel (e.g., "Main" and "Bypass 1" at the same time). `build_targets.py` ignores `channel_name`. 2,850 target sites are touched; 622 have more than 10% of rows affected. There are also about 4,600 exact duplicate rows. | Moves width at Q₂ by more than 10% at about 60% of those 622 sites. Headline numbers probably move by under a point (not yet rerun). | Sum Q, width and area per site-timestamp (or keep the total/main row), and drop exact duplicates. | Rerun targets + retrains + product, about 2–3 h compute |
| 2 | **Q₂ uses the full peak record** (median start 1959), while geometry is from 2000 onward. There is no filter on peak qualifiers. 23% of sites are mostly regulated. | A and B share Q₂, so **the comparison is unaffected**. Absolute target levels and the product shift: Q₂ changes by more than 20% at 14% of sites when restricted to water years ≥ 1990. | At minimum, disclose it. Optionally restrict to a recent period, drop LESSTHAN/GREATERTHAN/DAMFAILURE peaks, and de-duplicate water years. | Same rerun as #1 |
| 3 | **Spatial dependence in CV.** Folds are grouped correctly by reach, but 61% of sites share a river (level path) with another site. The denser B2 network has more near neighbours, so the CV may flatter B2 slightly for truly ungaged reaches. | A reviewer at OWP will likely ask. | Add one HUC-blocked CV row (e.g., grouped by HUC-4) as a robustness check. | About 1 h compute; adds a line to the memo |
| 4 | **Conformal details:** no finite-sample correction, and reaches with drainage area 0 or missing get no interval. | Small. | Use the ceil((n+1)(1−α))/n quantile; fill missing drainage areas. | Small |
| 5 | **AHGestimation's `qva_filter` is a no-op here:** depth and velocity are derived from area, so continuity holds exactly. The `viable` flag never filters anything. | None on results, but "mass-conserving fit" shouldn't be claimed as a check. | Wording only (the memos don't currently claim it). | None |
| 6 | **Small inconsistencies:** the < 10 km² site count is 111 in the targets table, 110/113 in CV scoring, and 117/120 using E2NHDPlus drainage area for all targets. | Cosmetic. | Pick one drainage-area source throughout. | Small |
| 7 | **Stale docstrings:** `train_eval.py` (still says "fixed trees, no early stopping") and `build_targets.py` line 12. A could also be slightly handicapped (≤ 0.5 pt) because the settings were tuned on B2 data. | None on results. | Update the docstrings; mention the tuning note in the long form if asked. | Trivial |

**Checked and fine:**
- **Units:** every conversion checked. HYDRoSWOT widths equal the matched USGS widths (median ratio 1.000).
- **No leakage across folds.** All variants are scored on identical held-out sites.
- **The test set was not used for tuning.**
- **The conformal calibration is genuinely out-of-sample,** and the coverage check is honest.
- **No row multiplication in any join.**
- **Product numbers match the memo:** > 1 km widths, exponents, coverage.

**Metric definitions** (for the long form, if asked):
- Median absolute error = median of |pred − obs| / obs.
- Bias = median(pred / obs) − 1.
- Within 2× = share with 0.5 ≤ pred/obs ≤ 2.
- Log R² is in natural-log metres and computed within each group.
- All metrics are averaged over 3 seeds.

## C. Separate observation: published target files vs deployed models

The width and depth target files in the RiverShapeML research repo sit **1.6–1.9× below** the deployed
models' predictions at the same sites. Our targets match the deployed scale (ratio 1.0 for depth, 1.13 for
width). Most likely the deployed models were trained on a later target set. The long form poses this as a
question; it doesn't affect our comparison.

## Recommendation

- **Before sending:** fix #1 (split channels) and rerun, since it touches the training data itself. Add #3
  (HUC-blocked CV) as one robustness line.
- **Disclose #2** in the long form.
- **Leave #4–7 for later.**
- **Expected effect:** small changes to the numbers, with the message unchanged.
