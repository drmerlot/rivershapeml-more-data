# Adding USGS channel measurements to RiverShapeML

*More gage sites, better-constrained targets, and a re-predicted national product with prediction intervals*

*Andrew Sommerlot · andrewsommerlot@gmail.com · October 2026 · long form, draft 3*

## Summary

RiverShapeML (Modaresi Rad et al., 2024) learns bankfull width and depth from gage measurements and
predicts them for all 2.65 million NHDPlus reaches in CONUS. Its training data come from HYDRoSWOT, which
holds acoustic Doppler (ADCP) measurements through 2014. USGS now publishes every channel measurement it
makes, by every method (wading, bridge, cableway, boat), through the Water Data OGC API: 4.77 million
records. I rebuilt the training targets from those records using the RiverShapeML and AHGestimation
conventions, kept the model, features and deployment pipeline unchanged, and asked one question: **does the
added data improve the model where it is weakest?**

- **More training sites, mostly where they were scarcest.** Bankfull-width training sites rise from 4,375
  to 6,928 (+58%). Sites under 10 km² rise from 66 to 111, and sites of 10–100 km² from 596 to 1,157.
- **The added measurements are sound.** At the same sites, bankfull widths from all measurements and from
  HYDRoSWOT alone agree at the median (ratio 1.00); depths measured the same way agree within 0.1–2.4%.
- **Held-out error falls in every size class, most in small streams.** Against the same model trained on
  the existing RiverShapeML data, median absolute error in bankfull width falls from 78% to 41% for sites
  under 10 km² (bias +67% → +5%), and from 25% to 22% overall; depth improves similarly. The gain comes
  from the new sites: at existing sites, judged on the original HYDRoSWOT targets, the two models match.
- **A re-predicted national product with 90% prediction intervals.** The retrained model was run through
  the NOAA-OWP deployment pipeline for every reach, with split-conformal intervals by drainage-area class.
  It sits closer to the Bieger et al. (2015) regional curves in headwaters and removes most physically
  implausible values.

The work builds directly on Arash Modaresi Rad's RiverShapeML and the public
[NOAA-OWP/predict-riverML](https://github.com/NOAA-OWP/predict-riverML) pipeline, and uses Mike Johnson's
AHGestimation package for the target fits. The deployment pipeline is clean and complete enough that its
national outputs reproduced exactly from the published models, which made a controlled experiment like this
one possible in a few weeks.

## 1. Motivation

I walked a stretch of Shoal Creek in Austin, Texas, to see how a reach-averaged geometry product compares
with the channel on the ground. Over a few hundred meters the channel moves from a walled, rock-lined
section with a built low-flow channel, to an incised section held by gabions and a bedrock ledge, to a
section rebuilding a small floodplain below a failed grade control. A single bankfull value for the whole
8.2 km reach (COMID 5781223) is the right abstraction for a national product, so that variability is not
the subject here.

The gage at the downstream end of the walk is. USGS 08156675 (Shoal Creek at Silverway Drive) has
**68 width measurements**. HYDRoSWOT has **2**, both boat measurements from 2009, so the site falls below the
5-measurement minimum. The other 66, mostly wading measurements, are not in HYDRoSWOT at all. The question
this note answers is how common that situation is, and what the model gains when those measurements are
used.

![](../docs/process_overview.svg)

*Figure 1. Process. Each step corresponds to scripts in the accompanying repository, and the full table-level lineage
(36 tables, 18 operations, each with source, retrieval date and commit) is checked automatically on every
run.*

## 2. Data and methods

**Measurements.** All USGS channel measurements were retrieved from the `channel-measurements` collection
of the USGS Water Data OGC API (4,772,063 records), with site metadata and annual peaks from the
`monitoring-locations` and `peaks` collections. Gages were linked to NHDPlus reaches with the NLDI; existing
RiverShapeML sites keep their published COMIDs. Candidate new sites are CONUS gages with at least 5
measurements since 2000 and at least 10 years of daily flow (4,501 sites; 3,192 linked to a reach).

**Targets.** For each site, width, mean depth (area / width) and velocity were converted to SI units and
screened with RiverShapeML's sanity rules (width < 4 km, depth < 65 m, Q ≤ 1.1 × maximum annual peak).
Hydraulic geometry was then fit with AHGestimation (`mad_filter` → `qva_filter` → `ahg_estimate`), which
fits W = aQ^b, D = cQ^f and V = kQ^m jointly under continuity, and each curve was evaluated at the 2-year
flood, Q₂. I took Q₂ as the median of the annual peak series, which is the 2-year flood by definition on an
annual-maximum basis (England et al., 2019; Institute of Hydrology, 1999); it agrees with RiverShapeML's
bankfull flows at the 4,315 shared sites (median ratio 1.02). The target is y = ln(1 + W_bf) with W_bf in meters, as in
RiverShapeML, and likewise for depth. Sites where depth decreases with flow (f < 0) were dropped (53 sites).

**Target filter.** RiverShapeML keeps sites whose width–discharge fit has R² ≥ 0.6. Many well-measured
sites fail that test because width barely changes with flow, not because the target is uncertain: 3,636 of
the 6,928 width targets here would fail it, with a median standard error of 0.040 on the log target. I
replaced it with a precision filter: standard error of the log target at Q₂ ≤ 0.2, Q₂ ≤ 3 × the largest
measured flow (limits extrapolation), and at least 5 measurements.

**Model.** Features, feature engineering, missing-value handling and inference follow predict-riverML exactly
(30 features from `ml_inputs.parquet`). I first reproduced the deployed models' national predictions: on
the same reaches, the published output statistics match to six decimals (e.g., minimum and maximum width
1.813823 m and 14,114.583008 m; median 18.43 m). For retraining I kept their XGBoost
learning rate and sampling settings but chose tree depth, minimum child weight and number of trees by a
regularization sweep with early stopping on a reach-grouped 70/15/15 split (depth 6, min_child_weight 5,
about 2,450 trees). This matched the deployed settings' held-out accuracy with roughly a quarter of the
trees and much less fit to the training data (test log R² 0.84 for width, 0.76 for depth). All runs below
use the same settings, so only the training data differ.

**Experiment.** One comparison, with everything but the training data held fixed:

| Run | Training sites | Targets built from |
|---|---|---|
| A, existing RiverShapeML data | Existing RiverShapeML sites | HYDRoSWOT measurements only |
| B2, expanded training set | Existing plus new sites | All USGS measurements |

Both are scored with 5-fold cross-validation grouped by reach (3 seeds) on the same held-out sites, against
the all-measurement targets, which rest on about ten times more measurements per site. Because A never
trained on those targets, existing sites are also scored against the HYDRoSWOT-only targets (the same-target
check). HYDRoSWOT records maximum rather than mean depth, so A's depth target uses mean depth from the USGS
records of the same HYDRoSWOT visits (97% match by date and discharge). Error is reported as
ln(predicted / observed), summarized as median absolute error (%), median bias (%), share of sites within a
factor of 2, and R² in log space.

**Prediction intervals.** The B2 model was refit on all sites and applied to every reach through the
deployment pipeline. Split-conformal 90% prediction intervals (Lei et al., 2018) were computed per
drainage-area class from held-out residuals. Coverage was then checked on sites unseen by both the model and
the calibration (0.88–0.90 across classes).

## 3. Results

### More sites, concentrated in small drainages

| Drainage area (km²) | Published width targets (R² ≥ 0.6) | Expanded set: existing sites | Expanded set: new sites | Expanded total |
|---|---:|---:|---:|---:|
| < 10 | 66 | 38 | 73 | 111 |
| 10–100 | 596 | 560 | 597 | 1,157 |
| 100–1,000 | 1,766 | 1,919 | 876 | 2,795 |
| 1,000–10,000 | 1,395 | 1,620 | 465 | 2,085 |
| > 10,000 | 485 | 577 | 193 | 770 |
| **Total** | **4,375** | **4,719** | **2,209** | **6,928** |

Totals include 10 sites with a zero or missing drainage area. Depth targets follow the same pattern (7,017 sites). The
gain is largest relative to need: 58% of NHDPlus reaches drain less than 10 km², and those reaches were
represented by 66 training sites.

### The added measurements are consistent with HYDRoSWOT

Across all target sites, 97% of HYDRoSWOT measurements since 2000 match a record in the USGS collection
(within one day, discharge within 3%; 147,074 of 152,100), and in a 300-site sample a typical site has 13
HYDRoSWOT measurements against 142 in total. For width, the
curve fit to the added measurements matches the HYDRoSWOT curve over shared flows at the median (+0.6%; 62%
of sites within ±10%), and across 3,666 existing sites the bankfull width targets from all measurements and
from HYDRoSWOT alone agree at the median (ratio 0.996). For depth, where the same deployment type appears in
both, depths agree closely (bridge +0.1%, boat +2.4%, cableway +1.0%). Wading sections are about 33% shallower
than bridge sections at the same flows, because they are taken at different cross sections; this is a
location difference rather than a measurement error, and the per-site fit pools both. Repeat-measurement
scatter sets a noise floor of 8–15% for width and 14–19% for depth. The joint AHG fit changes the median width target by less than 1% relative to a plain
log-log fit but corrects poorly conditioned sites (e.g., Walnut Creek: 28.1 m → 19.0 m).

### Held-out error falls in every size class, most in small streams

![](fig_errors.png)

*Figure 2. Held-out median absolute error and median bias by drainage-area class: the same model trained
on the existing RiverShapeML data (A) and on the expanded training set (B2). Same settings, same held-out
sites.*

| | Width: A → B2 | Depth: A → B2 |
|---|---|---|
| Median absolute error, all sites | 25.4% → 22.4% | 23.8% → 20.9% |
| Median absolute error, < 10 km² | 78% → 41% | 72% → 32% |
| Median bias, < 10 km² | +67% → +5% | +72% → +7% |
| Within a factor of 2, all sites | 89% → 92% | 89% → 94% |
| Log R², all sites | 0.77 → 0.82 | 0.66 → 0.76 |

The model trained on the existing data overpredicts small streams because it has almost no small streams to
learn from; the new sites remove most of that bias. Gains are smaller in larger basins.

**Where the gain comes from.** At sites outside RiverShapeML's training set, the closest proxy here for an
unseen reach, median absolute error falls from 32% to 26% for width and from 38% to 25% for depth. At
existing sites, judged on the HYDRoSWOT-only targets (the same-target check), the two models match: width
20.4% (A) vs 19.6% (B2), depth 18.2% vs 18.2%. Scored on the all-measurement targets, A also looks worse at
existing sites (21.4% vs 19.7% for width), but most of that difference reflects the change in target
definition rather than a better model, so I do not count it as a gain. The improvement comes from the new
sites, and the expanded model gives up nothing on the original problem. For context, repeat-measurement
scatter means no model can do much better than about 10–15% median error on width.

### A re-predicted national product

![](fig_product.png)

*Figure 3. Median bankfull width (left) and depth (right) by drainage area across all CONUS reaches for the
current model (predict-riverML), the 2024 HydroShare release, the expanded-set retrain with its 90% prediction
interval, and the Bieger et al. (2015) regional curves. The shaded region holds 44% of reaches.*

| Check (all 2.6 M reaches) | Expanded-set retrain | Current model | 2024 release |
|---|---:|---:|---:|
| Width > 1 km | 0.002% | 0.47% | 0.013% |
| Width / depth > 100 | 0.09% | 1.94% | 0.05% |
| Width halves going downstream | 2.2% | 3.6% | 5.3% |
| Width–drainage-area exponent (Bieger: 0.36) | 0.23 | 0.15 | 0.06 |
| Depth–drainage-area exponent (Bieger: 0.25) | 0.16 | 0.12 | 0.00 |
| Median ratio to Bieger width | 1.80 | 2.88 | 3.28 |

Between 100 and 10,000 km² the three ML products agree closely (median widths within about 10%). Above
10,000 km² the current model runs wider (median 152 m, against 122 m for the retrain and 110 m for the 2024
release), and all curves wobble there because few reaches fall in each bin and large rivers of different
character (e.g., the Colorado and the Missouri) share them. Below 10 km² they diverge: median widths are
9.1 m (expanded-set retrain), 17.6 m (current model), 26.5 m (2024 release) and 3.5 m (Bieger). Downstream
scaling moves toward the regional curves, and implausibly wide reaches nearly disappear.

Depth tells the same story in small streams. Below 10 km² median depths are 0.72 m (expanded-set retrain),
0.93 m (current model), 1.63 m (2024 release) and 0.36 m (Bieger). The 2024 release is nearly flat in depth
from the smallest reaches to about 1,000 km² (median 1.5 m), which is the exponent of 0.00 in the table.
Between 1,000 and 10,000 km² the retrain and the current model run deeper than both the 2024 release and
Bieger (median 2.5 m against 1.7–1.8 m), so the depth improvement is concentrated in small streams. In both
panels the Bieger curve rises again below about 0.3 km²; each regional equation is monotone, so this
probably reflects which regions those very small reaches fall in. Because most reaches are small, the national median width falls from 18.4 m (current model) to
10.9 m; this is the largest practical change for users of the product. The prediction intervals widen where
data are thin: for width they span roughly 0.50–1.9× the prediction above 10 km² and 0.36–4.2× below it.
Below 10 km² they rest on about 110 calibration sites, and gaged small streams are not typical of NHDPlus
headwaters, so they should be read as indicative there.

The Silverway reach illustrates both the improvement and its limit. The retrain predicts 15.8 m wide and
0.99 m deep (current model 14.8 m / 1.29 m; 2024 release 19.6 m / 1.51 m). The Silverway site itself is
not in the expanded training set: its Q₂ (2,700 cfs, 76 m³/s) is 5.2 times the largest measured flow, so
its target would be an extrapolation and the filter excludes it.

## 4. Discussion and limitations

**The headwater gap is narrower but not closed.** Below about 5 km² there are still only ~50 training sites
for 1.16 million reaches. Tree ensembles cannot extrapolate beyond the range of their training targets, so
predictions flatten (and in all three products bend upward) at the smallest drainage areas rather than
continuing the downstream trend. The ratio to Bieger widths remains 1.8 overall. Part of that gap is
definitional: Bieger's field bankfull indicators and a Q₂-based target are not the same quantity, and the
Q₂ convention may overstate bankfull flow in small, flashy basins.

**Gaged is not ungaged.** All accuracy figures are for reaches with gages, which skew toward larger rivers
than the network as a whole. The conformal intervals inherit that assumption: they are calibrated
on gaged sites and are exchangeable only to the extent that ungaged reaches resemble them.

**Some useful data remain unused.** About 1,300 candidate gages did not link to a reach through the NLDI, and
about 7,000 sites have five or more measurements but less than 10 years of daily flow. The latter could
use National Water Model flood frequencies; RiverShapeML's training code already has an NWM
flood-frequency option, used there in place of gage floods at the same sites. Most of these sites are small
streams (985 under 10 km² and 2,159 at 10–100 km², of those with a known drainage area). In an exploratory
check at 7,403 gaged reaches, the NWM 2-year flood ran about 20% below the gage value at the median, and
about 50% below under 10 km², where the two barely correlate (squared correlation of logs 0.72 overall,
0.03 for the 137 small sites). That argues for bringing these sites in with lower weight or a flag, and for
checking the NWM flows first.

**Simplifications to revisit.** Three choices here are deliberately simple and worth tightening. (i) Where
USGS records a split-flow measurement as one row per channel, the rows are treated as separate points; this
affects a minority of sites (about 620 of 6,928 with more than 10% of rows split). (ii) Q₂ uses each gage's full peak record,
which often starts decades before the 2000+ measurements, and regulated peaks are kept; both runs share Q₂, so
the comparison is unaffected, but absolute target levels are. (iii) Cross-validation holds out whole reaches,
not whole basins, so neighbouring gages on the same river can sit on either side of a split; a basin-blocked
check would be a useful robustness test.

**Two observations I could not resolve from the public record.** The current deployed models and the 2024
HydroShare release differ more than I expected: the squared correlation between their log widths is 0.36. Separately, 932 of
the 2.65 million reach IDs in the HydroShare file appear more than once (66,341 rows in total). The repeat
counts are perfect squares (4, 9, 16, 64, …), which is the signature of a many-to-many merge, and most of
these IDs carry two distinct predictions, so they may be two features sharing an ID rather than true
copies. For the comparison here each was collapsed to its median; with 932 reaches the effect is
negligible. Both points are noted as questions, not findings. Relatedly, the training-target files in the
RiverShapeML research repository sit 1.6–1.9 times below the deployed models' predictions at the same sites,
while the targets built here match the deployed scale (ratio 1.0 for depth, 1.13 for width). That suggests
the deployed models were trained on a later target set than the one in the repository.

## 5. Questions and possible next steps

These are offered as starting points for a conversation; the team will know which matter and which have
already been considered.

1. **Was the HYDRoSWOT-only choice deliberate?** For example, to keep a single, consistent measurement
   method. The agreement statistics above suggest the other methods can be added with little risk, but
   there may be reasons I have not seen.
2. **Headwater behavior.** Monotone constraints on drainage area and discharge, or blending with regional
   curves below a threshold, would keep predictions on a physically sensible trend where data are sparse.
3. **More sites.** Recover the unlinked gages, and use RiverShapeML's NWM flood-frequency option to bring in
   the ~7,000 short-record sites, mostly small streams.
4. **In-channel geometry.** The same measurements support targets at lower flows (in-channel flow is about
   6% of bankfull flow at the median), and these sites are far better sampled there than at Q₂. This is the
   natural next experiment and maps directly onto the existing in-channel model.
5. **Training settings.** Early stopping and modest regularization reach the same accuracy with about a
   quarter of the trees, which shortens training and inference.
6. **Release housekeeping.** An updated HydroShare release could carry prediction intervals and resolve
   the repeated IDs noted above.

I would be glad to prototype any of these within the predict-riverML structure so that the results drop
straight into the existing pipeline.

## Code and data

Repository: [github.com/drmerlot/rivershapeml-more-data](https://github.com/drmerlot/rivershapeml-more-data). It contains every script, a single `run_all.sh` that rebuilds all
results from public sources, a checked data-lineage record, and an interactive notebook (marimo) for
exploring sites, targets, runs and the national product. The re-predicted product ("ML-derived channel width
and depth, expanded-set retrain", with 90% prediction intervals for every reach) is attached to the
repository's [v1.0 release](https://github.com/drmerlot/rivershapeml-more-data/releases/tag/v1.0) as a 38 MB Parquet file, and is regenerated by the pipeline.

## Acknowledgments

This work stands on Arash Modaresi Rad's RiverShapeML research and the NOAA Office of Water Prediction's
open [predict-riverML](https://github.com/NOAA-OWP/predict-riverML) deployment pipeline, whose published
models, inputs and inference code made exact reproduction straightforward. Target fitting uses
[AHGestimation](https://github.com/mikejohnson51/AHGestimation) by Mike Johnson, with contributions from
Arash Modaresi Rad. Data are from the USGS (HYDRoSWOT; Water Data APIs; NLDI), the E2NHDPlus attributes,
and the Bieger et al. (2015) regional curves (USGS ScienceBase). Any errors of interpretation are mine.

## References

- Bieger, K., Rathjens, H., Allen, P. M., and Arnold, J. G. (2015). Development and evaluation of bankfull
  hydraulic geometry relationships for the physiographic regions of the United States. *JAWRA*, 51(3),
  842–858. doi:10.1111/jawr.12282. Data: doi:10.5066/F7765D7V.
- Canova, M. G., Fulton, J. W., and Bjerklie, D. M. (2016). USGS HYDRoSWOT: USGS Hydroacoustic dataset in
  support of the Surface Water Oceanographic Topography satellite mission. U.S. Geological Survey data
  release. doi:10.5066/F7D798H6.
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. KDD*, 785–794.
- England, J. F., et al. (2019). Guidelines for determining flood flow frequency: Bulletin 17C. *USGS
  Techniques and Methods* 4-B5.
- Institute of Hydrology (1999). *Flood Estimation Handbook*, Vol. 3. Wallingford, UK.
- Johnson, J. M., et al. AHGestimation: An R package for computing robust, mass preserving hydraulic
  geometries and rating curves. github.com/mikejohnson51/AHGestimation.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-free
  predictive inference for regression. *JASA*, 113(523), 1094–1111.
- Leopold, L. B., and Maddock, T. (1953). The hydraulic geometry of stream channels and some physiographic
  implications. *USGS Professional Paper* 252.
- Modaresi Rad, A., et al. (2024). [title to confirm].
  *JGR: Machine Learning and Computation*. doi:10.1029/2024JH000173.
