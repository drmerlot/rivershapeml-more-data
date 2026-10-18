<style>@page { margin: 0.55in 0.75in; } body { font-size: 10pt; line-height: 1.36; } p { margin-bottom: 6pt; }
img { margin: 2pt auto 0; } table { margin: 2pt auto 6pt; }</style>

# Adding USGS channel measurements to RiverShapeML

*Andrew Sommerlot · andrewsommerlot@gmail.com · October 2026*

RiverShapeML (Modaresi Rad et al., 2024) predicts bankfull width and depth for all 2.65 million NHDPlus
reaches in CONUS from a model trained on HYDRoSWOT, the USGS collection of acoustic Doppler (ADCP)
measurements through 2014. USGS now publishes every channel measurement it makes, by every method (wading,
bridge, cableway, boat), through its Water Data OGC API: 4.77 million records. This note asks whether
those measurements improve the model, with everything else held fixed.

**How it started.** The USGS gage on Shoal Creek at Silverway Drive in Austin (08156675), at the end of a
reach I walked, has 68 width measurements. HYDRoSWOT has 2, both from 2009, so the site falls below the
5-measurement minimum and is not used. The other 66, mostly wading measurements, are simply outside HYDRoSWOT.
Across CONUS, that pattern turns out to be common, and most common in small streams.

**What I did.** I rebuilt the bankfull targets from all USGS measurements, using RiverShapeML's sanity rules
and Mike Johnson's AHGestimation for the hydraulic-geometry fits, evaluated at the 2-year flood (the median
annual peak). I replaced the R² ≥ 0.6 site filter with a filter on the precision of the target itself
(many well-measured sites have nearly constant width and fail R² for that reason alone). The
features, feature engineering and inference are those of the public NOAA-OWP/predict-riverML pipeline,
whose national outputs I could reproduce exactly from the published models; the model is the same XGBoost
setup with modest regularization and early stopping.

**More sites where they were scarcest.** Bankfull-width training sites rise from 4,375 to 6,928 (+58%).
Sites draining under 10 km² rise from 66 to 111, and 10–100 km² from 596 to 1,157; 58% of NHDPlus reaches
are under 10 km². At the same sites, bankfull widths from all measurements and from HYDRoSWOT alone agree
at the median (ratio 1.00), and depths measured the same way agree within 0.1–2.4%.

**Lower error in every size class, most in small streams.** I trained the same model twice: on the
existing RiverShapeML data (HYDRoSWOT measurements at existing sites) and on the expanded set (all
measurements, existing and new sites), and scored both on the same held-out sites with cross-validation
grouped by reach.

<figure><img src="fig_errors.png" style="width:100%"><figcaption><em>Figure 1. Held-out median absolute error and median bias by drainage area, existing RiverShapeML data (A)
vs the expanded training set (B2).</em></figcaption></figure>

| | Width | Depth |
|---|---|---|
| Median absolute error, all sites | 25.4% → 22.4% | 23.8% → 20.9% |
| Median absolute error, < 10 km² | 78% → 41% | 72% → 32% |
| Median bias, < 10 km² | +67% → +5% | +72% → +7% |
| Median absolute error, sites new to training | 32% → 26% | 38% → 25% |
| Existing sites, on the original HYDRoSWOT targets | 20.4% → 19.6% | 18.2% → 18.2% |

The existing-data model overpredicts small streams because it has few to learn from. The gain comes from
the new sites: at existing sites, judged on the original HYDRoSWOT targets, the two models match.

**A re-predicted national product.** I ran the retrained model through the predict-riverML pipeline for
every reach, with split-conformal 90% prediction intervals by drainage-area class (coverage 0.88–0.90 on
held-out sites). Between 100 and 10,000 km² it agrees with the current model within about 10%. In
headwaters it is narrower and closer to the Bieger et al. (2015) regional curves, so the national median
width falls from 18.4 m to 10.9 m. Implausibly wide reaches (> 1 km) drop from 0.47% to 0.002%.

<figure><img src="fig_product.png" style="width:100%"><figcaption><em>Figure 2. Median bankfull width (left) and depth (right) by drainage area, all CONUS reaches. The shaded
region holds 44% of reaches but only ~50 training sites; the gap to the regional curves narrows but does not
close.</em></figcaption></figure>

**Questions and discussion.**

1. What was the original motivation for training on HYDRoSWOT, for example a single, consistent
   measurement method? The agreement above suggests data from other methods could be added.
2. About 7,000 more sites, mostly small streams, have enough measurements but short flow records.
   RiverShapeML's existing NWM flood-frequency option could be used to bring them in.
3. The same measurements are much denser at in-channel flows, which maps directly onto the in-channel
   model. This would be a reasonable next step to try with the process described here.

**Credit and citations.** This builds directly on Arash Modaresi Rad's RiverShapeML and the NOAA Office of
Water Prediction's open predict-riverML pipeline, whose published models and inference code made a
controlled comparison possible, and on AHGestimation by Mike Johnson, with contributions from Arash Modaresi
Rad. The code (one script rebuilds everything from public sources), a longer write-up and an interactive
notebook are at [github.com/drmerlot/rivershapeml-more-data](https://github.com/drmerlot/rivershapeml-more-data); the re-predicted product with intervals for every reach is in the
[v1.0 release](https://github.com/drmerlot/rivershapeml-more-data/releases/tag/v1.0).

<p style="font-size:8.5pt; line-height:1.3; color:#444"><em>References.</em> Modaresi Rad, A., et al. (2024),
<em>JGR: Machine Learning and Computation</em>, doi:10.1029/2024JH000173. Bieger, K., et al. (2015), <em>JAWRA</em>
51(3), 842–858, doi:10.1111/jawr.12282. Johnson, J. M., et al., AHGestimation (R package).</p>
