# More measurements for RiverShapeML? Notes from a walk on Shoal Creek

*Andrew Sommerlot · October 2026 · draft v1.1*

## How this started

I walked a stretch of Shoal Creek near my home in Austin to see how a channel-geometry model's
view of a reach compares with what's on the ground. Within a few hundred yards the channel shifts
from a walled, rock-lined channel with a small built low-flow channel, to an incised section held in
place by gabions and a bedrock ledge, to a section rebuilding a small floodplain below a failed grade
control. Where I'd put "bankfull" changed several-fold between those sections. RiverShapeML gives
the whole 8 km reach (COMID 5781223) a single bankfull value: 19.6 m wide, 1.5 m deep.

That's expected for a reach-averaged product, so it isn't what these notes are about. What caught my
attention was the USGS gage at the downstream end of my walk, Shoal Ck at Silverway Dr (08156675).
USGS has **68 width measurements** there. RiverShapeML's training data has **2**, both boat ADCP
measurements from 2009, so the site falls under the 5-measurement minimum and isn't used. The other 66,
mostly wading measurements, aren't in HYDRoSWOT at all.

So I looked at how general that is.

## What I found

**1. There's a lot more data, and much of it is in the streams the model sees least.**
HYDRoSWOT contains ADCP measurements only, through about 2015. USGS's current channel-measurement
records include every method (wading, bridge, cableway, boat) through 2026: about 4.8 million records, 1.8 million of
them usable width measurements since 2000. I counted CONUS sites that aren't in training today but meet the paper's
filters (≥ 5 measurements since 2000, ≥ 10 years of daily flow for flood frequency), and assumed they'd pass the
R² > 0.6 fit filter at the same rate as existing sites (75%):

| | Training sites today | With additions (est.) |
|---|---|---|
| All sites | 4,375 | ~7,800 (**+~80%**) |
| < 10 km² | 25 | ~240 (**~9×**) |
| 10–100 km² | 567 | ~1,500 (~2.6×) |

About 7,000 more sites have ≥ 5 measurements but no long daily record. They could be usable with
NWM-based flood frequencies, which I see the repo already supports.

**2. The extra measurements agree with the ADCP ones when taken the same way.**
At 300 randomly sampled training sites, with the same deployment (from a bridge, boat or cableway),
non-ADCP and ADCP depths agree within 1–3% (median, within-site). So "not in HYDRoSWOT" doesn't
mean "lower quality."

**3. They make the existing targets better determined.**
At those same sites there's a median of 13 HYDRoSWOT measurements vs 142 others since 2000. Adding them:

- the in-channel flow falls below every measured flow at 37% of sites with HYDRoSWOT alone, but only 3% with all data;
  the bankfull flow lies above every measured flow at 32% vs 10%;
- the 90% bootstrap interval on in-channel width shrinks from ±21% to ±8% (median), and bankfull from ±25% to ±19%.

The in-channel flow is low (median ~6% of the bankfull flow), which is exactly the range where wading
measurements are dense and boat measurements are sparse.

**4. Different measurements see different parts of the reach.**
At the same flows, wading sections are about **33% shallower** than bridge sections (162 sites,
shallower at 90% of them), and wading far from the gage runs shallower than wading close to it.
Neither is "the" depth. Since HYDRoSWOT leans toward boat and bridge sections, today's depth targets may
lean toward the deeper, more confined parts of the reach, which matches the paper's own note that gages
sit at confined sections.

**5. Measurement noise vs model error.**
Treating each set of measurements at a site as an equally valid look at the reach, the targets
computed from different sets disagree by a typical **8–15% for width and 14–19% for depth**. The model's
held-out error implied by the paper's R² is roughly **50–64% for width and 30–33% for depth**. For
width, nearly all the error is in the model and its inputs; for depth, measurement variability is a
meaningful share.

## Questions this raised for me

1. **Was HYDRoSWOT-only a deliberate choice?** Consistency, quality and convenience would all be good
   reasons. I'd like to understand the thinking before going further.
2. **How should multiple collections at a site be combined?** Pooling them all as looks at the same reach
   seems natural. The spread between them could also serve as a per-site target uncertainty, for weighting
   training rows or as an uncertainty output.
3. **Where is the headroom?** With width error several times the measurement noise, is the bottleneck
   the features (for example, the NWM 1-year flows, which the paper notes agree weakly with gages)
   or the model form? Would small streams benefit from their own model once there's enough small-stream data?
4. **Would a measurement-level model help?** That is, f(reach attributes, flow) = width, trained on all
   measurements instead of per-site targets. Sites with only a few measurements, like Silverway, would
   contribute, and sites with no high-flow data could borrow from similar ones.

## Caveats

- New-site counts assume the existing R² pass rate (75%); small or new sites could pass less often. About 14
  records the USGS API couldn't serve were skipped.
- I fit plain log-log power laws, not the AHGestimation ensemble, so absolute target values differ
  from the published ones; the comparisons are relative.
- Model error is approximated from the paper's reported R², not computed site by site.
- The noise floor reflects variation within a few hundred feet of each gage, not along a whole reach.

## Code

Everything above is reproducible from public data (USGS Water Data APIs, the RiverShapeML repo and
HydroShare outputs): **[repo link]**. I'd enjoy talking through any of this, and I'd be happy to
prototype adding the measurements to your pipeline if it's useful.
