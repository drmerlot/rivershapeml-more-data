#!/usr/bin/env bash
# Download everything the analysis needs into data/raw/. Re-runnable; downloads resume.
# The USGS downloads need a free api.data.gov key (https://api.waterdata.usgs.gov/signup/)
# in .env or the API_USGS_PAT environment variable; without one, USGS rate-limits after ~1,000 requests.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw/repo data/raw/rivershapeml data/derived

# RiverShapeML training data (HYDRoSWOT copy + training targets), pinned to the commit analyzed
if [ ! -d data/raw/repo/RiverShapeML ]; then
  git clone -q https://github.com/lynker-spatial/RiverShapeML.git data/raw/repo/RiverShapeML
  git -C data/raw/repo/RiverShapeML checkout -q 29c1209
fi

# Published RiverShapeML outputs for 2.7M reaches (HydroShare d147fcf554a54b2aaa4f146f85da0e03, ~76 MB)
[ -f data/raw/rivershapeml/ml_channel_dimension.parquet ] || curl -sL -o data/raw/rivershapeml/ml_channel_dimension.parquet \
  "https://www.hydroshare.org/resource/d147fcf554a54b2aaa4f146f85da0e03/data/contents/ml_channel_dimension.parquet"

# USGS Water Data APIs: stream sites, daily-discharge series, and every channel measurement (~4.8M rows, ~1 hr)
python src/fetch_usgs_ogc.py monitoring-locations
python src/fetch_usgs_ogc.py time-series-metadata
python src/fetch_usgs_ogc.py channel-measurements

# E2NHDPlusV2 / NWM v2.1 network attributes (USGS ScienceBase 63cb311ed34e06fef14f40a3, ~170 MB)
[ -f data/raw/enhd/enhd_nhdplusatts.parquet ] || curl -sL -o data/raw/enhd/enhd_nhdplusatts.parquet \
  "https://www.sciencebase.gov/catalog/file/get/63cb311ed34e06fef14f40a3?f=__disk__fa%2Fb9%2F03%2Ffab9035a4ece1500fdb1ead631a9f364d34d1e30"
python src/fetch_usgs_ogc.py peaks

# NWM v2.1 retrospective hourly flows at ~18k USGS gages, 1979-2020 (USGS ScienceBase 612e264ed34e40dd9c091228, 2.2 GB)
mkdir -p data/raw/nwm_gages
[ -f data/raw/nwm_gages/nwmv21_nwis.nc ] || curl -sL -o data/raw/nwm_gages/nwmv21_nwis.nc \
  "https://www.sciencebase.gov/catalog/file/get/612e264ed34e40dd9c091228?f=__disk__1d%2F9c%2F75%2F1d9c75f22b026a5d676c174ad73b46dfb4a367f7"

# RiverShapeML deployment (NOAA-OWP/predict-riverML): inputs for all ~2.65M reaches (~1.3 GB) and the
# deployed XGBoost models, reassembled the way their deploy.bash does (cat parts | unxz)
R=data/raw/predict-riverML
mkdir -p "$R"
curl -s "https://api.github.com/repos/NOAA-OWP/predict-riverML/git/trees/main?recursive=1" | python -c "
import sys, json
for t in json.load(sys.stdin)['tree']:
    p = t['path']
    if t['type'] == 'blob' and (p.startswith(('data/', 'models/', 'model_space/')) or p in ('data_loader.py', 'inference.py')):
        print(p)" | while read -r f; do
  mkdir -p "$R/$(dirname "$f")"
  [ -s "$R/$f" ] || curl -sfL --retry 3 -o "$R/$f" "https://raw.githubusercontent.com/NOAA-OWP/predict-riverML/main/$f"
done
[ -f "$R/data/ml_inputs.parquet" ] || cat "$R"/data/ml_inputs_split_part_*.xz | xz -d > "$R/data/ml_inputs.parquet"
for m in TW_bf Y_bf r; do
  f="$R/models/trained_xgboost_model_update_${m}_final.pickle.dat"
  [ -f "$f" ] || cat "$R"/models/trained_xgboost_model_update_${m}_final_split_part_*.xz | xz -d > "$f"
done
