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
