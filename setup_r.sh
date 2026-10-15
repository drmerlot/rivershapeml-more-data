#!/usr/bin/env bash
# Install the R packages used to fit at-a-station hydraulic geometry (AHGestimation) into a project-local
# library (r_lib/). Needs R >= 4.2 (e.g. `brew install --cask r` on macOS).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p r_lib
SHA=$(cat r_ahg_commit.txt)
R_LIBS_USER="$PWD/r_lib" Rscript -e "
options(repos = c(CRAN = 'https://cloud.r-project.org'))
install.packages(c('remotes', 'DescTools', 'dplyr', 'geodist', 'mco', 'pbapply', 'sf', 'arrow'), lib = 'r_lib')
remotes::install_github('mikejohnson51/AHGestimation@$SHA', lib = 'r_lib', upgrade = 'never', dependencies = FALSE)
library(AHGestimation, lib.loc = 'r_lib'); cat('AHGestimation', as.character(packageVersion('AHGestimation', lib.loc = 'r_lib')), 'ready\n')"
