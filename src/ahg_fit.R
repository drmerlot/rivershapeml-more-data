# Fit at-a-station hydraulic geometry (AHG) per site with AHGestimation (Johnson et al.; NOAA OWP),
# the package RiverShapeML used to build its training targets.
#
# Input : data/derived/ahg_input.parquet with columns site_no, set, Q, TW, Y, V (SI: m3/s, m, m, m/s).
#         `set` is "all" (all USGS measurements; TW, Y, V) or "hs" (HYDRoSWOT records; TW only).
# Steps : per site and set: mad_filter() then qva_filter() (when Y and V are present), as in the
#         package README, then ahg_estimate(); keep the best viable solution.
# Output: data/derived/ahg_fits.parquet (one row per site/set: coefficients, exponents, method,
#         viability, n before/after filtering).
#
# Usage : R_LIBS_USER=r_lib Rscript src/ahg_fit.R [n_cores]

suppressPackageStartupMessages({
  library(AHGestimation)
  library(arrow)
  library(dplyr)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
n_cores <- if (length(args) > 0) as.integer(args[1]) else max(1, detectCores() - 2)
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
inp <- read_parquet(file.path(root, "data/derived/ahg_input.parquet"))
groups <- split(inp, interaction(inp$site_no, inp$set, drop = TRUE))
cat(length(groups), "site/set groups on", n_cores, "cores\n")

fit_one <- function(g) {
  site <- g$site_no[1]; set <- g$set[1]
  out <- data.frame(site_no = site, set = set, n_in = nrow(g), n_used = NA_integer_,
                    TW_coef = NA_real_, TW_exp = NA_real_, Y_coef = NA_real_, Y_exp = NA_real_,
                    V_coef = NA_real_, V_exp = NA_real_, TW_method = NA_character_,
                    viable = NA, error = NA_character_)
  res <- tryCatch({
    df <- if (set == "hs") select(g, Q, TW) else select(g, Q, TW, Y, V)
    df <- as.data.frame(df)
    df <- mad_filter(df)
    if (all(c("Y", "V") %in% names(df))) df <- qva_filter(df)
    if (nrow(df) < 5) stop("fewer than 5 measurements after filtering")
    fit <- ahg_estimate(df)
    out$n_used <- nrow(df)
    if (set == "hs") {
      best <- fit[fit$type == "TW", ][1, ]
      out$TW_coef <- best$coef; out$TW_exp <- best$exp; out$TW_method <- best$method; out$viable <- TRUE
    } else {
      best <- fit[1, ]
      out$TW_coef <- best$TW_coef; out$TW_exp <- best$TW_exp
      out$Y_coef <- best$Y_coef;   out$Y_exp <- best$Y_exp
      out$V_coef <- best$V_coef;   out$V_exp <- best$V_exp
      out$TW_method <- best$TW_method
      out$viable <- if ("viable" %in% names(best)) best$viable else TRUE
    }
    out
  }, error = function(e) { out$error <- conditionMessage(e); out })
  res
}

t0 <- Sys.time()
fits <- bind_rows(mclapply(groups, fit_one, mc.cores = n_cores))
write_parquet(fits, file.path(root, "data/derived/ahg_fits.parquet"))
cat("done in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min;",
    sum(is.na(fits$error)), "fitted,", sum(!is.na(fits$error)), "failed\n")
print(table(fits$set, is.na(fits$error)))
