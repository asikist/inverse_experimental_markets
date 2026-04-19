#!/usr/bin/env Rscript
# Clustered Wilcoxon signed-rank tests via clusrank (Rosner-Glynn-Lee).
#
# Input CSV columns: group_id (string), cluster (int), d (numeric).
# For each group_id, runs clusWilcox.test(d, cluster, paired=TRUE, method="rgl",
# alternative="two.sided") and reports Z, p-value, n_obs, n_clusters, n_zeros.
#
# Usage:
#   Rscript clustered_wilcoxon.R <input.csv> <output.csv>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: clustered_wilcoxon.R <input.csv> <output.csv>")
input_path <- args[1]
output_path <- args[2]

.libPaths(c(path.expand("~/R/library"), .libPaths()))
suppressPackageStartupMessages(library(clusrank))

df <- read.csv(input_path, stringsAsFactors = FALSE)
stopifnot(all(c("group_id", "cluster", "d") %in% names(df)))

run_one <- function(g, df) {
    sub <- df[df$group_id == g, ]
    n_obs <- nrow(sub)
    n_cl <- length(unique(sub$cluster))
    n_zeros <- sum(sub$d == 0)
    out <- data.frame(
        group_id = g, n_obs = n_obs, n_clusters = n_cl,
        n_zeros = n_zeros, Z = NA_real_, p_value = NA_real_,
        stringsAsFactors = FALSE
    )
    nz <- sub[sub$d != 0, ]
    if (nrow(nz) < 2 || length(unique(nz$cluster)) < 2) return(out)
    res <- tryCatch(
        clusWilcox.test(
            x = nz$d, cluster = nz$cluster,
            paired = TRUE, method = "rgl", alternative = "two.sided"
        ),
        error = function(e) NULL
    )
    if (!is.null(res)) {
        out$Z <- as.numeric(res$statistic)
        out$p_value <- as.numeric(res$p.value)
    }
    out
}

groups <- unique(df$group_id)
cat(sprintf("[clustered_wilcoxon.R] %d groups, %d rows total\n", length(groups), nrow(df)))
t0 <- Sys.time()
rows <- lapply(groups, run_one, df = df)
res <- do.call(rbind, rows)
cat(sprintf("[clustered_wilcoxon.R] done in %.1fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

write.csv(res, output_path, row.names = FALSE)
