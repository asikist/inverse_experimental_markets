"""Session-clustered Wilcoxon tests.

Two variants, both paired and two-sided:

  (a) clustered   — Rosner-Glynn-Lee (RGL) clustered signed-rank test via R's
                    ``clusrank`` package.  Cluster unit is ``(treatment, game)``,
                    treating one experimental game as one session.  Operates on
                    all per-row paired differences within each bucket.

  (b) aggregated  — per-cluster median of the paired difference ``d``, then
                    ``scipy.stats.wilcoxon`` across clusters.  One number per
                    game enters the test, so independence at the game level is
                    respected by construction; power is lower than RGL.

Holm-Bonferroni adjustment is applied to p-values within each task and each
variant (across the 4 buckets x 6 pairs = 24 tests).

Outputs
-------
  code/data/results/wilcoxon_clustered/
      {ae,ce}_results.ft          per-pair x bucket records, all metrics
      diagnostics.json            per-bucket summaries (K, N, %zeros, ...)

LaTeX tables are rendered in a separate postprocess:
``uv run python scripts/wilcoxon_clustered_tables.py``.

Run: ``uv run python scripts/wilcoxon_clustered.py``
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent
R_SCRIPT = HERE / "clustered_wilcoxon.R"

OUT_DIR = CODE_ROOT / "data" / "results" / "wilcoxon_clustered"

MODELS = ["EMH", "CEMH", "OB-RLM", "GBT"]
RESULT_FILES = {
    "ce": {
        "EMH": "data/results/ce_price/emh.ft",
        "CEMH": "data/results/ce_price/cemh.ft",
        "OB-RLM": "data/results/ce_price/ob_rlm.ft",
        "GBT": "data/results/ce_price/gbt.ft",
    },
    "ae": {
        "EMH": "data/results/allocative_efficiency/emh.ft",
        "CEMH": "data/results/allocative_efficiency/cemh.ft",
        "OB-RLM": "data/results/allocative_efficiency/ob_rlm.ft",
        "GBT": "data/results/allocative_efficiency/gbt.ft",
    },
}
APE_COL = {"ce": "ce_ape", "ae": "ae_ape"}
DEALS_LABEL = {"ce": "Price realizations", "ae": "# Total Deals"}

BUCKETS = [("1", "0"), ("1", "1+"), ("2+", "0"), ("2+", "1+")]


def load_sample_orient(task: str) -> pd.DataFrame:
    """Return a wide frame with one row per sample x trial and one column per model."""
    frames = []
    for m, rel in RESULT_FILES[task].items():
        df = pd.read_feather(CODE_ROOT / rel).copy()
        df["model"] = m
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    idx = ["sample_id", "treatment", "game", "round", "time", "n_unique_deals_round"]
    wide = long.set_index(idx + ["model"])[APE_COL[task]].unstack("model")[MODELS]
    wide = wide.reset_index()
    wide["Round"] = np.where(wide["round"] > 1, "2+", "1")
    wide["Deals"] = np.where(wide["n_unique_deals_round"] >= 1, "1+", "0")
    wide["cluster"] = pd.factorize(
        wide["treatment"].astype(str) + "|" + wide["game"].astype(str)
    )[0]
    return wide


def bucket_mask(df: pd.DataFrame, round_b: str, deal_b: str) -> pd.Series:
    return (df["Round"] == round_b) & (df["Deals"] == deal_b)


def aggregated_wilcoxon(d: pd.Series, cluster: pd.Series) -> tuple[float, float, int]:
    """Per-cluster median of d, paired Wilcoxon across clusters, two-sided."""
    per_cluster = pd.DataFrame({"d": d.values, "c": cluster.values}).groupby("c")["d"].median()
    n_cl = int(per_cluster.size)
    if n_cl < 2 or (per_cluster == 0).all():
        return (float("nan"), float("nan"), n_cl)
    try:
        stat, p = scipy.stats.wilcoxon(per_cluster.values, alternative="two-sided")
    except ValueError:
        return (float("nan"), float("nan"), n_cl)
    return (float(stat), float(p), n_cl)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni p-value adjustment. NaNs are preserved."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    m = int(mask.sum())
    if m == 0:
        return out
    order = np.argsort(p[mask])
    sorted_p = p[mask][order]
    adj = np.minimum.accumulate((m - np.arange(m)) * sorted_p[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out_mask = np.empty(m)
    out_mask[order] = adj
    out[mask] = out_mask
    return out


def run_clustered_tests(long_csv: Path, out_csv: Path) -> pd.DataFrame:
    cmd = ["Rscript", str(R_SCRIPT), str(long_csv), str(out_csv)]
    print(f"[py] calling: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return pd.read_csv(out_csv)


def build_records(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-pair x bucket records for one task.

    Returns (records, long_d_frame).  ``long_d_frame`` is the table passed to
    ``clustered_wilcoxon.R``.
    """
    wide = load_sample_orient(task)
    records = []
    long_rows = []
    for rd, dl in BUCKETS:
        sub = wide[bucket_mask(wide, rd, dl)]
        for A, B in combinations(MODELS, 2):
            d = sub[A] - sub[B]
            finite = d.notna()
            dd = d[finite]
            cc = sub["cluster"][finite]
            n_obs = int(dd.size)
            n_clusters = int(cc.nunique())
            median_diff = float(dd.median()) if n_obs else float("nan")
            n_zeros = int((dd == 0).sum())
            stat_agg, p_agg, n_cl_agg = aggregated_wilcoxon(dd, cc)
            gid = f"{task}|R{rd}|D{dl}|{A}|{B}"
            records.append(
                {
                    "task": task,
                    "round_bucket": rd,
                    "deal_bucket": dl,
                    "A": A,
                    "B": B,
                    "group_id": gid,
                    "n_obs": n_obs,
                    "n_clusters": n_clusters,
                    "n_zeros": n_zeros,
                    "median_diff": median_diff,
                    "agg_stat": stat_agg,
                    "agg_p": p_agg,
                    "agg_n_clusters": n_cl_agg,
                }
            )
            long_rows.append(
                pd.DataFrame(
                    {
                        "group_id": gid,
                        "cluster": cc.values.astype(np.int64),
                        "d": dd.values.astype(np.float64),
                    }
                )
            )
    rec = pd.DataFrame(records)
    long_df = pd.concat(long_rows, ignore_index=True)
    return rec, long_df




def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    diagnostics = {}
    all_records = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for task in ("ae", "ce"):
            records, long_df = build_records(task)
            in_csv = tmp / f"{task}_long.csv"
            out_csv = tmp / f"{task}_out.csv"
            long_df.to_csv(in_csv, index=False)
            rgl = run_clustered_tests(in_csv, out_csv)
            rgl = rgl.rename(columns={"Z": "rgl_Z", "p_value": "p_rgl",
                                       "n_zeros": "rgl_n_zeros"})
            rec = records.merge(
                rgl[["group_id", "rgl_Z", "p_rgl", "rgl_n_zeros"]],
                on="group_id", how="left",
            )
            rec["p_rgl_holm"] = holm_adjust(rec["p_rgl"].to_numpy())
            rec["agg_p_holm"] = holm_adjust(rec["agg_p"].to_numpy())

            out_ft = OUT_DIR / f"{task}_results.ft"
            rec.to_feather(out_ft)
            print(f"[py] wrote {out_ft}")

            diagnostics[task] = {
                "n_rows": int(rec["n_obs"].sum()),
                "n_clusters_max": int(rec["n_clusters"].max()),
                "n_clusters_min": int(rec["n_clusters"].min()),
                "n_zeros_total": int(rec["n_zeros"].sum()),
                "n_comparisons": int(len(rec)),
                "n_signif_clustered_raw": int((rec["p_rgl"] < 0.05).sum()),
                "n_signif_clustered_holm": int((rec["p_rgl_holm"] < 0.05).sum()),
                "n_signif_aggregated_raw": int((rec["agg_p"] < 0.05).sum()),
                "n_signif_aggregated_holm": int((rec["agg_p_holm"] < 0.05).sum()),
                "median_diff_range": [float(rec["median_diff"].min()),
                                       float(rec["median_diff"].max())],
            }
            all_records[task] = rec

    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[py] wrote {OUT_DIR / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
