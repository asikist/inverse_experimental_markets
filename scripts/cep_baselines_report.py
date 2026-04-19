"""Reproducibility-friendly report for the CEP appendix baselines.

Reads the four paper model results plus the two baseline .ft files produced
by ``scripts/cep_baselines.py`` and writes:

  data/results/ce_price/baseline_diagnostics/
      trainset_vs_testset_ce_gap.csv   -- per (sample_id, treatment) train/test
                                          mean ce_round and their % gap.
      loto_median_ape.csv              -- leave-one-treatment-out median APE
                                          for the treatment-mean predictor,
                                          per treatment.
      summary.json                     -- headline numbers used in the paper
                                          text (overall medians, brittleness).

  paper/epj/tables/
      tab_cep_baseline_comparison.tex  -- 6 rows (EMH / CEMH / OB-RLM / GBT /
                                          treatment_mean / book_midpoint)
                                          x 4 buckets (Round in {1, 2+} x
                                          Deals in {0, 1+}), median CE APE.
      tab_cep_baseline_brittleness.tex -- two-row brittleness summary for the
                                          treatment-mean baseline (worst-case
                                          split and LOTO).

Run with: ``uv run python scripts/cep_baselines_report.py``. Assumes
``cep_baselines.py`` has already been run (or re-runs it on demand).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent

PREP = CODE_ROOT / "data" / "preprocessed"
CE_DIR = CODE_ROOT / "data" / "results" / "ce_price"
DIAG_DIR = CE_DIR / "baseline_diagnostics"
FINAL_TABLES = REPO_ROOT / "paper" / "epj" / "tables"

MODELS = [
    ("EMH",            "emh.ft"),
    ("CEMH",           "cemh.ft"),
    ("OB-RLM",         "ob_rlm.ft"),
    ("GBT",            "gbt.ft"),
    ("Treatment-Mean", "treatment_mean_ce.ft"),
    ("Book-Midpoint",  "book_midpoint.ft"),
]


def ensure_baselines_exist() -> None:
    missing = [name for name, fn in MODELS if not (CE_DIR / fn).exists()]
    if missing:
        print(f"Generating missing baseline feathers via cep_baselines.py: {missing}")
        subprocess.run([sys.executable, str(HERE / "cep_baselines.py")], check=True)


def load_all_models() -> pd.DataFrame:
    frames = []
    for name, fn in MODELS:
        df = pd.read_feather(CE_DIR / fn).copy()
        df["model"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# (1) Appendix comparison table
# ---------------------------------------------------------------------------

def comparison_pivot(all_df: pd.DataFrame) -> pd.DataFrame:
    df = all_df.copy()
    df["Round"] = np.where(df["round"] > 1, "2+", "1")
    df["Price realizations"] = np.where(df["n_unique_deals_round"] >= 1, "1+", "0")
    g = df.groupby(["model", "Round", "Price realizations"], observed=True)["ce_ape"].median()
    piv = g.unstack(["Round", "Price realizations"])
    model_order = [name for name, _ in MODELS]
    return piv.reindex(model_order)


def write_comparison_latex(piv: pd.DataFrame, out_path: Path) -> None:
    cols = [("1", "0"), ("1", "1+"), ("2+", "0"), ("2+", "1+")]
    header = (
        "\\begin{tabular}{l cccc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Round 1} & \\multicolumn{2}{c}{Round $\\geq 2$} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Model & 0 deals & $\\geq 1$ deal & 0 deals & $\\geq 1$ deal \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for model in piv.index:
        vals = [piv.loc[model, c] for c in cols]
        formatted = [f"{v:.3f}" for v in vals]
        label = model
        if model in {"Treatment-Mean", "Book-Midpoint"}:
            label = f"\\textit{{{model}}}"
        rows.append(f"{label} & {' & '.join(formatted)} \\\\")
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(rows) + footer)


# ---------------------------------------------------------------------------
# (2) Brittleness diagnostics
# ---------------------------------------------------------------------------

def train_test_gap(tad: pd.DataFrame, tts: pd.DataFrame) -> pd.DataFrame:
    uniq = tad.drop_duplicates(["treatment", "game", "round"])[["treatment", "game", "round", "ce_round"]]
    rows = []
    for (sid, tr), g in tts.groupby(["sample_id", "treatment"]):
        train_games = g.loc[g["dataset_type"] == "train", "game"].tolist()
        test_games = g.loc[g["dataset_type"] == "test", "game"].tolist()
        tr_ce = uniq.loc[(uniq["treatment"] == tr) & uniq["game"].isin(train_games), "ce_round"]
        te_ce = uniq.loc[(uniq["treatment"] == tr) & uniq["game"].isin(test_games), "ce_round"]
        if len(tr_ce) == 0 or len(te_ce) == 0:
            continue
        rows.append({
            "sample_id": sid,
            "treatment": tr,
            "train_mean_ce": tr_ce.mean(),
            "test_mean_ce": te_ce.mean(),
            "gap_pct": 100.0 * (te_ce.mean() - tr_ce.mean()) / tr_ce.mean(),
        })
    return pd.DataFrame(rows)


def worst_split_tmean_ape(tmean_res: pd.DataFrame, gap_df: pd.DataFrame) -> dict:
    worst = gap_df.reindex(gap_df["gap_pct"].abs().sort_values(ascending=False).index).iloc[0]
    sid, tr = int(worst["sample_id"]), worst["treatment"]
    sl = tmean_res.query("sample_id == @sid and treatment == @tr")
    return {
        "sample_id": sid,
        "treatment": tr,
        "train_mean_ce": float(worst["train_mean_ce"]),
        "test_mean_ce": float(worst["test_mean_ce"]),
        "gap_pct": float(worst["gap_pct"]),
        "treatment_mean_median_ape": float(sl["ce_ape"].median()),
        "n_rows": int(len(sl)),
    }


def loto_treatment_mean(tad: pd.DataFrame, emh_grid: pd.DataFrame) -> pd.DataFrame:
    uniq = tad.drop_duplicates(["treatment", "game", "round"])[["treatment", "ce_round"]]
    treatments = uniq["treatment"].unique()
    global_excl = {t: uniq.loc[uniq["treatment"] != t, "ce_round"].mean() for t in treatments}

    merged = emh_grid.merge(
        tad[["treatment", "game", "round", "time", "ce_round"]],
        on=["treatment", "game", "round", "time"],
    )
    merged["loto_pred"] = merged["treatment"].map(global_excl)
    merged["loto_ape"] = np.abs(merged["loto_pred"] - merged["ce_round"]) / merged["ce_round"]

    per_trt = merged.groupby("treatment")["loto_ape"].median().rename("loto_median_ape").reset_index()
    per_trt["loto_prediction"] = per_trt["treatment"].map(global_excl)
    return per_trt, float(merged["loto_ape"].median())


def write_brittleness_latex(worst: dict, loto_rows: pd.DataFrame, loto_overall: float,
                            tmean_res: pd.DataFrame, out_path: Path) -> None:
    overall_tmean = float(tmean_res["ce_ape"].median())
    top3 = loto_rows.sort_values("loto_median_ape", ascending=False).head(3)
    top3_str = ", ".join(f"{r['treatment']} = {r['loto_median_ape']:.3f}" for _, r in top3.iterrows())
    body = (
        "\\begin{tabular}{p{0.78\\linewidth} c}\n"
        "\\toprule\n"
        "Scenario & Median APE on CEP \\\\\n"
        "\\midrule\n"
        f"Treatment-Mean (aggregate, 50 splits) & {overall_tmean:.3f} \\\\\n"
        f"Treatment-Mean on the worst-asymmetry split "
        f"(sample \\#{worst['sample_id']}, treatment {worst['treatment']}; "
        f"train/test mean gap {worst['gap_pct']:+.1f}\\%) "
        f"& {worst['treatment_mean_median_ape']:.3f} \\\\\n"
        f"Treatment-Mean under leave-one-treatment-out (novel treatment at test time) & {loto_overall:.3f} \\\\\n"
        f"\\quad Three worst LOTO treatments: {top3_str} & --- \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_baselines_exist()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)

    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tts = pd.read_feather(PREP / "train_test_split.ft")
    emh = pd.read_feather(CE_DIR / "emh.ft")
    tmean_res = pd.read_feather(CE_DIR / "treatment_mean_ce.ft")

    all_df = load_all_models()
    piv = comparison_pivot(all_df)
    piv.to_csv(DIAG_DIR / "median_ape_comparison.csv")
    write_comparison_latex(piv, FINAL_TABLES / "tab_cep_baseline_comparison.tex")

    gap_df = train_test_gap(tad, tts)
    gap_df.to_csv(DIAG_DIR / "trainset_vs_testset_ce_gap.csv", index=False)
    worst = worst_split_tmean_ape(tmean_res, gap_df)

    loto_rows, loto_overall = loto_treatment_mean(tad, emh)
    loto_rows.to_csv(DIAG_DIR / "loto_median_ape.csv", index=False)

    write_brittleness_latex(worst, loto_rows, loto_overall, tmean_res,
                            FINAL_TABLES / "tab_cep_baseline_brittleness.tex")

    summary = {
        "overall_median_ape": {m: float(piv.loc[m].mean()) for m in piv.index},
        "treatment_mean_overall_median_ape": float(tmean_res["ce_ape"].median()),
        "book_midpoint_overall_median_ape": float(pd.read_feather(CE_DIR / "book_midpoint.ft")["ce_ape"].median()),
        "worst_asymmetric_split": worst,
        "leave_one_treatment_out_overall_median_ape": loto_overall,
        "leave_one_treatment_out_per_treatment": loto_rows.to_dict(orient="records"),
        "train_test_gap_summary_pct": {
            "median_abs": float(gap_df["gap_pct"].abs().median()),
            "p95_abs": float(gap_df["gap_pct"].abs().quantile(0.95)),
            "max_abs": float(gap_df["gap_pct"].abs().max()),
        },
    }
    (DIAG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote diagnostics:")
    print(f"  {DIAG_DIR}/")
    print(f"  {FINAL_TABLES}/tab_cep_baseline_comparison.tex")
    print(f"  {FINAL_TABLES}/tab_cep_baseline_brittleness.tex")
    print("Headline:")
    print(f"  Treatment-Mean overall median APE: {summary['treatment_mean_overall_median_ape']:.3f}")
    print(f"  Book-Midpoint  overall median APE: {summary['book_midpoint_overall_median_ape']:.3f}")
    print(f"  Treatment-Mean on worst split    : {worst['treatment_mean_median_ape']:.3f}")
    print(f"  Treatment-Mean under LOTO        : {loto_overall:.3f}")


if __name__ == "__main__":
    main()
