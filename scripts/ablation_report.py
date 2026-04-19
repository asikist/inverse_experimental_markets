"""Build the appendix ablation tables for the orderbook-only and no-deal-price ablations.

Reads the full-model reference results from ``data/results/{ce_price,
allocative_efficiency}/`` and the ablation feathers from
``data/results/ablations/{r25,r26}/``, computes median APE broken down by
round and price-realisation bucket (the same buckets as Tabs.\\ \\ref{tab:
ae_ape} and \\ref{tab:cep_ape} in the paper), and writes the LaTeX tables to
``paper/epj/reproducibility_report/final_tables/``.

Run with: ``uv run python scripts/ablation_report.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent

CE_DIR = CODE_ROOT / "data" / "results" / "ce_price"
AE_DIR = CODE_ROOT / "data" / "results" / "allocative_efficiency"
ABL = CODE_ROOT / "data" / "results" / "ablations"
FINAL = REPO_ROOT / "paper" / "epj" / "reproducibility_report" / "final_tables"


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Round"] = np.where(df["round"] > 1, "2+", "1")
    df["Price realizations"] = np.where(df["n_unique_deals_round"] >= 1, "1+", "0")
    return df


def pivot(df: pd.DataFrame, err: str) -> pd.DataFrame:
    df = add_buckets(df)
    g = df.groupby(["Round", "Price realizations"], observed=True)[err].median()
    return g.unstack(["Round", "Price realizations"]).reindex(
        index=["1", "2+"], columns=None)


def median_row(df: pd.DataFrame, err: str) -> dict[tuple[str, str], float]:
    df = add_buckets(df)
    out = {}
    for (rd, pr), sub in df.groupby(["Round", "Price realizations"]):
        out[(rd, pr)] = float(sub[err].median())
    return out


ORDER = [("1", "0"), ("1", "1+"), ("2+", "0"), ("2+", "1+")]


def format_row(label: str, data: dict, err_fmt: str = "{:.3f}") -> str:
    vals = [err_fmt.format(data[c]) if c in data else "---" for c in ORDER]
    return f"{label} & " + " & ".join(vals) + " \\\\"


def write_r25_table(out: Path) -> dict:
    full = {
        ("AE", "OB-RLM"): pd.read_feather(AE_DIR / "ob_rlm.ft"),
        ("AE", "GBT"): pd.read_feather(AE_DIR / "gbt.ft"),
        ("CEP", "CEMH"): pd.read_feather(CE_DIR / "cemh.ft"),
        ("CEP", "GBT"): pd.read_feather(CE_DIR / "gbt.ft"),
    }
    abl = {
        ("AE", "OB-RLM"): pd.read_feather(ABL / "r25" / "ob_rlm_ae.ft"),
        ("AE", "GBT"): pd.read_feather(ABL / "r25" / "gbt_ae.ft"),
        ("CEP", "CEMH"): pd.read_feather(ABL / "r25" / "cemh_global_cep.ft"),
        ("CEP", "GBT"): pd.read_feather(ABL / "r25" / "gbt_cep.ft"),
    }

    rows = []
    all_rows = {}
    for (target, model), df in full.items():
        err = "ce_ape" if target == "CEP" else "ae_ape"
        data_full = median_row(df, err)
        data_abl = median_row(abl[(target, model)], err)
        rows.append((f"{model} ({target})", data_full, data_abl))
        all_rows[(target, model)] = (data_full, data_abl)

    body = (
        "\\begin{tabular}{l l cccc}\n"
        "\\toprule\n"
        " & & \\multicolumn{2}{c}{Round 1} & \\multicolumn{2}{c}{Round $\\geq 2$} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Model & Variant & 0 deals & $\\geq 1$ deal & 0 deals & $\\geq 1$ deal \\\\\n"
        "\\midrule\n"
    )
    for idx, (label, data_full, data_abl) in enumerate(rows):
        body += format_row(f"\\multirow{{2}}{{*}}{{{label}}} & original", data_full) + "\n"
        body += format_row("& orderbook-only", data_abl) + "\n"
        if idx < len(rows) - 1:
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    return {f"{t}:{m}": {"full": v[0], "ablated": v[1]} for (t, m), v in all_rows.items()}


def write_r26_table(out: Path) -> dict:
    full = pd.read_feather(AE_DIR / "ob_rlm.ft")
    abl = pd.read_feather(ABL / "r26" / "ob_rlm_ae.ft")
    data_full = median_row(full, "ae_ape")
    data_abl = median_row(abl, "ae_ape")

    body = (
        "\\begin{tabular}{l cccc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Round 1} & \\multicolumn{2}{c}{Round $\\geq 2$} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Variant & 0 deals & $\\geq 1$ deal & 0 deals & $\\geq 1$ deal \\\\\n"
        "\\midrule\n"
    )
    body += format_row("OB-RLM (AE), with realised price", data_full) + "\n"
    body += format_row("OB-RLM (AE), without realised price", data_abl) + "\n"
    body += "\\bottomrule\n\\end{tabular}\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    return {"full": data_full, "ablated": data_abl}


def write_r25_cep_vs_treatment_mean_note() -> None:
    """Document that the CEMH-CEP-without-realized_price variant reduces to
    Treatment-Mean (already reported in the appendix baselines table)."""
    pass


def main() -> None:
    out_r25 = FINAL / "tab_ablation_r25.tex"
    out_r26 = FINAL / "tab_ablation_r26.tex"

    summary_r25 = write_r25_table(out_r25)
    summary_r26 = write_r26_table(out_r26)

    out_diag = ABL / "ablation_summary.json"
    out_diag.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(d: dict) -> dict:
        return {str(k): v for k, v in d.items()}

    out_diag.write_text(json.dumps({
        "r25": {k: {"full": _fmt(v["full"]), "ablated": _fmt(v["ablated"])}
                for k, v in summary_r25.items()},
        "r26": {"full": _fmt(summary_r26["full"]),
                "ablated": _fmt(summary_r26["ablated"])},
    }, indent=2))

    print(f"Wrote {out_r25}")
    print(f"Wrote {out_r26}")
    print(f"Wrote {out_diag}")


if __name__ == "__main__":
    main()
