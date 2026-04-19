"""Render LaTeX tables from per-pair clustered-Wilcoxon results.

Reads the feathers produced by ``wilcoxon_clustered.py`` and emits the four
4x4 pairwise tables consumed by the paper.  Separating this step keeps the
~90 s R pipeline decoupled from table formatting.

Run: ``uv run python scripts/wilcoxon_clustered_tables.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent

FINAL = REPO_ROOT / "paper" / "epj" / "reproducibility_report" / "final_tables"
IN_DIR = CODE_ROOT / "data" / "results" / "wilcoxon_clustered"

MODELS = ["EMH", "CEMH", "OB-RLM", "GBT"]
BUCKETS = [("1", "0"), ("1", "1+"), ("2+", "0"), ("2+", "1+")]
DEALS_LABEL = {"ce": "Price realizations", "ae": "# Total Deals"}


def pvalue_glyph(p: float) -> str:
    """1/2/3 stars, keeping the star-only convention of tab_{ae,cep}_ape_wilcoxon.tex
    while mapping to the three thresholds used in paper_artifacts._pvalue_glyph
    (>0.10 none, >0.05 *, >0.01 **, else ***). Only ``*`` is emitted so LaTeX
    text mode stays happy."""
    if not np.isfinite(p):
        return ""
    if p > 0.10:
        return ""
    if p > 0.05:
        return "*"
    if p > 0.01:
        return "**"
    return "***"


def render_tex(records: pd.DataFrame, variant: str, task: str) -> str:
    p_col = "p_rgl_holm" if variant == "clustered" else "agg_p_holm"
    deals_label = DEALS_LABEL[task]

    def cell(row) -> str:
        if not np.isfinite(row["median_diff"]):
            return ""
        diff = round(float(row["median_diff"]), 3)
        if diff == 0:
            return "n/a"
        glyph = pvalue_glyph(float(row[p_col]))
        body = f"{diff:,.3f}\\textsuperscript{{{glyph}}}" if glyph else f"{diff:,.3f}"
        if diff < 0:
            body = f"\\B {body}"
        return body

    mirrored = []
    for _, r in records.iterrows():
        mirrored.append(r.to_dict())
        flipped = r.to_dict()
        flipped["A"], flipped["B"] = r["B"], r["A"]
        flipped["median_diff"] = -r["median_diff"]
        mirrored.append(flipped)
    df = pd.DataFrame(mirrored)
    df["cell"] = df.apply(cell, axis=1)
    df["Round"] = df["round_bucket"]
    df[deals_label] = df["deal_bucket"]

    df["A"] = pd.Categorical(df["A"], MODELS)
    df["B"] = pd.Categorical(df["B"], MODELS)
    piv = (
        df.set_index(["Round", deals_label, "A", "B"])["cell"]
        .unstack("A")
        .reindex(columns=MODELS)
    )
    idx_order = [(rd, dl, m) for rd, dl in BUCKETS for m in MODELS]
    piv = piv.reindex(idx_order)
    piv = piv.rename_axis(index=["Round", deals_label, "Model B"])
    piv.columns.name = "Model A"
    piv = piv.fillna("")

    lines = ["\\begin{tabular}{lllllll}", "\\toprule"]
    lines.append(" &  &  & \\multicolumn{4}{r}{Median APE Difference} \\\\")
    lines.append(" &  & Model A & " + " & ".join(MODELS) + " \\\\")
    lines.append("Round & " + deals_label.replace("#", r"\#") + " & Model B &  &  &  &  \\\\")
    lines.append("\\midrule")
    rd_prev = dl_prev = None
    for i, ((rd, dl, mb), row) in enumerate(piv.iterrows()):
        cells = [str(row[m]) if row[m] else "" for m in MODELS]
        parts = []
        if rd != rd_prev:
            parts.append(f"\\multirow[c]{{8}}{{*}}{{{rd}}}")
            rd_prev = rd
        else:
            parts.append("")
        if (rd, dl) != (rd_prev, dl_prev) or dl != dl_prev:
            parts.append(f"\\multirow[c]{{4}}{{*}}{{{dl}}}")
            dl_prev = dl
        else:
            parts.append("")
        parts.append(mb)
        parts.extend(cells)
        lines.append(" & ".join(parts) + " \\\\")
        if (i + 1) % 4 == 0 and (i + 1) < len(piv):
            if (i + 1) % 8 == 0:
                lines.append("\\cline{1-7} \\cline{2-7}")
            else:
                lines.append("\\cline{2-7}")
    lines.append("\\cline{1-7} \\cline{2-7}")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    for task, task_tex in (("ae", "ae"), ("ce", "cep")):
        rec = pd.read_feather(IN_DIR / f"{task}_results.ft")
        for variant in ("clustered", "aggregated"):
            tex = render_tex(rec, variant, task)
            out = FINAL / f"tab_{task_tex}_ape_wilcoxon_{variant}.tex"
            out.write_text(tex)
            print(f"[py] wrote {out}")


if __name__ == "__main__":
    main()
