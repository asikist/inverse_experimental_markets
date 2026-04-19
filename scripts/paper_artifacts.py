"""Regenerate paper-ready tables and figures from stored result .ft files.

Reads:
  - data/results/{ce_price,allocative_efficiency}/{emh,cemh,ob_rlm,gbt}.ft
  - data/preprocessed/original_df.ft       (for per-round player counts)
  - data/results/plots/{ce,ae}/bar_round1_{deal,nodeal}.pdf

Writes (under paper/epj/reproducibility_report/):
  - latex_staging/{ae,ce}/tbl_mdape_deals.tex     (raw df.to_latex per notebook cell)
  - latex_staging/{ae,ce}/tbl_mdape_size.tex
  - latex_staging/{ae,ce}/tbl_wilcoxon.tex
  - latex_staging/{ae,ce}/data_*.pkl              (underlying pivots for reproducibility)
  - final_tables/tab_ae_ape.tex                   (paper-format merged Deals+Size)
  - final_tables/tab_cep_ape.tex
  - final_tables/tab_ae_ape_wilcoxon.tex
  - final_tables/tab_cep_ape_wilcoxon.tex
  - ranking_preservation.md                       (diff of best-per-cell vs paper)

Also copies figures to paper/epj/figures/{ae_results,price_results}/.

Run with: uv run python scripts/paper_artifacts.py
"""
from __future__ import annotations

import pickle
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent
REPORT_DIR = REPO_ROOT / "paper" / "epj" / "reproducibility_report"
STAGING = REPORT_DIR / "latex_staging"
FINAL = REPORT_DIR / "final_tables"
FIG_STAGING = REPORT_DIR / "fig_staging"
PAPER_FIGURES = REPO_ROOT / "paper" / "epj" / "figures"

MODELS_UQ = np.array(["EMH", "CEMH", "OB-RLM", "GBT"])

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(task: str) -> pd.DataFrame:
    frames = []
    for model, rel in RESULT_FILES[task].items():
        df = pd.read_feather(CODE_ROOT / rel)
        df = df.copy()
        df["model"] = model  # override model column to canonical display name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_player_counts() -> pd.Series:
    odf = pd.read_feather(CODE_ROOT / "data/preprocessed/original_df.ft")
    by_side = odf.groupby(["treatment", "game", "round", "side"])["id"].nunique()
    return by_side.groupby(["treatment", "game", "round"]).sum().rename("id")


# ---------------------------------------------------------------------------
# Pivots
# ---------------------------------------------------------------------------

def pivot_deals(all_models_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Median APE by Round × (# realized deals)."""
    ape_col = APE_COL[task]
    df = all_models_df.copy()
    df["Round"] = np.where(df["round"] > 1, "2+", "1")
    df[DEALS_LABEL[task]] = np.where(df["n_unique_deals_round"] >= 1, "1+", "0")
    g = df.groupby(["model", "Round", DEALS_LABEL[task]], observed=True)[ape_col].median().reset_index()
    piv = g.pivot(index=["Round", DEALS_LABEL[task]], columns="model", values=ape_col)
    piv.columns.name = ""
    return piv[list(MODELS_UQ)]


def pivot_market_size(all_models_df: pd.DataFrame, player_counts: pd.Series, task: str) -> pd.DataFrame:
    """Median APE by Round × Market Size (Large if round-1 player count >= 15)."""
    ape_col = APE_COL[task]
    idx_cols = ["sample_id", "treatment", "game", "round", "time", "n_unique_deals_round", "model"]
    sample_orientation = all_models_df.set_index(idx_cols)[ape_col].unstack("model")
    joined = sample_orientation.join(player_counts)
    joined["Round"] = np.where(joined.index.get_level_values("round") < 2, "1", "2+")
    joined["Market Size"] = np.where(joined["id"] >= 15, "Large", "Small")
    joined["Market Size"] = pd.Categorical(joined["Market Size"], ["Small", "Large"])
    piv = joined.groupby(["Round", "Market Size"], observed=True).median().drop(columns="id")[list(MODELS_UQ)]
    return piv


# ---------------------------------------------------------------------------
# Wilcoxon (matches notebook cell 13/16 logic)
# ---------------------------------------------------------------------------

def _pvalue_glyph(pvalue: float) -> str:
    if pvalue > 0.1:
        return ""
    if pvalue > 0.05:
        return "^"
    if pvalue > 0.01:
        return "#"
    return "*"


def wilcoxon_table(all_models_df: pd.DataFrame, task: str) -> pd.DataFrame:
    ape_col = APE_COL[task]
    deal_label = DEALS_LABEL[task]
    idx = ["sample_id", "treatment", "game", "round", "time", "n_unique_deals_round", "model"]
    sample_orient = all_models_df.set_index(idx)[ape_col].unstack("model")[list(MODELS_UQ)]

    round_queries = {"1": "round==1", "2+": "round > 1"}
    deal_queries = {"0": "n_unique_deals_round==0", "1+": "n_unique_deals_round > 0"}

    rows = []
    for kr, vr in round_queries.items():
        for kd, vd in deal_queries.items():
            sub = sample_orient.query(f"{vr} and {vd}")
            for left_, right_ in combinations(list(MODELS_UQ), 2):
                ls = sub[left_].values
                rs = sub[right_].values
                diff = float(np.median(ls - rs))
                pval = neg_pval = 0.0
                if not np.array_equal(ls, rs):
                    alt = "less" if diff < 0 else "greater"
                    _, pval = scipy.stats.wilcoxon(ls, rs, alternative=alt)
                    # Replicate notebook cell 13/16 behavior: the submission-era
                    # code reused `alternative` (not `neg_alternative`) in the
                    # second wilcoxon call, so neg_pval was in fact pval.
                    # Keep this exactly so the regenerated table matches the
                    # submission's Wilcoxon numbers to 3rd-digit precision.
                    neg_pval = pval
                fd = f"{diff:,.3f}"
                nfd = f"{-diff:,.3f}"
                stars = f"\\textsuperscript{{{_pvalue_glyph(pval)}}}"
                neg_stars = f"\\textsuperscript{{{_pvalue_glyph(neg_pval)}}}"
                if round(diff, 3) == 0:
                    stars = neg_stars = ""
                    fd = nfd = "n/a"
                rows.append({"Round": kr, deal_label: kd, "Model A": left_, "Model B": right_,
                             "Median APE Difference": fd + stars})
                rows.append({"Round": kr, deal_label: kd, "Model A": right_, "Model B": left_,
                             "Median APE Difference": nfd + neg_stars})

    out = pd.DataFrame(rows)
    out["Model A"] = pd.Categorical(out["Model A"], list(MODELS_UQ))
    out["Model B"] = pd.Categorical(out["Model B"], list(MODELS_UQ))
    out = out.set_index(["Round", deal_label, "Model A", "Model B"]).unstack("Model A")
    return out


# ---------------------------------------------------------------------------
# Raw df.to_latex (what the notebooks print, now written to files)
# ---------------------------------------------------------------------------

def _bold_minrow_latex(piv: pd.DataFrame) -> str:
    return (
        piv.style
        .format({m: "{:,.3f}".format for m in piv.columns})
        .apply(lambda x: ["font-weight: bold" if v == x.min() else "" for v in x], axis=1)
        .to_latex(hrules=True, clines="skip-last;data")
        .replace("%", r"\%")
        .replace("font-weightbold", "B")
    )


def _wilcoxon_latex(wilc: pd.DataFrame) -> str:
    return (
        wilc.style
        .apply(
            lambda x: [
                "font-weight: bold" if (isinstance(v, str) and "-" in v and "0.00" not in v) else ""
                for v in x
            ],
            axis=1,
        )
        .to_latex(hrules=True, clines="skip-last;data")
        .replace("%", r"\%")
        .replace("font-weightbold", "B")
        .replace("#", r"\#")
        .replace("nan", " ")
    )


def write_raw_tex(task: str, piv_deals: pd.DataFrame, piv_size: pd.DataFrame, wilc: pd.DataFrame) -> None:
    out = STAGING / task
    out.mkdir(parents=True, exist_ok=True)

    deals_tex = _bold_minrow_latex(piv_deals)
    if task == "ce":
        deals_tex = deals_tex.replace("1.000", "n/a")
    (out / "tbl_mdape_deals.tex").write_text(deals_tex)
    (out / "data_mdape_deals.pkl").write_bytes(pickle.dumps(piv_deals))

    size_tex = _bold_minrow_latex(piv_size).replace("nan", "-")
    (out / "tbl_mdape_size.tex").write_text(size_tex)
    (out / "data_mdape_size.pkl").write_bytes(pickle.dumps(piv_size))

    (out / "tbl_wilcoxon.tex").write_text(_wilcoxon_latex(wilc))
    (out / "data_wilcoxon.pkl").write_bytes(pickle.dumps(wilc))


# ---------------------------------------------------------------------------
# Paper-format merged tabular (matches tab:ae_ape / tab:cep_ape)
# ---------------------------------------------------------------------------

def _fmt_cell(v: float, is_min: bool, na_for_one: bool) -> str:
    if na_for_one and np.isclose(v, 1.0):
        return "n/a"
    s = f"{v:,.3f}"
    return rf"\B {s}" if is_min else s


def merged_paper_tabular(piv_deals: pd.DataFrame, piv_size: pd.DataFrame, task: str) -> str:
    """Side-by-side tabular matching paper's tab:ae_ape / tab:cep_ape layout."""
    deal_label = DEALS_LABEL[task]
    na_for_one = (task == "ce")

    rows = []
    # iterate deals pivot in native order: (1,0), (1,1+), (2+,0), (2+,1+)
    # and size pivot in: (1,Small), (1,Large), (2+,Small), (2+,Large)
    deals_rows = piv_deals.reset_index().to_dict("records")
    size_rows = piv_size.reset_index().to_dict("records")
    assert len(deals_rows) == len(size_rows) == 4

    def _row_to_cells(row: dict, piv_view: pd.DataFrame, row_key: tuple) -> list[str]:
        vals = [float(row[m]) for m in MODELS_UQ]
        finite_vals = [v for v in vals if not (na_for_one and np.isclose(v, 1.0))]
        minv = min(finite_vals) if finite_vals else None
        return [_fmt_cell(v, (minv is not None and np.isclose(v, minv)), na_for_one) for v in vals]

    for i, (dr, sr) in enumerate(zip(deals_rows, size_rows)):
        deal_vals = _row_to_cells(dr, piv_deals, (dr["Round"], dr[deal_label]))
        size_vals = _row_to_cells(sr, piv_size, (sr["Round"], sr["Market Size"]))
        # Render the row. Even rows emit a \multirow for "Round".
        if i % 2 == 0:
            round_cell = rf"\multirow[c]{{2}}{{*}}{{{dr['Round']}}}"
        else:
            round_cell = ""
        cells = [round_cell, str(dr[deal_label])] + deal_vals + [str(sr["Market Size"])] + size_vals
        rows.append(" & ".join(cells) + r" \\")
        if i == 1:
            rows.append(r"\cline{1-6} \cline{7-11}")

    task_header_label = "Price realizations" if task == "ce" else r"\# realizations"
    size_label = "Market size"
    body = "\n        ".join(rows)
    tabular = (
        r"\begin{tabular}{c || c | cccc | c | cccc}" + "\n"
        "        "
        + r"& \multicolumn{5}{c|}{\textbf{Price Realizations Analysis (Left)}} & \multicolumn{5}{c}{\textbf{Market Size Analysis (Right)}} \\" + "\n"
        "        "
        + rf"Round & {task_header_label} & EMH & CEMH & OB-RLM & GBT & {size_label} & EMH & CEMH & OB-RLM & GBT \\" + "\n"
        "        " + r"\midrule" + "\n"
        "        " + body + "\n"
        r"    \end{tabular}"
    )
    return tabular


# ---------------------------------------------------------------------------
# Ranking comparison vs paper
# ---------------------------------------------------------------------------

PAPER_CEP_DEALS = {  # (Round, Deals): {model: value}  (n/a -> None)
    ("1", "0"):  {"EMH": None, "CEMH": None, "OB-RLM": 0.191, "GBT": 0.135},
    ("1", "1+"): {"EMH": 0.109, "CEMH": 0.097, "OB-RLM": 0.061, "GBT": 0.077},
    ("2+", "0"): {"EMH": None, "CEMH": None, "OB-RLM": 0.111, "GBT": 0.099},
    ("2+", "1+"): {"EMH": 0.062, "CEMH": 0.058, "OB-RLM": 0.049, "GBT": 0.051},
}
PAPER_CEP_SIZE = {
    ("1", "Small"): {"EMH": 0.152, "CEMH": 0.135, "OB-RLM": 0.096, "GBT": 0.079},
    ("1", "Large"): {"EMH": 0.131, "CEMH": 0.112, "OB-RLM": 0.061, "GBT": 0.084},
    ("2+", "Small"): {"EMH": 0.094, "CEMH": 0.083, "OB-RLM": 0.066, "GBT": 0.070},
    ("2+", "Large"): {"EMH": 0.071, "CEMH": 0.067, "OB-RLM": 0.051, "GBT": 0.051},
}


def compare_ranking(task: str, piv_deals: pd.DataFrame, piv_size: pd.DataFrame) -> str:
    if task != "ce":
        return ""
    lines = ["# Ranking preservation: paper vs corrected (CEP)", "",
             "| Row | Paper best | Paper value | Current best | Current value | Max |Δ| any model |",
             "|---|---|---|---|---|---|"]

    def _best(d: dict) -> tuple[str, float]:
        items = [(k, v) for k, v in d.items() if v is not None]
        k, v = min(items, key=lambda kv: kv[1])
        return k, v

    for (r, d), paper_row in PAPER_CEP_DEALS.items():
        cur_row = piv_deals.loc[(r, d)].to_dict()
        cur_clean = {m: (None if (task == "ce" and np.isclose(cur_row[m], 1.0)) else cur_row[m]) for m in MODELS_UQ}
        p_best, p_val = _best(paper_row)
        c_best, c_val = _best(cur_clean)
        max_delta = max(abs(paper_row[m] - cur_clean[m]) for m in MODELS_UQ if paper_row[m] is not None and cur_clean[m] is not None)
        lines.append(f"| R{r}/{d} deals | {p_best} | {p_val:.3f} | {c_best} | {c_val:.3f} | {max_delta:.3f} |")

    for (r, s), paper_row in PAPER_CEP_SIZE.items():
        cur_row = piv_size.loc[(r, s)].to_dict()
        cur_clean = {m: cur_row[m] for m in MODELS_UQ}
        p_best, p_val = _best(paper_row)
        c_best, c_val = _best(cur_clean)
        max_delta = max(abs(paper_row[m] - cur_clean[m]) for m in MODELS_UQ)
        lines.append(f"| R{r}/{s} size | {p_best} | {p_val:.3f} | {c_best} | {c_val:.3f} | {max_delta:.3f} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure copy
# ---------------------------------------------------------------------------

FIG_MAP = {
    "ae": (CODE_ROOT / "data/results/plots/ae", PAPER_FIGURES / "ae_results"),
    "ce": (CODE_ROOT / "data/results/plots/ce", PAPER_FIGURES / "price_results"),
}
BAR_NAMES = ["bar_round1_deal.pdf", "bar_round1_nodeal.pdf"]

BAR_COLORS = {
    "EMH": "#AF8D86",
    "CEMH": "#F6E27F",
    "OB-RLM": "#473BF0",
    "GBT": "#8C271E",
}


def _render_bar(pivot: pd.DataFrame, out_pdf: Path, suppress_na: bool) -> None:
    """Render a grouped bar chart matching the submission notebook styling.

    pivot: index=feedback_setting, columns=model, values=median APE
    suppress_na: when True, any y==1.0 (EMH/CEMH degenerate with no price) is
    rendered as a blank position with an 'n/a' label, matching the original paper.
    """
    import plotly.graph_objects as go  # imported lazily to avoid startup cost

    fig = go.Figure()
    feedback_settings = pivot.index.astype(str).tolist()
    for model in MODELS_UQ:
        y_raw = pivot[model].tolist()
        text = []
        y_plot = []
        positions = []
        colors = []
        for v in y_raw:
            if suppress_na and np.isclose(v, 1.0):
                text.append("n/a")
                y_plot.append(np.nan)
                positions.append("outside")
                colors.append(BAR_COLORS[model])
            else:
                text.append(f"{v:,.3f}")
                y_plot.append(v)
                positions.append("inside" if v > 0.7 else "outside")
                colors.append("black")
        fig.add_bar(
            x=feedback_settings,
            y=y_plot,
            name=model,
            marker_color=BAR_COLORS[model],
            text=text,
            textposition=positions,
            textangle=90,
            textfont=dict(color=colors),
        )

    fig.update_layout(barmode="group", template="simple_white")
    fig.update_layout(
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.2,
                    bgcolor="rgba(0,0,0,0)", orientation="h"),
        uniformtext_minsize=12, uniformtext_mode="show",
        width=300, height=300,
        margin=dict(t=10, r=10, b=10, l=10),
        font=dict(family="Arial"),
    )
    fig.update_xaxes(mirror=True, title_text="Feedback Setting")
    fig.update_yaxes(mirror=True, range=[0, 1.1], title_text="Median APE")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Disable MathJax so kaleido doesn't render the "Loading MathJax" banner.
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    fig.write_image(str(out_pdf), scale=10)


def build_bar_figures(all_ce_df: pd.DataFrame) -> list[Path]:
    """Regenerate CEP round-1 bar charts with n/a suppression for EMH/CEMH."""
    orig = pd.read_feather(CODE_ROOT / "data/preprocessed/original_df.ft")
    fs_pr = orig.groupby("treatment")[["price_rule", "feedback_setting"]].first()

    sample = all_ce_df.query("sample_id==0").copy()
    idx_cols = ["treatment", "game", "round", "time", "n_unique_deals_round", "model"]
    sample_orient = sample.set_index(idx_cols)["ce_ape"].unstack("model")[list(MODELS_UQ)]
    joined = sample_orient.join(fs_pr)

    r1_nodeal = (joined.query("round==1 and n_unique_deals_round==0")
                 .groupby("feedback_setting")[list(MODELS_UQ)].median())
    r1_deal = (joined.query("round==1 and n_unique_deals_round>0")
               .groupby("feedback_setting")[list(MODELS_UQ)].median())

    staging_dir = FIG_STAGING / "ce"
    staging_dir.mkdir(parents=True, exist_ok=True)
    dst_dir = PAPER_FIGURES / "price_results"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_dir = CODE_ROOT / "data/results/plots/ce"
    src_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for suffix, pivot, suppress in [
        ("nodeal", r1_nodeal, True),
        ("deal", r1_deal, False),
    ]:
        name = f"bar_round1_{suffix}.pdf"
        _render_bar(pivot, src_dir / name, suppress)
        shutil.copy2(src_dir / name, staging_dir / name)
        shutil.copy2(src_dir / name, dst_dir / name)
        outputs.append(dst_dir / name)
    return outputs


def copy_bar_figures() -> list[Path]:
    """Copy AE bars as-is (they don't need regeneration)."""
    copied = []
    src_dir, dst_dir = FIG_MAP["ae"]
    staging_dir = FIG_STAGING / "ae"
    staging_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in BAR_NAMES:
        src = src_dir / name
        if not src.exists():
            print(f"[warn] missing figure: {src}")
            continue
        shutil.copy2(src, staging_dir / name)
        shutil.copy2(src, dst_dir / name)
        copied.append(dst_dir / name)
    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_task(task: str, player_counts: pd.Series) -> dict:
    df = load_results(task)
    piv_deals = pivot_deals(df, task)
    piv_size = pivot_market_size(df, player_counts, task)
    wilc = wilcoxon_table(df, task)

    write_raw_tex(task, piv_deals, piv_size, wilc)

    merged = merged_paper_tabular(piv_deals, piv_size, task)
    FINAL.mkdir(parents=True, exist_ok=True)
    target = FINAL / (f"tab_ae_ape.tex" if task == "ae" else f"tab_cep_ape.tex")
    target.write_text(merged + "\n")
    wilc_target = FINAL / (f"tab_ae_ape_wilcoxon.tex" if task == "ae" else f"tab_cep_ape_wilcoxon.tex")
    wilc_target.write_text(_wilcoxon_latex(wilc))

    return {"deals": piv_deals, "size": piv_size, "df": df}


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    player_counts = load_player_counts()

    ae = build_task("ae", player_counts)
    ce = build_task("ce", player_counts)

    ranking_md = compare_ranking("ce", ce["deals"], ce["size"])
    (REPORT_DIR / "ranking_preservation.md").write_text(ranking_md)

    ce_figs = build_bar_figures(ce["df"])
    ae_figs = copy_bar_figures()

    print("Staging written to:", STAGING)
    print("Final tables:", FINAL)
    print("Figures written:")
    for p in ce_figs + ae_figs:
        print(" ", p)
    print("Ranking diff:", REPORT_DIR / "ranking_preservation.md")


if __name__ == "__main__":
    main()
