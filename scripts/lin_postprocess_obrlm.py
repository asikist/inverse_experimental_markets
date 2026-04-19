"""Post-process the R2.4 Lin tables + figure to include OB-RLM (CEP).

Run this *after* ``lin_eval.py``, ``lin_cep_diagnostic.py`` and
``lin_obrlm_cep_check.py``. It consumes the persisted .ft files and rewrites
the three final artefacts referenced by ``response.tex``:

  paper/epj/tables/tab_lin_external.tex
  paper/epj/tables/tab_lin_cep_spread.tex
  paper/epj/figures/diagnostics/lin_cs_spread_diagnostic.pdf

so the OB-RLM (CEP) row / column / panel appears alongside GBT (CEP).

Existing GBT numbers are kept bit-identical by reading from the same .ft
files the GBT-only tables were built from. No existing script is modified.

Run: ``uv run python scripts/lin_postprocess_obrlm.py``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
RES = CODE_ROOT / "data" / "results"
LIN = RES / "lin_eval"
OBRLM = RES / "lin_obrlm_cep_check"
MAIN_CEP = RES / "ce_price"

PAPER_ROOT = CODE_ROOT.parent / "paper" / "epj"
TAB_DIR = PAPER_ROOT / "tables"
FIG_DIR = PAPER_ROOT / "figures" / "diagnostics"

SPREAD_BANDS = [
    ("crossed (sp/mid <= 0.07)", -np.inf, 0.07, r"crossed ($\mathit{sp/m} \leq 0.07$)"),
    ("Ikica IQR (0.07-0.58)", 0.07, 0.58, r"Ikica IQR ($0.07$--$0.58$)"),
    ("intermediate (0.58-1.0)", 0.58, 1.0, r"intermediate ($0.58$--$1.0$)"),
    ("Lin-only regime (>1.0)", 1.0, np.inf, r"Lin-only ($\mathit{sp/m} > 1.0$)"),
]


def bucket_median(df: pd.DataFrame, col: str) -> dict:
    """Median APE in the four (round, n_deals) buckets."""
    out = {}
    for key, q in [
        ("r1_0", "round == 1 and n_unique_deals_round == 0"),
        ("r1_d", "round == 1 and n_unique_deals_round >= 1"),
        ("r2_0", "round > 1 and n_unique_deals_round == 0"),
        ("r2_d", "round > 1 and n_unique_deals_round >= 1"),
    ]:
        out[key] = float(df.query(q)[col].median())
    return out


# ---------------------------------------------------------------------------
# tab_lin_external.tex
# ---------------------------------------------------------------------------

def _fmt_dash_or_number(v: float, prec: int = 3) -> str:
    if not np.isfinite(v):
        return "---"
    return f"{v:.{prec}f}"


def regenerate_lin_external() -> None:
    # GBT / OB-RLM / CEMH references in the "orderbook-only (Ikica)" row
    # are pre-existing numbers from the R2.5 orderbook-only ablation
    # and the main-paper OB-RLM (CEP) file.  We re-quote them verbatim so
    # this postprocess is idempotent with the submitted table.
    ikica_ref = {
        ("OB-RLM (AE)",     "orderbook-only (Ikica)"):     (0.281, 0.164, 0.154, 0.136),
        ("GBT (AE)",        "orderbook-only (Ikica)"):     (0.316, 0.172, 0.098, 0.100),
        ("CEMH-global (CEP)", "orderbook-only (Ikica)"):   (1.000, 0.092, 1.000, 0.054),
        ("GBT (CEP)",       "orderbook-only (Ikica)"):     (0.135, 0.077, 0.099, 0.051),
    }

    # OB-RLM (CEP) reference comes from the main-paper 50-CV file: it is
    # already orderbook-only and was not refit in R2.5.
    ob_rlm_cep_ref = pd.read_feather(MAIN_CEP / "ob_rlm.ft")
    ob_rlm_cep_ref_buckets = bucket_median(ob_rlm_cep_ref, "ce_ape")

    # Lin external numbers -- read from the .ft files.
    lin_numbers = {
        "OB-RLM (AE)":        bucket_median(
            pd.read_feather(LIN / "ob_rlm_ae.ft"), "ae_ape_ob_rlm"),
        "GBT (AE)":           bucket_median(
            pd.read_feather(LIN / "gbt_ae.ft"), "ae_ape_gbt"),
        "CEMH-global (CEP)":  bucket_median(
            pd.read_feather(LIN / "cemh_global_cep.ft"), "ce_ape_cemh"),
        "OB-RLM (CEP)":       bucket_median(
            pd.read_feather(OBRLM / "ob_rlm_cep_lin.ft"), "ce_ape_ob_rlm"),
        "GBT (CEP)":          bucket_median(
            pd.read_feather(LIN / "gbt_cep.ft"), "ce_ape_gbt"),
    }

    order = [
        ("OB-RLM (AE)",       ikica_ref[("OB-RLM (AE)", "orderbook-only (Ikica)")]),
        ("GBT (AE)",          ikica_ref[("GBT (AE)", "orderbook-only (Ikica)")]),
        ("CEMH-global (CEP)", ikica_ref[("CEMH-global (CEP)", "orderbook-only (Ikica)")]),
        ("OB-RLM (CEP)",      (ob_rlm_cep_ref_buckets["r1_0"],
                               ob_rlm_cep_ref_buckets["r1_d"],
                               ob_rlm_cep_ref_buckets["r2_0"],
                               ob_rlm_cep_ref_buckets["r2_d"])),
        ("GBT (CEP)",         ikica_ref[("GBT (CEP)", "orderbook-only (Ikica)")]),
    ]

    lines = [
        r"\begin{tabular}{l l cccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Round 1} & \multicolumn{2}{c}{Round $\geq 2$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Model & Variant & 0 deals & $\geq 1$ deal & 0 deals & $\geq 1$ deal \\",
        r"\midrule",
    ]

    last = len(order) - 1
    for i, (model, (ref_r1_0, ref_r1_d, ref_r2_0, ref_r2_d)) in enumerate(order):
        lin = lin_numbers[model]
        # CEMH no-deal buckets are not defined (no realized price) --> dashes
        def lin_val(val, bucket):
            if model == "CEMH-global (CEP)" and bucket in ("r1_0", "r2_0"):
                return "---"
            return _fmt_dash_or_number(val)

        lines.append(
            rf"\multirow{{2}}{{*}}{{{model}}} & orderbook-only (Ikica) & "
            rf"{_fmt_dash_or_number(ref_r1_0)} & {_fmt_dash_or_number(ref_r1_d)} & "
            rf"{_fmt_dash_or_number(ref_r2_0)} & {_fmt_dash_or_number(ref_r2_d)} \\"
        )
        lines.append(
            rf"& Lin external (preliminary) & "
            rf"{lin_val(lin['r1_0'], 'r1_0')} & {lin_val(lin['r1_d'], 'r1_d')} & "
            rf"{lin_val(lin['r2_0'], 'r2_0')} & {lin_val(lin['r2_d'], 'r2_d')} \\"
        )
        if i < last:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "tab_lin_external.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# tab_lin_cep_spread.tex
# ---------------------------------------------------------------------------

def _add_spread_cols(df: pd.DataFrame) -> pd.DataFrame:
    bid = df["running_buyer_bid_quant_10"].astype(float)
    ask = df["running_seller_bid_quant_00"].astype(float)
    df = df.copy()
    df["spread"] = ask - bid
    df["mid"] = (ask + bid) / 2
    df["spread_rel"] = df["spread"] / df["mid"]
    return df.replace([np.inf, -np.inf], np.nan)


def _spread_band_stats(merged: pd.DataFrame, ape_col: str, pred_col: str) -> pd.DataFrame:
    cs = merged[merged["n_unique_deals_round"] == 0].copy()
    cs = cs.dropna(subset=["spread_rel", ape_col])
    cs = cs[np.isfinite(cs["spread_rel"]) & np.isfinite(cs[ape_col])]
    rows = []
    for label, lo, hi, _pretty in SPREAD_BANDS:
        sub = cs[(cs["spread_rel"] > lo) & (cs["spread_rel"] <= hi)]
        if len(sub) == 0:
            continue
        rows.append({
            "band": label,
            "n": int(len(sub)),
            "share": float(len(sub) / len(cs)),
            "spread_rel_median": float(sub["spread_rel"].median()),
            "ape_median": float(sub[ape_col].median()),
            "pred_median": float(sub[pred_col].median()),
            "truth_median": float(sub["ce_target"].median()),
        })
    rho = cs[["spread_rel", ape_col]].corr(method="spearman").iloc[0, 1]
    return pd.DataFrame(rows), float(rho)


def regenerate_spread_table() -> None:
    lin = pd.read_feather(LIN / "lin_features.ft")
    lin = _add_spread_cols(lin)
    sp = lin[["treatment", "game", "round", "time",
              "spread", "mid", "spread_rel"]]

    gbt = pd.read_feather(LIN / "gbt_cep.ft").merge(
        sp, on=["treatment", "game", "round", "time"], how="left")
    obr = pd.read_feather(OBRLM / "ob_rlm_cep_lin.ft").merge(
        sp, on=["treatment", "game", "round", "time"], how="left")

    gbt_tab, rho_gbt = _spread_band_stats(gbt, "ce_ape_gbt", "ce_pred_gbt")
    obr_tab, rho_obr = _spread_band_stats(obr, "ce_ape_ob_rlm", "ce_pred_ob_rlm")

    merged = gbt_tab.merge(
        obr_tab[["band", "ape_median", "pred_median"]],
        on="band", suffixes=("_gbt", "_obr"),
    )

    lines = [
        r"\begin{tabular}{l rr r rr rr r}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{Median APE} & \multicolumn{2}{c}{Median pred.} & \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Spread band & $n$ & Share & Med.\ $\mathit{sp/m}$ & GBT & OB-RLM & GBT & OB-RLM & Med.\ truth \\",
        r"\midrule",
    ]
    pretty = {label: pr for label, _lo, _hi, pr in SPREAD_BANDS}
    for _, r in merged.iterrows():
        lines.append(
            f"{pretty[r['band']]} & {int(r['n']):,} & {r['share']:.2f} & "
            f"{r['spread_rel_median']:.2f} & "
            f"{r['ape_median_gbt']:.3f} & {r['ape_median_obr']:.3f} & "
            f"{r['pred_median_gbt']:.1f} & {r['pred_median_obr']:.1f} & "
            f"{r['truth_median']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "tab_lin_cep_spread.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")
    print(f"  Spearman rho GBT = {rho_gbt:.3f}   OB-RLM = {rho_obr:.3f}")


# ---------------------------------------------------------------------------
# lin_cs_spread_diagnostic.pdf  (three-panel composite, OB-RLM overlaid)
# ---------------------------------------------------------------------------

def regenerate_figure() -> None:
    # Ikica cold-start spread for panel (a)
    tad = pd.read_feather(CODE_ROOT / "data" / "preprocessed" / "time_aggregate_dataset.ft")
    tad = tad[~tad["treatment"].isin(("FullSShift", "FullFromBtoS", "FullFromStoB"))]
    tad = tad.query("round <= 5 and time <= 120").reset_index(drop=True)
    tad = _add_spread_cols(tad)

    lin = pd.read_feather(LIN / "lin_features.ft")
    lin = _add_spread_cols(lin)
    sp = lin[["treatment", "game", "round", "time", "spread_rel"]]

    gbt = pd.read_feather(LIN / "gbt_cep.ft").merge(
        sp, on=["treatment", "game", "round", "time"], how="left")
    obr = pd.read_feather(OBRLM / "ob_rlm_cep_lin.ft").merge(
        sp, on=["treatment", "game", "round", "time"], how="left")

    def binned(df: pd.DataFrame, ape_col: str, pred_col: str) -> pd.DataFrame:
        cs = df[df["n_unique_deals_round"] == 0].copy()
        cs = cs.dropna(subset=["spread_rel", ape_col])
        cs = cs[np.isfinite(cs["spread_rel"]) & np.isfinite(cs[ape_col])]
        cs = cs[(cs["spread_rel"] >= -1) & (cs["spread_rel"] <= 4)]
        cs["bin"] = pd.qcut(cs["spread_rel"], 30, duplicates="drop")
        return (
            cs.groupby("bin", observed=True)
            .agg(
                spread_rel=("spread_rel", "median"),
                ape=(ape_col, "median"),
                pred=(pred_col, "median"),
                truth=("ce_target", "median"),
            )
            .reset_index(drop=True)
        )

    g_gbt = binned(gbt, "ce_ape_gbt", "ce_pred_gbt")
    g_obr = binned(obr, "ce_ape_ob_rlm", "ce_pred_ob_rlm")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    ax0 = axes[0]
    for df, label, ls in [
        (tad[tad["n_unique_deals_round"] == 0], "Ikica", "-"),
        (lin[lin["n_unique_deals_round"] == 0], "Lin", "--"),
    ]:
        s = df["spread_rel"].dropna()
        s = s[np.isfinite(s)]
        s = s[(s >= -2) & (s <= 4)]
        x = np.sort(s.values)
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax0.plot(x, y, ls, label=label, lw=1.6)
    ax0.axvline(0.0, color="grey", lw=0.6, ls=":")
    ax0.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax0.set_xlabel(r"Relative spread $(\mathrm{ask}-\mathrm{bid})/\mathrm{mid}$")
    ax0.set_ylabel("Cumulative share")
    ax0.set_xlim(-1, 4)
    ax0.legend(loc="lower right", frameon=False)
    ax0.set_title("(a) Spread CDF, cold-start")

    ax1 = axes[1]
    ax1.plot(g_gbt["spread_rel"], g_gbt["ape"], "-o", ms=3, lw=1.4,
             color="C3", label="GBT (CEP)")
    ax1.plot(g_obr["spread_rel"], g_obr["ape"], "-s", ms=3, lw=1.4,
             color="C1", label="OB-RLM (CEP)")
    ax1.axvline(0.58, color="grey", lw=0.6, ls=":")
    ax1.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax1.set_xlabel("Relative spread (bin median)")
    ax1.set_ylabel("Median APE")
    ax1.legend(loc="best", frameon=False, fontsize=9)
    ax1.set_title("(b) APE on Lin cold-start")

    ax2 = axes[2]
    ax2.plot(g_gbt["spread_rel"], g_gbt["pred"], "-o", ms=3, lw=1.2,
             color="C3", label="GBT pred.")
    ax2.plot(g_obr["spread_rel"], g_obr["pred"], "-s", ms=3, lw=1.2,
             color="C1", label="OB-RLM pred.")
    ax2.plot(g_gbt["spread_rel"], g_gbt["truth"], "--", lw=1.2,
             color="C2", label="truth")
    ax2.axvline(0.58, color="grey", lw=0.6, ls=":")
    ax2.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax2.set_xlabel("Relative spread (bin median)")
    ax2.set_ylabel("Price")
    ax2.legend(loc="best", frameon=False, fontsize=9)
    ax2.set_title("(c) Prediction vs.\\ truth")

    fig.tight_layout()
    out = FIG_DIR / "lin_cs_spread_diagnostic.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print("Step 1/3: regenerate tab_lin_external.tex (adds OB-RLM (CEP) row)")
    regenerate_lin_external()
    print("\nStep 2/3: regenerate tab_lin_cep_spread.tex (adds OB-RLM columns)")
    regenerate_spread_table()
    print("\nStep 3/3: regenerate lin_cs_spread_diagnostic.pdf (overlay)")
    regenerate_figure()
    print("\ndone.")


if __name__ == "__main__":
    main()
