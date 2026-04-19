"""Diagnostic for the GBT (CEP) cold-start blowup on Lin.

Tests the hypothesis: the cold-start CEP error on Lin is driven by the
pre-deal bid-ask spread distribution lying outside the Ikica training
support. Produces:

  1. Spread / mid distribution comparison (Ikica vs Lin), per bucket
  2. APE-by-spread-band table for GBT (CEP) on Lin cold-start
  3. CDF plot: relative spread, Ikica vs Lin, cold-start only
  4. Binned plot: GBT (CEP) median APE vs relative spread, with
     prediction-vs-truth medians overlaid

Best bid is approximated by the 10th (highest) running buyer-side
quantile, best ask by the 0th (lowest) running seller-side quantile.
The 22 quantile features are all that the orderbook-only models see.

Outputs:
  code/data/results/lin_eval/diagnostic/spread_summary.json
  code/data/results/lin_eval/diagnostic/ape_by_spread.csv
  paper/epj/reproducibility_report/final_tables/tab_lin_cep_spread.tex
  paper/epj/figures/diagnostics/lin_cs_spread_cdf.pdf
  paper/epj/figures/diagnostics/lin_cs_ape_vs_spread.pdf

Run with: ``uv run python scripts/lin_cep_diagnostic.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PREP = CODE_ROOT / "data" / "preprocessed"
LIN_OUT = CODE_ROOT / "data" / "results" / "lin_eval"
DIAG_OUT = LIN_OUT / "diagnostic"
DIAG_OUT.mkdir(parents=True, exist_ok=True)

PAPER_ROOT = CODE_ROOT.parent / "paper" / "epj"
TAB_OUT = PAPER_ROOT / "reproducibility_report" / "final_tables" / "tab_lin_cep_spread.tex"
FIG_DIR = PAPER_ROOT / "figures" / "diagnostics"

DYNAMIC_TREATMENTS = ("FullSShift", "FullFromBtoS", "FullFromStoB")
SPREAD_BANDS = [
    ("crossed (sp/mid <= 0.07)", -np.inf, 0.07),
    ("Ikica IQR (0.07-0.58)", 0.07, 0.58),
    ("intermediate (0.58-1.0)", 0.58, 1.0),
    ("Lin-only regime (>1.0)", 1.0, np.inf),
]


def load_ikica() -> pd.DataFrame:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    return tad.query("round <= 5 and time <= 120").reset_index(drop=True)


def load_lin() -> pd.DataFrame:
    return pd.read_feather(LIN_OUT / "lin_features.ft")


def add_spread_cols(df: pd.DataFrame) -> pd.DataFrame:
    bid = df["running_buyer_bid_quant_10"].astype(float)
    ask = df["running_seller_bid_quant_00"].astype(float)
    df = df.copy()
    df["best_bid"] = bid
    df["best_ask"] = ask
    df["spread"] = ask - bid
    df["mid"] = (ask + bid) / 2
    df["spread_rel"] = df["spread"] / df["mid"]
    return df.replace([np.inf, -np.inf], np.nan)


def summarise_distribution(df: pd.DataFrame, label: str) -> dict:
    sp = df["spread"].dropna()
    sr = df["spread_rel"].dropna()
    return {
        "label": label,
        "n": int(len(df)),
        "spread_q25": float(sp.quantile(0.25)),
        "spread_median": float(sp.median()),
        "spread_q75": float(sp.quantile(0.75)),
        "spread_rel_q25": float(sr.quantile(0.25)),
        "spread_rel_median": float(sr.median()),
        "spread_rel_q75": float(sr.quantile(0.75)),
        "frac_crossed": float((df["spread"] <= 0).mean()),
    }


def summarise_buckets(df: pd.DataFrame, source: str) -> list[dict]:
    out = []
    df = df.copy()
    df["round_band"] = np.where(df["round"] == 1, "round 1", "round >= 2")
    df["deal_band"] = np.where(df["n_unique_deals_round"] == 0, "no deals", ">=1 deal")
    for (rb, db), grp in df.groupby(["round_band", "deal_band"]):
        s = summarise_distribution(grp, f"{source} | {rb} | {db}")
        s["source"] = source
        s["round_band"] = rb
        s["deal_band"] = db
        out.append(s)
    return out


def ape_by_spread_band(merged: pd.DataFrame) -> pd.DataFrame:
    cs = merged[merged["n_unique_deals_round"] == 0].copy()
    cs = cs.dropna(subset=["spread_rel", "ce_ape_gbt"])
    cs = cs[np.isfinite(cs["spread_rel"]) & np.isfinite(cs["ce_ape_gbt"])]
    rows = []
    for label, lo, hi in SPREAD_BANDS:
        sub = cs[(cs["spread_rel"] > lo) & (cs["spread_rel"] <= hi)]
        if len(sub) == 0:
            continue
        rows.append({
            "band": label,
            "n": len(sub),
            "share": len(sub) / len(cs),
            "spread_rel_median": float(sub["spread_rel"].median()),
            "ape_median": float(sub["ce_ape_gbt"].median()),
            "pred_median": float(sub["ce_pred_gbt"].median()),
            "truth_median": float(sub["ce_target"].median()),
        })
    rho = cs[["spread_rel", "ce_ape_gbt"]].corr(method="spearman").iloc[0, 1]
    return pd.DataFrame(rows), float(rho)


def write_latex_table(df_band: pd.DataFrame, rho: float) -> None:
    # Columns: band, n, share, spread_rel_median, ape_median, pred_median, truth_median
    lines = [
        r"\begin{tabular}{l rr rrrr}",
        r"\toprule",
        r"Spread band & $n$ & Share & Median $\mathit{sp/m}$ & Median APE & Med.\ pred. & Med.\ truth \\",
        r"\midrule",
    ]
    band_short = {
        "crossed (sp/mid <= 0.07)": r"crossed ($\mathit{sp/m} \leq 0.07$)",
        "Ikica IQR (0.07-0.58)": r"Ikica IQR ($0.07$--$0.58$)",
        "intermediate (0.58-1.0)": r"intermediate ($0.58$--$1.0$)",
        "Lin-only regime (>1.0)": r"Lin-only ($\mathit{sp/m} > 1.0$)",
    }
    for _, r in df_band.iterrows():
        lines.append(
            f"{band_short.get(r['band'], r['band'])} & {int(r['n']):,} & {r['share']:.2f} & "
            f"{r['spread_rel_median']:.2f} & {r['ape_median']:.3f} & "
            f"{r['pred_median']:.1f} & {r['truth_median']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    TAB_OUT.parent.mkdir(parents=True, exist_ok=True)
    TAB_OUT.write_text("\n".join(lines) + "\n")
    print(f"  wrote {TAB_OUT}  (Spearman rho = {rho:.3f})")


def plot_combined(ikica_cs: pd.DataFrame, lin_cs: pd.DataFrame, merged: pd.DataFrame) -> None:
    """One composite figure, three panels: (a) spread CDF, (b) APE vs spread,
    (c) prediction vs truth medians vs spread."""
    cs = merged[merged["n_unique_deals_round"] == 0].copy()
    cs = cs.dropna(subset=["spread_rel", "ce_ape_gbt"])
    cs = cs[np.isfinite(cs["spread_rel"]) & np.isfinite(cs["ce_ape_gbt"])]
    cs = cs[(cs["spread_rel"] >= -1) & (cs["spread_rel"] <= 4)]
    cs["bin"] = pd.qcut(cs["spread_rel"], 30, duplicates="drop")
    g = cs.groupby("bin", observed=True).agg(
        spread_rel=("spread_rel", "median"),
        ape=("ce_ape_gbt", "median"),
        pred=("ce_pred_gbt", "median"),
        truth=("ce_target", "median"),
    ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    ax0 = axes[0]
    for df, label, ls in [
        (ikica_cs, "Ikica", "-"),
        (lin_cs, "Lin", "--"),
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
    ax1.plot(g["spread_rel"], g["ape"], "-o", ms=3, lw=1.4, color="C3")
    ax1.set_xlabel("Relative spread (bin median)")
    ax1.set_ylabel("Median APE")
    ax1.axvline(0.58, color="grey", lw=0.6, ls=":")
    ax1.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax1.set_title("(b) GBT (CEP) APE on Lin cold-start")

    ax2 = axes[2]
    ax2.plot(g["spread_rel"], g["pred"], "-o", ms=3, lw=1.4, label="median prediction", color="C0")
    ax2.plot(g["spread_rel"], g["truth"], "-s", ms=3, lw=1.4, label="median truth", color="C2")
    ax2.set_xlabel("Relative spread (bin median)")
    ax2.set_ylabel("Price")
    ax2.axvline(0.58, color="grey", lw=0.6, ls=":")
    ax2.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_title("(c) Prediction vs.\\ truth")

    fig.tight_layout()
    out = FIG_DIR / "lin_cs_spread_diagnostic.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print("Step 1/4: load Ikica + Lin features")
    ikica = add_spread_cols(load_ikica())
    lin = add_spread_cols(load_lin())

    print("Step 2/4: per-bucket spread distributions")
    summary = summarise_buckets(ikica, "ikica") + summarise_buckets(lin, "lin")
    (DIAG_OUT / "spread_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {DIAG_OUT / 'spread_summary.json'}")
    for s in summary:
        print(
            f"    {s['source']:5s} {s['round_band']:>11s} {s['deal_band']:>9s}  "
            f"n={s['n']:>6d}  sp_med={s['spread_median']:.2f}  "
            f"sp/m_med={s['spread_rel_median']:.3f}  "
            f"sp/m_IQR=[{s['spread_rel_q25']:.3f},{s['spread_rel_q75']:.3f}]  "
            f"crossed={s['frac_crossed']:.3f}"
        )

    print("\nStep 3/4: GBT (CEP) APE by spread band on Lin cold-start")
    gbt = pd.read_feather(LIN_OUT / "gbt_cep.ft")
    merged = gbt.merge(
        lin[["treatment", "game", "round", "time",
             "best_bid", "best_ask", "spread", "mid", "spread_rel"]],
        on=["treatment", "game", "round", "time"],
        how="left",
    )
    df_band, rho = ape_by_spread_band(merged)
    df_band.to_csv(DIAG_OUT / "ape_by_spread.csv", index=False)
    print(f"  Spearman rho(spread_rel, APE) = {rho:.3f}")
    print(df_band.to_string(index=False))
    write_latex_table(df_band, rho)

    print("\nStep 4/4: composite figure")
    ikica_cs = ikica[ikica["n_unique_deals_round"] == 0]
    lin_cs = lin[lin["n_unique_deals_round"] == 0]
    plot_combined(ikica_cs, lin_cs, merged)
    print("done.")


if __name__ == "__main__":
    main()
