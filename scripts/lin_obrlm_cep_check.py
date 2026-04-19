"""Supplemental OB-RLM (CEP) Lin transfer check for R2.4.

Adds the orderbook-only linear CEP model as a second witness for the
wide-spread extrapolation hypothesis currently tested on GBT (CEP) only.
No existing files are modified; all outputs are written to a separate
directory so the main pipeline remains untouched.

Procedure
---------
1. Refit OB-RLM (CEP) on the full non-dynamic Ikica dataset, using only the
   running bid/ask quantile features (same feature set as `lin_eval.py` uses
   for the other four orderbook-only models).
2. Apply the refit model to Lin (cached features from `lin_eval`).
3. Compute bucket-level median APE (same four buckets as the main paper).
4. Re-use the R2.4 spread-band decomposition on OB-RLM predictions so the
   wide-spread claim can be cross-checked against a second model.
5. For reference, also predict in-sample on Ikica so the reader can see the
   "Ikica full-data" baseline for the same refit.

Inputs (read-only)
------------------
  code/data/preprocessed/time_aggregate_dataset.ft
  code/data/results/lin_eval/lin_features.ft       (produced by lin_eval.py)
  code/data/results/lin_eval/gbt_cep.ft            (for the overlay figure)

Outputs
-------
  code/data/results/lin_obrlm_cep_check/
      ob_rlm_cep_lin.ft              per-row Lin predictions + APE
      ob_rlm_cep_ikica.ft            per-row Ikica in-sample predictions + APE
      summary.json                   bucket medians (Lin + Ikica reference)
      ape_by_spread_ob_rlm.csv       cold-start spread-band table (Lin)
      tab_lin_external_obrlm_cep.tex supplemental row for tab_lin_external
      tab_lin_cep_spread_obrlm.tex   supplemental spread-band table
      lin_cs_spread_obrlm_vs_gbt.pdf overlay: OB-RLM vs GBT across spread

Run: ``uv run python scripts/lin_obrlm_cep_check.py``
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_helpers.preprocessors.scalers import (
    denormalize_feats,
    normalize_feats,
    normalize_other,
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PREP = CODE_ROOT / "data" / "preprocessed"
LIN_IN = CODE_ROOT / "data" / "results" / "lin_eval"
OUT = CODE_ROOT / "data" / "results" / "lin_obrlm_cep_check"
OUT.mkdir(parents=True, exist_ok=True)

DYNAMIC_TREATMENTS = ("FullSShift", "FullFromBtoS", "FullFromStoB")
MIN_Q, MAX_Q = 0.35, 0.65

SPREAD_BANDS = [
    ("crossed (sp/mid <= 0.07)", -np.inf, 0.07),
    ("Ikica IQR (0.07-0.58)", 0.07, 0.58),
    ("intermediate (0.58-1.0)", 0.58, 1.0),
    ("Lin-only regime (>1.0)", 1.0, np.inf),
]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "running_" in c and "change" not in c]


def load_ikica() -> pd.DataFrame:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    return tad.query("round <= 5 and time <= 120").reset_index(drop=True)


def load_lin() -> pd.DataFrame:
    return pd.read_feather(LIN_IN / "lin_features.ft")


def fit_ob_rlm_cep(train: pd.DataFrame, feats: list[str]):
    """Parallel to diagnostics.fit_ob_rlm_cep; training on full Ikica."""
    tr = train.copy()
    tr[feats], X_med, X_iqr = normalize_feats(tr[feats].values, MIN_Q, MAX_Q)
    y = tr["ce_round"].values[:, np.newaxis]
    tr["ce_round_norm"] = normalize_other(y, X_med, X_iqr).squeeze()
    formula = "ce_round_norm ~ " + " + ".join(feats) + " - 1"
    return smf.rlm(formula, data=tr).fit()


def predict_ob_rlm_cep(model, test: pd.DataFrame, feats: list[str]) -> np.ndarray:
    te = test.copy()
    te[feats], med, iqr = normalize_feats(te[feats].values, MIN_Q, MAX_Q)
    raw = model.predict(te).to_numpy()
    return denormalize_feats(raw[:, np.newaxis], med, iqr).squeeze()


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


def bucket_summary(df: pd.DataFrame, err_col: str) -> dict:
    return {
        "rows": int(df[err_col].notna().sum()),
        "median_ape_overall": float(df[err_col].median()),
        "median_ape_round_1_no_deals": float(
            df.query("round == 1 and n_unique_deals_round == 0")[err_col].median()),
        "median_ape_round_1_with_deals": float(
            df.query("round == 1 and n_unique_deals_round >= 1")[err_col].median()),
        "median_ape_round_ge2_no_deals": float(
            df.query("round > 1 and n_unique_deals_round == 0")[err_col].median()),
        "median_ape_round_ge2_with_deals": float(
            df.query("round > 1 and n_unique_deals_round >= 1")[err_col].median()),
    }


def ape_by_spread_band(merged: pd.DataFrame, err_col: str, pred_col: str):
    cs = merged[merged["n_unique_deals_round"] == 0].copy()
    cs = cs.dropna(subset=["spread_rel", err_col])
    cs = cs[np.isfinite(cs["spread_rel"]) & np.isfinite(cs[err_col])]
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
            "ape_median": float(sub[err_col].median()),
            "pred_median": float(sub[pred_col].median()),
            "truth_median": float(sub["ce_target"].median()),
        })
    rho = cs[["spread_rel", err_col]].corr(method="spearman").iloc[0, 1]
    return pd.DataFrame(rows), float(rho)


def write_latex_spread_table(df_band: pd.DataFrame, rho: float) -> None:
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
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = OUT / "tab_lin_cep_spread_obrlm.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}  (Spearman rho = {rho:.3f})")


def write_latex_external_row(summ_lin: dict, summ_ikica: dict) -> None:
    lines = [
        r"% Supplemental row for tab_lin_external; splice in if adopted.",
        r"% Ikica reference here is OB-RLM (CEP) refit on the *full* Ikica dataset,",
        r"% predicted in-sample -- matches the refit protocol used for GBT in lin_eval.",
        r"\multirow{2}{*}{OB-RLM (CEP)} "
        rf"& full-Ikica in-sample        & "
        rf"{summ_ikica['median_ape_round_1_no_deals']:.3f} & "
        rf"{summ_ikica['median_ape_round_1_with_deals']:.3f} & "
        rf"{summ_ikica['median_ape_round_ge2_no_deals']:.3f} & "
        rf"{summ_ikica['median_ape_round_ge2_with_deals']:.3f} \\",
        rf"& Lin external (preliminary) & "
        rf"{summ_lin['median_ape_round_1_no_deals']:.3f} & "
        rf"{summ_lin['median_ape_round_1_with_deals']:.3f} & "
        rf"{summ_lin['median_ape_round_ge2_no_deals']:.3f} & "
        rf"{summ_lin['median_ape_round_ge2_with_deals']:.3f} \\",
    ]
    out = OUT / "tab_lin_external_obrlm_cep.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def overlay_figure(ob_rlm_merged: pd.DataFrame, gbt_merged: pd.DataFrame) -> None:
    """Three panels: (a) CDF of relative spread (kept as context);
    (b) median APE vs spread, OB-RLM vs GBT;
    (c) median prediction vs truth per spread bin, OB-RLM vs GBT."""

    def binned(df: pd.DataFrame, err_col: str, pred_col: str) -> pd.DataFrame:
        cs = df[df["n_unique_deals_round"] == 0].copy()
        cs = cs.dropna(subset=["spread_rel", err_col])
        cs = cs[np.isfinite(cs["spread_rel"]) & np.isfinite(cs[err_col])]
        cs = cs[(cs["spread_rel"] >= -1) & (cs["spread_rel"] <= 4)]
        cs["bin"] = pd.qcut(cs["spread_rel"], 30, duplicates="drop")
        return (
            cs.groupby("bin", observed=True)
            .agg(
                spread_rel=("spread_rel", "median"),
                ape=(err_col, "median"),
                pred=(pred_col, "median"),
                truth=("ce_target", "median"),
            )
            .reset_index(drop=True)
        )

    g_ob = binned(ob_rlm_merged, "ce_ape_ob_rlm", "ce_pred_ob_rlm")
    g_gb = binned(gbt_merged, "ce_ape_gbt", "ce_pred_gbt")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    ax0 = axes[0]
    s = ob_rlm_merged[ob_rlm_merged["n_unique_deals_round"] == 0]["spread_rel"].dropna()
    s = s[np.isfinite(s)]
    s = s[(s >= -2) & (s <= 4)]
    x = np.sort(s.values)
    y = np.linspace(0, 1, len(x), endpoint=True)
    ax0.plot(x, y, "--", lw=1.6, label="Lin cold-start")
    ax0.axvline(0.0, color="grey", lw=0.6, ls=":")
    ax0.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax0.set_xlabel(r"Relative spread $(\mathrm{ask}-\mathrm{bid})/\mathrm{mid}$")
    ax0.set_ylabel("Cumulative share")
    ax0.set_xlim(-1, 4)
    ax0.legend(loc="lower right", frameon=False)
    ax0.set_title("(a) Spread CDF, Lin cold-start")

    ax1 = axes[1]
    ax1.plot(g_gb["spread_rel"], g_gb["ape"], "-o", ms=3, lw=1.4,
             color="C3", label="GBT (CEP)")
    ax1.plot(g_ob["spread_rel"], g_ob["ape"], "-s", ms=3, lw=1.4,
             color="C1", label="OB-RLM (CEP)")
    ax1.set_xlabel("Relative spread (bin median)")
    ax1.set_ylabel("Median APE")
    ax1.axvline(0.58, color="grey", lw=0.6, ls=":")
    ax1.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax1.legend(loc="best", frameon=False, fontsize=9)
    ax1.set_title("(b) Median APE vs spread, cold-start")

    ax2 = axes[2]
    ax2.plot(g_gb["spread_rel"], g_gb["pred"], "-o", ms=3, lw=1.2,
             color="C3", label="GBT pred")
    ax2.plot(g_ob["spread_rel"], g_ob["pred"], "-s", ms=3, lw=1.2,
             color="C1", label="OB-RLM pred")
    ax2.plot(g_gb["spread_rel"], g_gb["truth"], "--", lw=1.2,
             color="C2", label="truth")
    ax2.set_xlabel("Relative spread (bin median)")
    ax2.set_ylabel("Price")
    ax2.axvline(0.58, color="grey", lw=0.6, ls=":")
    ax2.axvline(1.0, color="grey", lw=0.6, ls=":")
    ax2.legend(loc="best", frameon=False, fontsize=9)
    ax2.set_title("(c) Prediction vs.\\ truth per bin")

    fig.tight_layout()
    out = OUT / "lin_cs_spread_obrlm_vs_gbt.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    np.random.seed(1)

    print("=" * 60)
    print("Step 1/5: load Ikica and Lin features")
    t0 = time.perf_counter()
    tad = load_ikica()
    lin = load_lin()
    feats = feature_cols(tad)
    print(f"  Ikica rows: {len(tad):,}  features: {len(feats)}  "
          f"Lin rows: {len(lin):,}  ({time.perf_counter()-t0:.1f}s)")

    print("\nStep 2/5: fit OB-RLM (CEP) on full Ikica")
    t0 = time.perf_counter()
    rlm = fit_ob_rlm_cep(tad, feats)
    print(f"  fit done ({time.perf_counter()-t0:.1f}s)")

    print("\nStep 3/5: predict on Lin + Ikica in-sample")
    # Lin predictions
    pred_lin = predict_ob_rlm_cep(rlm, lin, feats)
    tgt_lin = lin["ce_round"].to_numpy()
    ape_lin = np.where(tgt_lin > 0, np.abs(pred_lin - tgt_lin) / tgt_lin, np.nan)
    keys = ["treatment", "game", "round", "time", "n_unique_deals_round"]
    df_lin = pd.DataFrame({
        **{k: lin[k].to_numpy() for k in keys},
        "ce_pred_ob_rlm": pred_lin,
        "ce_target": tgt_lin,
        "ce_ape_ob_rlm": ape_lin,
    })
    df_lin.to_feather(OUT / "ob_rlm_cep_lin.ft")

    # Ikica in-sample reference
    pred_ik = predict_ob_rlm_cep(rlm, tad, feats)
    tgt_ik = tad["ce_round"].to_numpy()
    ape_ik = np.where(tgt_ik > 0, np.abs(pred_ik - tgt_ik) / tgt_ik, np.nan)
    df_ik = pd.DataFrame({
        **{k: tad[k].to_numpy() for k in keys},
        "ce_pred_ob_rlm": pred_ik,
        "ce_target": tgt_ik,
        "ce_ape_ob_rlm": ape_ik,
    })
    df_ik.to_feather(OUT / "ob_rlm_cep_ikica.ft")

    summ_lin = bucket_summary(df_lin, "ce_ape_ob_rlm")
    summ_ik = bucket_summary(df_ik, "ce_ape_ob_rlm")
    (OUT / "summary.json").write_text(json.dumps(
        {"lin": summ_lin, "ikica_in_sample": summ_ik}, indent=2,
    ))
    print(f"  Lin overall median APE = {summ_lin['median_ape_overall']:.4f}  "
          f"(r1-0={summ_lin['median_ape_round_1_no_deals']:.3f}, "
          f"r1-d={summ_lin['median_ape_round_1_with_deals']:.3f}, "
          f"r2+-0={summ_lin['median_ape_round_ge2_no_deals']:.3f}, "
          f"r2+-d={summ_lin['median_ape_round_ge2_with_deals']:.3f})")
    print(f"  Ikica in-sample median APE = {summ_ik['median_ape_overall']:.4f}  "
          f"(r1-0={summ_ik['median_ape_round_1_no_deals']:.3f}, "
          f"r1-d={summ_ik['median_ape_round_1_with_deals']:.3f}, "
          f"r2+-0={summ_ik['median_ape_round_ge2_no_deals']:.3f}, "
          f"r2+-d={summ_ik['median_ape_round_ge2_with_deals']:.3f})")

    print("\nStep 4/5: spread-band decomposition on Lin cold-start")
    lin_sp = add_spread_cols(lin)
    merged_ob = df_lin.merge(
        lin_sp[["treatment", "game", "round", "time",
                "best_bid", "best_ask", "spread", "mid", "spread_rel"]],
        on=["treatment", "game", "round", "time"], how="left",
    )
    df_band, rho = ape_by_spread_band(merged_ob, "ce_ape_ob_rlm", "ce_pred_ob_rlm")
    df_band.to_csv(OUT / "ape_by_spread_ob_rlm.csv", index=False)
    print(f"  Spearman rho(spread_rel, APE) = {rho:.3f}")
    print(df_band.to_string(index=False))
    write_latex_spread_table(df_band, rho)
    write_latex_external_row(summ_lin, summ_ik)

    print("\nStep 5/5: overlay figure (OB-RLM vs GBT)")
    gbt = pd.read_feather(LIN_IN / "gbt_cep.ft")
    merged_gbt = gbt.merge(
        lin_sp[["treatment", "game", "round", "time",
                "best_bid", "best_ask", "spread", "mid", "spread_rel"]],
        on=["treatment", "game", "round", "time"], how="left",
    )
    overlay_figure(merged_ob, merged_gbt)
    print("\ndone.")


if __name__ == "__main__":
    main()
