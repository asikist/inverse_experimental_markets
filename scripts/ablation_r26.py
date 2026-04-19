"""No-deal-price ablation.

Refits the OB-RLM AE predictor without the realised deal price to test whether
the AE score is driven by circular dependence on already-observed prices.

As documented in the paper feature audit (Sec.\\ ``Model Comparison Overview'',
\\cref{tab:model:features}), the realised deal price only enters three of the
proposed predictors: CEMH for CEP, EMH for CEP and OB-RLM for AE. Of these,
CEMH for CEP reduces to a per-treatment constant once $\\vdealprice$ is
removed, which coincides numerically with the Treatment-Mean baseline already
reported in the appendix and is cross-referenced there. EMH for CEP has no
non-trivial variant without $\\vdealprice$. Only OB-RLM for AE remains and is
refit here.

Outputs:
  code/data/results/ablations/r26/ob_rlm_ae.ft
  code/data/results/ablations/r26/summary.json

Run with: ``uv run python scripts/ablation_r26.py``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_helpers.preprocessors.scalers import normalize_feats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PREP = CODE_ROOT / "data" / "preprocessed"
OUT = CODE_ROOT / "data" / "results" / "ablations" / "r26"

DYNAMIC_TREATMENTS = ("FullSShift", "FullFromBtoS", "FullFromStoB")
MIN_Q, MAX_Q = 0.35, 0.65
ROUNDS = range(1, 5)
N_DEAL_PRICES = range(0, 6)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    tad = tad.query("round <= 5 and time <= 120").reset_index(drop=True)
    tts = pd.read_feather(PREP / "train_test_split.ft")
    return tad, tts


def running_feature_cols(tad: pd.DataFrame) -> list[str]:
    return [c for c in tad.columns if "running_" in c and "change" not in c]


def get_split(tad, tts, sid):
    split = tts.query("sample_id == @sid")
    train_ids = split.loc[split["dataset_type"] == "train", ["treatment", "game"]]
    test_ids = split.loc[split["dataset_type"] == "test", ["treatment", "game"]]
    return (tad.merge(train_ids, on=["treatment", "game"]),
            tad.merge(test_ids, on=["treatment", "game"]))


def _ob_rlm_ae_no_deal_price(train_df, test_df, feature_cols, sid):
    """OB-RLM AE matched to the main model (feedback_setting grouping and
    n_unique_deals_round retained, per-round fit on ``round <= rd``) but with
    the realised_price term removed."""
    frames = []
    formula = (
        "allocative_efficiency_round ~ feedback_setting:("
        + " + ".join(feature_cols)
        + " + n_unique_deals_round)"
    )
    for rd in ROUNDS:
        tr = train_df.query("round <= @rd").copy()
        if len(tr) == 0:
            continue
        tr[feature_cols], _, _ = normalize_feats(tr[feature_cols].values, MIN_Q, MAX_Q)
        model = smf.rlm(formula, data=tr).fit()
        for nd in N_DEAL_PRICES:
            te = test_df.query("round == @rd and n_unique_deals_round == @nd").copy()
            if len(te) == 0:
                continue
            te[feature_cols], _, _ = normalize_feats(te[feature_cols].values, MIN_Q, MAX_Q)
            pred = np.clip(model.predict(te).to_numpy(), 0.0, 1.0)
            tgt = te["allocative_efficiency_round"].to_numpy()
            denom = np.where(tgt == 0, np.where(pred == 0, 1.0, pred), tgt)
            frames.append(pd.DataFrame({
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ae_ape": np.abs(pred - tgt) / denom,
                "sample_id": sid,
                "model": "ob_rlm_ae_r26",
            }))
    return pd.concat(frames, ignore_index=True)


def run() -> None:
    np.random.seed(1)
    OUT.mkdir(parents=True, exist_ok=True)

    tad, tts = load_data()
    feature_cols = running_feature_cols(tad)

    frames = []
    total = tts["sample_id"].max() + 1
    for sid in range(total):
        train_df, test_df = get_split(tad, tts, sid)
        frames.append(_ob_rlm_ae_no_deal_price(train_df, test_df, feature_cols, sid))
        if (sid + 1) % 10 == 0:
            print(f"  finished sample {sid + 1}/{total}", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_feather(OUT / "ob_rlm_ae.ft")

    summary = {
        "ob_rlm_ae_r26": {
            "rows": int(len(df)),
            "median_ape_overall": float(df["ae_ape"].median()),
            "median_ape_round_1_no_deals": float(
                df.query("round == 1 and n_unique_deals_round == 0")["ae_ape"].median()),
            "median_ape_round_1_with_deals": float(
                df.query("round == 1 and n_unique_deals_round >= 1")["ae_ape"].median()),
            "median_ape_round_ge2_no_deals": float(
                df.query("round > 1 and n_unique_deals_round == 0")["ae_ape"].median()),
            "median_ape_round_ge2_with_deals": float(
                df.query("round > 1 and n_unique_deals_round >= 1")["ae_ape"].median()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  ob_rlm_ae_r26  n={len(df)}  median APE = {summary['ob_rlm_ae_r26']['median_ape_overall']:.4f}")
    print("Summary written to", OUT / "summary.json")


if __name__ == "__main__":
    run()
