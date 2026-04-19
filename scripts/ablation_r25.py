"""Orderbook-only ablation.

Refits four models with only the running bid/ask quantiles as inputs, removing
all treatment descriptors, the realised deal price, $n_{\\text{deals}}$ and the
round index from the feature vector. This tests whether the paper's models
learn market microstructure or merely exploit experimental-protocol
information.

Models refit:
  * OB-RLM AE, orderbook-only linear regression (no feedback_setting grouping,
    no realised_price, no n_unique_deals_round)
  * CEMH CEP with the $price\\_rule \\times feedback\\_setting$ grouping removed
    (collapses to a single global scalar rescaling of the realised deal
    price); reported as CEMH-global
  * GBT CEP, orderbook-only CatBoost (no price-rule / feedback-setting dummies,
    no n, no round)
  * GBT AE, orderbook-only CatBoost (same as above)

The OB-RLM CEP model is already orderbook-only and is not refit here; CEMH AE
has no orderbook inputs and the reverse ablation does not apply.

Outputs:
  code/data/results/ablations/r25/{ob_rlm_ae, cemh_global_cep, gbt_cep,
                                    gbt_ae}.ft
  code/data/results/ablations/r25/summary.json

Run with: ``uv run python scripts/ablation_r25.py``.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import catboost as cb
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
OUT = CODE_ROOT / "data" / "results" / "ablations" / "r25"

DYNAMIC_TREATMENTS = ("FullSShift", "FullFromBtoS", "FullFromStoB")
MIN_Q, MAX_Q = 0.35, 0.65
ROUNDS = range(1, 5)
N_DEAL_PRICES = range(0, 6)

GBT_GRID_CEP = {
    "n_estimators": [1, 2, 3],
    "depth": [1, 2, 3],
    "learning_rate": [0.001, 0.005, 0.01],
    "max_leaves": [10, 20, 30, 40, 50, 70],
}
GBT_GRID_AE = {
    "n_estimators": [10, 20, 30],
    "depth": [5, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "min_data_in_leaf": [1, 2, 5],
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    tad = tad.query("round <= 5 and time <= 120").reset_index(drop=True)
    tts = pd.read_feather(PREP / "train_test_split.ft")
    return tad, tts


def running_feature_cols(tad: pd.DataFrame) -> list[str]:
    return [c for c in tad.columns if "running_" in c and "change" not in c]


def get_split(tad: pd.DataFrame, tts: pd.DataFrame, sid: int):
    split = tts.query("sample_id == @sid")
    train_ids = split.loc[split["dataset_type"] == "train", ["treatment", "game"]]
    test_ids = split.loc[split["dataset_type"] == "test", ["treatment", "game"]]
    return (tad.merge(train_ids, on=["treatment", "game"]),
            tad.merge(test_ids, on=["treatment", "game"]))


def _ob_rlm_ae_sample(train_df, test_df, feature_cols, sid):
    frames = []
    formula = "allocative_efficiency_round ~ (" + " + ".join(feature_cols) + ")"
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
                "model": "ob_rlm_ae_r25",
            }))
    return pd.concat(frames, ignore_index=True)


def _cemh_global_cep_sample(train_df, test_df, sid):
    frames = []
    formula = "ce_round ~ realized_price - 1"
    for nd in N_DEAL_PRICES:
        tr = train_df.query("n_unique_deals_round == @nd")
        if len(tr) == 0:
            continue
        model = smf.rlm(formula, data=tr).fit()
        for rd in ROUNDS:
            te = test_df.query("round == @rd and n_unique_deals_round == @nd")
            if len(te) == 0:
                continue
            pred = model.predict(te).to_numpy()
            tgt = te["ce_round"].to_numpy()
            frames.append(pd.DataFrame({
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ce_ape": np.abs(pred - tgt) / tgt,
                "sample_id": sid,
                "model": "cemh_global_cep_r25",
            }))
    return pd.concat(frames, ignore_index=True)


def _gbt_cep_sample(train_df, test_df, feature_cols, sid):
    X = train_df[feature_cols].values
    X_norm, X_median, X_iqr = normalize_feats(X, MIN_Q, MAX_Q)
    y = train_df["ce_round"].values[:, np.newaxis]
    y_norm = normalize_other(y, X_median, X_iqr)

    reg = cb.CatBoostRegressor(
        boost_from_average=True, loss_function="Quantile",
        grow_policy="Lossguide", task_type="CPU", verbose=False,
    )
    reg.grid_search(
        GBT_GRID_CEP, X_norm, y=y_norm, cv=5, partition_random_seed=0,
        calc_cv_statistics=True, search_by_train_test_split=True,
        refit=True, shuffle=True, stratified=None, train_size=0.8,
        verbose=False, plot=False,
    )

    frames = []
    for rd in ROUNDS:
        for nd in N_DEAL_PRICES:
            te = test_df.query("round == @rd and n_unique_deals_round == @nd")
            if len(te) == 0:
                continue
            X_test = te[feature_cols].values
            X_test_norm, X_test_median, X_test_iqr = normalize_feats(X_test, MIN_Q, MAX_Q)
            pred_norm = reg.predict(X_test_norm)
            pred = denormalize_feats(pred_norm[:, np.newaxis], X_test_median, X_test_iqr).squeeze()
            tgt = te["ce_round"].to_numpy()
            frames.append(pd.DataFrame({
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ce_ape": np.abs(pred - tgt) / tgt,
                "sample_id": sid,
                "model": "gbt_cep_r25",
            }))
    return pd.concat(frames, ignore_index=True)


def _gbt_ae_sample(train_df, test_df, feature_cols, sid):
    X = train_df[feature_cols].values
    X_norm, _, _ = normalize_feats(X, MIN_Q, MAX_Q)
    y = train_df["allocative_efficiency_round"].values[:, np.newaxis]

    reg = cb.CatBoostRegressor(
        boost_from_average=True, loss_function="Quantile",
        grow_policy="SymmetricTree", task_type="CPU", verbose=False,
    )
    reg.grid_search(
        GBT_GRID_AE, X_norm, y=y, cv=5, partition_random_seed=0,
        calc_cv_statistics=False, search_by_train_test_split=True,
        refit=True, shuffle=False, stratified=None, train_size=0.5,
        verbose=False, plot=False,
    )

    frames = []
    for rd in ROUNDS:
        for nd in N_DEAL_PRICES:
            te = test_df.query("round == @rd and n_unique_deals_round == @nd")
            if len(te) == 0:
                continue
            X_test = te[feature_cols].values
            X_test_norm, _, _ = normalize_feats(X_test, MIN_Q, MAX_Q)
            pred = np.clip(reg.predict(X_test_norm), 0.0, 1.0)
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
                "model": "gbt_ae_r25",
            }))
    return pd.concat(frames, ignore_index=True)


def run() -> None:
    np.random.seed(1)
    OUT.mkdir(parents=True, exist_ok=True)

    tad, tts = load_data()
    feature_cols = running_feature_cols(tad)

    per_model = {
        "ob_rlm_ae_r25": [],
        "cemh_global_cep_r25": [],
        "gbt_cep_r25": [],
        "gbt_ae_r25": [],
    }
    total = tts["sample_id"].max() + 1
    for sid in range(total):
        train_df, test_df = get_split(tad, tts, sid)
        per_model["ob_rlm_ae_r25"].append(
            _ob_rlm_ae_sample(train_df, test_df, feature_cols, sid))
        per_model["cemh_global_cep_r25"].append(
            _cemh_global_cep_sample(train_df, test_df, sid))
        per_model["gbt_cep_r25"].append(
            _gbt_cep_sample(train_df, test_df, feature_cols, sid))
        per_model["gbt_ae_r25"].append(
            _gbt_ae_sample(train_df, test_df, feature_cols, sid))
        if (sid + 1) % 5 == 0:
            print(f"  finished sample {sid + 1}/{total}", flush=True)

    files = {
        "ob_rlm_ae_r25": OUT / "ob_rlm_ae.ft",
        "cemh_global_cep_r25": OUT / "cemh_global_cep.ft",
        "gbt_cep_r25": OUT / "gbt_cep.ft",
        "gbt_ae_r25": OUT / "gbt_ae.ft",
    }
    summary: dict = {}
    for model, frames in per_model.items():
        df = pd.concat(frames, ignore_index=True)
        df.to_feather(files[model])
        err_col = "ce_ape" if "cep" in model else "ae_ape"
        summary[model] = {
            "rows": int(len(df)),
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
        print(f"  {model:>25s}  n={len(df):>7d}  median APE = {summary[model]['median_ape_overall']:.4f}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary written to", OUT / "summary.json")


if __name__ == "__main__":
    run()
