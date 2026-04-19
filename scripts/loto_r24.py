"""Leave-one-treatment-out cross-validation.

Operationalises "leave-one-schedule-out" as holding out all games of one
experimental treatment at a time. The treatment label fixes the
reservation-value generation rule, which is the closest match in this
dataset to a supply-demand "schedule". Dynamic-schedule treatments
(FullSShift, FullFromBtoS, FullFromStoB) are excluded as in the main
analysis.

Feature set is orderbook quantiles only (matches ablation_r25). Dropping the
categorical treatment / feedback / price-rule dummies avoids the degenerate
case in which a held-out treatment introduces a categorical level never
observed in training, which would test categorical extrapolation rather than
structural generalisation.

Models refit per fold:
  * OB-RLM AE (orderbook-only linear regression over rounds 1--4)
  * CEMH-global CEP (scalar rescaling of the realised price, per n_deals)
  * GBT CEP (CatBoost, quantile loss, grid-search)
  * GBT AE (CatBoost, quantile loss, grid-search)

Outputs:
  code/data/results/loto_r24/{ob_rlm_ae, cemh_global_cep, gbt_cep, gbt_ae}.ft
  code/data/results/loto_r24/summary.json

Run with: ``uv run python scripts/loto_r24.py``.
"""
from __future__ import annotations

import json
import time
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
OUT = CODE_ROOT / "data" / "results" / "loto_r24"

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


def load_data() -> pd.DataFrame:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    return tad.query("round <= 5 and time <= 120").reset_index(drop=True)


def running_feature_cols(tad: pd.DataFrame) -> list[str]:
    return [c for c in tad.columns if "running_" in c and "change" not in c]


def split_by_treatment(tad: pd.DataFrame, held: str):
    tr = tad[tad["treatment"] != held].reset_index(drop=True)
    te = tad[tad["treatment"] == held].reset_index(drop=True)
    return tr, te


def _ob_rlm_ae_fold(train_df, test_df, feats, held):
    frames = []
    formula = "allocative_efficiency_round ~ (" + " + ".join(feats) + ")"
    for rd in ROUNDS:
        tr = train_df.query("round <= @rd").copy()
        if len(tr) == 0:
            continue
        tr[feats], _, _ = normalize_feats(tr[feats].values, MIN_Q, MAX_Q)
        model = smf.rlm(formula, data=tr).fit()
        for nd in N_DEAL_PRICES:
            te = test_df.query("round == @rd and n_unique_deals_round == @nd").copy()
            if len(te) == 0:
                continue
            te[feats], _, _ = normalize_feats(te[feats].values, MIN_Q, MAX_Q)
            pred = np.clip(model.predict(te).to_numpy(), 0.0, 1.0)
            tgt = te["allocative_efficiency_round"].to_numpy()
            denom = np.where(tgt == 0, np.where(pred == 0, 1.0, pred), tgt)
            frames.append(pd.DataFrame({
                "held_out_treatment": held,
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ae_ape": np.abs(pred - tgt) / denom,
                "model": "ob_rlm_ae_loto",
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cemh_global_cep_fold(train_df, test_df, held):
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
                "held_out_treatment": held,
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ce_ape": np.abs(pred - tgt) / tgt,
                "model": "cemh_global_cep_loto",
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _gbt_cep_fold(train_df, test_df, feats, held):
    X = train_df[feats].values
    X_norm, X_med, X_iqr = normalize_feats(X, MIN_Q, MAX_Q)
    y = train_df["ce_round"].values[:, np.newaxis]
    y_norm = normalize_other(y, X_med, X_iqr)

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
            X_test = te[feats].values
            X_test_norm, X_test_med, X_test_iqr = normalize_feats(X_test, MIN_Q, MAX_Q)
            pred_norm = reg.predict(X_test_norm)
            pred = denormalize_feats(pred_norm[:, np.newaxis], X_test_med, X_test_iqr).squeeze()
            tgt = te["ce_round"].to_numpy()
            frames.append(pd.DataFrame({
                "held_out_treatment": held,
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ce_ape": np.abs(pred - tgt) / tgt,
                "model": "gbt_cep_loto",
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _gbt_ae_fold(train_df, test_df, feats, held):
    X = train_df[feats].values
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
            X_test = te[feats].values
            X_test_norm, _, _ = normalize_feats(X_test, MIN_Q, MAX_Q)
            pred = np.clip(reg.predict(X_test_norm), 0.0, 1.0)
            tgt = te["allocative_efficiency_round"].to_numpy()
            denom = np.where(tgt == 0, np.where(pred == 0, 1.0, pred), tgt)
            frames.append(pd.DataFrame({
                "held_out_treatment": held,
                "treatment": te["treatment"].to_numpy(),
                "game": te["game"].to_numpy(),
                "round": te["round"].to_numpy(),
                "time": te["time"].to_numpy(),
                "n_unique_deals_round": te["n_unique_deals_round"].to_numpy(),
                "ae_ape": np.abs(pred - tgt) / denom,
                "model": "gbt_ae_loto",
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run() -> None:
    np.random.seed(1)
    OUT.mkdir(parents=True, exist_ok=True)

    tad = load_data()
    feats = running_feature_cols(tad)
    treatments = sorted(tad["treatment"].unique())
    print(f"LOTO folds: {len(treatments)}  |  orderbook features: {len(feats)}")

    per_model = {
        "ob_rlm_ae_loto": [],
        "cemh_global_cep_loto": [],
        "gbt_cep_loto": [],
        "gbt_ae_loto": [],
    }
    t_all = time.perf_counter()
    for i, held in enumerate(treatments, start=1):
        tr, te = split_by_treatment(tad, held)
        t0 = time.perf_counter()
        per_model["ob_rlm_ae_loto"].append(_ob_rlm_ae_fold(tr, te, feats, held))
        per_model["cemh_global_cep_loto"].append(_cemh_global_cep_fold(tr, te, held))
        per_model["gbt_cep_loto"].append(_gbt_cep_fold(tr, te, feats, held))
        per_model["gbt_ae_loto"].append(_gbt_ae_fold(tr, te, feats, held))
        dt = time.perf_counter() - t0
        print(f"  [{i:2d}/{len(treatments)}] held={held:12s}  test rows={len(te):4d}  "
              f"train rows={len(tr):5d}  ({dt:5.1f} s)", flush=True)

    files = {
        "ob_rlm_ae_loto": OUT / "ob_rlm_ae.ft",
        "cemh_global_cep_loto": OUT / "cemh_global_cep.ft",
        "gbt_cep_loto": OUT / "gbt_cep.ft",
        "gbt_ae_loto": OUT / "gbt_ae.ft",
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
            "median_ape_per_fold": {
                t: float(df[df["held_out_treatment"] == t][err_col].median())
                for t in treatments
                if len(df[df["held_out_treatment"] == t]) > 0
            },
        }
        print(f"  {model:>25s}  n={len(df):>7d}  median APE = {summary[model]['median_ape_overall']:.4f}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Total wall time: {(time.perf_counter() - t_all)/60:.2f} min")
    print("Summary written to", OUT / "summary.json")


if __name__ == "__main__":
    run()
