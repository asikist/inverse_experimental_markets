"""External-dataset evaluation on Lin et al. (2020), preliminary.

Refits the four orderbook-only models on the full Ikica dataset (no train/test
split, no 50-CV ensemble) and applies them to Lin et al. orderbook features.

Lin orderbook is keyed on (MarketID, period); we treat MarketID as both
``treatment`` and ``game`` and rename ``period`` to ``round`` to mirror the
Ikica per-round target structure. Time is converted from milliseconds to
seconds (integer floor) and capped at 120s, matching Ikica's auction window.

Targets are taken from Lin_marketProfile_processed.csv (one row per
(MarketID, period); within a market eq_price is constant by construction).
Buckets follow the main paper: (round=1, n_deals=0), (round=1, n_deals>=1),
(round>=2, n_deals=0), (round>=2, n_deals>=1).

Caveats this script does NOT correct for:
  (a) Lin is multi-unit (lenval up to 3); features aggregate across units
      and players whereas Ikica is single-unit.
  (b) Feedback setting and price rule do not map 1-to-1 to Ikica
      categoricals; we use the orderbook-only models throughout.
  (c) Within a Lin market eq_price is constant across periods; we report
      a per-market dispersion diagnostic for CEP alongside the bucket APE.
  (d) Ikica per-round assumes value re-draws each round; Lin periods are
      repetitions of a fixed schedule with no dropout.

Outputs:
  code/data/results/lin_eval/lin_features.ft
  code/data/results/lin_eval/{ob_rlm_ae,cemh_global_cep,gbt_cep,gbt_ae}.ft
  code/data/results/lin_eval/summary.json

Run with: ``uv run python scripts/lin_eval.py``.
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

from data_helpers.distribution_features import (
    calc_quantile_feats,
    calculate_running_distribution,
)
from data_helpers.preprocessors.scalers import (
    denormalize_feats,
    normalize_feats,
    normalize_other,
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PREP = CODE_ROOT / "data" / "preprocessed"
LIN_DIR = CODE_ROOT / "new_data" / "Experimental-double-auctions-main" / "Data"
OUT = CODE_ROOT / "data" / "results" / "lin_eval"

DYNAMIC_TREATMENTS = ("FullSShift", "FullFromBtoS", "FullFromStoB")
MIN_Q, MAX_Q = 0.35, 0.65
ROUNDS = range(1, 6)
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


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "running_" in c and "change" not in c]


def load_ikica() -> pd.DataFrame:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    return tad.query("round <= 5 and time <= 120").reset_index(drop=True)


def load_lin_market_targets() -> pd.DataFrame:
    mp = pd.read_csv(LIN_DIR / "Lin_marketProfile_processed.csv")
    mp = mp.sort_values("MarketID").reset_index(drop=True)
    mp["round"] = mp.groupby("MarketID").cumcount() + 1
    return mp[["MarketID", "round", "eq_price", "efficiency"]].rename(
        columns={"MarketID": "treatment", "eq_price": "ce_round",
                 "efficiency": "allocative_efficiency_round"}
    )


def preprocess_lin_orderbook() -> pd.DataFrame:
    """Build a per-(treatment, game, round, time) feature frame from Lin orderbook.

    Mirrors the Ikica time_aggregate_dataset structure for the orderbook-only
    feature set. Returns one row per second-resolution snapshot up to 120s.
    """
    ob = pd.read_csv(LIN_DIR / "Lin_orderBook_processed.csv")
    ob = ob.dropna(subset=["bid"]).copy()
    ob["side"] = ob["side"].str.lower()
    ob["game"] = ob["treatment"]
    ob["time"] = (ob["time"] // 1000).astype(int)
    ob = ob[(ob["time"] >= 0) & (ob["time"] <= 120)].copy()
    # Lin is multi-unit (Smith 1962 induced-value DA, lenval up to 20):
    # one player posts multiple bids/asks for distinct units, all sharing the
    # same `id`. The shared `accumulate_dictionary` helper keys on `id` and
    # overwrites, which collapses all units of a player to the latest order.
    # Compose a unique (id, unit) key so the running distribution captures
    # every standing offer. Ikica is single-unit so this would be a no-op
    # there; we apply it only on the Lin path.
    ob["id"] = ob["id"].astype(int) * 100 + ob["unit"].fillna(0).astype(int)

    rd = calculate_running_distribution(
        ob[["treatment", "game", "round", "time", "side", "id", "bid"]],
        distribution_col="bid",
    )
    qf = rd.apply(calc_quantile_feats, axis=1).reset_index()

    deals_mask = ob["status"].isin({"FILLED", "FILLED, MARKET ORDER"})
    deals = ob[deals_mask].copy()
    deals["deal_id"] = deals.apply(
        lambda r: (r.get("match_id"), r.get("match_time")) if pd.notna(r.get("match_id"))
        else (r["id"], r["time"]),
        axis=1,
    )

    snap_keys = qf[["treatment", "game", "round", "time"]].drop_duplicates()

    def per_round_n_deals_realised(group_keys: pd.DataFrame, deals_g: pd.DataFrame) -> pd.DataFrame:
        out = group_keys.sort_values("time").copy()
        if deals_g.empty:
            out["n_unique_deals_round"] = 0
            out["realized_price"] = np.nan
            return out
        deals_g = deals_g.sort_values("time")
        deals_list = [(t_, d_, p_) for t_, d_, p_ in
                      zip(deals_g["time"].to_numpy(),
                          deals_g["deal_id"].to_numpy(),
                          deals_g["price"].to_numpy())]
        n_deals = []
        last_price = []
        seen: set = set()
        cur_price = np.nan
        di = 0
        for t in out["time"].to_numpy():
            while di < len(deals_list) and deals_list[di][0] <= t:
                seen.add(deals_list[di][1])
                cur_price = deals_list[di][2]
                di += 1
            n_deals.append(len(seen))
            last_price.append(cur_price)
        out["n_unique_deals_round"] = n_deals
        out["realized_price"] = last_price
        return out

    out_chunks = []
    for (t, g, r), grp in snap_keys.groupby(["treatment", "game", "round"]):
        deals_g = deals[(deals["treatment"] == t) & (deals["round"] == r)]
        out_chunks.append(per_round_n_deals_realised(grp.copy(), deals_g))
    deals_per_snap = pd.concat(out_chunks, ignore_index=True)

    qf = qf.merge(deals_per_snap, on=["treatment", "game", "round", "time"], how="left")
    return qf


def assemble_lin_dataset(cache: bool = True) -> pd.DataFrame:
    cache_path = OUT / "lin_features.ft"
    if cache and cache_path.exists():
        print(f"  loading cached features from {cache_path}", flush=True)
        return pd.read_feather(cache_path)
    feats = preprocess_lin_orderbook()
    targets = load_lin_market_targets()
    df = feats.merge(targets, on=["treatment", "round"], how="inner")
    df = df.query("round <= 5").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_feather(cache_path)
    return df


def fit_full_ikica_models(tad: pd.DataFrame, feats: list[str]):
    """Refit the four orderbook-only models on the full Ikica dataset."""
    print("fitting OB-RLM AE on full Ikica ...", flush=True)
    formula = "allocative_efficiency_round ~ (" + " + ".join(feats) + ")"
    rlm_per_round = {}
    for rd in ROUNDS:
        tr = tad.query("round <= @rd").copy()
        if len(tr) == 0:
            continue
        tr[feats], _, _ = normalize_feats(tr[feats].values, MIN_Q, MAX_Q)
        rlm_per_round[rd] = smf.rlm(formula, data=tr).fit()

    print("fitting CEMH-global CEP on full Ikica ...", flush=True)
    cemh_per_nd = {}
    cemh_formula = "ce_round ~ realized_price - 1"
    for nd in N_DEAL_PRICES:
        tr = tad.query("n_unique_deals_round == @nd")
        if len(tr) == 0:
            continue
        cemh_per_nd[nd] = smf.rlm(cemh_formula, data=tr).fit()

    print("fitting GBT CEP on full Ikica ...", flush=True)
    Xc = tad[feats].values
    Xc_norm, Xc_med, Xc_iqr = normalize_feats(Xc, MIN_Q, MAX_Q)
    yc = tad["ce_round"].values[:, np.newaxis]
    yc_norm = normalize_other(yc, Xc_med, Xc_iqr)
    gbt_cep = cb.CatBoostRegressor(
        boost_from_average=True, loss_function="Quantile",
        grow_policy="Lossguide", task_type="CPU", verbose=False,
    )
    gbt_cep.grid_search(
        GBT_GRID_CEP, Xc_norm, y=yc_norm, cv=5, partition_random_seed=0,
        calc_cv_statistics=True, search_by_train_test_split=True,
        refit=True, shuffle=True, stratified=None, train_size=0.8,
        verbose=False, plot=False,
    )

    print("fitting GBT AE on full Ikica ...", flush=True)
    Xa = tad[feats].values
    Xa_norm, _, _ = normalize_feats(Xa, MIN_Q, MAX_Q)
    ya = tad["allocative_efficiency_round"].values[:, np.newaxis]
    gbt_ae = cb.CatBoostRegressor(
        boost_from_average=True, loss_function="Quantile",
        grow_policy="SymmetricTree", task_type="CPU", verbose=False,
    )
    gbt_ae.grid_search(
        GBT_GRID_AE, Xa_norm, y=ya, cv=5, partition_random_seed=0,
        calc_cv_statistics=False, search_by_train_test_split=True,
        refit=True, shuffle=False, stratified=None, train_size=0.5,
        verbose=False, plot=False,
    )

    return rlm_per_round, cemh_per_nd, gbt_cep, gbt_ae


def predict_on_lin(lin: pd.DataFrame, feats: list[str], rlm_per_round, cemh_per_nd, gbt_cep, gbt_ae):
    frames = []
    keys = ["treatment", "game", "round", "time", "n_unique_deals_round"]

    for rd in ROUNDS:
        if rd not in rlm_per_round:
            continue
        sub = lin.query("round == @rd").copy()
        if sub.empty:
            continue
        sub[feats], _, _ = normalize_feats(sub[feats].values, MIN_Q, MAX_Q)
        pred = np.clip(rlm_per_round[rd].predict(sub).to_numpy(), 0.0, 1.0)
        tgt = sub["allocative_efficiency_round"].to_numpy()
        denom = np.where(tgt == 0, np.where(pred == 0, 1.0, pred), tgt)
        frames.append(pd.DataFrame({
            **{k: sub[k].to_numpy() for k in keys},
            "ae_pred_ob_rlm": pred,
            "ae_target": tgt,
            "ae_ape_ob_rlm": np.abs(pred - tgt) / denom,
        }))
    ob_rlm_ae = pd.concat(frames, ignore_index=True)

    frames = []
    for nd in N_DEAL_PRICES:
        if nd not in cemh_per_nd:
            continue
        sub = lin.query("n_unique_deals_round == @nd").copy()
        if sub.empty:
            continue
        valid = sub.dropna(subset=["realized_price"]).copy()
        if valid.empty:
            continue
        pred = cemh_per_nd[nd].predict(valid).to_numpy()
        tgt = valid["ce_round"].to_numpy()
        ape = np.where(tgt > 0, np.abs(pred - tgt) / tgt, np.nan)
        frames.append(pd.DataFrame({
            **{k: valid[k].to_numpy() for k in keys},
            "ce_pred_cemh": pred,
            "ce_target": tgt,
            "ce_ape_cemh": ape,
        }))
    cemh_global_cep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    Xn, Xn_med, Xn_iqr = normalize_feats(lin[feats].values, MIN_Q, MAX_Q)
    pred_norm = gbt_cep.predict(Xn)
    pred = denormalize_feats(pred_norm[:, np.newaxis], Xn_med, Xn_iqr).squeeze()
    tgt = lin["ce_round"].to_numpy()
    ape = np.where(tgt > 0, np.abs(pred - tgt) / tgt, np.nan)
    gbt_cep_df = pd.DataFrame({
        **{k: lin[k].to_numpy() for k in keys},
        "ce_pred_gbt": pred,
        "ce_target": tgt,
        "ce_ape_gbt": ape,
    })

    Xn2, _, _ = normalize_feats(lin[feats].values, MIN_Q, MAX_Q)
    pred_ae = np.clip(gbt_ae.predict(Xn2), 0.0, 1.0)
    tgt_ae = lin["allocative_efficiency_round"].to_numpy()
    denom = np.where(tgt_ae == 0, np.where(pred_ae == 0, 1.0, pred_ae), tgt_ae)
    gbt_ae_df = pd.DataFrame({
        **{k: lin[k].to_numpy() for k in keys},
        "ae_pred_gbt": pred_ae,
        "ae_target": tgt_ae,
        "ae_ape_gbt": np.abs(pred_ae - tgt_ae) / denom,
    })

    return ob_rlm_ae, cemh_global_cep, gbt_cep_df, gbt_ae_df


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


def per_market_cep_dispersion(gbt_cep_df: pd.DataFrame) -> dict:
    snapshot = (gbt_cep_df.dropna(subset=["ce_pred_gbt"])
                .groupby(["treatment", "round"])
                .agg(pred=("ce_pred_gbt", "median"), target=("ce_target", "first"))
                .reset_index())
    multi = (snapshot.groupby("treatment").size() > 1)
    multi_ids = multi[multi].index
    sub = snapshot[snapshot["treatment"].isin(multi_ids)].copy()
    if sub.empty:
        return {"n_markets": 0, "n_periods": 0, "median_within_market_pred_dispersion": None}
    disp = sub.groupby("treatment")["pred"].agg(lambda v: float(np.std(v) / max(np.mean(v), 1e-9)))
    return {
        "n_markets": int(len(multi_ids)),
        "n_periods": int(len(sub)),
        "median_within_market_pred_cv": float(disp.median()),
    }


def main() -> None:
    np.random.seed(1)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1/4: load and assemble Lin features + targets")
    t0 = time.perf_counter()
    lin = assemble_lin_dataset()
    print(f"  Lin rows: {len(lin):,}  markets: {lin['treatment'].nunique():,}  "
          f"periods used: {sorted(lin['round'].unique().tolist())}  "
          f"({time.perf_counter()-t0:.1f}s)")

    print("\nStep 2/4: load Ikica training data")
    tad = load_ikica()
    feats = feature_cols(tad)
    print(f"  Ikica rows: {len(tad):,}  features: {len(feats)}")

    print("\nStep 3/4: refit four orderbook-only models on full Ikica")
    t0 = time.perf_counter()
    rlm_per_round, cemh_per_nd, gbt_cep, gbt_ae = fit_full_ikica_models(tad, feats)
    print(f"  done ({(time.perf_counter()-t0)/60:.2f} min)")

    print("\nStep 4/4: predict on Lin and summarise")
    ob_rlm_ae, cemh_cep, gbt_cep_df, gbt_ae_df = predict_on_lin(
        lin, feats, rlm_per_round, cemh_per_nd, gbt_cep, gbt_ae,
    )
    ob_rlm_ae.to_feather(OUT / "ob_rlm_ae.ft")
    cemh_cep.to_feather(OUT / "cemh_global_cep.ft")
    gbt_cep_df.to_feather(OUT / "gbt_cep.ft")
    gbt_ae_df.to_feather(OUT / "gbt_ae.ft")

    summary = {
        "ob_rlm_ae": bucket_summary(ob_rlm_ae, "ae_ape_ob_rlm"),
        "cemh_global_cep": bucket_summary(cemh_cep, "ce_ape_cemh") if len(cemh_cep) else {},
        "gbt_cep": bucket_summary(gbt_cep_df, "ce_ape_gbt"),
        "gbt_ae": bucket_summary(gbt_ae_df, "ae_ape_gbt"),
        "gbt_cep_within_market_dispersion": per_market_cep_dispersion(gbt_cep_df),
        "lin_data": {
            "snapshot_rows": int(len(lin)),
            "markets_with_targets": int(lin["treatment"].nunique()),
            "rounds_used": sorted(int(r) for r in lin["round"].unique()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote", OUT / "summary.json")
    for m, s in summary.items():
        if isinstance(s, dict) and "median_ape_overall" in s:
            print(f"  {m:>20s}  median APE = {s['median_ape_overall']:.4f}  "
                  f"(r1-0={s['median_ape_round_1_no_deals']:.3f}, "
                  f"r1-deal={s['median_ape_round_1_with_deals']:.3f}, "
                  f"r2+-0={s['median_ape_round_ge2_no_deals']:.3f}, "
                  f"r2+-deal={s['median_ape_round_ge2_with_deals']:.3f})")


if __name__ == "__main__":
    main()
