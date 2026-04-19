"""Simple CEP baselines.

Two baselines are evaluated on the same per-sample-id test rows used by the
four paper models (we reuse data/results/ce_price/emh.ft as the evaluation
grid so the row set is identical down to the time-step), producing feather
files with the same schema:

* treatment_mean_ce: constant prediction per treatment, equal to the mean of
  ce_round over distinct training (game, round) pairs of that treatment.
  Pure prior with no per-prediction information.

* book_midpoint: (best-bid + best-ask) / 2 at prediction time, using the
  running max buyer bid (running_buyer_bid_quant_10) and the running min
  seller ask (running_seller_bid_quant_00). When one side is zero (no bids
  or no asks yet) the other side is used; when both are zero, the
  treatment-mean prediction is used as a fallback.

Evaluation target is ce_round.

Outputs:
  data/results/ce_price/treatment_mean_ce.ft
  data/results/ce_price/book_midpoint.ft
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PREP = CODE_ROOT / "data" / "preprocessed"
OUT = CODE_ROOT / "data" / "results" / "ce_price"

TA_COLS = [
    "treatment", "game", "round", "time", "ce_round",
    "n_unique_deals_round",
    "running_buyer_bid_quant_10",
    "running_seller_bid_quant_00",
]

RESULT_COLS = [
    "index", "treatment", "game", "round", "time",
    "n_unique_deals_round", "ce_ape", "sample_id", "model",
]


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tts = pd.read_feather(PREP / "train_test_split.ft")
    emh = pd.read_feather(OUT / "emh.ft")
    return tad[TA_COLS].copy(), tts, emh


def compute_treatment_mean(tad: pd.DataFrame, train_games: pd.DataFrame) -> pd.Series:
    train_rows = tad.merge(train_games, on=["treatment", "game"])
    uniq_round = train_rows.drop_duplicates(["treatment", "game", "round"])
    return uniq_round.groupby("treatment")["ce_round"].mean()


def ape(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(pred - target) / target


def make_frame(test_rows: pd.DataFrame, pred: np.ndarray, sid: int, model: str) -> pd.DataFrame:
    return pd.DataFrame({
        "treatment": test_rows["treatment"].to_numpy(),
        "game": test_rows["game"].to_numpy(),
        "round": test_rows["round"].to_numpy(),
        "time": test_rows["time"].to_numpy(),
        "n_unique_deals_round": test_rows["n_unique_deals_round"].to_numpy(),
        "ce_ape": ape(pred, test_rows["ce_round"].to_numpy()),
        "sample_id": sid,
        "model": model,
    })


def main() -> None:
    tad, tts, emh = load_base()
    tmean_frames: list[pd.DataFrame] = []
    bmid_frames: list[pd.DataFrame] = []

    for sid, split in tts.groupby("sample_id"):
        train_games = split.loc[split["dataset_type"] == "train", ["treatment", "game"]]
        tmean = compute_treatment_mean(tad, train_games)

        eval_grid = emh.loc[emh["sample_id"] == sid,
                            ["treatment", "game", "round", "time"]]
        test_rows = eval_grid.merge(tad, on=["treatment", "game", "round", "time"], how="left")
        if test_rows["ce_round"].isna().any():
            n_missing = int(test_rows["ce_round"].isna().sum())
            raise RuntimeError(f"{n_missing} eval rows missing from time_aggregate for sample {sid}")

        tmean_pred = test_rows["treatment"].map(tmean).to_numpy()
        tmean_frames.append(make_frame(test_rows, tmean_pred, sid, "treatment_mean"))

        bb = test_rows["running_buyer_bid_quant_10"].to_numpy()
        sa = test_rows["running_seller_bid_quant_00"].to_numpy()
        mid = np.where(
            (bb > 0) & (sa > 0), (bb + sa) / 2,
            np.where(bb > 0, bb, np.where(sa > 0, sa, tmean_pred)),
        )
        bmid_frames.append(make_frame(test_rows, mid, sid, "book_midpoint"))

    tmean_df = pd.concat(tmean_frames, ignore_index=True)
    bmid_df = pd.concat(bmid_frames, ignore_index=True)
    tmean_df.insert(0, "index", np.arange(len(tmean_df)))
    bmid_df.insert(0, "index", np.arange(len(bmid_df)))

    OUT.mkdir(parents=True, exist_ok=True)
    tmean_path = OUT / "treatment_mean_ce.ft"
    bmid_path = OUT / "book_midpoint.ft"
    tmean_df[RESULT_COLS].to_feather(tmean_path)
    bmid_df[RESULT_COLS].to_feather(bmid_path)

    print(f"wrote {tmean_path} ({len(tmean_df):,} rows)")
    print(f"wrote {bmid_path} ({len(bmid_df):,} rows)")
    for name, df in [("treatment_mean", tmean_df), ("book_midpoint", bmid_df)]:
        med = df["ce_ape"].median()
        print(f"  {name:>15s} overall median APE: {med:.4f}")


if __name__ == "__main__":
    main()
