"""Model diagnostics + feature-response panels.

Computes per-task (AE, CEP) for the two non-trivial models (OB-RLM, GBT):

  Residual / fit panels
  ---------------------
  * predicted-vs-actual scatter (rasterized; CEP outliers clipped for legibility)
  * residual distribution (histogram with kernel density)
  * residual boxplot across the four (Round, Deals) buckets used in
    tab_ae_ape / tab_cep_ape

  Feature-response panels (GBT only; OB-RLM interpretability is covered by
  the existing coefficient heatmaps in
  fig:ae:obrlm:params / fig:ce_ape:obrlm:params)
  -----------------------------------------------------------------------------
  * partial dependence plots for the bid and ask medians (D5) and the
    highest-importance bid and ask extremes; PDPs are reported in raw target
    units (AE, raw price for CEP) and centered on the grand-mean prediction
    so the response amplitude is directly readable
  * SHAP beeswarm (TreeSHAP via CatBoost) across the top 10 features by
    mean |phi|; SHAP is converted to raw target units (CEP uses the per-row
    IQR scale that the GBT was trained on)
  * grouped-|SHAP| bar: sum |phi| across bid quantiles, ask quantiles and
    categoricals.  This sidesteps TreeSHAP's credit splitting on correlated
    quantiles and exposes the bid/ask asymmetry directly.

The script refits each model ONCE on sample_id=0's train split (not across
the 50 CV splits).  Explanations are about model response, not estimator
variance, so repeated refits would not add information.

Outputs
-------
  code/data/results/diagnostics/
      predictions_{task}_{model}.ft   test-set row-level predictions + residuals
      shap_{task}_gbt.ft              per-row SHAP values (GBT only, raw units)
      pdp_{task}_gbt.ft               PDP curves in raw target units
      importance_{task}_gbt.json      CatBoost feature_importances_
      summary.json                    aggregate metrics + numbers cited in prose

  paper/epj/figures/diagnostics/
      m2_{task}_{model}.pdf           predicted vs actual + residual hist + boxplot
      r29_{task}_gbt.pdf              PDPs + SHAP beeswarm + grouped |phi|

All raster artists (scatters, beeswarm) are rendered at 300 dpi inside the
PDF; axes, labels and lines remain vector for sharp print quality.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import catboost as cb
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
REPO_ROOT = CODE_ROOT.parent
PREP = CODE_ROOT / "data" / "preprocessed"
OUT_DATA = CODE_ROOT / "data" / "results" / "diagnostics"
OUT_FIG = REPO_ROOT / "paper" / "epj" / "figures" / "diagnostics"

DYNAMIC_TREATMENTS = ("FullSShift", "FullFromBtoS", "FullFromStoB")
MIN_Q, MAX_Q = 0.35, 0.65
BUCKET_ORDER = ["R1/D0", "R1/D1+", "R2+/D0", "R2+/D1+"]

GBT_GRID_AE = {"n_estimators": [10, 20, 30], "depth": [5, 6, 8],
               "learning_rate": [0.001, 0.01, 0.1], "min_data_in_leaf": [1, 2, 5]}
GBT_GRID_CEP = {"n_estimators": [1, 2, 3], "depth": [1, 2, 3],
                "learning_rate": [0.001, 0.005, 0.01],
                "max_leaves": [10, 20, 30, 40, 50, 70]}

# Rendering: vector axes + rasterized heavy artists at 300 dpi keeps PDFs small
# and fast to render in Acrobat / Evince while preserving label sharpness.
SAVE_DPI = 300
RASTER_KW = dict(rasterized=True)


def bucket_of(df: pd.DataFrame) -> pd.Series:
    r = np.where(df["round"] > 1, "2+", "1")
    d = np.where(df["n_unique_deals_round"] >= 1, "1+", "0")
    return pd.Series([f"R{a}/D{b}" for a, b in zip(r, d)], index=df.index)


def load_split(sid: int = 0):
    tad = pd.read_feather(PREP / "time_aggregate_dataset.ft").reset_index(drop=True)
    tad = tad[~tad["treatment"].isin(DYNAMIC_TREATMENTS)]
    tad = tad.query("round <= 5 and time <= 120").reset_index(drop=True)
    tad = pd.get_dummies(tad, columns=["feedback_setting", "price_rule"])
    dummies = [c for c in tad.columns
               if c.startswith("feedback_setting_") or c.startswith("price_rule_")]
    quantiles = [c for c in tad.columns if "running_" in c and "change" not in c]
    tts = pd.read_feather(PREP / "train_test_split.ft").query("sample_id == @sid")
    train_ids = tts.loc[tts["dataset_type"] == "train", ["treatment", "game"]]
    test_ids = tts.loc[tts["dataset_type"] == "test", ["treatment", "game"]]
    train = tad.merge(train_ids, on=["treatment", "game"])
    test = tad.merge(test_ids, on=["treatment", "game"])
    return train, test, quantiles, dummies


def fit_gbt(task, train, quantiles, dummies):
    target = "allocative_efficiency_round" if task == "ae" else "ce_round"
    X_q = train[quantiles].values
    X_q_norm, X_median, X_iqr = normalize_feats(X_q, MIN_Q, MAX_Q)
    y = train[target].values[:, np.newaxis]
    y_train = y if task == "ae" else normalize_other(y, X_median, X_iqr)
    X_other = train[dummies + ["n_unique_deals_round", "round"]].values
    X_all = np.concatenate([X_q_norm, X_other], axis=1)

    params = dict(boost_from_average=True, loss_function="Quantile",
                  task_type="CPU", verbose=False)
    params["grow_policy"] = "SymmetricTree" if task == "ae" else "Lossguide"
    reg = cb.CatBoostRegressor(**params)
    grid = GBT_GRID_AE if task == "ae" else GBT_GRID_CEP
    reg.grid_search(grid, X_all, y=y_train, cv=5, partition_random_seed=0,
                    calc_cv_statistics=(task == "ce"),
                    search_by_train_test_split=True, refit=True,
                    shuffle=(task == "ce"),
                    train_size=0.5 if task == "ae" else 0.8,
                    verbose=False, plot=False)

    feature_names = quantiles + dummies + ["n_unique_deals_round", "round"]
    return reg, feature_names


def predict_gbt_raw(reg, X_norm):
    """Return predictions in the model's TRAINING units (raw AE in [0,1]; for CEP
    the value is in normalized-CE units that still need per-row denormalization)."""
    return reg.predict(X_norm)


def assemble_X_test(test, quantiles, dummies):
    X_q_te, X_med_te, X_iqr_te = normalize_feats(test[quantiles].values, MIN_Q, MAX_Q)
    X_other_te = test[dummies + ["n_unique_deals_round", "round"]].values
    X_all_te = np.concatenate([X_q_te, X_other_te], axis=1)
    return X_all_te, X_med_te, X_iqr_te


def fit_ob_rlm_ae(train, quantiles):
    tr = train.copy()
    tr[quantiles], _, _ = normalize_feats(tr[quantiles].values, MIN_Q, MAX_Q)
    formula = ("allocative_efficiency_round ~ ("
               + " + ".join(quantiles)
               + ") * C(feedback_setting_BB) + n_unique_deals_round")
    tr["feedback_setting_BB"] = tr["feedback_setting_BB"].astype(int)
    return smf.rlm(formula, data=tr).fit()


def predict_ob_rlm_ae(model, test, quantiles):
    te = test.copy()
    te[quantiles], _, _ = normalize_feats(te[quantiles].values, MIN_Q, MAX_Q)
    te["feedback_setting_BB"] = te["feedback_setting_BB"].astype(int)
    return np.clip(model.predict(te).to_numpy(), 0.0, 1.0)


def fit_ob_rlm_cep(train, quantiles):
    tr = train.copy()
    tr[quantiles], X_median, X_iqr = normalize_feats(tr[quantiles].values, MIN_Q, MAX_Q)
    y = tr["ce_round"].values[:, np.newaxis]
    tr["ce_round_norm"] = normalize_other(y, X_median, X_iqr).squeeze()
    formula = "ce_round_norm ~ " + " + ".join(quantiles) + " - 1"
    return smf.rlm(formula, data=tr).fit()


def predict_ob_rlm_cep(model, test, quantiles):
    te = test.copy()
    te[quantiles], med_t, iqr_t = normalize_feats(te[quantiles].values, MIN_Q, MAX_Q)
    raw = model.predict(te).to_numpy()
    return denormalize_feats(raw[:, np.newaxis], med_t, iqr_t).squeeze()


def predict_gbt_target_units(reg, task, X_all_te, X_med_te, X_iqr_te):
    raw = reg.predict(X_all_te)
    if task == "ae":
        return np.clip(raw, 0.0, 1.0)
    return denormalize_feats(raw[:, np.newaxis], X_med_te, X_iqr_te).squeeze()


def compute_shap_target_units(reg, X_all_te, task, X_iqr_te):
    """SHAP values converted to raw target units.

    For AE the model trains on raw units, so SHAP is already in [0,1] units.
    For CEP the model trains on normalised CE; multiplying by the per-row IQR
    expresses each contribution in raw price units.
    """
    pool = cb.Pool(X_all_te)
    shap = reg.get_feature_importance(data=pool, type="ShapValues")[:, :-1]
    if task == "ce":
        shap = shap * np.asarray(X_iqr_te).reshape(-1, 1)
    return shap


def compute_pdp_target_units(reg, task, feature_idx, X_all_te, X_med_te, X_iqr_te,
                             n_points: int = 30):
    """Mean prediction in raw target units as feature j is swept across the empirical
    quantile grid (2nd–98th percentile).  All other inputs stay at their joint test
    distribution.  CEP predictions are denormalised per row before averaging."""
    grid = np.quantile(X_all_te[:, feature_idx], np.linspace(0.02, 0.98, n_points))
    ys = np.zeros_like(grid)
    for i, v in enumerate(grid):
        X_mod = X_all_te.copy()
        X_mod[:, feature_idx] = v
        pred = reg.predict(X_mod)
        if task == "ce":
            pred = denormalize_feats(pred[:, np.newaxis], X_med_te, X_iqr_te).squeeze()
        else:
            pred = np.clip(pred, 0.0, 1.0)
        ys[i] = float(np.mean(pred))
    return grid, ys


def non_monotone_swings(ys: np.ndarray) -> int:
    diffs = np.diff(ys)
    return int(np.sum(np.diff(np.sign(diffs)) != 0))


def make_m2_figure(df, task, model_name, out: Path) -> dict:
    resid = df["residual"].values
    actual = df["actual"].values
    pred = df["pred"].values
    buckets = df["bucket"].values

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    ax = axes[0]
    if task == "ce":
        lo = float(np.min(actual))
        hi = float(np.max(actual))
        margin = 0.5 * (hi - lo)
        clip_lo, clip_hi = lo - margin, hi + margin
        n_clip = int(np.sum((pred < clip_lo) | (pred > clip_hi)))
    else:
        clip_lo, clip_hi = 0.0, 1.0
        n_clip = 0
    ax.scatter(actual, np.clip(pred, clip_lo, clip_hi),
               s=3, alpha=0.25, color="#3366cc", **RASTER_KW)
    ax.plot([clip_lo, clip_hi], [clip_lo, clip_hi],
            color="black", lw=0.8, linestyle="--")
    ax.set_xlim(clip_lo, clip_hi)
    ax.set_ylim(clip_lo, clip_hi)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    title = f"Predicted vs actual --- {model_name.upper()} ({task.upper()})"
    if n_clip:
        title += f"\n[{n_clip} of {len(pred)} pred clipped to view]"
    ax.set_title(title, fontsize=9)

    ax = axes[1]
    qlo, qhi = np.quantile(resid, [0.005, 0.995])
    show = resid[(resid >= qlo) & (resid <= qhi)]
    ax.hist(show, bins=60, color="#3366cc", alpha=0.7, density=True, **RASTER_KW)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel(r"Residual $\hat{y}-y$")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution (0.5--99.5\\%)", fontsize=9)

    ax = axes[2]
    data = [resid[buckets == b] for b in BUCKET_ORDER]
    ax.boxplot(data, labels=BUCKET_ORDER, showfliers=False)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Residual")
    ax.set_title("Residuals by (Round, Deals)", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)

    return {"n_clipped_in_scatter": n_clip,
            "scatter_clip_lo": float(clip_lo), "scatter_clip_hi": float(clip_hi)}


def make_r29_figure(reg, feature_names, X_all_te, X_med_te, X_iqr_te,
                    task: str, out: Path, out_shap_ft: Path,
                    out_pdp_ft: Path) -> dict:
    importance = dict(zip(feature_names, reg.feature_importances_.tolist()))

    d5_bid = next((f for f in feature_names if "buyer" in f and f.endswith("_05")), None)
    d5_ask = next((f for f in feature_names if "seller" in f and f.endswith("_05")), None)
    bid_top = max((f for f in feature_names if "buyer" in f),
                  key=lambda f: importance.get(f, 0.0), default="")
    ask_top = max((f for f in feature_names if "seller" in f),
                  key=lambda f: importance.get(f, 0.0), default="")
    pdp_features = [f for f in (d5_bid, bid_top, d5_ask, ask_top) if f]
    pdp_features = list(dict.fromkeys(pdp_features))

    shap_raw = compute_shap_target_units(reg, X_all_te, task, X_iqr_te)
    shap_df = pd.DataFrame(shap_raw, columns=feature_names)
    shap_df.to_feather(out_shap_ft)

    bid_cols = [f for f in feature_names if "buyer" in f]
    ask_cols = [f for f in feature_names if "seller" in f]
    cat_cols = [f for f in feature_names if f not in bid_cols + ask_cols]
    group_phi = {
        "Bid quantiles": float(np.mean(np.sum(np.abs(shap_df[bid_cols].values), axis=1))),
        "Ask quantiles": float(np.mean(np.sum(np.abs(shap_df[ask_cols].values), axis=1))),
        "Categoricals+time": float(np.mean(np.sum(np.abs(shap_df[cat_cols].values), axis=1))),
    }

    pdp_records = []
    pdp_summary = {}
    for fname in pdp_features:
        idx = feature_names.index(fname)
        xs, ys = compute_pdp_target_units(reg, task, idx, X_all_te,
                                          X_med_te, X_iqr_te, n_points=25)
        pdp_records.append(pd.DataFrame({"feature": fname, "x_norm": xs, "y_target": ys}))
        pdp_summary[fname] = {
            "y_min": float(np.min(ys)),
            "y_max": float(np.max(ys)),
            "y_swing": float(np.max(ys) - np.min(ys)),
            "y_grand_mean": float(np.mean(ys)),
            "n_sign_flips": non_monotone_swings(ys),
        }
    pd.concat(pdp_records, ignore_index=True).to_feather(out_pdp_ft)

    n_pdp = len(pdp_features)
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, max(n_pdp, 4), hspace=0.55, wspace=0.45)

    target_label = "AE" if task == "ae" else "CEP"
    target_unit = "" if task == "ae" else "  (price)"
    for i, fname in enumerate(pdp_features):
        ax = fig.add_subplot(gs[0, i])
        sub = pdp_records[i]
        xs = sub["x_norm"].values
        ys = sub["y_target"].values
        ymean = float(np.mean(ys))
        ax.plot(xs, ys - ymean, color="#cc3333", lw=1.4, marker="o", markersize=2)
        ax.axhline(0, color="black", lw=0.4, linestyle=":")
        ax.set_xlabel(fname.replace("running_", "").replace("_", " "), fontsize=8)
        ax.set_ylabel(rf"$E[\hat{{{target_label}}}]-\bar{{E}}${target_unit}", fontsize=8)
        swing = float(np.max(ys) - np.min(ys))
        flips = non_monotone_swings(ys)
        ax.set_title(f"PDP $D_{{{fname.split('_')[-1]}}}$\n"
                     f"swing={swing:.3g}, flips={flips}", fontsize=8)

    ax_bee = fig.add_subplot(gs[1, :2])
    mean_abs = np.abs(shap_df).mean().sort_values(ascending=False)
    top = mean_abs.head(10).index.tolist()
    positions = np.arange(len(top))
    rng = np.random.default_rng(0)
    for k, fname in enumerate(top):
        vals = shap_df[fname].values
        feat_vals = X_all_te[:, feature_names.index(fname)]
        rng_v = max(feat_vals.max() - feat_vals.min(), 1e-9)
        feat_norm = (feat_vals - feat_vals.min()) / rng_v
        jitter = (rng.random(len(vals)) - 0.5) * 0.6
        ax_bee.scatter(vals, np.full_like(vals, positions[k]) + jitter,
                       s=2, alpha=0.35, c=feat_norm, cmap="coolwarm",
                       vmin=0, vmax=1, **RASTER_KW)
    ax_bee.axvline(0, color="black", lw=0.6)
    ax_bee.set_yticks(positions)
    ax_bee.set_yticklabels([f.replace("running_", "").replace("_", " ")
                            for f in top], fontsize=8)
    ax_bee.set_xlabel(rf"SHAP $\phi_j$ ({target_label} units)", fontsize=9)
    ax_bee.set_title("SHAP beeswarm --- top 10 features", fontsize=9)

    ax_grp = fig.add_subplot(gs[1, 2:])
    labels = list(group_phi.keys())
    values = [group_phi[k] for k in labels]
    ax_grp.barh(labels, values, color=["#3366cc", "#cc3333", "#888888"])
    ax_grp.set_xlabel(rf"$\langle\sum_{{j\in g}}|\phi_j|\rangle$ ({target_label} units)",
                      fontsize=9)
    ax_grp.set_title("Grouped $|\\phi|$ (supply/demand asymmetry)", fontsize=9)

    fig.suptitle(f"GBT feature response --- {target_label}", y=1.0, fontsize=11)
    fig.savefig(out, bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)

    return {"pdp_features": pdp_features, "group_phi": group_phi,
            "pdp_summary": pdp_summary,
            "top_importance": {k: float(v) for k, v in
                               sorted(importance.items(), key=lambda kv: -kv[1])[:15]},
            "top_mean_abs_shap": {k: float(v) for k, v in
                                  mean_abs.head(15).to_dict().items()}}


def summarise_residuals(df) -> dict:
    out = {}
    for b in BUCKET_ORDER:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            continue
        r = sub["residual"].values
        a = sub["actual"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(np.abs(a) > 1e-9, np.abs(r / a), np.nan)
        out[b] = {
            "n": int(len(sub)),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "mae": float(np.mean(np.abs(r))),
            "bias": float(np.mean(r)),
            "median_abs_pct_err": float(np.nanmedian(ape)) if not np.all(np.isnan(ape)) else None,
            "p99_abs_resid": float(np.quantile(np.abs(r), 0.99)),
        }
    return out


def run_task(task, train, test, quantiles, dummies) -> dict:
    summary: dict = {"task": task}
    target = "allocative_efficiency_round" if task == "ae" else "ce_round"

    print(f"[py] {task}: fit GBT")
    reg, feature_names = fit_gbt(task, train, quantiles, dummies)
    X_all_te, X_med_te, X_iqr_te = assemble_X_test(test, quantiles, dummies)
    pred_gbt = predict_gbt_target_units(reg, task, X_all_te, X_med_te, X_iqr_te)
    actual = test[target].values
    df_gbt = test[["treatment", "game", "round", "time", "n_unique_deals_round"]].copy()
    df_gbt["actual"] = actual
    df_gbt["pred"] = pred_gbt
    df_gbt["residual"] = pred_gbt - actual
    df_gbt["bucket"] = bucket_of(test)
    df_gbt.to_feather(OUT_DATA / f"predictions_{task}_gbt.ft")

    summary["gbt_buckets"] = summarise_residuals(df_gbt)
    summary["gbt_m2"] = make_m2_figure(df_gbt, task, "gbt",
                                        OUT_FIG / f"m2_{task}_gbt.pdf")

    print(f"[py] {task}: SHAP + PDPs")
    summary["gbt_r29"] = make_r29_figure(reg, feature_names, X_all_te,
                                         X_med_te, X_iqr_te, task,
                                         OUT_FIG / f"r29_{task}_gbt.pdf",
                                         OUT_DATA / f"shap_{task}_gbt.ft",
                                         OUT_DATA / f"pdp_{task}_gbt.ft")
    (OUT_DATA / f"importance_{task}_gbt.json").write_text(json.dumps(
        {k: float(v) for k, v in zip(feature_names, reg.feature_importances_.tolist())},
        indent=2))

    print(f"[py] {task}: fit OB-RLM")
    if task == "ae":
        rlm = fit_ob_rlm_ae(train, quantiles)
        pred_rlm = predict_ob_rlm_ae(rlm, test, quantiles)
    else:
        rlm = fit_ob_rlm_cep(train, quantiles)
        pred_rlm = predict_ob_rlm_cep(rlm, test, quantiles)
    df_rlm = df_gbt[["treatment", "game", "round", "time",
                     "n_unique_deals_round", "actual"]].copy()
    df_rlm["pred"] = pred_rlm
    df_rlm["residual"] = pred_rlm - actual
    df_rlm["bucket"] = df_gbt["bucket"].values
    df_rlm.to_feather(OUT_DATA / f"predictions_{task}_ob_rlm.ft")
    summary["ob_rlm_buckets"] = summarise_residuals(df_rlm)
    summary["ob_rlm_m2"] = make_m2_figure(df_rlm, task, "ob_rlm",
                                           OUT_FIG / f"m2_{task}_ob_rlm.pdf")

    return summary


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    train, test, quantiles, dummies = load_split(sid=0)
    print(f"[py] loaded split 0: train={len(train)} test={len(test)}"
          f" quantiles={len(quantiles)} dummies={len(dummies)}")
    summary = {}
    for task in ("ae", "ce"):
        summary[task] = run_task(task, train, test, quantiles, dummies)
    (OUT_DATA / "summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: str(o)))
    print(f"[py] wrote {OUT_DATA / 'summary.json'}")


if __name__ == "__main__":
    main()
