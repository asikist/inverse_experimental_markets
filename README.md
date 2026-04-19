# Inverse Experimental Framework — reproduction guide

This folder contains the code, cached intermediates, and result tables that
back the EPJ Data Science submission *"An 'Inverse' Experimental Framework
to Estimate Market Efficiency"* and its R1 / R2 revision response. Running
the scripts below against the included data reproduces every numeric table
and figure that appears in the manuscript and in `paper/epj/response.tex`.

The pipeline has three layers:

1. **Data preparation** — numbered scripts (`1.…` through `3.5.…`) that turn
   the raw Ikica CSV into the cached feather and torch artefacts every model
   reads.
2. **Analyses** — five model evaluations: leave-one-treatment-out
   cross-validation, the orderbook-only and no-deal-price ablations, the
   CEP baselines appendix, the session-clustered Wilcoxon tests, and the
   external-dataset evaluation on Lin et al. 2020.
3. **Reporting** — scripts that read the result `.ft` files, write the LaTeX
   tables under `paper/epj/reproducibility_report/final_tables/`, and copy
   PDFs into `paper/epj/figures/`.

The main-text bar charts and the four model-fit notebooks live in
`notebooks/experiments/{allocative_efficiency,price}/`. The cached `.ft`
outputs in `data/results/{ce_price,allocative_efficiency}/` are the inputs
that the reporting scripts consume, so you do not need to re-run the
notebooks to produce the appendix tables. They remain available in the
repo so the line-numbered references in
`paper/epj/reproducibility_report/README.md` continue to resolve, and so
the main-paper results can be regenerated end-to-end.

## Data

Two datasets are needed.

### Ikica (Ikica, Nax & Pradelski 2023)

Already in the repo: `data/Data.csv` (the raw subject-level CSV from the
Ikica paper). After running the data-preparation scripts the following
caches are written under `data/preprocessed/`:

- `original_df.ft` — long-format dataframe of bids, asks, valuations,
  deal prices, indexed by `(treatment, game, round, time, side, id)`.
- `time_aggregate_dataset.ft` — per-snapshot feature matrix used by every
  model: running orderbook quantiles (`running_buyer_bid_quant_*`,
  `running_seller_bid_quant_*`), categorical labels
  (`treatment`, `feedback_setting`, `price_rule`), targets
  (`ce_round`, `allocative_efficiency_round`).
- `train_test_split.ft` — the 50 stratified train/test splits used across
  the paper. Each row carries `sample_id`, `dataset_type`, `treatment`,
  `game`. All evaluations re-merge against this file rather than re-drawing
  splits, so the per-split results are bit-stable across reruns.
- `supply_demand_lines_round.ft` — fitted linear supply / demand
  intercepts and slopes per round, used by the CEMH and EMH baselines.
- `torch/raw/*.pt` — running bid / ask / valuation tensors for the torch
  models. Optional; only needed to re-run the deep-learning notebooks.

### Lin et al. (2020)

Bundled in `new_data/Experimental-double-auctions-main/` (a snapshot of
Lin's public replication archive). Files used:

- `Data/Lin_marketProfile_processed.csv` — per-(MarketID, period) targets:
  `eq_price` (the competitive-equilibrium price the script names
  `ce_round`), period boundaries.
- `Data/Lin_orderBook_processed.csv` — order-by-order book log, converted
  to running quantiles by `lin_eval.py` to mirror the Ikica feature
  schema.

No download is required. If you need to re-fetch from the source, the
upstream archive is the supplementary material of Lin, Palfrey, Plott
(2020) at the `Experimental-double-auctions-main` repository linked from
their paper.

## Environment

The project pins exact dependency versions in `pyproject.toml` and
`uv.lock`. Install with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra torch              # required: base + torch (CPU)
uv sync --extra torch --extra jupyter   # add jupyterlab if re-running the model-fit notebooks
uv sync --extra torch-cu121 \      # optional: CUDA 12.1 torch instead of CPU torch
        --index-url https://download.pytorch.org/whl/cu121
uv sync --extra dev                # optional: pytest, black, nbstripout
```

The `torch` extra is **required** for the reproduction pipeline, not
optional: `scripts/2.generate_time_aggregate_dataset.py` calls
`torch.save` to write the raw tensors, `scripts/3.generate_supply_demand_line_data.py`
uses `torch.linalg.lstsq`, and `scripts/3.{,5.}generate_tensor_datasets.py`
serialise the canonical split as torch tensors. The analysis scripts
(LOTO, ablations, CEP baselines, Wilcoxon, Lin eval) themselves do not
import torch, but they depend on the feather caches produced by those
numbered scripts, so torch must be installed to regenerate the caches
from `Data.csv`. If the caches in `data/preprocessed/` are already
present, you can run the analyses from the base env.

The `jupyter` extra is only needed to re-execute the model-fit notebooks
under `notebooks/experiments/`. Their cached `.ft` outputs in
`data/results/{ce_price,allocative_efficiency}/` are the inputs for the
paper-table scripts, so re-running them is not on the critical
reproduction path.

Python: `>=3.10, <3.12`.

### R for the clustered Wilcoxon step

`scripts/wilcoxon_clustered.py` shells out to `Rscript` with the
`clusrank` package for the Rosner–Glynn–Lee clustered signed-rank test.
Install R ≥ 4.0 from your distribution, then install `clusrank` into the
user library:

```bash
R -e 'user_lib <- Sys.getenv("R_LIBS_USER"); dir.create(user_lib, recursive=TRUE, showWarnings=FALSE); install.packages("clusrank", lib=user_lib, repos="https://cloud.r-project.org")'
```

`scripts/wilcoxon_clustered.py` passes `R_LIBS_USER` through to the
subprocess so a user-library install is picked up without further
configuration.

### Jupyter config conflict (optional workaround)

If your shell has a global Jupyter config that loads
`jupyter_contrib_nbextensions` (or another legacy notebook extension),
`nbconvert --execute` will fail at import time. Bypass the global config
by setting `JUPYTER_CONFIG_DIR` to an empty directory for the notebook
run:

```bash
mkdir -p /tmp/empty-jupyter
JUPYTER_CONFIG_DIR=/tmp/empty-jupyter uv run jupyter nbconvert --to notebook --execute <nb>.ipynb --output /tmp/out.ipynb
```

This is only relevant when re-running the notebooks, not the numbered
scripts or the analysis scripts.

## Pipeline execution order

If the cached `.ft` files in `data/preprocessed/` are already present (they
ship with the repo), you can skip directly to the analysis section.
Re-running the data preparation from `Data.csv` takes ~10–20 minutes total.

### Working-directory conventions

The scripts split into two groups with different path-resolution
behaviour. Follow the conventions below when invoking them manually:

- **Numbered data-prep scripts** (`1.preprocess_original_dataset.py`,
  `2.generate_time_aggregate_dataset.py`,
  `3.generate_supply_demand_line_data.py`,
  `3.generate_tensor_datasets.py`,
  `3.5.generate_tensor_datasets.py`) use **relative paths** like
  `'../data/Data.csv'`. Run them from `code/scripts/`.
- **Analysis and report scripts** (`loto_r24.py`, `ablation_r25.py`,
  `ablation_r26.py`, `cep_baselines.py`, `cep_baselines_report.py`,
  `wilcoxon_clustered.py`, `wilcoxon_clustered_tables.py`, `lin_eval.py`,
  `lin_cep_diagnostic.py`, `paper_artifacts.py`,
  `regenerate_bar_charts.py`, `ablation_report.py`, `diagnostics.py`,
  `voronoi_composite.py`) resolve paths via
  `Path(__file__).resolve().parent.parent` and are
  location-independent. You can invoke them from `code/` or
  `code/scripts/`.
- **Notebooks** (`notebooks/experiments/{allocative_efficiency,price}/*.ipynb`)
  use `'../../../data/...'` paths. Execute them with their own directory
  as cwd — e.g., via
  `cd code/notebooks/experiments/allocative_efficiency && uv run jupyter nbconvert --execute ...`.

For brevity, the commands below are shown with
`uv run python scripts/<x>.py` and assume you are one level up. If you
are inside `code/scripts/`, drop the `scripts/` prefix. Both forms work
for the analysis scripts; only the second form works for the numbered
scripts.

### Step 1 — preprocess the raw Ikica CSV

```bash
cd code/scripts
uv run python 1.preprocess_original_dataset.py
```

- Reads: `data/Data.csv`
- Writes: `data/preprocessed/original_df.ft`
- Cost: ~30 s.

Casts dtypes, attaches treatment / feedback / price-rule labels, drops
malformed rows.

### Step 2 — build the per-snapshot feature matrix

```bash
uv run python 2.generate_time_aggregate_dataset.py    # from code/scripts/
```

- Reads: `data/preprocessed/original_df.ft`
- Writes:
  - `data/preprocessed/time_aggregate_dataset.ft`
  - `data/preprocessed/torch/raw/{deal_prices,running_*}.pt`
- Cost: ~3–5 min. Requires `--extra torch` (uses `torch.save`).

This is the single most expensive step. Produces both the pandas feature
matrix consumed by every model in the analysis section and the torch
tensors consumed by the optional deep-learning notebooks.

### Step 3 — fit per-round supply / demand lines

```bash
uv run python 3.generate_supply_demand_line_data.py    # from code/scripts/
```

- Reads: `data/preprocessed/original_df.ft`
- Writes:
  - `data/preprocessed/supply_demand_lines_round.ft`
  - `data/preprocessed/supply_demand_lines_game.ft`
- Cost: ~30 s. Requires `--extra torch` (uses `torch.linalg.lstsq`).

Per-round OLS of valuations against rank gives the linear supply and
demand intercepts and slopes, plus the analytic `linear_eq` and
`linear_cep` quantities used by the EMH and CEMH baselines.

### Step 4 — generate the canonical train/test split

```bash
uv run python 3.5.generate_tensor_datasets.py     # from code/scripts/; canonical split (50 samples, ratio 0.5)
# 3.generate_tensor_datasets.py is an alternative entry point with
# the same configuration; either works.
```

- Reads: `data/preprocessed/{time_aggregate_dataset,original_df}.ft` and
  `data/preprocessed/torch/raw/*.pt`
- Writes:
  - `data/preprocessed/train_test_split.ft`
  - `data/preprocessed/torch/<dataset_name>/*` (only used by the deep models)
- Cost: ~2 min. Requires `--extra torch`.

The split file is what every analysis script merges against to define
`train` and `test` rows. Static-treatment filtering and the 5-round /
120-second cap are applied here.

## Analyses

Each script below is self-contained, reads only from `data/preprocessed/`
and (for the report scripts) `data/results/`, and writes to a fixed
sub-directory of `data/results/`. Re-runs are idempotent.

### Leave-one-treatment-out cross-validation

```bash
uv run python scripts/loto_r24.py
```

- Reads: `data/preprocessed/{time_aggregate_dataset,supply_demand_lines_round}.ft`
- Writes:
  - `data/results/loto_r24/{ob_rlm_ae,cemh_global_cep,gbt_cep,gbt_ae}.ft`
  - `data/results/loto_r24/summary.json`
- Cost: ~15–25 min on a single CPU (16 folds × 4 models, CatBoost grid).

Holds out one experimental treatment at a time and refits the four
orderbook-only models on the remaining 15 treatments. Result: per-fold
median APE per `(round, n_deals)` bucket, plus the aggregate LOTO numbers
reported in the appendix.

### Orderbook-only ablation

```bash
uv run python scripts/ablation_r25.py
```

- Reads: `data/preprocessed/{time_aggregate_dataset,train_test_split}.ft`
- Writes:
  - `data/results/ablations/r25/{ob_rlm_ae,cemh_global_cep,gbt_cep,gbt_ae}.ft`
  - `data/results/ablations/r25/summary.json`
- Cost: ~30–60 min (50 splits × 4 models).

Drops every non-orderbook input (treatment / feedback / price-rule
categoricals, realised deal price, `n_deals`, round) and refits.

### No-deal-price ablation

```bash
uv run python scripts/ablation_r26.py
```

- Reads: `data/preprocessed/{time_aggregate_dataset,train_test_split}.ft`
- Writes:
  - `data/results/ablations/r26/ob_rlm_ae.ft`
  - `data/results/ablations/r26/summary.json`
- Cost: ~5 min.

Refits OB-RLM (AE) without the realised deal price. Only OB-RLM (AE) is
refit because the other three predictors that use the realised price either
collapse to a previously-reported baseline (CEMH for CEP) or have no
non-trivial variant without it (EMH for CEP).

### CEP baselines

```bash
uv run python scripts/cep_baselines.py            # produces the two baseline .ft files
uv run python scripts/cep_baselines_report.py     # writes diagnostics + LaTeX
```

- Reads (baselines): `data/preprocessed/time_aggregate_dataset.ft`,
  `data/preprocessed/train_test_split.ft`,
  `data/results/ce_price/emh.ft` (used as the per-row evaluation grid).
- Writes (baselines):
  - `data/results/ce_price/{treatment_mean_ce,book_midpoint}.ft`
- Reads (report): the four paper model results plus the two baseline files.
- Writes (report):
  - `data/results/ce_price/baseline_diagnostics/{trainset_vs_testset_ce_gap.csv,loto_median_ape.csv,summary.json}`
  - `paper/epj/reproducibility_report/final_tables/tab_cep_baseline_comparison.tex`
  - `paper/epj/reproducibility_report/final_tables/tab_cep_baseline_brittleness.tex`
- Cost: ~3 min total.

### Session-clustered Wilcoxon

```bash
uv run python scripts/wilcoxon_clustered.py            # ~90 s; needs Rscript + clusrank
uv run python scripts/wilcoxon_clustered_tables.py     # ~5 s; pure Python
```

- Reads: `data/results/{ce_price,allocative_efficiency}/*.ft`
- Writes:
  - `data/results/wilcoxon_clustered/{ae,ce}_results.ft`
  - `data/results/wilcoxon_clustered/diagnostics.json`
  - `paper/epj/reproducibility_report/final_tables/tab_{ae,cep}_ape_wilcoxon_clustered.tex`

Two test variants per pair, per bucket: Rosner-Glynn-Lee clustered
signed-rank (via the R `clusrank` package; install with
`R -e 'install.packages("clusrank")'`) and a per-cluster median +
Wilcoxon. Holm-Bonferroni adjustment is applied within (task, variant).

### Lin external evaluation

```bash
uv run python scripts/lin_eval.py                # full external eval
uv run python scripts/lin_cep_diagnostic.py      # spread-driven cold-start diagnostic
```

- Reads (eval): `data/preprocessed/{time_aggregate_dataset,supply_demand_lines_round}.ft`,
  `new_data/Experimental-double-auctions-main/Data/Lin_*.csv`
- Writes (eval):
  - `data/results/lin_eval/lin_features.ft`
  - `data/results/lin_eval/{ob_rlm_ae,cemh_global_cep,gbt_cep,gbt_ae}.ft`
  - `data/results/lin_eval/summary.json`
- Reads (diagnostic): `data/results/lin_eval/{lin_features,gbt_cep}.ft` and
  the Ikica feature matrix.
- Writes (diagnostic):
  - `data/results/lin_eval/diagnostic/{spread_summary.json,ape_by_spread.csv}`
  - `paper/epj/reproducibility_report/final_tables/tab_lin_cep_spread.tex`
  - `paper/epj/figures/diagnostics/lin_cs_spread_diagnostic.pdf`
- Cost: ~20 min (eval) + ~30 s (diagnostic).

Refits the four orderbook-only models on the full Ikica set (no train/test
split, no 50-CV ensemble) and applies them to Lin's order book. The
diagnostic script then explains the GBT (CEP) cold-start failure as a
spread-distribution mismatch between the two datasets.

## Reporting and paper artefacts

These scripts read result `.ft` files and emit the actual files included in
the manuscript LaTeX. Run them after the corresponding analysis above.

### Main paper tables and bar charts

```bash
uv run python scripts/paper_artifacts.py
uv run python scripts/regenerate_bar_charts.py
```

`paper_artifacts.py` produces:

- `paper/epj/reproducibility_report/final_tables/tab_{ae,cep}_ape.tex` —
  the four-bucket median APE tables (paper Tabs.\\ 3 and 5).
- `paper/epj/reproducibility_report/final_tables/tab_{ae,cep}_ape_wilcoxon.tex`
  — the unclustered pairwise Wilcoxon tables.
- `paper/epj/reproducibility_report/latex_staging/{ae,ce}/` — the raw
  `.to_latex()` tables and pickle-cached pivots, kept for line-by-line
  audit against the original notebook outputs.
- `paper/epj/reproducibility_report/ranking_preservation.md` — diff of the
  best-per-cell winner against the original submission, used to argue that
  reranking is below the rounding threshold.

`regenerate_bar_charts.py` reproduces the round-1 / no-deal and round-1 /
deal bar charts (`paper/epj/figures/{ae_results,price_results}/`) without
re-running the original `result_analysis.ipynb` notebooks.

### Ablation report

```bash
uv run python scripts/ablation_report.py
```

Produces `tab_ablation_r25.tex` and `tab_ablation_r26.tex` in
`paper/epj/reproducibility_report/final_tables/`.

### Diagnostics, Voronoi, deal-counts

```bash
uv run python scripts/diagnostics.py             # PDP, SHAP, residual + predicted-vs-actual panels
uv run python scripts/voronoi_composite.py       # four CEP / AE Voronoi composite PDFs
```

`diagnostics.py` writes per-task figures to `paper/epj/figures/diagnostics/`
(predicted-vs-actual scatter, residual histogram + boxplot, partial
dependence plots, SHAP beeswarms, grouped |SHAP| bar charts).
`voronoi_composite.py` writes the four `voronoi_composite_*.pdf` files used
in the Voronoi appendix figure.

## Mapping between scripts and paper artefacts

| Analysis                                    | Source script(s)                                         | Result artefact(s)                                                              |
|---------------------------------------------|----------------------------------------------------------|---------------------------------------------------------------------------------|
| Tab. 3 (CEP main)                           | `paper_artifacts.py`                                     | `final_tables/tab_cep_ape.tex`                                                  |
| Tab. 5 (AE main)                            | `paper_artifacts.py`                                     | `final_tables/tab_ae_ape.tex`                                                   |
| Round-1 bar charts (Figs. 5, 7)             | `regenerate_bar_charts.py`                               | `paper/epj/figures/{ae_results,price_results}/bar_round1_{deal,nodeal}.pdf`     |
| Voronoi appendix figure                     | `voronoi_composite.py`                                   | `paper/epj/figures/voronoi/voronoi_composite_*.pdf`                             |
| Model diagnostics (PDP, SHAP, residuals)    | `diagnostics.py`                                         | `paper/epj/figures/diagnostics/*.pdf`                                           |
| Leave-one-treatment-out cross-validation    | `loto_r24.py`                                            | `data/results/loto_r24/`, appendix LOTO table                                   |
| Lin external evaluation                     | `lin_eval.py` + `lin_cep_diagnostic.py`                  | `data/results/lin_eval/`, `final_tables/tab_lin_cep_spread.tex`,                |
|                                             |                                                          | `paper/epj/figures/diagnostics/lin_cs_spread_diagnostic.pdf`                    |
| Orderbook-only ablation                     | `ablation_r25.py` → `ablation_report.py`                 | `final_tables/tab_ablation_r25.tex`                                             |
| No-deal-price ablation                      | `ablation_r26.py` → `ablation_report.py`                 | `final_tables/tab_ablation_r26.tex`                                             |
| Session-clustered Wilcoxon                  | `wilcoxon_clustered.py` → `wilcoxon_clustered_tables.py` | `final_tables/tab_{ae,cep}_ape_wilcoxon_clustered.tex`                          |
| CEP baselines                               | `cep_baselines.py` → `cep_baselines_report.py`           | `final_tables/tab_cep_baseline_{comparison,brittleness}.tex`                    |

All paths above are relative to the repository root.

## Notebooks under `notebooks/experiments/`

The two sub-folders (`allocative_efficiency/`, `price/`) contain the
notebooks that produced the original submission's main results: one
notebook per model family (`emh.ipynb`, `cemh.ipynb`, `ob-rlm.ipynb`,
`gradient_boosting.ipynb`) under each task, plus `result_analysis.ipynb`
for aggregation and bar charts.

The per-model `.ft` outputs of these notebooks are treated as fixed
inputs to `paper_artifacts.py`, `regenerate_bar_charts.py` and
`ablation_report.py`. You do **not** need to re-run the notebooks to
reproduce the appendix tables — the cached results in
`data/results/{ce_price,allocative_efficiency}/*.ft` are sufficient.

If you do want to re-run them (e.g.\\ to verify the cached results), the
Jupyter kernel needs the `--extra torch --extra jupyter` extras. The
notebooks read from `data/preprocessed/` and write to
`data/results/{ce_price,allocative_efficiency}/`. Two notebooks are
explicitly cited in `paper/epj/reproducibility_report/README.md` because
the AE-vs-CEP split sensitivity discussion turns on their internals:

- `notebooks/experiments/allocative_efficiency/cemh.ipynb`
- `notebooks/experiments/price/cemh.ipynb`

## Where things live

```
code/
├── data/
│   ├── Data.csv                     # raw Ikica subject-level data
│   ├── preprocessed/                # caches written by 1./2./3./3.5. scripts
│   └── results/
│       ├── allocative_efficiency/   # main AE model .ft files (notebooks)
│       ├── ce_price/                # main CEP model .ft files (notebooks) + treatment-mean / book-midpoint baselines
│       ├── ablations/{r25,r26}/     # orderbook-only + no-deal-price ablation outputs
│       ├── loto_r24/                # leave-one-treatment-out CV outputs
│       ├── lin_eval/                # Lin external-eval outputs + diagnostic/
│       └── wilcoxon_clustered/      # session-clustered Wilcoxon outputs
├── new_data/
│   └── Experimental-double-auctions-main/
│       ├── Data/Lin_{marketProfile,orderBook}_processed.csv   # only files lin_eval.py reads
│       ├── LICENSE
│       └── README.md
├── data_helpers/                    # shared preprocessors, scalers, training utilities
├── models/
│   ├── sim/operators.py             # competitive-equilibrium matching engine
│   └── utilities.py                 # minimal stub (`dw`/`DummyWriter`) used by the GBT notebooks
├── notebooks/experiments/           # per-model fit notebooks (frozen for the revision)
├── scripts/                         # numbered data-prep + analysis + reporting scripts
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

Anything not on the critical reproduction path (deep-learning experiments,
Mona-Lisa fits, third-party PDFs, the streamlit demo, archival notebooks,
the full original `models/utilities.py` with its unused torch helpers, and
the non-`Lin_*_processed.csv` files from Lin's replication archive) has
been moved to the sibling `extra_code/` folder so the submission artefact
is minimal. In particular, the reference copies against which this
reproduction was verified live under `extra_code/data/{preprocessed,results}/`.

### Shipped data vs.\ regenerated artefacts

All cached intermediates under `data/preprocessed/` and result feathers under
`data/results/` are checked in (each file is below 50 MB), so a reviewer can
clone the repository and run any of the analysis or reporting scripts
directly without first re-executing the numbered preparation pipeline.

One artefact is intentionally **not** tracked because it exceeds the size
threshold and is fully regenerable from the smaller tracked tensors:

- `data/preprocessed/torch/original_dataset/tensor_dataset.safetensors`
  (≈289 MB) — rebuilt by

  ```bash
  cd code/scripts && uv run python 3.5.generate_tensor_datasets.py
  ```

  using the `.pt` tensors under `data/preprocessed/torch/raw/` (those are
  tracked).

The third-party Springer supplementary PDF
(`new_data/41562_2020_916_MOESM1_ESM.pdf`) is also excluded for copyright
reasons and is not needed by any script.

## Container

A reference container is provided:

```bash
docker compose -f docker-compose.yaml up --build
```

The image installs the base + `torch` + `jupyter` extras and exposes a
JupyterLab on `:8888`. Use the host machine for the heavy CatBoost grids
(LOTO and the orderbook-only ablation) — they parallelise over CPU cores
and are slower in the container.

## Reproducibility checklist

The following sequence was executed from a clean state (only the raw
Ikica `Data.csv` and the two Lin CSVs present; `data/preprocessed/` and
`data/results/` empty) and every output was verified bitwise-identical
to the reference copies under `extra_code/data/{preprocessed,results}/`
(via `pd.read_feather` + `numpy.assert_array_equal` on the numeric
columns and shape/column-set equality on the categoricals):

```bash
# --- environment ------------------------------------------------------
cd code
uv sync --extra torch
R -e 'install.packages("clusrank", lib=Sys.getenv("R_LIBS_USER"), repos="https://cloud.r-project.org")'

# --- data preparation (from code/scripts/) ---------------------------
cd scripts
uv run python 1.preprocess_original_dataset.py
uv run python 2.generate_time_aggregate_dataset.py
uv run python 3.generate_supply_demand_line_data.py
uv run python 3.5.generate_tensor_datasets.py

# --- analyses (still in code/scripts/, order doesn't matter except that
#     cep_baselines_report reads ce_price/emh.ft, which must exist from
#     either the notebooks or a prior run) -----------------------------
uv run python loto_r24.py                   # leave-one-treatment-out CV
uv run python ablation_r25.py               # orderbook-only ablation
uv run python ablation_r26.py               # no-deal-price ablation
uv run python cep_baselines.py              # CEP baselines (treatment-mean, book-midpoint)
uv run python cep_baselines_report.py       # CEP-baseline LaTeX tables
uv run python wilcoxon_clustered.py         # session-clustered Wilcoxon (needs R + clusrank)
uv run python wilcoxon_clustered_tables.py  # clustered-Wilcoxon LaTeX tables
uv run python lin_eval.py                   # Lin external evaluation
uv run python lin_cep_diagnostic.py         # Lin spread-driven diagnostic

# --- paper artefacts ------------------------------------------------
uv run python paper_artifacts.py            # Tabs. 3/5 + unclustered pairwise Wilcoxon
uv run python regenerate_bar_charts.py      # round-1 bar charts (Figs. 5, 7)
uv run python ablation_report.py            # tab_ablation_r{25,26}.tex
uv run python diagnostics.py                # PDP, SHAP, residual + predicted-vs-actual panels
uv run python voronoi_composite.py          # Voronoi appendix figure
```

Cost on a single machine, no GPU: ~60–90 min for data prep, ~60–90 min
for all analyses (LOTO and the orderbook-only ablation dominate; both run
50-split × 4-model CatBoost grids).

Re-running the model-fit notebooks is **not required** to reproduce the
appendix tables — their outputs in
`data/results/{ce_price,allocative_efficiency}/` are shipped with the
repo and consumed as inputs by `paper_artifacts.py`. If you want to
regenerate them from scratch:

```bash
cd code
uv sync --extra torch --extra jupyter
mkdir -p data/results/allocative_efficiency data/results/ce_price
mkdir -p /tmp/empty-jupyter    # see "Jupyter config conflict" above
for task in allocative_efficiency price; do
  cd code/notebooks/experiments/$task
  for nb in emh cemh ob-rlm gradient_boosting; do
    JUPYTER_CONFIG_DIR=/tmp/empty-jupyter uv run jupyter nbconvert \
      --to notebook --execute $nb.ipynb --output /tmp/${task}_${nb}_executed.ipynb
  done
done
```

Each analysis script's headline numbers are cross-checked against the
LaTeX tables under `paper/epj/reproducibility_report/final_tables/`
(e.g., `tab_cep_baseline_comparison.tex` for the CEP baselines and
`tab_cep_baseline_brittleness.tex` for the LOTO brittleness row that
ties the CEP baselines back to the leave-one-treatment-out CV).
