# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CLEAR (*Clearance Learning & Equity Assessment on gRaphs*) predicts whether a US
homicide case gets solved (`Crime Solved`), using the Murder Accountability
Project / Kaggle "Homicide Reports, 1980–2014" dataset (~638k rows). The project
is structured as one question — "what determines clearance?" — pursued in three
stages: **predict** (flat baseline now, GNN later) → **diagnose** (do clearance
rates/errors differ by victim race/sex?) → **prescribe** (how much can mitigation
techniques close that gap, and at what accuracy cost?).

Written docs with the full design rationale (Korean), in `reports/`:
`문제정의서_CLEAR.md` (problem definition), `AI모델_개발계획서_살인사건검거_GNN.md`
(dev plan — graph/edge design, GNN architecture, 4-week roadmap),
`EDA_보고서.md` (EDA findings).

## Commands

```bash
pip install -r requirements.txt
cd src
python 01_clean.py            # raw CSV -> data/processed/clean.parquet
python 02_sample.py           # -> data/processed/sample.parquet
python 03_features.py         # -> data/processed/features.parquet
python eda.py                 # -> outputs/eda_*.csv, outputs/eda_*.png
python 05_train_baseline.py   # -> outputs/baseline_metrics.csv, outputs/baseline_xgb_top_features.csv
python 04_build_graph.py            # -> data/processed/graph/edges_{geo,temporal,weapon}_k{k}.npy (k=config.K_NEIGHBORS)
python 04_build_graph.py --k 20     # rebuild at a different degree cap (needed before --k_neighbors 20 below works)
python 06_train_gnn.py              # -> outputs/gnn_metrics.csv (appends; defaults to config.GNN_DEFAULT_EDGE_TYPE="geo" only)
python 06_train_gnn.py --edge_type all                                   # train geo+temporal+weapon for comparison
python 06_train_gnn.py --edge_type geo --hidden_dim 128 --tag my_sweep   # any hyperparam overridable via CLI
python 06_train_gnn.py --edge_type geo --k_neighbors 20                  # use a --k 20 graph built above
python 06_train_gnn.py --edge_type geo_temporal                          # union of geo+temporal edges
python ablation_sweep.py            # one-factor-at-a-time hyperparam sweep around config.GNN_* defaults
```

Scripts must run from `src/` and in this order — each stage reads the parquet
the previous stage wrote (see `config.py` paths). `04_build_graph.py` only
needs `02`/`03`'s output, not `05`; it's listed last above because it and
`06_train_gnn.py` were built after `05` despite the lower number (see
numbering-gap note below). There is no test suite; there is no build/lint
step configured.

## Architecture

**Shared config**: every script does `import config as C`. `src/config.py` is
the single source of truth for paths, target encoding, column lists, and
hyperparameters (random state, split ratios, CV folds). When changing any of
these, edit `config.py` rather than a script.

**Pipeline is a chain of scripts, not a package**: each `NN_*.py` file is a
standalone stage that reads a parquet (or `.npy`) from `data/processed/`,
transforms it, and writes the next artifact. No script imports another
numbered script — a leading digit isn't a valid Python identifier, so
`import 05_train_baseline` simply doesn't work — everything shares state only
through `config.py` and files on disk. `05_train_baseline.py` was built
before `04_build_graph.py`/`06_train_gnn.py` despite the lower number: it's
the pre-graph performance floor (flat XGBoost/LogReg) the GNN has to beat, so
it needed to exist first. `07_fairness.py` (fairness diagnosis, week 3) is
still unimplemented — that's the one remaining numbering gap.

```
dataset/kaggle_homicide_Reports_1980_2014.csv  (not in git, ~638k rows)
  -> 01_clean.py    target-encode; drop leakage + low-info columns; clean age
  -> 02_sample.py   filter to config.SAMPLE_STATES (default: California+Texas+Michigan)
  -> 03_features.py 5-yr age bins, decade bins, full one-hot; splits off
                     sens__Victim Race / sens__Victim Sex for later fairness work
  -> eda.py / 05_train_baseline.py       (read features.parquet or sample.parquet)
  -> 04_build_graph.py                   (reads sample.parquet + features.parquet ->
                                           data/processed/graph/edges_{geo,temporal,weapon}.npy)
  -> 06_train_gnn.py                     (reads features.parquet + edges_*.npy)
```

**Graph construction (`04_build_graph.py`) uses blocking, not literal
nearest-neighbor search.** The dev-plan doc describes edges as "top-k by
feature similarity," but within any of the 3 candidate blocking keys (same
State+City / same Year+Month / same Weapon+Sex+Race) there's no finer
similarity signal to rank by — `City` is already the finest geography
available (`Agency Code`/`Agency Name` are dropped in `01_clean.py` and never
come back). So each candidate blocks rows by exact key match, then caps
degree at `config.K_NEIGHBORS` via a deterministic shuffle-ring construction
(`zlib.crc32`-seeded per block, not Python's `hash()`, which isn't stable
across process runs) to avoid an O(n²) blowup on huge blocks (e.g. LA alone is
~44k rows). Edge arrays are stored pre-symmetrized (both `(i,j)` and `(j,i)`
present) — already the format PyG's `edge_index` wants directly.

**`06_train_gnn.py` reproduces `05_train_baseline.py`'s exact test split**
by calling `train_test_split` with the same `n`/`stratify`/`random_state`
(row assignment depends only on those three things, not on what's in `X`),
so GraphSAGE's test metrics are directly comparable to the XGBoost baseline's.
It carves an additional validation slice out of the training portion only,
for early stopping on validation balanced accuracy (not validation loss —
the reweighted `BCEWithLogitsLoss` doesn't track balanced accuracy 1:1).
Unlike every other script, its hyperparameters are `argparse`-overridable
(defaults still live in `config.py` as `GNN_*` constants, so a no-args run
matches every other script's zero-CLI-arg convention) because it's meant to
be invoked repeatedly for ablation sweeps. Its output,
`outputs/gnn_metrics.csv`, **appends** rather than overwrites (unlike
`baseline_metrics.csv`, which is a snapshot re-generated each run) — it's an
accumulating experiment log, distinguished by the `edge_type`/`tag` columns.
`04_build_graph.py --k` similarly appends to `outputs/graph_edge_comparison.csv`
and encodes `k` into the `.npy` filename (`edges_geo_k20.npy`) — `06_train_gnn.py
--k_neighbors` picks which of those to load, so a `--k_neighbors N` run
requires `04_build_graph.py --k N` to have been run first, else it's a missing-file
error, not a silent fallback.

**Of the 3 edge candidates, only `geo` (State+City blocking) beat the XGBoost
baseline** on all 4 metrics (see `outputs/gnn_metrics.csv`) — `config.GNN_DEFAULT_EDGE_TYPE
= "geo"` reflects that decision; `temporal`/`weapon`/`geo_temporal` (their
union) remain runnable via `--edge_type` but aren't the default.
`src/ablation_sweep.py` (unnumbered utility, same tier as `eda.py` — no
pipeline artifact of its own) drives `06_train_gnn.py` repeatedly to vary one
hyperparameter at a time from the `config.GNN_*` defaults, then summarizes
`gnn_metrics.csv` by dimension. Best config found so far: `k_neighbors=20,
lr=0.005` (rest at defaults) → balanced_accuracy 0.6525, vs. 0.6456 baseline
and 0.6507 at all-defaults — `num_layers=1` and `aggr="max"` are clearly
worse (graph structure and mean-aggregation both matter), everything else is
fairly flat around the defaults.

**Target leakage is the load-bearing constraint of this dataset.** Perpetrator
columns (`Perpetrator Sex/Age/Race/Ethnicity/Count`, `Relationship`) are
90–99% `Unknown` specifically on unsolved cases — including them makes the
target trivially predictable. `config.LEAKAGE_COLS` encodes this and must stay
excluded from any model input; a suspiciously high AUC (~0.99) is the signal
this was reintroduced (see Plan B risk table in the dev plan doc).

**Methodology follows Campedelli (2022, *Journal of Criminal Justice*)** by
deliberate design choice, not just convention: full one-hot encoding (not
embeddings/ordinal), 5-year age binning, 70/30 random stratified split,
5-fold stratified CV, and Balanced Accuracy + Precision as primary metrics
(ROC-AUC/PR-AUC are secondary). Deviating from these should be a conscious
decision, since the baseline is meant to be paper-comparable.

**Sensitive attributes** (`Victim Race`, `Victim Sex`) are carried through
`03_features.py` as `sens__*` columns *unencoded*, separate from the model
input matrix `X` — they exist for the fairness diagnosis stage, not for
training. `05_train_baseline.py:load_xy()` explicitly strips them before
fitting.

`outputs/` and `data/processed/` are gitignored (regenerable); the raw CSV in
`dataset/` is also gitignored (too large to commit).

## Environment notes (Windows, non-ASCII user path)

The dev machine's Windows profile path contains non-ASCII (Korean) characters.
`config.py` redirects `TEMP`/`TMP`/`JOBLIB_TEMP_FOLDER` to `<ROOT>/.joblib_tmp`
(ASCII, gitignored) before any other import, because joblib's
`resource_tracker` crashes with `UnicodeEncodeError` under the default temp
path whenever something uses `n_jobs>1` (`GridSearchCV`, and later any
DataLoader multiprocessing for the GNN). Don't remove this — it's not
optional cleanup, it's what makes parallel sklearn/joblib calls run at all
here. `requirements.txt` pins `xgboost>=2.1.4` for the same reason 2.1.3
shipped: earlier xgboost is incompatible with scikit-learn>=1.6's
`__sklearn_tags__` API and crashes `GridSearchCV.fit()`.

**matplotlib output text is English-only** (titles/labels/legends in `eda.py`
and any future plotting script) — the default font has no Hangul glyphs, so
Korean text silently renders as blank boxes in saved PNGs. Console
`print()`/comments can stay Korean; only rendered plot text must be English.

**`import torch` needs `KMP_DUPLICATE_LIB_OK=TRUE`** on this machine or it
crashes with `OMP: Error #15` (duplicate OpenMP runtime, torch's bundled
`libiomp5md.dll` conflicting with one sklearn/xgboost/mkl already loaded).
`config.py` sets this via `os.environ.setdefault` before any script gets a
chance to `import torch` — same "fix once, centrally, before the first
import" pattern as the joblib/TEMP block above. `requirements.txt` lists
`torch`/`torch-geometric` but flags that a plain `pip install` on a fresh
machine pulls CPU-only wheels; the CUDA 12.4 build this project actually
runs on needs `pip install torch --index-url
https://download.pytorch.org/whl/cu124` installed first.
