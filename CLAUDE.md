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
python 05_train_baseline.py   # -> outputs/metrics.csv (shared ledger, appends), outputs/baseline_xgb_top_features.csv
python 04_build_graph.py            # -> data/processed/graph/edges_{geo,temporal,weapon}_k{k}.npy (k=config.K_NEIGHBORS)
python 04_build_graph.py --k 20     # rebuild at a different degree cap (needed before --k_neighbors 20 below works)
python 06_train_gnn.py              # -> outputs/metrics.csv (appends; geo only, one row per config.GNN_SEEDS seed)
python 06_train_gnn.py --edge_type all                                   # train geo+temporal+weapon for comparison
python 06_train_gnn.py --edge_type geo --hidden_dim 128 --tag my_sweep   # any hyperparam overridable via CLI
python 06_train_gnn.py --edge_type geo --k_neighbors 20                  # use a --k 20 graph built above
python 06_train_gnn.py --edge_type geo --seeds 42,43,44                  # torch seeds to repeat over (split stays fixed)
python 06_train_gnn.py --edge_type geo_temporal                          # union of geo+temporal edges
python ablation_sweep.py            # in-process one-factor-at-a-time sweep; mean±std over config.GNN_SEEDS
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

**Numbered scripts are thin stages; shared logic lives in the `clear/`
package.** Each `NN_*.py` file is a standalone pipeline stage that reads a
parquet (or `.npy`) from `data/processed/`, transforms it, and writes the next
artifact. A leading digit isn't a valid Python identifier, so
`import 05_train_baseline` simply doesn't work — which is exactly *why* logic
that two stages must share can't live in a numbered file. That shared logic
lives in `src/clear/` (a normal importable package): `clear.data` (`load_xy`,
`get_split`), `clear.metrics` (`evaluate` — the one AUC/MCC/F1/Sens/Spec
definition), `clear.graph` (edge loading + `geo_temporal`-style unions),
`clear.gnn` (the GraphSAGE model + training loop), `clear.ledger` (the shared
`outputs/metrics.csv` log). The numbered scripts (`05`, `06`) and the
unnumbered `ablation_sweep.py` are thin CLIs over `clear/`; `config.py` still
holds all paths/constants. This replaced an earlier state where `load_xy`,
the metric block, and the split were copy-pasted across `05`/`06` and kept in
sync by hand. `05_train_baseline.py` was built before
`04_build_graph.py`/`06_train_gnn.py` despite the lower number: it's the
pre-graph performance floor (flat XGBoost/LogReg) the GNN has to beat, so it
needed to exist first. `07_fairness.py` (fairness diagnosis, week 3) is still
unimplemented — that's the one remaining numbering gap.

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

**`05` and `06` share one test split via `clear.data.get_split`**, so
GraphSAGE's test metrics are directly comparable to the XGBoost baseline's —
comparability is now guaranteed by *calling the same function*, not (as
before) by re-calling `train_test_split` with matching args and trusting the
"row assignment depends only on `n`/`stratify`/`random_state`" invariant. The
baseline trains on the split's `trainval` (order preserved, so its CV folds
are byte-identical to before); the GNN carves a further validation slice for
early stopping on validation MCC (not validation loss — the reweighted
`BCEWithLogitsLoss` doesn't track MCC 1:1). `06`'s hyperparameters are
`argparse`-overridable (defaults live in `config.py` as `GNN_*` constants, so a
no-args run matches the zero-CLI-arg convention) because it's meant to be
invoked repeatedly. **Both `05` and `06` append to one shared ledger,
`outputs/metrics.csv`** (`clear.ledger`), each row tagged with a `model`
column (`logreg`/`xgboost`/`graphsage`) — so "did the GNN beat baseline" is a
single `ledger.summarize()` groupby, not a cross-file comparison. Rows are
reindexed to a fixed schema on append, so baseline rows (which leave the GNN
hyperparameter columns empty) stay column-aligned with GNN rows. **`06` runs
each config over `config.GNN_SEEDS`** (varying only the torch seed; the split
stays fixed so every seed shares the baseline's test set), because GPU
scatter-aggregation is non-deterministic and the threshold-dependent metrics
(F1/Sens/Spec) swing ±3–4 points run-to-run — MCC/AUC are stable. `summarize`
reports mean±std so a margin can be judged against that noise; the baseline is
deterministic and stays one reference row per model.
`04_build_graph.py --k` similarly appends to `outputs/graph_edge_comparison.csv`
and encodes `k` into the `.npy` filename (`edges_geo_k20.npy`) — `06_train_gnn.py
--k_neighbors` picks which of those to load, so a `--k_neighbors N` run
requires `04_build_graph.py --k N` to have been run first, else it's a missing-file
error, not a silent fallback.

**Of the 3 edge candidates, `geo` (State+City blocking) is the default**
(`config.GNN_DEFAULT_EDGE_TYPE = "geo"`): it clears the XGBoost baseline on the
**stable** metrics — MCC (0.285 ± 0.0001 vs 0.274) and AUC (0.712 ± 0.001 vs
0.703), margins ~10–100× the GNN's own seed-to-seed noise. The apparent
F1/Sensitivity ordering between GNN and baseline is **within** that noise
(sensitivity std ≈ 0.02), so it isn't claimed as a win — this is exactly what
the seed-repeats were added to expose. `temporal`/`weapon`/`geo_temporal`
(their union) remain runnable via `--edge_type` but aren't the default.
`src/ablation_sweep.py` (unnumbered utility, same tier as `eda.py` — no
pipeline artifact of its own) now **imports `clear.gnn` and runs in-process**
(loads the data once) instead of `subprocess`-relaunching `06` per config; it
varies one hyperparameter at a time from the `config.GNN_*` defaults over
`config.GNN_SEEDS`, then summarizes the `outputs/metrics.csv` ledger by
dimension as mean±std, ranked on MCC. Best config found so far:
`k_neighbors=20, lr=0.005` (rest at defaults, both since promoted into
`config.py`) — `num_layers=1` and `aggr="max"` are clearly worse (graph depth
and mean-aggregation both matter), everything else is fairly flat around the
defaults (which is itself a mean±std judgment now, not a single-run one).

**Target leakage is the load-bearing constraint of this dataset.** Perpetrator
columns (`Perpetrator Sex/Age/Race/Ethnicity/Count`, `Relationship`) are
90–99% `Unknown` specifically on unsolved cases — including them makes the
target trivially predictable. `config.LEAKAGE_COLS` encodes this and must stay
excluded from any model input; a suspiciously high AUC (~0.99) is the signal
this was reintroduced (see Plan B risk table in the dev plan doc).

**Methodology follows Campedelli (2022, *Journal of Criminal Justice*)** by
deliberate design choice, not just convention, for everything except the
evaluation metrics: full one-hot encoding (not embeddings/ordinal), 5-year
age binning, 70/30 random stratified split, 5-fold stratified CV. Deviating
from these should be a conscious decision, since the baseline is meant to be
paper-comparable on the data/split/encoding side.

**Evaluation metrics are AUC/MCC/F1/Sensitivity/Specificity** (the project's
own choice, a deliberate deviation from the paper) **plus Balanced Accuracy +
Precision** — the latter two added later purely so our models can be compared
to the literature (Campedelli 2022 reports *only* those two) on a shared axis;
they are secondary reporting metrics, not selection criteria. There is now a
single definition of this 7-metric set, `clear.metrics.evaluate`, that both
`05` and `06` call (it used to be copy-pasted into each and kept in sync by
hand), so they can't drift apart. The ledger schema (`clear.ledger`) carries
all seven; `balanced_accuracy`/`precision` were appended *after* the original
five so existing `outputs/metrics.csv` rows stay column-aligned, and historical
rows were backfilled by exact algebra (BA = (Sens+Spec)/2, Precision =
F1·Sens/(2·Sens−F1)) rather than re-running. On this CA+TX+MI sample our
Balanced Accuracy (XGB ≈ 0.65) sits well below the paper's (national XGB 0.767,
California 0.802) — expected, because the paper's top-2 SHAP predictors are
unavailable here: `Circumstance` is absent from the Kaggle CSV, and
`Number of Offenders` (= `Perpetrator Count`) is excluded as leakage. See
`reports/모델벤치마크_선행연구비교.md` for the full comparison and analysis.
MCC is the
representative scalar for model selection (`GridSearchCV(scoring=
"matthews_corrcoef")` in the baseline, validation-MCC early stopping in the
GNN, MCC-sorted ablation summaries) because it stays informative under this
dataset's ~32%/68% class imbalance the way plain accuracy wouldn't.
Sensitivity = recall on the positive (solved) class; Specificity =
`recall_score(y, pred, pos_label=0)`, i.e. recall on the negative (unsolved)
class — both via sklearn's `recall_score` rather than manual confusion-matrix
arithmetic.

**Sensitive attributes** (`Victim Race`, `Victim Sex`) are carried through
`03_features.py` as `sens__*` columns *unencoded*, separate from the model
input matrix `X` — they exist for the fairness diagnosis stage, not for
training. `clear.data.load_xy()` (shared by `05`/`06`) explicitly strips them
before fitting.

`outputs/` and `data/processed/` are gitignored (regenerable); the raw CSV in
`dataset/` is also gitignored (too large to commit). The shared metrics ledger
is `outputs/metrics.csv` (both trainers append; `model` column distinguishes
rows); it supersedes the old per-model `baseline_metrics.csv`/`gnn_metrics.csv`.

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
