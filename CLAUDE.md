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

Written docs with the full design rationale (Korean): `문제정의서_CLEAR.md`
(problem definition), `AI모델_개발계획서_살인사건검거_GNN.md` (dev plan —
graph/edge design, GNN architecture, 4-week roadmap), `EDA_보고서.md` (EDA
findings).

## Commands

```bash
pip install -r requirements.txt
cd src
python 01_clean.py            # raw CSV -> data/processed/clean.parquet
python 02_sample.py           # -> data/processed/sample.parquet
python 03_features.py         # -> data/processed/features.parquet
python eda.py                 # -> outputs/eda_*.csv, outputs/eda_*.png
python 05_train_baseline.py   # -> outputs/baseline_metrics.csv, outputs/baseline_xgb_top_features.csv
```

Scripts must run from `src/` and in this order — each stage reads the parquet
the previous stage wrote (see `config.py` paths). There is no test suite; there
is no build/lint step configured.

## Architecture

**Shared config**: every script does `import config as C`. `src/config.py` is
the single source of truth for paths, target encoding, column lists, and
hyperparameters (random state, split ratios, CV folds). When changing any of
these, edit `config.py` rather than a script.

**Pipeline is a chain of scripts, not a package**: each `NN_*.py` file is a
standalone stage that reads a parquet from `data/processed/`, transforms it,
and writes the next parquet. There's an intentional numbering gap —
`04_build_graph.py`, `06_train_gnn.py`, `07_fairness.py` are planned (weeks
2–3 per the dev plan) but not yet implemented; `05_train_baseline.py` exists
now purely as the pre-graph performance floor the future GNN must beat.

```
dataset/kaggle_homicide_Reports_1980_2014.csv  (not in git, ~638k rows)
  -> 01_clean.py    target-encode; drop leakage + low-info columns; clean age
  -> 02_sample.py   filter to config.SAMPLE_STATES (default: California only)
  -> 03_features.py 5-yr age bins, decade bins, full one-hot; splits off
                     sens__Victim Race / sens__Victim Sex for later fairness work
  -> eda.py / 05_train_baseline.py  (both read features.parquet or sample.parquet)
```

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
