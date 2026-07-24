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
python 05_train_baseline.py   # -> outputs/metrics.csv (shared ledger, appends), outputs/baseline_xgb_top_features.csv, outputs/predictions_{logreg,xgboost}.csv
python 04_build_graph.py            # -> data/processed/graph/edges_{geo,temporal,weapon}_k{k}.npy (k=config.K_NEIGHBORS)
python 04_build_graph.py --k 20     # rebuild at a different degree cap (needed before --k_neighbors 20 below works)
python 06_train_gnn.py              # -> outputs/metrics.csv (appends; geo only, one row per config.GNN_SEEDS seed)
python 06_train_gnn.py --edge_type all                                   # train geo+temporal+weapon for comparison
python 06_train_gnn.py --edge_type geo --hidden_dim 128 --tag my_sweep   # any hyperparam overridable via CLI
python 06_train_gnn.py --edge_type geo --k_neighbors 20                  # use a --k 20 graph built above
python 06_train_gnn.py --edge_type geo --seeds 42,43,44                  # torch seeds to repeat over (split stays fixed)
python 06_train_gnn.py --edge_type geo_temporal                          # union of geo+temporal edges
python ablation_sweep.py            # in-process one-factor-at-a-time sweep; mean±std over config.GNN_SEEDS
python 07_fairness.py               # diagnoses EVERY outputs/predictions_*.csv found -> fairness_group_metrics.csv + fairness_gaps.csv
python 07_fairness.py --models graphsage_geo xgboost   # restrict to specific dumps
python 07_fairness.py --min_n 5000 --n_boot 2000       # stricter group floor / more bootstrap reps
python 05_train_baseline.py --blind   # same, but with race/sex/ethnicity dummies dropped from X
python 06_train_gnn.py --blind        # ditto (graph unchanged) -> predictions_graphsage_geo_blind.csv
python edge_homophily.py              # -> outputs/edge_homophily.csv (are edges a sensitive-attribute proxy?)
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
`outputs/metrics.csv` log), `clear.predictions` (the test-prediction dump
format that hands off from training to diagnosis), `clear.fairness` (group
metrics, gaps, bootstrap CIs). The numbered scripts (`05`, `06`, `07`) and the
unnumbered `ablation_sweep.py` are thin CLIs over `clear/`; `config.py` still
holds all paths/constants. This replaced an earlier state where `load_xy`,
the metric block, and the split were copy-pasted across `05`/`06` and kept in
sync by hand. `05_train_baseline.py` was built before
`04_build_graph.py`/`06_train_gnn.py` despite the lower number: it's the
pre-graph performance floor (flat XGBoost/LogReg) the GNN has to beat, so it
needed to exist first. `07_fairness.py` (fairness diagnosis, week 3) is now
implemented, so the numbering is contiguous; the mitigation stage (week 3's
"prescribe" half) has no script yet.

```
dataset/kaggle_homicide_Reports_1980_2014.csv  (not in git, ~638k rows)
  -> 01_clean.py    target-encode; drop leakage + low-info columns; clean age
  -> 02_sample.py   filter to config.SAMPLE_STATES (default: California+Texas+Michigan)
  -> 03_features.py 5-yr age bins, decade bins, full one-hot; *copies* race/sex
                     off as sens__* for fairness work — the one-hot dummies of the
                     same columns stay in X unless load_xy(blind=True)
  -> eda.py / 05_train_baseline.py       (read features.parquet or sample.parquet;
                                           05 -> outputs/predictions_{logreg,xgboost}[_blind].csv)
  -> 04_build_graph.py                   (reads sample.parquet + features.parquet ->
                                           data/processed/graph/edges_{geo,temporal,weapon}.npy)
  -> 06_train_gnn.py                     (reads features.parquet + edges_*.npy ->
                                           outputs/predictions_graphsage_{edge}[_blind].csv)
  -> 07_fairness.py                      (reads every prediction dump ->
                                           outputs/fairness_{group_metrics,gaps,model_contrasts}.csv)
     edge_homophily.py                   (reads edges_*.npy + sens__* ->
                                           outputs/edge_homophily.csv)
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

**Sensitive attributes are in the model input by default — `sens__*` is a
diagnosis copy, not a removal.** `03_features.py` carries `Victim Race` and
`Victim Sex` through as unencoded `sens__*` columns for the fairness stage,
and `clear.data.load_xy()` strips those. But `config.CATEGORICAL_COLS`
*also* lists `Victim Sex`/`Victim Race`/`Victim Ethnicity`, so the same
attributes survive as one-hot dummies (`Victim Race=Black`, …) — 11 of `X`'s
73 columns. Every result recorded before 2026-07-24 (baseline, GNN, and the
first fairness diagnosis) therefore comes from models that saw race and sex
**directly**; this doc previously claimed the opposite, and the claim was
wrong. `load_xy(blind=True)` drops the dummies whose prefix is in
`config.SENSITIVE_FEATURE_COLS`, and `05`/`06` expose it as `--blind`
(ledger `tag="blind"`, dumps suffixed `_blind`), which is both the simplest
mitigation (fairness through unawareness) and the only condition under which
the graph-as-proxy question is answerable. `Victim Ethnicity` is included in
that list even though it isn't one of the two diagnosed attributes, because
Hispanic-origin is a direct race proxy and leaving it in would keep a
non-graph path open. `clear.data.load_sensitive()` is the other half of the
`sens__*` pair:
it returns the same `sens__*` columns in the *same row order*, so the split
indices from `get_split` `iloc` into it directly. That's what lets `06` dump
test-node predictions joined to sensitive attributes
(`outputs/predictions_graphsage_{edge}.csv`, on by default,
`--no_dump_predictions` to skip) and `07_fairness.py` diagnose group-wise
fairness **without re-running training** — it reads a CSV, not a model. The
dumped `proba` is the mean over `config.GNN_SEEDS` (the seeds share one test
set), so the diagnosis isn't reading one seed's threshold noise.

**`07_fairness.py` measures three group-wise gaps** over each sensitive
attribute: demographic-parity gap (max−min `selection_rate`), equalized-odds
gaps (max−min TPR and FPR), and — as the reference the other two are read
against — `base_rate_gap`, the max−min of the *actual* clearance rate. The
headline quantity is the **amplification ratio** = model gap ÷ base-rate gap:
above 1 means the model widened a disparity that was already in the data,
which is the diagnosis the project exists to make. Gaps are computed over
named groups only (`Unknown` excluded — it's a recording artifact, not a
population), and always at **two group floors** (all named groups, and
n ≥ `config.FAIRNESS_MIN_GROUP_N`) because on this sample the race max and min
are `Native American/Alaska Native` (n=178) and `Asian/Pacific Islander`
(n=1,431) — reporting only the unfiltered number would hand a small-sample
artifact the headline. This is the "민감속성 소수 그룹 희소" risk the dev-plan
doc's Plan B table anticipated. Note that max−min is an upward-biased
statistic (it's the max of noisy estimates); the bootstrap CI quantifies the
interval but does **not** remove that bias, which is why both floors ship.

**The bootstrap in `clear.fairness` resamples within groups, and shares its
draws across models.** Group sizes are fixed by the data rather than sampled,
so resampling is stratified within each group. Two implementation points
matter. (1) Instead of drawing n×B row indices, note that every row falls in
one of the four TN/FP/FN/TP cells, so a within-group resample is *exactly* a
`Multinomial(n, cell_proportions)` draw — a `(B,4)` matrix per group, which is
why B=1000 over 57k rows is instant. (2) Model-vs-model differences are computed **paired within
replicate**, by lifting the draw to **joint cells** — a row's cell becomes the
pair of its two per-model cells (16 of them), and one multinomial draw is
marginalized back to each model's `(B,4)`. Pairwise, not all-models-at-once:
an earlier version jointly drew 4^M cells across every dump, which dies once
dumps accumulate (10 dumps × 1000 replicates ≈ 8 GB). Restricting the join to
the pair being contrasted is statistically equivalent — marginalizing a
multinomial over grouped cells gives back a multinomial — and diagnosing seven
dumps takes 1.9 s. Point estimates are deterministic; CI bounds are Monte
Carlo and move in the third decimal if the draw structure changes. This is load-bearing, not decoration — comparing
each model's independent CI for overlap is invalid on a shared test set, and
in practice it flips a conclusion here: GraphSAGE's and XGBoost's sex-gap
amplification CIs overlap almost entirely (2.67 [2.39, 2.98] vs 2.55
[2.29, 2.85]) while the paired difference is a clean +0.11 [+0.04, +0.20].
`07` prints the contrast table and writes it to
`outputs/fairness_model_contrasts.csv`; read model comparisons **only** from
there.

**`edge_homophily.py` asks whether the edges themselves encode the sensitive
attributes** — the mechanism question behind any GNN-vs-flat fairness gap,
since message passing can carry a neighbour's race into a node's
representation even when race is absent from `X`. It reports raw homophily,
the null expectation under group-size-preserving rewiring (`sum_g p_g^2`), the
normalized assortativity between them, and a direct "can neighbours recover
this node's group by majority vote" accuracy. Normalizing matters: `geo`'s
raw sex-homophily is 0.68, which looks high until you notice the null is
0.675 (the sample is ~80% male). Measured at k=20: `weapon` is 1.000 on both
attributes — expected, since `config.WEAPON_BLOCK_COLS` blocks *on* race and
sex, which makes it a perfect proxy by construction and a standing warning
against `--edge_type weapon`; `temporal` is ~0 on both (a useful null
control); `geo` is race-assortative (0.174, neighbour-majority recovers race
+7.8pp over the majority baseline) but **not** sex-assortative (0.019, +0.0pp).
Read against the sighted-model diagnosis alone this looks *backwards* — the
significant GraphSAGE-over-XGBoost amplification was on *sex* (where geo
carries no signal) and unresolvable on *race* (where it does). The confound
is that race was in `X` all along (next note), so the graph's copy of it was
redundant and invisible. `--blind` closes the direct path, and then the
homophily measurement predicts the outcome correctly — see the blind-run
findings below. Keep that ordering in mind before treating this table as a
null result.

**`05` and `06` both dump test predictions** (`clear.predictions`) joined to
the unencoded sensitive attributes, so diagnosis reads CSVs and never
re-instantiates a model — `07` re-runs in seconds against a GNN that took
minutes to train, and `08` (mitigation, not yet written) can write mitigated
predictions in the same format to be diagnosed by the same code. `07` with no
arguments diagnoses *every* dump it finds, which is what makes the flat-vs-graph
fairness comparison the default rather than an extra step.
`clear.predictions.assert_same_test_set` fails loudly if two dumps disagree on
`row_index`, since a silently mismatched test set would still produce a
plausible-looking comparison table.

**The `--blind` runs settled where each disparity comes from, and the two
sensitive attributes answer differently.** All figures below are
demographic-parity amplification (model gap ÷ base-rate gap) with 95%
bootstrap CIs; model-vs-model figures are paired differences.

- **Sex is entirely the direct feature.** Blinding collapses every model from
  2.55–3.29× to 0.88–0.96×, i.e. no amplification left. Consistent with geo
  having no sex assortativity: there is no graph path to fall back on, so
  deleting the column deletes the disparity.
- **Race is partly carried by the graph.** On the White-vs-Black contrast
  (`--min_n 5000`), sighted models are indistinguishable — GraphSAGE 2.19×
  vs XGBoost 2.23×, paired difference −0.04 [−0.15, +0.08]. Blind, they
  separate: LogReg 0.05×, XGBoost 0.67×, **GraphSAGE 1.48×**, paired
  GraphSAGE−XGBoost **+0.82 [+0.65, +1.03]**. Once the direct path is closed,
  the flat models stop amplifying and the GNN does not — the residual is the
  race-assortative `geo` edge structure. This is the project's mechanism
  result, and it is the thing a graph-level mitigation (FairDrop-style
  de-homophilizing) would target.
- **The edge-swap control holds architecture fixed.** GraphSAGE on `temporal`
  edges (assortativity 0.002) blind gives 0.36× [0.20, 0.54] against `geo`'s
  1.48×, paired difference **+1.12 [+0.93, +1.36]** — same model, same
  features, same training, only the wiring differs, so the GNN-vs-XGBoost
  architecture confound is gone. `temporal` even lands *below* blind XGBoost
  (−0.31 [−0.41, −0.22]), i.e. a non-assortative graph appears to dilute
  group signal rather than merely not carry it; that one is an unexpected
  observation, not a claim. Residual confound: `geo` and `temporal` differ in
  block-size distribution and density too, so isolating homophily *alone*
  needs the within-`geo` edge intervention — which is exactly what the
  mitigation stage will be. Accuracy tracks the same axis: `geo` blind MCC
  0.266 vs `temporal` blind 0.249, so the edge type that buys accuracy is the
  one that buys race bias.
- **Blinding costs ~0.02 MCC** for all three models (GraphSAGE 0.287→0.266,
  XGBoost 0.274→0.255, LogReg 0.233→0.215) and does **not** cost the GNN its
  accuracy lead (+0.013 sighted, +0.011 blind, against a GNN seed std of
  0.0005).

Caveat to carry forward: `--min_n 5000` was chosen *after* seeing that the
n≥1000 floor left this contrast borderline (+0.40 [−0.01, +0.86]). It is
defensible — the White-vs-Black pair was flagged as the robust contrast in
the very first diagnosis, both floors agree in direction, and the direction
was predicted in advance by `edge_homophily.py` — but it is a post-hoc floor
and all three floors stay in `outputs/fairness_gaps.csv` so the choice is
visible rather than buried.

`data/processed/` is gitignored, as are the raw CSV in `dataset/` (too large),
`outputs/*.png`, and `outputs/predictions_*.csv` (a few MB each, regenerable by
re-running `05`/`06`). The small result CSVs under `outputs/` **are**
tracked — they're the experiment record. The shared metrics ledger
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
