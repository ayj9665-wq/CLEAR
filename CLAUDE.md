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

Everything runs from `src/`. The numbered scripts at the root are the data
pipeline and **must run in order**; everything under `experiments/` reads the
prepared artifacts and has **no order among itself**, so those run as modules.

```bash
pip install -r requirements.txt
cd src

# --- data pipeline (ordered; each stage reads what the previous one wrote) ---
python 01_clean.py            # raw CSV -> data/processed/clean.parquet
python 02_sample.py           # -> data/processed/sample.parquet
python 03_features.py         # -> data/processed/features.parquet
python 04_build_graph.py      # -> data/processed/graph/edges_{geo,temporal,weapon}_k{k}.npy (k=config.K_NEIGHBORS)
python 04_build_graph.py --k 20     # rebuild at a different degree cap (needed before --k_neighbors 20 below works)

# --- experiments (unordered; all write outputs/results.csv) ------------------
python -m experiments.train_gnn           # -> results.csv (family=train; geo only, one row per config.GNN_SEEDS seed)
python -m experiments.train_gnn --edge_type all                                 # train geo+temporal+weapon for comparison
python -m experiments.train_gnn --edge_type geo --hidden_dim 128 --tag my_sweep # any hyperparam overridable via CLI
python -m experiments.train_gnn --edge_type geo --k_neighbors 20                # use a --k 20 graph built above
python -m experiments.train_gnn --edge_type geo --seeds 42,43,44                # torch seeds to repeat over (split stays fixed)
python -m experiments.train_gnn --edge_type geo_temporal                        # union of geo+temporal edges
python -m experiments.train_gnn --blind        # race/sex/ethnicity dummies dropped from X (graph unchanged)

python -m experiments.diagnose_fairness   # diagnoses EVERY dump in outputs/predictions/ -> fairness_{group_metrics,gaps,model_contrasts}.csv
python -m experiments.diagnose_fairness --models graphsage_geo xgboost   # restrict to specific dumps
python -m experiments.diagnose_fairness --min_n 5000 --n_boot 2000       # stricter group floor / more bootstrap reps
python -m experiments.edge_homophily      # -> outputs/edge_homophily.csv (are edges a sensitive-attribute proxy?)

python -m experiments.mitigate_graph      # -> results.csv (family=mitigate_graph; drop same-race edges + matched random control)
python -m experiments.mitigate_loss       # -> results.csv (family=mitigate_loss; fairness penalty in the training loss)
python -m experiments.mitigate_loss --beta 1.0                              # early-stop on val MCC - beta*val gap instead of val MCC alone
python -m experiments.mitigate_loss --alphas 0 25 50 75 --seeds 42,43,44,45 # finer alpha grid near the useful range

python -m experiments.poster_figures      # -> outputs/poster_fig{1,2}_*.png (reads result CSVs only, no retraining)
```

There is no test suite; there is no build/lint step configured.

## Scope: this repo is the graph line of work

Four scripts were removed once their results were in hand, to leave one clean
line — build graph → train GNN → diagnose → mitigate:

| removed | produced | still available as |
|---|---|---|
| `train_baseline.py` | flat LogReg/XGBoost floor | `results.csv` family=`train` (logreg/xgboost rows) **+ committed dumps** |
| `mitigate_threshold.py`, `clear/mitigate.py` | group-wise threshold sweep | `results.csv` family=`mitigate_threshold` (4,004 rows) |
| `ablation.py` | hyperparameter OFAT sweep | `results.csv` family=`ablation` (480 rows); winners promoted into `config.py` |
| `eda.py`, `migrate_results.py` | EDA plots; one-time schema migration | `eda_clearance_by_group.csv`; migration already applied |

**All of it is recoverable from commit `96d1dd1`** — nothing is lost, only
removed from the working tree. Findings sections below that describe these
experiments still stand; they document results, not live code.

The one thing that was *not* recoverable is the flat models' test predictions:
they are gitignored and only `train_baseline.py` could make them, yet the
project's headline is a model contrast that needs them —
`graphsage_geo_blind − xgboost_blind = +0.82 [+0.65, +1.03]`. So those four
dumps are committed as `.gitignore` exceptions (8.4 MB), and
`diagnose_fairness` reproduces that table exactly without the trainer. Sixty
citations across the fairness and benchmark reports depend on this.

`xgboost` was dropped from `requirements.txt` for the same reason;
`scikit-learn` stays (splitting + metrics).

## Architecture

**Shared config**: every script does `import config as C`. `src/config.py` is
the single source of truth for paths, target encoding, column lists, and
hyperparameters (random state, split ratios, CV folds). When changing any of
these, edit `config.py` rather than a script.

**Two tiers: an ordered pipeline, and an unordered experiment surface.**
`src/01_clean.py` … `src/04_build_graph.py` are the pipeline — each reads what
the previous one wrote, so the numbers encode a real dependency and they stay.
Everything in `src/experiments/` reads `features.parquet` + `edges_*.npy` and
writes to `outputs/`; **none of them depend on each other.**

Those experiment files used to be numbered `05`–`10` too, and that was wrong in
two ways. The numbers encoded *authorship order*, not dependency — you can run
`mitigate_loss` without `mitigate_threshold`, and `mitigate_graph` is a failed
branch rather than a prerequisite. And a leading digit isn't a valid Python
identifier, so `import 05_train_baseline` was a syntax error: scripts physically
could not share code with each other, which pushed every shared line into
`clear/` or, when that didn't happen, into copy-paste. They are now plain
modules run as `python -m experiments.train_gnn` (`-m` puts the CWD on
`sys.path`, so `config`/`clear` resolve when run from `src/`). Old number → new
name: `05` train_baseline, `06` train_gnn, `07` diagnose_fairness,
`08` mitigate_threshold, `09` mitigate_graph, `10` mitigate_loss,
`ablation_sweep` → `ablation`.

The pipeline scripts stayed at `src/` root rather than moving into a
`pipeline/` package: leading digits block `python -m`, so a subpackage would
need a `sys.path` shim in every file — reintroducing exactly the per-file
boilerplate this refactor removes — and buys nothing, since `01`–`04` never
import each other.

**Shared logic lives in `src/clear/`** (a normal importable package):
`clear.data` (`load_xy`, `get_split`), `clear.metrics` (`evaluate` — the one
7-metric definition), `clear.graph` (edge loading + `geo_temporal`-style
unions), `clear.gnn` (the GraphSAGE model + training loop), `clear.results`
(the one result schema, `outputs/results.csv`), `clear.predictions` (the
test-prediction dump format that hands off from training to diagnosis),
`clear.fairness` (group metrics, gaps, bootstrap CIs), `clear.fairgraph`
(homophilous-edge dropping),
`clear.sweep` (the mitigation sweep harness `mitigate_graph`/`mitigate_loss`
share). The experiment scripts are thin CLIs over `clear/`; `config.py` holds
all paths/constants. The package originally existed because of the import
constraint above; it stays because its modules genuinely have multiple callers
(`clear.data` 6, `clear.results` 6, `clear.predictions` 6, `clear.metrics` 5,
`clear.gnn` 5). The rule for what belongs there is **two or more callers** —
single-caller computation stays in its script. `clear.fairgraph` is the one
exception left (only `mitigate_graph` uses it); it stays because its docstrings
carry the method rationale for the edge intervention, which would be buried in
a CLI file. `clear.mitigate` was the other and left with its script.

The prescribe stage was three experiments because the first two answers were
incomplete — `mitigate_threshold` post-processed thresholds (works, but needs
the attribute at decision time), `mitigate_graph` rewires edges (**fails**, and
the failure is what located the real leak), `mitigate_loss` penalises the gap in
the training loss (works, no decision-time attribute). Read the findings in that
order; each exists because of what the previous one could not do. Only the last
two are still code — `mitigate_threshold`'s curve lives on in `results.csv` and
is still plotted on the poster's trade-off panel.

```
dataset/kaggle_homicide_Reports_1980_2014.csv  (not in git, ~638k rows)
  -> 01_clean.py    target-encode; drop leakage + low-info columns; clean age
  -> 02_sample.py   filter to config.SAMPLE_STATES (default: California+Texas+Michigan)
  -> 03_features.py 5-yr age bins, decade bins, full one-hot; *copies* race/sex
                     off as sens__* for fairness work — the one-hot dummies of the
                     same columns stay in X unless load_xy(blind=True)
  -> (removed: eda.py / train_baseline.py)  (read features.parquet or sample.parquet;
                                           -> outputs/predictions/{logreg,xgboost}[_blind].csv)
  -> 04_build_graph.py                   (reads sample.parquet + features.parquet ->
                                           data/processed/graph/edges_{geo,temporal,weapon}.npy)
  -> experiments/train_gnn.py                     (reads features.parquet + edges_*.npy ->
                                           outputs/predictions/graphsage_{edge}[_blind].csv)
  -> experiments/diagnose_fairness.py                      (reads every prediction dump ->
                                           outputs/fairness_{group_metrics,gaps,model_contrasts}.csv)
     experiments/edge_homophily.py                   (reads edges_*.npy + sens__* ->
                                           outputs/edge_homophily.csv)
  -> (removed: mitigate_threshold.py)   (read every dump -> results.csv family=mitigate_threshold)
     experiments/mitigate_graph.py      (edges + sens__* -> retrains -> family=mitigate_graph)
     experiments/mitigate_loss.py       (penalty in the loss -> family=mitigate_loss)
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

**Everything shares one test split via `clear.data.get_split`**, so
GraphSAGE's test metrics are directly comparable to the XGBoost baseline's —
comparability is now guaranteed by *calling the same function*, not (as
before) by re-calling `train_test_split` with matching args and trusting the
"row assignment depends only on `n`/`stratify`/`random_state`" invariant. The
baseline trains on the split's `trainval` (order preserved, so its CV folds
are byte-identical to before); the GNN carves a further validation slice for
early stopping on validation MCC (not validation loss — the reweighted
`BCEWithLogitsLoss` doesn't track MCC 1:1). `train_gnn`'s hyperparameters are
`argparse`-overridable (defaults live in `config.py` as `GNN_*` constants, so a
no-args run matches the zero-CLI-arg convention) because it's meant to be
invoked repeatedly.

**Every experiment writes one table, `outputs/results.csv`** (`clear.results`),
in long format — one row per (run, metric):

```
family timestamp model tag seed attribute blind group_set params notes metric value lo hi std
```

`family` is `train` / `mitigate_graph` / `mitigate_loss` for live code, plus
`ablation` / `mitigate_threshold` from the removed scripts (see Scope). `params` is a JSON object of the knobs that were *chosen*
(`alpha`, `lambda`, `mode`+`p`, the GNN hyperparameters); `notes` is JSON for
values the run *produced* (fitted thresholds, graph statistics). That split is
load-bearing: `params` is part of a row's identity and `notes` is not, so
re-running a config **replaces** its rows instead of appending near-duplicates.
Getting this wrong is not hypothetical — putting fitted thresholds in the
identity key made one re-run add 78 duplicate rows during this refactor.

This replaced four wide tables (`metrics.csv` plus three `*_tradeoff.csv`) that
held the same quantities under different names — `mcc` vs `acc_mcc`,
`selection_rate_gap` vs `dp_gap` — with a different "config column" set each.
Keeping wide schemas in sync cost real complexity: fixed-schema reindexing on
append, a hand-written merge-on-key in the fairloss sweep, and an
exact-algebra backfill when Balanced Accuracy/Precision were added late. Long
format removes the class: **a new metric is new rows, never a new column.**
It also removes a live bug — the old ledger wrote its header once at file
creation while `LEDGER_COLS` grew twice, so `metrics.csv` ended up with 25/26/28
field rows under a 25-field header and `pandas.read_csv` raised `ParserError`
on it. The ablation summary, which reads that file, was simply broken. The
migration recovered those rows positionally (columns were only ever appended,
so an N-field row maps to the first N columns) and the originals are preserved
in `outputs/legacy/`.

Each row is tagged with `family` and `model`, so "did the GNN beat baseline" is
a single `results.summarize()` groupby, not a cross-file comparison. **`train_gnn` runs
each config over `config.GNN_SEEDS`** (varying only the torch seed; the split
stays fixed so every seed shares the baseline's test set), because GPU
scatter-aggregation is non-deterministic and the threshold-dependent metrics
(F1/Sens/Spec) swing ±3–4 points run-to-run — MCC/AUC are stable. `summarize`
reports mean±std so a margin can be judged against that noise; the baseline is
deterministic and stays one reference row per model.
`04_build_graph.py --k` similarly appends to `outputs/graph_edge_comparison.csv`
and encodes `k` into the `.npy` filename (`edges_geo_k20.npy`) — `experiments/train_gnn.py
--k_neighbors` picks which of those to load, so a `--k_neighbors N` run
requires `04_build_graph.py --k N` to have been run first, else it's a missing-file
error, not a silent fallback.

**Of the 3 edge candidates, `geo` (State+City blocking) is the default**
(`config.GNN_DEFAULT_EDGE_TYPE = "geo"`): it clears the XGBoost baseline on the
**stable** metrics — MCC (0.285 ± 0.0001 vs 0.274) and AUC (0.712 ± 0.001 vs
0.703), margins ~10–100× the GNN's own seed-to-seed noise. **Which metrics can
carry the accuracy claim is a measured question, not a stylistic one** — the
full margin-vs-noise table for GraphSAGE(geo) − XGBoost, sighted:

| metric | margin | GNN seed std | ratio | |
|---|---|---|---|---|
| MCC | +0.0111 | ±0.0006 | **19.0×** | claimable |
| AUC | +0.0083 | ±0.0006 | 14.6× | claimable |
| Balanced Accuracy | +0.0061 | ±0.0009 | **6.6×** | claimable |
| Precision | +0.0062 | ±0.0050 | **1.2×** | within noise |
| Specificity | +0.0156 | ±0.0222 | 0.7× | within noise |
| F1 | +0.0003 | ±0.0101 | 0.0× | within noise |
| Sensitivity | −0.0035 | ±0.0205 | −0.2× | within noise |

So **Precision joins F1/Sens/Spec as a metric whose GNN-vs-baseline margin
cannot be claimed** — it looks comparable to Balanced Accuracy's (+0.0062 vs
+0.0061) but its own seed std is 5× larger, because it depends on the 0.5
decision threshold and ignores the negative class entirely. Only MCC and AUC
(threshold-free) and Balanced Accuracy clear the floor. This is exactly what the
seed-repeats were added to expose.

**Metric ordering is identical under MCC and Balanced Accuracy** — GraphSAGE >
XGBoost > LogReg sighted, and GraphSAGE(geo) > XGBoost > GraphSAGE(temporal) >
LogReg blind. The representative-metric choice therefore does not change any
conclusion; it only changes which number leads the presentation. The project's
resolution: **Balanced Accuracy + Precision lead in literature-comparison
contexts** (`reports/모델벤치마크_선행연구비교.md` §1, and the results page),
because Campedelli 2022 reports only those two; **MCC stays the selection
criterion** (GridSearchCV scoring, validation early stopping, ablation ranking)
and is labelled as such rather than as "the representative metric". Switching
selection to Balanced Accuracy is coherent but is not a relabelling: it would
move the operating point, and every fairness number here is
threshold-dependent, so the diagnosis and both mitigations would need re-running. `temporal`/`weapon`/`geo_temporal`
(their union) remain runnable via `--edge_type` but aren't the default.
`src/experiments/ablation.py` (since removed — see Scope; a utility with no
pipeline artifact of its own) now **imports `clear.gnn` and runs in-process**
(loads the data once) instead of `subprocess`-relaunching `train_gnn` per config; it
varies one hyperparameter at a time from the `config.GNN_*` defaults over
`config.GNN_SEEDS`, then summarizes `outputs/results.csv` by
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
to the literature (Campedelli 2022 reports *only* those two) on a shared axis.
**They lead the presentation wherever the paper comparison is the point** —
`reports/모델벤치마크_선행연구비교.md` §1 and the published results page put
Balanced Accuracy and Precision first — but they are **not** selection criteria;
MCC is (see the margin-vs-noise table above for why this split is measured
rather than stylistic). There is now a
single definition of this 7-metric set, `clear.metrics.evaluate`, that both
every trainer calls (it used to be copy-pasted per script and kept in sync by
hand), so they can't drift apart. `clear.results` carries all seven as ordinary
rows. (Historically `balanced_accuracy`/`precision` were appended to a *wide*
ledger schema after the fact, which needed column-alignment care and an exact-
algebra backfill of old rows; the long format removes that whole class of
change — a new metric is new rows, never a new column.) On this CA+TX+MI sample our
Balanced Accuracy (XGB ≈ 0.65) sits well below the paper's (national XGB 0.767,
California 0.802) — expected, because the paper's top-2 SHAP predictors are
unavailable here: `Circumstance` is absent from the Kaggle CSV, and
`Number of Offenders` (= `Perpetrator Count`) is excluded as leakage. See
`reports/모델벤치마크_선행연구비교.md` for the full comparison and analysis.
MCC is the **selection criterion** (`GridSearchCV(scoring=
"matthews_corrcoef")` in the baseline, validation-MCC early stopping in the
GNN, MCC-sorted ablation summaries) because it stays informative under this
dataset's ~32%/68% class imbalance the way plain accuracy wouldn't, and because
it is the sharpest discriminator available here (19× the seed noise, against
Balanced Accuracy's 6.6×). Call it that — "selection criterion" — rather than
"the representative metric", so a table that reports Balanced Accuracy is not
read as reporting a quantity the models were tuned on.

**Keeping the paper-comparison metrics pays off concretely at the mitigation
stage.** Under MCC alone the two working mitigations look interchangeable
(−0.0025 vs −0.0043). Across the wider set they are doing different things:
post-processing moves the two group thresholds in *opposite* directions
(White 0.500→0.517, Black 0.500→0.470), so the global operating point cancels
out and every metric drifts down together by ~0.001; the loss penalty shifts the
operating point *globally* — Sensitivity +0.0245, Specificity −0.0315, Precision
−0.0075 — i.e. it predicts "solved" more often and pays for it in precision.
The load-bearing evidence is the **sign pattern** (Sens up / Spec down), not the
magnitudes, since the loss-penalty figures carry retraining noise
(Sens ±0.021, Spec ±0.022, Precision ±0.005) while post-processing has none.
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
`config.SENSITIVE_FEATURE_COLS` **and then asserts the drop actually happened**
(`clear.data._assert_blind`: non-empty drop list, and no column left starting
with a sensitive prefix). The drop condition depends on `03_features.py`
one-hot-ing as `Victim Race=Black`; if that separator ever changes, the old code
would produce an empty drop list and `--blind` would become a silent no-op that
still exits 0, invalidating every blind result without a single error message.
That doc-vs-reality drift is the exact failure this project already shipped once,
so it is now enforced rather than documented. `train_gnn` exposes the flag as `--blind`
(ledger `tag="blind"`, dumps suffixed `_blind`), which is both the simplest
mitigation (fairness through unawareness) and the only condition under which
the graph-as-proxy question is answerable. `Victim Ethnicity` is included in
that list even though it isn't one of the two diagnosed attributes, because
Hispanic-origin is a direct race proxy and leaving it in would keep a
non-graph path open. `clear.data.load_sensitive()` is the other half of the
`sens__*` pair:
it returns the same `sens__*` columns in the *same row order*, so the split
indices from `get_split` `iloc` into it directly. That's what lets `train_gnn` dump
test-node predictions joined to sensitive attributes
(`outputs/predictions/graphsage_{edge}.csv`, on by default,
`--no_dump_predictions` to skip) and `experiments/diagnose_fairness.py` diagnose group-wise
fairness **without re-running training** — it reads a CSV, not a model. The
dumped `proba` is the mean over `config.GNN_SEEDS` (the seeds share one test
set), so the diagnosis isn't reading one seed's threshold noise.

**`experiments/diagnose_fairness.py` measures three group-wise gaps** over each sensitive
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
`diagnose_fairness` prints the contrast table and writes it to
`outputs/fairness_model_contrasts.csv`; read model comparisons **only** from
there.

**`experiments/mitigate_threshold.py` is the prescribe stage's post-processing half** (`clear.mitigate`):
no retraining, just a per-group threshold applied to a saved dump. Strength is a
single knob λ that interpolates each group's *target rate* — not its threshold —
from its own observed rate (λ=0, reproduces the dump) to the pooled rate (λ=1,
gap 0); rates interpolate meaningfully across models where raw thresholds do not.
Two criteria: `dp` equalizes selection rate, `tpr` equalizes recall on solved
cases. **Thresholds are fitted on a stratified half of the test set and scored on
the other half**, so accuracy here is not comparable to the numbers elsewhere in
this file — the λ=0 row is the baseline for that table. Findings: closing 93–99%
of the gap costs at most 0.008 MCC on every model and both criteria, mild
mitigation (λ≈0.2–0.5) often *improves* MCC by correcting between-group
miscalibration, and the model ranking is unchanged after full mitigation
(GraphSAGE 0.280 > XGBoost 0.267 > LogReg 0.228) — the graph's fairness cost is
repayable and its accuracy edge survives repayment. It is cheap because the
disparity sits in a narrow band at the boundary: even at λ=1 the thresholds move
only ±0.03. Two things to keep attached to that result — demographic parity is a
value choice here (the groups' *actual* clearance rates really do differ, 70.2%
vs 65.5%), and post-processing needs the sensitive attribute **at decision time**, which
an edge-level mitigation *may* avoid depending on the variant: permanently
rewiring the graph still needs race to place a new case's edges, so only the
"drop during training, infer on the full graph" variant actually escapes the
constraint. `experiments/mitigate_graph.py` does the permanent-rewiring form, because that is
the one that doubles as the mechanism test.

**`mitigate_graph` and `mitigate_loss` are the same experiment loop, and it lives in `clear.sweep`.**
Both sweep one knob, and at each point train the GNN over `config.GNN_SEEDS`,
dump test predictions, re-diagnose the gap with `diagnose_fairness`'s own definitions
(`clear.fairness`), and emit one accuracy-vs-fairness row. Only the intervention
differs: `mitigate_graph` swaps the graph (`run_point(..., data=modified)`), `mitigate_loss` changes the
loss (`run_point(..., fair_alpha=...)`). `clear.sweep` holds the shared argparse
block, the load/split/hyperparameter setup, and `run_point`; the scripts keep
only their intervention and their own trade-off columns (which is why `run_point`
returns the metric/gap block *without* `tag`/`blind`/`attribute` — the caller
owns its own knobs). Before this, ~90 lines — the argparse defaults, the seed
loop, the six-line re-diagnosis, the seven-metric aggregation, the six gap
columns — were copy-pasted into both, so changing the gap definition in `diagnose_fairness`
meant editing two more places, and editing only one would silently put the two
mitigation curves on different axes. `clear.gnn.train_eval` grew `data=` and
`**extra` in the same change: it already owned the seed loop but couldn't cover
`mitigate_graph`/`mitigate_loss` (which needed a prebuilt graph and `fair_*` kwargs), so both had
reimplemented it. Adding a fourth mitigation should mean a tag and an
intervention argument, nothing else.

**`experiments/mitigate_loss.py` is the mitigation that works, and it exists because `mitigate_graph`
failed.** Instead of deleting the sensitive information from the input, it stops
the model *using* it: `loss = BCE + alpha * (size-weighted variance of per-group
mean predicted probability)`, computed on training nodes only
(`clear.gnn.fairness_penalty`). With two groups that reduces to the squared
selection-rate gap, i.e. the diagnosis' own quantity written into the loss; the
sigmoid mean stands in for the non-differentiable hard rate. It sidesteps `mitigate_graph`'s
failure mode because it never has to locate the leak — edge, block, or feature,
the penalty presses on the *output*. And it needs race **only at training time**,
which is the one practical objection to the post-processing route.

Blind geo, White-vs-Black, 3 seeds: amplification 1.37 → **0.06 [0.00, 0.24]** at
alpha=50, for 0.0044 MCC. Comparable in cost to post-processing (−0.0025) with
no decision-time attribute requirement — compare the two on *relative* cost, since
`mitigate_threshold` scores on the eval half and `mitigate_loss` on the whole test set.

**The curve inverts above alpha≈50** (amplification 0.06 → 0.19 at 200 → 0.52 at
1000). The first suspicion was model selection: early stopping picks on
validation **MCC**, which knows nothing about the penalty. `--beta` tests that by
selecting on `val MCC − beta·val gap` instead. It is **half the story** — beta=1
softens the inversion (0.52 → 0.31 at alpha=1000) but does not remove it, and
the residual is optimisation instability rather than selection: MCC's seed std
jumps 0.0029 → 0.0178 at alpha=1000 as the penalty swamps the BCE term. So the
sweep stays a trade-off curve only for **alpha ≲ 50**, whatever beta is.

**The two mitigations differ in kind, which only the paper-comparison metrics
reveal.** Post-processing moves group thresholds in *opposite* directions
(White 0.500→0.517, Black 0.500→0.470), so the global operating point cancels
out: Sens/Spec/Precision barely move (Precision −0.0003, and XGBoost's actually
rises). The loss penalty instead shifts the operating point globally —
sensitivity up, specificity down (0.648→0.673, 0.636→0.605) — so it predicts
"solved" more often and pays an order of magnitude more Precision (−0.0075
against post-processing's −0.0003).
Both are small in absolute terms, but if precision matters for the use case,
post-processing is the better instrument; if decision-time attribute use is
blocked, the penalty is. Neither dominates. MCC alone hides this, which is one
concrete payoff of having kept Balanced Accuracy and Precision around.

Two things fell out of that experiment. Fairness-aware early stopping is a
mitigation **on its own** — at alpha=0, beta=1 alone takes amplification 1.37 →
0.91 for 0.0075 MCC, just by choosing a different checkpoint. And the best
accuracy of any mitigated config is beta=1, alpha=50 (MCC 0.2645, amplification
0.101), which beats beta=0, alpha=50 on accuracy while giving up some fairness
(0.058). Pick per what the frontier point needs to be; both are in
`outputs/results.csv` under `family=mitigate_loss`; `clear.results` replaces
rows with a matching identity key, so re-running one beta neither overwrites
the other curve nor duplicates rows.

**`experiments/mitigate_graph.py` attacks the diagnosed cause instead of the symptom**
(`clear.fairgraph`): it drops same-race `geo` edges with probability p and
retrains, so message passing carries less race information. It only means
anything in the `--blind` condition (the default here) — with race still in `X`
the model reads it directly and any edge intervention is masked.

**The random-drop control arm is what makes it a test rather than a demo.**
Removing homophilous edges also removes *edges*, so a shrinking gap could just
be graph thinning. Every p therefore runs twice: `homophily` drops same-race
pairs at rate p, and `random` keeps a **matched count** chosen without regard to
race. At p=1 both arms hold 1,601,328 pairs while homophily is 0.000 vs 0.576 —
identical density, isolated attribute. The conclusion only stands if the gap
falls in the homophily arm and not in the random arm. The random arm doubles as
a noise floor: since its homophily is constant across p, whatever it does across
p is sparsification plus training stochasticity.

That stochasticity is worth watching. Bootstrap CIs cover test-set sampling
only, **not** GNN training randomness, and amplification is a
threshold-dependent quantity — the same blind geo config scored 1.48, 1.33 and
1.37 on three separate runs. Judge effects against the control arm's spread, not
against the CI alone.

**The result: it does not work, and why is the interesting part.** Differences
between arms (−0.07 to −0.16 at p=0.25–0.75) sit inside that ±0.15 training
spread. At p=1 it clearly *backfires* (homophily arm 1.74 vs control 1.11) for
two measurable reasons: homophily 0 is not neutral but the opposite extreme
(assortativity −0.949 against a null of 0.487, so race stays perfectly
recoverable with the sign flipped — the race-independent point is p≈0.30), and
node isolation becomes race-correlated (2.84% of White victims lose every
neighbour vs 0.03% of Black ones, making "has no neighbours" a race signal).
Even at the principled p≈0.30 the amplification stays ~1.25 rather than dropping
toward `temporal`'s 0.36. **So the residual is not edge-level homophily — it is
block membership.** Aggregating over city-mates transmits "which city", and city
correlates with race, no matter how the edges inside the block are paired.
That revises the §5-5 reading: `temporal` scored low not because it is
non-assortative but because it does not block on geography. Rewiring inside a
block cannot delete the block, which is what sent the mitigation effort to
`experiments/mitigate_loss.py`.

Edges are stored symmetrized, so `clear.fairgraph` folds to undirected pairs
(`src < dst`), drops there, and re-symmetrizes; dropping one direction only
would silently make message passing asymmetric.

**`experiments/edge_homophily.py` asks whether the edges themselves encode the sensitive
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

**Trainers dump test predictions** (`clear.predictions`) joined to
the unencoded sensitive attributes, so diagnosis reads CSVs and never
re-instantiates a model — `diagnose_fairness` re-runs in seconds against a GNN that took
minutes to train, and `mitigate_threshold` (mitigation, not yet written) can write mitigated
predictions in the same format to be diagnosed by the same code. `diagnose_fairness` with no
arguments diagnoses *every* dump it finds, which is what makes the flat-vs-graph
fairness comparison the default rather than an extra step.
`clear.predictions.assert_same_test_set` fails loudly if two dumps disagree on
`row_index`, since a silently mismatched test set would still produce a
plausible-looking comparison table.

**Dumps live in `outputs/predictions/`, one file per configuration.** They were
loose in `outputs/` until the mitigation sweeps started leaving one per sweep
point — 42 files / 79 MB, which buried the 9 small result CSVs (1.5 MB) that are
the actual experiment record. They are all gitignored and regenerable, so the
whole directory is ignored. Note the consequence for discovery: because every
sweep point leaves a dump, a no-argument `diagnose_fairness` run now diagnoses
all 42, not the ~7 model dumps the tracked `fairness_*.csv` were produced from.
Use `--models` to reproduce those tables.

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
was predicted in advance by `experiments/edge_homophily.py` — but it is a post-hoc floor
and all three floors stay in `outputs/fairness_gaps.csv` so the choice is
visible rather than buried.

`data/processed/` is gitignored, as are the raw CSV in `dataset/` (too large),
`outputs/*.png`, and all of `outputs/predictions/` (a few MB each, regenerable by
re-running `train_baseline`/`train_gnn`). The small result CSVs under `outputs/` **are**
tracked — they're the experiment record. The shared metrics ledger
is `outputs/results.csv` (every experiment writes there; `family`/`model`
columns distinguish rows). It supersedes `metrics.csv` and the three
`*_tradeoff.csv` files, whose pre-unification contents are preserved under
`outputs/legacy/`.

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
