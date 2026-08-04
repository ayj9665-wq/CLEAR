# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CLEAR (*Clearance Learning & Equity Assessment on gRaphs*) predicts whether a US
homicide case gets solved (`Crime Solved`), using the Murder Accountability
Project / Kaggle "Homicide Reports, 1980–2014" dataset (~638k rows). The project
is structured as one question — "what determines clearance?" — pursued in four
stages: **predict** (flat baseline → GraphSAGE) → **diagnose** (do clearance
rates/errors differ by victim race/sex?) → **prescribe** (how much can mitigation
techniques close that gap, and at what accuracy cost?) → **apply** (where does the
residual that mitigation did *not* repay concentrate?). All four are executed; the
findings below are the record.

Written docs with the full design rationale (Korean), in `reports/` — 18 files. The
long ones are **kept in sync with the code rather than frozen at their writing date**,
so a stale claim in one of them is a bug, not history:

- **`최종보고서_CLEAR.md`** — the whole four-stage narrative, the negative results, and
  the limits. Read this first; the rest are its depth.
- `문제정의서_CLEAR.md` (problem definition + ethics gate), `데이터카드_CLEAR.md`
  (data card), `EDA_보고서.md`, `AI모델_개발계획서_살인사건검거_GNN.md` (dev plan —
  graph/edge design, GNN architecture, 4-week roadmap).
- `공정성진단_보고서.md` (the diagnose+prescribe stage in full),
  `모델벤치마크_선행연구비교.md` (Campedelli 2022 comparison),
  `모델보고서_{GraphSAGE,XGBoost,LogReg}.md`.
- `확장설계서_CLEAR.md` (extension plan — **largely executed now**; its own status table
  at the top says which track is done, and tracks B/C/D1/D3 are the ones that are not).
- Six tracks were **pre-registered in their own plan doc before the code was written**,
  and each doc carries its result afterwards: `크로스피팅`, `카운티차분`,
  `짝지은표준화대조`, `SHR정황조인`, `로컬배포`, `포트폴리오웹페이지`
  (`*_계획서_CLEAR.md`). Follow that convention for the next intervention —
  pre-register, tag a rollback point, then execute.
- `ReturnA교차검증_계획서_CLEAR.md` is pre-registered and **not yet executed** — no code
  exists for it. It would bound how much of `ρ = +0.083` the measured race-differential
  recording (+4.1pp) can account for, by treating UCR Return A as an independent second
  measurement of the same agency's clearances: a disagreement between two measurements of
  the same police activity is by definition a recording problem, not discrimination or
  resourcing, so its size caps the contamination.

`README.md` is the outward-facing entry point (3-minute quick start + generated metric
tables). Its `<!-- METRICS:START/END -->` block is **generated**, not written — see
`experiments/dashboard.py --emit_readme_tables` below.

## Commands

Everything runs from `src/`. The numbered scripts at the root are the data
pipeline and **must run in order**; everything under `experiments/` reads the
prepared artifacts and has **no order among itself**, so those run as modules.

```bash
pip install -r requirements.txt
cd src

# --- scope (which sample everything below operates on) ----------------------
# Unset  = "ca_tx_mi" (California+Texas+Michigan, 190,326 rows) — every path below
#          is exactly as it has always been.
# Set    = "national" (no state filter, 638,454 rows) — sample/features/graph move
#          to data/processed/national/, dumps to outputs/predictions/national/,
#          result CSVs to outputs/national/. results.csv stays shared.
export CLEAR_SCOPE=national         # (or CLEAR_SCOPE=national python ... per command)

# --- data pipeline (ordered; each stage reads what the previous one wrote) ---
python 01_clean.py            # raw CSV -> data/processed/clean.parquet  (scope-independent)
python 02_sample.py           # -> data/processed/sample.parquet
python 03_features.py         # -> data/processed/features.parquet
python 04_build_graph.py      # -> data/processed/graph/edges_{geo,temporal,weapon}_k{k}.npy (k=config.K_NEIGHBORS)
python 04_build_graph.py --k 20     # rebuild at a different degree cap (needed before --k_neighbors 20 below works)
python 04_build_graph.py --rank onehot --control --candidates geo
                                    # -> edges_geo_k20_rank_onehot[_control].npy (similarity-ranked graph + degree-matched control; REJECTED, see Scope)

# --- experiments (unordered; all write outputs/results.csv) ------------------
python -m experiments.train_gnn           # -> results.csv (family=train; geo only, one row per config.GNN_SEEDS seed)
python -m experiments.train_gnn --edge_type all                                 # train geo+temporal+weapon for comparison
python -m experiments.train_gnn --edge_type geo --hidden_dim 128 --tag my_sweep # any hyperparam overridable via CLI
python -m experiments.train_gnn --edge_type geo --k_neighbors 20                # use a --k 20 graph built above
python -m experiments.train_gnn --edge_type geo --seeds 42,43,44                # torch seeds to repeat over (split stays fixed)
python -m experiments.train_gnn --edge_type geo --blind --minibatch --tag blind_mb
                                    # NeighborLoader mini-batch (required at national scale;
                                    # dumps to graphsage_geo_blind_mb.csv, NOT comparable to
                                    # full-batch rows — see the bridge-run note below)
python -m experiments.train_gnn --edge_type geo_temporal                        # union of geo+temporal edges
python -m experiments.train_gnn --blind        # race/sex/ethnicity dummies dropped from X (graph unchanged)
python -m experiments.train_gnn --blind --minibatch --save_model
                                    # also -> outputs[/{scope}]/models/
                                    #         graphsage_{edge}[_mode][_blind][_mb]_seed{N}.pt
                                    #         (same naming as the prediction dumps)
                                    # (weights + hyper + FEATURE COLUMN ORDER + blind/scope/
                                    #  edge_type/k + calibrated:False). gitignored on purpose —
                                    # see the deployment-layer section below

python -m experiments.diagnose_fairness   # diagnoses EVERY dump in outputs/predictions/ -> fairness_{group_metrics,gaps,model_contrasts,gaps_standardized}.csv
python -m experiments.diagnose_fairness --models graphsage_geo xgboost   # restrict to specific dumps
python -m experiments.diagnose_fairness --min_n 5000 --n_boot 2000       # stricter group floor / more bootstrap reps
python -m experiments.diagnose_fairness --stratum State --stratum_min_n 100
                                          # also writes fairness_gaps_standardized.csv (default;
                                          # --stratum none skips). Pooled gaps carry a
                                          # region-composition confound — see below.
python -m experiments.diagnose_fairness --stratum State,City --stratum_min_n 20 30 50 \
       --min_n 5000 --out_suffix _county --models graphsage_geo_blind_mb graphsage_fairloss_a100_mb
                                          # county stratum. `State,City` is mandatory (county
                                          # names collide across states); --out_suffix is
                                          # mandatory for ANY non-default run — this script
                                          # replaces the tracked fairness_*.csv wholesale.
                                          # Any --stratum run also writes a 5th table,
                                          # fairness_model_contrasts_standardized*.csv:
                                          # PAIRED model differences on the stratum axis.
                                          # Read `quantity="signed_gap"`, not amplification.
python -m experiments.edge_homophily      # -> outputs/edge_homophily.csv (are edges a sensitive-attribute proxy?)
python -m experiments.edge_relatedness    # -> outputs/edge_relatedness.csv (do edges link genuinely related cases?)
python -m experiments.edge_relatedness --no_oracle                  # skip the O(n^2)-per-block oracle
python -m experiments.edge_relatedness --oracle_cap 3000            # larger per-block sample for the oracle
python -m experiments.edge_relatedness --similarity onehot onehot_idf ordinal ordinal_idf
                                          # -> also outputs/edge_relatedness_similarity.csv (encoding bake-off)

python -m experiments.mitigate_graph      # -> results.csv (family=mitigate_graph; drop same-race edges + matched random control)
python -m experiments.mitigate_loss       # -> results.csv (family=mitigate_loss; fairness penalty in the training loss)
python -m experiments.mitigate_loss --beta 1.0                              # early-stop on val MCC - beta*val gap instead of val MCC alone
python -m experiments.mitigate_loss --alphas 0 25 50 75 --seeds 42,43,44,45 # finer alpha grid near the useful range

python -m experiments.detect_cold_blocks  # -> outputs/cold_blocks.csv (blocks with unexplained excess unsolved; family=cold_blocks)
python -m experiments.detect_cold_blocks --blocks county hargrove --min_n 50
python -m experiments.detect_cold_blocks --min_n 20 50 100   # re-runs the test at each floor
                                    # (BH q depends on how many blocks were tested together,
                                    #  so this is NOT the same as filtering one run by n)
python -m experiments.detect_cold_blocks --model graphsage_geo_blind --calibrate none   # diagnostic only, see below
python -m experiments.detect_cold_blocks --out cold_blocks_cv5.csv   # separate file for a different split (cross-fitting)

python -m experiments.crossfit_predictions --minibatch   # 5-fold out-of-fold p̂ for EVERY row
                                    # -> predictions/national/graphsage_fairloss_a100_mb_cv5.csv (638,454 rows)
                                    # ~76 min at national scope; alpha/blind/minibatch fixed to the a100 gate value
python -m experiments.crossfit_compare    # -> outputs/national/crossfit_compare.csv (the §2-1 judgment)

python -m experiments.county_race_residual   # -> outputs[/{scope}]/county_race_residual{,_summary}.csv
                                    # (county x race) O/E residual; rho = log(SMR_Black/SMR_White)
                                    # holds the county fixed, so any channel acting as an
                                    # additive county intercept cancels. Reads the cv5 dump.
python -m experiments.county_race_residual --min_race_n 10 20 30 --haldane off
                                    # per-race floor sweep / drop the +0.5 log correction

python -m experiments.audit_resources     # -> outputs[/{scope}]/resource_audit.csv
                                    # joins LEOKA agency-year officer counts onto the county
                                    # residual; needs dataset/geo/LEOKA_parquet_1960_2024_year/
python -m experiments.audit_resources --src cold_blocks.csv --min_n 20   # test-split table instead
python -m experiments.shr_circumstance    # -> outputs[/{scope}]/shr_recording_{summary,by_county,by_circumstance}.csv
                                    # SHR circumstance recording: is it downstream of the
                                    # outcome, and is it race-differential within county?
                                    # Needs dataset/geo/shr_1976_2016_csv/ (openICPSR 100699).
                                    # Reads the RAW csv for Agency Code, like audit_resources.
python -m experiments.shr_circumstance --min_race_n 10 20 30 --min_unsolved 50

python -m experiments.audit_resources --target rho    # -> resource_audit_rho.csv
                                    # regress the WITHIN-county race contrast instead of z.
                                    # `target` rides in params only when rho, so the existing
                                    # 42 z rows in results.csv keep their identity KEY

python -m experiments.audit_priority      # -> outputs[/{scope}]/priority_audit{.csv,_lift.png}
                                    # D3 priority-list fairness audit; defaults to the
                                    # a0-vs-a100 contrast on the shared test split
python -m experiments.audit_priority --models graphsage_fairloss_a100_mb_cv5 --qs 0.001 0.005 0.01
python -m experiments.audit_priority --attrs "Victim Race" --min_n 2000 --no_figure

python -m experiments.build_web_map       # -> outputs[/{scope}]/web/map_{scope}.html
                                    # self-contained, zero external requests; needs the
                                    # three min_n levels above for the sample-floor slider
python -m experiments.build_web_map --simplify_km 1.5   # payload knob (1.0km=468KB paths, 2.0km=399KB)
# the two national maps. --src picks the split and drives the caption; --out must move with it.
python -m experiments.build_web_map --src cold_blocks_cv5.csv --out map_national.html --simplify_km 1.5
                                    # PRIMARY: cross-fit, 1,800 counties carry data
python -m experiments.build_web_map --src cold_blocks.csv --out map_national_test.html --simplify_km 1.5
                                    # reference: test split, 850 counties, same split as fairness_*.csv

python -m experiments.map_figures         # -> outputs/map_fig{1,2}_*.png (county choropleth of the residuals + the race panel)
python -m experiments.poster_figures      # -> outputs/poster_fig{1,2}_*.png (reads result CSVs only, no retraining)

# --- deployment layer (reads COMMITTED result CSVs only; no torch, no dataset) ---
pip install -r requirements-dashboard.txt   # pandas + numpy, nothing else
python -m experiments.dashboard           # -> outputs/dashboard.html + opens the browser
python -m experiments.dashboard --no_open           # build only (CI / remote)
python -m experiments.dashboard --scope national    # one scope instead of both tabs
python -m experiments.dashboard --verify_clone      # rebuild inside a temp tree holding ONLY
                                          # git-tracked files — the clone-and-run contract
python -m experiments.dashboard --emit_readme_tables  # rewrites README's METRICS block from results.csv

CLEAR_SCOPE=national python -m experiments.build_story_page
                                          # -> outputs/national/web/clear_story.html
                                          # scroll-narrative portfolio page; defaults
                                          # --src cold_blocks_cv5.csv --simplify_km 1.5

npm install                               # once, at the repo root: jsdom (dev only)
CLEAR_SCOPE=national python -m experiments.verify_html
                                          # runs the built HTML in jsdom, drives every
                                          # control, and cross-checks screen counts
                                          # against cold_blocks*.csv. NOT part of any
                                          # build — the dashboard must stay buildable
                                          # on pandas+numpy alone
python -m experiments.verify_html --only map        # one kind
python -m experiments.verify_html --selftest        # prove the checker catches the
                                          # original ReferenceError, then exit
```

**Tests: `pytest` from the repo root** (39 tests, ~1.4 s, `tests/`). There is no build or
lint step. The suite is deliberately narrow — it does not check performance or
conclusions, it checks that **this repo's specific silent-wrong failures stay loud**:
ledger identity (`params` is identity, `notes` is not — the 78-duplicate-row bug),
`from_run` dropping `None` knobs, `--blind`'s no-op assertions, the seven-metric
definitions (especially specificity-as-negative-recall, whose inversion would flip every
mitigation claim), calibration being total-matching *and* rank-preserving, county
canonicalization merging renames while **not** merging independent cities into their
surrounding counties, `assert_one_row_per_fips` raising, the `flag == ""` vs `NaN` trap,
and `_htmlcheck` actually failing on broken JS. Tests requiring `data/processed/` skip
rather than fail, because a clone legitimately has no data — the same rule the dashboard's
contract runs on. Both fixture directions were mutation-checked: emptying `ALIASES` or
neutering the FIPS guard makes the corresponding tests fail.

The HTML builders
(`build_web_map`, `build_story_page`, `dashboard`) each carry their own build-time
self-check instead — external-request scan, size budget, data invariants, and
`_htmlcheck.node_check` (inline-JS syntax via `node --check`, skipped with a notice when
`node` is absent). **None of it covers DOM behaviour**, which is where the one shipped JS
defect lived — see the deployment-layer section.

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

**Scope is an axis, because the national extension re-runs the whole pipeline.**
`CLEAR_SCOPE` (`ca_tx_mi` default, `national`) picks the sample, and
`config.SAMPLE_STATES` is *derived* from it — one switch, so the state filter and
the output paths cannot drift apart. Three things follow, and the reasons are not
interchangeable:

- **`C.SCOPE_DIR`** holds `sample.parquet`/`features.parquet`/`graph/`; **`C.scoped_output(name)`**
  holds the result CSVs and figures; **`predictions_dir()`** holds the dumps. In the
  default scope all three resolve to exactly the old paths, so nothing moved and
  there was no migration.
- **`clean.parquet` stays outside the scope.** `01_clean.py` has no state filter, so
  its output is already national and both scopes share it. Scoping it would make a
  national run look for a file that is never written.
- **`results.csv` is *not* split by scope.** It's long format and `scope` rides in
  `params`, so one file separates by groupby — the "a new metric is new rows, not a
  new column" rule applied to samples.

The dump directory split is the load-bearing part, not tidiness. `row_index` in a
prediction dump is a **row position into `features.parquet`**, and `diagnose_fairness`
`iloc`s it into `load_sensitive()`. Point that at a different sample and the sensitive
attributes join to the wrong rows while still producing a plausible table —
`assert_same_test_set` compares dumps to each other, so it passes. That would silently
void the committed flat-model dumps the headline
`graphsage_geo_blind − xgboost_blind = +0.82` rests on. Directory isolation is the
primary guard (`discover()` doesn't recurse, and the legacy `outputs/predictions_*.csv`
fallback is gated to the default scope); `assert_same_test_set` additionally rejects a
dump whose max `row_index` exceeds the current scope's row count, which catches files
moved by hand.

`config.scope_param()` returns `{}` in the default scope and `clear.results.rows()`
merges it into `params` — **`None`-by-default for the same reason `GNN_EDGE_MODE` is**:
existing rows' `params` JSON stays byte-identical, so their identity KEY holds and a
re-run still *replaces* instead of duplicating. Injecting it in `rows()` rather than at
each call site is deliberate — `rows()` is the one funnel every family passes through,
and a missed call site would let a national run overwrite three-state rows in a tracked
file.

**`--minibatch` (NeighborLoader) is the other half, and it is a separate axis, not a
speedup.** National `geo` is 25.0M directed edges — ~20 GB for 2 layers plus backward,
against 12.9 GB, so full-batch is out. Training samples `config.GNN_NUM_NEIGHBORS`
(`[25, 10]`) per layer, but **val/test inference uses every neighbour**: the dumped
`proba` feeds every fairness gap, `detect_cold_blocks`'s `E_b`, and the map's colours,
so sampling noise there would propagate into all of it. The three loaders are built once
per `train_one` — building the val loader per epoch rebuilds the CSC index every time and
walked resident memory from 1.8 GB to 5.3 GB on three states alone. Dumps and tags get an
`_mb` suffix (`graphsage_geo_blind_mb.csv`), applied in `train_gnn` and in
`sweep.run_point` so both mitigation scripts inherit it; without it a bridge run
overwrites the very full-batch dump it is meant to be compared against. Needs `pyg-lib`
(PyG's own wheel index — `torch-sparse` has no wheel for win/torch2.6+cu124/cp313), but
only when `--minibatch` is used: the loader import sits inside the function.

**The bridge run says accuracy carries over and fairness does not.** Three-state blind
`geo`, 3 seeds each, mini-batch minus full-batch: AUC +0.00007 (0.06× the full-batch seed
std), MCC +0.00069 (1.45×), Balanced Accuracy −0.00253 (3.90×). Small — and consistent
with the graph's value here being *an unbiased sample of the block's feature
distribution*, which sampling 25 of ~40 neighbours preserves. But the **operating point
moves**: predicted clearance 53.2% → 60.7% (actual 68.2%), and race amplification
**1.48 [1.24, 1.78] → 1.26 [1.04, 1.54], paired difference +0.23 [+0.13, +0.34]**, CI
excluding zero. Every fairness number here is threshold-dependent, so **national results
compare only against three-state *mini-batch* values**. The full-batch 1.48 is a number
on a different axis.

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
`clear.similarity` (within-block case-similarity feature maps — the ranking criterion
candidates; used by `edge_relatedness`'s oracle and, once built, by the ranked-graph
mode of `04_build_graph.py`),
`clear.sweep` (the mitigation sweep harness `mitigate_graph`/`mitigate_loss`
share), `clear.counties` (FIPS join, equal-area projection, and the OKLab
lightness-matched colour arms — shared by `map_figures`, `build_web_map` and
`build_story_page`). The experiment scripts are thin CLIs over `clear/`; `config.py` holds
all paths/constants. The package originally existed because of the import
constraint above; it stays because its modules genuinely have multiple callers
(`clear.results` 14, `clear.predictions` 10, `clear.gnn` 10, `clear.data` 9,
`clear.fairness` 6, `clear.graph` 6, `clear.metrics` 5). The rule for what belongs there
is **two or more callers** —
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
                                           outputs/fairness_{group_metrics,gaps,model_contrasts,gaps_standardized}.csv)
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

**That "no finer similarity signal to rank by" premise has now been tested, and it
is false** (`experiments/edge_relatedness.py`, `outputs/edge_relatedness.csv`). The
test uses the perpetrator columns as an *external label*: they are excluded from
model input as leakage, so they have never touched training and are legitimate for
evaluation. Restricting to edges whose endpoints are both solved with a known
perpetrator profile, the measurement splits relatedness into what the blocking key
contributes (`block_lift`) versus what the pairing inside the block contributes
(`edge_lift`). Three results:

- **`edge_lift` is 0 across all 9 (candidate × profile-definition) cells** (max
  |0.0005|), confirming the shuffle-ring is an unbiased within-block random pairing.
  So **per-edge importance is structurally meaningless here — the unit of any edge
  analysis must be the block or the edge attribute, never the individual edge.**
  This is why GAT attention, if added, must be read aggregated by edge attribute.
- **`block_lift` orders weapon > geo > temporal** consistently: `geo` is substantial
  (+0.0768 on perpetrator sex×race, +18.6% relative to its 0.4129 null), `temporal`
  is ~0 (+0.0015). `weapon`'s +0.2200 is a construction artifact, not merit —
  `WEAPON_BLOCK_COLS` blocks on victim sex/race and homicide is largely intraracial,
  so victim race predicts perpetrator race. Same artifact as its homophily of 1.000.
- **A blind similarity ranking inside the block beats random pairing everywhere**
  (`oracle_lift_blind` > 0 in all 9 cells). For `geo`/relationship it is +0.0406,
  *larger than that block's own* `block_lift` of +0.0304. So ranking within blocks is
  a real available improvement — but **most of the apparent gain is sensitive-attribute
  driven**: sighted +0.1163 vs blind +0.0186 on `geo`, a 6× gap. A naive
  similarity-ranked rebuild would widen exactly the race-homophily leak this project
  diagnosed. Any upgrade must rank on blind features, at ~1/6 the sighted gain.

Unexpectedly, `temporal` has the *highest* `oracle_lift_blind` (+0.0452/+0.0524)
despite the lowest `block_lift` — its blocks are large and heterogeneous, so ranking
has the most room. The right reading is not "time carries no relatedness" but
"Year+Month blocking with random pairing wastes it."

Caveats: only solved–solved edges are checkable (unsolved cases have no perpetrator
record); `oracle_lift` carries no CI, though it is stable across `--oracle_cap`
1000/2000/3000 (blind geo 0.0186/0.0186/0.0191) and the sighted figures rise with
cap, so those are lower bounds.

**What that oracle actually ranks by is a 6-level step function, and better encodings
mostly lose** (`clear/similarity.py`, `outputs/edge_relatedness_similarity.csv`).
Because `03_features.py` one-hots without `drop_first` and keeps `Unknown` as its own
category, every categorical field contributes exactly one 1 per row, so row L2 norms
are near-constant (√6–√7) and **cosine similarity reduces to "fraction of fields that
match"**. Inside a `geo` block `State` is constant, leaving 5 varying fields — so the
similarity takes only ~6 distinct values. It shows: in the largest block (LA, 44,511
rows) the median group of byte-identical feature rows is 2 but **50.2% of nodes could
fill all k=20 neighbours from exact duplicates alone**, and only 34,189 of 190,326
rows (18.0%) have a distinct blind feature vector. So `oracle_lift` measures the value
of *finer exact-match blocking*, not of graded ranking.

Three encodings were tried against that baseline via the same oracle (35 s, no
training): IDF column weighting (`√(−log p_c)`, so a `Poison` match outweighs a
`Handgun` match), ordinal/cyclic embedding (Age Group·Decade·Victim Count onto a
quarter-circle, Month onto a full circle — both 2 columns with unit norm, so a dot
product gives `cos` of the gap and stays a single BLAS call), and their combination.
Results: **IDF loses in 9 cells out of 9** (mean −0.0044) — rare-category matches are
rare precisely because they are noisy. **Ordinal wins only on `temporal`** (3/3,
+0.0066/+0.0007/+0.0045) and **loses on `geo`** (0.0207→0.0152 on the finest profile).
`geo` blocks are already geographically homogeneous, so what is left to detect is
exact co-occurrence, not proximity: grading age lets a 25–29 case partially match a
30–34 case, and that dilutes rather than helps. The 6-level step function was the
right representation there. Decision: **`geo` ranks on `onehot`, `temporal` on
`ordinal`** — the scheme is a per-candidate parameter, not a global one. Scale: the
best gain over `onehot` is +0.0066 (15% relative), so **encoding is a second-order
effect; the first-order question is ranking-vs-shuffle at all.**

**Building the ranked graph and training on it settles the question: ranking is worse,
and why is the useful part.** `04_build_graph.py --rank {onehot,ordinal} [--control]`
builds it (geo in 42 s, CPU — `04` stays torch-free); `clear.similarity.block_topk` is
shared with the oracle so the measured gain and the built graph cannot drift apart.
Ranking works exactly as designed — `edge_relatedness --edge_mode rank_onehot` shows
`edge_lift` rising from ~0 to **+0.0155/+0.0205/+0.0393** across the three profile
definitions, matching the oracle's predictions (+0.0186/+0.0207/+0.0406) to within
0.003. But on blind geo over 3 seeds it **loses on every claimable metric**: MCC 0.2574
vs 0.2628 for its degree-matched control and 0.2659 for shuffle — **−0.0054 against the
control, 9.0× the seed std** (AUC −0.0056, 6.2×; Balanced Accuracy −0.0030, 10.0×).
Race assortativity also *rises*, 0.1739 → 0.1938 (control 0.1629), despite ranking on
blind features only. Worse on both axes, so not even a trade-off. Pre-registered rule
says reject.

The mechanism is measured, not guessed: the cosine between a node's **neighbourhood-mean
feature vector and its own** goes 0.6432 (shuffle) → **0.9477** (ranked), with the
control at 0.6373. Aggregating ranked neighbours hands back the node itself, so message
passing carries no new information and the GNN degenerates toward an MLP.

> **The graph's value here is contextual aggregation, not relatedness.** Twenty random
> neighbours inside a block are an unbiased sample of that block's feature distribution
> — an estimate of "what does this county look like" — and that is the signal. Ranking
> replaces the estimate with a copy of the node and erases it.

This retroactively explains two earlier findings: **`geo` beats `temporal`** because a
county's composition predicts clearance (state clearance ranges 34%–93%) while
"March 1993 across three states" does not; and **`aggr="max"` is clearly worse than
`mean`** because max picks one extreme neighbour where mean summarises the distribution.
It also yields a **pre-registered prediction for the model-swap track: GATv2 should
underperform mean-aggregation GraphSAGE**, since attention concentrates weight on
similar neighbours — the exact direction just measured to hurt.

The degree-matched control was load-bearing, not ceremony. Shuffle-ring is essentially
regular (geo degree std 2.63, temporal exactly 40 for every node) while ranking makes
hubs (geo std 22.06, max degree 300) and drops 24% of edges through mutual-top-k dedup —
so `rank` vs `shuffle` differs in ranking, degree distribution *and* density at once.
Only `rank` vs `rank_control` isolates ranking. The plumbing (`--rank`, `--control`,
`--edge_mode` on `train_gnn`/`edge_relatedness`/`edge_homophily`) is kept: block-size
distributions change completely at national scale, and the control arm plus the
neighbourhood-redundancy measure apply to any future graph intervention.

The `mode` axis is deliberately separate from `edge_type`: `build_edge_index` splits
`edge_type` on `_` to form unions (`geo_temporal`), so a `geo_rank` edge type would
parse as "geo ∪ rank". Mode lives in the filename suffix instead
(`edges_geo_k20_rank_onehot.npy`). `config.GNN_EDGE_MODE` defaults to **`None`, not the
string `"shuffle"`** — `RUN_PARAMS` carries `edge_mode` and `from_run` drops `None`
keys, so every pre-existing row's `params` JSON is byte-identical and re-running still
*replaces* instead of duplicating.

Normalization rule in `clear.similarity`: rank by **dot product, not cosine**, whenever
weights are non-uniform — L2 would penalise exactly the rows carrying rare (informative)
categories. Holding the query row fixed makes its own norm a constant across candidates,
so the dot product ranks by `Σ_c w_c·1[match]` directly. Only `onehot` keeps cosine, to
reproduce the already-published table. The same rule means **within-block constant
fields need no removal** — they add the same value to every candidate, leaving argsort
unchanged.

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
which is the diagnosis the project exists to make.

**All of those are *pooled* gaps, and at national scale that is not a neutral choice** —
they mix group composition with regional effects, which is what the sex-amplification
confound above turned out to be. `standardized_gap_row` therefore reports the same gaps
under **direct standardization** by `State`: rates within a stratum, averaged by stratum
weight, then max−min across groups. It reuses the pooled cell machinery (`_rates`, the
per-stratum multinomial bootstrap) precisely so the two numbers cannot come to mean
different things. Two rules: only strata where **every** compared group clears
`--stratum_min_n` are used (otherwise a stratum contributes differently per group and
"standardized to one population" breaks), and per-metric weights are renormalized over
strata where the rate is defined (a stratum with no negatives has no FPR; summing the NaN
would wipe the metric and zeroing it would invent an observation).

Gaps are computed over
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
**At national scope `geo`'s race assortativity *rises* to 0.2308 and the recovery
lift to +18.9pp** (`temporal` stays ~0 at 0.0033, `weapon` stays 1.000): the same
county blocking spans the whole country's segregation range instead of three
states', so the proxy path this project diagnosed gets *stronger* with scale, not
weaker. Pre-registered consequence — the national blind `geo` run should amplify
*above* the three-state 1.48.
Read against the sighted-model diagnosis alone this looks *backwards* — the
significant GraphSAGE-over-XGBoost amplification was on *sex* (where geo
carries no signal) and unresolvable on *race* (where it does). The confound
is that race was in `X` all along (next note), so the graph's copy of it was
redundant and invisible. `--blind` closes the direct path, and then the
homophily measurement predicts the outcome correctly — see the blind-run
findings below. Keep that ordering in mind before treating this table as a
null result.

**`experiments/detect_cold_blocks.py` is the "apply" stage: which case groupings went
unsolved more than the model expected.** It is the defensible form of the "serial
suspicion" idea — the dataset has no perpetrator ID, so *"these N cases share an
offender"* is unverifiable in principle; what MAP's Hargrove algorithm actually does is
**cluster-level anomaly detection**, and that this data supports. Per block:
`O_b = Σ(1−y)`, `E_b = Σ(1−p̂)`, `SMR = O/E`, with two-sided p-values from a parametric
bootstrap (`Σ Bernoulli(1−p̂)` simulated directly, since the normal approximation is
poor for small or extreme-p blocks) and **BH FDR** across the hundreds of blocks. It is
indirect standardization, the epidemiological SMR construction.

Two things had to be fixed before the numbers meant anything, and both generalize:

- **p̂ is not a calibrated probability.** `clear.gnn` trains with
  `BCEWithLogitsLoss(pos_weight=neg/pos)`, so predictions are correctly *ranked* but
  systematically shifted: on the test set the expected unsolved count exceeds the
  observed by **+47.9%** (a50), +53.8% (`geo_blind`), +49.4% (`geo`). Used raw, nearly
  every block reads "fewer unsolved than expected" and the test measures global
  miscalibration rather than local anomaly. The fix is a single logit shift δ chosen so
  `Σσ(logit p̂ + δ) = Σy` (δ = +0.7141 here, 30,253 → 38,953). One parameter, monotone,
  so **rankings are untouched** and the priority list is unaffected; the global residual
  becomes zero by construction, which is exactly what indirect standardization means.
  **Any future use of these dumps as probabilities rather than scores needs this.**
- **A blind, fairness-penalized model does not "remove race" from the expectation — it
  does the opposite, and that is the point.** Because the model never sees race, race-
  linked disparity is *not* absorbed into `E_b` and stays in the residual. A sighted
  model would bake "this victim is Black, so expect it to go unsolved" into the
  baseline and launder the disparity into normality — hiding the very thing the project
  measures. Even blind-but-unmitigated is unusable here (FPR amplification 1.82); only
  the α=50 model (FPR amplification 0.27) gives a **race-neutral baseline**. So the
  correct claim is not "concentration unexplained by race" but *"concentration
  unexplained by case mix, measured against a race-neutral baseline."*

Findings (test set, `min_n=20`): `county` blocking gives 143 blocks, **5 cold / 15 warm**
at FDR 5%; `hargrove` (State+City+Weapon+Victim Sex) gives 331 blocks, 16 cold / 19 warm.
Top counties are Dallas (SMR 1.29, z 11.4), Wayne/Detroit (1.17, z 9.1), San Francisco
(1.22), Genesee/Flint (1.29), Alameda/Oakland (1.08) — known low-clearance urban
counties, so face validity holds. The centre of the result is that **the residual
correlates with racial composition**: z ↔ `black_share` = **+0.292** (county) / +0.322
(hargrove), with cold blocks averaging 0.623 Black share against warm blocks' 0.221 and
an overall 0.227. Since the baseline is race-neutral, that correlation is a finding, not
a nuisance — it is the project's "who gets forgotten" question expressed geographically.
It is not exclusively a race story (Texas Cameron, SMR 1.90, is 0.000 Black share).

**The load-bearing caveat: the model does not know what county a case is in.** `City` is
a blocking key, not a feature — it is absent from `config.CATEGORICAL_COLS` and enters
only indirectly through the graph. So `E_b` carries almost no county-specific effect and
**z is effectively "how far does this county deviate from a state-and-case-mix
baseline"**. Case mix, discrimination, investigative resourcing, and recording practice
are **not separable** in this design. *Urbanicity is not a fifth channel but their common
cause* — it is a container for witness cooperation, caseload per agency, agency size and
stranger-homicide share, and it is simultaneously a mediator of the discrimination path
through residential segregation, so entering it as a regressor makes the race coefficient
uninterpretable in a new way rather than isolating anything; it is a stratifier only
(`Agency Type` partly proxies it). The z-vs-race correlation must not be read causally,
and that sentence has to travel with any map built on this table.

**The national run is done, and it reproduces the mechanism at scale.** 638,454 rows /
51 states / 123 feature columns; `geo` gives exactly the 3,042 blocks the design doc
predicted and 25.0M directed edges. Race assortativity *rises* with scale (0.174 →
0.2308), and blind amplification rises with it — **1.805 [1.715, 1.904]** against the
three-state mini-batch 1.26, the lower bound clear of that CI's upper bound. This was
pre-registered before the run. The alpha re-search picks **α=100** on the pre-registered
gate (FPR amplification CI upper bound ≤ 0.5): α=25 and α=50 have point estimates under
0.5 but upper bounds of 0.572 and 0.539, so they fail; α=100 gives 0.261 [0.157, 0.367]
for −0.0156 MCC. Every `detect_cold_blocks` / map artifact at national scope runs on
`graphsage_fairloss_a100_mb`. Two things did *not* carry over: the alpha curve does not
invert in 0..100 (three states inverted above 50 — the mini-batch effective range shifted
down, 47 optimizer steps per epoch against one), and **sex amplification appeared not to
collapse under blinding** (1.25 [1.17, 1.33] nationally vs 0.88–0.96 on three states)
even though geo's sex assortativity stays low at 0.0336.

**That last one was a confound, not a mechanism, and finding it changed how gaps are
measured.** Five hypotheses were eliminated from existing dumps alone (no retraining):
the denominator is unchanged (base-rate gap 0.0803 → 0.0826, so the *numerator* rose 33%);
the operating point accounts for about a quarter (at matched predicted-positive rate the
national gap is still +0.0254 larger); the feature-proxy channel did **not** strengthen
(blind X → sex recovery AUC 0.6921 → 0.6975, and 0.6925 without the State dummies —
whereas race goes 0.7001 → 0.7770, which is why race amplification *did* rise); edge
homophily cannot be it because **in the blind condition neighbours carry no sex either**;
and decisively, **scoring the national model on CA+TX+MI rows only gives 0.965**, the
three-state value. It is between-state aggregation: `State` is not a sensitive attribute,
so it stays in blind `X` as 51 dummies, and states differ in both clearance level and
victim-sex composition. Signed decomposition — the model's gap draws **32% from
between-state composition while the base-rate gap draws only 14%**, and the ratio keeps
the asymmetry.

So `clear.fairness` gained **direct standardization** (`standardized_gap_row`, wired as
`diagnose_fairness --stratum State` → a fourth table `fairness_gaps_standardized.csv`).
Sex goes 1.246 → **1.005 [0.925, 1.090]** nationally (three states 0.964 → 0.832), while
**race survives: 1.805 → 1.558 [1.457, 1.674]** (three states 1.483 → 1.417, against blind
XGBoost's 0.668 → 0.644). Sex disappears under standardization and race does not — so the
"sex is the direct column, race is the graph" split needs no narrowing to three states;
the confound correction strengthens it. It stays a **separate file**: adding columns to
`fairness_gaps.csv` would change what the reports citing it point at, and adding rows would
let anyone reading it unfiltered mix pooled with standardized. Standardization drops strata
below the per-group floor (39/51 states for race, 44/51 for sex) and provides no *paired*
model contrasts, so model comparisons still come from the pooled contrast table.

**Consequence for the alpha gate, and it does not resolve by turning the knob further.**
The gate used *pooled* FPR amplification; standardized, α=100 is **0.590 [0.456, 0.738]**
rather than 0.261 [0.157, 0.367], and no alpha in the grid passes. The penalty is defined on
the pooled gap, so it is not even monotone in the standardized metric (α=25/50/100 give
0.639/0.701/0.590) — pressing harder is the wrong instrument, and α>100 is where three
states documented optimization instability. The principled fix is a **stratified penalty**
(`fairness_penalty` over within-stratum group variance), which is a new intervention needing
its own pre-registration, not a re-tune. The bias direction is not adverse: α=100's
within-state race gap is still negative, so `E_b` is overstated in high-Black-share counties
and **`z ↔ black_share = +0.290` is a lower bound**.

National cold blocks (test set, α=100, δ=+0.8170): 853 county blocks at n≥20, 38 cold /
52 warm. Top: Fulton/Atlanta (SMR 1.55, z 15.66), Baltimore city, St. Louis city,
Orleans, Richmond, DC, Wayne, Cook — far stronger face validity than three states could
show. `z ↔ black_share` strengthens to **+0.381** (+0.546 at n≥100), cold blocks
averaging 0.680 Black share against warm 0.233; still not exclusively a race story
(2 of 38 cold counties are under 0.30). The §5-4-1 caveat travels with the map: the model
has no county feature, so z carries the whole county effect and case mix / discrimination
/ resourcing / recording practice are not separable — **but which *layer* the race
association lives in is now measured, see the county-differencing section below.**

**Cross-fitting doubled that coverage and cut the correlation, exactly as pre-registered.**
`experiments/crossfit_predictions.py` gives every one of the 638,454 rows an out-of-fold
p̂ — 5 folds, transductive (the graph stays whole; only the loss moves to the fold's learn
nodes, which is the regime `clear.gnn` already runs), α=100/blind/minibatch held fixed,
76 min. Blocks at n≥20 go 853 → **1,803** (28% → 59% of counties) and `z ↔ black_share`
drops **+0.381 → +0.290 [+0.245, +0.331]**, CI excluding the old point estimate at all
three floors. **The matched control is what makes that attributable to coverage**: three
things moved at once (block set, split, 3-seed mean → 1 seed), so `crossfit_compare`
recomputes the cv5 correlation on the *same 853 counties* — it lands at +0.413/+0.500/+0.556,
i.e. the split/seed effect is ≈0 (CI covers 0 at n≥50 and n≥100) and the whole −0.11
belongs to the 950 newly admitted small counties (median n=34, mean Black share 0.200,
internal correlation only +0.158). So the claim is **limited, not withdrawn**: the
association is clearly positive nationwide, but its strength scales with county size, and
+0.381 is a large-county-weighted estimate rather than a national average. Two by-products:
warm blocks grew 5.3× against cold's 1.95× (power reveals "solves better than expected"
counties faster), and fold MCC 0.2969 ± 0.0041 matches the test-split a100's 0.2979 — 14%
more training nodes bought no accuracy, so learning is saturated at this sample size.

Three plumbing facts follow from that dump being **full-sample rather than test-set**.
`predictions.discover()` skips `_cv\d+$` labels **in auto-discovery only** (an all-rows dump
makes no-arg `diagnose_fairness` fail `assert_same_test_set`, legitimately — model contrasts
are only meaningful on a shared test set; `--models` still reads it). `detect_cold_blocks
--out` keeps the cv5 table in its own file, because `cold_blocks.csv` is the test-split
table the map and reports cite. And the fairness-penalty group set must be counted on the
**canonical test split**, not on all rows: nationally Asian/PI is 2,940 in test but 9,890
overall, so re-deriving it naively presses 3 groups instead of 2 and silently makes α=100 a
different intervention. The verification gate also had to change — p̂ is deliberately
uncalibrated (`pos_weight`), so out-of-fold clearance is checked against the test-split
model's own 0.5328, never against the actual 0.7020.

**`experiments/audit_resources.py` measures one of `detect_cold_blocks`'s four confounded
channels from outside the model.** LEOKA agency-year sworn-officer counts (Kaplan's concatenated
files, openICPSR 102180) join onto the raw `Agency Code` directly — **(ORI, year) exact join is
100.0%** (638,452/638,454), every year 1980–2014 at 100%. `Agency Code` is dropped by
`01_clean.py`, so this reads the raw CSV; that is safe because the result is only ever used at
`(State, City)` county granularity, never per row. LEAIC (ICPSR 35158) is **not** used: no
ORI→county mapping is needed (`City` already is the county), and its county column has a real
error — `PAPEP00` (Philadelphia PD) is assigned Indiana County (42063), which would silently
move 12,848 homicides to a rural county.

**Normalize per homicide, not per capita.** Per-capita is wrong twice: within a county the
municipal/sheriff/state jurisdictional populations overlap so the denominator double-counts
(officers don't — each is employed by exactly one agency), and more importantly cold counties
have *more* officers per capita (2.20 vs warm 1.64) while carrying 1.75× the homicide load, so
only the per-homicide figure reveals that their capacity per case is 26% *lower*. The sign flips
if you skip this.

**The headline is the attenuation of the `black_share` coefficient, not R².** The question is
not "what explains z" but "is the race association a resourcing story". Measured attenuation is
−0.8% [−2.3%, +0.5%] at n≥20 (CI covers zero), +2.6% at n≥50, +7.7% [+3.9%, +12.7%] at n≥100 —
the same size-dependence the cross-fitting comparison found. **Read these as a lower bound, not
a ceiling**: the regressor is compressed (below), so the control under-corrects and the race
coefficient stays higher than it should, making measured attenuation smaller than the true one.
What this design licenses is "resources account for *at least* this much"; **the upper bound is
not obtainable here** — that needs exogenous variation (an instrument or a quasi-experiment).
That it is *not* a flat null still carries information: the proxy demonstrably works at n≥100
(partial R² +0.018), so "nothing showed up because the measure is dead" is weakened, which is
evidence *against* the strong version ("resourcing explains most of the race association") —
evidence, not proof.

**Resources are deliberately kept out of the model.** Putting them in `X` would bake "few
officers per homicide here, so expect it unsolved" into `E_b`, and Detroit's anomaly would
vanish because the model pre-approved it as normal — the same laundering argument
`detect_cold_blocks` already makes for race. Staffing is worse than a neutral covariate here
because it is **endogenous**: police are assigned in response to homicide volume, so the thing
being controlled is not an input but the county severity z exists to measure. Hence a county-level
post-hoc regression only; every existing map, `cold_blocks*.csv`, and report citation stays valid.
Two limits travel with the result. The proxy is total sworn officers, **not** detectives (LEOKA
has no such column — `total_detective_*` is an assault-circumstance breakdown, 808 nationally in
2010 against 759,323 officers). And endogeneity biases the attenuation *downward*, which cuts
**against** the finding rather than for it: staffing tracks caseload almost exactly
(`corr(log officers, log homicides) = +0.883`, elasticity 1.219), so the regressor
`log(officers/homicide)` carries only **49% of the spread** of officers itself (SD 0.865 vs
1.747) and a variable with no variation cannot explain anything even when the true effect is
real; reverse causation pushes the same way, since "more resources → better clearance" (negative)
is partly cancelled by "unsolved cases accumulate → more officers assigned" (positive). The true
resource share could therefore exceed 8%. So the claim is "**the staffing level we measured**
does not explain it", never "resourcing is not the cause".

**`experiments/audit_priority.py` audits the priority list the project deliberately does not
ship.** The extension design split "cluster the unsolved cases" into D1 typology / D2 block
anomaly / D3 case-level priority, and its ethics gate said to draw D3 only from a mitigated
model — carrying an untested assumption that **mitigating the model mitigates anything derived
from it**. One quantity tests it: `lift_g(τ) = P(p̂ ≥ τ | y=0, g) / P(p̂ ≥ τ | y=0)` over
unsolved cases only. At `τ=0.5` that *is* the group FPR ratio — where the loss penalty presses
and where `fpr_gap` lives; at `τ = ` the top-q quantile it is the list's own composition. Same
curve, two points, so "the model is fair but the list is not" becomes checkable. No calibration
needed (§`detect_cold_blocks`'s δ): only ranks are used and τ is a quantile, so the whole thing
is invariant to the monotone shift.

**The answer splits by attribute, and the axis is whether the penalty targeted it.**
`mitigate_loss --attr` defaults to `Victim Race`, so `a100` pressed race only. Race: the
operating point goes 1.21 → 1.03 and **the tail follows, 1.54 → 1.07 [0.95, 1.21]** — CI covers
1, so the mitigated list is proportional. Sex: the operating point improves slightly
(1.26 → 1.17) but **the tail gets worse, 1.85 → 2.01 [1.69, 2.32]**. So the gate's assumption
holds only for the penalized attribute; single-attribute mitigation does not generalize to the
derived artifact along other attributes. Note this table is **evidence to report alongside, not
a verdict** — treating the unsolved population's composition as the list's fair target is a
value choice of the same kind as demographic parity, and the opposite reading is coherent
(female-victim homicides usually get solved, so an unsolved one really is unusual and a high p̂
is the correct signal).

The case-level list itself is computed and **not written to disk**, for three reasons that
generalize: there is no verification path (no re-investigation outcome column, so D3 is a
ranking with nothing to test against — the same wall as "same offender" being unverifiable);
the model's lack of a county feature is *deliberate* in `detect_cold_blocks` but becomes
contamination here, since "solvable case" and "case in a low-clearance county" stop being
separable without aggregation; and it violates the unit rule this repo already derived for
edges — at MCC 0.30 an individual-case output is noise-dominated, and the unit has to be the
block. Only the audit statistics ship.

**Holding the county fixed cancels three of those four channels, at zero new data — and
that relocates the whole race association.** (`reports/카운티차분_계획서_CLEAR.md`,
pre-registered at `95be50a`, rollback tag `pre-county-diff`.) Adding covariates to a
county-level regression on z only partitions R² among regressors correlated 0.7–0.9 with
each other; `audit_resources` already hit that wall. Stratifying instead needs no measure
of a channel, only that it is *constant within a county*: any channel acting as an
additive county intercept cancels in a within-county comparison, which is resourcing,
urbanicity and recording practice. Only an interaction with victim race survives — not
because discrimination is stronger but because **only discrimination is defined as that
interaction.** Two tracks, no retraining:

- **`diagnose_fairness --stratum State,City`** pushes `standardized_gap_row` from `State`
  down to county. `clear/fairness.py` needed **no change** (`strat_counts` only does
  `pd.unique` + `get_indexer`). `--stratum City` alone would be silently wrong — county
  names collide across dozens of states, so the composite key is mandatory, same as
  `BLOCK_KEYS["county"]`. `--stratum_min_n` drops to 20–30 (national test is ~63 rows per
  county) and takes several values; `--out_suffix` is now **required** for any non-default
  run because this script replaces four tracked CSVs wholesale on every invocation.
- **`experiments/county_race_residual.py`** is the track that actually decomposes z. It
  computes `O/E/V` per (county × race) with the same `calibrate` δ and reports
  `ρ_b = log(SMR_Black / SMR_White)`. **`ρ`, not `z`-difference, is primary**: `z` scales
  with √n, so correlating it with county covariates confounds effect size with county
  size — the exact trap the cross-fitting track found. Variance comes from the delta
  method, the mean is inverse-variance weighted, and the CI resamples **counties**
  (the observation unit), as in `audit_resources`.

**The result: the model's blind race gap is 88% between-county, and the residual's race
association is 100% between-county.** Nationally, blind GraphSAGE's race
selection-rate gap goes 0.1378 pooled → 0.1029 (State) → **0.0169** (county, 314 strata)
while the base-rate gap goes 0.0763 → 0.0661 → 0.0256 — numerator down 88%, denominator
66%, so amplification 1.805 → 1.558 → **0.662 [0.452, 0.888]**, CI upper bound below 1 at
all three floors. Read it as: the model has no county feature, so its only route to "which
county" is the graph — this is `mitigate_graph`'s block-membership channel measured
directly rather than inferred by elimination, and **within a county the model *understates*
the real race gap.** The pre-registered trap (§3-1) matters here: because `City` *is* the
`geo` blocking key, this fall confirms the mechanism instead of refuting it; sex and race
undergo the same arithmetic with opposite meaning (geo sex assortativity 0.0336 vs race
0.2308).

Meanwhile `ρ` is clearly non-zero — **+0.083 [+0.056, +0.128]** (651 counties, race floor
20; stable at +0.081/+0.083/+0.085 across floors), i.e. within the same county Black-victim
cases stay unsolved ~8.7% more than a race-neutral baseline expects. But
**`corr(ρ, black_share) ≈ 0`** (−0.015 [−0.108, +0.105], all floors). So `z ↔ black_share
= +0.290` is **entirely a county-level intercept effect**, and "who gets forgotten" splits
into two independent components: a *uniform* within-county race gap that does not vary with
county composition, and a county-level deficit that does and that resourcing / urbanicity /
recording practice / case mix remain unseparated within. `audit_resources --target rho`
adds that the staffing proxy explains 0.5–0.9% of `ρ`'s variance (the attenuation headline
is undefined here — its denominator is the ~0 race coefficient — so read partial R²).
Both `ρ > 0` and the county-standardized amplification `< 1` are the same statement from
opposite sides, which is the cross-check. Caveats that travel: `ρ` is a **lower bound**
(α=100's within-state race gap is still negative), the inverse-variance mean is
large-county weighted (the unweighted mean only clears zero at floor 30), and three leaks
survive county differencing — within-county case mix, channel × race interaction (partly
definitional: race-differential triage inside one department *is* institutional
discrimination), and race-differential recording. **The SHR join settled the first and
measured the third** (see below).

**`experiments/shr_circumstance.py` closes the case-mix question by proving it cannot be
closed, and measures the recording channel instead.** (`reports/SHR정황조인_계획서_CLEAR.md`.)
`Circumstance` is absent from the Kaggle CSV but present in SHR (Kaplan, **openICPSR
100699** — a different deposit from LEOKA's 102180). The join works: the raw CSV carries
`Incident`, so `(Agency Code, Year, Month, Incident)` hits the SHR incident directly —
**96.10% matched, White 95.90% vs Black 96.16%, a 0.27pp race difference**, rows preserved
at 638,454. So the data is usable; the problem is what it is.

**Recorded circumstance is downstream of the outcome, which makes it a collider.** With
`C*` the true circumstance, `C` the recorded one, `Y` solved and `R` victim race, the graph
is `R → C* → Y` (the confound we wanted to block), `R → D → Y` (the target), and
`C* → C ← Y`. Conditioning on `C` opens that collider, inducing a non-causal `C*`–`Y`
association that flows into `R`–`Y` via `R → C*`. **Not a noisier control — a different
operation, with unknown sign.** The `Y → C` edge is real: `P(undetermined)` is **14.6% on
solved vs 56.4% on unsolved**. And the decisive part is that the coding depends on race
*given the county*: unsolved cases with Black victims are **+4.1pp [+2.0, +6.8]** more
likely to be coded undetermined (floors 10/20/30 give +4.0/+4.1/+4.1, all CIs excluding
zero; pooled is +8.2pp, so roughly half is between-county and half within). Adjusting `E_b`
for circumstance would therefore absorb "police recorded less for Black victims' unsolved
cases" as if it were case mix — the same laundering this repo already refused for race and
for staffing. Both `--undetermined`-as-category and `--undetermined`-excluded variants fail:
the second selects a subsample on solvedness (85.4% vs 43.6%), breaking the O/E
construction. **So the within-county case-mix leak is permanently open, and ρ still has
only an upper bound.** That is a settled answer to a question that was open, and the
+4.1pp is direct evidence for the race-differential recording leak.

Two traps worth keeping. A "survival ratio" (share among unsolved ÷ share among solved)
looks like a contamination diagnostic and is **a tautology** — it equals
`(1−q_c)/q_c × (N_solved/N_unsolved)` exactly, so it carries nothing beyond the category's
own clearance rate (predicted 1.113 vs observed 1.11 for robbery, and so on). And the
pre-registered discriminating test **failed**: correlating county clearance with each
category's share among unsolved was supposed to split offender-dependent categories
(positive under contamination) from felony-type ones (negative under genuine case mix);
observed medians are +0.007 vs +0.106, the wrong way round, and
`corr(clearance, undetermined share)` is **+0.090** rather than negative. It is kept in the
script rather than dropped. The conclusion does not rest on it — it rests on the semantics
of "undetermined" (a statement about investigation, not about the case) plus the
within-county race dependence, which invalidates `C` as a control whether the mechanism is
`Y → C` or `D → C`.

**`experiments/returna_crosscheck.py` bounds how much of `ρ` could be recording, using an
independent second measurement.** (`reports/ReturnA교차검증_계획서_CLEAR.md`, tag
`pre-returna`.) UCR Return A (openICPSR **100707** — a third deposit, distinct from SHR's
100699 and LEOKA's 102180) is the same agency counting the same homicides and clearances on
a different form, so a disagreement between it and `Crime Solved` cannot be discrimination
or resourcing — by construction it is a recording problem, and its size caps the
contamination. `(ORI, year)` joins at **100.0%**, matching LEOKA's precedent on the same
axis. The shipped version is *wide monthly* (one row per agency-year, months as columns,
1,448 of them), so the loader sums the twelve `{MON}_ACT_MURDER` / `_CLR_MURDER` columns and
reads only 27 of them; `MANSLAUGHTER` is excluded because Part I `murder` already means
murder and nonnegligent manslaughter while the separate column is negligent.

**The verdict is the pre-registered second row: state the bound, keep `ρ`.** Recorded
homicide counts agree — `SHR/ReturnA` median **1.029**, so the heaviest scenario (the sample
itself is biased) does not fire. Clearance rates disagree substantially — `d = q_RA − q_SHR`
median **−0.0913** — but **`corr(d, black_share) = +0.013 [−0.136, +0.052]`, i.e. zero**, so
the disagreement carries no racial pattern. Worst case, with the entire measured error piled
onto one race, `ρ = +0.0819` moves to **[−0.0658, +0.2328]**: the sign does not survive that
extreme, and nothing supports the extreme. So the sibling reading of the SHR `+4.1pp` holds
but is **not promoted**, and the artifact reading is not excluded either. Read `d`'s sign
with care — `Crime Solved` is a snapshot that includes clearances years later while Return A
books them in the month they happen and the file ends in 2016, so part of −0.091 is
structural censoring rather than error. That is why this track yields an interval and not a
point estimate. One new by-product: `corr(SHR/ReturnA, black_share) = −0.240 [−0.294,
−0.196]` — coverage itself is not race-neutral, at 3% overall magnitude.

Three defects surfaced while running it, all of the plausible-but-wrong kind. `find_source`
expanded a directory into its file list and read **only the 1960 file**, giving G1 = 0% —
and the pre-registered failure condition merely *warned* and carried on, saving a table full
of NaN. It now aborts. `MONTHS_REPORTED` is a sentence (`"december is the last month
reported"`), not a number, so `to_numeric` would silently mark every agency-year as
unreported. And the contamination bound clipped `d` at zero, which erased essentially every
county's error because the measured median is **negative**; fixing the sign moved the bound
from [+0.039, +0.164] to **[−0.066, +0.233]**, i.e. across zero — an error large enough to
change the conclusion.

**The national flat baseline now exists, and it revises half of the mechanism claim.**
`train_baseline.py` was restored from `96d1dd1` and run at national scope only — this is
the one thing on the backlog that genuinely needed training, and it is **additive**: it
writes new dumps and leaves every GNN model untouched, so `cold_blocks`, the maps and `ρ`
did not move. The three-state dumps are **never regenerated**; xgboost went 2.x → 3.3.0 and
a changed prediction would shift the committed headline. Eleven national dumps now share
one test split (191,537 rows).

| blind, White-vs-Black | pooled amplification | State-standardized | MCC |
|---|---|---|---|
| **GraphSAGE (geo)** | **1.805** [1.72, 1.90] | **1.558** [1.46, 1.67] | **0.3133** |
| XGBoost | 1.632 [1.55, 1.72] | 1.301 [1.20, 1.41] | 0.2989 |
| LogReg | 1.429 [1.34, 1.51] | 1.064 [0.98, 1.16] | 0.2638 |

**One half of the three-state finding replicates and the other does not.** The GNN does
amplify race more than either flat model — paired GraphSAGE−XGBoost **+0.173 [+0.135,
+0.212]** pooled, **+0.257 [+0.208, +0.310]** standardized, both significant, same
direction as the three-state +0.82. But *"close the direct path and the flat models stop
amplifying"* is **false at national scale**: blind XGBoost goes 0.67 → **1.632** and blind
LogReg 0.05 → **1.429**. That is the same scaling already measured on the feature side —
blind-X race recovery AUC 0.7001 → 0.7770 — so nationally the graph is **an increment on
top of a strong feature-proxy channel**, not the sole surviving route. Three states made
the graph look like the only carrier because the proxy channel was weak in that sample.

The accuracy side replicates cleanly: blind MCC +0.0144 over XGBoost against a GNN seed std
of 0.00061, **23.6×**. And sighted the sign flips — GraphSAGE 2.245 *below* XGBoost 2.544
and LogReg 2.785 (paired −0.739 and −0.980, both significant), where three states could not
separate them (−0.04). Handed the attribute directly, the flat models lean on it harder;
the graph's cost only becomes visible once the direct path is shut.

**That test is now paired, and it says the two models land on opposite sides of zero.**
(`reports/짝지은표준화대조_계획서_CLEAR.md`, pre-registered `278dd7a`, tag
`pre-paired-std`.) `standardized_contrast_rows` lifts the pooled joint-cell trick to the
stratum axis: draw a 16-cell multinomial per (stratum, group) and marginalize back to each
model, so differences are computed within replicate. `marginal()` needed **no change** —
its reshape takes an arbitrary prefix, so `(B,S,16)` folds to `(B,S,4,4)` correctly.
Pairing also makes the denominators identical replicate-by-replicate, since `base_rate` is
`(fn+tp)/n` and a joint draw preserves the y-marginal.

**The catch, and it changed a reported result: `_span` is `max−min`, so it is unsigned.**
When two models' gaps point *opposite ways*, their absolute values move in opposite
directions under resampling — measured `corr = −0.72…−0.75`, so pairing *widened* the CI
to 0.76–0.79× of the independent guess. Signed gaps flip this to `corr = +0.73…+0.76` and
**1.9–2.0× narrowing**. `standardized_contrast_rows` therefore also emits
`quantity="signed_gap"` (two-group sets only — sign is undefined for three or more), and
`signed_as` records the direction. This is not only precision: county-standardized,
GraphSAGE's race gap is **+0.0103 (Black − White)** while blind XGBoost's is **−0.0175**,
so the earlier "GraphSAGE's gap is 0.59× XGBoost's" understated it — they are not
differently sized, they are **on opposite sides of zero**, and paired differences are
significant at all three floors (+0.0277 [+0.0206, +0.0345] at n≥20, +0.0261 and +0.0253
at 30 and 50). Remove the between-county channel and the GNN has **no White-favouring race
residual left**, while the flat model's feature-proxy channel is untouched by county
standardization. Sex is significant too and in the same direction for both models
(+0.0368 vs +0.0537, difference −0.0168 [−0.0244, −0.0091]).

Judge these on **`signed_gap`, not amplification**: the standardized `base_rate` gap has a
sample CV of 0.108, so dividing by that small noisy denominator hands the ratio its full
multiplicative noise and the amplification contrast stays wide even paired. A national
positive control (`geo_blind_mb − a100_mb`, 314 strata) is what separated "low power" from
"broken implementation" when the CI-narrowing check first failed — keep it in any rerun.
Output is a fifth file, `fairness_model_contrasts_standardized{suffix}.csv`, separate for
the same reason the standardized gaps are: the pooled contrast table is what the reports
cite.

**Side result: the alpha ladder is confirmed non-monotone in the county-standardized
metric.** Signed gaps go −0.0208 (α=0) → +0.0005 (α=10) → +0.0065 (α=25) → +0.0013 (α=50)
→ +0.0079 (α=100) — the penalty overshoots past zero, and `a25−a50` (+0.0052 [+0.0029,
+0.0077]) and `a50−a100` (−0.0066 [−0.0093, −0.0041]) are both significant, so the wobble
is real rather than noise. On the within-county gap the best point is **α=10, not α=100**.
That is a measured argument that turning the pooled-defined knob harder is the wrong
instrument, and it is input to the stratified-penalty design rather than a re-tune.

**`experiments/build_web_map.py` is the national deliverable** — one self-contained HTML,
zero external requests, no chart library (a choropleth is fill on `<path>`; projection and
simplification stay in Python). 3,090 counties / 50 states / 626 KB. Three details are
load-bearing. The **sample-floor slider consumes three separate `detect_cold_blocks` runs**,
because BH q-values depend on how many blocks were tested together — filtering one run by
`n` would report wrong q's, and the effect is not even monotone (three states give 5 cold
at n≥20 but 6 at n≥50). The **three-tier encoding is resolved to integer colour indices at
build time**, so the browser never compares the flag strings that once read back from CSV
as `NaN` and painted all 143 counties as signal. And **per-level payload carries only what
changes** — `n`/`z`/`smr`/`black_share` are properties of a county, so shipping them once
instead of three times took 729 KB (over budget) to 664 KB, with 1.5 km simplification
closing to 626 KB. Verified without a browser by re-rendering the emitted paths and fills,
`node --check`, and 13 data invariants.

**Read that last sentence carefully, because it was half false for months.** The path
re-rendering and the 13 invariants were run **by hand, once, during development** —
`build_web_map.py` had no `check()` function at all — and `node --check` had **no call site
anywhere in the repo** while five documents described it as a build-time check. Both are
now real: `experiments/_htmlcheck.py` (`node_check`) extracts the inline `<script>` bodies
and runs `node --check` on each, wired into all three HTML builders, and a syntax error
fails the build. It **skips with a printed notice when `node` is absent**, because the
dashboard's contract is that it builds on `pandas + numpy` alone and a hard node dependency
would break it — but skipping-because-absent and passing-because-checked are different
events and the output says which one happened. Two things this repo already knew showed up
again while wiring it: `subprocess(text=True)` decodes with the locale codec, so reporting
a syntax error **died in cp949** on the non-ASCII temp path (fixed to explicit UTF-8), and
`OFL.txt` and `_htmlcheck.py` each broke `--verify_clone` the moment they became build
inputs while untracked — which is exactly what that check is for.

**`node --check` is syntax only, and the bug that shipped was not a syntax error.**

**"DOM behaviour is unverified" was a real hole, and it had already shipped.** `legend()`
opened with `const d=D[LV[li]];` — `D` was never defined anywhere, so the line threw a
`ReferenceError` on every call. `paint()` had already filled the counties by then, so all
three shipped maps *looked* correct while their **legend and ranked table were silently
empty**; the failure landed after the visible part. `node --check` passes it (syntax is
valid) and the emitted-path validator never touches it. It was found by querying the DOM
in a headless browser. So the rule from the figure bugs generalizes to HTML: **the
validator checks what you told it to check — open the artifact and look at it**, and for
a page, "look" means query the DOM, not read the source.

**`experiments/verify_html.py` closes that, and its assertions are read off the bug.**
`_domprobe.js` runs the page in jsdom (`runScripts: "dangerously"`), collects every
uncaught error, then **drives the controls** — every slider level, every layer button, the
table toggle, every dashboard tab — and dumps observations as JSON; `verify_html.py` does
the asserting in Python, because the CSV cross-check belongs in the language that owns the
data. The four requirements: zero uncaught errors (this alone catches the shipped bug),
non-empty legend and ranked table, **the same assertions again after each interaction**
(`legend()` runs on every repaint and the failure landed after the first paint, so a
load-only check would miss it), and screen counts equal to counts recomputed from
`cold_blocks*.csv`.

**It is verified against the real bug, not assumed to work.** `--selftest` re-injects the
original `const d=D[LV[li]];` into a copy of the shipped map and fails if the checker still
passes; it currently reports the defect as 28 findings, the first being the uncaught
`ReferenceError`. It runs by default before the real artifacts — same discipline as
deliberately planting a syntax error to prove `node --check` fires. It runs by default before the real artifacts — same discipline as
deliberately planting a syntax error to prove `node --check` fires.

**All four artifacts are covered, and the story page needed a shim rather than a browser.**
Reading what it actually does with the browser-only APIs settled that: `IntersectionObserver`
drives scroll-reveal only (add `in`, `unobserve`), so a shim that reports immediate
intersection *is* the fully-scrolled state we want to assert; `getBoundingClientRect`
appears once as `void c.getBoundingClientRect()` to force reflow, so jsdom's zeros are
harmless; and `requestAnimationFrame` is deliberately **not** used. So the shim fakes one
visibility event, not layout — any future assertion that depends on layout does need a real
browser. The story page gets its own selftest with a different injected defect, because its
fatal mode differs: `.js` goes on at the top of the script and the CSS hides content only
under `.js`, so a script that dies *after* that line leaves every section at `opacity:0` —
a blank page. Injecting a throw right after the class is added reproduces exactly that.

**The CSV cross-check immediately found a defect that is not in the map at all.** Screen
counts ran 3 short at n≥20 and 1 short at n≥50/100. Two are legitimate — `City` is
`Repressed` (the agency suppressed the county name), so there is no geography to draw and
the checker reports them as an expected exclusion. The third is real: **`Dade` and
`Miami-Dade` are the same county**, renamed in 1997, and both names are present as separate
blocks. `clear.counties.ALIASES` maps them to one FIPS at *map* time, so `color_tables`'s
`drop_duplicates("fips")` silently discards one — but the deeper problem is upstream.
`detect_cold_blocks` blocks on raw `["State", "City"]` (as do `county_race_residual` and
`audit_resources`), so the county was **split into two blocks and tested twice**:
Miami-Dade n=523 z=+6.86 **cold**, Dade n=9,054 z=+1.12 not significant.

**Fixed by `clear.counties.canonicalize()`, applied in the analysis layer only.**
`detect_cold_blocks`, `county_race_residual`, `audit_resources` and
`diagnose_fairness --stratum State,City` all call it, so the four share one county
definition. It is deliberately **not** in the pipeline (`01`–`04`): `City` is the `geo`
blocking key, so merging there would change the graph and force a national retrain for no
gain. Merging at aggregation is sound anyway — SMR is O and E summed over a county's cases,
and the sum is valid however p̂ was produced; the model having no county feature is the
design, so its blocking is not the definition of a county.

What moved: Florida merges to n=9,577 and SMR **1.408 → 1.035**, which drops the cold flag
on the test split (38 → 37 cold) and leaves it standing but weakened on cross-fit (z 6.86 →
2.70, q 0.0023 → 0.0334); no other block changed verdict in either table. Virginia's
`Clifton Forge` (3) + `Alleghany` (19) = 22 **newly clears the n≥20 floor**, so the cv5
county count stays 1,803 (−1 +1). Aggregates moved in the third decimal only —
`z ↔ black_share` +0.290 → +0.288, `ρ` +0.083 → +0.084, county-standardized amplification
0.662 → 0.671 with the CI upper bound still under 1. Recurrence is blocked by
`clear.counties.assert_one_row_per_fips()`, which **kills the map build** when one FIPS has
two rows: a silent `drop_duplicates` is what kept this alive for three months, so turning
silent loss into loud failure is half the fix.

**Two national maps ship, and the cross-fit one is primary.** `map_national.html` is built
from `cold_blocks_cv5.csv` (1,800 counties carry data at n≥20); `map_national_test.html` is
the test-split reference, kept because it shares a split with `fairness_*.csv` and can be
cited alongside them. The decision rests on the two tables agreeing on effect size while
disagreeing on precision — Fulton SMR 1.55→1.49, Wayne 1.19→1.26, but |z| max 15.66→26.60 —
so cross-fitting is the same answer measured better, and on the 4 cold counties visible
*only* there (Lauderdale TN SMR 2.69, Greensville VA 2.98, Pulaski MO 2.06, Grenada MS 2.01):
the strongest SMRs in the whole table, invisible on the test map because they are small. A
map answering "who gets forgotten" must not structurally drop small counties. Cost of the
choice: the primary map no longer shares a split with the fairness tables, and its z values
are not comparable to the test map's, so **`--src` drives a mandatory caption** (`SPLIT_NOTES`)
carrying both facts plus "do not compare colour intensity between the two maps". Both ship at
1.5 km simplification deliberately — differing geometry between two maps a reader compares
side by side would read as a data difference. The cross-fit map is 681 KB against the 650 KB
budget; the excess is per-county payload for 2.1× the counties, not geometry, so simplifying
harder would trade border fidelity for someone else's data. Note that several top
cold counties are independent cities and are nearly invisible on a national choropleth —
small in area, not weak in signal — so **hover and the table view are how this result is
read**, and any static figure needs the ranked table beside it.

**`experiments/map_figures.py` draws that table as a county choropleth** (`clear.counties`).
`City` really is a county field here, so the choropleth — not a city dot map — is the
correct form. No geopandas: GDAL/GEOS binaries are heavy and fragile on Windows, and all
that is needed is a FIPS join, polygon coordinates, and an equal-area projection, which
`json` + `numpy` cover. The same judgment carries into the national web map (no chart
library, project in Python, ship SVG paths). Reference data (Census county codes, a
FIPS-keyed county GeoJSON) is cached under `dataset/geo/`, gitignored and re-downloadable.
The FIPS join is **143/143** on CA+TX+MI, but `clear.counties.ALIASES` and the unmatched-
name printout stay, because national scale reintroduces independent cities
(`Baltimore city` vs `Baltimore`), renames (`Dade` → `Miami-Dade`) and parishes.
Projection is CONUS Albers **equal-area** — an equal-angle projection inflates large
high-latitude counties and makes them look important.

**The palette is computed, not eyeballed.** The blue arm is the documented sequential
ramp; the red arm is generated at **matched OKLab lightness** per step, giving a worst
arm-to-arm L error of 0.00105 with blue monotone 0.905→0.338 and the neutral gray at
0.952, brighter than either arm's lightest step. Chroma is capped at 0.85 of the gamut
boundary — the boundary itself yields a near-fluorescent red at mid lightness, which
makes the colour rather than the data raise the alarm. Lightness is untouched, so the
diverging symmetry holds.

Three bugs surfaced only by rendering the figure and looking at it, and all three were
the "plausible but wrong" kind:
(1) **every county was being coloured, not just the significant ones** — `flag` is an
empty string in memory but reads back from CSV as `NaN`, so `flag == ""` silently failed
and 123 non-significant counties were painted as signal. Fixed to a three-tier encoding:
significant → diverging colour, assessed-but-not-significant → neutral gray, not assessed
→ hatch. This is the same discipline that keeps F1/Precision out of the accuracy claims,
applied to a figure.
(2) legend bands were off by one against `searchsorted`, so the legend named different
numbers than the map drew.
(3) only the first 3 steps of the 5-step ramp were used, so the strongest county (z=11.4)
came out mid-red; `clear.counties.arms(n_steps)` now **subsamples the ramp evenly** so
both extremes are always present. **Render and look at the output** — the validator
checks colour, not whether the picture says what you think.

**`experiments/dashboard.py` is the deployment layer, and its whole contract is "clone
and run".** (`reports/로컬배포_계획서_CLEAR.md`.) One self-contained
`outputs/dashboard.html` — six panels (predict / diagnose / mitigate / graph diagnostics /
apply / post-hoc audits), two scope tabs, opened via the stdlib `webbrowser`. It reads
**only the committed result CSVs**: not `data/processed/`, not `dataset/`, not the
uncommitted dumps. That is why `requirements-dashboard.txt` is pandas + numpy and nothing
else — `config.py` imports `os`/`pathlib` only and `clear.results` imports pandas only, so
the torch/PyG install (CUDA index, manual `pyg-lib`) that `requirements.txt` needs never
enters the path. The **43 committed result CSVs (13 MB) are what makes this possible**;
that gitignore rule stops being tidiness and becomes the product.

`--verify_clone` builds the dashboard inside a temp tree holding **only git-tracked
files**, so the contract is a command rather than a claim. Its self-check is a **runtime
path trace, not a source grep**: `load()` records every path it opens and only those are
checked. Grepping the source for `data/processed` failed 4 times on *doc comments* naming
the path — comments don't get caught now, and a path read without a comment still does.

Two data bugs surfaced only by rendering the panel, both of the "plausible table" kind.
(1) **`results.csv`'s `std` column is empty on `family=train` rows** — the ledger stores
one row per seed and aggregation belongs to the reader, so trusting it printed `-`
everywhere and made the whole margin÷noise column meaningless. Std is now computed across
seeds. (2) **A label that omits any config key silently merges different experiments.**
Three-state blind `geo` holds shuffle, ranked and degree-matched-control rows under the
same model/tag/edge_type; grouping without `edge_mode` averaged 0.2659/0.2628/0.2574 into
**0.2620, a number that exists nowhere**, and matching `tag` by substring merged
`fairness_handoff` into plain sighted. Labels now carry every config key (same fix for
`group_set` in the contrast table and `measure` in the resource audit). **The identity key
of a row is the full config, in the reader as well as in the writer.**

One caveat is printed on the panel itself: the MCC margin÷noise ratio there is 99.4×
against the report's 19.0×, because std comes from the 3 seeds left in `results.csv`.
**The ordering of metrics is the claim, not the absolute ratio** — which metrics clear
their own noise is the same under both computations.

**`experiments/build_story_page.py` (+ `experiments/_fontpack.py`) is the portfolio page.**
(`reports/포트폴리오웹페이지_계획서_CLEAR.md`.) Scroll narrative ending in the county map,
one self-contained file, same rules as `build_web_map` (no chart library, project in
Python, `--src` drives the caption through the shared `SPLIT_NOTES`). Three things carry:
**verifiable numbers are read from the CSVs, not typed into the prose** (county counts,
cold/warm, correlations, top counties), with only the un-derivable ones as constants
carrying a source comment; **red is reserved for the gap alarm**, so the warm arm is the
gray ramp (`clear.counties.gray_arm`, which went into `clear/` because the OKLab
lightness-matching discipline already lives there) and the page never spends alarm colour
on anything else; and **fonts are subset per weight** — text is emitted as (string,
weight) pairs so the build knows which glyphs each of 400/700/800 needs, and a missing
glyph fails the build rather than rendering a box.
The globe hero was built and then **withdrawn** — the plan doc keeps the section marked
철회 rather than deleting it.

`_fontpack.py` (2 callers — story page and dashboard) and `_htmlcheck.py` (3 — both of
those plus `build_web_map`) stay in `experiments/` even though they clear this repo's
two-or-more bar for `clear/`. The bar is necessary, not sufficient: `clear/` holds the data
and model logic that experiment scripts are thin CLIs over, and these two are **artifact
plumbing** — font subsetting and HTML validation, which no analysis ever imports. The
leading underscore marks that they are not part of the experiment surface.

**Licensing is split three ways, and the split is load-bearing rather than bureaucratic.**
The source data (Kaggle / Murder Accountability Project) is **CC BY-SA 4.0**, so
share-alike reaches every data-derived artifact: `outputs/**`, `reports/**`, the docs, and
the generated HTML all ship CC BY-SA 4.0, while `src/**` is MIT because code holds no data
and is not an adaptation. The sharpest instance is `outputs/predictions/*.csv` — the four
committed flat-model dumps reproduce licensed *column values* row by row (`y_true` is
`Crime Solved`; victim race and sex are verbatim), so they are unambiguously adapted
material. The raw CSV itself is never redistributed.

**The font is a carve-out, not part of that.** The two artifacts that embed a NanumSquare
subset (`dashboard.html`, `clear_story.html`) would otherwise purport to release a font
derivative under CC BY-SA, which SIL OFL 1.1 forbids ("cannot be released under any other
type of license"). Three consequences are wired into the code, not just documented:
`_fontpack.license_comment()` embeds the **full OFL text** as a leading HTML comment,
because OFL §2 demands the licence travel with every copy and a *link* would make the
licence an external dependency of a page whose whole design rule is zero external requests;
`_fontpack._relicense_names()` rewrites the subset's internal name table to a neutral name
(OFL §3 — subsetting is a Modified Version) while **preserving nameID 0**, which is the
notice §2 requires, and restoring the nameID 13/14 licence fields that `fontTools` drops;
and `--no_fonts` builds omit the OFL comment, since shipping it on a file with no font
would be a false notice. `OFL.txt` is therefore a **build input** — `--verify_clone`
catches its absence, and did.

Two false positives came out of this and both generalize. The self-checks flagged the URLs
inside the OFL text as external requests; the check measures *requests*, not *strings*, so
HTML comments are now stripped before the scan exactly as base64 payloads already were.
And `--verify_clone`'s **success** message crashed on a cp949 console over an em-dash —
console output is a rendering target too, and the failure was in the one path nobody
exercises.

**`clear.gnn.save_checkpoint` / `load_checkpoint` exist; the weights are deliberately not
committed.** `train_gnn --save_model` writes one `.pt` per seed under
`outputs[/{scope}]/models/`, and the file holds the **feature column order** alongside the
`state_dict`, hyperparameters, `blind`, scope, edge type and `k`. The column order is the
load-bearing field: `03_features.py` one-hots in a data-dependent order, so a pipeline
re-run on another machine can permute `X`'s columns and silently attach weights to the
wrong features — plausible metrics, no error, exactly the failure mode `--blind`'s silent
no-op already inflicted once. `load_checkpoint` compares the list and **dies on
mismatch**. The file also records `calibrated: False`, because `pos_weight` training makes
`proba` a score and not a probability (§`detect_cold_blocks`'s δ) and someone will
otherwise read it as one. Not committed (`.gitignore`: `outputs/models/`,
`outputs/*/models/`) even at ~100 KB: a clone has neither the graph nor the features, so
shipping weights would only *look* like deploying a model.

**Trainers dump test predictions** (`clear.predictions`) joined to
the unencoded sensitive attributes, so diagnosis reads CSVs and never
re-instantiates a model — `diagnose_fairness` re-runs in seconds against a GNN that took
minutes to train, and `mitigate_threshold` (since removed — see Scope) wrote mitigated
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
`outputs/*.png`, `outputs/*.log`, and all of `outputs/predictions/` (a few MB each,
regenerable by re-running `train_baseline`/`train_gnn`) — bar the four committed flat-model
dumps above. The generated HTML follows the figures: `outputs/web/`, `outputs/*/web/`
(maps + story page) and `outputs/dashboard.html` are ignored because they are **rebuilt
from the committed CSVs**, which is the whole point of the clone-and-run contract. Model
checkpoints (`outputs/models/`, `outputs/*/models/`) are ignored for the opposite reason —
a clone cannot use them. The small result CSVs under `outputs/` **are**
tracked — they're the experiment record, and the deployment layer's only input. The shared metrics ledger
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
