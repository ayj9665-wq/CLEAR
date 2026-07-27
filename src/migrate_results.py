"""migrate_results.py — 옛 결과 CSV 4종 -> outputs/results.csv (일회성, 재실행 가능)

clear.results의 long 스키마로 옮긴다. 원본은 지우지 않고 outputs/legacy/로
옮겨 보존한다(실험 기록이므로).

  metrics.csv              -> family=train | ablation | mitigate_graph | mitigate_loss
                              (tag로 분류. seed별 raw 행)
  fairgraph_tradeoff.csv   -> family=mitigate_graph      (seed 평균 + 격차)
  fairloss_tradeoff.csv    -> family=mitigate_loss       (seed 평균 + 격차)
  mitigation_tradeoff.csv  -> family=mitigate_threshold

**metrics.csv는 깨져 있어서 복구도 겸한다.** clear.ledger.append가 헤더를
파일 생성 시에만 쓰는데 LEDGER_COLS가 두 번 늘어서, 헤더는 25열인데 행은
25/26/28열이 섞였다 — pandas.read_csv가 ParserError로 죽는다(즉 ledger.load()에
의존하는 ablation 요약이 현재 고장). 열은 **뒤에만** 추가됐으므로 행 길이 N은
LEDGER_COLS[:N]에 대응한다. 그걸로 위치 기반 복구가 가능하다.
long 스키마에서는 지표가 늘어도 행이 늘 뿐이라 이 실패 자체가 재발하지 않는다.

실행: python migrate_results.py [--dry_run]
"""
import argparse
import csv
import pathlib
import io
import json
import shutil

import pandas as pd

import config as C
from clear import results as R


# 삭제된 clear.ledger의 마지막 스키마. 위치 기반 복구가 이걸 기준으로 하므로
# 마이그레이션 스크립트가 사본을 들고 있어야 한다(원본 모듈은 통합과 함께 제거됐다).
LEDGER_COLS = [
    "timestamp", "model", "tag", "seed",
    "auc", "mcc", "f1", "sensitivity", "specificity",
    "balanced_accuracy", "precision",
    "edge_type", "k_neighbors", "hidden_dim", "num_layers", "dropout",
    "lr", "weight_decay", "aggr", "max_epochs", "patience", "val_size",
    "best_epoch", "train_seconds", "n_params",
    "fair_alpha", "fair_beta", "best_val_gap",
]


def _params(row, keys):
    """빈 값·NaN을 뺀 노브 dict."""
    out = {}
    for k in keys:
        v = row.get(k)
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            continue
        out[k] = v
    return out


def _family_for_tag(tag):
    tag = tag or ""
    if tag.startswith("fairloss"):
        return "mitigate_loss"
    if tag.startswith("fair_"):
        return "mitigate_graph"
    if tag == "default" or tag.startswith("sweep_"):
        return "ablation"
    return "train"


def _attribute_for_tag(tag, family):
    if family not in ("mitigate_loss", "mitigate_graph"):
        return None
    return "Victim Sex" if "_sex" in (tag or "") else "Victim Race"


def _blind_for_tag(tag, family):
    tag = tag or ""
    if family in ("mitigate_loss", "mitigate_graph"):
        return "sighted" not in tag          # 이 실험들의 기본 조건이 blind
    return tag == "blind" or tag.endswith("_blind")


# ---- 1) metrics.csv (위치 기반 복구 + 변환) ----------------------------------
HP_KEYS = ["edge_type", "k_neighbors", "hidden_dim", "num_layers", "dropout",
           "lr", "weight_decay", "aggr", "max_epochs", "patience", "val_size",
           "fair_alpha", "fair_beta"]
RUN_METRICS = ["best_epoch", "train_seconds", "n_params", "best_val_gap"]


def from_ledger(path):
    raw = list(csv.reader(io.open(path, encoding="utf-8-sig")))
    if not raw:
        return [], {}
    body, widths = raw[1:], {}
    out = []
    for rec in body:
        if not any(rec):
            continue
        n = len(rec)
        widths[n] = widths.get(n, 0) + 1
        # 열은 뒤에만 추가됐다 -> 길이 N인 행은 LEDGER_COLS[:N]에 대응.
        row = {k: v for k, v in zip(LEDGER_COLS[:n], rec)}
        tag = row.get("tag") or ""
        family = _family_for_tag(tag)
        num = {}
        for m in R.ACCURACY + RUN_METRICS:
            v = row.get(m)
            if v not in (None, ""):
                num[m] = float(v)
        out += R.rows(
            family, num,
            model=row.get("model") or None,
            tag=tag or None,
            seed=int(float(row["seed"])) if row.get("seed") else None,
            attribute=_attribute_for_tag(tag, family),
            blind=_blind_for_tag(tag, family),
            params=_params(row, HP_KEYS),
            timestamp=row.get("timestamp") or None,
        )
    return out, widths


# ---- 2~4) 트레이드오프 3종 ----------------------------------------------------
GAP_METRICS = {
    "dp_gap": "selection_rate_gap",
    "dp_amplification": "selection_rate_amplification",
    "tpr_gap": "tpr_gap",
    "tpr_amplification": "tpr_amplification",
    "base_rate_gap": "base_rate_gap",
}


def from_tradeoff(path, family, param_keys, note_keys=(), model=None,
                  group_set="named_n>=5000"):
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        d = row.to_dict()
        num, ci, std = {}, {}, {}
        for m in R.ACCURACY:
            if f"acc_{m}" in d and pd.notna(d[f"acc_{m}"]):
                num[m] = float(d[f"acc_{m}"])
        if pd.notna(d.get("acc_mcc_std")):
            std["mcc"] = float(d["acc_mcc_std"])
        for old, new in GAP_METRICS.items():
            if old in d and pd.notna(d[old]):
                num[new] = float(d[old])
                lo, hi = d.get(f"{old}_lo"), d.get(f"{old}_hi")
                if pd.notna(lo) and pd.notna(hi):
                    ci[new] = (float(lo), float(hi))
        if pd.notna(d.get("best_val_gap")):
            num["best_val_gap"] = float(d["best_val_gap"])
        if pd.notna(d.get("n_eval")):
            num["n_eval"] = float(d["n_eval"])
        out += R.rows(
            family, num,
            model=d.get("model", model) or model,
            tag=d.get("tag"),
            seed=None,                     # seed 평균 요약 행(원자료는 family별 seed 행)
            attribute=d.get("attribute"),
            blind=d.get("blind"),
            group_set=group_set,
            params=_params(d, param_keys),
            ci=ci, std=std,
        )
    return out


# (파일, family, 기본 model, 노브=identity, 파생값=기록만)
SOURCES = [
    ("fairgraph_tradeoff.csv", "mitigate_graph", "graphsage",
     ["mode", "p"],
     ["n_pairs", "n_pairs_full", "kept_frac", "homophily", "homophily_full"]),
    ("fairloss_tradeoff.csv", "mitigate_loss", "graphsage",
     ["alpha", "beta", "grad_clip"], []),
    ("mitigation_tradeoff.csv", "mitigate_threshold", None,
     ["criterion", "lambda"], ["groups", "thresholds"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="변환 결과만 요약 출력, 파일 쓰기·이동 없음")
    ap.add_argument("--src", default=None,
                    help="옛 CSV가 있는 디렉터리(기본 outputs/). 이미 legacy/로 "
                         "옮긴 뒤 다시 돌릴 때 outputs/legacy를 준다.")
    args = ap.parse_args()
    src = pathlib.Path(args.src) if args.src else C.OUTPUT_DIR

    rows, legacy = [], []

    lp = src / "metrics.csv"
    if lp.exists():
        got, widths = from_ledger(lp)
        rows += got
        legacy.append(lp)
        print(f"[metrics.csv] 행 길이 분포 {dict(sorted(widths.items()))} "
              f"-> {len(got):,} long 행 (위치 기반 복구)")

    for fname, family, model, keys, note_keys in SOURCES:
        p = src / fname
        if not p.exists():
            print(f"[{fname}] 없음, 건너뜀")
            continue
        got = from_tradeoff(p, family, keys, note_keys, model=model)
        rows += got
        legacy.append(p)
        print(f"[{fname}] -> family={family}, {len(got):,} long 행")

    df = pd.DataFrame(rows)

    # 옛 원장은 append-only라 같은 설정을 다시 돌리면 행이 하나 더 쌓였다(실제로
    # 몇 건 있다 — 같은 tag·seed인데 best_epoch가 79 vs 67로 다르다). 새 스키마의
    # 규약은 "같은 KEY는 최신이 이긴다"이므로 여기서 맞춰 놓는다. 버리는 게
    # 아니라 outputs/legacy/ 원본에 그대로 남는다.
    n_before = len(df)
    df = (df.sort_values("timestamp", na_position="first")
            .drop_duplicates(subset=R.KEY, keep="last")
            .reset_index(drop=True))
    if len(df) < n_before:
        print(f"[dedup] 같은 설정의 과거 재실행 {n_before - len(df):,}행 -> 최신만 유지")
    rows = df.to_dict("records")

    print(f"\n합계 {len(df):,} 행 / family별:")
    print(df.groupby("family").size().to_string())
    print(f"\n지표 어휘({df['metric'].nunique()}종): {sorted(df['metric'].unique())}")

    if args.dry_run:
        print("\n[dry-run] 아무것도 쓰지 않았다.")
        return

    path = R.write(rows)
    print(f"\n[save] {path} ({len(pd.read_csv(path)):,}행)")

    legacy_dir = C.OUTPUT_DIR / "legacy"
    if src != legacy_dir:
        legacy_dir.mkdir(exist_ok=True)
        for p in legacy:
            shutil.move(str(p), str(legacy_dir / p.name))
        print(f"[move] 원본 {len(legacy)}개 -> {legacy_dir}/ (실험 기록이므로 보존)")


if __name__ == "__main__":
    main()
