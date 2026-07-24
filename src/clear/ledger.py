"""공용 실험 결과 원장(ledger) — 05·06이 같은 CSV(outputs/metrics.csv)에 append.

이전에는 baseline_metrics.csv(overwrite 스냅샷)와 gnn_metrics.csv(append 로그)가
서로 다른 생애주기·다른 열 구조라 "GNN이 baseline을 이겼나"가 파일 간 수동
대조였다. 이제 둘 다 model 열을 달고 한 원장에 append하므로 비교가 groupby 한
줄이다. 한 config를 여러 seed로 돌린 각 행이 그대로 쌓이고(long 포맷),
집계(mean±std)는 summarize()가 view로 제공한다 — 원본은 seed별 raw 행이 진실.

열 정렬 주의: baseline 행은 GNN 전용 열이 없으므로, append가 모든 행을
LEDGER_COLS로 reindex해(없는 열은 NaN) CSV append 시 열이 어긋나지 않게 한다.
"""
import pandas as pd

import config as C

LEDGER_PATH = C.OUTPUT_DIR / "metrics.csv"

# clear.metrics.METRIC_NAMES와 동일 순서 유지. 뒤쪽 둘은 논문(Balanced Acc/
# Precision) 대조용 부가 지표라 기존 다섯 뒤에 붙였다(기존 CSV 열 정렬 보존).
METRIC_COLS = ["auc", "mcc", "f1", "sensitivity", "specificity",
               "balanced_accuracy", "precision"]

# 고정 스키마: 공통 앞부분 + GNN 전용 뒷부분(baseline 행에선 NaN).
LEDGER_COLS = [
    "timestamp", "model", "tag", "seed",
    *METRIC_COLS,
    "edge_type", "k_neighbors", "hidden_dim", "num_layers", "dropout",
    "lr", "weight_decay", "aggr", "max_epochs", "patience", "val_size",
    "best_epoch", "train_seconds", "n_params",
    # 완화 실험(10_fairloss.py)의 손실 벌점 세기. 이전 행에서는 비어 있다.
    "fair_alpha",
]


def append(row, path=LEDGER_PATH):
    """실행 결과(dict) 한 줄을 원장에 append. 스키마로 reindex해 열 정렬 보장."""
    df = pd.DataFrame([row]).reindex(columns=LEDGER_COLS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    return path


def load(path=LEDGER_PATH):
    return pd.read_csv(path)


def summarize(df, group_cols=("model", "tag"), metrics=METRIC_COLS, sort_by="mcc"):
    """(config)별 seed 반복의 mean/std/n. 지표 대표값(기본 MCC) mean 내림차순 정렬.

    반환 열: group_cols + n + 각 지표의 {m}_mean/{m}_std(평탄화 — MultiIndex 아님).
    """
    group_cols = list(group_cols)
    agg = {"n": ("seed", "count")}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    out = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    return out.sort_values(f"{sort_by}_mean", ascending=False)
