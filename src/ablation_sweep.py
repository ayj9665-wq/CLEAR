"""ablation_sweep.py — GNN 하이퍼파라미터 one-factor-at-a-time 스윕 (in-process)

파이프라인 번호가 없는 유틸리티(eda.py와 같은 위치). config.py 기본값에서
한 축씩만 바꿔가며 각 하이퍼파라미터의 개별 효과를 본다.

이전에는 06_train_gnn.py를 subprocess로 config마다 다시 띄웠다(매번 Python·
torch 재import, parquet 재로드, 게다가 "기본값 행이 미리 있어야 한다"는 숨은
전제까지). 이제 clear.gnn을 직접 import해 데이터를 한 번만 로드하고, 기본값도
여기서 tag="default"로 직접 돌린다. 각 config는 config.GNN_SEEDS로 반복해
outputs/metrics.csv에 seed별 행을 남기고, 요약은 mean±std로 낸다 — 임계값
의존 지표가 run마다 흔들려 단일 run 비교가 신뢰할 수 없기 때문.

실행: python ablation_sweep.py
"""
import config as C
from clear.data import load_xy
from clear.gnn import prepare, train_eval
from clear import ledger

import torch

# (하이퍼파라미터 이름/열, 기본값에서 시도해볼 값 목록). 기본값은 tag="default"로
# 따로 한 번 돌린다 — 각 축 비교에서 공통 기준점이 된다.
GRID = [
    ("hidden_dim", [32, 128, 256]),
    ("num_layers", [1, 3, 4]),
    ("dropout", [0.0, 0.5]),
    ("lr", [0.001, 0.002, 0.01, 0.02]),
    ("weight_decay", [0.0, 5e-3]),
    ("aggr", ["max"]),
]

SUMMARY_COLS = ["n", "mcc_mean", "mcc_std", "auc_mean", "auc_std",
                "f1_mean", "sensitivity_mean", "specificity_mean"]


def default_hp():
    return dict(
        hidden_dim=C.GNN_HIDDEN_DIM, num_layers=C.GNN_NUM_LAYERS, dropout=C.GNN_DROPOUT,
        lr=C.GNN_LR, weight_decay=C.GNN_WEIGHT_DECAY, aggr=C.GNN_AGGR,
        max_epochs=C.GNN_MAX_EPOCHS, patience=C.GNN_PATIENCE, val_size=C.GNN_VAL_SIZE,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_xy()
    y_t, train_t, val_t, test_t = prepare(X, y, C.GNN_VAL_SIZE, device)
    edge, k, seeds = C.GNN_DEFAULT_EDGE_TYPE, C.K_NEIGHBORS, C.GNN_SEEDS
    print(f"[setup] edge={edge} k={k} seeds={seeds} device={device}")

    def run(hp, tag):
        train_eval(edge, k, hp, seeds, tag, X, y_t, train_t, val_t, test_t, device, ledger=ledger)

    # 기본값(공통 기준점) + 각 축 한 요인씩
    run(default_hp(), "default")
    for name, values in GRID:
        for v in values:
            hp = default_hp()
            hp[name] = v
            run(hp, f"sweep_{name}")

    # ---- 요약: 축별 mean±std(기본값 포함), MCC 기준 ----
    gnn = ledger.load()
    gnn = gnn[(gnn["model"] == "graphsage") & (gnn["edge_type"] == edge)]
    print("\n[summary] one-factor-at-a-time (기본값 포함, seed 반복 mean±std, MCC 정렬)")
    for name, _values in GRID:
        subset = gnn[gnn["tag"].isin(["default", f"sweep_{name}"])]
        summary = ledger.summarize(subset, group_cols=[name])
        cols = [name] + [c for c in SUMMARY_COLS if c in summary.columns]
        print(f"\n-- {name} --")
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 전체 best config(하이퍼파라미터 조합별 mean±std) ----
    hp_cols = ["hidden_dim", "num_layers", "dropout", "lr", "weight_decay", "aggr"]
    overall = ledger.summarize(gnn, group_cols=hp_cols)
    best = overall.iloc[0]
    print("\n[best config] " + "  ".join(f"{c}={best[c]}" for c in hp_cols))
    print(f"  mcc {best['mcc_mean']:.4f}±{best['mcc_std']:.4f}  "
          f"auc {best['auc_mean']:.4f}±{best['auc_std']:.4f}  (n={int(best['n'])})")


if __name__ == "__main__":
    main()
