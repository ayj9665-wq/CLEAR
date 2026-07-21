"""
ablation_sweep.py — 06_train_gnn.py 하이퍼파라미터 one-factor-at-a-time 스윕

파이프라인 번호가 없는 유틸리티(eda.py와 같은 위치) — 새 산출물을 만들지
않고 06_train_gnn.py를 반복 호출해 outputs/gnn_metrics.csv를 채우는
드라이버다. 기본값(hidden=64, layers=2, dropout=0.3, lr=0.01, wd=5e-4,
aggr=mean)에서 한 축씩만 바꿔가며 각 하이퍼파라미터의 개별 효과를 본다
(그리드 전체 조합이 아니라 ablation 정의 그대로 "한 요인씩 떼어보기").

모든 실행은 --edge_type geo(config.GNN_DEFAULT_EDGE_TYPE), --tag
hparam_sweep으로 고정 — 스윕이 끝난 뒤 tag로 걸러 요약한다.

실행: python ablation_sweep.py
"""
import subprocess
import sys

import pandas as pd

import config as C

# (하이퍼파라미터 이름, CLI 플래그, 기본값에서 시도해볼 값 목록)
# 기본값 자체는 따로 실행하지 않는다 — 이미 gnn_metrics.csv에 있음(default_run).
GRID = [
    ("hidden_dim", "--hidden_dim", [32, 128, 256]),
    ("num_layers", "--num_layers", [1, 3, 4]),
    ("dropout", "--dropout", [0.0, 0.5]),
    ("lr", "--lr", [0.005, 0.02, 0.05]),
    ("weight_decay", "--weight_decay", [0.0, 5e-3]),
    ("aggr", "--aggr", ["max"]),
]

def run_one(flag, value, tag):
    cmd = [
        sys.executable, "06_train_gnn.py",
        "--edge_type", C.GNN_DEFAULT_EDGE_TYPE,
        flag, str(value),
        "--tag", tag,
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    for name, flag, values in GRID:
        tag = f"sweep_{name}"
        for v in values:
            run_one(flag, v, tag)

    df = pd.read_csv(C.OUTPUT_DIR / "gnn_metrics.csv")
    default_row = df[df["tag"].isna()].sort_values("timestamp").iloc[[-1]]  # 가장 최근 기본값 실행

    print(f"\n[summary] one-factor-at-a-time 스윕 결과 (기본값 대비)")
    best_overall = default_row.iloc[0]
    for name, flag, values in GRID:
        tag = f"sweep_{name}"
        sub = pd.concat([default_row, df[df["tag"] == tag]])
        sub = sub[[name, "balanced_accuracy", "precision", "roc_auc", "pr_auc"]] \
            .sort_values("balanced_accuracy", ascending=False)
        print(f"\n-- {name} (기본값 포함) --")
        print(sub.to_string(index=False))

        best_in_dim = df[df["tag"] == tag].sort_values("balanced_accuracy", ascending=False).iloc[0]
        if best_in_dim["balanced_accuracy"] > best_overall["balanced_accuracy"]:
            best_overall = best_in_dim

    print(f"\n[best] balanced_accuracy={best_overall['balanced_accuracy']:.4f}  "
          f"tag={best_overall['tag']}  "
          f"({', '.join(f'{n}={best_overall[n]}' for n, _, _ in GRID)})")
    print(f"[baseline(default)] balanced_accuracy={default_row.iloc[0]['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
