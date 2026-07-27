"""experiments/ablation.py — GNN 하이퍼파라미터 one-factor-at-a-time 스윕 (in-process)

파이프라인 번호가 없는 유틸리티(eda.py와 같은 위치). config.py 기본값에서
한 축씩만 바꿔가며 각 하이퍼파라미터의 개별 효과를 본다.

이전에는 experiments/train_gnn.py를 subprocess로 config마다 다시 띄웠다(매번 Python·
torch 재import, parquet 재로드, 게다가 "기본값 행이 미리 있어야 한다"는 숨은
전제까지). 이제 clear.gnn을 직접 import해 데이터를 한 번만 로드하고, 기본값도
여기서 tag="default"로 직접 돌린다. 각 config는 config.GNN_SEEDS로 반복해
outputs/results.csv에 seed별 행을 남기고, 요약은 mean±std로 낸다 — 임계값
의존 지표가 run마다 흔들려 단일 run 비교가 신뢰할 수 없기 때문.

실행: python experiments/ablation.py            (전 축)
      python experiments/ablation.py --dims lr  (한 축만)
      python experiments/ablation.py --dry_run  (돌릴 config만 출력, 학습 없음)

argparse가 있는 이유: 이 스크립트는 GPU로 수 분~수십 분을 쓰고 그 결과를 추적
대상인 outputs/results.csv에 기록한다. argparse가 없던 시절에는 오타든
--help든 인자가 그냥 무시되고 **본 스윕이 시작됐다** — 결과 테이블에 의도치 않은
행이 쌓이는 경로였다(실제로 한 번 발생). 이제 모르는 인자는 즉시 에러다.
"""
import argparse

import config as C
from clear.data import load_xy
from clear.gnn import prepare, train_eval, default_hp, parse_seeds
from clear import results

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


def main():
    ap = argparse.ArgumentParser(
        description="GNN 하이퍼파라미터 one-factor-at-a-time 스윕. "
                    "GPU로 수 분 이상 쓰고 outputs/results.csv에 기록한다.")
    ap.add_argument("--dims", nargs="+", default=[n for n, _ in GRID],
                    choices=[n for n, _ in GRID],
                    help="훑을 축. 생략 시 전부.")
    ap.add_argument("--edge_type", default=C.GNN_DEFAULT_EDGE_TYPE)
    ap.add_argument("--k_neighbors", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--seeds", type=parse_seeds, default=C.GNN_SEEDS,
                    help="쉼표구분 torch seed. split은 고정.")
    ap.add_argument("--dry_run", action="store_true",
                    help="돌릴 config 목록만 출력하고 종료(학습·결과 기록 없음)")
    args = ap.parse_args()

    grid = [(n, v) for n, v in GRID if n in args.dims]
    edge, k, seeds = args.edge_type, args.k_neighbors, args.seeds
    n_runs = (1 + sum(len(v) for _, v in grid)) * len(seeds)

    if args.dry_run:
        print(f"[dry-run] edge={edge} k={k} seeds={seeds} -> 학습 {n_runs}회")
        print("  default")
        for name, values in grid:
            print(f"  {name}: {values}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_xy()
    y_t, train_t, val_t, test_t = prepare(X, y, C.GNN_VAL_SIZE, device)
    print(f"[setup] edge={edge} k={k} seeds={seeds} device={device} -> 학습 {n_runs}회")

    def run(hp, tag):
        train_eval(edge, k, hp, seeds, tag, X, y_t, train_t, val_t, test_t, device,
                   family="ablation")

    # 기본값(공통 기준점) + 각 축 한 요인씩
    run(default_hp(), "default")
    for name, values in grid:
        for v in values:
            hp = default_hp()
            hp[name] = v
            run(hp, f"sweep_{name}")

    # ---- 요약: 축별 mean±std(기본값 포함), MCC 기준 ----
    # 하이퍼파라미터는 이제 params(JSON)에 있으므로 축 하나를 열로 꺼내 group_by한다.
    df = results.read(family="ablation")
    df = df[(df["model"] == "graphsage")
            & (results.param(df, "edge_type", str) == edge)]
    print("\n[summary] one-factor-at-a-time (기본값 포함, seed 반복 mean±std, MCC 정렬)")
    for name, _values in grid:
        subset = df[df["tag"].isin(["default", f"sweep_{name}"])].copy()
        subset[name] = results.param(subset, name)
        summary = results.summarize(subset, group_cols=[name])
        cols = [name] + [c for c in SUMMARY_COLS if c in summary.columns]
        print(f"\n-- {name} --")
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 전체 best config(하이퍼파라미터 조합별 mean±std) ----
    hp_cols = ["hidden_dim", "num_layers", "dropout", "lr", "weight_decay", "aggr"]
    overall = df.copy()
    for c in hp_cols:
        overall[c] = results.param(overall, c, str)
    overall = results.summarize(overall, group_cols=hp_cols)
    best = overall.iloc[0]
    print("\n[best config] " + "  ".join(f"{c}={best[c]}" for c in hp_cols))
    print(f"  mcc {best['mcc_mean']:.4f}±{best['mcc_std']:.4f}  "
          f"auc {best['auc_mean']:.4f}±{best['auc_std']:.4f}  (n={int(best['n'])})")


if __name__ == "__main__":
    main()
