"""
experiments/train_gnn.py — GraphSAGE 학습 CLI (얇은 래퍼)

모델·학습 로직은 clear/gnn.py에 있다(스윕 스크립트들이 subprocess 대신
in-process로 재사용하기 위함). 이 파일은 인자 파싱 → 데이터 로드 → 분할 →
config별 seed 반복 학습 → 원장(outputs/metrics.csv) append만 담당한다.

인자 없이 실행하면 config.py 기본 하이퍼파라미터(geo, k=20, lr=0.005 등 —
ablation 결과로 승격된 값)와 config.GNN_SEEDS로 학습한다. 하이퍼파라미터는
전부 argparse로 오버라이드 가능(개별 실험용).

평가지표: AUC / MCC / F1 / Sensitivity / Specificity (+ 논문 대조용 Balanced
Accuracy / Precision) — clear.metrics의 단일 정의를 쓴다. MCC를
대표 지표로 val 조기 종료·순위에 쓴다.

기본으로 edge_type별 test 노드 예측(seed 평균 proba + 민감속성)을
outputs/predictions_graphsage_{edge}.csv로 저장한다(--no_dump_predictions로 끔) —
experiments/diagnose_fairness.py의 그룹별 공정성 진단 입력.

seed 반복: split은 config.RANDOM_STATE로 고정(=커밋된 평면 모델 덤프와 동일 test 집합, 비교
가능성 유지)하고 torch seed만 --seeds로 바꿔 학습 분산(mean±std)을 남긴다.
임계값 의존 지표는 run마다 ±3~4점 흔들리므로 단일 run 비교는 신뢰 불가.

흐름:
  features.parquet 로드(X/y) → clear.gnn.prepare로 분할·텐서화
   → 후보별(edge_type): edges_{edge_type}_k{k}.npy 로드 → seed마다 GraphSAGE
     학습(val MCC 조기 종료) → test 평가 → outputs/metrics.csv에 seed별 append
  --edge_type all은 geo/temporal/weapon 3종을, geo_temporal은 두 엣지의
  합집합(clear.graph)을 쓴다.
"""
import argparse

import config as C   # torch import 전에 필요 (KMP_DUPLICATE_LIB_OK 등 env 설정)
from clear.data import load_xy
from clear.gnn import prepare, train_eval, dump_test_predictions, parse_seeds
from clear import predictions, results

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_type",
                         choices=["geo", "temporal", "weapon", "geo_temporal", "all"],
                         default=C.GNN_DEFAULT_EDGE_TYPE)
    parser.add_argument("--k_neighbors", type=int, default=C.K_NEIGHBORS,
                         help="04_build_graph.py --k로 만든 그래프 중 어느 것을 쓸지 선택")
    parser.add_argument("--seeds", type=parse_seeds, default=C.GNN_SEEDS,
                         help="쉼표구분 torch seed 목록(예: 42,43,44). split은 고정.")
    parser.add_argument("--hidden_dim", type=int, default=C.GNN_HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=C.GNN_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=C.GNN_DROPOUT)
    parser.add_argument("--lr", type=float, default=C.GNN_LR)
    parser.add_argument("--weight_decay", type=float, default=C.GNN_WEIGHT_DECAY)
    parser.add_argument("--aggr", default=C.GNN_AGGR)
    parser.add_argument("--max_epochs", type=int, default=C.GNN_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.GNN_PATIENCE)
    parser.add_argument("--val_size", type=float, default=C.GNN_VAL_SIZE)
    parser.add_argument("--tag", default="", help="원장에서 구분할 자유 라벨")
    parser.add_argument("--blind", action="store_true",
                         help="민감속성 더미(config.SENSITIVE_FEATURE_COLS)를 X에서 제외. "
                              "그래프는 그대로이므로 '그래프가 민감속성의 우회 경로인가'의 검정 조건이 된다.")
    parser.add_argument("--dump_predictions", action="store_true", default=True,
                         help="edge_type별 test 노드 예측을 outputs/predictions_graphsage_{edge}.csv로 저장(공정성 진단 07용)")
    parser.add_argument("--no_dump_predictions", dest="dump_predictions", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_xy(blind=args.blind)
    print(f"[load] features {X.shape}, 검거율 {y.mean():.1%}, device={device}, seeds={args.seeds}"
          + (f"  (blind: {C.SENSITIVE_FEATURE_COLS} 더미 제외)" if args.blind else ""))

    y_t, train_t, val_t, test_t = prepare(X, y, args.val_size, device)
    print(f"[split] train {len(train_t):,} / val {len(val_t):,} / test {len(test_t):,} "
          f"(test는 outputs/predictions/의 평면 모델 덤프와 동일 집합)")

    hp = dict(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay, aggr=args.aggr,
        max_epochs=args.max_epochs, patience=args.patience, val_size=args.val_size,
    )
    edge_types = ["geo", "temporal", "weapon"] if args.edge_type == "all" else [args.edge_type]

    test_idx = test_t.cpu().numpy()
    suffix = "_blind" if args.blind else ""
    tag = args.tag or suffix.lstrip("_")
    for edge_type in edge_types:
        rows = train_eval(edge_type, args.k_neighbors, hp, args.seeds, tag,
                          X, y_t, train_t, val_t, test_t, device,
                          family="train", blind=args.blind)
        if args.dump_predictions:
            pred_path = predictions.path_for("graphsage", f"{edge_type}{suffix}")
            dump_test_predictions(rows, test_idx, y, pred_path)
            print(f"[save] {pred_path} (test 노드 {len(test_idx):,}개, seed 평균 proba)")
    print(f"[save] {results.results_path()} (family=train, {len(edge_types)}×{len(args.seeds)}회)")


if __name__ == "__main__":
    main()
