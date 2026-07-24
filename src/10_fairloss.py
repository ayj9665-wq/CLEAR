"""
10_fairloss.py — 손실 벌점 완화 스윕 (처방 단계, in-processing)

08(후처리)과 09(엣지 수정)에 이은 세 번째 처방이며, 앞선 두 실패·제약에서 나온 것이다.

  - 09는 실패했다. geo 엣지의 인종 동종성을 없애도 잔여 격차가 안 줄었다. 누수가
    엣지 짝짓기가 아니라 **블록 소속(지리)** 이기 때문이다. 정보를 입력에서 지우려는
    접근의 한계다.
  - 08은 성공했지만 **배포 시점에 인종을 알아야** 한다. 형사사법 맥락에서 그 제약은
    가볍지 않다.

그래서 여기서는 정보를 지우는 대신 **모델이 그걸 쓰지 못하게** 한다:

    loss = BCEWithLogits + alpha * (그룹 평균 예측확률의 크기가중 분산)

누수 경로가 무엇이든(엣지든 블록이든 특성이든) 결과 단계에서 격차를 누르므로
09의 실패 원인에 걸리지 않는다. 그리고 **민감속성은 학습 때만 필요하고 추론 때는
불필요**하다 — 08의 제약을 정확히 피한다.

alpha가 트레이드오프 곡선의 노브다. alpha=0은 완화 없음이고, 08의 곡선과 같은 축
(MCC vs 증폭비)에 겹쳐 그릴 수 있다.

기본은 blind 조건이다. 인종 열이 X에 있으면 모델이 인종을 직접 보므로 완화의
난이도가 달라진다(보고서 §5-1) — 다만 --sighted로 그 조건도 잴 수 있다.

출력:
  outputs/fairloss_tradeoff.csv                alpha당 정확도·격차·CI
  outputs/predictions_graphsage_fairloss_a{...}.csv
  outputs/metrics.csv                          원장에 seed별 append(fair_alpha 열)
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear.data import load_xy, load_sensitive
from clear import fairness as F, gnn, ledger, predictions

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="Victim Race", help="벌점을 걸 민감속성")
    ap.add_argument("--edge_type", default=C.GNN_DEFAULT_EDGE_TYPE)
    ap.add_argument("--k_neighbors", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 10.0, 50.0, 200.0, 1000.0],
                    help="손실 벌점 세기 격자")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="조기 종료 기준을 'val MCC - beta * val 선택률격차'로 바꾼다. "
                         "0이면 예전처럼 val MCC만 본다. alpha가 클 때 손실과 모델 선택이 "
                         "서로 싸우는 문제(곡선 뒤집힘)를 겨냥한 것.")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--min_n", type=int, default=5000,
                    help="벌점·격차 대상 그룹의 최소 표본수(기본 5000 = 인종은 White/Black)")
    ap.add_argument("--sighted", action="store_true", help="민감속성 열을 X에 남긴 채 실행")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else C.GNN_SEEDS)
    blind = not args.sighted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_xy(blind=blind)
    sens = load_sensitive()
    attr_col = f"sens__{args.attr}"

    # 벌점 대상 그룹 = 격차를 재는 그룹집합과 동일하게 맞춘다. 그 외(소수 그룹·
    # Unknown)는 -1로 두어 벌점에서 빠진다 — 재는 것과 누르는 것이 다르면 곡선을
    # 해석할 수 없다.
    counts = sens[attr_col].value_counts()
    targets = [g for g in counts.index
               if g != C.FAIRNESS_UNKNOWN_LABEL and counts[g] >= args.min_n]
    idx_of = {g: i for i, g in enumerate(sorted(targets))}
    codes_np = sens[attr_col].map(lambda g: idx_of.get(g, -1)).values.astype(np.int64)
    fair_codes = torch.tensor(codes_np, device=device)
    print(f"[load] X {X.shape} (blind={blind}), 벌점 대상 그룹 {sorted(targets)} "
          f"/ 제외 {int((codes_np < 0).sum()):,}행, device={device}")

    y_t, train_t, val_t, test_t = gnn.prepare(X, y, C.GNN_VAL_SIZE, device)
    test_idx = test_t.cpu().numpy()
    data = gnn.build_data(X, args.edge_type, args.k_neighbors, device)
    print(f"[graph:{args.edge_type}] 엣지 {data.edge_index.shape[1]:,}개(방향)")

    hp = dict(hidden_dim=C.GNN_HIDDEN_DIM, num_layers=C.GNN_NUM_LAYERS,
              dropout=C.GNN_DROPOUT, lr=C.GNN_LR, weight_decay=C.GNN_WEIGHT_DECAY,
              aggr=C.GNN_AGGR, max_epochs=C.GNN_MAX_EPOCHS, patience=C.GNN_PATIENCE,
              val_size=C.GNN_VAL_SIZE)

    rows = []
    for alpha in args.alphas:
        tag = (f"fairloss_a{alpha:g}" + (f"_b{args.beta:g}" if args.beta else "")
               + ("" if blind else "_sighted"))
        print(f"\n=== {tag} ===")
        seed_rows = []
        for seed in seeds:
            r = gnn.train_one(args.edge_type, data, y_t, train_t, val_t, test_t,
                              seed=seed, k_neighbors=args.k_neighbors, tag=tag,
                              device=device, fair_alpha=alpha, fair_codes=fair_codes,
                              fair_beta=args.beta, **hp)
            ledger.append(r)
            seed_rows.append(r)

        path = predictions.path_for("graphsage", tag)
        predictions.dump(path, test_idx, y,
                         np.stack([r["_test_proba"] for r in seed_rows]).mean(0))

        dump = predictions.load(path)
        cnt, _ = F.joint_counts({"_": dump}, attr_col)
        groups = F.select_groups(cnt, args.min_n)
        pt, st = F.gap_samples(cnt, F.bootstrap_joint(cnt), groups)
        gap = F.gap_row(st, pt, groups, f"named_n>={args.min_n}")

        acc = {k: float(np.mean([r[k] for r in seed_rows])) for k in
               ["auc", "mcc", "f1", "sensitivity", "specificity",
                "balanced_accuracy", "precision"]}
        rows.append({
            "tag": tag, "alpha": alpha, "beta": args.beta,
            "blind": blind, "attribute": args.attr,
            "best_val_gap": float(np.mean([r["best_val_gap"] for r in seed_rows]))
            if args.beta else np.nan,
            **{f"acc_{k}": v for k, v in acc.items()},
            "acc_mcc_std": float(np.std([r["mcc"] for r in seed_rows], ddof=1))
            if len(seeds) > 1 else np.nan,
            "dp_gap": gap["selection_rate_gap"],
            "dp_amplification": gap["selection_rate_amplification"],
            "dp_amplification_lo": gap["selection_rate_amplification_lo"],
            "dp_amplification_hi": gap["selection_rate_amplification_hi"],
            "tpr_gap": gap["tpr_gap"], "base_rate_gap": gap["base_rate_gap"],
        })
        print(f"[{tag}] MCC {acc['mcc']:.4f}  격차 {gap['selection_rate_gap']:.4f}  "
              f"증폭비 {gap['selection_rate_amplification']:.2f} "
              f"[{gap['selection_rate_amplification_lo']:.2f}, "
              f"{gap['selection_rate_amplification_hi']:.2f}]")

    out = pd.DataFrame(rows)
    path = C.OUTPUT_DIR / "fairloss_tradeoff.csv"
    # beta별 곡선을 한 파일에 모은다(같은 축에 겹쳐 그려야 비교가 된다). 같은
    # (alpha, beta, blind) 조합은 최신 실행으로 교체 — 덮어쓰면 이전 beta 곡선이,
    # 그냥 append하면 재실행분이 중복으로 남는다.
    if path.exists():
        old = pd.read_csv(path)
        if {"alpha", "beta", "blind"} <= set(old.columns):
            key = ["alpha", "beta", "blind"]
            merged = pd.MultiIndex.from_frame(out[key])
            old = old[~pd.MultiIndex.from_frame(old[key]).isin(merged)]
            out = pd.concat([old, out], ignore_index=True)
    out = out.sort_values(["blind", "beta", "alpha"]).reset_index(drop=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {path} (누적 {len(out)}행)")
    print(out[["alpha", "beta", "acc_mcc", "acc_mcc_std", "dp_gap", "dp_amplification"]]
          .round(4).to_string(index=False))
    print("\n[해석] alpha를 키우면 격차는 줄고 정확도는 떨어져야 한다. 08(후처리) 곡선과 "
          "같은 축에 겹쳐 '같은 공정성 수준에서 어느 쪽이 정확한가'를 본다. 이 방식은 "
          "추론 시점에 민감속성이 필요 없다는 점이 08과의 실질적 차이다.")


if __name__ == "__main__":
    main()
