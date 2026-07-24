"""
09_fairgraph.py — 엣지 수준 완화 스윕 (처방 단계, 08의 그래프판)

08_mitigate.py가 예측을 사후 보정했다면, 여기서는 **원인을 직접 건드린다** —
인종이 같은 사건끼리 이어진 geo 엣지를 확률 p로 제거하고 GNN을 다시 학습한다.
계산은 clear.fairgraph(엣지 조작) + clear.gnn(학습)에 있고 이 파일은 스윕만 돈다.

**항상 두 팔을 돌린다**:
  homophily  같은 인종끼리 이어진 쌍을 확률 p로 제거
  random     그 결과와 **같은 개수**를 무작위로 남긴 대조군

대조군이 없으면 아무것도 못 말한다 — 동종 엣지를 빼면 엣지 수도 줄기 때문에,
격차가 줄어도 "동종성이 빠져서"인지 "그래프가 희박해져서"인지 구분되지 않는다.
동종 제거에서만 격차가 줄고 무작위 제거에서는 안 줄어야 동종성이 원인이다.

**blind 조건에서만 의미가 있다**(기본값). 인종 열이 X에 있으면 그래프를 어떻게
고쳐도 모델이 인종을 직접 보므로 엣지 개입의 효과가 가려진다(보고서 §5-1).

출력:
  outputs/fairgraph_tradeoff.csv           (mode x p)당 정확도·격차·그래프 통계
  outputs/predictions_graphsage_fair{...}.csv  각 설정의 test 예측(07로 재진단 가능)
  outputs/metrics.csv                      원장에 seed별 append(tag=fairgraph_...)

주의: 여기서 만든 그래프는 학습·추론에 모두 쓰인다. 새 사건의 엣지를 정할 때도
인종을 알아야 하므로, "배포 시 민감속성 불필요"는 이 변형에서는 성립하지 않는다.
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear.data import load_xy, load_sensitive
from clear.graph import load_edges
from clear import fairgraph, fairness as F, gnn, ledger, predictions

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="Victim Race", help="동종성을 깰 민감속성")
    ap.add_argument("--edge_type", default=C.GNN_DEFAULT_EDGE_TYPE)
    ap.add_argument("--k_neighbors", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="동종 엣지 제거 확률 격자")
    ap.add_argument("--modes", nargs="+", default=["homophily", "random"],
                    choices=["homophily", "random"])
    ap.add_argument("--seeds", default=None, help="쉼표구분 torch seed(기본 config.GNN_SEEDS)")
    ap.add_argument("--min_n", type=int, default=5000, help="격차 대상 그룹 최소 표본수")
    ap.add_argument("--sighted", action="store_true",
                    help="민감속성 열을 X에 남긴 채 실행(기본은 blind). 진단상 권장하지 않음")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else C.GNN_SEEDS)
    blind = not args.sighted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_xy(blind=blind)
    sens = load_sensitive()
    attr_col = f"sens__{args.attr}"
    codes = pd.Categorical(sens[attr_col]).codes.astype(np.int64)
    edges = load_edges(args.edge_type, args.k_neighbors)

    print(f"[load] X {X.shape} (blind={blind}), {args.edge_type} k={args.k_neighbors}, "
          f"device={device}, seeds={seeds}")

    y_t, train_t, val_t, test_t = gnn.prepare(X, y, C.GNN_VAL_SIZE, device)
    test_idx = test_t.cpu().numpy()
    hp = dict(hidden_dim=C.GNN_HIDDEN_DIM, num_layers=C.GNN_NUM_LAYERS,
              dropout=C.GNN_DROPOUT, lr=C.GNN_LR, weight_decay=C.GNN_WEIGHT_DECAY,
              aggr=C.GNN_AGGR, max_epochs=C.GNN_MAX_EPOCHS, patience=C.GNN_PATIENCE,
              val_size=C.GNN_VAL_SIZE)

    x_tensor = torch.tensor(X.values.astype(np.float32))
    rows = []
    # p=0은 두 팔이 같은 그래프이므로 한 번만 돈다.
    configs = [(m, p) for p in args.p for m in args.modes if not (p == 0 and m != args.modes[0])]

    for mode, p in configs:
        edge_index, stats = fairgraph.build(edges, codes, p, mode)
        tag = f"fair_{mode}_p{p:g}" + ("" if blind else "_sighted")
        print(f"\n=== {tag} === 쌍 {stats['n_pairs']:,}/{stats['n_pairs_full']:,} "
              f"({stats['kept_frac']:.1%}), 동종성 {stats['homophily_full']:.3f} -> "
              f"{stats['homophily']:.3f}")

        from torch_geometric.data import Data
        data = Data(x=x_tensor, edge_index=torch.from_numpy(edge_index).long()).to(device)

        seed_rows = []
        for seed in seeds:
            r = gnn.train_one(args.edge_type, data, y_t, train_t, val_t, test_t,
                              seed=seed, k_neighbors=args.k_neighbors, tag=tag,
                              device=device, **hp)
            ledger.append(r)
            seed_rows.append(r)

        path = predictions.path_for("graphsage", tag)
        predictions.dump(path, test_idx, y, np.stack([r["_test_proba"] for r in seed_rows]).mean(0))

        # 공정성: 07과 같은 정의(clear.fairness)로 계산해 완화 전후가 비교 가능하게 한다.
        dump = predictions.load(path)
        counts, _ = F.joint_counts({"_": dump}, attr_col)
        groups = F.select_groups(counts, args.min_n)
        boot = F.bootstrap_joint(counts)
        pt, st = F.gap_samples(counts, boot, groups)
        gap = F.gap_row(st, pt, groups, f"named_n>={args.min_n}")

        acc = {k: float(np.mean([r[k] for r in seed_rows])) for k in
               ["auc", "mcc", "f1", "sensitivity", "specificity", "balanced_accuracy", "precision"]}
        mcc_std = float(np.std([r["mcc"] for r in seed_rows], ddof=1)) if len(seeds) > 1 else np.nan
        rows.append({**stats, "tag": tag, "blind": blind, "attribute": args.attr,
                     **{f"acc_{k}": v for k, v in acc.items()}, "acc_mcc_std": mcc_std,
                     "dp_gap": gap["selection_rate_gap"],
                     "dp_amplification": gap["selection_rate_amplification"],
                     "dp_amplification_lo": gap["selection_rate_amplification_lo"],
                     "dp_amplification_hi": gap["selection_rate_amplification_hi"],
                     "tpr_gap": gap["tpr_gap"], "base_rate_gap": gap["base_rate_gap"]})
        print(f"[{tag}] MCC {acc['mcc']:.4f}  증폭비 {gap['selection_rate_amplification']:.2f} "
              f"[{gap['selection_rate_amplification_lo']:.2f}, "
              f"{gap['selection_rate_amplification_hi']:.2f}]")

    out = pd.DataFrame(rows)
    path = C.OUTPUT_DIR / "fairgraph_tradeoff.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {path} ({len(out)}행)")
    print(out[["mode", "p", "kept_frac", "homophily", "acc_mcc", "dp_amplification"]]
          .round(4).to_string(index=False))
    print("\n[해석] homophily 팔에서만 증폭비가 떨어지고 random 팔은 그대로여야 "
          "동종성이 원인이라는 결론이 선다. 두 팔이 같이 떨어지면 원인은 동종성이 "
          "아니라 그래프 희박화다.")


if __name__ == "__main__":
    main()
