"""
experiments/mitigate_graph.py — 엣지 수준 완화 스윕 (처방 단계, mitigate_threshold의 그래프판)

experiments/mitigate_threshold.py가 예측을 사후 보정했다면, 여기서는 **원인을 직접 건드린다** —
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

from clear.graph import load_edges
from clear import fairgraph, results, sweep

import torch
from torch_geometric.data import Data


def main():
    ap = argparse.ArgumentParser()
    sweep.add_common_args(ap)   # --attr/--edge_type/--k_neighbors/--seeds/--min_n/--sighted
    ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="동종 엣지 제거 확률 격자")
    ap.add_argument("--modes", nargs="+", default=["homophily", "random"],
                    choices=["homophily", "random"])
    args = ap.parse_args()

    su = sweep.setup(args)
    codes = pd.Categorical(su.sens[su.attr_col]).codes.astype(np.int64)
    edges = load_edges(args.edge_type, args.k_neighbors)
    print(f"[load] X {su.X.shape} (blind={su.blind}), {args.edge_type} k={args.k_neighbors}, "
          f"device={su.device}, seeds={su.seeds}")

    # 노드 특성은 p·mode에 무관하므로 한 번만 텐서화하고 그래프만 갈아 끼운다.
    x_tensor = torch.tensor(su.X.values.astype(np.float32))
    # p=0은 두 팔이 같은 그래프이므로 한 번만 돈다.
    configs = [(m, p) for p in args.p for m in args.modes if not (p == 0 and m != args.modes[0])]

    for mode, p in configs:
        edge_index, stats = fairgraph.build(edges, codes, p, mode)
        tag = f"fair_{mode}_p{p:g}" + ("" if su.blind else "_sighted")
        print(f"\n=== {tag} === 쌍 {stats['n_pairs']:,}/{stats['n_pairs_full']:,} "
              f"({stats['kept_frac']:.1%}), 동종성 {stats['homophily_full']:.3f} -> "
              f"{stats['homophily']:.3f}")

        data = Data(x=x_tensor, edge_index=torch.from_numpy(edge_index).long()).to(su.device)
        # 노브는 (mode, p)뿐이다. 그래프 통계(kept_frac·homophily 등)는 그 결과로
        # 만들어진 값이므로 notes에 남긴다 — "동종성을 얼마나 없앴을 때 격차가
        # 어떻게 됐나"를 되짚는 데 필요하지만 결과의 identity는 아니다.
        sweep.run_point(su, tag, data, "mitigate_graph",
                        {"mode": mode, "p": p}, notes=stats)

    # 결과는 run_point가 outputs/results.csv에 남긴다(예전 fairgraph_tradeoff.csv).
    df = results.read(family="mitigate_graph")
    df = df[df["seed"].isna() & df["metric"].isin(["mcc", "selection_rate_amplification"])]
    wide = results.wide(df, index=["tag"])
    wide.insert(1, "p", results.param(df.drop_duplicates("tag"), "p").values)
    print(f"\n[save] {results.results_path()} (family=mitigate_graph)")
    print(wide.round(4).to_string(index=False))
    print("\n[해석] homophily 팔에서만 증폭비가 떨어지고 random 팔은 그대로여야 "
          "동종성이 원인이라는 결론이 선다. 두 팔이 같이 떨어지면 원인은 동종성이 "
          "아니라 그래프 희박화다.")


if __name__ == "__main__":
    main()
