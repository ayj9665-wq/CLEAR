"""
experiments/mitigate_loss.py — 손실 벌점 완화 스윕 (처방 단계, in-processing)

08(후처리)과 09(엣지 수정)에 이은 세 번째 처방이며, 앞선 두 실패·제약에서 나온 것이다.

  - mitigate_graph는 실패했다. geo 엣지의 인종 동종성을 없애도 잔여 격차가 안 줄었다. 누수가
    엣지 짝짓기가 아니라 **블록 소속(지리)** 이기 때문이다. 정보를 입력에서 지우려는
    접근의 한계다.
  - 08은 성공했지만 **배포 시점에 인종을 알아야** 한다. 형사사법 맥락에서 그 제약은
    가볍지 않다.

그래서 여기서는 정보를 지우는 대신 **모델이 그걸 쓰지 못하게** 한다:

    loss = BCEWithLogits + alpha * (그룹 평균 예측확률의 크기가중 분산)

누수 경로가 무엇이든(엣지든 블록이든 특성이든) 결과 단계에서 격차를 누르므로
mitigate_graph의 실패 원인에 걸리지 않는다. 그리고 **민감속성은 학습 때만 필요하고 추론 때는
불필요**하다 — mitigate_threshold의 제약을 정확히 피한다.

alpha가 트레이드오프 곡선의 노브다. alpha=0은 완화 없음이고, mitigate_threshold의 곡선과 같은 축
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

import config as C
from clear import gnn, results, sweep

import torch


def main():
    ap = argparse.ArgumentParser()
    sweep.add_common_args(ap)   # --attr/--edge_type/--k_neighbors/--seeds/--min_n/--sighted
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 10.0, 50.0, 200.0, 1000.0],
                    help="손실 벌점 세기 격자")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="조기 종료 기준을 'val MCC - beta * val 선택률격차'로 바꾼다. "
                         "0이면 예전처럼 val MCC만 본다. alpha가 클 때 손실과 모델 선택이 "
                         "서로 싸우는 문제(곡선 뒤집힘)를 겨냥한 것.")
    ap.add_argument("--grad_clip", type=float, default=0.0,
                    help="0보다 크면 매 스텝 grad L2 norm을 이 값으로 클리핑한다. "
                         "높은 alpha에서 벌점이 BCE를 압도해 생기는 최적화 불안정"
                         "(§10-2, MCC seed std 급증)을 겨냥한 것. 0이면 클리핑 없음.")
    args = ap.parse_args()

    su = sweep.setup(args)

    # 벌점 대상 그룹 = 격차를 재는 그룹집합과 동일하게 맞춘다. 그 외(소수 그룹·
    # Unknown)는 -1로 두어 벌점에서 빠진다 — 재는 것과 누르는 것이 다르면 곡선을
    # 해석할 수 없다.
    #
    # **개수는 test 집합에서 센다.** sweep.run_point가 격차를 잴 때 test 예측 덤프로
    # F.select_groups(cnt, min_n)을 부르므로, 전체 표본으로 세면 두 기준이 갈린다.
    # 3개 주에서는 우연히 같았다(Asian/PI가 전체 4,770 / test 1,431로 양쪽 다 5000
    # 미만) 전국에서 갈라진다 — 전체 9,890(포함) vs test 2,940(제외). 그대로 두면
    # 벌점은 3개 그룹을 누르는데 보고되는 격차는 White-Black 2개 그룹이라,
    # "두 그룹이면 이 벌점은 선택률 격차의 제곱"이라는 이 방법의 근거 자체가 깨진다.
    counts = su.sens.iloc[su.test_idx][su.attr_col].value_counts()
    targets = [g for g in counts.index
               if g != C.FAIRNESS_UNKNOWN_LABEL and counts[g] >= args.min_n]
    idx_of = {g: i for i, g in enumerate(sorted(targets))}
    codes_np = su.sens[su.attr_col].map(lambda g: idx_of.get(g, -1)).values.astype(np.int64)
    fair_codes = torch.tensor(codes_np, device=su.device)
    print(f"[load] X {su.X.shape} (blind={su.blind}), 벌점 대상 그룹 {sorted(targets)} "
          f"/ 제외 {int((codes_np < 0).sum()):,}행, device={su.device}")

    # 그래프는 alpha 격자 내내 동일하므로 한 번만 만들어 재사용한다(mitigate_graph와 달리
    # 개입이 손실 쪽에 있어 그래프가 안 바뀐다).
    data = gnn.build_data(su.X, args.edge_type, args.k_neighbors, su.device,
                          minibatch=su.hp["minibatch"])
    print(f"[graph:{args.edge_type}] 엣지 {data.edge_index.shape[1]:,}개(방향)")

    for alpha in args.alphas:
        tag = (f"fairloss_a{alpha:g}" + (f"_b{args.beta:g}" if args.beta else "")
               + (f"_gc{args.grad_clip:g}" if args.grad_clip else "")
               + ("" if args.attr == "Victim Race" else f"_{args.attr.split()[-1].lower()}")
               + ("" if su.blind else "_sighted"))
        print(f"\n=== {tag} ===")
        sweep.run_point(su, tag, data, "mitigate_loss",
                        {"alpha": alpha, "beta": args.beta, "grad_clip": args.grad_clip},
                        fair_alpha=alpha, fair_codes=fair_codes, fair_beta=args.beta)

    # 결과는 run_point가 outputs/results.csv에 남긴다(예전 fairloss_tradeoff.csv).
    # 여러 곡선(attribute·beta·grad_clip·blind 조합)이 한 테이블에 공존하고, 같은
    # 조합의 재실행은 clear.results.write의 KEY 교체가 처리한다 — 예전에는 이걸
    # 이 파일에서 merge-on-key로 손수 했고, 새 열이 생길 때마다 backfill이 필요했다.
    df = results.read(family="mitigate_loss")
    df = df[df["seed"].isna() & (df["attribute"] == args.attr)
            & (df["blind"] == su.blind)
            & df["metric"].isin(["mcc", "selection_rate_gap",
                                 "selection_rate_amplification"])]
    view = results.wide(df, index=["tag"])
    for knob in ["alpha", "beta", "grad_clip"]:
        view[knob] = results.param(df.drop_duplicates("tag"), knob).values
    view = view.sort_values(["grad_clip", "beta", "alpha"])
    print(f"\n[save] {results.results_path()} (family=mitigate_loss)")
    print(view[["alpha", "beta", "grad_clip", "mcc", "selection_rate_gap",
                "selection_rate_amplification"]].round(4).to_string(index=False))
    print("\n[해석] alpha를 키우면 격차는 줄고 정확도는 떨어져야 한다. 08(후처리) 곡선과 "
          "같은 축에 겹쳐 '같은 공정성 수준에서 어느 쪽이 정확한가'를 본다. 이 방식은 "
          "추론 시점에 민감속성이 필요 없다는 점이 mitigate_threshold과의 실질적 차이다.")


if __name__ == "__main__":
    main()
