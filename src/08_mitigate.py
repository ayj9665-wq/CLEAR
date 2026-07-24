"""
08_mitigate.py — 완화기법(후처리) CLI (진단 → **처방** 단계, week 3)

07_fairness.py가 "모델이 격차를 키운다"를 보였다면, 여기서는 **얼마나 줄일 수 있고
정확도를 얼마나 잃는가**를 잰다. 계산은 clear.mitigate에 있고 이 파일은 덤프 수집 →
lambda 훑기 → CSV 저장만 한다.

방법은 **그룹별 임계값 후처리**다. 모델을 재학습하지 않고 저장된 확률에 그룹마다
다른 임계값을 적용한다. lambda=0이 원본, lambda=1이 격차 0이며, 그 사이를 훑으면
**정확도-공정성 트레이드오프 곡선**이 나온다(포스터 두 번째 패널).

이 완화는 **모든 모델에 똑같이 적용된다.** GNN 전용 완화(엣지 수준, 다음 단계)만
하고 원본 XGBoost와 대면시키면 조작이므로, 공통 축을 먼저 세워둔다.

임계값은 test를 층화 분할한 tune 절반에서 정하고 eval 절반에서만 평가한다 —
같은 데이터에서 맞추고 재면 완화 효과가 낙관적으로 나오기 때문. 따라서 여기 나오는
정확도는 07의 수치와 직접 비교하면 안 된다(평가 표본이 절반이고 다르다).
lambda=0 행이 그 조건에서의 정당한 기준선이다.

출력: outputs/mitigation_tradeoff.csv
  (model x attribute x criterion x lambda)당 정확도 지표 + 격차·증폭비 + 사용 임계값

주의: AUC는 순위 기반이라 임계값을 바꿔도 변하지 않는다. 곡선에서 acc_auc가
평평한 것은 버그가 아니라 정의상 당연하며, 그래서 대표 지표로는 MCC를 본다.
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear import mitigate, predictions

SENS_ATTRS = [f"sens__{a}" for a in C.SENSITIVE_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="완화할 덤프 라벨. 생략 시 찾은 전부.")
    ap.add_argument("--criteria", nargs="+", default=list(mitigate.CRITERIA),
                    choices=list(mitigate.CRITERIA),
                    help="dp=선택률 균등, tpr=검거 재현율 균등")
    ap.add_argument("--steps", type=int, default=11, help="lambda 격자 점 개수(0~1)")
    ap.add_argument("--min_n", type=int, default=5000,
                    help="완화·격차 대상 그룹의 최소 표본수(기본 5000 = 인종은 White/Black)")
    ap.add_argument("--n_boot", type=int, default=C.FAIRNESS_BOOTSTRAP_N)
    args = ap.parse_args()

    paths = predictions.discover(args.models)
    if not paths:
        raise SystemExit("[에러] outputs/predictions_*.csv 없음. 먼저 05/06을 실행할 것.")
    dumps = {label: predictions.load(p) for label, p in paths.items()}
    predictions.assert_same_test_set(dumps)
    lambdas = np.round(np.linspace(0, 1, args.steps), 4)
    print(f"[load] 덤프 {len(dumps)}종 / lambda {args.steps}점 / 기준 {args.criteria} "
          f"/ 그룹하한 {args.min_n}")

    out = []
    for label, df in dumps.items():
        for attr in SENS_ATTRS:
            if attr not in df.columns:
                continue
            name = attr.replace("sens__", "")
            for crit in args.criteria:
                res = mitigate.sweep(df, attr, lambdas=lambdas, criterion=crit,
                                     min_n=args.min_n, n_boot=args.n_boot)
                res.insert(0, "attribute", name)
                res.insert(0, "model", label)
                out.append(res)

                base, full = res.iloc[0], res.iloc[-1]
                # 기준이 겨냥한 격차를 찍는다(dp면 선택률, tpr이면 재현율).
                # 기준을 무시하고 늘 dp를 찍으면 tpr 완화가 실패한 것처럼 보인다.
                col = f"{crit}_gap"
                best = res.loc[res["acc_mcc"].idxmax()]
                print(f"  [{label} / {name} / {crit}] "
                      f"MCC {base['acc_mcc']:.3f} -> {full['acc_mcc']:.3f} "
                      f"({full['acc_mcc'] - base['acc_mcc']:+.3f}, "
                      f"최고 {best['acc_mcc']:.3f} @L={best['lambda']:.1f}) | "
                      f"{crit} 격차 {base[col]:.3f} -> {full[col]:.3f}")

    df_out = pd.concat(out, ignore_index=True)
    path = C.OUTPUT_DIR / "mitigation_tradeoff.csv"
    df_out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {path}  ({len(df_out)}행)")
    print("[해석] lambda를 키우면 격차는 줄고 정확도는 떨어진다. 두 모델의 곡선을 "
          "겹쳐 그려 '같은 공정성 수준에서 누가 더 정확한가'를 본다. 한 점끼리 "
          "비교하면 완화 세기가 달라 무의미하다.")


if __name__ == "__main__":
    main()
