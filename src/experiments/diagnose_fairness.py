"""
experiments/diagnose_fairness.py — 공정성 진단 CLI (예측 → **진단** 단계, week 3)

05/06이 저장한 test 예측 덤프(outputs/predictions_*.csv)를 읽어, 피해자 인종·성별
그룹별로 검거 예측이 **체계적으로 불공평한지**를 측정한다. 계산은 전부
clear.fairness에 있고(08 완화 단계가 같은 정의로 재계산해야 하므로) 이 파일은
덤프 수집 → 진단 호출 → CSV 저장만 한다.

기본으로 **찾은 덤프를 전부** 진단한다(graphsage / xgboost / logreg). 모델을
나란히 놓는 게 요점이기 때문이다 — "격차가 큰가"만이 아니라 **"그래프가 flat
모델보다 더 키우는가"**를 물어야 그래프 접근의 공정성 대가를 말할 수 있다.
모든 덤프가 동일 test 집합인지는 clear.predictions가 검증한다.

출력:
  outputs/fairness_group_metrics.csv    그룹별 지표(+ CI) — 그림용 long 포맷
  outputs/fairness_gaps.csv             (model x attribute x group_set)당 격차·증폭비·CI
  outputs/fairness_model_contrasts.csv  모델 쌍의 격차 차이 + CI(짝지은 부트스트랩)
  outputs/fairness_gaps_standardized.csv  같은 격차를 **주(州)로 층 표준화**한 값

네 번째 표가 왜 필요한지는 clear.fairness의 docstring에 있다. 요지: pooled 격차는
지역 구성과 그룹 구성의 교란을 담고, 그 크기가 작지 않다 — 전국 blind 성별 증폭비
1.25는 표준화하면 **1.02**로, 기제가 아니라 주간 구성 효과였다. 같은 검정에서 인종은
1.805 -> 1.558로 **살아남는다.** 두 표를 나란히 읽어야 한다(표준화는 층 하한 때문에
표본 일부를 버리는 다른 추정량이다).

모델 비교는 **반드시** 대조표로 읽는다. 세 모델이 같은 test 행을 쓰므로 개별
모델의 CI가 겹치는지로 차이를 판정하면 틀린다 — 겹치는 독립 CI는 유의한 차이와
얼마든지 공존한다. clear.fairness가 복제본을 모델 간에 공유해 차이를 짝지어 낸다.

핵심 수치는 **증폭비** = 모델 격차 ÷ base_rate(실제) 격차. 1보다 크면 모델이
데이터에 이미 있던 불균형을 키운 것이다. 격차는 명명된 전체 그룹과
n>=config.FAIRNESS_MIN_GROUP_N 두 벌로 함께 낸다 — 이 표본에서 인종 max-min은
n=178 그룹이 결정해 버려서, 한 벌만 보고하면 오도된다.

주의: 아직 **진단만** 한다. 격차를 줄이는 완화(처방, 단계 3)는 다음 스크립트.
"""
import argparse
import sys

import pandas as pd

import config as C
from clear import fairness, predictions

SENS_ATTRS = [f"sens__{a}" for a in C.SENSITIVE_COLS]   # Victim Race, Victim Sex


def _fmt(row, m):
    """격차 한 줄을 '값 [lo, hi]' 꼴로."""
    return f"{row[f'{m}_gap']:.3f} [{row[f'{m}_gap_lo']:.3f}, {row[f'{m}_gap_hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="진단할 덤프 라벨(예: graphsage_geo xgboost). 생략 시 찾은 전부.")
    ap.add_argument("--min_n", type=int, nargs="+", default=[C.FAIRNESS_MIN_GROUP_N],
                    help="격차를 낼 최소 그룹 표본수(여러 개 가능: --min_n 1000 5000). "
                         "명명된 전체 그룹 집합은 항상 함께 낸다.")
    ap.add_argument("--n_boot", type=int, default=C.FAIRNESS_BOOTSTRAP_N,
                    help="부트스트랩 반복수(0이면 CI 없이 점추정만)")
    ap.add_argument("--stratum", default="State",
                    help="층 표준화에 쓸 sample.parquet의 열(기본 State). "
                         "pooled 격차는 지역 구성과 그룹 구성의 교란을 담으므로 "
                         "**표준화 격차를 나란히** 낸다(clear.fairness 참고). "
                         "none이면 표준화를 건너뛴다.")
    ap.add_argument("--stratum_min_n", type=int, default=100,
                    help="층 하나를 표준화에 쓰기 위해 **모든 비교 그룹**이 가져야 하는 "
                         "최소 표본. 한 그룹이라도 비면 그 층의 기여가 그룹마다 달라져 "
                         "'같은 인구에 표준화한다'는 전제가 깨진다.")
    args = ap.parse_args()

    paths = predictions.discover(args.models)
    if not paths:
        sys.exit("[에러] outputs/predictions/ 에 덤프 없음. 먼저 실행: "
                 "python -m experiments.train_gnn")

    dumps = {label: predictions.load(p) for label, p in paths.items()}
    test_idx = predictions.assert_same_test_set(dumps)
    print(f"[load] 덤프 {len(dumps)}종: {', '.join(dumps)} / 공통 test {len(test_idx):,}행")

    for label, df in dumps.items():
        print(f"  {label:16s} 예측 검거율 {df['pred'].mean():.1%} "
              f"(실제 {df['y_true'].mean():.1%})")

    # 층 라벨은 예측 덤프에 없다(row_index가 sample.parquet 행 위치라 join한다 —
    # detect_cold_blocks와 같은 규약). 덤프마다 같은 test 행이므로 한 번 붙이면 된다.
    strata = None
    if args.stratum.lower() != "none":
        sample = pd.read_parquet(C.SCOPE_DIR / "sample.parquet")
        if args.stratum not in sample.columns:
            sys.exit(f"[에러] sample.parquet에 '{args.stratum}' 열 없음")
        strata = sample[args.stratum].values[test_idx]
        print(f"[strat] 층 = {args.stratum} ({pd.Series(strata).nunique()}개), "
              f"층당 그룹 최소 {args.stratum_min_n}")

    all_groups, all_gaps, all_contrasts, all_std = [], [], [], []
    for attr in SENS_ATTRS:
        missing = [l for l, df in dumps.items() if attr not in df.columns]
        if missing:
            print(f"[skip] {attr} 열 없는 덤프: {', '.join(missing)}")
            continue
        name = attr.replace("sens__", "")
        gt, gaps, contrasts = fairness.diagnose(
            dumps, attr, min_n=args.min_n, n_boot=args.n_boot)
        for t in (gt, gaps, contrasts):
            t.insert(0, "attribute", name)
        all_groups.append(gt); all_gaps.append(gaps); all_contrasts.append(contrasts)

        print(f"\n{'=' * 78}\n=== {name} ===")
        for label in dumps:
            sub = gt[gt["model"] == label]
            print(f"\n[{label}]")
            print(sub[["group", "n", "base_rate", "selection_rate", "tpr", "fpr"]]
                  .round(3).to_string(index=False))
            for _, r in gaps[gaps["model"] == label].iterrows():
                amp = (r["selection_rate_amplification"],
                       r["selection_rate_amplification_lo"],
                       r["selection_rate_amplification_hi"])
                print(f"  [{r['group_set']}] n_groups={r['n_groups']}  "
                      f"DP {_fmt(r, 'selection_rate')} | TPR {_fmt(r, 'tpr')} | "
                      f"데이터 {r['base_rate_gap']:.3f}  "
                      f"-> 증폭비 {amp[0]:.2f}배 [{amp[1]:.2f}, {amp[2]:.2f}]")

        # 층 표준화: pooled 격차에 섞인 지역 구성 교란을 뺀 값. pooled를 대체하지
        # 않고 나란히 낸다 -- 표준화는 층 하한 때문에 표본 일부를 버리므로 어느 쪽도
        # 단독으로는 불충분하다.
        if strata is not None:
            std_rows = []
            for label, df in dumps.items():
                d = df.assign(**{args.stratum: strata})
                for _, r in gaps[gaps["model"] == label].iterrows():
                    grp = [g for g in str(r["groups"]).split(" | ") if g]
                    sr = fairness.standardized_gap_row(
                        d, attr, grp, args.stratum, r["group_set"],
                        min_stratum_n=args.stratum_min_n, n_boot=args.n_boot)
                    if sr is not None:
                        std_rows.append({"attribute": name, "model": label, **sr})
            if std_rows:
                std = pd.DataFrame(std_rows)
                all_std.append(std)
                print(f"\n  -- 층 표준화({args.stratum}) 후 증폭비 --")
                for _, r in std.iterrows():
                    pooled = gaps[(gaps["model"] == r["model"])
                                  & (gaps["group_set"] == r["group_set"])]
                    p = float(pooled["selection_rate_amplification"].iloc[0])
                    print(f"  [{r['group_set']}] {r['model']:26s} "
                          f"pooled {p:5.3f} -> 표준화 "
                          f"{r['selection_rate_amplification']:5.3f} "
                          f"[{r['selection_rate_amplification_lo']:.3f}, "
                          f"{r['selection_rate_amplification_hi']:.3f}]  "
                          f"(층 {r['n_strata_used']}/{r['n_strata']}, "
                          f"{r['n_standardized']:,}행)")

        # 모델 대조: 개별 CI 겹침이 아니라 짝지은 차이로 판정한다.
        dp = contrasts[(contrasts["metric"] == "selection_rate")
                       & (contrasts["quantity"] == "amplification")]
        if len(dp):
            print(f"\n  -- 모델 간 증폭비 차이(DP, 짝지은 부트스트랩) --")
            for _, r in dp.iterrows():
                verdict = "유의" if r["significant"] else "판정불가(CI가 0을 걸침)"
                print(f"  [{r['group_set']}] {r['model_a']} - {r['model_b']}: "
                      f"{r['diff']:+.2f}배 [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]  {verdict}")

    paths_out = [
        (C.scoped_output("fairness_group_metrics.csv"), all_groups),
        (C.scoped_output("fairness_gaps.csv"), all_gaps),
        (C.scoped_output("fairness_model_contrasts.csv"), all_contrasts),
    ]
    # 별도 파일이다. fairness_gaps.csv에 열을 더하면 그 표를 인용한 보고서들이
    # 가리키는 수치의 뜻이 바뀌고, 행을 더하면 필터 없이 읽는 쪽이 pooled와 표준화를
    # 섞어 센다. 표준화는 표본 일부를 버리는 **다른 추정량**이라 파일을 나눈다.
    if all_std:
        paths_out.append((C.scoped_output("fairness_gaps_standardized.csv"), all_std))
    print()
    for p, frames in paths_out:
        pd.concat(frames, ignore_index=True).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[save] {p}")
    print("[해석] 증폭비 > 1 이면 모델이 데이터에 이미 있던 격차를 키운 것. "
          f"기준별(named_all / {' / '.join(f'named_n>={m}' for m in sorted(args.min_n))}) "
          "값이 크게 다르면 헤드라인 격차가 소수 그룹에서 나온 것이다. 모델 비교는 "
          "개별 CI 겹침이 아니라 대조표의 짝지은 차이로 읽을 것.")


if __name__ == "__main__":
    main()
