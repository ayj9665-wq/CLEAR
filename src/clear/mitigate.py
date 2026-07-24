"""완화기법(후처리) — 그룹별 임계값으로 격차를 줄이고 그 대가를 잰다.

진단(07)이 "모델이 격차를 키운다"까지 보였다면, 처방은 "얼마나 줄일 수 있고
정확도를 얼마나 잃는가"에 답해야 한다. 이 모듈은 그중 **후처리** 계열을 맡는다 —
모델을 다시 학습하지 않고, 저장된 확률에 **그룹마다 다른 임계값**을 적용한다.

후처리를 먼저 두는 이유:
  - 재학습이 없어 05/06의 예측 덤프만으로 돌아간다(GNN 재학습 수 분 → 0초).
  - **세 모델에 똑같이 적용된다.** GNN 전용 완화(엣지 수준)만 하고 원본 XGBoost와
    대면시키면 조작이므로, 모든 모델에 적용 가능한 공통 축이 먼저 있어야 한다.
  - 임계값 세기를 연속적으로 바꾸면 **정확도-공정성 트레이드오프 곡선**이 그대로
    나온다(포스터의 두 번째 패널).

**세기 조절(lambda)**: 그룹 g의 목표 비율을
    target_g(L) = (1-L) * (그룹 g의 원래 비율) + L * (전체 비율)
로 두고, 그 비율을 달성하는 임계값을 분위수로 찾는다. L=0이면 원본 그대로,
L=1이면 모든 그룹이 같은 비율(격차 0)이다. 임계값을 직접 보간하지 않고 **비율**을
보간하는 이유: 임계값은 모델마다 스케일이 달라 보간의 의미가 불분명한 반면,
비율은 공정성 정의 그 자체라 L이 "격차를 몇 % 닫았는가"로 바로 읽힌다.

**기준(criterion)**:
  dp   그룹별 '검거 예측 비율'을 맞춘다(Demographic Parity). 배포 시 정답 라벨이
       필요 없다.
  tpr  그룹별 '실제 검거 사건을 맞히는 비율'을 맞춘다(Equalized Odds의 한 축).
       기회 균등에 더 가깝지만 임계값을 정하려면 라벨이 필요하다.

**임계값은 반드시 다른 데이터에서 정한다.** 평가할 바로 그 test 집합에서 임계값을
맞추고 같은 집합에서 성능을 재면 완화 효과가 낙관적으로 나온다. 그래서 test를
층화 분할해 한쪽(tune)에서 임계값을 정하고 다른 쪽(eval)에서만 평가한다. 05는
val 예측을 남기지 않으므로 이 방식이 재학습 없이 쓸 수 있는 가장 정직한 선택이다.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
from clear import fairness as F
from clear.metrics import evaluate_pred

CRITERIA = ("dp", "tpr")


def _rate(proba, y, criterion, threshold):
    """한 그룹에서 기준 비율(dp=선택률, tpr=양성 재현율)을 임계값 기준으로 계산."""
    pred = proba >= threshold
    if criterion == "dp":
        return float(pred.mean()) if len(pred) else np.nan
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else np.nan


def _threshold_for_rate(proba, y, criterion, rate):
    """목표 비율을 달성하는 임계값. 상위 rate 비율이 뽑히도록 분위수를 쓴다."""
    pool = proba if criterion == "dp" else proba[y == 1]
    if len(pool) == 0 or not np.isfinite(rate):
        return np.inf                      # 아무도 뽑지 않음
    rate = min(max(rate, 0.0), 1.0)
    if rate <= 0:
        return np.inf
    if rate >= 1:
        return -np.inf                     # 전원 선택
    return float(np.quantile(pool, 1.0 - rate))


def group_thresholds(proba, groups, y, lam, criterion="dp", base_threshold=0.5,
                     unknown=None, restrict=None):
    """그룹별 임계값 dict. lam=0이면 전부 base_threshold(원본 재현).

    restrict를 주면 그 그룹들만 조정하고 나머지는 base_threshold로 둔다 —
    격차를 재는 그룹집합(예: n>=5000)과 완화 대상 그룹집합을 일치시키기 위한 것.
    Unknown은 인구집단이 아니므로 기본적으로 조정하지 않는다.
    """
    unknown = C.FAIRNESS_UNKNOWN_LABEL if unknown is None else unknown
    proba, groups, y = np.asarray(proba), np.asarray(groups), np.asarray(y)
    labels = [g for g in pd.unique(groups) if g != unknown]
    if restrict is not None:
        labels = [g for g in labels if g in set(restrict)]

    masks = {g: (groups == g) for g in labels}
    rates = {g: _rate(proba[m], y[m], criterion, base_threshold) for g, m in masks.items()}
    # 전체 목표: 조정 대상 그룹을 합친 비율(= 완화가 수렴할 공통 값)
    pooled = np.concatenate([np.flatnonzero(m) for m in masks.values()]) if masks else np.array([], int)
    overall = _rate(proba[pooled], y[pooled], criterion, base_threshold)

    out = {}
    for g, m in masks.items():
        target = (1 - lam) * rates[g] + lam * overall
        out[g] = (base_threshold if lam == 0
                  else _threshold_for_rate(proba[m], y[m], criterion, target))
    return out


def apply_thresholds(proba, groups, thresholds, base_threshold=0.5):
    """그룹별 임계값 적용 → 예측 레이블. dict에 없는 그룹은 base_threshold."""
    proba, groups = np.asarray(proba), np.asarray(groups)
    t = np.full(len(proba), base_threshold, dtype=float)
    for g, thr in thresholds.items():
        t[groups == g] = thr
    return (proba >= t).astype(int)


def tune_eval_split(n, y, test_size=0.5, seed=None):
    """test 집합을 tune/eval로 층화 분할. 임계값은 tune에서, 성능은 eval에서."""
    seed = C.RANDOM_STATE if seed is None else seed
    tune, ev = train_test_split(np.arange(n), test_size=test_size,
                                stratify=y, random_state=seed)
    return tune, ev


def sweep(df, attr, *, lambdas, criterion="dp", min_n=None, n_boot=None,
          seed=None, base_threshold=0.5, tune_size=0.5):
    """한 (덤프 x 민감속성 x 기준)에 대해 lambda를 훑어 트레이드오프 점들을 낸다.

    각 lambda마다: tune 절반에서 그룹별 임계값을 정하고, eval 절반에 적용해
    (1) 정확도 지표와 (2) 공정성 격차·증폭비를 잰다. 둘 다 **eval 절반에서만**
    계산하므로 임계값 적합에 쓰인 데이터가 평가에 새지 않는다.

    격차 계산은 clear.fairness의 함수를 그대로 쓴다 — 완화 전후가 같은 정의로
    비교돼야 곡선이 의미를 갖는다.
    """
    min_n = C.FAIRNESS_MIN_GROUP_N if min_n is None else min_n
    proba = df["proba"].values
    y = df["y_true"].values
    groups = df[attr].values

    tune, ev = tune_eval_split(len(df), y, test_size=1 - tune_size, seed=seed)
    ev_df = df.iloc[ev].reset_index(drop=True)

    # 격차를 잴 그룹집합은 eval 절반의 그룹 크기로 정한다(평가 대상과 일치).
    ev_counts, _ = F.joint_counts({"_": ev_df}, attr)
    target_groups = F.select_groups(ev_counts, min_n)

    rows = []
    for lam in lambdas:
        thr = group_thresholds(proba[tune], groups[tune], y[tune], lam,
                               criterion=criterion, base_threshold=base_threshold,
                               restrict=target_groups)
        pred_ev = apply_thresholds(proba[ev], groups[ev], thr, base_threshold)

        acc = evaluate_pred(y[ev], pred_ev, proba[ev])
        scored = ev_df.assign(pred=pred_ev)
        counts, _ = F.joint_counts({"_": scored}, attr)
        boot = F.bootstrap_joint(counts, n_boot=n_boot, seed=seed)
        _, stats = F.gap_samples(counts, boot, target_groups)
        gap = F.gap_row(stats, {g: F._rates(counts[g]) for g in target_groups},
                        target_groups, f"named_n>={min_n}")

        rows.append({
            "criterion": criterion, "lambda": lam, "n_eval": len(ev),
            **{f"acc_{k}": v for k, v in acc.items()},
            "dp_gap": gap["selection_rate_gap"],
            "dp_amplification": gap["selection_rate_amplification"],
            "dp_amplification_lo": gap["selection_rate_amplification_lo"],
            "dp_amplification_hi": gap["selection_rate_amplification_hi"],
            "tpr_gap": gap["tpr_gap"],
            "tpr_amplification": gap["tpr_amplification"],
            "base_rate_gap": gap["base_rate_gap"],
            "groups": gap["groups"],
            "thresholds": "; ".join(f"{g}={t:.3f}" for g, t in sorted(thr.items())),
        })
    return pd.DataFrame(rows)
