"""그룹별 공정성 지표·격차·부트스트랩 신뢰구간 — 07(진단)과 08(처방)의 공통 출처.

experiments/diagnose_fairness.py에 있던 계산을 여기로 옮겼다. 완화기법(08)은 완화 전후의 격차를
**같은 정의로** 재계산해야 트레이드오프 곡선이 성립하므로, 이 로직이 07 안에
남아 있으면 mitigate_threshold이 복붙할 수밖에 없다 — clear/ 패키지를 만든 이유 그대로다.

측정하는 것:
  base_rate      실제 검거율            — 모델 이전, 데이터 자체의 격차
  selection_rate 검거로 예측한 비율      — Demographic Parity 대상
  tpr / fpr      검거·미해결 각각의 오류 — Equalized Odds 대상
  격차(gap)      명명된 그룹 간 max-min
  증폭비          모델 격차 ÷ base_rate 격차 — 1보다 크면 모델이 데이터에
                 이미 있던 불균형을 **키운** 것. 이 프로젝트의 핵심 주장.

**부트스트랩 설계 (1) 그룹 내 복원추출.** 그룹 크기(n)는 표본이 아니라 데이터가
정한 값이므로 고정하고, 각 그룹 '안에서' 복원추출한다(층화 부트스트랩).

**부트스트랩 설계 (2) 재추출 대신 정확한 등가물.** 한 그룹의 각 행은 (y, pred)
조합에 따라 TN/FP/FN/TP 네 칸 중 하나에 속하고, 그 그룹에서 n개를 복원추출하는
것은 네 칸 비율에 대한 Multinomial(n, [tn,fp,fn,tp]/n) 추출과 정확히 같다.
n x B개의 인덱스를 뽑을 필요 없이 그룹당 (B,4) 행렬 하나면 된다.

**부트스트랩 설계 (3) 모델 간 비교는 짝지어야 한다.** 여러 모델은 *같은 test 행*을
쓰므로, 모델마다 독립으로 CI를 내고 "구간이 겹치니 차이 없다"고 읽으면 틀린다
(겹치는 독립 CI는 유의한 차이와 얼마든지 공존한다). 그래서 두 모델을 대조할 때는
칸을 **결합 칸(joint cell)**으로 올린다 — 한 행은 (모델A 칸, 모델B 칸) 조합 중
하나에 속하므로 16칸이고, 그룹 내 복원추출은 그 16칸에 대한 multinomial과 같다.
한 번 뽑은 결합 표본에서 각 모델의 4칸을 주변화(marginalize)해 쓰므로, 두 모델의
격차 차이를 복제본마다 **짝지어** 계산할 수 있다.

**결합은 쌍 단위로만 한다(중요).** 처음엔 M개 모델 전체를 4^M칸으로 한 번에 뽑았다.
6개까지는 됐지만(4096칸) 덤프가 늘면 파탄난다 — 10개면 100만 칸, 복제본 1000개에
8GB다. 완화기법 산출물까지 쌓이면 금방 그 지경이 된다. 쌍별 결합(16칸)으로 낮춰도
**통계적으로 동등**하다: 다항분포를 칸끼리 묶어 주변화하면 다시 다항분포이므로,
쌍 대조에 필요한 결합분포는 전체 결합에서 그 쌍만 주변화한 것과 같은 분포다.
개별 모델 CI도 같은 이유로 단일 모델(4칸) 추출로 낸다. 다만 난수 추출이 서로 달라,
대조표가 함축하는 개별 값과 개별 모델 CI가 소수 셋째 자리에서 미세하게 다를 수 있다
(같은 분포의 다른 표본이라 해석에는 영향 없다).

**한계(반드시 함께 보고할 것)**: max-min 격차는 잡음 섞인 추정치들의 최댓값이라
**위쪽으로 편향**된다. 그룹이 작을수록 심하고, 부트스트랩 백분위 CI는 이 편향을
없애주지 않는다(구간 폭만 알려준다). 그래서 격차는 항상 명명된 전체 그룹과
n>=FAIRNESS_MIN_GROUP_N 두 벌로 낸다 — 후자가 편향이 덜한 쪽이다.
"""
import itertools

import numpy as np
import pandas as pd

import config as C

# 혼동행렬 4칸 순서. 행의 칸 번호는 2*y + pred 로 매긴다(= 이 순서).
CELLS = ["tn", "fp", "fn", "tp"]
RATE_COLS = ["base_rate", "selection_rate", "tpr", "fpr", "tnr"]
# 격차를 재는 지표. base_rate는 '모델 이전' 기준선이라 증폭비의 분모로만 쓴다.
GAP_METRICS = ["selection_rate", "tpr", "fpr"]


def _rates(c):
    """(...,4) 카운트 → 비율 dict. 분모 0이면 nan(그룹에 양성/음성이 아예 없는 경우)."""
    tn, fp, fn, tp = (c[..., i] for i in range(4))
    n = tn + fp + fn + tp
    div = lambda a, b: np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float),
                                 where=(b > 0))
    return {
        "base_rate": div(fn + tp, n),        # 실제 검거율
        "selection_rate": div(fp + tp, n),   # 예측 검거율(DP)
        "tpr": div(tp, tp + fn),             # sensitivity
        "fpr": div(fp, fp + tn),             # 1 - specificity
        "tnr": div(tn, tn + fp),             # specificity
        "accuracy": div(tn + tp, n),
    }


def _ci(arr, pct=None):
    """부트스트랩 표본 → (lo, hi) 백분위 신뢰구간. nan은 무시."""
    pct = C.FAIRNESS_CI_PCT if pct is None else pct
    tail = (100 - pct) / 2
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan, np.nan
    return tuple(np.nanpercentile(arr, [tail, 100 - tail]))


def _span(values):
    """max-min. 그룹이 2개 미만이거나 전부 nan이면 nan."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] < 2 or np.all(np.isnan(v)):
        return np.nan if v.ndim == 1 else np.full(v.shape[1:], np.nan)
    return np.nanmax(v, axis=0) - np.nanmin(v, axis=0)


# ---- 결합 칸(joint cell) 카운트 ----

def joint_counts(dumps, attr, y_col="y_true", pred_col="pred"):
    """그룹별 결합 칸 카운트 {group: (4**M,)} 와 모델 라벨 순서.

    행 i의 결합 칸 = sum_m cell_m(i) * 4**m, cell_m = 2*y + pred_m. 모든 덤프가
    같은 행 순서라는 전제는 clear.predictions.assert_same_test_set이 검증한다.
    """
    labels = list(dumps)
    ref = dumps[labels[0]]
    y = ref[y_col].values.astype(np.int64)
    joint = np.zeros(len(ref), dtype=np.int64)
    for m, lab in enumerate(labels):
        joint += (2 * y + dumps[lab][pred_col].values.astype(np.int64)) * (4 ** m)
    size = 4 ** len(labels)
    g = ref[attr].values
    return {grp: np.bincount(joint[g == grp], minlength=size) for grp in pd.unique(g)}, labels


def marginal(joint_arr, m, n_models):
    """(...,4**M) 결합 카운트 → (...,4) 모델 m의 주변 카운트.

    결합 인덱스에서 모델 m의 계수가 4**m이므로, (4,)*M 로 reshape 했을 때(C order,
    마지막 축이 가장 빠르게 변함) 모델 m은 뒤에서 (m+1)번째 축이다.
    """
    a = np.asarray(joint_arr)
    prefix = a.shape[:-1]
    a = a.reshape(prefix + (4,) * n_models)
    axis_of = lambda i: len(prefix) + (n_models - 1 - i)
    other = tuple(axis_of(i) for i in range(n_models) if i != m)
    return a.sum(axis=other) if other else a


def bootstrap_joint(jcounts, n_boot=None, seed=None):
    """그룹별 (B, 4**M) 부트스트랩 결합 카운트. 그룹 내 복원추출 = 결합 칸 multinomial."""
    n_boot = C.FAIRNESS_BOOTSTRAP_N if n_boot is None else n_boot
    seed = C.RANDOM_STATE if seed is None else seed
    rng = np.random.default_rng(seed)
    out = {}
    for g, c in jcounts.items():
        n = int(c.sum())
        out[g] = (rng.multinomial(n, c / n, size=n_boot) if n > 0
                  else np.zeros((n_boot, len(c)), dtype=np.int64))
    return out


# ---- 그룹 선택 · 표 ----

def select_groups(counts, min_n=0, unknown=None):
    """격차 계산에 쓸 그룹. Unknown 제외 + n 하한. n은 모델과 무관하므로
    모든 모델이 같은 그룹집합을 쓴다(비교 가능성)."""
    unknown = C.FAIRNESS_UNKNOWN_LABEL if unknown is None else unknown
    return [g for g, c in counts.items() if g != unknown and c.sum() >= min_n]


def group_table(counts, boot):
    """그룹별 지표 표(점추정 + selection_rate/tpr/fpr의 CI). 그림의 오차막대 재료."""
    rows = []
    for g, c in counts.items():
        pt, bt = _rates(c), _rates(boot[g])
        row = {"group": g, "n": int(c.sum())}
        row.update({m: pt[m] for m in [*RATE_COLS, "accuracy"]})
        for m in GAP_METRICS:
            row[f"{m}_lo"], row[f"{m}_hi"] = _ci(bt[m])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ---- 격차 · 증폭비 ----

def gap_samples(counts, boot, groups):
    """지표별 (점추정 gap, 부트스트랩 gap 배열) + 증폭비 표본.

    증폭비는 **같은 복제본 안에서** 분자(모델 격차)와 분모(base_rate 격차)를 함께
    계산한다(paired) — 둘이 같은 test 행에서 나오므로 독립으로 두면 구간이 과대해진다.
    분모가 0에 가까운 복제본은 nan으로 떨어뜨린다.
    """
    pt = {g: _rates(counts[g]) for g in groups}
    bt = {g: _rates(boot[g]) for g in groups}
    span_pt = lambda m: _span([pt[g][m] for g in groups])
    span_bt = lambda m: _span(np.stack([bt[g][m] for g in groups]))

    base_pt, base_bt = span_pt("base_rate"), span_bt("base_rate")
    out = {"base_rate": (base_pt, base_bt)}
    for m in GAP_METRICS:
        g_pt, g_bt = span_pt(m), span_bt(m)
        amp_pt = (g_pt / base_pt) if base_pt > 0 else np.nan
        amp_bt = np.divide(g_bt, base_bt, out=np.full_like(g_bt, np.nan),
                           where=(base_bt > 1e-9))
        out[m] = (g_pt, g_bt)
        out[f"{m}_amplification"] = (amp_pt, amp_bt)
    return pt, out


def gap_row(stats, pt, groups, label):
    """gap_samples 결과 → 격차 한 행(지표별 gap·CI, 최대/최소 그룹, 증폭비·CI)."""
    row = {"group_set": label, "n_groups": len(groups),
           "groups": " | ".join(map(str, groups))}
    row["base_rate_gap"] = stats["base_rate"][0]
    row["base_rate_gap_lo"], row["base_rate_gap_hi"] = _ci(stats["base_rate"][1])
    for m in GAP_METRICS:
        row[f"{m}_gap"] = stats[m][0]
        row[f"{m}_gap_lo"], row[f"{m}_gap_hi"] = _ci(stats[m][1])
        vals = [pt[g][m] for g in groups]
        if not np.all(np.isnan(vals)):
            row[f"{m}_max_group"] = groups[int(np.nanargmax(vals))]
            row[f"{m}_min_group"] = groups[int(np.nanargmin(vals))]
        amp_pt, amp_bt = stats[f"{m}_amplification"]
        row[f"{m}_amplification"] = amp_pt
        row[f"{m}_amplification_lo"], row[f"{m}_amplification_hi"] = _ci(amp_bt)
    return row


# ---- 진단 진입점 ----

def diagnose(dumps, attr, *, min_n=None, n_boot=None, seed=None):
    """여러 예측 덤프를 한 민감속성에 대해 동시 진단.

    dumps: {라벨: DataFrame} (같은 test 행 순서). 반환 (그룹표, 격차표, 모델대조표).
    격차는 명명된 전체 그룹 + min_n(정수 또는 정수 리스트)마다 한 벌씩 낸다.
    표본 하한을 올릴수록 소수그룹 잡음에 덜 휘둘리므로, 여러 기준을 **한 파일에
    나란히** 남겨야 헤드라인 수치가 어느 기준에서 나왔는지 독자가 판단할 수 있다
    (기준을 하나만 내면 사후에 유리한 기준을 고른 것처럼 보인다).

    모델대조표는 모델 쌍의 격차·증폭비 **차이**를 짝지은 복제본에서 계산한 것이다.
    "그래프가 flat 모델보다 격차를 더 키우는가"는 이 표로만 답할 수 있다 —
    개별 모델 CI의 겹침 여부로 읽으면 안 된다.
    """
    if min_n is None:
        min_n = [C.FAIRNESS_MIN_GROUP_N]
    elif isinstance(min_n, (int, np.integer)):
        min_n = [int(min_n)]
    labels = list(dumps)

    def _one(subset):
        """모델 부분집합의 (칸 카운트, 부트스트랩) — subset 크기만큼만 결합한다."""
        jc, labs = joint_counts({l: dumps[l] for l in subset}, attr)
        return jc, bootstrap_joint(jc, n_boot=n_boot, seed=seed), labs

    # 그룹 집합은 그룹 n으로만 정해지므로(모델 무관) 아무 모델 하나로 계산한다.
    jc0, jb0, _ = _one([labels[0]])
    group_sets = [("named_all", select_groups(jc0))]
    group_sets += [(f"named_n>={m}", select_groups(jc0, m)) for m in sorted(min_n)]

    # 개별 모델: 4칸 추출
    tables, gaps = [], []
    for lab in labels:
        jc, jb, _ = _one([lab]) if lab != labels[0] else (jc0, jb0, None)
        gt = group_table(jc, jb)
        gt.insert(0, "model", lab)
        tables.append(gt)
        for set_label, groups in group_sets:
            pt, stats = gap_samples(jc, jb, groups)
            gaps.append({"model": lab, **gap_row(stats, pt, groups, set_label)})

    # 모델 쌍: 16칸 결합 추출 → 복제본마다 짝지어 차이를 낸다
    contrasts = []
    for a, b in itertools.combinations(labels, 2):
        jc, jb, labs = _one([a, b])
        side = {}
        for i, lab in enumerate(labs):
            c_i = {g: marginal(c, i, 2) for g, c in jc.items()}
            b_i = {g: marginal(v, i, 2) for g, v in jb.items()}
            side[lab] = (c_i, b_i)
        for set_label, groups in group_sets:
            sa = gap_samples(*side[a], groups)[1]
            sb = gap_samples(*side[b], groups)[1]
            contrasts += _contrast_rows(sa, sb, a, b, set_label)

    return (pd.concat(tables, ignore_index=True), pd.DataFrame(gaps),
            pd.DataFrame(contrasts))


def _contrast_rows(sa, sb, a, b, set_label):
    """한 모델 쌍·한 그룹집합의 대조 행들.

    diff > 0 이고 CI가 0을 안 걸치면 model_a가 model_b보다 격차를 더 키운다는 뜻.
    """
    rows = []
    for m in GAP_METRICS:
        for kind, key in [("gap", m), ("amplification", f"{m}_amplification")]:
            d_pt = sa[key][0] - sb[key][0]
            d_bt = np.asarray(sa[key][1]) - np.asarray(sb[key][1])
            lo, hi = _ci(d_bt)
            rows.append({
                "group_set": set_label, "model_a": a, "model_b": b,
                "metric": m, "quantity": kind,
                "diff": d_pt, "diff_lo": lo, "diff_hi": hi,
                # CI가 0을 안 걸치면 방향성 있는 차이로 읽는다.
                "significant": bool(np.isfinite(lo) and np.isfinite(hi)
                                    and (lo > 0 or hi < 0)),
            })
    return rows
