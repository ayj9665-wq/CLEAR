"""전역 보정과 '빈 문자열이 NaN이 되는' 함정.

두 사고를 겨냥한다.

1. **보정 없이는 잔차가 전부 한쪽으로 쏠린다.** `clear.gnn`은 `pos_weight`로 학습해
   p̂이 순위는 옳지만 확률로는 치우쳐 있다(전국 기대 미해결이 관측보다 +48%). 그대로
   쓰면 거의 모든 블록이 "기대보다 적게 미해결"로 나와, 재는 것이 지역 이상이 아니라
   전역 miscalibration이 된다. 보정은 로짓 이동 하나이므로 **순위를 바꾸면 안 된다** --
   바꾸면 D3 우선순위가 조용히 달라진다.

2. **`flag`는 메모리에서 빈 문자열이지만 CSV에서 읽으면 `NaN`이다.** `flag == ""`가
   조용히 실패해 비유의 카운티 123개가 신호로 칠해졌다. 지도·검사기·보고서가 전부
   이 열을 읽으므로, 읽는 쪽이 NaN을 채우는지 본다.
"""
import numpy as np
import pandas as pd

from clear import predictions as P


def _skewed(n=2000, seed=0):
    """pos_weight 학습을 흉내낸다: 순위는 맞지만 확률이 위로 치우친 p̂."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.70, n)
    base = rng.beta(2, 2, n)
    proba = np.clip(base * 0.5 + y * 0.25 + 0.25, 1e-6, 1 - 1e-6)
    return proba, y


def test_calibration_matches_the_totals():
    """δ는 Σσ(logit p̂ + δ) = Σy 가 되도록 잡힌다. 그것이 간접 표준화의 정의다."""
    proba, y = _skewed()
    p_cal, cal = P.calibrate(proba, y, "shift")
    assert abs(p_cal.sum() - y.sum()) < 1e-3 * len(y), (
        f"보정 후 기대 검거 {p_cal.sum():.1f} != 실제 {y.sum()}")
    assert abs(cal["sum_p_after"] - cal["sum_y"]) < 1.0


def test_calibration_is_monotone_so_rankings_survive():
    """파라미터가 하나이고 단조라 **순위가 전혀 안 바뀐다.**"""
    proba, y = _skewed()
    p_cal, _ = P.calibrate(proba, y, "shift")
    before = np.argsort(np.argsort(proba))
    after = np.argsort(np.argsort(p_cal))
    assert np.array_equal(before, after), "보정이 순위를 바꿨다 — D3 목록이 달라진다"


def test_calibration_none_leaves_the_skew():
    """`--calibrate none`은 진단용이다. 치우침이 남아 있어야 그 경고가 참이 된다."""
    proba, y = _skewed()
    p_raw, _ = P.calibrate(proba, y, "none")
    assert np.array_equal(p_raw, proba)
    assert abs(p_raw.sum() - y.sum()) > 1.0


def test_empty_flag_reads_back_as_nan(tmp_path):
    """이 함정이 실재하는지부터 고정한다 — 안 그러면 아래 방어가 무의미해진다."""
    p = tmp_path / "cold.csv"
    pd.DataFrame({"City": ["A", "B"], "flag": ["cold", ""]}).to_csv(p, index=False)
    back = pd.read_csv(p)
    assert back["flag"].isna().iloc[1], "빈 문자열이 NaN으로 안 읽힌다면 이 테스트는 낡았다"
    assert not (back["flag"] == "").iloc[1], "flag == '' 는 CSV에서 통하지 않는다"


def test_significance_filter_must_fill_na_first():
    """유의/비유의를 가르는 올바른 방법. 채우지 않으면 비유의가 신호로 샌다."""
    df = pd.DataFrame({"City": list("ABCD"),
                       "flag": ["cold", None, "warm", None]})
    naive = df[df["flag"] != ""]                 # 옛 방식
    fixed = df[df["flag"].fillna("") != ""]      # 지금 방식

    assert len(naive) == 4, "옛 방식이 실패하지 않는다면 이 회귀 검사는 무의미하다"
    assert len(fixed) == 2
    assert set(fixed["City"]) == {"A", "C"}
