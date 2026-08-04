"""7지표 정의와 `--blind`의 무음 no-op 방어.

겨냥하는 사고:

1. **지표 정의가 스크립트마다 복사돼 있었다.** 손으로 동기화하다 어긋나면 모델 비교가
   조용히 무의미해진다. `clear.metrics.evaluate` 하나로 합친 뒤로는 정의가 갈릴 수
   없지만, **정의 자체가 바뀌는 것**은 여전히 가능하므로 골든 값으로 고정한다.
   특히 `specificity`는 음성 클래스(미해결)의 재현율이라 `sensitivity`와 대칭이다 --
   둘이 뒤바뀌면 모든 완화 서술의 부호가 뒤집힌다(벌점은 Sens↑/Spec↓이 특징이다).

2. **`--blind`가 무음 no-op였다.** 열이 안 지워졌는데 exit 0이라, 모든 blind 결과가
   무효인 채로 통과했다. 지금은 `_assert_blind`가 (a) 드롭 목록이 비었는지,
   (b) 민감 접두사로 시작하는 열이 남았는지를 검사한다. 이 방어가 살아 있는지 본다.
"""
import numpy as np
import pandas as pd
import pytest

import config as C
from clear import data as D
from clear.metrics import evaluate


def test_seven_metrics_present():
    y = np.array([0, 0, 1, 1, 1, 1])
    m = evaluate(y, np.array([0.1, 0.6, 0.4, 0.8, 0.9, 0.7]))
    assert set(m) == {"auc", "mcc", "f1", "sensitivity", "specificity",
                      "balanced_accuracy", "precision"}


def test_specificity_is_recall_on_the_negative_class():
    """미해결(0)을 얼마나 잡아내는가. sensitivity와 뒤바뀌면 완화 서술이 뒤집힌다."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])   # 음성 2/4, 양성 2/4
    m = evaluate(y, proba)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_perfect_and_inverted_predictions():
    y = np.array([0, 0, 1, 1])
    good = evaluate(y, np.array([0.01, 0.02, 0.98, 0.99]))
    assert good["auc"] == pytest.approx(1.0) and good["mcc"] == pytest.approx(1.0)

    bad = evaluate(y, np.array([0.99, 0.98, 0.02, 0.01]))
    assert bad["auc"] == pytest.approx(0.0) and bad["mcc"] == pytest.approx(-1.0)


def test_threshold_free_metrics_ignore_the_operating_point():
    """AUC는 임계값과 무관하고 F1은 아니다 — 이 프로젝트가 둘을 나눠 쓰는 근거다."""
    y = np.array([0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.45, 0.4, 0.8, 0.9, 0.7])
    a, b = evaluate(y, proba, threshold=0.5), evaluate(y, proba, threshold=0.44)
    assert a["auc"] == pytest.approx(b["auc"])
    assert a["f1"] != pytest.approx(b["f1"])


# --- --blind ---------------------------------------------------------------

def _X(cols):
    return pd.DataFrame({c: [0, 1] for c in cols})


def test_assert_blind_rejects_an_empty_drop_list():
    """드롭할 열을 하나도 못 찾았으면 blind가 아니다 -- 조용히 통과하면 안 된다."""
    with pytest.raises((AssertionError, SystemExit, ValueError)):
        D._assert_blind(_X(["Age Group=25-29", "Weapon=Firearm"]), [])


def test_assert_blind_rejects_a_surviving_sensitive_column():
    """접두사가 남아 있으면 실패해야 한다(구분자가 바뀌면 여기서 걸린다)."""
    prefix = C.SENSITIVE_FEATURE_COLS[0]
    X = _X([f"{prefix}=Black", "Weapon=Firearm"])
    with pytest.raises((AssertionError, SystemExit, ValueError)):
        D._assert_blind(X, ["dropped_something"])


def test_assert_blind_accepts_a_real_drop():
    prefix = C.SENSITIVE_FEATURE_COLS[0]
    D._assert_blind(_X(["Weapon=Firearm"]), [f"{prefix}=Black"])


def test_sensitive_columns_are_actually_in_the_feature_list():
    """민감속성은 기본적으로 **모델 입력에 있다** — sens__* 는 진단용 사본이다.

    이 사실이 흐려지면 '모델은 인종을 못 본다'는 잘못된 서술이 다시 들어온다.
    문서가 한 번 그렇게 틀렸고, 그래서 여기에 못 박는다.
    """
    assert set(C.SENSITIVE_FEATURE_COLS) <= set(C.CATEGORICAL_COLS), (
        "민감속성이 CATEGORICAL_COLS에서 빠졌다 — 그렇다면 sighted/blind 구분과 "
        "모든 blind 서술의 전제가 바뀐 것이다")
