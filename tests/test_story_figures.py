"""소개 페이지(v1/ver2)의 본문 수치가 결과 CSV와 같은 것을 말하는지 본다.

## 왜 이 파일이 생겼는가

ver2를 만들면서 v1의 벌점 사다리가 **라벨과 다른 양을 그리고 있다**는 것이 드러났다.
라벨은 "미제를 해결됨으로 잘못 판정하는 비율의 인종 격차"(= 전국 일괄 FPR 증폭)인데,
네 칸 중 하나만 그 값이었다. 벌점 0은 선택률 증폭(1.805, FPR은 1.893)이었고, 25와
50은 **주 단위 표준화** 값(0.639/0.701, 일괄은 0.451/0.429)이었다. 셋이 서로 다른
축의 값이었는데도 사다리는 매끄럽게 내려가는 모양이었다.

이 부류는 conftest의 목록에 이미 있는 것과 같다 — "설정키 누락으로 그룹 병합:
0.2620이라는 어디에도 없는 숫자". 화면에는 그럴듯한 값이 있고, 아무 예외도 안 난다.
그래서 값을 CSV에서 읽도록 고친 뒤, **그 결선이 유지되는지**를 여기서 잠근다.

데이터(`data/processed/`)는 필요 없다. 읽는 것이 커밋된 결과 CSV뿐이라 클론에서
그대로 돈다 — 대시보드의 clone-and-run 계약과 같은 조건이다.
"""
import re

import pytest

import config as C
from experiments import build_story_page as V1
from experiments import build_story_page_v2 as V2

# 소개 페이지는 전국 산출물이다(FAIR_MODEL이 전국 미니배치 모델 이름이다). 그런데
# config.SCOPE는 import 시점의 환경변수로 정해지고, 같은 세션의 다른 테스트가 config를
# 먼저 import했으면 이미 기본 스코프로 굳어 있다. **환경변수로 맞추려 하면 테스트를
# 혼자 돌릴 때만 통과한다** -- 실제로 그렇게 한 번 틀렸다. 그러니 프로세스 전역에
# 기대지 말고 테스트마다 명시적으로 바꾸고 되돌린다.
NAT = C.OUTPUT_DIR / "national"

pytestmark = pytest.mark.skipif(
    not (NAT / "fairness_gaps.csv").exists(),
    reason="전국 fairness_gaps.csv 없음")


@pytest.fixture(autouse=True)
def _national_scope(monkeypatch):
    monkeypatch.setattr(C, "SCOPE", "national")
    V1._FAIR_CACHE.clear()          # 캐시가 다른 스코프의 표를 물고 있으면 안 된다
    yield
    V1._FAIR_CACHE.clear()

ALPHAS = [(0, "graphsage_fairloss_a0_mb"), (25, "graphsage_fairloss_a25_mb"),
          (50, "graphsage_fairloss_a50_mb"), (100, "graphsage_fairloss_a100_mb")]


def _values(html, cls):
    return [float(x) for x in re.findall(rf'<span class="{cls}">([\d.]+)</span>', html)]


def test_alpha_ladder_is_one_quantity():
    """벌점 사다리 네 칸이 **전부** 전국 일괄 FPR 증폭이어야 한다.

    한 칸이라도 다른 파일/다른 지표에서 오면 사다리는 두 축을 오가게 되고, 그건
    화면으로 구분되지 않는다.
    """
    want = [V1.fair("fairness_gaps.csv", "Victim Race", metric="fpr", model=m)[0]
            for _, m in ALPHAS]
    got = _values(V1.prescribe_parts()["core"], "aval")
    assert got == [round(v, 3) for v in want], (
        f"사다리 {got} 가 일괄 FPR 증폭 {[round(v, 3) for v in want]} 와 다르다")


def test_diagnose_ladder_matches_csv():
    """진단 사다리 다섯 칸이 각각 제 파일에서 온 값이어야 한다."""
    want = [
        V1.fair("fairness_gaps.csv", "Victim Race")[0],
        V1.fair("fairness_gaps_standardized.csv", "Victim Race")[0],
        V1.fair("fairness_gaps_standardized_county.csv", "Victim Race",
                min_stratum_n=V1.COUNTY_FLOOR)[0],
        V1.fair("fairness_gaps.csv", "Victim Sex")[0],
        V1.fair("fairness_gaps_standardized.csv", "Victim Sex")[0],
    ]
    got = _values(V1.diagnose_parts()["core"], "bval")
    assert got == [round(v, 3) for v in want]


def test_fair_refuses_an_ambiguous_row():
    """행이 하나로 안 좁혀지면 **멈춰야** 한다.

    카운티 표준화 표는 층 하한 20/30/50이 각각 한 행씩이다. 하한을 안 주면 세 행이
    잡히는데, 조용히 첫 행을 쓰면 페이지가 어느 하한의 값인지 말할 수 없는 숫자를
    싣게 된다.
    """
    with pytest.raises(SystemExit):
        V1.fair("fairness_gaps_standardized_county.csv", "Victim Race")


def test_findings_agree_with_the_metric_table():
    """핵심 결과의 마진이 같은 페이지의 표에서 나온 차이와 같아야 한다.

    ver2는 요약과 근거를 한 장에 담으므로, 둘이 다르면 그 자리에서 자기모순이다.
    """
    g, x, _ = V1.PREDICT_ROWS
    nums = dict(V2.findings()[0]["nums"])
    for label, k in (("MCC", 0), ("AUC", 1), ("Precision", 3)):
        key = next(s for s in nums if s.startswith(label))
        assert key.endswith(f"{g[1][k] - x[1][k]:.4f}"), f"{label}: {key}"


def test_findings_report_the_national_flat_model():
    """전국에서는 blind 정형 모델도 증폭한다는 사실이 요약에 남아 있어야 한다.

    v1 진단 절은 3개 주 값(정형 0.67)을 라벨 없이 싣는다. 요약 카드로 압축하면
    '변수를 지우면 정형 모델은 증폭을 멈춘다'로 읽히는데, 전국에서는 거짓이다.
    """
    xb = V1.fair("fairness_gaps.csv", "Victim Race", model="xgboost_blind")[0]
    assert xb > 1.0, "전제가 바뀌었다 -- 이 테스트의 의미를 다시 볼 것"
    card = V2.findings()[1]
    assert f"{xb:.3f}" in " ".join(a + b for a, b in card["nums"]) + card["limit"]


def test_every_caveat_is_placed_exactly_once():
    """ver2는 v1의 경고문 12개를 재배치한다. 재배치하다 하나를 떨어뜨려도 화면에는
    아무 일도 안 일어난다 -- 그 문장이 빠지면 지도가 인과로 읽힐 뿐이다."""
    used = [i for _, idx in V2.LIMIT_GROUPS for i in idx] + V2.MAP_NOTE_IDX
    assert sorted(used) == list(range(len(V1.NOTES)))
