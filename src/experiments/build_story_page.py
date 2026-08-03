"""
experiments/build_story_page.py -- CLEAR 프로젝트 소개 페이지(자체 완결 HTML 한 장)

reports/포트폴리오웹페이지_계획서_CLEAR.md 의 구현. 프로젝트의 문제-방법-결과를
스크롤로 서술하고, 마지막에 카운티 잔차 지도가 주인공으로 등장한다.

## build_web_map.py를 고치지 않는다

지도 데이터를 만드는 세 함수(load_blocks / build_paths / color_tables)를 **import 해서
쓴다.** 셋 다 이미 모듈 레벨이라 리팩터링이 필요 없고, 그 덕에 이미 배포된
map_national.html / map_national_test.html / map_ca_tx_mi.html 은 회귀 위험이 0이다.
HTML 템플릿만 새로 쓴다 -- 저쪽은 밝은 배경 전용이고 이쪽은 어둡다. 템플릿을
파라미터화하면 조건 분기가 늘어 두 산출물이 같이 깨지므로, **이 구조에서는 템플릿
중복이 결합보다 싸다.** 값을 만드는 쪽(구간 경계·색 인덱스·판정 상태)은 한 곳에
있으므로 범례와 그림이 어긋나는 사고는 여전히 구조적으로 막혀 있다.

## 붉은색은 어휘가 아니라 단어 하나다

페이지 전체에서 빨강이 나오는 곳은 둘뿐이다: 지도의 cold 카운티, 그리고 신뢰구간이
1.0을 넘어선 격차 막대. 링크·버튼·호버에는 쓰지 않는다. 발산 팔레트의 반대편(warm)은
clear.counties.gray_arm()으로 채도를 빼고 **밝기까지 압축**한다 -- 밝기를 그대로 두면
밝은 표면에서 무채색이 유채색보다 대비가 세서 규칙이 뒤집힌다(그 함수의 docstring 참고).

## 경고문을 그대로 가져오지 않고 쉬운 말로 다시 쓴다

build_web_map.CAVEATS는 분석을 읽는 사람을 독자로 가정한다(SMR, BH FDR, 간접 표준화,
충돌자). 이 페이지의 독자는 그렇지 않으므로 NOTES/HOWTO에 **같은 내용을 다시 썼다.**
쉽게 쓰는 것과 약하게 쓰는 것은 다르므로 내용은 하나도 빼지 않았고 숫자도 전부 남겼다
-- 이 문장들이 빠지면 지도는 "빨간 카운티 = 경찰이 일을 안 한다"로 읽힌다.

## 폰트는 굵기별로 깎아서 싣는다

_fontpack 참고. "어느 텍스트가 어느 굵기인가"를 빌드가 알아야 하므로, 화면에 나가는
문자열은 전부 T(text, weight)를 통과한다. JS가 조립하는 텍스트(표·툴팁)는 LABELS에
모아 두고 거기서 한 번에 등록한다 -- JS 안에 한글 리터럴을 흩뿌리면 등록에서 빠진
글자가 두부(□)로 렌더된다.

출력: outputs[/{scope}]/web/clear_story.html (외부 요청 0)
"""
import argparse
import json
import re

import numpy as np

import config as C
from clear import counties as CT
from experiments import _fontpack as FP
from experiments.build_web_map import (
    VIEW_W, build_paths, color_tables, load_blocks,
)

SIMPLIFY_KM = 1.5
SIZE_BUDGET_KB = 850

# 다크 팔레트. 지도 카드만 밝게 반전한다(계획서 §5-3) -- 발산 팔레트는 밝은 표면
# 기준으로 명도가 검증돼 있어서, 검정 위에 그대로 얹으면 가장 밝은 단계(약한 신호)가
# 가장 강해 보인다. 의미가 뒤집힌다.
BG, BG2 = "#08080a", "#101013"
INK, INK2, MUTED, HAIR = "#f2f2f0", "#a8a8a4", "#6a6a67", "#232327"
ACCENT = CT.RED_ANCHOR

# ---------------------------------------------------------------------------
# 글리프 예산: 화면에 나가는 모든 문자열이 여기를 통과한다
# ---------------------------------------------------------------------------

class Glyphs:
    def __init__(self):
        self.w = {w: set() for w in FP.WEIGHTS}

    def add(self, text, weight=400):
        text = FP.normalize(str(text))
        self.w[weight].update(text)
        return text


G = Glyphs()


def T(text, weight=400):
    """등록하고 그대로 돌려준다. 본문은 400, 강조/표머리 700, 타이틀 800."""
    return G.add(text, weight)


# JS가 조립하는 텍스트. 한 곳에 모아 두어야 등록에서 빠지지 않는다.
LABELS = {
    "layer_res": "기대와의 차이",
    "layer_pri": "재수사 후보 비율",
    "layer_raw": "실제 검거율",
    "minsample": "최소 사건 수",
    "cases": "건 이상",
    "table": "표로 보기",
    "more": "기대보다 <b>많이</b> 잔존",
    "less": "기대보다 <b>적게</b> 잔존",
    "nonsig": "판정했으나 우연 수준",
    "low": "낮음",
    "high": "높음",
    "nodata": "사건 수 부족으로 판정 제외",
    "th": ["주", "카운티", "사건", "미해결", "기대", "배수", "확신도",
           "헛짚을 확률", "흑인비중"],
    "tip_cases": "사건",
    "tip_unsolved": "미해결",
    "tip_expected": "기대",
    "tip_black": "흑인 피해자 비중",
    # 레이어를 바꿀 때 뜨는 설명. 세 레이어가 **서로 다른 것을 센다**는 걸 안 쓰면
    # "기대보다 40% 더 미해결인데 왜 다시 볼 사건은 0.5%뿐이냐"에서 반드시 걸린다.
    "exp_res": "기대 미제 건수와 실제 건수를 견준 값이다. <b>전체 사건 대비가 "
               "아니다.</b> 배수 1.40은 기대보다 40% 더 남았다는 뜻이며, 그 초과분을 "
               "전체 사건으로 나누면 훨씬 작아진다. 뉴올리언스는 배수 1.40, 초과 "
               "1,113건으로 전체 7,847건의 14.2%에 해당한다.",
    "exp_pri": "미제 사건 중 '전국에서 가장 해결 가능성이 높게 예측된 상위 10%'에 "
               "해당하는 비율이다. 모델은 카운티를 모르므로 이 값은 해당 지역의 "
               "<b>사건 구성</b>을 따라간다. 대도시는 사건 자체가 어려워 낮게 "
               "나오고(뉴올리언스는 미제의 1.0%), 소규모 지역은 높게 나온다"
               "(로더데일은 82.4%). 첫 번째 레이어와 방향은 같으나(+0.67) "
               "<b>같은 것을 측정하지 않는다.</b>",
    "exp_raw": "기댓값과 무관하게 실제로 해결된 비율이다. 사건 구성이 수월한 지역은 "
               "그 자체로 높게 나오므로, 이 값만으로 수사의 성패를 판단할 수 없다. "
               "이 프로젝트가 기준으로 삼는 것은 첫 번째 레이어다.",
    "v_none": "우연으로 설명되는 수준",
    "v_more": "기대보다 %s% 더 미제로 잔존",
    "v_less": "기대보다 %s% 적게 잔존",
}


def register_labels():
    for k, v in LABELS.items():
        vals = v if isinstance(v, list) else [v]
        for s in vals:
            # 표 머리와 강조는 700으로도 렌더된다
            G.add(s, 400)
            G.add(s, 700)


# ---------------------------------------------------------------------------
# 본문
# ---------------------------------------------------------------------------

def section(num, label, title, lead, body_html=""):
    return (
        f'<section class="sec"><div class="inner">'
        f'<div class="shead"><span class="snum">{T(num, 800)}</span>'
        f'<span class="slab">{T(label, 800)}</span></div>'
        f'<h2>{T(title, 800)}</h2>'
        f'<p class="lead">{T(lead)}</p>{body_html}</div></section>')


def bar(label, value, vmax, red=None, ci=None):
    """증폭비 막대. **빨강은 신뢰구간이 1.0을 넘어설 때만** 칠한다.

    1.0은 "데이터에 이미 있던 격차를 모델이 더 벌리지 않았다"는 선이다. 값에
    상관없이 전부 빨강으로 칠하면 완화가 성공한 막대까지 경보색이 되어, 이 페이지의
    "빨강은 격차 경보 한 가지 뜻"이라는 규칙이 무너진다.

    점추정만 보고 칠하면 성별 주 표준화 1.005 [0.925, 1.090]이 빨강이 된다 --
    구간이 1을 덮으므로 "격차를 벌렸다"고 말할 수 없는 값인데도 경보로 읽힌다.
    이 저장소가 마진을 시드 노이즈로 나눠 판단하는 것과 같은 규율을 색에 적용한다.
    """
    if red is None:
        red = (ci[0] > 1.0) if ci else (value > 1.0)
    w = max(1.0, min(100.0, value / vmax * 100.0))
    ci_html = ""
    if ci:
        lo = max(0.0, min(100.0, ci[0] / vmax * 100.0))
        hi = max(0.0, min(100.0, ci[1] / vmax * 100.0))
        ci_html = (f'<i class="ci" style="left:{lo:.1f}%;width:{max(hi-lo,0.6):.1f}%">'
                   f'</i>')
    cls = "bar red" if red else "bar"
    return (f'<div class="brow"><span class="blab">{T(label)}</span>'
            f'<span class="btrack"><span class="{cls}" style="width:{w:.1f}%">'
            f'</span>{ci_html}</span>'
            f'<span class="bval">{T(f"{value:.3f}", 700)}</span></div>')


def metric_table(rows):
    head = "".join(f"<th>{T(h, 700)}</th>" for h in
                   ["모델", "MCC", "AUC", "Balanced Acc", "Precision"])
    body = ""
    for name, vals, best in rows:
        cells = "".join(
            f'<td class="{"win" if best else ""}">{T(f"{v:.4f}", 700 if best else 400)}</td>'
            for v in vals)
        body += f'<tr><td class="mname">{T(name, 700 if best else 400)}</td>{cells}</tr>'
    return f'<table class="mt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def build_body(stats):
    """페이지 본문(지도 섹션 제외)."""
    n_rows, n_counties, n_assessed, n_cold, n_warm, corr = stats
    h = []

    # --- S1 문제 ---------------------------------------------------------
    counters = [
        (f"{n_rows:,}", "기록된 살인사건", "1980-2014"),
        ("29.8%", "끝내 미제로 남음", "전국 평균"),
        (f"{n_counties:,}", "카운티", "50개 주 + DC"),
        (f"{n_cold}", "설명되지 않는 집중", "우연으로 보기 어려움"),
    ]
    cnt = '<div class="counters">' + "".join(
        f'<div class="cnt"><b data-count="{v}">{T(v, 800)}</b>'
        f'<span>{T(lab, 700)}</span><i>{T(sub)}</i></div>'
        for v, lab, sub in counters) + "</div>"
    h.append(section(
        "01", "THE PROBLEM",
        "검거율 70.2%, 그리고 남는 29.8%",
        "미국에서는 매년 수만 건의 살인사건이 발생하지만, 그중 상당수는 끝내 범인을 "
        "특정하지 못한 채 미제로 남는다. Murder Accountability Project가 집계한 "
        "1980~2014년 데이터 638,454건 기준 검거율은 70.2%에 그치며, 나머지 29.8%는 "
        "단 한 번도 해결되지 못했다. 이 숫자 뒤에는 단순한 수사 역량의 문제만이 아니라 "
        "'어떤 사건이, 어떤 피해자가 더 쉽게 잊히는가'라는 구조적 질문이 놓여 있다.",
        cnt
        + f'<p class="lead2">{T("격차는 모델을 돌리기 전, 원자료 수준에서 이미 나타난다. "
                                "피해자가 백인일 때 검거율은 <b>74.1%</b>인 반면 흑인일 "
                                "때는 <b>66.3%</b>로 7.8%p의 차이가 있고, 여성 "
                                "피해자(76.9%)와 남성 피해자(68.3%) 사이에도 8.6%p의 "
                                "차이가 확인된다. 이는 통계적 잡음이 아니라 실증적으로 "
                                "드러나는 구조적 격차이며, 이 프로젝트가 예측에서 멈추지 "
                                "않는 이유이기도 하다.", 700)}</p>'))

    # --- S2 접근 ---------------------------------------------------------
    steps = [
        ("PREDICT", "예측", "사건을 독립된 표본으로 다루는 대신, 같은 지역에서 발생한 "
                            "사건들을 엣지로 이어 그래프로 재구성하고 검거 여부를 "
                            "예측한다."),
        ("DIAGNOSE", "진단", "그 예측이 피해자의 인종·성별에 따라 체계적으로 "
                             "달라지는지를 공정성 지표로 측정한다. 두 속성은 서로 다른 "
                             "답을 낸다."),
        ("PRESCRIBE", "처방", "손실 함수에 집단 간 격차를 벌점으로 부과해 예측을 "
                              "교정하고, 그 대가로 지불하는 정확도를 정량화한다."),
        ("APPLY", "적용", "완화된 모델을 기준선으로 삼아, 사건 구성으로 설명되지 않는 "
                          "미제가 집중된 지역을 식별한다."),
    ]
    st = '<div class="steps">' + "".join(
        f'<div class="step"><span class="sn">{T(f"0{i+1}", 800)}</span>'
        f'<span class="se">{T(en, 700)}</span><b>{T(ko, 700)}</b>'
        f'<p>{T(d)}</p></div>'
        for i, (en, ko, d) in enumerate(steps)) + "</div>"
    h.append(section(
        "02", "THE APPROACH",
        "예측 · 진단 · 처방 · 적용의 네 단계",
        "각 단계는 앞 단계가 답하지 못한 것 때문에 존재한다. 특히 처방 단계는 두 번의 "
        "실패를 거친 뒤에야 작동하는 형태를 찾았으며, 실패한 두 시도가 오히려 격차의 "
        "원인을 특정하는 근거가 되었다.", st))

    # --- S3 예측 ---------------------------------------------------------
    mt = metric_table([
        ("GraphSAGE (그래프 신경망)", [0.2854, 0.7117, 0.6513, 0.7957], True),
        ("XGBoost (정형 모델)", [0.2744, 0.7033, 0.6456, 0.7915], False),
        ("LogReg (선형 기준선)", [0.2330, 0.6726, 0.6248, 0.7834], False),
    ])
    noise = "".join(
        f'<div class="nrow"><span>{T(m, 700)}</span>'
        f'<span class="nbar"><i style="width:{min(100, r/19*100):.0f}%" '
        f'class="{"ok" if ok else "no"}"></i></span>'
        f'<span class="nval">{T(f"{r:.1f}배", 700)}</span>'
        f'<span class="ntag">{T("주장 가능" if ok else "우연 범위", 400)}</span></div>'
        for m, r, ok in [("MCC", 19.0, True), ("AUC", 14.6, True),
                         ("Balanced Acc", 6.6, True), ("Precision", 1.2, False),
                         ("Specificity", 0.7, False), ("F1", 0.0, False)])
    h.append(section(
        "03", "PREDICT",
        "관계 구조로 재해석한 예측, 그리고 주장 가능한 지표",
        "사건을 낱개로 다루지 않고 같은 지역에서 발생한 사건들을 이어 붙인 뒤 검거 "
        "여부를 예측했다. 관계 구조를 반영한 모델이 정형 모델을 앞선다. 다만 "
        "'앞선다'를 어떤 지표로 주장할 수 있는지는 별도로 검증해야 한다.",
        f'<p class="note">{T("네 지표 모두 예측 성능을 측정하지만 측정 방식이 다르므로 "
                             "함께 본다. 모두 값이 클수록 좋다.")}</p>'
        + mt
        + f'<p class="lead2">{T("문제는 이 모델의 결과가 실행할 때마다 미세하게 "
                                "흔들린다는 점이다. 따라서 앞선 폭 자체가 아니라, 그 폭이 "
                                "실행 간 변동의 몇 배인지를 기준으로 삼아야 한다. 19배는 "
                                "확실한 차이지만 1.2배는 우연으로도 발생할 수 있는 "
                                "범위다.")}</p>'
        + f'<p class="note">{T("앞선 폭 ÷ 실행 간 변동 폭")}</p>'
        + f'<div class="noise">{noise}</div>'
        + f'<p class="callout">{T("성능이 좋아 보이는 지표를 스스로 배제한 결과다. "
                                  "Precision은 0.0042 앞섰고 이는 Balanced Accuracy의 "
                                  "0.0057과 유사해 보이지만, Precision은 실행 간 변동 폭이 "
                                  "5배 크기 때문에 이 정도 차이는 우연으로도 발생한다. "
                                  "따라서 성능 주장에 <b>사용하지 않았다.</b>")}</p>'
        + f'<p class="lead2">{T("관계 구조가 왜 도움이 되는지는 실패한 실험이 밝혀 "
                                "주었다. 이웃을 무작위가 아니라 <b>가장 유사한 사건</b>으로 "
                                "선택했더니 성능이 오히려 하락했다(0.0054 하락, 실행 간 "
                                "변동의 9배). 이웃들의 평균이 자기 자신과 거의 일치하게 "
                                "되어(유사도 0.64 → 0.95) 이웃으로부터 새로 얻을 정보가 "
                                "사라진 탓이다.", 700)}</p>'
        + f'<blockquote>{T("즉 그래프의 가치는 <b>유사한 사건을 찾는 데</b> 있지 않고 "
                           "<b>주변 맥락을 요약하는 데</b> 있다. 같은 지역에서 무작위로 "
                           "선택한 이웃 20건은 \'이 지역이 어떤 곳인가\'를 편향 없이 "
                           "추정하게 해 준다.", 700)}</blockquote>'))

    # --- S4 진단 ---------------------------------------------------------
    ladder = (
        f'<div class="bars"><div class="bhead">'
        f'{T("인종에 따른 격차 (백인 대 흑인)", 700)}</div>'
        + bar("전국 일괄", 1.805, 2.0, ci=(1.715, 1.904))
        + bar("주 단위 표준화", 1.558, 2.0, ci=(1.457, 1.674))
        + bar("카운티 단위 표준화", 0.662, 2.0, ci=(0.452, 0.888))
        + f'<div class="bhead">{T("성별에 따른 격차", 700)}</div>'
        + bar("전국 일괄", 1.246, 2.0)
        + bar("주 단위 표준화", 1.005, 2.0, ci=(0.925, 1.090))
        + f'<p class="note">{T("1.0이 기준선이며, 데이터에 이미 존재하던 격차를 모델이 "
                               "몇 배로 확대하는지를 뜻한다. 표준화는 해당 단위 안에서 "
                               "각각 산출한 뒤 평균낸 값이다. 막대 위 가는 선은 값이 놓일 "
                               "수 있는 범위(95% 신뢰구간)이며, 그 범위가 1.0을 넘어설 "
                               "때만 붉게 표시했다.")}</p></div>'
    )
    h.append(section(
        "04", "DIAGNOSE",
        "성별은 변수에서, 인종은 그래프에서",
        "입력에서 인종·성별 변수를 제거해 보았다. 성별 격차는 2.55~3.29배에서 "
        "0.88~0.96배로 무너진다. 변수를 지우면 격차도 함께 사라진 것이다. 인종은 "
        "그렇지 않다.",
        ladder
        + f'<p class="lead2">{T("변수를 제거한 조건에서 그래프 모델은 <b>1.48배</b>, "
                                "정형 모델은 0.67배를 기록했다. 두 모델을 짝지어 비교한 "
                                "차이는 <b>+0.82 [+0.65, +1.03]</b>이다. 동일한 모델을 "
                                "\'같은 지역\'이 아니라 \'같은 시기\'로 이어 붙인 그래프에 "
                                "올리면 0.36배로 하락한다. 모델도 특성도 학습 절차도 "
                                "동일하고 <b>연결 방식만</b> 다르므로, 남는 경로는 같은 "
                                "지역끼리 이어진 연결뿐이다.", 700)}</p>'
        + f'<p class="lead2">{T("카운티 단위로 표준화하면 0.662로 떨어진다. 이는 반증이 "
                                "아니라 확증이다. 사건을 이어 붙일 때 사용한 기준이 곧 "
                                "카운티이므로, 카운티를 고정하면 해당 경로가 통째로 "
                                "제거되는 것이 당연하다. 모델이 확대한 인종 격차의 "
                                "<b>88%가 카운티 간 차이</b>에서 비롯되었다는 "
                                "의미다.", 700)}</p>'
        + f'<blockquote>{T("다만 카운티 내부가 깨끗하다는 뜻은 아니다. 동일한 카운티 "
                           "안에서도 흑인 피해자 사건은 기대 대비 <b>8.7% 더</b> 미제로 "
                           "남는다(+0.083 [+0.056, +0.128]). 0이 아니다. 그러나 이 값과 "
                           "해당 카운티의 흑인 비중 사이에는 <b>상관이 없다</b>(-0.015). "
                           "카운티 내부의 격차는 지역에 관계없이 균일하게 존재하며, "
                           "지도가 보여 주는 것은 카운티 간 차이다.", 700)}</blockquote>'))

    # --- S5 처방 ---------------------------------------------------------
    alpha = "".join(
        f'<div class="arow"><span>{T(f"벌점 {a}", 700)}</span>'
        f'<span class="abar"><i class="{"red" if v > 1.0 else ""}" '
        f'style="width:{min(100, v/1.9*100):.0f}%"></i></span>'
        f'<span class="aval">{T(f"{v:.3f}", 700)}</span></div>'
        for a, v in [(0, 1.805), (25, 0.639), (50, 0.701), (100, 0.261)])
    h.append(section(
        "05", "PRESCRIBE",
        "손실 함수에 기록한 공정성 제약",
        "입력에서 인종을 제거하는 대신, 모델이 그것을 사용하지 못하도록 제약한다. "
        "학습 시 예측 오차를 알려 주는 손실에 '집단 간 예측이 벌어졌다'는 벌점을 함께 "
        "부과하는 방식이다. 인종 정보는 학습 시점에만 필요하므로, 실제 운영 시점에는 "
        "해당 속성을 몰라도 된다.",
        f'<div class="bars">{alpha}'
        f'<p class="note">{T("전국 기준, 미제 사건을 \'해결됨\'으로 잘못 판정하는 비율의 "
                             "인종 격차다. 사전에 정한 통과 기준은 \'값이 놓일 수 있는 "
                             "범위의 상한이 0.5 이하\'였고 벌점 100만 이를 통과했다"
                             "(0.261 [0.157, 0.367]). 그 대가로 정확도 0.0156을 "
                             "지불했다.")}</p></div>'
        + f'<p class="callout">{T("다만 이 통과 기준은 <b>전국을 일괄 산출한</b> 값에 "
                                  "근거한다. 주 단위로 표준화하면 0.590 [0.456, 0.738]이 "
                                  "되어 어떤 벌점도 기준을 통과하지 못한다. 벌점 자체가 "
                                  "일괄 격차를 기준으로 정의되어 있어 강도를 높인다고 "
                                  "개선되지도 않는다(25/50/100에서 0.639/0.701/0.590). "
                                  "이는 강도를 조절할 문제가 아니라 <b>주 단위로 벌점을 "
                                  "층화</b>해야 한다는 뜻이며, 그 작업은 아직 수행하지 "
                                  "않았다.", 700)}</p>'))
    return "".join(h)


# 이 지도는 오독되기 쉽고, 아래 문장들이 빠지면 "빨간 카운티 = 경찰이 일을 안 한다"로
# 읽힌다. build_web_map.CAVEATS를 그대로 쓰지 않고 **같은 내용을 쉬운 말로 다시 쓴다** --
# 저쪽은 분석을 읽는 사람이 독자이고 여기는 아니다. 다만 쉽게 쓰는 것과 약하게 쓰는
# 것은 다르므로, 내용은 하나도 빼지 않았고 숫자는 전부 남겼다.
HOWTO = [
    ("색의 의미",
     "카운티마다 '이 정도 사건 구성이라면 통상 이만큼은 미제로 남는다'는 기댓값을 먼저 "
     "산출한 뒤, 실제로 남은 건수를 그 기댓값과 견준다. <b>붉을수록 기대보다 많이 남은 "
     "지역</b>이며, 회색은 그 반대다."),
    ("색이 없는 지역",
     "차이가 우연으로 설명될 수 있는 범위이면 칠하지 않는다. 옅은 회색은 '판정했으나 "
     "우연 수준'이고, 빗금은 '사건 수가 부족해 판정 대상에서 제외'했다는 뜻이다. 색이 "
     "칠해진 지역은 모두 우연으로 보기 어려운 카운티다."),
    ("마우스를 올렸을 때의 숫자",
     "<b>배수</b> 1.4는 기대보다 40% 더 많다는 뜻이다. <b>확신도</b>는 그 차이가 얼마나 "
     "분명한지를 사건 수까지 반영해 산출한 값이므로, 같은 배수라도 사건이 많은 카운티에서 "
     "크게 나온다. <b>헛짚을 확률</b>은 해당 판정이 잘못되었을 가능성이며, 0.05 이하만 "
     "색을 칠했다."),
]

NOTES = [
    ("붉은 카운티는 '수사를 못한다'는 뜻이 아니다.",
     "무기 종류, 피해자 연령과 성별, 사건을 담당한 기관 유형, 소속 주까지 비슷하게 맞춘 "
     "뒤에도 <b>설명되지 않는 미제가 남는다</b>는 뜻이다. 왜 남는지까지는 이 그림이 "
     "말해 주지 않는다."),
    ("연쇄살인을 탐지하는 지도가 아니다.",
     "이 데이터에는 가해자를 식별하는 항목이 없다. 따라서 '이 사건들은 동일범의 소행'이라는 "
     "주장은 <b>원리적으로 검증이 불가능하다.</b> 여기서 수행하는 것은 개별 사건을 "
     "지목하는 일이 아니라, 지역 단위로 이상 여부를 판정하는 일이다."),
    ("모델은 사건이 발생한 카운티를 모른다 - 의도적으로 알려 주지 않았다.",
     "알려 줄 경우 모델은 '이 지역은 원래 검거가 어려우므로 미제로 남는 것이 정상'이라고 "
     "학습한다. 그러면 탐지하려던 이상이 정상으로 처리되어 지도에서 사라진다. 대신 해당 "
     "카운티의 사정이 통째로 색에 남는다. 그 안에는 <b>사건 구성, 차별, 수사 자원, 기록 "
     "관행</b>이 섞여 있으며 이 설계로는 넷을 분리할 수 없다. 색을 근거로 원인을 단정해서는 "
     "안 되는 이유다."),
    ("색과 인종 구성의 상관은 '카운티 간'의 관계이지 '카운티 내부'의 관계가 아니다.",
     "흑인 피해자 비중이 높은 카운티일수록 색이 진한 것은 사실이다(+0.29). 그러나 카운티를 "
     "고정한 뒤 그 안에서 인종별로 다시 산출하면, 흑인 피해자 사건은 기대 대비 "
     "<b>8.7% 더</b> 미제로 남는다. 0이 아니다. 그런데 이 8.7%는 흑인 비중이 높은 "
     "카운티든 낮은 카운티든 <b>거의 동일하다</b>(-0.015). 카운티 내부의 격차는 지역과 "
     "무관하게 균일하게 존재하며, 지도가 보여 주는 것은 카운티 간 차이다."),
    ("'수사 인력이 부족해서가 아닌가'는 실제로 측정했다 - 다만 하한만 안다.",
     "기관별·연도별 경찰관 수를 결합해(누락 없이 연결되었다) 사건 1건당 경찰관 수를 통제한 "
     "뒤 다시 산출했다. 인종 구성의 영향은 전수 기준으로 <b>거의 감소하지 않았고</b> 사건이 "
     "많은 카운티에서만 8~10% 감소했다. 다만 경찰 인력은 사건 수에 따라 배치되므로 이 "
     "변수 자체의 변동 폭이 절반으로 압축되어 있다. 따라서 <b>8%는 상한이 아니라 "
     "하한</b>이다. 실제 자원의 몫은 더 클 수 있으며, 상한은 이 설계로 구할 수 없다. "
     "게다가 측정한 것은 전체 경찰관 수이지 형사 인원이 아니다."),
    ("기댓값을 산출한 모델은 공정성 완화를 거쳤다.",
     "완화하지 않은 모델은 인종에 따라 오차가 1.8배 기울어 있다. 그런 모델로 기댓값을 "
     "만들면 '흑인 피해자 사건이므로 미제로 남는 것이 정상'이라는 판단이 기준선에 포함되어, "
     "측정하려던 격차가 사라진다. 완화한 모델만이 인종에 중립적인 기준선을 제공한다. "
     "따라서 색이 인종 구성과 상관되면 그것은 잡음이 아니라 <b>결과</b>다."),
    ("사건 수가 적은 카운티는 칠하지 않는다.",
     "표본이 작으면 우연만으로도 이상해 보이기 쉽다. 최소 사건 수 기준은 슬라이더로 조정할 "
     "수 있으며, 기준마다 <b>처음부터 다시 검정한</b> 결과다. 이미 산출된 결과를 필터링한 "
     "것이 아니다. 동시에 검정한 지역이 몇 곳인지에 따라 판정 기준 자체가 달라지기 "
     "때문이다."),
    ("예측값은 해당 사건을 학습하지 않은 상태에서 산출한 것이다.",
     "전체를 다섯 조각으로 나누어, 네 조각으로 학습하고 나머지 한 조각을 예측하는 절차를 "
     "다섯 번 반복했다. 그 결과 638,454건 전부에 '학습에 사용되지 않은 예측'이 부여되고 "
     "카운티 1,803곳(57%)을 판정할 수 있다. 다만 다룬 사건 수가 늘어난 만큼 확신도가 "
     "전반적으로 크게 나온다. <b>다른 지도와 색의 강도를 비교해서는 안 된다.</b> "
     "이상의 크기(배수)는 두 지도에서 거의 같고, 달라지는 것은 확신의 정도다."),
    ("재수사 우선순위 목록은 산출했으나 공개하지 않는다.",
     "정확도를 검증할 경로가 없기 때문이다. 재수사 결과가 데이터에 존재하지 않는다. 또한 "
     "개별 사건 단위로 내려가면 예측 정확도가 낮아 사실상 잡음이 지배한다. 따라서 "
     "<b>지역 단위까지만</b> 공개한다."),
    ("사건 정황으로 사건 구성을 통제하는 것은 오히려 편향을 만든다.",
     "별도 자료에서 사건 정황을 결합하는 데는 성공했다(96% 연결). 그러나 그 정황은 사건이 "
     "해결된 뒤에 기록되는 값이다. 미제 사건의 <b>56%가 '미확정'</b>으로 기록되어 있고, "
     "동일한 카운티 안에서도 흑인 피해자 사건이 <b>4.1%p 더 자주</b> '미확정'으로 "
     "기록된다. 이것으로 통제하면 '기록을 덜 남겼다'는 사실이 '사건이 원래 어려웠다'로 "
     "치환된다."),
    ("지도에서 알래스카만 축척이 다르다.",
     "면적이 왜곡되지 않는 투영을 사용했으나, 알래스카는 지면에 담기 위해 "
     "<b>0.35배로 축소</b>했다. 알래스카 카운티의 화면 면적을 본토와 비교해서는 안 된다. "
     "하와이는 본토와 같은 축척이다."),
    ("면적이 작지만 신호가 강한 카운티가 있다.",
     "상위권의 상당수가 독립시(세인트루이스, 볼티모어, 리치먼드 등)라 전국 지도에서는 점처럼 "
     "보인다. <b>면적이 작은 것이지 신호가 약한 것이 아니다.</b> 마우스를 올리거나 "
     "'표로 보기'로 확인할 수 있다."),
]


def build_notes(split):
    return "".join(f'<li><b>{T(h, 700)}</b> {T(b)}</li>' for h, b in NOTES)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:__BG__;color:__INK__;
  font-family:__FAM__,-apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",
  system-ui,sans-serif;font-weight:400;font-size:16px;line-height:1.75;
  letter-spacing:-.01em;font-synthesis:none;
  -webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
b,strong{font-weight:700}
.inner{max-width:1080px;margin:0 auto;padding:0 24px}
section.sec{padding:110px 0;border-top:1px solid __HAIR__}
section.sec:nth-child(even){background:__BG2__}
.shead{display:flex;align-items:center;gap:14px;margin-bottom:18px}
.snum{font-weight:800;font-size:13px;color:__ACC__;letter-spacing:.14em}
.slab{font-weight:800;font-size:13px;color:__MUT__;letter-spacing:.22em}
h2{font-weight:800;font-size:clamp(26px,3.6vw,40px);line-height:1.32;
  margin:0 0 20px;letter-spacing:-.025em}
.lead{color:__INK2__;max-width:60ch;margin:0 0 30px;font-size:17px}
.lead2{color:__INK2__;max-width:64ch;margin:26px 0 0}
.note{color:__MUT__;font-size:13px;margin:12px 0 0;max-width:62ch}
.callout{margin:26px 0 0;padding:18px 20px;border-left:2px solid __ACC__;
  background:#ffffff06;color:__INK2__;font-size:15px;max-width:66ch}
blockquote{margin:28px 0 0;padding:22px 24px;border:1px solid __HAIR__;
  border-radius:12px;background:#ffffff05;color:__INK__;font-size:17px;max-width:66ch}
/* HERO -- 타이포그래피만. 그림이 없으므로 여백과 크기 대비가 전부 일을 한다. */
#hero{min-height:100vh;display:flex;align-items:center;position:relative;
  padding:120px 0 110px}
.hlab{font-weight:800;font-size:12px;letter-spacing:.26em;color:__MUT__;
  margin:0 0 26px}
.htitle{font-weight:800;font-size:clamp(58px,12vw,168px);letter-spacing:-.055em;
  line-height:.9;margin:0;color:#fff}
.hkr{font-weight:800;font-size:clamp(22px,3.4vw,42px);letter-spacing:-.035em;
  margin:22px 0 0;color:__INK__;line-height:1.25}
.hkr em{font-style:normal;color:__ACC__}
.hsub{color:__INK2__;margin:26px 0 0;max-width:52ch;font-size:16px}
.hmeta{display:flex;flex-wrap:wrap;gap:12px 30px;margin:44px 0 0;
  color:__MUT__;font-size:12.5px;letter-spacing:.04em}
.hmeta span{white-space:nowrap}
.hhint{position:absolute;bottom:34px;left:0;right:0;text-align:center;
  color:__MUT__;font-size:11.5px;letter-spacing:.2em}
.hhint i{display:block;margin-top:8px;font-style:normal;
  animation:bob 1.8s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(5px)}}
.hhint i{display:block;margin-top:8px;animation:bob 1.8s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(5px)}}
/* counters */
.counters{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:26px;margin-top:8px}
.cnt b{display:block;font-weight:800;font-size:clamp(30px,4.4vw,46px);
  letter-spacing:-.03em;line-height:1.1}
.cnt span{display:block;font-weight:700;font-size:14px;margin-top:8px}
.cnt i{display:block;font-style:normal;color:__MUT__;font-size:12px;margin-top:2px}
.cnt:last-child b{color:__ACC__}
/* steps */
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:20px}
.step{border:1px solid __HAIR__;border-radius:12px;padding:20px 18px;
  background:#ffffff04}
.sn{font-weight:800;font-size:12px;color:__MUT__;letter-spacing:.1em}
.se{display:block;font-weight:700;font-size:11px;color:__ACC__;
  letter-spacing:.18em;margin:8px 0 2px}
.step b{font-weight:700;font-size:18px}
.step p{color:__INK2__;font-size:14px;margin:8px 0 0;line-height:1.65}
/* metric table */
table.mt{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 0}
table.mt th,table.mt td{padding:11px 12px;border-bottom:1px solid __HAIR__;
  text-align:right}
table.mt th:first-child,table.mt td:first-child{text-align:left}
table.mt th{font-weight:700;color:__MUT__;font-size:12px;letter-spacing:.06em}
td.win{color:__INK__}
.mname{color:__INK2__}
tr:first-child .mname,tr:first-child td.win{color:__INK__}
/* noise bars */
.noise{margin-top:8px}
.nrow{display:grid;grid-template-columns:118px 1fr 54px 74px;gap:12px;
  align-items:center;padding:5px 0;font-size:13px}
.nbar{background:#ffffff0d;height:6px;border-radius:3px;overflow:hidden}
.nbar i{display:block;height:100%;border-radius:3px}
.nbar i.ok{background:__INK__}
.nbar i.no{background:#ffffff26}
.nval{font-weight:700;text-align:right}
.ntag{color:__MUT__;font-size:12px}
/* generic bars */
.bars{margin-top:6px}
.bhead{font-weight:700;font-size:13px;color:__MUT__;letter-spacing:.06em;
  margin:22px 0 8px}
.brow{display:grid;grid-template-columns:118px 1fr 62px;gap:14px;align-items:center;
  padding:5px 0;font-size:14px}
.blab{color:__INK2__;font-size:13px}
.btrack{position:relative;background:#ffffff0d;height:9px;border-radius:5px}
.btrack:before{content:"";position:absolute;left:50%;top:-4px;bottom:-4px;
  width:1px;background:#ffffff2e}
.bar{display:block;height:100%;border-radius:5px;background:#ffffff5c}
.bar.red{background:__ACC__}
.ci{position:absolute;top:50%;height:1px;background:__INK__;opacity:.55;
  transform:translateY(-50%)}
.bval{font-weight:700;text-align:right;font-size:13px}
.arow{display:grid;grid-template-columns:110px 1fr 62px;gap:14px;align-items:center;
  padding:5px 0;font-size:14px}
.abar{background:#ffffff0d;height:9px;border-radius:5px;overflow:hidden}
.abar i{display:block;height:100%;background:#ffffff5c;border-radius:5px}
.abar i.red{background:__ACC__}
.aval{font-weight:700;text-align:right;font-size:13px}
/* MAP CARD (밝은 반전) */
#mapsec{padding:110px 0;border-top:1px solid __HAIR__}
.card{background:__SURF__;color:#0b0b0b;border-radius:16px;padding:26px 24px 30px;
  margin-top:26px}
.card h3{font-weight:800;font-size:19px;margin:0 0 4px;letter-spacing:-.02em}
.card .csub{color:#52514e;font-size:13px;margin:0 0 18px;max-width:78ch}
.bar2{display:flex;flex-wrap:wrap;gap:18px;align-items:center;padding:11px 13px;
  border:1px solid #e1e0d9;border-radius:10px;margin-bottom:13px}
.bar2 label{font-size:12px;color:#52514e;margin-right:6px;font-weight:700}
.card button{font:inherit;font-size:12px;font-weight:700;padding:5px 12px;
  border:1px solid #c3c2b7;background:transparent;color:#52514e;border-radius:999px;
  cursor:pointer;transition:background .12s,color .12s}
.card button[aria-pressed="true"]{background:#0b0b0b;color:#fff;border-color:#0b0b0b}
.lexp{margin:0 0 13px;padding:11px 14px;background:#0b0b0b0a;
  border-left:2px solid #c3c2b7;border-radius:0 8px 8px 0;
  font-size:13px;line-height:1.68;color:#52514e;max-width:88ch}
.lexp b{color:#0b0b0b;font-weight:700}
.mapwrap{position:relative;border:1px solid #e1e0d9;border-radius:10px;
  overflow-x:auto;background:#fff}
svg.map{display:block;width:100%;height:auto}
svg.map path{stroke:#fff;stroke-width:.35;vector-effect:non-scaling-stroke}
svg.map path.sig{stroke:#0b0b0b;stroke-width:.9}
#hl path{pointer-events:none;transform-box:fill-box;transform-origin:center;
  transform:scale(1);transition:transform .13s cubic-bezier(.2,.7,.3,1);
  stroke:#0b0b0b;stroke-width:1.1;vector-effect:non-scaling-stroke;
  filter:drop-shadow(0 2px 7px rgba(0,0,0,.38))}
#hl path.on{transform:scale(1.09)}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;
  background:#0b0b0b;color:#fff;padding:9px 11px;border-radius:8px;font-size:12px;
  line-height:1.55;max-width:290px;z-index:9;box-shadow:0 6px 20px #0004}
#tip b{font-size:13px;font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:20px;margin:13px 0 0;font-size:12px;
  color:#52514e;align-items:center}
.sw{display:inline-block;width:20px;height:11px;vertical-align:-1px;margin-right:5px;
  border:1px solid #0002}
.tablewrap{max-height:420px;overflow:auto;border:1px solid #e1e0d9;border-radius:10px;
  margin-top:10px}
table.ct{border-collapse:collapse;width:100%;font-size:12px}
table.ct th,table.ct td{padding:6px 9px;border-bottom:1px solid #e1e0d9;
  text-align:right;white-space:nowrap}
table.ct th:first-child,table.ct td:first-child,
table.ct th:nth-child(2),table.ct td:nth-child(2){text-align:left}
table.ct th{color:#52514e;font-weight:700;position:sticky;top:0;background:__SURF__}
table.ct tbody tr{cursor:default}
table.ct tbody tr:hover{background:#0000000a}
/* 읽는 법 카드 */
.howto{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:18px;margin:26px 0 6px}
.hcard{border:1px solid __HAIR__;border-radius:12px;padding:18px 19px;
  background:#ffffff05}
.hcard b{font-weight:700;font-size:15px;display:block;margin-bottom:7px}
.hcard p{margin:0;color:__INK2__;font-size:14px;line-height:1.7}
.hcard p b{display:inline;font-size:inherit;color:__INK__;margin:0}
/* notes / foot */
.notes{margin-top:52px}
.notes h3{font-weight:800;font-size:19px;margin:0 0 16px;letter-spacing:-.02em}
.notes ol{padding-left:22px;margin:0}
.notes li{margin-bottom:16px;color:__INK2__;font-size:14.5px;line-height:1.78;
  padding-left:4px}
.notes li b{color:__INK__}
.notes li>b:first-child{display:block;margin-bottom:3px;font-size:15px}
footer{border-top:1px solid __HAIR__;padding:70px 0 90px}
.who{font-weight:800;font-size:clamp(24px,3vw,34px);letter-spacing:-.03em;margin:0}
.who a{color:__INK__;text-decoration:none;border-bottom:1px solid #ffffff33}
.who a:hover{border-bottom-color:__INK__}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:24px;margin-top:34px;color:__INK2__;font-size:13.5px}
.meta b{display:block;font-weight:700;color:__MUT__;font-size:11px;
  letter-spacing:.16em;margin-bottom:7px}
.meta code{font-size:12px;color:__INK2__;display:block;line-height:1.9;
  word-break:break-all}
.lic{color:__MUT__;font-size:11.5px;margin-top:36px;line-height:1.7}
/* scroll reveal -- .js가 붙었을 때만 숨긴다. 스크립트가 죽으면 IntersectionObserver도
   안 도는데, 기본값이 opacity:0이면 페이지 전체가 빈 화면이 된다. */
.js .sec .inner,.js #mapsec .inner,.js footer .inner{opacity:0;
  transform:translateY(16px);
  transition:opacity .6s ease,transform .6s cubic-bezier(.2,.7,.3,1)}
.js .sec.in .inner,.js #mapsec.in .inner,.js footer.in .inner{opacity:1;
  transform:none}
@media (max-width:640px){
  section.sec,#mapsec{padding:76px 0}
  body{font-size:15px}
  .nrow{grid-template-columns:96px 1fr 46px}
  .nrow .ntag{display:none}
  .brow,.arow{grid-template-columns:104px 1fr 54px}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *{animation:none!important;transition:none!important}
  .js .sec .inner,.js #mapsec .inner,.js footer .inner{opacity:1;transform:none}
}
"""


def css():
    return (CSS.replace("__BG2__", BG2).replace("__BG__", BG)
            .replace("__INK2__", INK2).replace("__INK__", INK)
            .replace("__MUT__", MUTED).replace("__HAIR__", HAIR)
            .replace("__ACC__", ACCENT).replace("__SURF__", CT.SURFACE)
            .replace("__FAM__", FP.FAMILY))


JS = r"""
document.documentElement.classList.add('js');

/* ---------------- 카운터 ---------------- */
(function(){
var io=new IntersectionObserver(function(es){
  es.forEach(function(e){
    if(!e.isIntersecting) return;
    e.target.classList.add('in');
    io.unobserve(e.target);
  });
},{threshold:0.12});
document.querySelectorAll('.sec,#mapsec,footer').forEach(function(s){io.observe(s);});
})();

/* ---------------- 지도 ---------------- */
(function(){
var g=document.getElementById('g'), hl=document.getElementById('hl'),
    tip=document.getElementById('tip');
var layer='res', li=0, hoverIdx=-1;
var nodes=FI.map(function(f){
  var p=document.createElementNS('http://www.w3.org/2000/svg','path');
  p.setAttribute('d',PATHS[f]); g.appendChild(p); return p;
});
function st(i){ return PL[LV[li]].s[i]; }
function paint(){
  for(var i=0;i<nodes.length;i++){
    var s=st(i), c;
    if(s===0) c=-1;
    else if(layer==='res') c = (s===2 ? S.res[i] : 0);
    else c = S[layer][i];
    nodes[i].setAttribute('fill', c<0 ? 'url(#na)' : P[c]);
    nodes[i].classList.toggle('sig', layer==='res' && s===2);
  }
  document.getElementById('mnv').textContent=LV[li];
  document.getElementById('lexp').innerHTML=L['exp_'+layer];
  hover(-1); legend(); table();
}
function sw(c){ return '<span class="sw" style="background:'+c+'"></span>'; }
function legend(){
  var h='';
  if(layer==='res'){
    h+='<span>'+L.more+' '+[10,9,8,7,6].map(function(i){return sw(P[i]);}).join('')+'</span>';
    h+='<span>'+[1,2,3,4,5].map(function(i){return sw(P[i]);}).join('')+' '+L.less+'</span>';
    h+='<span>'+sw(P[0])+L.nonsig+'</span>';
  } else {
    h+='<span>'+L.low+' '+[1,2,3,4,5].map(function(i){return sw(P[i]);}).join('')+' '+L.high+'</span>';
  }
  h+='<span><span class="sw" style="background:#fff;background-image:'
   +'repeating-linear-gradient(45deg,#c3c2b7 0 1.4px,#fff 1.4px 5px)"></span>'
   +L.nodata+' (n &lt; '+LV[li]+')</span>';
  document.getElementById('leg').innerHTML=h;
}
function table(){
  var q=PL[LV[li]].q, rows=[];
  for(var i=0;i<FI.length;i++) if(st(i)) rows.push([i,S.stat[i],q[i]]);
  rows.sort(function(a,b){ return b[1][6]-a[1][6]; });
  document.querySelector('#ct tbody').innerHTML=rows.map(function(r){
    var t=r[1];
    return '<tr data-i="'+r[0]+'"><td>'+t[0]+'</td><td>'+t[1]+'</td><td>'+t[2]
      +'</td><td>'+t[3]+'</td><td>'+t[4]+'</td><td>'+t[5]+'</td><td>'+t[6]
      +'</td><td>'+r[2]+'</td><td>'+t[7]+'</td></tr>';
  }).join('');
}
/* 호버 확대: 원본은 그대로 두고 **맨 위 레이어에 복제본**을 올려 키운다.
   제자리에서 키우면 인접 카운티와 경계에 겹침/틈이 생겨 지도가 찢어져 보인다. */
function hover(i){
  if(i===hoverIdx) return;
  hoverIdx=i;
  hl.innerHTML='';
  if(i<0){ tip.style.opacity=0; return; }
  var c=nodes[i].cloneNode(false);
  c.setAttribute('fill', nodes[i].getAttribute('fill'));
  c.removeAttribute('class');
  hl.appendChild(c);
  /* scale(1)로 붙인 직후 scale(1.09)를 걸어야 transition이 돈다. 다음 프레임을
     기다리면(requestAnimationFrame) 프레임이 늦거나 스로틀될 때 확대가 통째로
     빠진다 -- 실제로 헤드리스 검사에서 클래스가 안 붙은 채로 잡혔다.
     getBoundingClientRect로 리플로우를 강제해 시작 상태를 확정하고 바로 건다. */
  void c.getBoundingClientRect();
  c.classList.add('on');
}
function showTip(i,x,y){
  var t=S.stat[i], qq=PL[LV[li]].q[i], s=st(i);
  /* SMR을 그대로 보여 주면 1.4가 무슨 뜻인지 아무도 모른다. 배수를 퍼센트 문장으로
     바꿔서 먼저 놓고, 원래 값은 그 아래에 남긴다. */
  var pct=Math.round((t[5]-1)*100);
  var v = s===1 ? L.v_none
        : (pct>=0 ? L.v_more.replace('%s',pct) : L.v_less.replace('%s',-pct));
  tip.innerHTML='<b>'+t[1]+', '+t[0]+'</b><br>'+v+'<br>'+L.tip_cases+' '+t[2]+' / '
    +L.tip_unsolved+' '+t[3]+'<br>'+L.tip_expected+' '+t[4]+' / SMR '+t[5]
    +'<br>z '+t[6]+' / q '+qq+'<br>'+L.tip_black+' '+t[7];
  tip.style.opacity=1;
  tip.style.left=Math.min(x+14, innerWidth-300)+'px';
  tip.style.top=(y+14)+'px';
}
g.addEventListener('pointermove',function(e){
  var i=nodes.indexOf(e.target);
  if(i<0||!st(i)){ hover(-1); return; }
  hover(i); showTip(i,e.clientX,e.clientY);
});
g.addEventListener('pointerleave',function(){ hover(-1); });
document.querySelector('#ct tbody').addEventListener('pointerover',function(e){
  var tr=e.target.closest('tr'); if(!tr) return;
  hover(+tr.dataset.i);
});
document.querySelector('#ct tbody').addEventListener('pointerleave',function(){
  hover(-1);
});
document.querySelectorAll('[data-layer]').forEach(function(b){
  b.onclick=function(){
    layer=b.dataset.layer;
    document.querySelectorAll('[data-layer]').forEach(function(o){
      o.setAttribute('aria-pressed', o===b);
    });
    paint();
  };
});
document.getElementById('mn').oninput=function(e){ li=+e.target.value; paint(); };
document.getElementById('tbl').onclick=function(e){
  var w=document.getElementById('tablewrap');
  w.hidden=!w.hidden;
  e.target.setAttribute('aria-pressed', !w.hidden);
};
paint();
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_key", default="county")
    ap.add_argument("--src", default="cold_blocks_cv5.csv",
                    help="블록 표. 기본은 크로스피팅(카운티 1,803개).")
    ap.add_argument("--simplify_km", type=float, default=SIMPLIFY_KM)
    ap.add_argument("--out", default="clear_story.html")
    ap.add_argument("--fonts_dir", default=None,
                    help="NanumSquare ttf가 있는 폴더(기본: 시스템 폰트 경로 탐색)")
    ap.add_argument("--no_fonts", action="store_true",
                    help="폰트 임베드를 건너뛰고 시스템 스택만 쓴다")
    args = ap.parse_args()

    # ---- 데이터 (build_web_map 재사용) ----------------------------------
    tab = load_blocks(args.block_key, args.src)
    tab, missing = CT.join_fips(tab)
    matched = tab["fips"].notna()
    print(f"[join] FIPS {int(matched.sum())}/{len(tab)} ({matched.mean()*100:.1f}%)")
    if len(missing):
        print(f"[경고] 미매칭 {len(missing)}건 - clear.counties.ALIASES 확인")
    tab = tab[matched].copy()

    state_fips = set(tab["fips"].str[:2])
    paths, height = build_paths(state_fips, args.simplify_km)
    fips_order = sorted(paths)
    palette, static, per, levels, vmax, _edges = color_tables(tab, fips_order)

    # 붉은색은 미해결 집중 한 가지 뜻으로만 쓴다: warm 팔의 채도를 죽인다.
    blue, red = CT.arms(5)
    assert palette == [CT.NEUTRAL] + blue + red, "color_tables 팔레트 구성이 바뀌었다"
    palette = [CT.NEUTRAL] + CT.gray_arm() + red

    lv0 = levels[0]
    n_cold = sum(1 for s in per[lv0]["s"] if s == 2 and True)
    cold_n = int(((tab["min_n"] == lv0) & (tab["flag"] == "cold")).sum())
    warm_n = int(((tab["min_n"] == lv0) & (tab["flag"] == "warm")).sum())
    assessed = sum(1 for s in per[lv0]["s"] if s)
    sub0 = tab[tab["min_n"] == lv0]
    corr = float(sub0["z"].corr(sub0["black_share"]))
    n_rows = int(sub0["n"].sum())
    print(f"[geo] 카운티 {len(fips_order):,}개 / 판정 {assessed:,}개 / "
          f"cold {cold_n} · warm {warm_n} / z~black {corr:+.3f}")

    split = "crossfit" if re.search(r"_cv\d+", args.src) else "test"

    # ---- 본문 -----------------------------------------------------------
    register_labels()
    # 표·툴팁에 나오는 지명. 툴팁의 카운티명은 <b>이므로 700에도 등록한다.
    for row in static["stat"]:
        if row:
            G.add(str(row[0]), 400); G.add(str(row[0]), 700)
            G.add(str(row[1]), 400); G.add(str(row[1]), 700)

    title = T("CLEAR", 800)
    hero_kr = T("미제로 남는 사건, 그 구조적 격차", 800)
    hero_sub = T("미국 살인사건 638,454건을 지역·시기·수법이라는 관계 정보로 이어 "
                 "그래프로 재구성하고, 그래프 신경망으로 각 사건의 검거 여부를 "
                 "예측한다. 다만 이 프로젝트의 핵심은 예측 성능이 아니라, 그 예측이 "
                 "피해자의 인종과 성별에 따라 어떻게 달라지는지를 진단하고 완화하는 "
                 "데 있다.")

    body = build_body((638454, len(fips_order), assessed, cold_n, warm_n, corr))
    notes = build_notes(split)

    # 지도 섹션
    map_head = (
        f'<div class="shead"><span class="snum">{T("06", 800)}</span>'
        f'<span class="slab">{T("APPLY", 800)}</span></div>'
        f'<h2>{T("설명되지 않는 미제의 지리적 집중", 800)}</h2>'
        f'<p class="lead">{T("이 프로젝트가 최종적으로 제시하는 것은 사건 목록이 "
                             "아니라 지도다. 어느 지역에 설명되지 않는 미제가 집중되는지, "
                             "그리고 그 색을 어디까지 신뢰할 수 있는지를 함께 제시한다.")}</p>')
    howto = '<div class="howto">' + "".join(
        f'<div class="hcard"><b>{T(q, 700)}</b><p>{T(a)}</p></div>'
        for q, a in HOWTO) + "</div>"
    th = "".join(f"<th>{T(x, 700)}</th>" for x in LABELS["th"])
    card = (
        f'<div class="card">'
        f'<h3>{T("기대 대비 미제 잔여량", 800)}</h3>'
        f'<p class="csub">{T(f"판정 대상 카운티 {assessed:,}곳 가운데 기대보다 많이 "
                             f"남은 지역이 {cold_n}곳, 적게 남은 지역이 {warm_n}곳이다. "
                             f"색의 강도와 해당 카운티의 흑인 피해자 비중은 {corr:+.2f}의 "
                             f"상관을 보인다.")}</p>'
        f'<div class="bar2">'
        f'<span><label>{T("레이어", 700)}</label>'
        f'<button data-layer="res" aria-pressed="true">{LABELS["layer_res"]}</button> '
        f'<button data-layer="pri" aria-pressed="false">{LABELS["layer_pri"]}</button> '
        f'<button data-layer="raw" aria-pressed="false">{LABELS["layer_raw"]}</button>'
        f'</span>'
        f'<span><label>{LABELS["minsample"]}</label>'
        f'<input id="mn" type="range" min="0" max="{len(levels)-1}" step="1" value="0">'
        f'<b id="mnv"></b> <span style="color:#898781">{LABELS["cases"]}</span></span>'
        f'<span><button id="tbl" aria-pressed="false">{LABELS["table"]}</button></span>'
        f'</div>'
        f'<p class="lexp" id="lexp"></p>'
        f'<div class="mapwrap"><svg class="map" viewBox="0 0 {VIEW_W:.0f} {height:.0f}" '
        f'role="img" aria-label="{T("카운티별로 기대 대비 미제가 얼마나 더 남았는지를 "
                                    "나타낸 미국 지도")}">'
        f'<defs><pattern id="na" width="5" height="5" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="5" height="5" fill="#fff"/>'
        f'<line x1="0" y1="0" x2="0" y2="5" stroke="#c3c2b7" stroke-width="1.4"/>'
        f'</pattern></defs><g id="g"></g><g id="hl"></g></svg></div>'
        f'<div class="legend" id="leg"></div>'
        f'<div id="tablewrap" class="tablewrap" hidden><table class="ct" id="ct">'
        f'<thead><tr>{th}</tr></thead><tbody></tbody></table></div>'
        f'</div>')

    foot_meta = (
        f'<div class="meta">'
        f'<div><b>{T("구현", 800)}</b>{T("Python · PyTorch Geometric · GraphSAGE")}'
        f'<br>{T("노드 638,454건, 엣지 2,500만 개, 사건당 특성 123개")}</div>'
        f'<div><b>{T("데이터", 800)}</b>'
        f'{T("Murder Accountability Project / Kaggle Homicide Reports 1980-2014")}'
        f'<br>{T("LEOKA openICPSR 102180 · SHR openICPSR 100699")}</div>'
        f'<div><b>{T("재현", 800)}</b><code>'
        f'{T("export CLEAR_SCOPE=national")}<br>'
        f'{T("python -m experiments.crossfit_predictions --minibatch")}<br>'
        f'{T("python -m experiments.build_story_page")}</code></div>'
        f'</div>')

    lic = T("본문 서체는 나눔스퀘어(네이버)이며, 사용된 글자만 추출해 파일에 "
            "포함했다. 재배포 조건은 배포처 고지를 따른다.")
    geo_note = T("경계 도형은 US Census 카운티 자료를 사용했다. 외부 요청이 전혀 없어 "
                 "네트워크 없이도 열람할 수 있다.")
    footer = (
        f'<footer><div class="inner">'
        f'<p class="who">{T("안윤지", 800)} · '
        f'<a href="mailto:ayj9665@gmail.com">{T("ayj9665@gmail.com", 800)}</a></p>'
        f'{foot_meta}'
        f'<p class="lic">{lic}<br>{geo_note}</p>'
        f'</div></footer>')

    hmeta = "".join(
        f'<span>{T(x, 700)}</span>' for x in
        [f"살인사건 {638454:,}건", "1980-2014", f"카운티 {len(fips_order):,}곳",
         "GraphSAGE", "공정성 완화 적용"])
    page = (
        f'<section id="hero"><div class="inner">'
        f'<p class="hlab">{T("CLEARANCE LEARNING & EQUITY ASSESSMENT ON GRAPHS", 800)}'
        f'</p>'
        f'<h1 class="htitle">{title}</h1>'
        f'<p class="hkr">{hero_kr}</p>'
        f'<p class="hsub">{hero_sub}</p>'
        f'<div class="hmeta">{hmeta}</div>'
        f'</div>'
        f'<div class="hhint">{T("아래로 스크롤", 700)}<i>&#8595;</i></div>'
        f'</section>'
        f'{body}'
        f'<section id="mapsec"><div class="inner">{map_head}{howto}{card}'
        f'<div class="notes"><h3>{T("지도 해석 시 유의사항", 800)}</h3>'
        f'<ol>{notes}</ol></div></div></section>'
        f'{footer}')

    # ---- 폰트 (본문 확정 후) --------------------------------------------
    face_css, report = ("", {"missing_fonts": ["skipped"]}) if args.no_fonts \
        else FP.build(G.w, args.fonts_dir)
    print(FP.format_report(report) if not args.no_fonts else "[font] 임베드 생략")

    data_js = (
        f'var S={json.dumps(static, ensure_ascii=False, separators=(",", ":"))},'
        f'PL={json.dumps(per, separators=(",", ":"))},'
        f'P={json.dumps(palette)},'
        f'PATHS={json.dumps(paths, separators=(",", ":"))},'
        f'LV={json.dumps([int(m) for m in levels])},'
        f'FI={json.dumps(fips_order, separators=(",", ":"))},'
        f'L={json.dumps(LABELS, ensure_ascii=False, separators=(",", ":"))};')

    html = (f'<title>{T("CLEAR - 미제사건의 구조적 격차 진단", 800)}</title>'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="description" content="'
            f'{T("미국 살인사건 검거 예측, 공정성 진단과 완화, 그리고 카운티 단위 잔차 지도")}">'
            f'<style>{face_css}{css()}</style>'
            f'{page}<div id="tip"></div>'
            f'<script>{data_js}{JS}</script>')

    out = C.scoped_output("web")
    out.mkdir(parents=True, exist_ok=True)
    p = out / args.out
    p.write_text(html, encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"[save] {p}  ({kb:.0f} KB, 외부 요청 0)")

    check(html, palette, static, per, levels, fips_order, kb)


def check(html, palette, static, per, levels, fips_order, kb):
    """빌드 자체 검사. 조용히 틀린 페이지를 내보내지 않는다."""
    errs = []
    # 1) 외부 요청 0. base64 페이로드에는 //xxxx 꼴이 무수히 들어 있으므로
    #    먼저 지우고 검사한다 -- 안 지우면 폰트를 심는 순간 검사가 40건씩 거짓양성을
    #    낸다(실제로 그랬다).
    scrub = re.sub(r"base64,[A-Za-z0-9+/=]+", "base64,X", html)
    for m in re.finditer(r"https?://[^\s\"')]+", scrub):
        if m.group(0).startswith("http://www.w3.org"):
            continue                              # SVG 네임스페이스는 요청이 아니다
        errs.append(f"외부 URL: {m.group(0)}")
    for m in re.finditer(r"[\"'(\s]//[a-z0-9.\-]{3,}", scrub, re.I):
        errs.append(f"프로토콜 상대 URL: {m.group(0).strip()}")
    # 2) 빨강은 z>0에만. res 인덱스 6..10은 z>0, 1..5는 z<0
    for i, f in enumerate(fips_order):
        c, stat = static["res"][i], static["stat"][i]
        if stat is None or c < 0:
            continue
        # stat의 z는 표시용으로 소수 2자리 반올림된 값이다. 정확히 0.00으로 찍히는
        # 카운티(실제 z=+0.0026 등)는 이 정밀도에서 부호를 판정할 수 없으므로 건너뛴다
        # -- 안 그러면 반올림이 색 규칙 위반으로 보고된다.
        if stat[6] == 0.0:
            continue
        if c >= 6 and stat[6] < 0:
            errs.append(f"{f}: 빨강인데 z={stat[6]}")
        if 1 <= c <= 5 and stat[6] > 0:
            errs.append(f"{f}: 회색인데 z={stat[6]}")
    # 3) 팔레트 구조
    if len(palette) != 11:
        errs.append(f"팔레트 길이 {len(palette)} != 11")
    grays = palette[1:6]
    for g in grays:
        if not (g[1:3] == g[3:5] == g[5:7]):
            errs.append(f"warm 팔이 무채색이 아니다: {g}")
    # 5) 마이너스 정규화
    if "−" in html:
        errs.append("U+2212가 본문에 남아 있다")
    # 6) 판정 상태 값 범위
    for m in levels:
        bad = [s for s in per[m]["s"] if s not in (0, 1, 2)]
        if bad:
            errs.append(f"min_n={m}: 판정 상태 이상값 {set(bad)}")
    # 7) 용량
    if kb > SIZE_BUDGET_KB:
        errs.append(f"용량 {kb:.0f}KB > 예산 {SIZE_BUDGET_KB}KB")
    if errs:
        print("[검사] 실패 " + str(len(errs)) + "건")
        for e in errs[:20]:
            print("  - " + e)
        raise SystemExit(1)
    print(f"[검사] 통과 (외부요청 0 · 빨강=z>0 · warm 무채색 "
          f"· 용량 {kb:.0f}/{SIZE_BUDGET_KB}KB)")


if __name__ == "__main__":
    main()
