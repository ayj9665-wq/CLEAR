"""experiments/build_story_page_v2.py -- 채용 담당자 동선으로 재구성한 소개 페이지(ver2)

기존 페이지(`build_story_page.py` -> clear_story.html)는 연구 서사 순서로 짜여 있다.
문제 -> 접근 -> 구조 -> 예측 -> 진단 -> 처방 -> 지도. 읽는 사람이 분석을 끝까지
따라올 것을 전제한 순서다. ver2는 같은 내용을 **30초 / 2분 / 5분** 세 층으로 다시
쌓는다. 첫 화면에서 목적·결과·역할이 끝나고, 근거는 접어 두고, 지도는 여전히 결론이다.

## v1을 고치지 않는다 -- 값은 import 하고 템플릿만 새로 쓴다

이 저장소는 같은 판단을 이미 한 번 내렸다. `build_story_page`가 `build_web_map`의
load_blocks / build_paths / color_tables를 **import 해서** 쓰고 HTML 템플릿만 새로
쓴 것이 그것이다(저쪽 파일 첫 주석). 여기서도 규칙은 같다:

- **값은 한 곳에서만 나온다.** 절의 문장과 수치는 v1의 `problem_parts()` /
  `predict_parts()` / `diagnose_parts()` / `prescribe_parts()`가 돌려주는 것을 그대로
  쓴다. 도해는 `ARCH_SVG` / `ARCH_STEPS`, 지도 어휘는 `LABELS`, 경고문은 `NOTES` /
  `HOWTO`를 쓴다. 문자열을 복사해 오면 한쪽만 고쳐졌을 때 **두 페이지가 서로 다른
  숫자를 주장한다** -- 이 저장소가 결과 CSV 라벨에서 이미 겪은 실패다.
- **템플릿은 소유한다.** ver2는 절 구성도 상호작용도 다르므로, 하나의 템플릿을
  분기로 공유하면 조건문이 늘어 두 산출물이 같이 깨진다. 대신 v1은 손대지 않으므로
  이미 배포된 clear_story.html의 회귀 위험이 0이다.

v1에 가한 변경은 세 가지뿐이고 모두 하위호환이다: `section(sid=)`, `check(budget=)`,
`css()`를 `subst()`로 분리. 셋 다 v1 산출물을 **바이트 단위로 바꾸지 않는다.**

## 채용 담당자가 30초 안에 읽어야 하는 것

첫 화면은 연구 배경이 아니라 (1) 무엇을 만들었나 (2) 무엇이 나왔나 (3) 내가 무엇을
했나 셋이다. 그래서 `04 KEY FINDINGS`와 `08 CONTRIBUTION`이 새로 생겼다.

**핵심 결과 절은 표본 범위를 반드시 밝힌다.** v1의 진단 절은 전국 사다리(1.805 /
1.558 / 0.662) 바로 아래에 3개 주 표본 수치(1.48 / 0.67 / +0.82)를 라벨 없이 잇는다.
연구 서사에서는 문맥이 받쳐 주지만, 요약 카드로 압축하면 "직접 경로를 닫으면 정형
모델은 증폭을 멈춘다"로 읽히고 **전국 기준으로 그 문장은 거짓이다**(blind XGBoost
1.632). 그래서 핵심 결과 2번은 3개 주와 전국을 나눠 적고, 전국에서 주장 가능한 것은
짝지은 차이 +0.173뿐임을 명시한다.

## 지도에 더한 것과 더하지 않은 것

레이어 3종·최소 사건 수 슬라이더·표·호버 9개 항목은 v1 그대로다. 더한 것은
**클릭 상세 패널 / 지도<->표 양방향 연동 / 스크롤 진입 시 1회 안내 / 키보드 이동**
넷이다. 색은 하나도 더하지 않았다 -- 선택 상태는 흰 테두리로만 표시한다. 색은 이
페이지에서 "기대 대비 잔여량"이라는 뜻 하나를 갖고, 선택이라는 뜻을 겸하면 그 규칙이
무너진다. 패널 문장도 배수에서 기계적으로 유도한 것만 쓰고, 원인을 말하지 않는다.

출력: outputs[/{scope}]/web/clear_story_v2.html (외부 요청 0)
"""
import argparse
import json
import re

import config as C
from clear import counties as CT
from experiments import _fontpack as FP
from experiments import _htmlcheck as HC
from experiments import build_story_page as V1
from experiments.build_web_map import (
    VIEW_W, build_paths, color_tables, load_blocks,
)

T = V1.T                     # 글리프 예산은 v1의 G 한 벌을 그대로 쓴다
SIMPLIFY_KM = V1.SIMPLIFY_KM

# v1은 792KB / 예산 850KB. ver2는 같은 기하(1.5km 단순화 -- 나란히 놓고 비교되는
# 산출물끼리 기하가 다르면 데이터 차이로 읽힌다)에 서사와 패널이 더 붙으므로 순증한다.
# 늘어나는 쪽은 지오메트리가 아니라 텍스트이므로, 예산만 올리고 단순화는 건드리지
# 않는다.
SIZE_BUDGET_KB = 900


# ---------------------------------------------------------------------------
# JS가 조립하는 문자열. v1의 LABELS에 ver2에서 새로 생긴 것만 더한다.
# 한글 리터럴을 JS 안에 흩뿌리면 서브셋 등록에서 빠져 글자가 통째로 사라진다.
# ---------------------------------------------------------------------------

LABELS2 = {
    "p_close": "상세 닫기",
    "p_verdict_more": "기대보다 %s% 더 남았다",
    "p_verdict_less": "기대보다 %s% 적게 남았다",
    "p_verdict_none": "차이가 우연으로 설명되는 범위다",
    "p_obs": "실제 미제",
    "p_exp": "기대 미제",
    "p_cases": "사건 수",
    "p_smr": "배수(SMR)",
    "p_z": "확신도(z)",
    "p_q": "헛짚을 확률(q)",
    "p_black": "흑인 피해자 비중",
    "p_flag_more": "기대보다 많이 잔존",
    "p_flag_less": "기대보다 적게 잔존",
    "p_flag_none": "판정했으나 우연 수준",
    "p_table": "표에서 보기",
    "p_cau": "이 값은 사건 구성으로 설명되지 않는 잔여량이다. 원인(수사 자원·차별·"
             "기록 관행·사건 구성)은 이 설계로 분리되지 않으므로, 수치로 원인을 "
             "단정할 수 없다.",
    "p_hint": "지도나 표에서 카운티를 선택하면 상세가 여기에 표시된다.",
    "spot_cold": "기대보다 많이 남은 지역",
    "spot_warm": "기대보다 적게 남은 지역",
    "spot_none": "사건 수가 부족해 판정에서 제외한 지역",
    "spot_skip": "안내 건너뛰기",
    "arch_easy": "쉬운 설명",
    "arch_tech": "기술적 설명",
    "arch_eff": "결과에 미친 영향",
    "fold_open": "자세히 보기",
    "fold_close": "접기",
}


def register_labels():
    V1.register_labels()
    for v in LABELS2.values():
        T(v, 400)
        T(v, 700)
    T("+-", 700)                 # 접이식 버튼이 JS로 바꿔 다는 부호


# ---------------------------------------------------------------------------
# 03 ARCHITECTURE -- 도해는 v1 것을 그대로 쓰고, 단계 선택만 얹는다
#
# SVG 좌표는 v1의 ARCH_SVG(viewBox 0 0 960 300)를 그대로 쓴다. 강조 영역은 그 위에
# **테두리만 덧그리는 사각형**이라 도형 자체는 하나도 바뀌지 않는다. LOSS는 그려진
# 도형이 없는 단계(손실은 그림이 아니다)이므로, 손실이 실제로 누르는 대상인 출력
# 확률 쪽을 가리킨다.
# ---------------------------------------------------------------------------

ARCH_ZONES = [                        # (x, y, w, h) -- ARCH_SVG의 viewBox 좌표계
    (4, 44, 208, 226),                # INPUT  : 흩어진 노드
    (250, 46, 400, 218),              # GRAPH  : 점선 블록 두 개
    (700, 58, 250, 134),              # MODEL  : 이웃 -> 평균
    (758, 140, 124, 106),             # LOSS   : 평균 -> 검거 확률
]

ARCH_DETAIL = [                       # (쉬운 설명, 결과에 미친 영향)
    ("사건 하나를 하나의 데이터 점으로 바꾼다.",
     "가해자 항목을 넣으면 정답을 그대로 알려 주는 셈이라 전부 제외했다. 성능이 "
     "갑자기 좋아지면(AUC 0.99 부근) 이 항목이 되돌아온 신호로 본다."),
    ("비슷한 지역에서 발생한 사건들을 서로 연결한다.",
     "모델에 카운티 변수를 주지 않았으므로, 지역의 맥락은 오직 이 연결을 통해서만 "
     "들어온다. 뒤에서 인종 격차의 경로로 지목되는 것도 같은 연결이다."),
    ("주변 사건들의 정보를 모아 검거 여부를 예측한다.",
     "이웃을 평균으로 요약하기 때문에 그래프가 값을 갖는다. 가장 비슷한 이웃만 "
     "고르도록 바꿨더니 오히려 성능이 떨어졌다 -- 요약이 자기 자신의 복사본이 된다."),
    ("잘 맞히면서도 집단 간 격차가 벌어지지 않도록 학습한다.",
     "인종은 학습할 때만 필요하고 예측할 때는 필요 없다. 다만 벌점이 전국 일괄 "
     "격차로 정의되어 있어, 주 단위로 다시 재면 통과 기준을 넘지 못한다."),
]


def build_arch2():
    """v1의 도해 + 단계 선택. 값(SVG·설명문)은 전부 v1에서 가져온다."""
    zones = "".join(
        f'<rect class="azone" data-zone="{i}" x="{x}" y="{y}" width="{w}" '
        f'height="{h}" rx="10"/>'
        for i, (x, y, w, h) in enumerate(ARCH_ZONES))
    svg = V1.ARCH_SVG.format(
        mut=V1.MUTED, ink=V1.INK, blk=V1.ARCH_BLOCK, edg=V1.ARCH_EDGE,
        aria=T("사건을 노드로 두고 같은 주·도시끼리 블록으로 묶어 연결한 뒤, 각 "
               "노드가 이웃의 평균을 요약해 검거 확률을 예측하는 과정을 나타낸 도해"),
        t0=T("사건 하나가 노드", 700), t1=T("같은 주·도시끼리 블록", 700),
        t2=T("이웃의 평균으로 예측", 700), t3=T("사건당 특성 123개"),
        t4=T("이웃 최대 20개 · 엣지 2,500만"), t5=T("GraphSAGE 2층 · 집계 평균"),
        t6=T("평균"), t7=T("검거 확률"))
    svg = svg.replace("</svg>", zones + "</svg>")

    tabs = "".join(
        f'<button class="atab" data-arch="{i}" role="tab" '
        f'aria-selected="{"true" if i == 0 else "false"}" '
        f'aria-controls="archexp">'
        f'<span class="sn">{T(f"0{i+1}", 800)}</span>'
        f'<span class="se">{T(en, 700)}</span>'
        f'<b>{T(ko, 700)}</b></button>'
        for i, (en, ko, _) in enumerate(V1.ARCH_STEPS))

    panes = ""
    for i, (en, ko, tech) in enumerate(V1.ARCH_STEPS):
        easy, eff = ARCH_DETAIL[i]
        panes += (
            f'<div class="apane" data-arch="{i}"{"" if i == 0 else " hidden"}>'
            f'<p class="aeasy">{T(easy, 700)}</p>'
            f'<p class="atech"><b>{T(LABELS2["arch_tech"], 700)}</b> {T(tech)}</p>'
            f'<p class="aeff"><b>{T(LABELS2["arch_eff"], 700)}</b> {T(eff)}</p>'
            f'</div>')

    return V1.section(
        "03", "ARCHITECTURE",
        "사건을 그래프로, 그래프를 예측으로",
        "예측 단계의 구조는 세 부분이다. 사건 하나를 노드로 두고, 같은 지역에서 "
        "발생한 사건끼리 이어 블록을 만들고, 각 노드가 이웃의 평균을 요약해 검거 "
        "여부를 예측한다. 아래 네 단계를 눌러 각 단계가 무엇을 하는지 볼 수 있다.",
        f'<div class="arch">{svg}</div>'
        + f'<p class="acap">{T("원과 사각형은 서로 다른 주·도시를 뜻한다. 블록이 "
                               "다르면 이어지지 않으며, 뒤에서 인종 격차의 경로로 "
                               "지목되는 것이 바로 이 구조다.")}</p>'
        + f'<div class="atabs" role="tablist" '
          f'aria-label="{T("예측 단계 네 가지")}">{tabs}</div>'
        + f'<div class="archexp" id="archexp">{panes}</div>'
        + f'<p class="note">{T("출력은 확률이 아니라 순위 점수다. 미제 쪽에 가중을 "
                               "주고 학습하므로 전체 수준이 밀려 있어, 지도 단계에서는 "
                               "기댓값 총합이 실제와 맞도록 상수 하나를 더한 뒤 "
                               "사용한다. 순서를 바꾸지 않는 보정이므로 순위는 "
                               "그대로다.")}</p>',
        "arch")


# ---------------------------------------------------------------------------
# 04 KEY FINDINGS -- 한 줄 결론 / 핵심 수치 / 내가 한 분석 / 한계
#
# 수치는 전부 reports/ 와 결과 CSV에 있는 값이다. 반올림하거나 새로 계산하지 않는다.
# 2번 항목이 3개 주와 전국을 나눠 적는 이유는 이 파일 첫 주석에 있다.
# ---------------------------------------------------------------------------

def findings():
    """핵심 결과 세 개. **수치는 전부 유도한다** -- 같은 페이지에 근거가 함께 있으므로,
    요약이 표나 사다리와 다른 숫자를 말하면 그 자리에서 자기모순이 된다."""
    g, x, _lr = V1.PREDICT_ROWS                 # GraphSAGE / XGBoost / LogReg
    d = [g[1][k] - x[1][k] for k in range(4)]   # MCC / AUC / BalAcc / Precision
    ratio = dict((m, r) for m, r, _ in V1.NOISE_RATIOS)

    r_pool = V1.fair("fairness_gaps.csv", "Victim Race")
    r_state = V1.fair("fairness_gaps_standardized.csv", "Victim Race")
    xb = V1.fair("fairness_gaps.csv", "Victim Race", model="xgboost_blind")
    pair = V1.contrast("fairness_model_contrasts.csv", "Victim Race")
    pair_s = V1.contrast("fairness_model_contrasts_standardized.csv", "Victim Race")

    grid = [(0, "graphsage_fairloss_a0_mb"), (25, "graphsage_fairloss_a25_mb"),
            (50, "graphsage_fairloss_a50_mb"), (100, "graphsage_fairloss_a100_mb")]
    pool = {a: V1.fair("fairness_gaps.csv", "Victim Race", metric="fpr", model=m)
            for a, m in grid}
    st = {a: V1.fair("fairness_gaps_standardized.csv", "Victim Race",
                     metric="fpr", model=m) for a, m in grid}

    return [
        {
            "tag": "PREDICT",
            "head": "그래프 모델이 정형 모델을 앞선다 - 다만 주장할 수 있는 지표는 "
                    "일곱 중 셋이다.",
            "nums": [(f"MCC +{d[0]:.4f}", f"실행 간 변동의 {ratio['MCC']:.1f}배"),
                     (f"AUC +{d[1]:.4f}", f"{ratio['AUC']:.1f}배"),
                     (f"Precision +{d[3]:.4f}",
                      f"{ratio['Precision']:.1f}배 - 주장 불가")],
            "mine": "일곱 지표를 다 재고, 각 지표의 앞선 폭을 그 지표 자신의 시드 간 "
                    "표준편차로 나눠 판단 기준을 만들었다. Precision은 Balanced "
                    f"Accuracy의 +{d[2]:.4f}와 비슷한 폭으로 앞섰지만 변동이 5배 커서 "
                    "주장에서 제외했다. 어떤 지표로 말할 수 있는지를 취향이 아니라 "
                    "측정으로 정한 것이 이 절의 요지다.",
            "limit": "임계값 0.5에 의존하는 네 지표(F1·Precision·Sensitivity·"
                     "Specificity)는 이 표본에서 우연 범위를 벗어나지 못한다. "
                     "정확도 우위는 MCC·AUC·Balanced Accuracy로만 주장한다.",
        },
        {
            "tag": "DIAGNOSE",
            "head": "성별 격차는 변수에서 오고, 인종 격차에는 그래프가 따로 기여한다.",
            "nums": [("성별 2.55~3.29배 -> 0.88~0.96배", "변수 제거만으로 소멸 "
                      "(3개 주 표본)"),
                     ("3개 주 표본: 그래프 1.48 vs 정형 0.67",
                      "짝지은 차이 +0.82 [+0.65, +1.03]"),
                     (f"전국: 그래프 {r_pool[0]:.3f} vs 정형 {xb[0]:.3f}",
                      f"짝지은 차이 +{pair[0]:.3f} "
                      f"[+{pair[1]:.3f}, +{pair[2]:.3f}]")],
            "mine": "인종·성별 변수를 지운 조건에서 같은 모델을 '같은 지역' 그래프와 "
                    "'같은 시기' 그래프에 각각 올려, 모델·특성·학습 절차를 고정하고 "
                    "연결 방식만 바꾼 대조를 만들었다. 신뢰구간은 같은 검정 집합을 "
                    "공유하므로 모델별로 따로 내지 않고 복원추출 표본 안에서 짝지어 "
                    "계산했다 - 구간이 겹치는데도 짝지은 차이는 0을 배제하는 경우가 "
                    "실제로 나온다.",
            "limit": "표본 범위를 반드시 붙여야 한다. 3개 주에서는 변수를 지우면 정형 "
                     "모델의 증폭이 사라지지만(0.67), <b>전국에서는 그렇지 않다</b>"
                     f"({xb[0]:.3f}). 전국 규모에서 그래프는 유일한 경로가 아니라 이미 "
                     "강한 특성 프록시 위에 얹히는 <b>증분</b>이며, 주장 가능한 것은 "
                     f"짝지은 차이 +{pair[0]:.3f}(주 단위 표준화 +{pair_s[0]:.3f})뿐이다.",
        },
        {
            "tag": "PRESCRIBE",
            "head": "완화는 작동한다. 다만 성공 여부가 격차를 어느 단위에서 재느냐에 "
                    "달려 있다.",
            "nums": [(f"전국 일괄: {pool[0][0]:.3f} -> {pool[100][0]:.3f} "
                      f"[{pool[100][1]:.3f}, {pool[100][2]:.3f}]",
                      "정확도 0.0156 지불"),
                     (f"주 단위 표준화: {st[100][0]:.3f} "
                      f"[{st[100][1]:.3f}, {st[100][2]:.3f}]",
                      "어떤 벌점도 기준 미통과"),
                     (f"벌점 25/50/100 -> {st[25][0]:.3f}/{st[50][0]:.3f}/"
                      f"{st[100][0]:.3f}", "단조가 아니다")],
            "mine": "통과 기준을 실험 전에 정해 두고(구간 상한 0.5 이하) 벌점을 훑었다. "
                    "일괄 기준으로는 벌점 100이 통과하지만, 같은 모델을 주 단위로 "
                    "표준화해 다시 재면 통과하지 못하고 강도를 올려도 개선되지 않는다. "
                    "벌점 자체가 일괄 격차로 정의되어 있기 때문이라는 것이 결론이며, "
                    "필요한 것은 재조정이 아니라 층화된 벌점이다.",
            "limit": "층화 벌점은 아직 만들지 않았다. 지도의 기댓값은 통과 기준을 "
                     "일괄로 적용한 모델에서 나오므로, 카운티 잔차의 인종 상관은 "
                     "<b>하한</b>으로 읽어야 한다.",
        },
    ]


def build_findings():
    cards = ""
    for f in findings():
        nums = "".join(
            f'<div class="fnum"><b>{T(a, 700)}</b><span>{T(b)}</span></div>'
            for a, b in f["nums"])
        cards += (
            f'<article class="find">'
            f'<span class="se">{T(f["tag"], 700)}</span>'
            f'<h3>{T(f["head"], 700)}</h3>'
            f'<div class="fnums">{nums}</div>'
            f'<p class="fmine"><b>{T("내가 한 분석", 700)}</b> {T(f["mine"])}</p>'
            f'<p class="flim"><b>{T("해석의 한계", 700)}</b> {T(f["limit"])}</p>'
            f'</article>')
    return V1.section(
        "04", "KEY FINDINGS",
        "세 가지 결과, 그리고 각각을 어디까지 주장할 수 있는가",
        "이 프로젝트에서 나온 결과는 세 줄로 요약된다. 각 결과에는 그것을 "
        "뒷받침하는 수치와 함께, 그 수치로 <b>말할 수 없는 것</b>을 붙였다. "
        "말할 수 없는 범위를 함께 적는 것이 이 프로젝트의 방법이다.",
        f'<div class="finds">{cards}</div>', "findings")


# ---------------------------------------------------------------------------
# 08 CONTRIBUTION
# ---------------------------------------------------------------------------

CONTRIB = [
    ("문제 정의와 연구 설계",
     "'검거를 얼마나 잘 맞히나'에서 멈추지 않고 예측 → 진단 → 처방 → 적용 네 "
     "단계로 문제를 재정의했다. 각 단계는 앞 단계가 답하지 못한 것 때문에 "
     "존재하며, 개입마다 계획서를 먼저 쓰고 되돌릴 지점을 표시한 뒤 실행했다."),
    ("데이터 정제와 피처 엔지니어링",
     "63만 건을 정제하고 범주형 항목을 펼쳐 사건당 특성 123개를 만들었다. "
     "가해자 항목이 미제 사건에서 90~99% '미상'이라는 점을 확인하고 타깃 누수로 "
     "규정해 전부 제외했다 - 이 데이터셋에서 가장 중요한 제약이다."),
    ("그래프 설계와 엣지 생성",
     "지역·시기·수법 세 가지 후보로 블록을 만들고 블록 안에서만 이웃을 잇되, "
     "실행마다 같은 그래프가 나오도록 고정했다. 전국 기준 엣지 2,500만 개다. "
     "세 후보를 성능과 인종 동질성 양쪽에서 비교해 지역 블록을 기본으로 정했다."),
    ("GraphSAGE 학습과 규모 확장",
     "전국 규모에서는 전체 그래프를 한 번에 올릴 수 없어 이웃 표집 학습으로 "
     "바꿨다. 다만 예측할 때는 이웃 전체를 쓴다 - 예측값이 공정성 지표와 지도의 "
     "색으로 이어지므로 표집 잡음을 남기면 안 된다. 표집 학습과 전체 학습을 "
     "잇는 대조 실험을 따로 돌려, 정확도는 이어지지만 임계값 의존 지표는 "
     "이어지지 않음을 확인했다."),
    ("공정성 지표와 신뢰구간",
     "집단별 선택률·재현율·오탐률 격차를 원자료의 격차로 나눈 증폭비로 진단했다. "
     "신뢰구간은 네 칸(TN/FP/FN/TP) 다항분포 복원추출로 계산하고, 모델 비교는 "
     "같은 표본을 공유하므로 16칸 결합분포에서 짝지어 뽑았다. 모델별 구간의 "
     "겹침으로 비교하는 것은 이 설계에서 무효다."),
    ("완화 실험과 실패한 접근의 규명",
     "동질 엣지를 끊는 방식은 실패했고, 실패의 원인을 측정해 격차가 엣지가 아니라 "
     "블록 소속에서 온다는 것을 특정했다. 그 결과가 손실 함수에 벌점을 넣는 방식 "
     "으로 이어졌다. 실패한 개입을 지우지 않고 남긴 이유는 그것이 원인을 지목한 "
     "근거이기 때문이다."),
    ("카운티 단위 잔차 분석",
     "지역별로 기대 미제와 실제 미제를 견주는 간접 표준화를 적용하고, 다중 검정 "
     "보정을 걸었다. 예측값이 확률이 아니라 순위 점수라는 점을 발견해 로짓 상수 "
     "하나로 총합을 맞췄다 - 이 보정 없이는 검정이 지역 이상이 아니라 전역 "
     "보정 오차를 재게 된다."),
    ("외부 자료 결합으로 대안 설명 검증",
     "경찰 인력, 사건 정황, 다른 집계 양식 세 가지를 기관 코드로 결합해 "
     "'자원 부족 아닌가', '사건 구성 차이 아닌가', '집계 오류 아닌가'를 각각 "
     "검증했다. 셋 다 결론을 뒤집지 못했고, 그 과정에서 정황 변수는 통제하면 "
     "오히려 편향을 만든다는 것을 밝혔다."),
    ("자체 완결형 웹 산출물과 검증 파이프라인",
     "외부 요청이 하나도 없는 단일 HTML로 지도와 소개 페이지를 만들었다. 차트 "
     "라이브러리를 쓰지 않고 투영과 단순화를 파이썬에서 끝냈다. 빌드가 스스로 "
     "외부 요청·색 규칙·용량을 검사하고, 별도로 헤드리스 DOM 검사가 조작까지 "
     "재현해 화면의 숫자를 CSV와 대조한다. 검사기가 실제 결함을 잡는지도 "
     "결함을 일부러 심어 확인한다."),
]


def build_contrib():
    items = "".join(
        f'<li><b>{T(h, 700)}</b><p>{T(b)}</p></li>' for h, b in CONTRIB)
    return V1.section(
        "08", "CONTRIBUTION",
        "이 프로젝트에서 내가 한 일",
        "혼자 수행한 프로젝트다. 아래는 사용한 기술의 목록이 아니라, 각 단계에서 "
        "무엇을 만들고 어떤 판단을 내렸는지다.",
        f'<ol class="contrib">{items}</ol>', "contrib")


# ---------------------------------------------------------------------------
# 10 LIMITS -- v1의 NOTES 12개를 세 묶음으로 재배치한다
#
# 문장은 하나도 바꾸지 않는다. 바뀌는 것은 순서와 묶음뿐이고, 12개 전부가 정확히
# 한 번씩 쓰였는지는 빌드가 검사한다(아래 check). "지도를 읽을 때"에 해당하는 두
# 개(축척·작은 면적)는 지도 절에 남겨 그림 옆에서 읽히게 한다.
# ---------------------------------------------------------------------------

LIMIT_GROUPS = [
    ("이 분석이 말할 수 있는 것", [5, 7, 6, 3]),
    ("이 분석이 말할 수 없는 것", [0, 1, 2, 8, 9]),
    ("다음에 확장할 수 있는 것", [4]),
]
MAP_NOTE_IDX = [10, 11]               # 지도 절에 남기는 '읽는 법' 항목

NEXT_STEPS = [
    ("벌점을 주 단위로 층화하는 것.",
     "지금 벌점은 전국을 일괄로 계산한 격차에 걸려 있어, 주 단위로 다시 재면 "
     "통과 기준을 넘지 못하고 강도를 올려도 개선되지 않는다. 강도의 문제가 "
     "아니라 정의의 문제이므로, 층화한 벌점을 새 개입으로 설계해야 한다."),
]


def build_limits():
    groups = ""
    for title, idx in LIMIT_GROUPS:
        items = "".join(
            f'<li><b>{T(V1.NOTES[i][0], 700)}</b> {T(V1.NOTES[i][1])}</li>'
            for i in idx)
        if title.startswith("다음"):
            items += "".join(f'<li><b>{T(h, 700)}</b> {T(b)}</li>'
                             for h, b in NEXT_STEPS)
        groups += (f'<div class="lgrp"><h3>{T(title, 800)}</h3>'
                   f'<ul>{items}</ul></div>')
    return V1.section(
        "10", "LIMITS",
        "어디까지 말할 수 있고, 어디부터 말할 수 없는가",
        "아래 문장들은 결과를 축소하기 위한 것이 아니라 결과의 범위를 정하기 위한 "
        "것이다. 특히 두 번째 묶음이 빠지면 이 프로젝트의 지도는 "
        "'붉은 지역 = 경찰이 일을 안 한다'로 읽힌다.",
        f'<div class="lims">{groups}</div>', "limits")


# ---------------------------------------------------------------------------
# 11 STACK
# ---------------------------------------------------------------------------

STACK = [
    ("모델·데이터", "Python · pandas · NumPy · PyTorch Geometric · GraphSAGE · "
                    "scikit-learn"),
    ("산출물", "HTML · CSS · JavaScript · SVG 기반 지도(차트 라이브러리 없음)"),
    ("검증", "pytest · node --check · jsdom DOM 검사 · CSV 교차대조 · "
             "clone-and-run 계약"),
]

REPRO = [
    "export CLEAR_SCOPE=national",
    "python -m experiments.crossfit_predictions --minibatch",
    "python -m experiments.detect_cold_blocks --out cold_blocks_cv5.csv",
    "python -m experiments.build_story_page_v2",
    "python -m experiments.verify_html",
]


def build_stack():
    rows = "".join(
        f'<div class="strow"><b>{T(h, 700)}</b><span>{T(b)}</span></div>'
        for h, b in STACK)
    cmds = "<br>".join(T(c) for c in REPRO)
    return V1.section(
        "11", "STACK",
        "사용한 기술과 재현 방법",
        "학습에는 GPU가 필요하지만, 이 페이지와 대시보드는 커밋된 결과 CSV만 읽어 "
        "pandas와 NumPy만으로 다시 만들어진다. 저장소를 받아 그대로 실행되는지를 "
        "확인하는 명령이 따로 있다.",
        f'<div class="stack">{rows}</div>'
        + fold(f'<code class="repro">{cmds}</code>', "repro"),
        "stack")


# ---------------------------------------------------------------------------
# 공통 조각
# ---------------------------------------------------------------------------

def fold(inner, fid):
    """접이식 상세. <details>가 아니라 버튼 + hidden 인 이유는 aria-expanded를
    직접 제어하고 헤드리스 검사에서 상태를 관측하기 위해서다."""
    return (f'<div class="fold">'
            f'<button class="foldb" data-fold="{fid}" aria-expanded="false" '
            f'aria-controls="fold-{fid}">'
            f'<span class="ftxt">{T(LABELS2["fold_open"], 700)}</span>'
            f'<i aria-hidden="true">+</i></button>'
            f'<div class="foldc" id="fold-{fid}" hidden>{inner}</div></div>')


def wrap_fold(fid):
    return lambda inner: fold(inner, fid)


# ---------------------------------------------------------------------------
# CSS -- v1의 전체 CSS를 그대로 싣고 ver2 전용 규칙만 덧붙인다
# ---------------------------------------------------------------------------

CSS2 = """
/* ---- 00 hero ------------------------------------------------------- */
#hero2{min-height:100svh;display:flex;flex-direction:column;justify-content:center;
  padding:80px 0 60px}
.hlab2{font-weight:800;font-size:11px;letter-spacing:.22em;color:__MUT__;margin:0}
#hero2 h1{font-weight:800;font-size:clamp(46px,9vw,104px);letter-spacing:-.045em;
  line-height:.98;margin:18px 0 0}
.hone{font-weight:700;font-size:clamp(19px,2.6vw,28px);line-height:1.5;
  letter-spacing:-.02em;margin:20px 0 0;max-width:24ch}
.hkpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:22px;margin:38px 0 0;max-width:760px}
.hkpi div b{display:block;font-weight:800;font-size:clamp(24px,3.4vw,34px);
  letter-spacing:-.03em;line-height:1.15}
.hkpi div span{display:block;color:__INK2__;font-size:13px;margin-top:6px}
.hrole{margin:34px 0 0;padding:16px 18px;border:1px solid __HAIR__;border-radius:12px;
  max-width:760px;background:#ffffff05}
.hrole b{font-weight:800;font-size:11px;letter-spacing:.18em;color:__MUT__;
  display:block;margin-bottom:6px}
.hrole p{margin:0;color:__INK2__;font-size:14.5px;line-height:1.7}
.cta{display:flex;gap:12px;flex-wrap:wrap;margin:30px 0 0}
.cta a{display:inline-block;padding:13px 22px;border-radius:999px;font-weight:700;
  font-size:14px;text-decoration:none;border:1px solid __INK__;color:__BG__;
  background:__INK__}
.cta a.ghost{background:transparent;color:__INK__;border-color:#3a3a40}
.cta a:focus-visible,.foldb:focus-visible,.atab:focus-visible{outline:2px solid __INK__;
  outline-offset:3px}
/* ---- 접이식 상세 ---------------------------------------------------- */
.fold{margin:26px 0 0}
.foldb{display:inline-flex;align-items:center;gap:8px;background:none;cursor:pointer;
  border:1px solid #3a3a40;border-radius:999px;padding:8px 16px;color:__INK2__;
  font:inherit;font-weight:700;font-size:13px}
.foldb:hover{border-color:#55555c;color:__INK__}
.foldb i{font-style:normal;font-size:15px;line-height:1}
.foldc{margin-top:18px;border-left:2px solid __HAIR__;padding-left:20px}
.foldc>*:first-child{margin-top:0}
/* ---- 03 architecture 상호작용 --------------------------------------- */
.azone{fill:none;stroke:none;stroke-width:1.6}
.azone.on{stroke:__INK__;stroke-dasharray:none}
.atabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px;margin:22px 0 0}
.atab{text-align:left;background:#ffffff04;border:1px solid __HAIR__;border-radius:12px;
  padding:14px 16px;color:__INK__;font:inherit;cursor:pointer}
.atab:hover{border-color:#3a3a40}
.atab[aria-selected="true"]{border-color:__INK__;background:#ffffff0d}
.atab b{display:block;font-weight:700;font-size:16px;margin-top:2px}
.archexp{margin:18px 0 0;border:1px solid __HAIR__;border-radius:12px;padding:20px}
.aeasy{margin:0;font-weight:700;font-size:17px;line-height:1.6}
.atech,.aeff{margin:12px 0 0;color:__INK2__;font-size:14px;line-height:1.7}
.atech b,.aeff b{color:__MUT__;font-size:11px;letter-spacing:.14em;margin-right:8px}
/* ---- 04 key findings ------------------------------------------------ */
.finds{display:grid;gap:20px}
.find{border:1px solid __HAIR__;border-radius:14px;padding:24px 22px;
  background:#ffffff04}
.find h3{font-weight:700;font-size:clamp(18px,2.2vw,22px);line-height:1.5;
  letter-spacing:-.02em;margin:8px 0 0}
.fnums{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:14px;margin:20px 0 0;padding:16px 0;border-top:1px solid __HAIR__;
  border-bottom:1px solid __HAIR__}
.fnum b{display:block;font-weight:700;font-size:15px;font-variant-numeric:tabular-nums}
.fnum span{display:block;color:__MUT__;font-size:12.5px;margin-top:3px}
.fmine,.flim{margin:16px 0 0;color:__INK2__;font-size:14.5px;line-height:1.75}
.fmine b,.flim b{display:block;color:__MUT__;font-size:11px;letter-spacing:.14em;
  margin-bottom:4px}
.flim{color:__INK2__}
/* ---- 02 approach 카드를 링크로 -------------------------------------- */
a.step{display:block;text-decoration:none;color:inherit}
a.step:hover{border-color:#3a3a40}
a.step:focus-visible{outline:2px solid __INK__;outline-offset:3px}
/* ---- 08 contribution ------------------------------------------------ */
ol.contrib{list-style:none;counter-reset:c;padding:0;margin:0;display:grid;gap:2px}
ol.contrib li{counter-increment:c;padding:20px 0;border-top:1px solid __HAIR__;
  display:grid;grid-template-columns:44px 1fr;gap:0 18px}
ol.contrib li::before{content:counter(c,decimal-leading-zero);font-weight:800;
  font-size:12px;color:__MUT__;letter-spacing:.08em;padding-top:5px}
ol.contrib li b{font-weight:700;font-size:17px}
ol.contrib li p{grid-column:2;margin:6px 0 0;color:__INK2__;font-size:14.5px;
  line-height:1.75}
/* ---- 10 limits ------------------------------------------------------ */
.lims{display:grid;gap:26px}
.lgrp h3{font-weight:800;font-size:13px;letter-spacing:.1em;color:__MUT__;
  margin:0 0 12px;padding-bottom:10px;border-bottom:1px solid __HAIR__}
.lgrp ul{margin:0;padding:0;list-style:none;display:grid;gap:16px}
.lgrp li{color:__INK2__;font-size:14.5px;line-height:1.75}
.lgrp li b{display:block;color:__INK__;font-weight:700;margin-bottom:3px}
/* ---- 11 stack ------------------------------------------------------- */
.stack{display:grid;gap:0}
.strow{display:grid;grid-template-columns:130px 1fr;gap:18px;padding:14px 0;
  border-top:1px solid __HAIR__;font-size:14.5px}
.strow b{font-weight:700;color:__MUT__;font-size:12px;letter-spacing:.08em;
  padding-top:3px}
.strow span{color:__INK2__}
code.repro{display:block;font-size:12.5px;line-height:2;color:__INK2__;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
/* ---- 09 지도: 상세 패널과 표 연동 ------------------------------------ */
.maprow{display:grid;grid-template-columns:1fr;gap:18px}
@media(min-width:1080px){.maprow{grid-template-columns:1fr 320px}}
.panel{border:1px solid #e1e0d9;border-radius:12px;padding:18px 18px 16px;
  background:#fff;align-self:start;position:relative;font-size:13.5px}
.panel h4{margin:0 22px 0 0;font-weight:800;font-size:18px;letter-spacing:-.02em;
  color:#141414}
.pflag{margin:8px 0 0;font-weight:700;font-size:12px;display:inline-block;
  padding:3px 10px;border-radius:999px;border:1px solid #d8d7cf;color:#4a4a45}
.pverdict{margin:14px 0 0;font-weight:700;font-size:15px;line-height:1.5;
  color:#141414}
.pbars{margin:14px 0 0;display:grid;gap:8px}
.pbrow{display:grid;grid-template-columns:58px 1fr 54px;gap:8px;align-items:center;
  font-size:12px;color:#54534d}
.pbtrack{height:9px;background:#eceae2;border-radius:5px;overflow:hidden}
.pbtrack i{display:block;height:100%;border-radius:5px}
.pbtrack i.obs{background:#141414}
.pbtrack i.exp{background:#b9b7ac}
.pbval{text-align:right;font-weight:700;color:#141414;font-variant-numeric:tabular-nums}
dl.pstats{margin:16px 0 0;display:grid;grid-template-columns:1fr auto;gap:7px 12px;
  font-size:12.5px}
dl.pstats dt{color:#6d6c65;margin:0}
dl.pstats dd{margin:0;text-align:right;font-weight:700;color:#141414;
  font-variant-numeric:tabular-nums}
.pcau{margin:16px 0 0;padding-top:12px;border-top:1px solid #eceae2;color:#6d6c65;
  font-size:11.5px;line-height:1.65}
.phint{color:#8a8981;font-size:12.5px;margin:0}
.pclose,.ptable{background:none;border:1px solid #d8d7cf;border-radius:8px;
  cursor:pointer;font:inherit;color:#333;padding:5px 10px;font-size:12px}
.pclose{position:absolute;top:12px;right:12px;padding:2px 8px;font-size:15px;
  line-height:1.3}
.ptable{margin-top:14px;font-weight:700}
.pclose:focus-visible,.ptable:focus-visible,.ct tbody tr:focus-visible{outline:2px
  solid #141414;outline-offset:2px}
/* 선택 표시는 색이 아니라 테두리다. 색은 '기대 대비 잔여량' 한 가지 뜻만 갖는다. */
#sel path,#spt path{fill:none;stroke:#141414;stroke-width:2;
  vector-effect:non-scaling-stroke;pointer-events:none}
#spt path{stroke-dasharray:5 4}
.spotcap{margin:10px 0 0;font-weight:700;font-size:13px;color:#333;min-height:20px}
.spotcap button{margin-left:10px;font-weight:400;font-size:12px;background:none;
  border:0;color:#6d6c65;cursor:pointer;text-decoration:underline;font-family:inherit}
.ct tbody tr{cursor:pointer}
.ct tbody tr.hl{background:#f2f1ea}
.ct tbody tr.sel{box-shadow:inset 2px 0 0 #141414}
@media(hover:none){#tip{display:none}}
@media(prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  .sec,#mapsec,footer{opacity:1!important;transform:none!important}
}
"""


# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

JS2 = r"""
document.documentElement.classList.add('js');

/* ---------------- 절 드러내기 ---------------- */
(function(){
var io=new IntersectionObserver(function(es){
  es.forEach(function(e){
    if(!e.isIntersecting) return;
    e.target.classList.add('in');
    io.unobserve(e.target);
  });
},{threshold:0.08});
document.querySelectorAll('.sec,#mapsec,footer').forEach(function(s){io.observe(s);});
})();

/* ---------------- 접이식 상세 ---------------- */
(function(){
document.querySelectorAll('.foldb').forEach(function(b){
  b.addEventListener('click',function(){
    var c=document.getElementById('fold-'+b.dataset.fold),
        open=b.getAttribute('aria-expanded')==='true';
    c.hidden=open;
    b.setAttribute('aria-expanded', String(!open));
    b.querySelector('.ftxt').textContent = open ? L2.fold_open : L2.fold_close;
    b.querySelector('i').textContent = open ? '+' : '-';
  });
});
})();

/* ---------------- 03 아키텍처 단계 선택 ---------------- */
(function(){
var tabs=[].slice.call(document.querySelectorAll('.atab')),
    panes=[].slice.call(document.querySelectorAll('.apane')),
    zones=[].slice.call(document.querySelectorAll('.azone'));
if(!tabs.length) return;
function pick(i){
  tabs.forEach(function(t,k){ t.setAttribute('aria-selected', String(k===i)); });
  panes.forEach(function(p,k){ p.hidden = k!==i; });
  zones.forEach(function(z,k){ z.classList.toggle('on', k===i); });
}
tabs.forEach(function(t,i){
  t.addEventListener('click',function(){ pick(i); });
  t.addEventListener('keydown',function(e){
    var d = e.key==='ArrowRight' ? 1 : e.key==='ArrowLeft' ? -1 : 0;
    if(!d) return;
    e.preventDefault();
    var n=(i+d+tabs.length)%tabs.length;
    tabs[n].focus(); pick(n);
  });
});
pick(0);
})();

/* ---------------- 09 지도 ---------------- */
(function(){
var g=document.getElementById('g'), hl=document.getElementById('hl'),
    sel=document.getElementById('sel'), spt=document.getElementById('spt'),
    tip=document.getElementById('tip'), panel=document.getElementById('panel'),
    cap=document.getElementById('spotcap'), tw=document.getElementById('tablewrap'),
    tb=document.querySelector('#ct tbody');
var layer='res', li=0, hoverIdx=-1, selIdx=-1, cursor=-1;
var nodes=FI.map(function(f){
  var p=document.createElementNS('http://www.w3.org/2000/svg','path');
  p.setAttribute('d',PATHS[f]); g.appendChild(p); return p;
});
function st(i){ return PL[LV[li]].s[i]; }
function outline(host,i){
  host.innerHTML='';
  if(i<0) return;
  var c=nodes[i].cloneNode(false);
  c.removeAttribute('class'); c.removeAttribute('fill');
  host.appendChild(c);
}
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
  if(selIdx>=0 && !st(selIdx)) select(-1); else if(selIdx>=0) select(selIdx);
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
  tb.innerHTML=rows.map(function(r){
    var t=r[1];
    return '<tr data-i="'+r[0]+'"><td>'+t[0]+'</td><td>'+t[1]+'</td><td>'+t[2]
      +'</td><td>'+t[3]+'</td><td>'+t[4]+'</td><td>'+t[5]+'</td><td>'+t[6]
      +'</td><td>'+r[2]+'</td><td>'+t[7]+'</td></tr>';
  }).join('');
  cursor=-1;
  var first=tb.firstElementChild;
  if(first) first.tabIndex=0;
}
function rowOf(i){ return tb.querySelector('tr[data-i="'+i+'"]'); }
/* 호버 확대: 원본은 그대로 두고 **맨 위 레이어에 복제본**을 올려 키운다.
   제자리에서 키우면 인접 카운티와 경계에 겹침/틈이 생겨 지도가 찢어져 보인다. */
function hover(i){
  if(i===hoverIdx) return;
  var prev=hoverIdx>=0 && rowOf(hoverIdx);
  if(prev) prev.classList.remove('hl');
  hoverIdx=i;
  hl.innerHTML='';
  if(i<0){ tip.style.opacity=0; return; }
  var c=nodes[i].cloneNode(false);
  c.setAttribute('fill', nodes[i].getAttribute('fill'));
  c.removeAttribute('class');
  hl.appendChild(c);
  /* scale(1)로 붙인 직후 scale(1.09)를 걸어야 transition이 돈다. */
  void c.getBoundingClientRect();
  c.classList.add('on');
  var r=rowOf(i);
  if(r) r.classList.add('hl');
}
function verdict(i){
  var t=S.stat[i], s=st(i);
  if(s===1) return L2.p_verdict_none;
  var pct=Math.round((t[5]-1)*100);
  return pct>=0 ? L2.p_verdict_more.replace('%s',pct)
                : L2.p_verdict_less.replace('%s',-pct);
}
function showTip(i,x,y){
  var t=S.stat[i], qq=PL[LV[li]].q[i];
  tip.innerHTML='<b>'+t[1]+', '+t[0]+'</b><br>'+verdict(i)+'<br>'+L.tip_cases+' '+t[2]
    +' / '+L.tip_unsolved+' '+t[3]+'<br>'+L.tip_expected+' '+t[4]+' / SMR '+t[5]
    +'<br>z '+t[6]+' / q '+qq+'<br>'+L.tip_black+' '+t[7];
  tip.style.opacity=1;
  tip.style.left=Math.min(x+14, innerWidth-300)+'px';
  tip.style.top=(y+14)+'px';
}
function bar(lab,val,max,cls){
  var w=Math.max(2,Math.min(100, max? val/max*100 : 0));
  return '<div class="pbrow"><span>'+lab+'</span><span class="pbtrack">'
    +'<i class="'+cls+'" style="width:'+w.toFixed(1)+'%"></i></span>'
    +'<span class="pbval">'+val+'</span></div>';
}
function select(i){
  selIdx=i;
  tb.querySelectorAll('tr.sel').forEach(function(r){ r.classList.remove('sel'); });
  outline(sel,i);
  if(i<0){
    panel.innerHTML='<p class="phint">'+L2.p_hint+'</p>';
    return;
  }
  var t=S.stat[i], qq=PL[LV[li]].q[i], s=st(i);
  var flag = s===1 ? L2.p_flag_none : (t[6]>0 ? L2.p_flag_more : L2.p_flag_less);
  var mx=Math.max(t[3],t[4]);
  panel.innerHTML=
    '<button class="pclose" id="pclose" aria-label="'+L2.p_close+'">×</button>'
    +'<h4>'+t[1]+', '+t[0]+'</h4>'
    +'<p class="pflag">'+flag+'</p>'
    +'<p class="pverdict">'+verdict(i)+'</p>'
    +'<div class="pbars">'+bar(L2.p_obs,t[3],mx,'obs')+bar(L2.p_exp,t[4],mx,'exp')
    +'</div>'
    +'<dl class="pstats">'
    +'<dt>'+L2.p_cases+'</dt><dd>'+t[2]+'</dd>'
    +'<dt>'+L2.p_smr+'</dt><dd>'+t[5]+'</dd>'
    +'<dt>'+L2.p_z+'</dt><dd>'+t[6]+'</dd>'
    +'<dt>'+L2.p_q+'</dt><dd>'+qq+'</dd>'
    +'<dt>'+L2.p_black+'</dt><dd>'+t[7]+'</dd></dl>'
    +'<p class="pcau">'+L2.p_cau+'</p>'
    +'<button class="ptable" id="ptable">'+L2.p_table+'</button>';
  document.getElementById('pclose').onclick=function(){ select(-1); };
  document.getElementById('ptable').onclick=function(){ toTable(i); };
  var r=rowOf(i);
  if(r){
    r.classList.add('sel');
    if(!tw.hidden) r.scrollIntoView({block:'nearest'});
  }
}
function toTable(i){
  if(tw.hidden) openTable(true);
  var r=rowOf(i);
  if(r){ r.scrollIntoView({block:'center'}); r.focus(); }
}
function openTable(open){
  tw.hidden=!open;
  document.getElementById('tbl').setAttribute('aria-pressed', String(open));
}
/* ---- 스크롤 진입 안내 : 1회만, 사용자가 만지면 즉시 중단 ---- */
var spotTimer=null, spotDone=false;
function stopSpot(){
  if(spotDone) return;
  spotDone=true;
  clearTimeout(spotTimer);
  spt.innerHTML=''; cap.textContent='';
}
function runSpot(){
  if(spotDone||!SPOT.length) return;
  var k=0;
  (function step(){
    if(spotDone||k>=SPOT.length){ stopSpot(); return; }
    var s=SPOT[k++];
    outline(spt,s[0]);
    cap.innerHTML=s[1]+'<button type="button" id="spotskip">'+L2.spot_skip+'</button>';
    document.getElementById('spotskip').onclick=stopSpot;
    spotTimer=setTimeout(step,2600);
  })();
}
/* ---- 입력 ---- */
g.addEventListener('pointermove',function(e){
  var i=nodes.indexOf(e.target);
  if(i<0||!st(i)){ hover(-1); return; }
  hover(i); showTip(i,e.clientX,e.clientY);
});
g.addEventListener('pointerleave',function(){ hover(-1); });
g.addEventListener('click',function(e){
  var i=nodes.indexOf(e.target);
  stopSpot();
  if(i<0||!st(i)){ select(-1); return; }
  select(i);
  if(!tw.hidden){ var r=rowOf(i); if(r) r.scrollIntoView({block:'center'}); }
});
tb.addEventListener('pointerover',function(e){
  var tr=e.target.closest('tr'); if(!tr) return;
  hover(+tr.dataset.i);
});
tb.addEventListener('pointerleave',function(){ hover(-1); });
tb.addEventListener('click',function(e){
  var tr=e.target.closest('tr'); if(!tr) return;
  stopSpot(); select(+tr.dataset.i);
});
/* 표는 행마다 tabindex를 박지 않고 하나만 이동시킨다(roving tabindex).
   1,800행에 속성을 박으면 탭 정지점이 1,800개가 되고 용량도 그만큼 늘어난다. */
tb.addEventListener('keydown',function(e){
  var rows=tb.children, tr=e.target.closest('tr');
  if(!tr) return;
  var i=[].indexOf.call(rows,tr), d=0;
  if(e.key==='ArrowDown') d=1;
  else if(e.key==='ArrowUp') d=-1;
  else if(e.key==='Home') d=-i;
  else if(e.key==='End') d=rows.length-1-i;
  else if(e.key==='Enter'||e.key===' '){
    e.preventDefault(); stopSpot(); select(+tr.dataset.i); return;
  } else return;
  e.preventDefault();
  var n=Math.max(0,Math.min(rows.length-1,i+d));
  tr.tabIndex=-1; rows[n].tabIndex=0; rows[n].focus();
  hover(+rows[n].dataset.i);
});
document.addEventListener('keydown',function(e){
  if(e.key==='Escape'&&selIdx>=0) select(-1);
});
document.querySelectorAll('[data-layer]').forEach(function(b){
  b.onclick=function(){
    stopSpot();
    layer=b.dataset.layer;
    document.querySelectorAll('[data-layer]').forEach(function(o){
      o.setAttribute('aria-pressed', String(o===b));
    });
    paint();
  };
});
document.getElementById('mn').oninput=function(e){ stopSpot(); li=+e.target.value; paint(); };
document.getElementById('tbl').onclick=function(){
  stopSpot(); openTable(tw.hidden);
};
paint(); select(-1);
if(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches){
  spotDone=true;
} else {
  var so=new IntersectionObserver(function(es){
    es.forEach(function(e){ if(e.isIntersecting){ so.disconnect(); runSpot(); } });
  },{threshold:0.25});
  so.observe(document.getElementById('mapsec'));
}
})();
"""


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------

def build_hero(n_counties):
    kpi = [
        (f"{638454:,}", "1980-2014년 미국 살인사건 기록"),
        (f"{n_counties:,}", "지도에 그린 카운티 (50개 주 + DC)"),
        ("GraphSAGE", "그래프 신경망 예측 + 공정성 완화"),
    ]
    kp = "".join(f'<div><b data-count="{v}">{T(v, 800)}</b>'
                 f'<span>{T(lab)}</span></div>' for v, lab in kpi)
    return (
        f'<section id="hero2"><div class="inner">'
        f'<p class="hlab2">{T("CLEARANCE LEARNING & EQUITY ASSESSMENT ON GRAPHS", 800)}'
        f'</p>'
        f'<h1>{T("CLEAR", 800)}</h1>'
        f'<p class="hone">{T("미제 사건의 격차를 예측하고, 그 격차가 어디에서 "
                             "발생하는지 진단한 프로젝트", 700)}</p>'
        f'<div class="hkpi">{kp}</div>'
        f'<div class="hrole"><b>{T("MY ROLE", 800)}</b>'
        f'<p>{T("데이터 파이프라인 · 그래프 설계 · 모델 학습 · 공정성 진단과 완화 · "
                "카운티 잔차 분석 · 자체 완결형 인터랙티브 지도 구현까지 "
                "혼자 수행했다.")}</p></div>'
        f'<div class="cta">'
        f'<a href="#arch">{T("프로젝트 구조 보기", 700)}</a>'
        f'<a class="ghost" href="#mapsec">{T("결과 지도 보기", 700)}</a>'
        f'</div></div></section>')


def build_approach(parts):
    """v1의 네 단계 카드를 해당 절로 가는 링크로 바꾼다. 문구는 v1 것 그대로."""
    hrefs = ["#s05", "#s06", "#s07", "#mapsec"]
    st = '<div class="steps">' + "".join(
        f'<a class="step" href="{hrefs[i]}"><span class="sn">{T(f"0{i+1}", 800)}</span>'
        f'<span class="se">{T(en, 700)}</span><b>{T(ko, 700)}</b>'
        f'<p>{T(d)}</p></a>'
        for i, (en, ko, d) in enumerate(V1.APPROACH_STEPS)) + "</div>"
    parts = dict(parts, core=st)
    return V1.render("02", parts, sid="approach")


def build_map_section(assessed, cold_n, warm_n, corr, levels, height):
    th = "".join(f"<th>{T(x, 700)}</th>" for x in V1.LABELS["th"])
    howto = '<div class="howto">' + "".join(
        f'<div class="hcard"><b>{T(q, 700)}</b><p>{T(a)}</p></div>'
        for q, a in V1.HOWTO) + "</div>"
    map_notes = "".join(
        f'<li><b>{T(V1.NOTES[i][0], 700)}</b> {T(V1.NOTES[i][1])}</li>'
        for i in MAP_NOTE_IDX)
    head = (
        f'<div class="shead"><span class="snum">{T("09", 800)}</span>'
        f'<span class="slab">{T("APPLY", 800)}</span></div>'
        f'<h2>{T("설명되지 않는 미제의 지리적 집중", 800)}</h2>'
        f'<p class="lead">{T("이 프로젝트가 최종적으로 제시하는 것은 사건 목록이 "
                             "아니라 지도다. 어느 지역에 설명되지 않는 미제가 집중되는지, "
                             "그리고 그 색을 어디까지 신뢰할 수 있는지를 함께 제시한다. "
                             "카운티를 누르면 오른쪽에 상세가 열리고, 표와 서로 "
                             "연동된다.")}</p>')
    card = (
        f'<div class="card">'
        f'<h3>{T("기대 대비 미제 잔여량", 800)}</h3>'
        f'<p class="csub">{T(f"판정 대상 카운티 {assessed:,}곳 가운데 기대보다 많이 "
                             f"남은 지역이 {cold_n}곳, 적게 남은 지역이 {warm_n}곳이다. "
                             f"색의 강도와 해당 카운티의 흑인 피해자 비중은 {corr:+.2f}의 "
                             f"상관을 보인다.")}</p>'
        f'<div class="bar2">'
        f'<span><label>{T("레이어", 700)}</label>'
        f'<button data-layer="res" aria-pressed="true">{V1.LABELS["layer_res"]}</button> '
        f'<button data-layer="pri" aria-pressed="false">{V1.LABELS["layer_pri"]}</button> '
        f'<button data-layer="raw" aria-pressed="false">{V1.LABELS["layer_raw"]}</button>'
        f'</span>'
        f'<span><label>{V1.LABELS["minsample"]}</label>'
        f'<input id="mn" type="range" min="0" max="{len(levels)-1}" step="1" value="0" '
        f'aria-label="{T("최소 사건 수 기준")}">'
        f'<b id="mnv"></b> <span style="color:#898781">{V1.LABELS["cases"]}</span></span>'
        f'<span><button id="tbl" aria-pressed="false">{V1.LABELS["table"]}</button></span>'
        f'</div>'
        f'<p class="lexp" id="lexp"></p>'
        f'<div class="maprow">'
        f'<div><div class="mapwrap"><svg class="map" '
        f'viewBox="0 0 {VIEW_W:.0f} {height:.0f}" '
        f'role="img" aria-label="{T("카운티별로 기대 대비 미제가 얼마나 더 남았는지를 "
                                    "나타낸 미국 지도. 개별 카운티의 값은 아래 표에서 "
                                    "키보드로도 확인할 수 있다.")}">'
        f'<defs><pattern id="na" width="5" height="5" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="5" height="5" fill="#fff"/>'
        f'<line x1="0" y1="0" x2="0" y2="5" stroke="#c3c2b7" stroke-width="1.4"/>'
        f'</pattern></defs><g id="g"></g><g id="hl"></g><g id="sel"></g>'
        f'<g id="spt"></g></svg></div>'
        f'<p class="spotcap" id="spotcap" aria-live="polite"></p>'
        f'<div class="legend" id="leg"></div></div>'
        f'<aside class="panel" id="panel" aria-live="polite"></aside>'
        f'</div>'
        f'<div id="tablewrap" class="tablewrap" hidden><table class="ct" id="ct">'
        f'<thead><tr>{th}</tr></thead><tbody></tbody></table></div>'
        f'</div>')
    return (f'<section id="mapsec"><div class="inner">{head}{howto}{card}'
            f'<div class="notes"><h3>{T("지도를 읽을 때", 800)}</h3>'
            f'<ol>{map_notes}</ol></div></div></section>')


def build_footer():
    foot_meta = (
        f'<div class="meta">'
        f'<div><b>{T("구현", 800)}</b>{T("Python · PyTorch Geometric · GraphSAGE")}'
        f'<br>{T("노드 638,454건, 엣지 2,500만 개, 사건당 특성 123개")}</div>'
        f'<div><b>{T("데이터", 800)}</b>'
        f'{T("Murder Accountability Project / Kaggle Homicide Reports 1980-2014")}'
        f'<br>{T("LEOKA openICPSR 102180 · SHR openICPSR 100699")}</div>'
        f'<div><b>{T("재현", 800)}</b><code>'
        f'{T("export CLEAR_SCOPE=national")}<br>'
        f'{T("python -m experiments.build_story_page_v2")}</code></div>'
        f'</div>')
    lic = T("데이터는 Murder Accountability Project의 "
            "“Homicide Reports, 1980-2014”(Kaggle)이며 CC BY-SA 4.0으로 "
            "제공된다. 정제·표본추출·집계 등 변경을 가했고, 이 페이지의 데이터 "
            "산출물도 동일 조건(CC BY-SA 4.0)으로 배포한다. "
            "본문 서체는 나눔스퀘어_ac(ⓒ 2010 NAVER Corporation)이며, 사용된 "
            "글자만 추출한 서브셋을 SIL Open Font License 1.1로 포함했다 — "
            "전문은 이 파일 상단 주석에 있다.")
    geo_note = T("경계 도형은 US Census 카운티 자료를 사용했다. 외부 요청이 전혀 없어 "
                 "네트워크 없이도 열람할 수 있다.")
    return (f'<footer><div class="inner">'
            f'<p class="who">{T("안윤지", 800)} · '
            f'<a href="mailto:ayj9665@gmail.com">{T("ayj9665@gmail.com", 800)}</a></p>'
            f'{foot_meta}'
            f'<p class="lic">{lic}<br>{geo_note}</p>'
            f'</div></footer>')


def pick_spotlights(static, per, levels, fips_order):
    """안내에 쓸 카운티 3곳. **데이터에서 고른다** -- 손으로 적으면 다음 빌드에서
    다른 카운티가 1위가 되었을 때 페이지가 조용히 거짓말을 한다."""
    lv0 = levels[0]
    s = per[lv0]["s"]
    cold = warm = none_i = -1
    zc, zw, area = -1e9, 1e9, -1
    for i, f in enumerate(fips_order):
        if s[i] == 2:
            z = static["stat"][i][6]
            if z > zc:
                zc, cold = z, i
            if z < zw:
                zw, warm = z, i
        elif s[i] == 0:
            # 판정에서 빠진 카운티 중 화면에서 가장 큰 것(안내가 보여야 한다)
            n = len(V1_PATHS.get(f, ""))
            if n > area:
                area, none_i = n, i
    out = []
    if cold >= 0:
        t = static["stat"][cold]
        out.append([cold, f'{T(LABELS2["spot_cold"], 700)} — '
                          f'{T(str(t[1]), 700)}, {T(str(t[0]), 700)}'])
    if warm >= 0:
        t = static["stat"][warm]
        out.append([warm, f'{T(LABELS2["spot_warm"], 700)} — '
                          f'{T(str(t[1]), 700)}, {T(str(t[0]), 700)}'])
    if none_i >= 0:
        name = V1_NAMES.get(fips_order[none_i], "")
        out.append([none_i, T(LABELS2["spot_none"], 700)
                    + (f' — {T(name, 700)}' if name else "")])
    return out


V1_PATHS = {}         # pick_spotlights가 쓰는 두 조회표. main에서 채운다
V1_NAMES = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_key", default="county")
    ap.add_argument("--src", default="cold_blocks_cv5.csv",
                    help="블록 표. 기본은 크로스피팅(카운티 1,803개).")
    ap.add_argument("--simplify_km", type=float, default=SIMPLIFY_KM)
    ap.add_argument("--out", default="clear_story_v2.html")
    ap.add_argument("--fonts_dir", default=None)
    ap.add_argument("--no_fonts", action="store_true")
    args = ap.parse_args()

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

    blue, red = CT.arms(5)
    assert palette == [CT.NEUTRAL] + blue + red, "color_tables 팔레트 구성이 바뀌었다"
    palette = [CT.NEUTRAL] + CT.gray_arm() + red

    lv0 = levels[0]
    cold_n = int(((tab["min_n"] == lv0) & (tab["flag"] == "cold")).sum())
    warm_n = int(((tab["min_n"] == lv0) & (tab["flag"] == "warm")).sum())
    assessed = sum(1 for s in per[lv0]["s"] if s)
    sub0 = tab[tab["min_n"] == lv0]
    corr = float(sub0["z"].corr(sub0["black_share"]))
    print(f"[geo] 카운티 {len(fips_order):,}개 / 판정 {assessed:,}개 / "
          f"cold {cold_n} · warm {warm_n} / z~black {corr:+.3f}")

    # ---- 본문 -----------------------------------------------------------
    register_labels()
    for row in static["stat"]:
        if row:
            V1.G.add(str(row[0]), 400); V1.G.add(str(row[0]), 700)
            V1.G.add(str(row[1]), 400); V1.G.add(str(row[1]), 700)

    V1_PATHS.update(paths)
    ft = CT.fips_table()
    V1_NAMES.update(dict(zip(ft["fips"], ft["COUNTYNAME"])))

    stats = (638454, len(fips_order), assessed, cold_n, warm_n, corr)
    body = "".join([
        V1.render("01", V1.problem_parts(stats), wrap_fold("problem"), "problem"),
        build_approach(V1.approach_parts()),
        build_arch2(),
        build_findings(),
        V1.render("05", V1.predict_parts(), wrap_fold("predict"), "s05"),
        V1.render("06", V1.diagnose_parts(), wrap_fold("diagnose"), "s06"),
        V1.render("07", V1.prescribe_parts(), wrap_fold("prescribe"), "s07"),
        build_contrib(),
    ])
    page = (build_hero(len(fips_order))
            + body
            + build_map_section(assessed, cold_n, warm_n, corr, levels, height)
            + build_limits()
            + build_stack()
            + build_footer())

    spots = pick_spotlights(static, per, levels, fips_order)

    # ---- 폰트 (본문 확정 후) --------------------------------------------
    face_css, report = ("", {"missing_fonts": ["skipped"]}) if args.no_fonts \
        else FP.build(V1.G.w, args.fonts_dir)
    print(FP.format_report(report) if not args.no_fonts else "[font] 임베드 생략")

    data_js = (
        f'var S={json.dumps(static, ensure_ascii=False, separators=(",", ":"))},'
        f'PL={json.dumps(per, separators=(",", ":"))},'
        f'P={json.dumps(palette)},'
        f'PATHS={json.dumps(paths, separators=(",", ":"))},'
        f'LV={json.dumps([int(m) for m in levels])},'
        f'FI={json.dumps(fips_order, separators=(",", ":"))},'
        f'SPOT={json.dumps(spots, ensure_ascii=False, separators=(",", ":"))},'
        f'L={json.dumps(V1.LABELS, ensure_ascii=False, separators=(",", ":"))},'
        f'L2={json.dumps(LABELS2, ensure_ascii=False, separators=(",", ":"))};')

    ofl = "" if face_css == "" else FP.license_comment()
    html = (f'{ofl}'
            f'<title>{T("CLEAR - 미제사건의 구조적 격차 진단", 800)}</title>'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="description" content="'
            f'{T("미국 살인사건 63만 건의 검거 예측, 공정성 진단과 완화, 그리고 "
                 "카운티 단위 잔차 지도. 문제 정의부터 배포까지 단독 수행한 프로젝트.")}">'
            f'<style>{face_css}{V1.css()}{V1.subst(CSS2)}</style>'
            f'{page}<div id="tip"></div>'
            f'<script>{data_js}{JS2}</script>')

    out = C.scoped_output("web")
    out.mkdir(parents=True, exist_ok=True)
    p = out / args.out
    p.write_text(html, encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"[save] {p}  ({kb:.0f} KB, 외부 요청 0)")

    V1.check(html, palette, static, per, levels, fips_order, kb,
             budget=SIZE_BUDGET_KB)
    check2(html, spots, per, levels)


def check2(html, spots, per, levels):
    """ver2 전용 검사. v1의 check가 보는 것(외부요청·색·용량)에 더해, ver2에서
    새로 생긴 '조용히 틀릴 수 있는 것'만 본다."""
    errs = []
    # 1) NOTES 12개가 정확히 한 번씩 쓰였는가. 재배치하다 한 문장을 떨어뜨리면
    #    경고문이 통째로 사라지는데, 화면상으로는 아무 일도 안 일어난다.
    used = [i for _, idx in LIMIT_GROUPS for i in idx] + MAP_NOTE_IDX
    if sorted(used) != list(range(len(V1.NOTES))):
        errs.append(f"NOTES 재배치 누락/중복: {sorted(used)} != 0..{len(V1.NOTES)-1}")
    for head, _ in V1.NOTES:
        if FP.normalize(head) not in html:
            errs.append(f"NOTES 문장이 페이지에 없다: {head[:24]}")
    # 2) 안내 대상이 실제로 판정 상태와 맞는가(cold/warm은 유의, 세 번째는 미판정)
    s = per[levels[0]]["s"]
    want = [2, 2, 0]
    for k, sp in enumerate(spots):
        if k < len(want) and s[sp[0]] != want[k]:
            errs.append(f"안내 대상 {k}의 판정 상태 {s[sp[0]]} != {want[k]}")
    # 3) ver2가 새로 약속한 것들이 실제로 문서에 있는가
    for need in ('id="panel"', 'id="sel"', 'id="spt"', 'id="spotcap"',
                 'class="atab"', 'data-zone="0"', 'href="#mapsec"', 'href="#arch"'):
        if need not in html:
            errs.append(f"필수 요소 누락: {need}")
    # 4) 선택 표시에 색을 쓰지 않았는가(테두리만)
    if re.search(r"#sel path\{[^}]*fill:(?!none)", html):
        errs.append("선택 표시가 채움색을 쓴다 -- 색은 잔여량 한 가지 뜻만 갖는다")
    if errs:
        print("[검사2] 실패 " + str(len(errs)) + "건")
        for e in errs[:20]:
            print("  - " + e)
        raise SystemExit(1)
    print("[검사2] 통과 (경고문 12개 배치 · 안내 대상 상태 · 필수 요소 · 선택=테두리)")


if __name__ == "__main__":
    main()
