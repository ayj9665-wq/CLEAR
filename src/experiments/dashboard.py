"""
experiments/dashboard.py -- 클론 직후 바로 뜨는 성능·공정성 대시보드

reports/로컬배포_계획서_CLEAR.md 의 구현.

    python -m experiments.dashboard          # HTML 생성 + 브라우저 자동 오픈
                                             # (소개 페이지가 이미 있으면 같이 연다)

## 계약: 커밋된 것만 읽는다

`data/processed/`, `dataset/`, 미커밋 예측 덤프는 **건드리지 않는다.** 이 스크립트가
읽는 것은 저장소에 커밋된 결과 CSV 43개뿐이라, `git clone` 직후 원본 데이터도 GPU도
없이 실행된다. 그게 이 산출물의 유일한 존재 이유이므로, 하나라도 어기면 산출물이
무의미해진다. main()의 자체 검사가 이를 기계적으로 확인한다.

같은 이유로 **torch를 import하지 않는다.** config.py는 os/pathlib만 쓰고 clear.results는
pandas만 쓰므로, requirements-dashboard.txt(pandas + numpy)만으로 돈다.

## 왜 서버가 아니라 HTML 파일인가

build_web_map / build_story_page 와 같은 판단이다. 자체 완결 HTML이면 서버를 띄워 둘
필요가 없고, 오프라인에서 열리고, 파일 하나로 남에게 보낼 수도 있다. 브라우저는
webbrowser 모듈로 연다(표준 라이브러리).

## 패널 1이 이 대시보드의 존재 이유다

단순 성능표가 아니라 **마진을 시드 표준편차로 나눈 비율**을 같이 낸다. "어떤 지표로
이겼다고 말할 수 있는가"를 취향이 아니라 측정으로 정한 것이 이 프로젝트의 특징이고,
평가자가 그것을 자기 기계에서 확인할 수 있어야 한다.

## 스코프를 섞지 않는다

3개 주와 전국 결과가 둘 다 커밋돼 있지만 작동점이 다르고 모든 공정성 수치가 임계값
의존이라 직접 비교가 성립하지 않는다. 탭으로 나누고 경고를 고정 표시한다.

출력: outputs/dashboard.html (자체 완결, 외부 요청 0)
"""
import argparse
import json
import re
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from experiments import _fontpack as FP

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
SIZE_BUDGET_KB = 2048

SCOPES = [("ca_tx_mi", "3개 주 (CA·TX·MI)", OUT_DIR),
          ("national", "전국 51개 주", OUT_DIR / "national")]

# 7지표 중 표에 세울 순서. MCC가 선택 기준이므로 앞에 둔다.
METRICS = ["mcc", "auc", "balanced_accuracy", "precision",
           "sensitivity", "specificity", "f1"]
METRIC_KO = {"mcc": "MCC", "auc": "AUC", "balanced_accuracy": "Balanced Acc",
             "precision": "Precision", "sensitivity": "Sensitivity",
             "specificity": "Specificity", "f1": "F1"}

# 마진÷노이즈가 이 값을 넘으면 "주장 가능"으로 표시한다. 3.0은 관례적 기준이 아니라
# 이 저장소가 이미 쓰던 선 -- Balanced Accuracy의 6.6배는 채택하고 Precision의 1.2배는
# 버렸으므로 그 사이에 선이 있다.
CLAIM_RATIO = 3.0

G = FP.WEIGHTS and None  # (자리표시용 -- 글리프 수집은 아래 Glyphs가 한다)


class Glyphs:
    """화면에 나가는 문자열을 굵기별로 모은다(_fontpack의 요구)."""

    def __init__(self):
        self.w = {w: set() for w in FP.WEIGHTS}

    def add(self, text, weight=400):
        text = FP.normalize(str(text))
        self.w[weight].update(text)
        return text


GL = Glyphs()


def T(text, weight=400):
    return GL.add(text, weight)


# ---------------------------------------------------------------------------
# 읽기 -- 없으면 None을 돌려주고, 패널이 "없다"고 말한다
# ---------------------------------------------------------------------------

MISSING = []
READ = []          # 실제로 열어 본 경로 -- 자체 검사가 이걸 본다


def load(path, hint, optional=False):
    """CSV를 읽는다. 없으면 None + 안내 문구를 기록한다.

    조용히 빈 표를 그리면 "결과가 없는 프로젝트"로 읽힌다. 파일이 없다는 사실과
    그것을 만드는 명령을 같이 보여 준다.

    optional=True는 **그 스코프에 원래 없는 파일**에 쓴다(사후 감사 표는 전국에서만
    돌렸다). 없는 게 정상인 파일을 "빠졌다"고 표시하면 안내가 잡음이 된다.
    """
    p = Path(path)
    READ.append(p)
    if not p.exists():
        if not optional:
            MISSING.append((str(p.relative_to(ROOT)), hint))
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:                      # 손상된 파일도 조용히 넘기지 않는다
        MISSING.append((str(p.relative_to(ROOT)), f"읽기 실패: {e}"))
        return None


def _params(df):
    """params JSON 열을 펼친다. 없는 키는 None."""
    return df["params"].fillna("{}").map(json.loads)


def train_rows(res, scope):
    """family=train 중 해당 스코프의 행. scope는 params에 실려 있고 기본 스코프는 없다."""
    t = res[res["family"] == "train"].copy()
    pj = _params(t)
    t["scope"] = pj.map(lambda d: d.get("scope", "ca_tx_mi"))
    t["edge"] = pj.map(lambda d: d.get("edge_type"))
    t["mode"] = pj.map(lambda d: d.get("edge_mode"))
    return t[t["scope"] == scope]


def config_label(row):
    """표에 세울 이름. **설정 키를 빠짐없이 담아야** 다른 실험이 한 줄로 섞이지 않는다.

    model/tag/edge_type만 쓰면 3개 주 blind geo에서 랭킹 그래프(rank_onehot)와 그
    대조군이 기본 shuffle과 평균돼 어디에도 없는 숫자가 나온다. tag도 substring이
    아니라 값 그대로 붙인다.
    """
    name = {"graphsage": "GraphSAGE", "xgboost": "XGBoost",
            "logreg": "LogReg"}.get(row["model"], row["model"])
    if row["model"] == "graphsage" and row["edge"]:
        name += f" ({row['edge']})"
    tag = str(row["tag"] or "")
    if tag == "blind":
        name += " · blind"
    elif tag == "blind_mb":
        name += " · blind · minibatch"
    elif tag:
        name += f" · {tag}"
    mode = str(row["mode"] or "")
    if mode:
        name += f" · {mode}"
    return name


# ---------------------------------------------------------------------------
# HTML 조각
# ---------------------------------------------------------------------------

def table(headers, rows, cls="dt", align_left=(0,)):
    th = "".join(f"<th>{T(h, 700)}</th>" for h in headers)
    body = ""
    for r in rows:
        tds = ""
        for i, cell in enumerate(r):
            klass = ' class="l"' if i in align_left else ""
            tds += f"<td{klass}>{cell}</td>"
        body += f"<tr>{tds}</tr>"
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def panel(num, title, note, body):
    return (f'<section class="pnl"><h2><span class="pn">{T(num, 800)}</span>'
            f'{T(title, 800)}</h2>'
            f'<p class="pnote">{T(note)}</p>{body}</section>')


def empty(msg, cmd):
    return (f'<div class="miss"><b>{T("이 패널의 데이터가 없다", 700)}</b>'
            f'<p>{T(msg)}</p><code>{T(cmd)}</code></div>')


def num(v, d=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else T(f"{v:.{d}f}", 400)


# ---------------------------------------------------------------------------
# 패널 1 -- 예측 성능과 "주장 가능한 지표"
# ---------------------------------------------------------------------------

def panel_predict(res, scope):
    """모델 × 7지표 + 마진÷시드노이즈.

    두 가지가 함정이라 명시해 둔다.

    (1) **`std` 열은 train 행에서 항상 비어 있다.** results.csv는 시드별로 한 줄씩
        남기고 집계는 읽는 쪽이 한다(`results.summarize`가 그 역할). 그래서 여기서
        시드에 대해 직접 std를 낸다. `std` 열을 그대로 믿으면 노이즈가 전부 사라지고
        마진÷노이즈가 통째로 무의미해진다.

    (2) **설정 키를 다 쓰지 않으면 다른 실험이 한 줄로 섞인다.** 3개 주 blind geo에는
        기본(shuffle) 말고도 랭킹 그래프(rank_onehot)와 그 차수 맞춤 대조군이 같은
        model/tag/edge_type으로 들어 있다. edge_mode를 빼고 묶으면 0.2659 / 0.2628 /
        0.2574가 평균돼 0.2620이라는, 어디에도 없는 숫자가 표에 오른다. tag도
        substring으로 보면 안 된다 -- `fairness_handoff`가 기본 sighted 행과 합쳐진다.
    """
    if res is None:
        return empty("outputs/results.csv 가 없다.", "python -m experiments.train_gnn")
    t = train_rows(res, scope)
    if t.empty:
        return empty(f"results.csv 에 scope={scope} 의 train 행이 없다.",
                     f"CLEAR_SCOPE={scope} python -m experiments.train_gnn")

    t = t[t["metric"].isin(METRICS)].copy()
    t["tag"] = t["tag"].fillna("")
    t["mode"] = t["mode"].fillna("")
    t["edge"] = t["edge"].fillna("")
    t["label"] = t.apply(config_label, axis=1)

    g = t.groupby(["label", "metric"])["value"]
    piv = g.mean().unstack()
    sd = g.std(ddof=1).unstack()
    nseed = g.size().unstack()

    order = sorted(piv.index, key=lambda s: -piv.loc[s].get("mcc", -9))
    rows = []
    for lab in order:
        cells = []
        for m in METRICS:
            v = piv.loc[lab].get(m)
            s = sd.loc[lab].get(m) if m in sd.columns else None
            if v is None or not np.isfinite(v):
                cells.append("-")
                continue
            txt = num(v)
            if s is not None and np.isfinite(s) and s > 0:
                txt += f'<span class="sd">±{s:.4f}</span>'
            cells.append(txt)
        n = int(nseed.loc[lab].get("mcc", 0) or 0)
        rows.append([T(lab, 700), T(f"{n}")] + cells)
    tbl = table(["모델", "시드"] + [METRIC_KO[m] for m in METRICS], rows)

    # --- 마진 ÷ 시드 노이즈 -------------------------------------------------
    # 기본 설정(shuffle 그래프, 특수 tag 없음)끼리만 짝짓는다.
    pairs = [("GraphSAGE (geo)", "XGBoost", "민감속성을 넣고 학습"),
             ("GraphSAGE (geo) · blind", "XGBoost · blind", "민감속성을 지우고 학습")]
    blocks = []
    for a, b, cond in pairs:
        if a not in piv.index or b not in piv.index:
            continue
        rws = []
        for m in METRICS:
            va, vb = piv.loc[a].get(m), piv.loc[b].get(m)
            s = sd.loc[a].get(m) if m in sd.columns else None
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            margin = va - vb
            if s is None or not np.isfinite(s) or s <= 0:
                rws.append([T(METRIC_KO[m], 700), num(margin), "-",
                            T("시드가 1개라 판정 불가")])
                continue
            r = margin / s
            ok = abs(r) >= CLAIM_RATIO
            bar = min(100, abs(r) / 19 * 100)
            rws.append([
                T(METRIC_KO[m], 700), num(margin), num(s),
                f'<span class="rbar"><i class="{"ok" if ok else "no"}" '
                f'style="width:{bar:.0f}%"></i></span>'
                f'<b>{T(f"{r:.1f}배", 700)}</b> '
                f'<span class="tag">{T("주장 가능" if ok else "우연 범위")}</span>'])
        if rws:
            blocks.append(f'<h3>{T(f"GraphSAGE − XGBoost · {cond}", 700)}</h3>'
                          + table(["지표", "마진", "시드 표준편차", "마진 ÷ 노이즈"],
                                  rws, align_left=(0, 3)))
    if blocks:
        ratio = ('<div class="sub">'
                 + f'<p class="pnote">{T("이 프로젝트는 마진 자체가 아니라 마진을 시드 "
                                         "표준편차로 나눈 값으로 성능 주장 여부를 정한다. "
                                         "GPU 집계가 비결정적이라 임계값에 의존하는 "
                                         "지표는 실행마다 흔들리기 때문이다. "
                                         f"{CLAIM_RATIO:g}배를 선으로 둔다.")}</p>'
                 + f'<p class="pnote">{T("표준편차는 results.csv에 남아 있는 시드만으로 "
                                         "계산한다(대개 3개). 시드가 적으면 표준편차 "
                                         "추정 자체가 불안정하므로 <b>비율의 절대값보다 "
                                         "지표 사이의 순서</b>를 본다 — 어느 지표가 "
                                         "노이즈를 넘고 어느 지표가 못 넘는지가 판단의 "
                                         "대상이다.", 700)}</p>'
                 + "".join(blocks) + "</div>")
    else:
        ratio = (f'<p class="pnote">{T("이 스코프에는 평면 모델(XGBoost) 기준선이 없어 "
                                       "마진÷노이즈를 낼 수 없다. 전국 XGBoost 덤프는 "
                                       "만들 수 없다 — 그 학습 스크립트가 저장소에서 "
                                       "제거됐고, 커밋된 덤프는 3개 주 표본이다. "
                                       "평면 대 그래프 판별 검정은 3개 주에서만 "
                                       "성립한다.")}</p>')
    return (f'<p class="pnote">{T("±는 시드 간 표준편차다. results.csv는 시드별로 한 "
                                  "줄씩 남기므로 여기서 직접 계산한다.")}</p>'
            + tbl + ratio)


# ---------------------------------------------------------------------------
# 패널 2 -- 공정성 진단
# ---------------------------------------------------------------------------

def panel_fairness(gaps, contrasts, std_gaps):
    if gaps is None:
        return empty("fairness_gaps.csv 가 없다.",
                     "python -m experiments.diagnose_fairness")
    g = gaps[gaps["group_set"].astype(str).str.startswith("min_n")].copy()
    if g.empty:
        g = gaps.copy()
    rows = []
    for _, r in g.sort_values(["attribute", "model"]).iterrows():
        amp = r["selection_rate_amplification"]
        rows.append([
            T(str(r["attribute"]), 700), T(str(r["model"])),
            T(str(r["group_set"])), num(r["base_rate_gap"]),
            num(r["selection_rate_gap"]),
            f'<b class="{"hot" if r["selection_rate_amplification_lo"] > 1 else ""}">'
            f'{T(f"{amp:.3f}", 700)}</b>'
            f'<span class="ci">[{r["selection_rate_amplification_lo"]:.2f}, '
            f'{r["selection_rate_amplification_hi"]:.2f}]</span>'])
    html = (f'<h3>{T("증폭비 — 데이터에 있던 격차를 모델이 몇 배로 벌리는가", 700)}</h3>'
            + table(["민감속성", "모델", "그룹 기준", "원자료 격차", "모델 격차",
                     "증폭비 [95%]"], rows, align_left=(0, 1, 2, 5)))

    if contrasts is not None and not contrasts.empty:
        c = contrasts[(contrasts["quantity"] == "amplification")
                      & (contrasts["metric"] == "selection_rate")]
        if c.empty:
            c = contrasts.head(40)
        # group_set을 빼면 같은 모델 쌍이 두 줄로 보인다(named_all과 n>=5000).
        shown = c.head(40)
        crows = [[T(str(r["attribute"]), 700), T(str(r.get("group_set", ""))),
                  T(str(r["model_a"])), T(str(r["model_b"])), num(r["diff"], 3),
                  f'<span class="ci">[{r["diff_lo"]:.3f}, {r["diff_hi"]:.3f}]</span>',
                  T("유의" if r["significant"] else "-", 700)]
                 for _, r in shown.iterrows()]
        more = ("" if len(c) <= len(shown) else
                f'<p class="pnote">{T(f"전체 {len(c):,}행 중 {len(shown)}행만 "
                                      "표시했다. 나머지는 "
                                      "outputs/**/fairness_model_contrasts.csv 에 "
                                      "있다.")}</p>')
        html += (f'<h3>{T("짝지은 모델 대조 — 같은 test 집합이므로 개별 CI 겹침으로 "
                          "판단하면 안 된다", 700)}</h3>'
                 + table(["민감속성", "그룹 기준", "모델 A", "모델 B", "차이",
                          "95% 구간", "판정"], crows, align_left=(0, 1, 2, 3, 5, 6))
                 + more)

    if std_gaps is not None and not std_gaps.empty:
        s = std_gaps.head(20)
        srows = [[T(str(r["attribute"]), 700), T(str(r["model"])),
                  T(str(r.get("group_set", ""))),
                  num(r.get("selection_rate_amplification"), 3)]
                 for _, r in s.iterrows()]
        html += (f'<h3>{T("주 단위 표준화 — 지역 구성이 만든 혼입을 제거한 값", 700)}</h3>'
                 + table(["민감속성", "모델", "그룹 기준", "증폭비"], srows,
                         align_left=(0, 1, 2)))
    return html


# ---------------------------------------------------------------------------
# 패널 3 -- 완화 트레이드오프
# ---------------------------------------------------------------------------

def panel_mitigate(res, scope):
    if res is None:
        return empty("outputs/results.csv 가 없다.",
                     "python -m experiments.mitigate_loss")
    m = res[res["family"].isin(["mitigate_loss", "mitigate_graph"])].copy()
    if m.empty:
        return empty("results.csv 에 완화 실험 행이 없다.",
                     "python -m experiments.mitigate_loss")
    pj = _params(m)
    m["scope"] = pj.map(lambda d: d.get("scope", "ca_tx_mi"))
    m["alpha"] = pj.map(lambda d: d.get("alpha"))
    m["beta"] = pj.map(lambda d: d.get("beta"))
    m = m[m["scope"] == scope]
    if m.empty:
        return empty(f"scope={scope} 의 완화 실험 행이 없다.",
                     f"CLEAR_SCOPE={scope} python -m experiments.mitigate_loss")

    key = ["family", "alpha", "beta"]
    acc = (m[m["metric"] == "mcc"].groupby(key, dropna=False)["value"]
           .mean().rename("mcc"))
    gap = (m[m["metric"].str.contains("amplification", na=False)]
           .groupby(key, dropna=False)["value"].mean().rename("amplification"))
    j = pd.concat([acc, gap], axis=1).reset_index().dropna(subset=["mcc"])
    if j.empty:
        return empty("완화 실험 행에서 MCC/증폭비를 찾지 못했다.",
                     "python -m experiments.mitigate_loss")
    j = j.sort_values(["family", "alpha"], na_position="first")
    rows = [[T(str(r["family"]), 700),
             T("-" if pd.isna(r["alpha"]) else f"{r['alpha']:g}"),
             T("-" if pd.isna(r["beta"]) else f"{r['beta']:g}"),
             num(r["mcc"]), num(r.get("amplification"), 3)]
            for _, r in j.iterrows()]
    return (f'<p class="pnote">{T("정확도(MCC)와 격차(증폭비)를 같은 표에 놓는다. "
                                  "격차를 되갚는 데 정확도를 얼마나 지불하는지가 이 "
                                  "단계의 결과다.")}</p>'
            + table(["실험", "alpha", "beta", "MCC", "증폭비"], rows,
                    align_left=(0,)))


# ---------------------------------------------------------------------------
# 패널 4 -- 그래프 진단
# ---------------------------------------------------------------------------

def panel_graph(homo, relat):
    html = ""
    if homo is not None and not homo.empty:
        rows = [[T(str(r["edge_type"]), 700), T(str(r["attribute"])),
                 T(f"{int(r['n_edges_directed']):,}"),
                 num(r["homophily"], 3), num(r["homophily_null"], 3),
                 f'<b class="{"hot" if r["assortativity"] > 0.1 else ""}">'
                 f'{T(f"{r['assortativity']:.4f}", 700)}</b>',
                 num(r["recovery_lift"], 4)]
                for _, r in homo.iterrows()]
        html += (f'<h3>{T("엣지가 민감속성의 프록시인가", 700)}</h3>'
                 f'<p class="pnote">{T("동류성(assortativity)은 그룹 크기를 보존한 "
                                       "무작위 재배선 대비 값이다. weapon 엣지는 "
                                       "구성상 1.000이 나오므로 경고 표시이지 결과가 "
                                       "아니다.")}</p>'
                 + table(["엣지", "민감속성", "엣지 수", "동류성", "귀무 기대",
                          "정규화 동류성", "이웃 복원 이득"], rows, align_left=(0, 1)))
    if relat is not None and not relat.empty:
        cols = [c for c in ["candidate", "profile", "block_lift", "edge_lift",
                            "oracle_lift_blind", "oracle_lift"] if c in relat.columns]
        rows = [[T(str(r[c]), 700 if i == 0 else 400) if isinstance(r[c], str)
                 else num(r[c], 4) for i, c in enumerate(cols)]
                for _, r in relat.iterrows()]
        html += (f'<h3>{T("엣지가 실제로 관련된 사건을 잇는가", 700)}</h3>'
                 f'<p class="pnote">{T("가해자 열은 모델 입력에서 누수로 제외돼 학습에 "
                                       "닿은 적이 없으므로, 외부 라벨로 쓸 수 있다. "
                                       "edge_lift가 0이면 블록 안 짝짓기는 무작위라는 "
                                       "뜻이고, 그래서 개별 엣지 중요도는 이 설계에서 "
                                       "의미가 없다.")}</p>'
                 + table(cols, rows, align_left=(0, 1)))
    if not html:
        return empty("edge_homophily.csv / edge_relatedness.csv 가 없다.",
                     "python -m experiments.edge_homophily")
    return html


# ---------------------------------------------------------------------------
# 패널 5 -- 적용(카운티 잔차)
# ---------------------------------------------------------------------------

def panel_apply(blocks, src_name):
    if blocks is None:
        return empty("cold_blocks.csv 가 없다.",
                     "python -m experiments.detect_cold_blocks --min_n 20 50 100")
    b = blocks[blocks["block_key"] == "county"].copy()
    if b.empty:
        return empty("cold_blocks.csv 에 county 블록이 없다.",
                     "python -m experiments.detect_cold_blocks --blocks county")
    lv = sorted(b["min_n"].unique())[0]
    b = b[b["min_n"] == lv].copy()
    b["flag"] = b["flag"].fillna("")
    cold, warm = int((b.flag == "cold").sum()), int((b.flag == "warm").sum())
    corr = float(b["z"].corr(b["black_share"]))

    top = b.nlargest(15, "z")
    rows = [[T(f"{r['City']}, {r['State']}", 700), T(f"{int(r['n']):,}"),
             T(f"{int(r['n_unsolved']):,}"), num(r["expected_unsolved"], 1),
             f'<b class="hot">{T(f"{r["smr"]:.3f}", 700)}</b>',
             num(r["z"], 2), num(r["black_share"], 3)]
            for _, r in top.iterrows()]
    summary = (f'<div class="kpis">'
               f'<div><b>{T(f"{len(b):,}", 800)}</b><span>{T("판정 카운티", 700)}</span></div>'
               f'<div><b class="hot">{T(str(cold), 800)}</b>'
               f'<span>{T("기대보다 많이 잔존", 700)}</span></div>'
               f'<div><b>{T(str(warm), 800)}</b>'
               f'<span>{T("기대보다 적게 잔존", 700)}</span></div>'
               f'<div><b>{T(f"{corr:+.3f}", 800)}</b>'
               f'<span>{T("잔차 ↔ 흑인비중 상관", 700)}</span></div></div>')
    return (summary
            + f'<p class="pnote">{T(f"출처 {src_name}, 최소 사건 수 {lv} 기준. "
                                    "지도는 이 표와 같은 데이터를 그린 것이고, "
                                    "지오메트리(약 10MB)를 내려받아야 하므로 별도 "
                                    "명령으로 만든다.")}</p>'
            + f'<code class="cmd">{T("python -m experiments.build_web_map "
                                     "--src cold_blocks_cv5.csv "
                                     "--out map_national.html --simplify_km 1.5")}</code>'
            + f'<h3>{T("상위 15개 카운티", 700)}</h3>'
            + table(["카운티", "사건", "미제", "기대", "배수(SMR)", "확신도(z)",
                     "흑인비중"], rows))


# ---------------------------------------------------------------------------
# 패널 6 -- 사후 감사
# ---------------------------------------------------------------------------

def panel_audit(resource, county_rho, shr, priority, crossfit, scope):
    parts = []
    if resource is not None and not resource.empty:
        # measure를 빼면 같은 min_n이 두 줄로 보인다(officers / officers_per_homicide).
        rows = [[T(f"n≥{int(r['min_n'])}", 700), T(str(r.get("measure", ""))),
                 T(f"{int(r['n_counties']):,}"),
                 num(r["race_coef"], 3), num(r["race_coef_controlled"], 3),
                 f'{num(r["race_attenuation"], 4)}'
                 f'<span class="ci">[{r["attenuation_lo"]:.3f}, '
                 f'{r["attenuation_hi"]:.3f}]</span>',
                 num(r["partial_r2_resource"], 4)]
                for _, r in resource.iterrows()]
        parts.append(f'<h3>{T("수사 인력으로 설명되는 몫", 700)}</h3>'
                     f'<p class="pnote">{T("경찰 인력이 사건 수를 따라 배치되는 탓에 "
                                           "이 변수의 변동이 절반으로 눌려 있다. 따라서 "
                                           "감쇠율은 상한이 아니라 하한이다.")}</p>'
                     + table(["표본 기준", "지표", "카운티", "인종 계수", "통제 후",
                              "감쇠율 [95%]", "부분 R²"], rows, align_left=(0, 1, 5)))
    if county_rho is not None and not county_rho.empty:
        rows = [[T(f"인종별 n≥{int(r['min_race_n'])}", 700),
                 T(f"{int(r['n_counties']):,}"),
                 f'{num(r["rho_wmean"], 4)}'
                 f'<span class="ci">[{r["rho_wmean_lo"]:.3f}, '
                 f'{r["rho_wmean_hi"]:.3f}]</span>',
                 f'{num(r["corr_rho_black_share"], 4)}'
                 f'<span class="ci">[{r["corr_rho_black_share_lo"]:.3f}, '
                 f'{r["corr_rho_black_share_hi"]:.3f}]</span>']
                for _, r in county_rho.iterrows()]
        parts.append(f'<h3>{T("카운티를 고정했을 때 남는 인종 격차", 700)}</h3>'
                     f'<p class="pnote">{T("ρ는 같은 카운티 안에서 흑인 대 백인 "
                                           "미제 잔존의 로그비다. ρ가 0이 아니면서 "
                                           "ρ와 흑인비중의 상관이 0이면, 지도의 색은 "
                                           "카운티 간 이야기이고 카운티 내부 격차는 "
                                           "지역과 무관하게 균일하다는 뜻이다.")}</p>'
                     + table(["표본 기준", "카운티", "ρ (가중평균) [95%]",
                              "ρ ↔ 흑인비중 [95%]"], rows, align_left=(0, 2, 3)))
    if shr is not None and not shr.empty:
        cols = [c for c in shr.columns if shr[c].dtype != object][:6]
        rows = [[T(str(r[shr.columns[0]]), 700)]
                + [num(r[c], 4) for c in cols] for _, r in shr.head(10).iterrows()]
        parts.append(f'<h3>{T("정황 기록의 인종 차등", 700)}</h3>'
                     f'<p class="pnote">{T("기록된 정황은 사건이 해결된 뒤에 적히는 "
                                           "값이라 통제 변수로 쓸 수 없다. 미제 사건에서 "
                                           "흑인 피해자 쪽이 카운티를 고정해도 더 자주 "
                                           "'미확정'으로 기록된다.")}</p>'
                     + table([shr.columns[0]] + cols, rows, align_left=(0,)))
    if priority is not None and not priority.empty:
        p = priority.copy()
        rows = [[T(str(r["model"]), 700), T(str(r["attribute"])),
                 T(str(r["point"])), T(str(r["group"]), 700),
                 f'{num(r["lift"], 3)}<span class="ci">[{r["lo"]:.2f}, '
                 f'{r["hi"]:.2f}]</span>',
                 T("유의" if r["significant"] else "-")]
                for _, r in p.head(24).iterrows()]
        parts.append(f'<h3>{T("재수사 우선순위 목록의 구성 감사", 700)}</h3>'
                     f'<p class="pnote">{T("목록 자체는 공개하지 않고 감사 통계만 낸다. "
                                           "벌점을 준 속성(인종)에서는 완화가 목록까지 "
                                           "전달되지만, 누르지 않은 속성(성별)에서는 "
                                           "오히려 나빠진다.")}</p>'
                     + table(["모델", "속성", "지점", "그룹", "lift [95%]", "판정"],
                             rows, align_left=(0, 1, 2, 3, 4, 5)))
    if crossfit is not None and not crossfit.empty:
        cols = list(crossfit.columns)[:7]
        rows = [[T(str(r[c]), 700 if i == 0 else 400) if isinstance(r[c], str)
                 else num(r[c], 4) for i, c in enumerate(cols)]
                for _, r in crossfit.iterrows()]
        parts.append(f'<h3>{T("교차적합 대조 — 늘어난 상관 하락이 무엇 때문인가", 700)}</h3>'
                     + table(cols, rows, align_left=(0,)))
    if not parts:
        if scope != "national":
            return (f'<p class="pnote">{T("사후 감사(수사 인력·정황 기록·카운티 차분·"
                                          "우선순위·교차적합)는 전국 스코프에서만 "
                                          "수행했다. 위 탭을 전국으로 바꾸면 여기에 "
                                          "표가 나온다.")}</p>')
        return empty("사후 감사 CSV가 없다.",
                     "CLEAR_SCOPE=national python -m experiments.audit_resources")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------

def build_scope(scope, label, d):
    res = load(OUT_DIR / "results.csv", "python -m experiments.train_gnn")
    # 사후 감사 6종은 전국 스코프에서만 돌렸다. 3개 주에서 "빠졌다"고 하면 잡음이다.
    opt = scope != "national"
    src = "cold_blocks_cv5.csv" if (d / "cold_blocks_cv5.csv").exists() \
        else "cold_blocks.csv"
    panels = [
        panel("01", "예측 성능", "모델별 7지표와, 그 마진을 시드 노이즈로 나눈 값.",
              panel_predict(res, scope)),
        panel("02", "공정성 진단",
              "예측이 피해자 인종·성별에 따라 체계적으로 달라지는가.",
              panel_fairness(
                  load(d / "fairness_gaps.csv",
                       "python -m experiments.diagnose_fairness"),
                  load(d / "fairness_model_contrasts.csv",
                       "python -m experiments.diagnose_fairness"),
                  load(d / "fairness_gaps_standardized.csv",
                       "python -m experiments.diagnose_fairness --stratum State"))),
        panel("03", "완화 트레이드오프",
              "격차를 되갚는 데 정확도를 얼마나 지불하는가.",
              panel_mitigate(res, scope)),
        panel("04", "그래프 진단",
              "엣지가 민감속성의 프록시인지, 그리고 실제로 관련된 사건을 잇는지.",
              panel_graph(load(d / "edge_homophily.csv",
                               "python -m experiments.edge_homophily"),
                          load(d / "edge_relatedness.csv",
                               "python -m experiments.edge_relatedness",
                               scope == "national"))),
        panel("05", "적용 — 카운티 잔차",
              "완화한 모델을 기준선으로 삼았을 때 설명되지 않는 미제가 남는 지역.",
              panel_apply(load(d / src,
                               "python -m experiments.detect_cold_blocks "
                               "--min_n 20 50 100"), src)),
        panel("06", "사후 감사",
              "잔차에 섞인 채널을 모델 밖에서 하나씩 재 본 결과.",
              panel_audit(
                  load(d / "resource_audit.csv",
                       "python -m experiments.audit_resources", opt),
                  load(d / "county_race_residual_summary.csv",
                       "python -m experiments.county_race_residual", opt),
                  load(d / "shr_recording_summary.csv",
                       "python -m experiments.shr_circumstance", opt),
                  load(d / "priority_audit.csv",
                       "python -m experiments.audit_priority", opt),
                  load(d / "crossfit_compare.csv",
                       "python -m experiments.crossfit_compare", opt), scope)),
    ]
    return "".join(panels)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fcfcfb;color:#0b0b0b;font-family:__FAM__,-apple-system,
  "Segoe UI","Malgun Gothic",system-ui,sans-serif;font-size:15px;line-height:1.7;
  font-synthesis:none;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px 90px}
header{border-bottom:1px solid #e1e0d9;padding:44px 0 26px;margin-bottom:8px}
h1{font-weight:800;font-size:30px;letter-spacing:-.03em;margin:0}
.sub{color:#52514e;margin:8px 0 0;font-size:14px;max-width:78ch}
.warn{margin:20px 0 0;padding:12px 15px;border-left:2px solid #e34948;
  background:#e3494810;font-size:13.5px;color:#52514e}
.warn b{color:#0b0b0b;font-weight:700}
.tabs{display:flex;gap:8px;margin:24px 0 0}
.tabs button{font:inherit;font-weight:700;font-size:13px;padding:7px 16px;
  border:1px solid #c3c2b7;background:transparent;color:#52514e;border-radius:999px;
  cursor:pointer}
.tabs button[aria-pressed="true"]{background:#0b0b0b;color:#fff;border-color:#0b0b0b}
.pnl{padding:38px 0;border-bottom:1px solid #e1e0d9}
.pnl h2{font-weight:800;font-size:20px;margin:0 0 4px;letter-spacing:-.02em;
  display:flex;align-items:baseline;gap:11px}
.pn{color:#e34948;font-size:12px;letter-spacing:.14em}
.pnl h3{font-weight:700;font-size:14.5px;margin:26px 0 8px}
.pnote{color:#52514e;font-size:13.5px;margin:0 0 14px;max-width:82ch}
.tw{overflow-x:auto;border:1px solid #e1e0d9;border-radius:10px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 11px;border-bottom:1px solid #eeede7;text-align:right;
  white-space:nowrap}
th{background:#f7f7f5;color:#52514e;font-weight:700;font-size:12px;
  position:sticky;top:0}
td.l,th:first-child{text-align:left}
tbody tr:hover{background:#0000000a}
.sd{color:#898781;font-size:11px;margin-left:4px}
.ci{color:#898781;font-size:11px;margin-left:5px}
b.hot{color:#a72629}
.rbar{display:inline-block;width:88px;height:6px;background:#0000000f;
  border-radius:3px;vertical-align:middle;margin-right:8px;overflow:hidden}
.rbar i{display:block;height:100%;border-radius:3px}
.rbar i.ok{background:#0b0b0b}
.rbar i.no{background:#00000026}
.tag{color:#898781;font-size:11.5px;margin-left:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:18px;margin:4px 0 18px}
.kpis b{display:block;font-weight:800;font-size:30px;letter-spacing:-.03em}
.kpis span{display:block;font-weight:700;font-size:12.5px;color:#52514e;margin-top:2px}
.miss{border:1px dashed #c3c2b7;border-radius:10px;padding:18px 20px;
  background:#f7f7f5;color:#52514e}
.miss b{display:block;color:#0b0b0b;margin-bottom:4px}
.miss p{margin:0 0 9px;font-size:13.5px}
code,.cmd{display:block;background:#0b0b0b;color:#f2f2f0;padding:9px 12px;
  border-radius:7px;font-size:12px;font-family:ui-monospace,Consolas,monospace;
  overflow-x:auto;white-space:pre;letter-spacing:0}
.cmd{margin:8px 0 4px}
.sub{margin-top:10px}
footer{padding:34px 0 0;color:#898781;font-size:12.5px}
@media (max-width:640px){.wrap{padding:0 14px 60px}h1{font-size:23px}}
"""

JS = """
var TABS=document.querySelectorAll('[data-scope]');
function show(s){
  TABS.forEach(function(b){b.setAttribute('aria-pressed', b.dataset.scope===s);});
  document.querySelectorAll('[data-pane]').forEach(function(p){
    p.hidden = p.dataset.pane!==s;
  });
}
TABS.forEach(function(b){b.onclick=function(){show(b.dataset.scope);};});
show(TABS[0].dataset.scope);
"""


# 커밋 전이지만 계약에 필요한 파일. 커밋되면 git ls-files에 잡히므로 중복돼도 무해하다.
_CONTRACT_FILES = ["src/experiments/dashboard.py", "src/experiments/_fontpack.py",
                   "requirements-dashboard.txt"]


def readme_tables():
    """README에 박을 마크다운 표를 결과 CSV에서 생성한다.

    숫자를 README에 손으로 적으면 재실행 때 조용히 어긋난다. 출처를 파일 하나로
    고정하고, 어긋나면 재생성으로 고친다. README의 METRICS 마커 사이만 갈아끼운다.
    """
    NL = "\n"
    res = load(OUT_DIR / "results.csv", "")
    out = []

    for scope, label, _d in SCOPES:
        if res is None:
            break
        t = train_rows(res, scope)
        t = t[t["metric"].isin(["mcc", "auc", "balanced_accuracy", "precision"])].copy()
        if t.empty:
            continue
        t["tag"] = t["tag"].fillna("")
        t["mode"] = t["mode"].fillna("")
        t["edge"] = t["edge"].fillna("")
        t["label"] = t.apply(config_label, axis=1)
        g = t.groupby(["label", "metric"])["value"]
        piv, sd = g.mean().unstack(), g.std(ddof=1).unstack()
        out.append("")
        out.append(f"**{label}** — 예측 성능 (± 는 시드 간 표준편차)")
        out.append("")
        out.append("| 모델 | MCC | AUC | Balanced Acc | Precision |")
        out.append("|---|---|---|---|---|")
        for lab in sorted(piv.index, key=lambda x: -piv.loc[x].get("mcc", -9)):
            cells = []
            for m in ("mcc", "auc", "balanced_accuracy", "precision"):
                v, e = piv.loc[lab].get(m), sd.loc[lab].get(m)
                if not np.isfinite(v):
                    cells.append("—")
                else:
                    cells.append(f"{v:.4f}"
                                 + (f" ±{e:.4f}" if np.isfinite(e) and e > 0 else ""))
            out.append(f"| {lab} | " + " | ".join(cells) + " |")

    d = OUT_DIR / "national"
    gaps = load(d / "fairness_gaps.csv", "")
    if gaps is not None:
        g = gaps[gaps["group_set"].astype(str).str.contains(">=5000")]
        g = g[g["model"].isin(["graphsage_geo_blind_mb", "graphsage_fairloss_a100_mb"])]
        if not g.empty:
            out.append("")
            out.append("**전국** — 공정성 증폭비 (1.0이 기준선, 대괄호는 95% 신뢰구간)")
            out.append("")
            out.append("| 민감속성 | 모델 | 원자료 격차 | 모델 격차 | 증폭비 |")
            out.append("|---|---|---|---|---|")
            for _, r in g.iterrows():
                out.append(
                    f"| {r['attribute']} | `{r['model']}` | "
                    f"{r['base_rate_gap']:.4f} | {r['selection_rate_gap']:.4f} | "
                    f"**{r['selection_rate_amplification']:.3f}** "
                    f"[{r['selection_rate_amplification_lo']:.2f}, "
                    f"{r['selection_rate_amplification_hi']:.2f}] |")

    blocks = load(d / "cold_blocks_cv5.csv", "")
    if blocks is not None:
        b = blocks[blocks["block_key"] == "county"]
        lv = sorted(b["min_n"].unique())[0]
        b = b[b["min_n"] == lv].copy()
        b["flag"] = b["flag"].fillna("")
        out.append("")
        out.append(f"**전국** — 적용 단계 (교차적합 예측, 최소 사건 수 {lv})")
        out.append("")
        out.append("| 판정 카운티 | 기대보다 많이 잔존 | 기대보다 적게 잔존 | "
                   "잔차 ↔ 흑인 피해자 비중 |")
        out.append("|---|---|---|---|")
        out.append(f"| {len(b):,} | {int((b.flag == 'cold').sum())} | "
                   f"{int((b.flag == 'warm').sum())} | "
                   f"{b['z'].corr(b['black_share']):+.3f} |")

    md = NL.join(out).strip() + NL
    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    a, b_ = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
    if a in txt and b_ in txt:
        merged = txt[:txt.index(a) + len(a)] + NL + md + txt[txt.index(b_):]
        readme.write_text(merged, encoding="utf-8")
        print(f"[readme] METRICS 구간을 {len(out)}줄로 갱신했다")
    else:
        print(f"[readme] 마커가 없어 표만 출력한다 ({a} ... {b_} 를 넣을 것)")
        print(md)


def verify_clone():
    """git이 추적하는 파일만 복사한 임시 트리에서 대시보드가 빌드되는지 확인한다.

    이 산출물의 계약은 "클론 직후 원본 데이터 없이 실행된다" 하나뿐이므로, 검사도
    거기에 건다. 소스를 읽어 추정하지 않고 **실제로 데이터 없는 트리에서 돌려 본다.**
    폰트가 없는 기계도 가정해 --no_fonts 경로까지 같이 확인한다.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    tracked = [t for t in subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8").stdout.splitlines() if t.strip()]
    tmp = Path(tempfile.mkdtemp(prefix="clearclone_"))
    n = 0
    for rel in tracked + _CONTRACT_FILES:
        src = ROOT / rel
        if not src.exists():
            continue
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp / rel)
        n += 1
    print(f"[clone] 추적 파일 {n}개 복사 -> {tmp}")
    for label, extra in (("기본", []), ("폰트 없는 기계", ["--no_fonts"])):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "experiments.dashboard", "--no_open"]
            + extra, cwd=tmp / "src", capture_output=True, text=True, encoding="utf-8")
        made = (tmp / "outputs" / "dashboard.html").exists()
        print(f"  [{label}] exit={r.returncode} html={made}")
        if r.returncode != 0 or not made:
            print(r.stdout[-1500:] or r.stderr[-1500:])
            shutil.rmtree(tmp, ignore_errors=True)
            raise SystemExit("[검사] 클론 직후 실행 실패 — 이 산출물의 계약이 깨졌다")
        (tmp / "outputs" / "dashboard.html").unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    print("[검사] 통과 — 원본 데이터·GPU·모델 없이 빌드된다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--no_open", action="store_true",
                    help="브라우저를 열지 않고 파일만 만든다(CI·원격용)")
    ap.add_argument("--scope", choices=[s for s, _, _ in SCOPES], default=None,
                    help="한 스코프만 담는다(기본: 둘 다)")
    ap.add_argument("--fonts_dir", default=None)
    ap.add_argument("--no_fonts", action="store_true")
    ap.add_argument("--verify_clone", action="store_true",
                    help="추적 파일만 있는 임시 트리에서 빌드되는지 확인하고 종료")
    ap.add_argument("--emit_readme_tables", action="store_true",
                    help="README의 METRICS 구간을 results.csv에서 다시 만든다")
    args = ap.parse_args()

    if args.verify_clone:
        verify_clone()
        return
    if args.emit_readme_tables:
        readme_tables()
        return

    scopes = [s for s in SCOPES if args.scope is None or s[0] == args.scope]
    tabs, panes = "", ""
    for scope, label, d in scopes:
        tabs += (f'<button data-scope="{scope}" aria-pressed="false">'
                 f'{T(label, 700)}</button>')
        panes += (f'<div data-pane="{scope}" hidden>{build_scope(scope, label, d)}'
                  f'</div>')

    head = (
        f'<header><div class="wrap">'
        f'<h1>{T("CLEAR — 성능 · 공정성 대시보드", 800)}</h1>'
        f'<p class="sub">{T("저장소에 커밋된 결과 CSV만 읽어 만든 화면이다. 원본 "
                            "데이터도, GPU도, 모델 가중치도 필요 없다 — git clone 직후 "
                            "바로 실행된다.")}</p>'
        f'<div class="warn">{T("<b>두 스코프의 값을 직접 비교하지 말 것.</b> 3개 주와 "
                               "전국은 작동점이 다르고, 이 프로젝트의 공정성 수치는 전부 "
                               "임계값에 의존한다. 같은 축 위의 숫자가 아니다.")}</div>'
        f'<div class="tabs">{tabs}</div></div></header>')

    miss_html = ""
    if MISSING:
        items = "".join(f'<li><code>{T(p)}</code> {T(h)}</li>'
                        for p, h in dict(MISSING).items())
        miss_html = (f'<section class="pnl"><h2>'
                     f'<span class="pn">{T("NOTE", 800)}</span>'
                     f'{T("이 실행에서 찾지 못한 파일", 800)}</h2>'
                     f'<p class="pnote">{T("아래 파일이 없어 일부 패널이 비어 있다. "
                                           "옆의 명령으로 만들 수 있다.")}</p>'
                     f'<ul class="miss-list">{items}</ul></section>')

    foot = (f'<footer>{T("출처: Murder Accountability Project / Kaggle Homicide "
                         "Reports 1980–2014. 이 화면의 모든 숫자는 outputs/ 아래 커밋된 "
                         "CSV에서 그대로 읽은 값이며, 재현 방법은 README를 참고할 것.")}'
            f'</footer>')

    face_css, report = ("", {"missing_fonts": ["skipped"]}) if args.no_fonts \
        else FP.build(GL.w, args.fonts_dir)
    print(FP.format_report(report) if not args.no_fonts else "[font] 임베드 생략")

    html = (f'<title>{T("CLEAR — 성능 대시보드", 800)}</title>'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<style>{face_css}{CSS.replace("__FAM__", FP.FAMILY)}</style>'
            f'{head}<div class="wrap">{panes}{miss_html}{foot}</div>'
            f'<script>{JS}</script>')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / args.out
    p.write_text(html, encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"[save] {p}  ({kb:.0f} KB, 외부 요청 0)")
    if MISSING:
        print(f"[note] 찾지 못한 파일 {len(dict(MISSING))}개 — 화면에 안내로 표시했다")
    check(html, kb)

    if not args.no_open:
        webbrowser.open(p.resolve().as_uri())
        print("[open] 브라우저에서 열었다")
        open_story()


# 소개 페이지는 별도 산출물이지만 같이 봐야 뜻이 통한다 -- 대시보드는 "숫자가 맞는가",
# 소개 페이지는 "그래서 무슨 이야기인가"다. 다만 이쪽은 gitignore 대상이라 클론 직후에는
# 없다. 없는 파일을 webbrowser에 넘기면 빈 탭이 뜨므로, 있을 때만 열고 없으면 만드는
# 명령을 안내한다. load()를 쓰지 않으므로 "커밋된 CSV만 읽는다"는 계약은 그대로다.
STORY = [d / "web" / "clear_story.html" for _, _, d in reversed(SCOPES)]


def open_story():
    for s in STORY:
        if s.exists():
            webbrowser.open(s.resolve().as_uri())
            print(f"[open] 소개 페이지도 열었다: {s.relative_to(ROOT)}")
            return
    print("[open] 소개 페이지가 없어 건너뛴다 — 만들려면:\n"
          "       CLEAR_SCOPE=national python -m experiments.build_story_page")


def check(html, kb):
    errs = []
    scrub = re.sub(r"base64,[A-Za-z0-9+/=]+", "base64,X", html)
    for m in re.finditer(r"https?://[^\s\"')]+", scrub):
        if m.group(0).startswith("http://www.w3.org"):
            continue
        errs.append(f"외부 URL: {m.group(0)}")
    if kb > SIZE_BUDGET_KB:
        errs.append(f"용량 {kb:.0f}KB > 예산 {SIZE_BUDGET_KB}KB")
    # 클론 직후 실행이 이 산출물의 유일한 계약이다. 소스를 grep하는 대신 **실제로
    # 열어 본 경로**를 검사한다 -- 문서 주석에 경로 이름이 나온다고 실패하면 안 되고,
    # 반대로 주석 없이 몰래 읽는 경로는 잡아야 한다.
    for p in READ:
        try:
            rel = p.resolve().relative_to(OUT_DIR.resolve())
        except ValueError:
            errs.append(f"outputs/ 밖을 읽었다: {p}")
            continue
        if rel.suffix != ".csv":
            errs.append(f"CSV가 아닌 파일을 읽었다: {rel}")
        if "web" in rel.parts or "predictions" in rel.parts:
            errs.append(f"gitignore 대상 디렉터리를 읽었다: {rel}")
    if errs:
        print("[검사] 실패 " + str(len(errs)) + "건")
        for e in errs[:10]:
            print("  - " + e)
        raise SystemExit(1)
    print(f"[검사] 통과 (외부요청 0 · 커밋된 CSV만 읽음 · 용량 {kb:.0f}KB)")


if __name__ == "__main__":
    main()
