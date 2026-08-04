"""
experiments/build_web_map.py -- 카운티 잔차 인터랙티브 지도(자체 완결 HTML)

확장설계서 트랙 E의 최종 산출물. detect_cold_blocks가 만든 블록별 관측-기대 잔차를
카운티 코로플레스로 그린다.

## 왜 웹인가 -- 스케일이 매체를 정한다

3개 주 392개 카운티는 정적 PNG로 읽힌다(experiments/map_figures.py). 전국 3,042개는
읽히지 않는다 -- 호버와 표가 있어야 개별 카운티에 접근할 수 있다. 취향이 아니라
필연이고, 두 산출물이 역할을 나눈다.

## 라이브러리를 쓰지 않는다

코로플레스는 결국 <path>에 fill을 칠하는 것이다. Plotly(약 3MB)나 D3 번들을 실을
이유가 없고, 외부 요청이 막힌 환경이면 그걸 전부 인라인해야 한다. 어려운 쪽
(투영·단순화)은 파이썬이 이미 잘하는 일이므로 여기서 끝내고, 브라우저는 색칠과
호버만 한다. geopandas를 쓰지 않은 것과 같은 판단이다(clear/counties.py 참고).

## 최소표본 슬라이더는 필터가 아니다

n>=20/50/100은 **각각 검정을 다시 돌린 결과**다. BH FDR의 q값이 동시에 검정한 블록
수에 의존하므로, 한 번 돌리고 n으로 거르면 q값이 틀린다. 그래서 이 스크립트는
cold_blocks.csv의 min_n 열에 여러 수준이 있기를 요구한다:

    python -m experiments.detect_cold_blocks --min_n 20 50 100

슬라이더가 있어야 하는 이유도 여기 있다 -- "이 결과가 표본 기준에 얼마나 민감한가"를
독자가 직접 확인하게 하는 것이, 이 프로젝트가 격차를 두 벌로 보고하는 규율의
인터랙티브 버전이다.

## 3단 인코딩

유의(발산색) / 판정했으나 비유의(중립 회색) / 미판정(빗금). 판정 결과를 **빌드 시점에
불리언으로 확정**해 JSON에 넣는다 -- map_figures가 CSV 왕복에서 빈 flag가 NaN이 되어
`flag == ""`가 조용히 전부 실패했고, 143개 카운티를 전부 신호처럼 칠했다. 브라우저에서
문자열을 비교하지 않는 이유가 그것이다.

출력: outputs[/{scope}]/web/map.html (자체 완결, 외부 요청 0)
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

import config as C
from experiments import _htmlcheck as HC
from clear import counties as CT

VIEW_W = 1000.0            # viewBox 폭(투영 좌표를 여기에 맞춘다)
SIMPLIFY_KM = 1.0          # 지오메트리 단순화 허용오차. 468KB(1km) vs 1.12MB(무단순화)
EARTH_SPAN_KM = 4500.0     # CONUS 동서 폭 -- 허용오차를 km로 환산하는 기준

# 페이지가 반드시 실어야 할 맥락(확장설계서 §6-9). 지도는 오독되기 쉽고, 이 문장들이
# 빠지면 "빨간 카운티 = 수사를 못한다"로 읽힌다.
CAVEATS = [
    ("빨간 카운티는 '수사를 못한다'는 뜻이 아니다.",
     "사건 구성(무기·피해자 연령·기관 유형 등)과 주(州)를 통제한 뒤에도 "
     "<b>설명되지 않는</b> 미해결 집중이 남는다는 뜻이다."),
    ("연쇄살인 탐지기가 아니다.",
     "이 데이터에는 가해자 ID가 없어 '이 사건들은 동일범'이라는 주장은 "
     "원리적으로 검증 불가능하다. 여기서 하는 것은 군집 단위 이상 탐지다."),
    ("모델은 카운티를 모른다.",
     "City는 특성이 아니라 블로킹 키일 뿐이라, z는 사실상 '이 카운티가 주·사건구성 "
     "기준선에서 얼마나 벗어나는가'다. 그 이탈이 사건 구성인지, 차별인지, 수사 자원인지, "
     "기록 관행인지는 <b>이 설계로 분리되지 않는다</b>(도시성은 이 넷과 나란한 채널이 "
     "아니라 그 공통원인이다). 인과로 읽으면 안 된다."),
    # 이 색이 인종 이야기인지 묻는 것이 지도의 첫 반론이다. 이제 "어느 층에 있는가"까지
    # 답할 수 있으므로 넣는다 -- 빼면 z~black_share=+0.290이 카운티 내부 차별로 읽힌다.
    ("색과 인종 구성의 상관은 <b>카운티 사이</b>에 있지 카운티 안에 있지 않다.",
     "카운티를 고정하고 인종별로 같은 잔차를 내면 흑인 피해자 사건이 기대 대비 "
     "<b>8.7% 더</b> 미해결로 남는다(log-SMR비 +0.083 [+0.056, +0.128], 카운티 651개). "
     "0이 아니다. 그런데 그 값과 카운티 흑인비중의 상관은 <b>0</b>이다"
     "(−0.015 [−0.108, +0.105]). 즉 흑인 비중이 높은 카운티는 잔차가 나쁘지만 그 안에서 "
     "흑인 피해자가 특별히 더 나쁘지는 않고, 카운티 내부 격차는 어디에나 균일하다. "
     "<b>지도의 색은 카운티 수준 결손을 보여 주는 것이지 카운티 내부 차별의 지도가 "
     "아니다.</b>"),
    # 넷 중 하나에 하한을 붙였다(§6-4-1). 여기서 빼면 지도만 본 사람은 여전히
    # "인력 부족 아니냐"를 첫 반론으로 들고 오는데, 그건 이제 숫자로 답할 수 있다.
    # 다만 상한이 아니라 하한이라는 것까지 같이 써야 오독되지 않는다.
    ("인력 규모는 재 봤다 — 다만 하한만 안다.",
     "LEOKA 기관-연도 경찰관 수를 붙여(조인 100%) 사건 1건당 경찰관을 통제하면 인종 구성 "
     "계수는 전수 기준 <b>거의 안 줄고</b>(감쇠 0) 큰 카운티에서 8~10% 줄어든다. "
     "단 인력이 사건 수를 따라 배치되는 탓에 이 변수의 변동이 절반으로 눌려 있어, "
     "<b>8%는 상한이 아니라 하한</b>이다 — 진짜 자원 몫은 더 클 수 있고 상한은 이 설계로 "
     "얻을 수 없다. 지표도 총 경찰관 수이지 형사 인원이 아니다."),
    ("기대값을 낸 모델은 공정성 완화를 거쳤다.",
     "완화 전 모델은 FPR 기준 인종 증폭이 1.8배라 기대값 자체가 기울어 있다. "
     "완화 모델만이 인종 중립적 기준선을 준다 -- 그래서 잔차가 인종 구성과 상관되면 "
     "그것은 잡음이 아니라 결과다."),
    ("소표본 카운티는 칠하지 않는다.",
     "최소 표본 기준은 슬라이더로 조절할 수 있고, 기준마다 <b>검정을 다시 돌린</b> "
     "결과다(BH FDR의 q값이 검정 개수에 의존하므로 단순 필터와 다르다)."),
]

# 어느 분할에서 나온 지도인지는 **캡션에 반드시 있어야 한다.** 두 지도가 나란히
# 배포되는데 z의 스케일이 분할마다 다르기 때문이다 -- 같은 카운티가 크로스피팅 지도에서
# 더 진하게 보이는 것은 신호가 커진 것이 아니라 표본이 커진 것이다(|z| 최대 15.66 대
# 26.60). 그 경고 없이 두 장을 비교하면 데이터가 달라진 것으로 읽힌다.
SPLIT_NOTES = {
    "crossfit": (
        "예측은 <b>5-fold cross-fitting</b>의 out-of-fold 값이다.",
        "전체 638,454행에 학습에 쓰이지 않은 예측이 있으므로 카운티 1,803개(57%)가 "
        "판정된다 -- test 분할 지도의 853개(27%)보다 2.1배다. 대신 이 지도는 공정성 "
        "지표 표(<code>fairness_*.csv</code>)와 <b>다른 분할</b> 위에 있고, 표본이 "
        "3.4배라 z가 전반적으로 크다. <b>두 지도의 색 강도를 비교하지 말 것</b> — "
        "SMR(이상성의 크기)은 두 분할에서 거의 같고 달라지는 것은 확신도다."),
    "test": (
        "예측은 <b>test 분할</b>(층화 무작위 30%)의 값이다.",
        "공정성 지표 표와 같은 분할이라 나란히 인용할 수 있다. 대신 판정되는 카운티가 "
        "853개(27%)뿐이고, 빗금은 작은 카운티에 몰려 있다 — 전수 판정본은 "
        "cross-fitting 지도다."),
}

# 알래스카만 축척이 다르다(0.35배). 하와이는 CONUS와 같은 축척이라 면적 비교가 성립한다.
PROJECTION_NOTE = ("투영은 Albers 등적(equal-area). 알래스카는 지면에 담기 위해 "
                   "<b>0.35배로 축소</b>했으므로 알래스카 카운티의 화면 면적을 "
                   "본토와 비교하면 안 된다. 하와이는 본토와 같은 축척이다.")


def load_blocks(block_key, src="cold_blocks.csv"):
    """블록 표를 읽는다. src를 바꾸면 **다른 분할**의 표로 지도를 그릴 수 있다 --
    크로스피팅 표(cold_blocks_cv5.csv)가 그 경우다. 두 분할은 파일이 갈려 있고
    (detect_cold_blocks --out) 지도도 --out으로 갈라 두어야 한다. 같은 파일에 섞으면
    어느 카운티가 어느 분할에서 판정된 것인지 지도 위에서 알 수 없다."""
    path = C.scoped_output(src)
    if not path.exists():
        raise SystemExit(f"[에러] {path} 없음. 먼저: "
                         f"python -m experiments.detect_cold_blocks --min_n 20 50 100")
    tab = pd.read_csv(path)
    tab = tab[tab["block_key"] == block_key].copy()
    if tab.empty:
        raise SystemExit(f"[에러] block_key={block_key} 행이 없다.")
    if "min_n" not in tab.columns:
        raise SystemExit(
            "[에러] cold_blocks.csv에 min_n 열이 없다(옛 형식). 슬라이더는 기준마다 "
            "**다시 돌린** 검정을 요구한다:\n"
            "  python -m experiments.detect_cold_blocks --min_n 20 50 100")
    # flag는 CSV 왕복에서 빈 문자열이 NaN이 된다. 여기서 확정해 브라우저로 넘긴다.
    tab["flag"] = tab["flag"].fillna("")
    return tab


def build_paths(state_fips, tol):
    """지도에 올릴 카운티 -> SVG path. 표본에 있는 주의 카운티만 그린다.

    표본에 없는 주까지 회색으로 깔면 '판정했으나 유의하지 않음'과 구분이 안 된다 --
    우리가 아무 말도 할 수 없는 지역이므로 지도에서 뺀다.
    """
    geo = CT.load_geometry()
    proj = {}
    for fips, rings in geo.items():
        if fips[:2] not in state_fips:
            continue
        proj[fips] = [np.column_stack(CT.albers_usa(r[:, 0], r[:, 1], fips))
                      for r in rings]
    if not proj:
        raise SystemExit("[에러] 그릴 카운티가 없다(FIPS 조인 실패?).")

    xs = np.concatenate([r[:, 0] for rs in proj.values() for r in rs])
    ys = np.concatenate([r[:, 1] for rs in proj.values() for r in rs])
    scale = VIEW_W / (xs.max() - xs.min())
    ox, oy = -xs.min() * scale, ys.max() * scale
    height = (ys.max() - ys.min()) * scale
    tol_proj = tol / EARTH_SPAN_KM * (xs.max() - xs.min())
    paths = {f: CT.svg_path(rs, scale, ox, oy, tol_proj) for f, rs in proj.items()}
    return paths, height


def color_tables(tab, fips_order):
    """정적 카운티 통계 1벌 + 수준별로 **실제로 달라지는 것만**.

    `n`·`n_unsolved`·`expected`·`smr`·`z`·`black_share`는 카운티의 성질이라
    min_n을 바꿔도 값이 같다. 달라지는 것은 **판정 대상인지**와 **q값/flag**뿐이다
    (BH FDR이 동시에 검정한 블록 수에 의존하므로). 처음에는 수준마다 전부 실었더니
    페이로드가 729KB로 예산 650KB를 넘었다 -- 같은 숫자를 세 번 보낸 탓이다.

    색은 **파이썬에서 확정**해 인덱스로 넘긴다. 구간 경계를 JS에 다시 구현하면
    범례와 그림이 어긋날 수 있고(map_figures에서 실제로 searchsorted 경계와 라벨이
    한 칸 밀렸다), 페이로드도 색 문자열보다 작은 정수가 낫다.
    """
    blue, red = CT.arms(5)
    palette = [CT.NEUTRAL] + blue + red          # 0=중립 회색, 1..5=파랑, 6..10=빨강
    levels = sorted(int(m) for m in tab["min_n"].unique())
    sig = tab["flag"] != ""
    # 발산 스케일은 모든 수준에서 **같은 vmax**를 써야 슬라이더를 움직여도 색의
    # 뜻이 안 바뀐다.
    vmax = float(np.nanmax(np.abs(tab.loc[sig, "z"]))) if sig.any() else 1.0
    edges = np.linspace(0, vmax, 6)[1:]

    # 정적 통계: 가장 낮은 수준(=가장 넓은 집합)에서 한 벌만 뽑는다.
    base = tab[tab["min_n"] == levels[0]].drop_duplicates("fips").set_index("fips")
    pri_rate = (base["n_priority"] / base["n"])
    pri_max = float(pri_rate.max() or 1.0)
    pri_edges = np.linspace(0, pri_max, 6)[1:]
    raw_edges = np.linspace(0, 1, 6)[1:]

    stat, res_static, pri_c, raw_c = [], [], [], []
    for f in fips_order:
        if f not in base.index:
            stat.append(None); res_static.append(-1); pri_c.append(-1); raw_c.append(-1)
            continue
        r = base.loc[f]
        stat.append([r["State"], r["City"], int(r["n"]), int(r["n_unsolved"]),
                     round(float(r["expected_unsolved"]), 1), round(float(r["smr"]), 3),
                     round(float(r["z"]), 2), round(float(r["black_share"]), 3),
                     int(r["n_priority"])])
        i = min(4, int(np.searchsorted(edges, abs(r["z"]))))
        res_static.append((6 + i) if r["z"] > 0 else (1 + i))
        pri_c.append(1 + min(4, int(np.searchsorted(pri_edges, r["n_priority"] / r["n"]))))
        raw_c.append(1 + min(4, int(np.searchsorted(raw_edges,
                                                    1 - r["n_unsolved"] / r["n"]))))

    # 수준별: 판정 여부(0=미판정 / 1=판정·비유의 / 2=판정·유의)와 q값만.
    per = {}
    for m in levels:
        sub = tab[tab["min_n"] == m].drop_duplicates("fips").set_index("fips")
        state, q = [], []
        for f in fips_order:
            if f not in sub.index:
                state.append(0); q.append(None)
                continue
            r = sub.loc[f]
            state.append(2 if r["flag"] != "" else 1)
            q.append(round(float(r["q_value"]), 4))
        per[m] = {"s": state, "q": q}
    static = {"stat": stat, "res": res_static, "pri": pri_c, "raw": raw_c}
    return palette, static, per, levels, vmax, edges


HTML = """<title>{title}</title>
<style>
:root{{--surface:{surface};--ink:{ink};--ink2:{ink2};--muted:{muted};
       --hair:{hair};--axis:{axis};}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--surface);color:var(--ink);
  font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 20px 64px}}
h1{{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--ink2);margin:0 0 22px;font-size:13px}}
.bar{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;
  padding:12px 14px;border:1px solid var(--hair);border-radius:10px;margin-bottom:14px}}
.bar label{{font-size:12px;color:var(--ink2);margin-right:6px}}
button{{font:inherit;font-size:12px;padding:5px 11px;border:1px solid var(--axis);
  background:transparent;color:var(--ink2);border-radius:999px;cursor:pointer}}
button[aria-pressed="true"]{{background:var(--ink);color:var(--surface);border-color:var(--ink)}}
input[type=range]{{vertical-align:middle}}
.mapwrap{{position:relative;border:1px solid var(--hair);border-radius:10px;
  overflow-x:auto;background:#fff}}
svg{{display:block;width:100%;height:auto}}
path{{stroke:#fff;stroke-width:.35;vector-effect:non-scaling-stroke}}
path.sig{{stroke:{ink};stroke-width:.9}}
#tip{{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;
  background:var(--ink);color:#fff;padding:9px 11px;border-radius:8px;
  font-size:12px;line-height:1.5;max-width:280px;z-index:9;box-shadow:0 6px 20px #0003}}
#tip b{{font-size:13px}}
.legend{{display:flex;flex-wrap:wrap;gap:20px;margin:14px 0 0;font-size:12px;
  color:var(--ink2);align-items:center}}
.sw{{display:inline-block;width:20px;height:11px;vertical-align:-1px;margin-right:5px;
  border:1px solid #0002}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:16px}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--hair);text-align:right;
  white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{cursor:pointer;color:var(--ink2);font-weight:600;position:sticky;top:0;
  background:var(--surface)}}
.tablewrap{{max-height:420px;overflow:auto;border:1px solid var(--hair);
  border-radius:10px;margin-top:8px}}
.notes{{margin-top:30px;border-top:1px solid var(--hair);padding-top:18px}}
.notes li{{margin-bottom:10px;color:var(--ink2)}}
.notes b{{color:var(--ink)}}
.foot{{margin-top:22px;font-size:12px;color:var(--muted)}}
</style>
<div class="wrap">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>

<div class="bar">
  <span><label>레이어</label>
    <button data-layer="res" aria-pressed="true">표준화 잔차 z</button>
    <button data-layer="pri" aria-pressed="false">재수사 우선순위 밀도</button>
    <button data-layer="raw" aria-pressed="false">원시 검거율</button>
  </span>
  <span><label>최소 표본</label>
    <input id="mn" type="range" min="0" max="{maxlv}" step="1" value="0">
    <b id="mnv"></b> <span style="color:var(--muted)">건 이상</span>
  </span>
  <span><button id="tbl" aria-pressed="false">표로 보기</button></span>
</div>

<div class="mapwrap">
  <svg viewBox="0 0 {vw:.0f} {vh:.0f}" role="img" aria-label="{title}">
    <defs><pattern id="na" width="5" height="5" patternUnits="userSpaceOnUse"
      patternTransform="rotate(45)">
      <rect width="5" height="5" fill="#fff"/>
      <line x1="0" y1="0" x2="0" y2="5" stroke="{axis}" stroke-width="1.4"/>
    </pattern></defs>
    <g id="g"></g>
  </svg>
</div>
<div class="legend" id="leg"></div>

<div id="tablewrap" class="tablewrap" hidden><table id="t">
<thead><tr><th>주</th><th>카운티</th><th>n</th><th>관측</th><th>기대</th>
<th>SMR</th><th>z</th><th>q</th><th>흑인비중</th></tr></thead><tbody></tbody>
</table></div>

<div id="tip"></div>

<div class="notes"><h2 style="font-size:15px;margin:0 0 12px">이 지도를 읽는 법</h2>
<ol>{caveats}</ol>
<p style="font-size:12px;color:var(--ink2)">{projnote}</p></div>
<p class="foot">{foot}</p>
</div>
<script>
// S = 카운티 정적 통계(수준과 무관), PL = 수준별로 달라지는 것만(판정상태·q).
const S={static}, PL={per}, P={palette}, PATHS={paths}, LV={levels}, FI={fips};
const g=document.getElementById('g'), tip=document.getElementById('tip');
let layer='res', li=0;
const nodes=FI.map(f=>{{
  const p=document.createElementNS('http://www.w3.org/2000/svg','path');
  p.setAttribute('d',PATHS[f]); g.appendChild(p); return p;
}});
// 이 수준에서 이 카운티가 어떤 상태인가: 0 미판정 / 1 판정·비유의 / 2 판정·유의
const st=i=>PL[LV[li]].s[i];
function paint(){{
  nodes.forEach((p,i)=>{{
    const s=st(i);
    let c;
    if(s===0) c=-1;                        // 미판정 -> 빗금
    else if(layer==='res') c = s===2 ? S.res[i] : 0;   // 유의한 것만 발산색
    else c = S[layer][i];                  // 기술 레이어는 판정 대상이면 색칠
    p.setAttribute('fill', c<0 ? 'url(#na)' : P[c]);
    // 색 단독으로 의미가 실리지 않도록 유의 카운티에 테두리를 함께 건다.
    p.classList.toggle('sig', layer==='res' && s===2);
  }});
  document.getElementById('mnv').textContent=LV[li];
  legend(); table();
}}
function legend(){{
  // 여기 한때 `const d=D[LV[li]];`가 있었다. D는 어디에도 정의된 적이 없어서
  // ReferenceError가 났고, paint()가 카운티를 칠한 **직후** 여기서 멈췄다.
  // 색은 칠해지므로 지도는 멀쩡해 보이는데 **범례와 표가 통째로 비어** 있었다.
  // 헤드리스 브라우저로 DOM을 조회해서야 잡혔다 -- 배포본 3장이 전부 그랬다.
  const L=document.getElementById('leg');
  const sw=c=>`<span class="sw" style="background:${{c}}"></span>`;
  let h='';
  if(layer==='res'){{
    h+='<span>미해결이 기대보다 <b>많음</b> '+[10,9,8,7,6].map(i=>sw(P[i])).join('')+'</span>';
    h+='<span>'+[1,2,3,4,5].map(i=>sw(P[i])).join('')+' 기대보다 <b>적음</b></span>';
    h+='<span>'+sw(P[0])+'판정했으나 유의하지 않음</span>';
  }} else {{
    h+='<span>낮음 '+[1,2,3,4,5].map(i=>sw(P[i])).join('')+' 높음</span>';
  }}
  h+='<span><span class="sw" style="background:#fff;background-image:'
   +'repeating-linear-gradient(45deg,'+'{axis}'+' 0 1.4px,#fff 1.4px 5px)"></span>'
   +'표본 부족(n &lt; '+LV[li]+')으로 판정 안 함</span>';
  L.innerHTML=h;
}}
function table(){{
  const b=document.querySelector('#t tbody'), q=PL[LV[li]].q;
  const rows=FI.map((f,i)=>st(i)?[S.stat[i], q[i]]:null).filter(Boolean)
    .sort((a,c)=>c[0][6]-a[0][6]);
  b.innerHTML=rows.map(([t,qq])=>`<tr><td>${{t[0]}}</td><td>${{t[1]}}</td><td>${{t[2]}}</td>
    <td>${{t[3]}}</td><td>${{t[4]}}</td><td>${{t[5]}}</td><td>${{t[6]}}</td>
    <td>${{qq}}</td><td>${{t[7]}}</td></tr>`).join('');
}}
g.addEventListener('mousemove',e=>{{
  const i=nodes.indexOf(e.target); const s=i<0?0:st(i);
  if(!s){{ tip.style.opacity=0; return; }}
  const t=S.stat[i], qq=PL[LV[li]].q[i];
  const verdict = s===1 ? '유의하지 않음'
                : (t[6]>0 ? '기대보다 많음(cold)' : '기대보다 적음(warm)');
  tip.innerHTML=`<b>${{t[1]}}, ${{t[0]}}</b><br>사건 ${{t[2]}}건 · 미해결 ${{t[3]}}건`
    +`<br>기대 ${{t[4]}} · SMR ${{t[5]}}<br>z ${{t[6]}} · q ${{qq}}`
    +`<br>${{verdict}}<br>흑인 피해자 비중 ${{t[7]}}`;
  tip.style.opacity=1;
  tip.style.left=Math.min(e.clientX+14, innerWidth-300)+'px';
  tip.style.top=(e.clientY+14)+'px';
}});
g.addEventListener('mouseleave',()=>tip.style.opacity=0);
document.querySelectorAll('[data-layer]').forEach(b=>b.onclick=()=>{{
  layer=b.dataset.layer;
  document.querySelectorAll('[data-layer]').forEach(o=>
    o.setAttribute('aria-pressed', o===b));
  paint();
}});
document.getElementById('mn').oninput=e=>{{ li=+e.target.value; paint(); }};
document.getElementById('tbl').onclick=e=>{{
  const w=document.getElementById('tablewrap'); w.hidden=!w.hidden;
  e.target.setAttribute('aria-pressed', !w.hidden);
}};
paint();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_key", default="county")
    ap.add_argument("--simplify_km", type=float, default=SIMPLIFY_KM,
                    help="지오메트리 단순화 허용오차(km). 0이면 단순화 안 함. "
                         "실측 페이로드: 0km 1.12MB / 1km 468KB / 2km 399KB")
    # 파일명에 스코프를 넣는다. 디렉터리 격리(outputs/{scope}/web/)만으로는 두 파일이
    # 똑같이 'map.html'이라, 파일 탐색기나 브라우저 탭에서 어느 쪽을 연 것인지
    # 알 수 없다 -- 실제로 3개 주 지도를 열고 "주가 3개뿐"이라고 읽는 일이 생겼다.
    ap.add_argument("--out", default=None,
                    help="출력 파일명(기본: map_{scope}.html)")
    ap.add_argument("--src", default="cold_blocks.csv",
                    help="블록 표 파일명. 크로스피팅 표(cold_blocks_cv5.csv)로 그리면 "
                         "카운티 커버리지가 853 -> 1,803개가 된다. **--out도 함께 "
                         "바꿔야** 기존 지도를 덮어쓰지 않는다.")
    args = ap.parse_args()

    tab = load_blocks(args.block_key, args.src)
    tab, missing = CT.join_fips(tab)
    matched = tab["fips"].notna()
    print(f"[join] FIPS {int(matched.sum())}/{len(tab)} ({matched.mean()*100:.1f}%)")
    if len(missing):
        print(f"[경고] 미매칭 {len(missing)}건 - clear.counties.ALIASES 확인:")
        print(missing.to_string(index=False))
    tab = tab[matched].copy()

    state_fips = set(tab["fips"].str[:2])
    paths, height = build_paths(state_fips, args.simplify_km)
    fips_order = sorted(paths)
    print(f"[geo] 주 {len(state_fips)}개 / 카운티 {len(fips_order):,}개, "
          f"단순화 {args.simplify_km}km, viewBox {VIEW_W:.0f}x{height:.0f}")

    palette, static, per, levels, vmax, edges = color_tables(tab, fips_order)
    assessed = {m: sum(1 for s in per[m]["s"] if s) for m in levels}
    sig = {m: sum(1 for s in per[m]["s"] if s == 2) for m in levels}
    print(f"[level] 판정 {assessed} / 유의 {sig}  (|z| 스케일 상한 {vmax:.2f})")

    # 분할은 --src에서 읽는다(파일명이 곧 분할이다 — detect_cold_blocks --out 규약).
    split = "crossfit" if re.search(r"_cv\d+", args.src) else "test"
    n_states = tab["fips"].str[:2].nunique()
    scope_label = (f"전국 {n_states}개 주" if C.SCOPE != C.DEFAULT_SCOPE
                   else "California · Texas · Michigan 3개 주")
    html = HTML.format(
        title=f"[{scope_label}] 설명되지 않는 미해결 집중 — 카운티별 표준화 잔차",
        subtitle=("각 카운티에서 관측된 미해결 건수를, 사건 구성과 주(州)를 통제한 "
                  "모델의 기대값과 비교한 값이다. 색이 칠해진 곳은 FDR 5%에서 "
                  "유의한 카운티뿐이다."),
        surface=CT.SURFACE, ink=CT.INK, ink2=CT.INK2, muted=CT.MUTED,
        hair=CT.HAIRLINE, axis=CT.AXIS,
        vw=VIEW_W, vh=height, maxlv=len(levels) - 1,
        caveats="".join(f"<li><b>{h}</b> {b}</li>"
                        for h, b in CAVEATS + [SPLIT_NOTES[split]]),
        projnote=PROJECTION_NOTE,
        foot=("출처: Murder Accountability Project / Kaggle Homicide Reports "
              "1980–2014 (CC BY-SA 4.0). 정제·집계 등 변경을 가했으며, 이 지도의 "
              "데이터 산출물도 동일 조건으로 배포한다. "
              "기대값은 공정성 완화(손실 벌점)를 거친 GraphSAGE 모델의 "
              f"{'out-of-fold' if split == 'crossfit' else 'test 집합'} 예측이며, "
              "전역 로짓 보정 후 간접 표준화(SMR)로 계산했다. "
              "재현 방법은 저장소의 CLAUDE.md 참고."),
        static=json.dumps(static, ensure_ascii=False, separators=(",", ":")),
        per=json.dumps(per, separators=(",", ":")),
        palette=json.dumps(palette), paths=json.dumps(paths, separators=(",", ":")),
        levels=json.dumps([int(m) for m in levels]),
        fips=json.dumps(fips_order, separators=(",", ":")),
    )

    # 이 스크립트에는 오랫동안 자체 검사가 없었다. 문서에 적혀 있던 "데이터 불변식
    # 13개"와 "node --check 통과"는 개발 중 손으로 한 번 돌린 것이지 배선된 검사가
    # 아니었고, 그 사이에 legend()의 ReferenceError가 지도 3장에 실려 나갔다.
    # 문법 검사만으로 그 결함을 잡지는 못하지만(DOM 조회의 몫이다), 최소한 문서가
    # 말하는 검사는 실제로 돌아야 한다.
    ok, msg = HC.node_check(html)
    print(msg)
    if not ok:
        raise SystemExit("[에러] 인라인 JS 문법 오류 - 지도를 쓰지 않는다.")

    out = C.scoped_output("web")
    out.mkdir(parents=True, exist_ok=True)
    p = out / (args.out or f"map_{C.SCOPE}.html")
    p.write_text(html, encoding="utf-8")
    kb = p.stat().st_size / 1024
    print(f"[save] {p}  ({kb:.0f} KB, 외부 요청 0)")
    if kb > 650:
        print("  [경고] 페이로드 예산 650KB 초과 - --simplify_km을 키울 것")


if __name__ == "__main__":
    main()
