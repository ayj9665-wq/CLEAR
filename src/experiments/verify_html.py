"""
experiments/verify_html.py -- 배포 HTML을 실제로 실행해 DOM을 조회하고 판정한다.

    python -m experiments.verify_html                 # 있는 산출물 전부
    python -m experiments.verify_html --only map      # 지도만
    python -m experiments.verify_html --selftest      # 검사기 자체를 검증한다

## 왜 빌더 안이 아니라 별도 명령인가

대시보드의 계약은 `pandas + numpy` 만으로 빌드된다는 것이다(README 빠른 시작).
DOM 검사는 node와 jsdom을 요구하므로 빌드에 넣으면 그 계약이 깨진다. 그래서
분리하고, 대신 각 빌더가 끝에 "DOM 미검증"을 찍어 잊지 않게 한다.

## 단언은 실제로 터진 결함에서 역산했다

`build_web_map`의 `legend()`가 정의된 적 없는 `D`를 참조해 `ReferenceError`를 냈다.
`paint()`가 카운티를 다 칠한 **직후** 거기서 멈췄으므로 지도는 멀쩡해 보였고 범례와
순위표만 비어 있었다. 배포본 3장이 전부 그랬다. 여기서 나오는 요구사항이 셋이다.

1. **로드 중 uncaught error 0건.** 이것 하나로 그 결함이 잡힌다.
2. **침묵 실패한 자리에 내용 단언.** 범례 스와치 > 0, 표 행 > 0.
3. **첫 렌더가 아니라 조작 뒤에도.** `legend()`는 repaint마다 불리고 결함은 첫
   페인트 **뒤에** 났다. 그래서 슬라이더를 모든 수준으로 옮기고 레이어를 전부
   눌러 본 뒤 다시 단언한다. 이 항목이 없으면 검사가 결함을 다시 놓친다.

여기에 넷째를 더한다 -- **화면의 수와 CSV의 수가 같은가.** 표 행 수와 유의 카운티
수는 `cold_blocks*.csv`에서 직접 셀 수 있다. 그럴듯한 표가 틀린 수를 담는 것이
이 저장소가 반복해서 당한 실패다(`flag == ""`가 `NaN`에서 조용히 실패한 건, `std`
열이 비어 마진÷노이즈가 통째로 무의미했던 건).

## 커버리지 한계

jsdom은 `IntersectionObserver`를 구현하지 않고 `getBoundingClientRect`가 0을
돌려준다. 소개 페이지(`clear_story.html`)는 그 둘과 `requestAnimationFrame`에
의존하므로 **여기서 검사하지 않는다.** 실제 브라우저가 필요하고, 별도 항목이다.
대시보드는 브라우저 전용 API를 하나도 쓰지 않고 지도는 `createElementNS`만 쓰므로
jsdom으로 충분하다 -- 그리고 결함이 있던 곳이 지도다.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config as C

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).resolve().parent / "_domprobe.js"

# (라벨, HTML 경로, 종류, 근거 CSV) -- CSV가 None이면 수 대조를 건너뛴다.
def targets():
    web = C.scoped_output("web")
    out = C.scoped_output("")
    return [
        ("dashboard", ROOT / "outputs" / "dashboard.html", "dashboard", None),
        ("map (cv5)", web / "map_national.html", "map", out / "cold_blocks_cv5.csv"),
        ("map (test)", web / "map_national_test.html", "map", out / "cold_blocks.csv"),
        ("story", web / "clear_story.html", "story", out / "cold_blocks_cv5.csv"),
        ("story v2", web / "clear_story_v2.html", "story2",
         out / "cold_blocks_cv5.csv"),
    ]


def probe(html_path, kind):
    """jsdom에서 페이지를 실행하고 관측치를 돌려준다."""
    node = shutil.which("node")
    if node is None:
        raise SystemExit("[에러] node가 없다. DOM 검사는 node + jsdom이 필요하다.")
    if not (ROOT / "node_modules" / "jsdom").is_dir():
        raise SystemExit("[에러] jsdom이 없다. 저장소 루트에서 `npm install`.")
    r = subprocess.run([node, str(PROBE), str(html_path), kind],
                       capture_output=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(f"[에러] 조회 실패 ({html_path.name})\n{r.stderr[:1500]}")
    return json.loads(r.stdout)


def expected_from_csv(csv_path):
    """수준별 기대치를 CSV에서 직접 센다.

    지도가 그리는 것과 같은 정의여야 한다 -- 판정 = 그 수준에 행이 있는 카운티,
    유의 = flag가 cold/warm인 카운티. flag는 메모리에서 빈 문자열이지만 CSV에서
    읽으면 NaN이라, 이 저장소는 바로 그 지점에서 한 번 당했다(비유의 123개가
    신호로 칠해졌다). 그래서 빈 값을 명시적으로 채운다.

    CSV 행 수를 그대로 기대치로 쓰면 안 된다. 지도는 **지리로 그릴 수 있는 것만**
    그리므로, 두 종류를 나눠 돌려준다.

      unmappable -- FIPS가 없는 행. `City`가 `Repressed`(기관이 카운티명을 숨긴
                    자리)인 경우가 이에 해당한다. 지리가 없으니 지도에서 빠지는
                    것이 맞고, 결함이 아니다.
      collided   -- FIPS는 있는데 같은 FIPS를 가진 다른 행이 있어 지도의
                    drop_duplicates에서 조용히 버려지는 행. **이것은 데이터 유실
                    이다.** 별도로 보고한다.

    returns {level: (그릴 수 있는 수, 유의 수, unmappable 목록, collided 목록)}
    """
    from clear import counties as CT

    df = pd.read_csv(csv_path)
    df = df[df["block_key"] == "county"].copy()
    df["flag"] = df["flag"].fillna("")
    tab, _ = CT.join_fips(df)

    out = {}
    for lv, g in tab.groupby("min_n"):
        no_fips = g[g["fips"].isna()]
        ok = g.dropna(subset=["fips"])
        dup = ok[ok.duplicated("fips", keep=False)]
        drawn = ok.drop_duplicates("fips")
        out[int(lv)] = (
            len(drawn),
            int((drawn["flag"] != "").sum()),
            [f"{r.State}/{r.City}" for r in no_fips.itertuples()],
            [f"{r.State}/{r.City}(n={r.n}, z={r.z:+.2f}"
             f"{', ' + r.flag if r.flag else ''})" for r in dup.itertuples()],
        )
    return out


def check_map(label, obs, csv_path, errs):
    r, seen = obs["result"], obs["errors"]
    for e in seen:
        errs.append(f"{label}: 페이지가 예외를 던졌다 -- {e[:200]}")

    if not r.get("levels"):
        errs.append(f"{label}: 수준이 하나도 없다(슬라이더를 못 찾았나)")
        return

    exp = expected_from_csv(csv_path) if csv_path and csv_path.exists() else {}
    for lv in r["levels"]:
        tag = f"{label} n>={lv['level']}"
        if lv["legendSwatches"] == 0:
            errs.append(f"{tag}: 범례가 비어 있다 (legend()가 죽었을 때의 증상)")
        if lv["legendSpans"] == 0:
            errs.append(f"{tag}: 범례 항목이 없다")
        # 요소가 없는 것과 요소가 있는데 비어 있는 것은 다른 사건이다. 섞으면
        # 선택자 실수가 "표가 비었다"로 읽힌다 -- 실제로 소개 페이지에서 그랬다.
        if not lv.get("tableFound", True):
            errs.append(f"{tag}: 순위표 요소를 못 찾았다 (선택자가 페이지와 안 맞는다)")
        elif lv["tableRows"] == 0:
            errs.append(f"{tag}: 순위표가 비어 있다 (table()이 안 불렸을 때의 증상)")
        if lv["painted"] == 0:
            errs.append(f"{tag}: 그려진 카운티가 없다")
        if lv["hatched"] >= lv["painted"]:
            errs.append(f"{tag}: 전부 빗금이다(판정된 카운티가 없다)")

        # 수준 라벨은 #mnv의 텍스트다. 스크립트가 죽으면 빈 문자열이 되므로
        # **정수로 못 읽는 것 자체가 소견**이다 -- 여기서 예외를 던지면 검사기가
        # 결함을 보고하는 대신 크래시한다(이 저장소가 이미 두 번 당한 양상이다).
        try:
            key = int(lv["level"])
        except (TypeError, ValueError):
            errs.append(f"{tag}: 수준 표시가 비어 있다({lv['level']!r}) "
                        f"-- paint()가 끝까지 못 갔다")
            continue

        if key in exp:
            n_drawn, n_sig, unmappable, collided = exp[key]
            if lv["tableRows"] != n_drawn:
                errs.append(f"{tag}: 표 {lv['tableRows']}행인데 CSV에서 그릴 수 "
                            f"있는 것은 {n_drawn}개")
            if lv["sigPaths"] != n_sig:
                errs.append(f"{tag}: 유의 테두리 {lv['sigPaths']}개인데 CSV는 "
                            f"{n_sig}개")
            if collided:
                # 같은 FIPS를 가진 행이 둘 이상이면 지도의 drop_duplicates가 하나만
                # 남긴다. 남은 쪽이 어느 쪽이든 **다른 쪽의 사건은 지도에서 사라진다.**
                errs.append(f"{tag}: 같은 FIPS로 겹쳐 지도에서 유실되는 행이 있다 "
                            f"-> {' vs '.join(collided)}. 이름이 다른 같은 카운티가"
                            f" 별도 블록으로 검정된 것이므로 지도가 아니라 "
                            f"detect_cold_blocks 쪽에서 합쳐야 한다.")
            if unmappable:
                print(f"    [note] {tag}: 지리가 없어 지도에서 제외 "
                      f"{len(unmappable)}건 ({', '.join(unmappable)}) - 정상")
        elif exp:
            errs.append(f"{tag}: CSV에 이 수준이 없다")

    for name, ly in r.get("layers", {}).items():
        if ly["legendSwatches"] == 0:
            errs.append(f"{label} layer={name}: 범례가 비어 있다")
        if ly["tableRows"] == 0:
            errs.append(f"{label} layer={name}: 순위표가 비어 있다")

    tt = r.get("tableToggle", {})
    if tt.get("before") == tt.get("after"):
        errs.append(f"{label}: 표 접기 토글이 동작하지 않는다")


def check_story(label, obs, csv_path, errs):
    """소개 페이지. 지도 절은 같은 단언을 재사용하고, 서사 쪽 셋을 더한다."""
    r, seen = obs["result"], obs["errors"]
    for e in seen:
        errs.append(f"{label}: 페이지가 예외를 던졌다 -- {e[:200]}")

    # 스크립트가 끝까지 돌았는지의 카나리아. CSS가 `.js`가 붙었을 때만 본문을
    # 숨기므로, 이 클래스가 붙은 채 리빌이 안 되면 **페이지가 빈 화면**이 된다.
    if not r.get("jsClass"):
        errs.append(f"{label}: <html>에 .js가 안 붙었다 (스크립트가 초반에 죽었다)")
    if not r.get("sections"):
        errs.append(f"{label}: 서사 절을 하나도 못 찾았다")
    elif r["revealed"] != r["sections"]:
        errs.append(f"{label}: 절 {r['sections']}개 중 {r['revealed']}개만 드러났다 "
                    f"-- .js가 붙은 채 리빌이 안 되면 그 절은 **투명한 채로 남는다**")
    if not r.get("hero"):
        errs.append(f"{label}: hero 절이 없다")

    # 본문에 박힌 수치가 표시된 텍스트와 같은가. 계획서가 "검증 가능한 숫자는 CSV에서
    # 읽는다"고 정했으므로, 읽어 온 값과 화면에 찍힌 값이 갈리면 안 된다.
    if not r.get("counters"):
        errs.append(f"{label}: data-count 카운터가 하나도 없다")
    for c in r.get("counters", []):
        if c["want"] and c["want"] != c["text"]:
            errs.append(f"{label}: 카운터 불일치 data-count={c['want']!r} "
                        f"화면={c['text']!r}")

    check_map(label + " 지도", {"result": r.get("map", {}), "errors": []},
              csv_path, errs)



def check_story2(label, obs, csv_path, errs):
    """ver2. 서사 쪽 단언은 v1과 같고, ver2가 새로 약속한 넷을 더 본다.

    각 단언은 "이게 조용히 깨지면 화면에서 무슨 일이 일어나는가"에서 역산했다.
    아키텍처 탭이 죽으면 설명 패널이 첫 단계에 고정된 채로 남고, 접이식이 죽으면
    상세가 **영영 안 열린다**(v1과 달리 ver2는 근거를 접어 두므로, 그건 근거가
    사라진 것과 같다). 패널이 엉뚱한 값을 담는 경우가 가장 위험한데, 화면상으로는
    아무 문제도 없어 보이기 때문이다 -- 그래서 값을 직접 대조한다.
    """
    r, seen = obs["result"], obs["errors"]
    for e in seen:
        errs.append(f"{label}: 페이지가 예외를 던졌다 -- {e[:200]}")

    if not r.get("jsClass"):
        errs.append(f"{label}: <html>에 .js가 안 붙었다 (스크립트가 초반에 죽었다)")
    if not r.get("sections"):
        errs.append(f"{label}: 서사 절을 하나도 못 찾았다")
    elif r["revealed"] != r["sections"]:
        errs.append(f"{label}: 절 {r['sections']}개 중 {r['revealed']}개만 드러났다 "
                    f"-- .js가 붙은 채 리빌이 안 되면 그 절은 **투명한 채로 남는다**")
    if not r.get("hero"):
        errs.append(f"{label}: hero 절이 없다(ver2는 #hero2)")
    if not r.get("counters"):
        errs.append(f"{label}: data-count 카운터가 하나도 없다")
    for c in r.get("counters", []):
        if c["want"] and c["want"] != c["text"]:
            errs.append(f"{label}: 카운터 불일치 data-count={c['want']!r} "
                        f"화면={c['text']!r}")

    # --- 03 아키텍처 단계 선택 ---------------------------------------------
    a = r.get("arch", {})
    if a.get("tabs", 0) != 4 or a.get("panes", 0) != 4 or a.get("zones", 0) != 4:
        errs.append(f"{label}: 아키텍처 단계가 4/4/4가 아니다 "
                    f"(탭 {a.get('tabs')} · 설명 {a.get('panes')} · 강조 {a.get('zones')})")
    for p in a.get("picks", []):
        if p["selected"] != p["clicked"] or p["openPane"] != p["clicked"] \
                or p["zoneOn"] != p["clicked"]:
            errs.append(f"{label}: 단계 {p['clicked']}를 눌렀는데 "
                        f"선택={p['selected']} 설명={p['openPane']} 강조={p['zoneOn']}")
        if p["openPanes"] != 1:
            errs.append(f"{label}: 설명 패널이 동시에 {p['openPanes']}개 열렸다")

    # --- 접이식 상세 -------------------------------------------------------
    folds = r.get("folds", [])
    if not folds:
        errs.append(f"{label}: 접이식 상세가 하나도 없다")
    for f in folds:
        if not f["found"]:
            errs.append(f"{label}: 접이식 '{f['id']}'의 내용 영역이 없다")
        elif not (f["before"] is True and f["opened"] is False
                  and f["closed"] is True):
            errs.append(f"{label}: 접이식 '{f['id']}'가 안 열리거나 안 닫힌다 "
                        f"({f['before']} -> {f['opened']} -> {f['closed']})")
        if f["expanded"] != "true":
            errs.append(f"{label}: 접이식 '{f['id']}'의 aria-expanded가 "
                        f"{f['expanded']!r}")

    # --- 스크롤 진입 안내 ---------------------------------------------------
    sp = r.get("spotlight", {})
    if not sp.get("outlines"):
        errs.append(f"{label}: 지도 안내가 카운티를 하나도 표시하지 않았다")
    if not sp.get("caption"):
        errs.append(f"{label}: 지도 안내 설명이 비어 있다")
    if not sp.get("skip"):
        errs.append(f"{label}: 안내를 건너뛸 방법이 없다")

    # --- 상세 패널: 값까지 대조한다 -----------------------------------------
    p = r.get("panel", {})
    if not p.get("hintOnLoad"):
        errs.append(f"{label}: 선택 전 패널 안내 문구가 없다")
    if p.get("idx", -1) < 0:
        errs.append(f"{label}: 유의 판정된 카운티를 못 찾았다 -- 패널을 검사할 수 없다")
        return
    want, got = p["expect"], p.get("afterMapClick", {})
    if got.get("name") != f"{want['county']}, {want['state']}":
        errs.append(f"{label}: 패널 제목이 {got.get('name')!r}, "
                    f"기대는 {want['county']}, {want['state']}")
    # **자리까지 본다.** 값이 들어 있기만 하면 통과시키면, 사건 수 자리에 미해결
    # 수가 찍히는 종류의 결함이 그대로 빠져나간다 -- 두 값 다 화면에는 있다.
    order = [("n", "사건 수"), ("smr", "배수"), ("z", "확신도"),
             ("q", "헛짚을 확률"), ("black", "흑인 비중")]
    shown = got.get("stats") or []
    if len(shown) != len(order):
        errs.append(f"{label}: 패널 통계가 {len(shown)}개다({len(order)}개여야 한다)")
    for k, (key, label_ko) in enumerate(order):
        if k >= len(shown) or shown[k] != str(want[key]):
            errs.append(f"{label}: 패널 {k+1}번째({label_ko})가 "
                        f"{shown[k] if k < len(shown) else None!r}, "
                        f"기대는 {want[key]}")
    if got.get("bars") != 2:
        errs.append(f"{label}: 실제/기대 막대가 2개가 아니다({got.get('bars')})")
    if not got.get("verdict"):
        errs.append(f"{label}: 패널 해석 문장이 비어 있다")
    if got.get("outlines") != 1:
        errs.append(f"{label}: 선택 표시가 {got.get('outlines')}개다(1이어야 한다)")
    if not p.get("rowSelectedIsRight"):
        errs.append(f"{label}: 지도에서 고른 카운티가 표에서 선택되지 않았다")
    if p.get("scrolledToRow") != p["idx"]:
        errs.append(f"{label}: 지도 클릭이 표의 {p.get('scrolledToRow')}행으로 "
                    f"스크롤했다 -- 고른 것은 {p['idx']}행이다")

    # --- 지도 <-> 표 양방향 -------------------------------------------------
    if not p.get("hoverMapMarksRow"):
        errs.append(f"{label}: 지도 호버가 표 행을 강조하지 않는다")
    if not p.get("hoverRowMarksMap"):
        errs.append(f"{label}: 표 호버가 지도를 강조하지 않는다")
    rc = p.get("afterRowClick", {})
    if not rc.get("name"):
        errs.append(f"{label}: 표 행을 눌렀는데 패널이 안 열린다")

    # --- 키보드 -------------------------------------------------------------
    if p.get("firstTabIndex") != 0:
        errs.append(f"{label}: 표 첫 행이 탭 대상이 아니다"
                    f"(tabIndex={p.get('firstTabIndex')})")
    ad = p.get("afterArrowDown", {})
    if not (ad.get("first") == -1 and ad.get("second") == 0 and ad.get("focused")):
        errs.append(f"{label}: 아래 화살표로 행 이동이 안 된다({ad})")
    if not (p.get("afterEnter") or {}).get("name"):
        errs.append(f"{label}: Enter로 상세가 열리지 않는다")
    esc = p.get("afterEscape", {})
    if esc.get("name") or not esc.get("hint") or esc.get("outlines"):
        errs.append(f"{label}: Escape로 선택이 해제되지 않는다({esc})")

    check_map(label + " 지도", {"result": r.get("map", {}), "errors": []},
              csv_path, errs)


def check_dashboard(label, obs, errs):
    r, seen = obs["result"], obs["errors"]
    for e in seen:
        errs.append(f"{label}: 페이지가 예외를 던졌다 -- {e[:200]}")

    tabs = r.get("tabs", [])
    if len(tabs) < 2:
        errs.append(f"{label}: 스코프 탭이 {len(tabs)}개다(2개여야 한다)")

    vis = [p for p in r.get("onLoad", []) if not p["hidden"]]
    if len(vis) != 1:
        errs.append(f"{label}: 로드 직후 보이는 패널이 {len(vis)}개다(1개여야 한다)")

    for pane in r.get("onLoad", []):
        if pane["panels"] != 6:
            errs.append(f"{label} [{pane['scope']}]: 패널이 {pane['panels']}개다"
                        f"(6개여야 한다)")
        if not pane["tables"] or max(pane["tables"] or [0]) == 0:
            errs.append(f"{label} [{pane['scope']}]: 표가 전부 비어 있다")

    # 탭을 눌렀을 때 실제로 그 스코프만 보여야 한다.
    for scope, st in r.get("clicks", {}).items():
        shown = [p["scope"] for p in st["panes"] if not p["hidden"]]
        if shown != [scope]:
            errs.append(f"{label}: '{scope}' 탭을 눌렀는데 보이는 것이 {shown}다")


def selftest(errs):
    """검사기가 실제로 그 결함을 잡는지 확인한다.

    Phase 1에서 일부러 문법 오류를 심어 `node --check`를 검증한 것과 같은 이유다.
    잡는지 보지 않은 검사는 "적어 놨는데 안 도는" 상태와 구분되지 않는다.
    재현하는 결함은 실제로 배포됐던 그것이다 -- legend()가 정의된 적 없는 D를
    참조하게 되돌린다.
    """
    src = C.scoped_output("web") / "map_national.html"
    if not src.exists():
        print(f"[selftest] {src.name} 이 없어 건너뛴다")
        return
    html = src.read_text(encoding="utf-8")
    anchor = "const L=document.getElementById('leg');"
    if anchor not in html:
        errs.append("[selftest] 결함을 심을 지점을 못 찾았다 -- 검사기가 낡았다")
        return
    broken = html.replace(anchor, "const d=D[LV[li]];" + anchor, 1)

    tmp = ROOT / ".joblib_tmp" / "_selftest_map.html"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(broken, encoding="utf-8")
    try:
        obs = probe(tmp, "map")
        found = []
        check_map("selftest", obs, None, found)
        if not found:
            errs.append("[selftest] **결함을 심었는데 검사가 통과했다.** "
                        "이 검사기는 신뢰할 수 없다.")
        else:
            print(f"[selftest] 지도 통과 - 심어 둔 ReferenceError를 {len(found)}건으로 "
                  f"잡았다 (예: {found[0][:70]})")
    finally:
        tmp.unlink(missing_ok=True)

    _selftest_story(errs)
    _selftest_story2(errs)


def _selftest_story(errs):
    """소개 페이지 쪽 단언도 실제로 무는지 확인한다.

    여기서 심는 결함은 지도와 다르다. 소개 페이지의 치명적 실패 양상은 **스크립트가
    초반에 죽어 리빌이 안 도는 것**이다 -- `.js`는 붙었는데 절에 `in`이 안 붙으면
    CSS가 본문을 `opacity:0`으로 숨긴 채 남겨, 페이지가 **빈 화면**이 된다. 그래서
    스크립트 첫머리에 예외를 심어 그 상태를 만든다.
    """
    src = C.scoped_output("web") / "clear_story.html"
    if not src.exists():
        print("[selftest] clear_story.html 이 없어 서사 쪽 검증은 건너뛴다")
        return
    html = src.read_text(encoding="utf-8")
    anchor = "document.documentElement.classList.add('js');"
    if anchor not in html:
        errs.append("[selftest] 소개 페이지에 결함을 심을 지점을 못 찾았다 "
                    "-- 검사기가 낡았다")
        return
    broken = html.replace(anchor, anchor + "throw new Error('selftest');", 1)

    tmp = ROOT / ".joblib_tmp" / "_selftest_story.html"
    tmp.write_text(broken, encoding="utf-8")
    try:
        found = []
        check_story("selftest", probe(tmp, "story"), None, found)
        if not found:
            errs.append("[selftest] **소개 페이지에 결함을 심었는데 통과했다.**")
        else:
            print(f"[selftest] 소개 페이지 통과 - {len(found)}건으로 잡았다 "
                  f"(예: {found[0][:70]})")
    finally:
        tmp.unlink(missing_ok=True)



def _selftest_story2(errs):
    """ver2가 새로 약속한 것들에 대해서도 검사기가 무는지 확인한다.

    여기서 심는 결함은 v1과 또 다르다. ver2의 특징적 실패는 **패널이 열리기는
    하는데 값이 틀린 것**이다 -- 화면상 아무 이상이 없어 보이므로 눈으로는 절대
    안 잡힌다. 그래서 '사건 수' 자리에 '미해결 수'를 찍게 만든다. 둘 다 그 카운티의
    실제 값이라 어느 것도 이상해 보이지 않고, **자리까지 대조하는 단언만이** 잡는다.

    처음 심었던 결함(통계 배열 첨자를 하나 밀기)은 이 검사를 통과했다. 옆 첨자가
    비어 있는 카운티가 많아 `|| 원래값` 폴백에 먹혔기 때문이다 -- 검사기를 검증하지
    않았다면 "결함을 잡는다"고 믿은 채로 배포됐을 자리다.
    """
    src = C.scoped_output("web") / "clear_story_v2.html"
    if not src.exists():
        print("[selftest] clear_story_v2.html 이 없어 ver2 검증은 건너뛴다")
        return
    html = src.read_text(encoding="utf-8")
    anchor = "'<dt>'+L2.p_cases+'</dt><dd>'+t[2]+'</dd>'"
    if anchor not in html:
        errs.append("[selftest] ver2에 결함을 심을 지점을 못 찾았다 -- 검사기가 낡았다")
        return
    broken = html.replace(anchor, "'<dt>'+L2.p_cases+'</dt><dd>'+t[3]+'</dd>'", 1)

    tmp = ROOT / ".joblib_tmp" / "_selftest_story2.html"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(broken, encoding="utf-8")
    try:
        found = []
        check_story2("selftest", probe(tmp, "story2"), None, found)
        if not found:
            errs.append("[selftest] **ver2에 결함을 심었는데 통과했다.** "
                        "패널 값 대조가 실제로 물지 않는다.")
        else:
            print(f"[selftest] ver2 통과 - 자리를 바꾼 결함을 {len(found)}건으로 "
                  f"잡았다 (예: {found[0][:70]})")
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["map", "story", "story2", "dashboard"],
                    help="한 종류만 검사한다")
    ap.add_argument("--selftest", action="store_true",
                    help="결함을 일부러 심어 검사기가 잡는지 확인한다")
    ap.add_argument("--no_selftest", action="store_true",
                    help="자체 검증을 건너뛴다(기본은 항상 함께 돈다)")
    args = ap.parse_args()

    errs = []
    if not args.no_selftest:
        selftest(errs)
    if args.selftest:
        raise SystemExit(1 if errs else 0)

    n = 0
    for label, path, kind, csv_path in targets():
        if args.only and kind != args.only:
            continue
        if not path.exists():
            print(f"[skip] {label}: {path.name} 없음 - 먼저 빌드할 것")
            continue
        n += 1
        obs = probe(path, kind)
        before = len(errs)
        if kind == "map":
            check_map(label, obs, csv_path, errs)
        elif kind == "story":
            check_story(label, obs, csv_path, errs)
        elif kind == "story2":
            check_story2(label, obs, csv_path, errs)
        else:
            check_dashboard(label, obs, errs)
        mark = "OK " if len(errs) == before else "실패"
        print(f"[{mark}] {label}  ({path.name})")

    if not n:
        raise SystemExit("[에러] 검사할 산출물이 없다.")
    if errs:
        print(f"\n[검사] 실패 {len(errs)}건")
        for e in errs:
            print("  - " + e)
        raise SystemExit(1)
    print(f"\n[검사] 통과 - 산출물 {n}개, DOM 실행·조작·CSV 대조 모두 정상")


if __name__ == "__main__":
    main()
