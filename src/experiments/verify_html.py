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
        if lv["tableRows"] == 0:
            errs.append(f"{tag}: 순위표가 비어 있다 (table()이 안 불렸을 때의 증상)")
        if lv["painted"] == 0:
            errs.append(f"{tag}: 그려진 카운티가 없다")
        if lv["hatched"] >= lv["painted"]:
            errs.append(f"{tag}: 전부 빗금이다(판정된 카운티가 없다)")

        key = int(lv["level"])
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
            print(f"[selftest] 통과 - 심어 둔 ReferenceError를 {len(found)}건으로 "
                  f"잡았다 (예: {found[0][:70]})")
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["map", "dashboard"],
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
