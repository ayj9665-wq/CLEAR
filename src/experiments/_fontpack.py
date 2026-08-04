"""
experiments/_fontpack.py -- NanumSquare를 굵기별로 서브셋해 base64로 심는다.

build_story_page 전용 보조 모듈. `clear/`에 올리지 않는 이유는 호출자가 하나뿐이기
때문이다(이 저장소의 규칙은 "호출자 2개 이상"). 그렇다고 build_story_page 본문에
두면 서브셋·인코딩·라이선스 처리가 서사 조립 코드와 뒤섞이므로 파일만 나눈다.

## 왜 CDN이 아니라 임베드인가

이 페이지의 1번 원칙은 "외부 요청 0"이다(build_web_map과 같은 규율). CDN <link>로
폰트를 부르면 네트워크가 없거나 막힌 환경에서 서체가 통째로 빠진다. 대신 **페이지
텍스트가 빌드 시점에 전부 확정**되므로 필요한 글리프를 정확히 셀 수 있다.

## 굵기별로 다른 문자 집합을 깎는다

본문 400은 전체 문자가 필요하지만 타이틀 800은 150자 남짓이면 된다. 공통 집합으로
세 벌을 깎으면 각 66KB(합 264KB)이고, 굵기별로 깎으면 합 ~172KB다. 그래서
호출자가 `{weight: set(chars)}`를 넘긴다 -- "어느 텍스트가 어느 굵기인가"를 빌드가
알아야 한다는 요구가 여기서 나온다.

## 조용히 실패하지 않는다

이 모듈의 실패 모드는 딱 하나, **글리프가 없어 두부(□)로 렌더되는 것**이고 그건
브라우저에서만 보인다. 그래서 커버리지 검사에서 걸리면 빌드를 세운다. 허용 목록에
있는 문자(≈, 州)만 시스템 폴백에 맡긴다 -- 브라우저는 글리프 단위로 폴백하므로
두부가 아니라 다른 서체로 렌더된다.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

# 굵기 3단. Light 300은 싣지 않는다 -- 어두운 배경에서 얇은 획은 헤일레이션으로
# 뭉개져 읽히지 않는다.
WEIGHTS = (400, 700, 800)

# 서브셋에 없어도 빌드를 세우지 않는 문자. 시스템 폴백으로 렌더된다.
#   ≈  : α≈50 한 곳
#   州 : 경고문 "주(州)를 통제한" -- 실제 페이지 내용이지만 한자 한 글자다
ALLOW_FALLBACK = frozenset("≈州")

# 마이너스 기호는 폴백에 맡기면 안 된다. `-0.0156` 같은 수치 문자열 한가운데
# 한 글자만 다른 서체가 되어 눈에 띈다. 그래서 정규화한다(폰트에 U+002D는 있다).
NORMALIZE = {"\u2212": "-", "\u2013": "-", "\u2014": "-"}

FAMILY = "NSQ"        # 로컬 설치본과 이름이 겹치지 않게 별도 패밀리명을 쓴다

# OFL 1.1 §3: Modified Version은 예약 이름(Reserved Font Name)을 "primary font
# name as presented to the users"로 쓸 수 없다. 서브셋은 글리프를 지우고 포맷을
# 바꾸므로 §DEFINITIONS가 정의하는 Modified Version이 맞다. 그래서 CSS 패밀리명
# (FAMILY)뿐 아니라 **바이너리 내부 name 테이블도** 중립 이름으로 바꾼다.
#
# 배포처 고지의 예약 목록에 NanumSquareRound / NanumSquareNeo 는 있고 우리가 쓰는
# NanumSquare 는 없지만, 목록 첫 항목이 "Nanum" 그 자체다. 해석을 다투는 것보다
# 이름을 바꾸는 편이 싸다.
SUBSET_NAME = "NSQ Subset"
SUBSET_PS = "NSQSubset"

# subset이 떨어뜨리는 nameID 13/14(라이선스)를 다시 넣는다. OFL §2가 허용하는
# "machine-readable metadata fields" 형태다. 저작권인 nameID 0은 subset이 보존하며
# 여기서도 건드리지 않는다 -- 그것이 §2가 사본마다 요구하는 고지다.
LICENSE_NOTE = ("Subset of NanumSquare_ac. Distributed by hangeul.naver.com "
                "under the SIL Open Font License, Version 1.1. "
                "Full text: OFL.txt in this repository.")
LICENSE_URL = "https://openfontlicense.org"


def normalize(text: str) -> str:
    """페이지에 들어가는 모든 문자열이 통과해야 하는 정규화."""
    for a, b in NORMALIZE.items():
        text = text.replace(a, b)
    return text


def _candidate_dirs(fonts_dir=None):
    if fonts_dir:
        yield Path(fonts_dir)
        return
    local = os.environ.get("LOCALAPPDATA")
    if local:
        yield Path(local) / "Microsoft/Windows/Fonts"
    yield Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    yield Path(__file__).resolve().parents[2] / "dataset/fonts"


def find_fonts(fonts_dir=None):
    """{굵기: 경로}. **파일명이 아니라 OS/2의 usWeightClass로 판정한다.**

    NanumSquare 배포본의 파일명은 굵기를 안 담는다(NanumSquare.ttf가 Bold,
    NanumSquare_0.ttf가 ExtraBold다). 파일명으로 매핑하면 다른 기계에서 조용히
    엉뚱한 굵기가 실린다.
    """
    from fontTools.ttLib import TTFont

    found = {}
    for d in _candidate_dirs(fonts_dir):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("NanumSquare*.ttf")):
            try:
                t = TTFont(p, lazy=True)
                w = int(t["OS/2"].usWeightClass)
                t.close()
            except Exception:
                continue
            if w in WEIGHTS and w not in found:
                found[w] = p
        if len(found) == len(WEIGHTS):
            break
    return found


def _relicense_names(font):
    """서브셋 바이너리의 이름 계열 레코드를 중립 이름으로 바꾸고 라이선스를 되넣는다.

    무엇을 바꾸고 무엇을 안 바꾸는지가 전부다:
      바꾼다  1/3/4/6/16/18/20/21/22 -- 사용자에게 보이는 이름 계열 (OFL §3)
      안 바꾼다 0 -- 저작권 고지. OFL §2가 사본마다 요구하는 바로 그것이다
      되넣는다 13/14 -- subset이 떨어뜨리는 라이선스 필드

    fontTools의 기본 name_IDs 는 [0..6]이라 13/14가 사라지는 것을 실측으로 확인했다.
    """
    nm = font["name"]
    w = int(font["OS/2"].usWeightClass)
    full = f"{SUBSET_NAME} {w}"
    ps = f"{SUBSET_PS}-{w}"
    new = {1: SUBSET_NAME, 3: f"{full}; subset", 4: full, 6: ps,
           16: SUBSET_NAME, 18: full, 20: ps, 21: SUBSET_NAME, 22: SUBSET_NAME}

    slots = set()
    for rec in list(nm.names):
        if rec.nameID == 0:
            slots.add((rec.platformID, rec.platEncID, rec.langID))
        if rec.nameID in new:
            nm.setName(new[rec.nameID], rec.nameID,
                       rec.platformID, rec.platEncID, rec.langID)
    for pid, eid, lid in slots:
        nm.setName(LICENSE_NOTE, 13, pid, eid, lid)
        nm.setName(LICENSE_URL, 14, pid, eid, lid)


def license_comment():
    """HTML 맨 위에 실을 OFL 고지 + 전문. 서체를 임베드하는 산출물은 필수다.

    OFL §2는 사본마다 저작권 고지와 **라이선스 전문**을 동봉하라고 요구하고,
    허용 형태로 "human-readable headers"를 명시한다. 링크가 아니라 임베드인 이유는
    이 산출물들의 설계 규율이 "외부 요청 0"이기 때문이다 -- 라이선스만 외부 의존으로
    두면 그 규율을 라이선스에서 깨는 셈이 된다.

    HTML 주석에 `--`가 연속으로 들어가는 것은 엄밀히는 비적합이지만, 주석을 실제로
    끝내는 시퀀스는 `-->` 하나뿐이라 파싱에는 영향이 없다. 그것만 검사한다.
    """
    p = Path(__file__).resolve().parents[2] / "OFL.txt"
    if not p.exists():
        raise SystemExit(
            f"[에러] {p} 가 없다. 서체를 임베드하는 산출물은 OFL 전문을 함께 "
            f"실어야 한다(OFL §2). --no_fonts 로 서체를 빼거나 OFL.txt를 둘 것.")
    text = p.read_text(encoding="utf-8").strip()
    if "-->" in text:
        raise SystemExit("[에러] OFL.txt 에 '-->' 가 있어 주석이 조기 종료된다.")
    return ("<!--\n"
            "이 파일에는 아래 서체의 서브셋이 base64로 포함되어 있다.\n"
            "SIL Open Font License 1.1 전문을 함께 싣는다(OFL 조항 2).\n\n"
            f"{text}\n-->\n")


def _subset(path, chars, hinting=True):
    """(bytes, flavor, 글리프수). woff2를 먼저 시도하고 brotli가 없으면 WOFF1."""
    from fontTools.subset import Options, Subsetter
    from fontTools.ttLib import TTFont

    font = TTFont(path)
    cmap = set(font.getBestCmap())
    keep = "".join(sorted(c for c in chars if ord(c) in cmap))

    opt = Options()
    opt.layout_features = ["*"]
    opt.hinting = hinting
    opt.desubroutinize = True
    opt.drop_tables += ["FFTM"]
    sub = Subsetter(options=opt)
    sub.populate(text=keep)
    sub.subset(font)
    _relicense_names(font)

    for flavor in ("woff2", "woff"):
        try:
            font.flavor = flavor
            buf = io.BytesIO()
            font.save(buf)
            return buf.getvalue(), flavor, len(keep)
        except Exception:
            continue                      # woff2는 brotli가 있어야 한다
    raise RuntimeError("폰트를 woff2/woff 어느 쪽으로도 저장하지 못했다")


def build(charsets, fonts_dir=None):
    """{굵기: set(chars)} -> (css, report).

    css는 @font-face 3개. 폰트를 못 찾으면 css=""를 돌려주고 호출자가 시스템
    스택으로 진행한다 -- 다만 **조용히 넘어가지 않도록** report에 이유를 담는다.
    """
    report = {"found": {}, "sizes": {}, "flavor": None,
              "missing_fonts": [], "fallback_chars": set(), "total_b64": 0}

    paths = find_fonts(fonts_dir)
    report["missing_fonts"] = [w for w in WEIGHTS if w not in paths]
    if report["missing_fonts"]:
        return "", report

    from fontTools.ttLib import TTFont

    blocks = []
    for w in WEIGHTS:
        chars = set(charsets.get(w, set()))
        chars |= set(chr(c) for c in range(32, 127))     # ASCII는 항상 넣는다
        cmap = set(TTFont(paths[w], lazy=True).getBestCmap())

        missing = {c for c in chars if ord(c) not in cmap}
        hard = missing - ALLOW_FALLBACK
        if hard:
            detail = ", ".join(f"U+{ord(c):04X}({c!r})" for c in sorted(hard))
            raise SystemExit(
                f"[에러] 굵기 {w} 서브셋에 없는 문자 {len(hard)}개: {detail}\n"
                f"  {paths[w].name} 의 cmap에 없다. 본문에서 바꾸거나 "
                f"_fontpack.ALLOW_FALLBACK 에 추가할 것 "
                f"(추가하면 그 글자만 시스템 폰트로 렌더된다).")
        report["fallback_chars"] |= missing & ALLOW_FALLBACK

        data, flavor, n = _subset(paths[w], chars)
        b64 = base64.b64encode(data).decode("ascii")
        report["found"][w] = paths[w].name
        report["sizes"][w] = (n, len(data), len(b64))
        report["flavor"] = flavor
        report["total_b64"] += len(b64)
        mime = "font/woff2" if flavor == "woff2" else "font/woff"
        blocks.append(
            f"@font-face{{font-family:{FAMILY};font-style:normal;"
            f"font-weight:{w};font-display:block;"
            f"src:url(data:{mime};base64,{b64}) format('{flavor}')}}")
    return "".join(blocks), report


def format_report(report):
    if report["missing_fonts"]:
        return ("[경고] NanumSquare 굵기 "
                + "/".join(str(w) for w in report["missing_fonts"])
                + " 를 찾지 못했다 -> 시스템 폰트 스택으로 빌드한다. "
                  "--fonts_dir 로 경로를 넘기거나 폰트를 설치할 것.")
    lines = [f"[font] NanumSquare {report['flavor']} 서브셋"]
    for w, (n, raw, b64) in report["sizes"].items():
        lines.append(f"  {w}: 글리프 {n:4d}  {raw/1024:6.1f}KB "
                     f"-> base64 {b64/1024:6.1f}KB  ({report['found'][w]})")
    lines.append(f"  합계 base64 {report['total_b64']/1024:.1f}KB")
    if report["fallback_chars"]:
        lines.append("  시스템 폴백 문자: "
                     + " ".join(sorted(report["fallback_chars"])))
    return "\n".join(lines)
