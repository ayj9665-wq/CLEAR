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
