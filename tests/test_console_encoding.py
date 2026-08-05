"""콘솔 출력이 이 기계의 코덱에서 인코딩되는가.

## 왜 테스트로 만드는가

같은 사고가 **네 번** 났다.

    dashboard --verify_clone   성공 메시지의 em-dash가 cp949에서 죽음
    _htmlcheck                 node의 stderr를 로케일 코덱으로 디코드하다 죽음
                               (오류를 **보고하려다** 죽는다)
    verify_html                빈 수준 라벨이 int()에서 크래시
    returna_crosscheck         [rho] 안내의 em-dash가 cp949에서 죽음

세 번은 우연일 수 있어도 네 번은 규칙이다. 그리고 매번 **실패를 보고하는 경로**나
**성공을 알리는 경로** -- 즉 아무도 평소에 안 밟는 자리에서 났다. 그래서 실행이 아니라
소스에서 잡는다.

## 무엇을 검사하고 무엇을 검사하지 않는가

`print(...)` / `raise SystemExit(...)` 가 있는 **줄**이 이 기계의 선호 코덱으로
인코딩되는지만 본다. docstring·주석·HTML 문자열은 대상이 아니다 -- 파일 안에 머물거나
UTF-8로 저장되므로 콘솔 코덱과 무관하다. matplotlib 텍스트가 영문 전용이어야 하는 것과
같은 부류의 제약이고, 이 저장소는 그 규칙을 이미 문서로 갖고 있다.

코덱이 UTF-8인 기계에서는 무엇이든 통과하므로 **cp949로 명시해 검사한다** -- 개발
기계가 그것이고, 거기서 죽는 코드가 CI에서 통과하면 검사의 의미가 없다.
"""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# 콘솔로 나가는 호출. 이 줄들만 본다.
EMITS = re.compile(r"\bprint\s*\(|\braise\s+SystemExit\s*\(")

# 개발 기계의 콘솔 코덱. 여기서 죽으면 실제로 죽는다.
CONSOLE_CODEC = "cp949"


def _offenders(path):
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not EMITS.search(line):
            continue
        try:
            line.encode(CONSOLE_CODEC)
        except UnicodeEncodeError as e:
            bad = line[e.start:e.end]
            out.append((i, bad, line.strip()[:70]))
    return out


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")),
                         ids=lambda p: str(p.relative_to(SRC)))
def test_console_output_is_encodable(path):
    bad = _offenders(path)
    assert not bad, (
        f"{path.relative_to(SRC)}: 콘솔 출력에 {CONSOLE_CODEC}로 인코딩할 수 없는 "
        f"문자가 있다. 이 기계에서 그 줄이 실행되면 UnicodeEncodeError로 죽는다.\n"
        + "\n".join(f"  {ln}행 {ch!r} : {src}" for ln, ch, src in bad)
        + "\n  em-dash(—)는 하이픈(-)으로, 가운뎃점(·)은 슬래시(/)로 바꿀 것.")


def test_the_detector_actually_detects(tmp_path):
    """검사기가 무는지 확인한다 -- 안 그러면 '통과'가 아무 뜻이 없다."""
    p = tmp_path / "sample.py"
    p.write_text('print("정상")\nprint("깨짐 — em-dash")\n', encoding="utf-8")
    found = _offenders(p)
    assert len(found) == 1 and found[0][0] == 2


def test_comments_and_docstrings_are_not_flagged(tmp_path):
    """주석과 docstring은 콘솔로 안 나간다. 과잉 검출은 검사를 무력화한다."""
    p = tmp_path / "sample.py"
    p.write_text('"""문서 — em-dash가 있어도 된다."""\n# 주석 — 마찬가지\nx = 1\n',
                 encoding="utf-8")
    assert _offenders(p) == []
