"""HTML 검사기 자신을 검사한다.

검사기가 틀리면 그것이 지키는 산출물 전부가 무방비가 된다. 이 저장소는 그것을 이미
겪었다 -- 다섯 문서가 `node --check`를 "빌드 시 자동 검사"라고 적어 왔는데 **호출부가
저장소 어디에도 없었다.** 그러므로 여기서 보는 것은 "검사가 존재하는가"가 아니라
**"틀린 것을 실제로 틀렸다고 하는가"**다.

`node`가 없는 기계에서는 문법 검사 부분을 skip한다 -- 대시보드는 pandas+numpy만으로
빌드돼야 한다는 계약이 있어 node가 하드 의존이 아니고, 그 설계를 테스트가 뒤집으면 안
된다.
"""
import shutil

import pytest

from experiments import _htmlcheck as HC

needs_node = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node 없음 — 문법 검사는 선택적 의존이다")


# --- 어떤 <script>를 검사 대상으로 보는가 -----------------------------------

def test_inline_script_is_collected():
    assert HC.scripts("<script>var a=1;</script>") == ["var a=1;"]


def test_external_script_is_skipped():
    """src=는 본문이 없다. 외부 요청은 각 빌더의 URL 검사가 따로 잡는다."""
    assert HC.scripts('<script src="x.js"></script>') == []


def test_non_javascript_type_is_skipped():
    """`type="text/plain"` 블록은 실행되지 않으므로 문법 검사 대상이 아니다."""
    assert HC.scripts('<script type="text/plain">{{{ not js</script>') == []


def test_module_and_explicit_javascript_are_collected():
    assert len(HC.scripts('<script type="module">let a=1;</script>')) == 1
    assert len(HC.scripts('<script type="text/javascript">var a=1;</script>')) == 1


def test_multiple_blocks_are_all_collected():
    html = "<script>var a=1;</script><p>x</p><script>var b=2;</script>"
    assert len(HC.scripts(html)) == 2


# --- 실제로 틀린 것을 틀렸다고 하는가 ---------------------------------------

@needs_node
def test_valid_javascript_passes():
    ok, msg = HC.node_check("<script>function f(){return 1;} f();</script>")
    assert ok, msg


@needs_node
def test_syntax_error_fails_and_says_where():
    """이 단언이 이 파일의 존재 이유다."""
    ok, msg = HC.node_check("<script>\nvar a=1;\nfunction f({ return 1;}\n</script>")
    assert not ok, "문법 오류를 통과시켰다 — 검사기가 죽어 있다"
    assert "블록 0" in msg


@needs_node
def test_license_comment_does_not_break_the_check():
    """OFL 전문은 `-----` 를 포함한다. 주석이 조기 종료되면 안 된다."""
    ok, _ = HC.node_check("<!--\n-----\n-- in part or in whole --\n-->\n"
                          "<script>var a=1;</script>")
    assert ok


def test_no_inline_js_is_not_a_failure():
    """지도 이전 형태나 순수 문서 페이지도 있다."""
    ok, msg = HC.node_check("<p>본문뿐</p>")
    assert ok and "없음" in msg


@needs_node
def test_error_message_survives_a_non_ascii_temp_path():
    """오류 **보고**가 죽으면 안 된다.

    `subprocess(text=True)`가 로케일 코덱(cp949)으로 디코드하는 바람에, 비ASCII
    임시 경로에서 오류를 보고하려다 UnicodeDecodeError로 죽은 적이 있다. 이번 주
    두 번째로 '실패를 보고하는 코드에서 실패'가 난 사례였다.
    """
    ok, msg = HC.node_check("<script>function f({</script>")
    assert not ok
    assert isinstance(msg, str) and len(msg) > 0


# --- OFL 전문이 임베드 가능한 형태인가 --------------------------------------

def test_ofl_text_is_embeddable_as_an_html_comment(repo_root):
    """`-->`가 있으면 주석이 조기 종료돼 페이지가 깨진다. 라이선스는 필수 동봉이다."""
    ofl = repo_root / "OFL.txt"
    if not ofl.exists():
        pytest.skip("OFL.txt 없음")
    text = ofl.read_text(encoding="utf-8")
    assert "-->" not in text
    for section in ("PREAMBLE", "DEFINITIONS", "PERMISSION & CONDITIONS",
                    "TERMINATION", "DISCLAIMER"):
        assert section in text, f"OFL 전문에 {section} 절이 없다 — 전사가 깨졌다"
    assert "distributed under any other license" in text, (
        "조항 5에서 'under'가 빠졌다 — 웹에서 복사한 판본의 알려진 전사 오류다")
