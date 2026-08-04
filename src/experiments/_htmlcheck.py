"""
experiments/_htmlcheck.py -- 배포용 HTML의 인라인 JS를 문법 검사한다.

## 이 검사가 무엇을 못 하는지부터 적는다

`node --check`는 **문법만** 본다. 이 저장소가 실제로 배포해 버린 유일한 JS 결함은
문법적으로 완벽히 유효한 코드였다 -- `build_web_map`의 `legend()`가 어디에도 정의된
적 없는 `D`를 참조해 호출마다 `ReferenceError`를 냈고, `paint()`가 카운티를 다 칠한
**직후** 거기서 멈췄다. 그래서 지도는 멀쩡해 보이는데 **범례와 순위표가 통째로 비어**
있었고, 배포한 3장이 전부 그랬다. 헤드리스 브라우저로 DOM을 조회해서야 잡혔다.

그러므로 이 모듈은 그 구멍을 메우지 않는다. 그건 DOM 검사의 몫이다.

## 그럼에도 넣는 이유

**문서 다섯 곳이 "빌드 시 node --check가 돈다"고 적어 왔는데 호출부가 0건이었다.**
개발 중에 손으로 한 번 돌린 것이 문서에는 자동 검사로 적혔다. 문서와 현실이 어긋난
채로 두는 것이 이 저장소가 가장 경계하는 실패이고(`--blind` 무음 no-op이 같은
부류다), 그것을 assert로 바꾸는 것이 여기서 하는 일이다. 오타와 괄호 짝이 덤으로
잡힌다.

## node가 없으면 실패시키지 않는다

대시보드는 `pandas + numpy`만으로 돌아야 한다는 계약이 있다(README 빠른 시작,
`requirements-dashboard.txt`). node를 하드 의존으로 만들면 그 계약이 깨지므로 없으면
건너뛴다. 다만 **건너뛴 사실을 반드시 출력한다** -- 조용히 통과시키면 지금 고치고
있는 상태가 그대로 되돌아온다. 없어서 건너뛴 것과 있어서 통과한 것은 다른 사건이다.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# <script ...>본문</script>. 이 저장소 산출물은 블록이 하나뿐이지만, 늘어나도
# 동작하도록 전부 순회한다.
_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)

# 실행되지 않는 블록은 검사 대상이 아니다. type이 없거나 JS 계열일 때만 본다.
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}


def scripts(html: str) -> list[str]:
    """실행되는 인라인 JS 본문만 돌려준다.

    `src=`가 있는 블록은 외부 파일이라 본문이 없고, 애초에 이 저장소 산출물에
    있으면 안 된다(외부 요청 0). 그건 각 빌더의 URL 검사가 먼저 잡으므로 여기서는
    조용히 건너뛴다.
    """
    out = []
    for attrs, body in _SCRIPT.findall(html):
        low = attrs.lower()
        if "src=" in low:
            continue
        m = re.search(r"type\s*=\s*[\"']?([^\"'\s>]*)", low)
        if m and m.group(1) not in _JS_TYPES:
            continue
        if body.strip():
            out.append(body)
    return out


def node_check(html: str) -> tuple[bool, str]:
    """(ok, 출력할 한 줄). node가 없으면 (True, "건너뜀") 이다.

    ok=False는 **문법 오류가 실재한다**는 뜻이며, 호출자는 빌드를 실패시켜야 한다.
    """
    blocks = scripts(html)
    if not blocks:
        return True, "[검사] 인라인 JS 없음 - 문법 검사 생략"

    exe = shutil.which("node")
    if exe is None:
        return True, ("[검사] node 없음 - JS 문법 미검사(설치하면 빌드가 자동으로 "
                      "검사한다). 산출물 자체는 정상이다.")

    # text=True 는 로케일 코덱(이 기계에서는 cp949)으로 디코드한다. node는 UTF-8로
    # 출력하고, 오류 메시지에는 임시 파일 경로가 실린다 -- 이 기계의 프로필 경로가
    # 비ASCII라 그대로 두면 **오류를 보고하려다 디코드에서 죽는다**. 실제로 그랬다.
    _dec = dict(capture_output=True, encoding="utf-8", errors="replace")

    ver = subprocess.run([exe, "--version"], **_dec).stdout.strip()
    errs = []
    with tempfile.TemporaryDirectory() as d:
        for i, body in enumerate(blocks):
            # 임시 파일로 검사한다. node가 내는 줄 번호는 **그 블록 안에서의**
            # 줄 번호이므로, 블록 번호와 함께 읽어야 위치를 찾을 수 있다.
            p = Path(d) / f"block{i}.js"
            p.write_text(body, encoding="utf-8")
            r = subprocess.run([exe, "--check", str(p)], **_dec)
            if r.returncode != 0:
                msg = (r.stderr or r.stdout).strip().splitlines()
                errs.append(f"블록 {i}: " + " / ".join(msg[:4])[:500])

    if errs:
        return False, ("[검사] JS 문법 오류 " + str(len(errs)) + "건 (node " + ver
                       + ")\n  " + "\n  ".join(errs)
                       + "\n  줄 번호는 해당 <script> 블록 안에서의 위치다.")
    return True, (f"[검사] JS 문법 OK (node {ver}, 블록 {len(blocks)}개) - 문법만 "
                  f"본다. DOM 동작은 `python -m experiments.verify_html` 로 따로 "
                  f"검사할 것.")
