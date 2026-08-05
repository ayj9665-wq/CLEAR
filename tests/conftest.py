"""pytest 공통 설정.

## 이 테스트 묶음이 무엇을 겨냥하는가

이 저장소가 실제로 겪은 사고는 **거의 전부 예외를 안 던지고 통과하는** 부류였다.

    --blind 무음 no-op            열이 안 지워졌는데 exit 0
    flag == "" 가 NaN에서 실패    비유의 카운티 123개를 신호로 칠함
    std 열이 비어 있음            마진÷노이즈가 통째로 무의미
    설정키 누락으로 그룹 병합     0.2620이라는 어디에도 없는 숫자
    적합 임계값을 identity에      재실행 한 번이 78행 중복
    legend()의 ReferenceError     배포본 3장의 범례가 빈 채 출고
    Dade / Miami-Dade             한 카운티가 두 블록으로 두 번 검정
    --out_suffix 누락             추적 중인 표를 부분집합으로 덮어씀

전부 **그럴듯한 결과가 나온다**는 공통점이 있다. 그래서 여기 있는 테스트는 성능이나
결론을 검증하지 않는다 — 위 목록이 재발했을 때 **시끄럽게 실패하는지**만 본다.

## 데이터가 없어도 도는 것과 아닌 것을 나눈다

`data/processed/` 와 `dataset/` 은 gitignore이고, 이 저장소의 계약은 "클론만 해도
결과를 볼 수 있다"이다(README 빠른 시작, `dashboard --verify_clone`). 그러므로
**기본 테스트는 원본 데이터 없이 통과해야 한다.** 데이터가 필요한 테스트는
`needs_data` 픽스처로 표시하고 없으면 skip한다 — 실패가 아니라 skip인 이유는,
데이터가 없는 것이 이 저장소에서는 정상 상태이기 때문이다.

실행: 저장소 루트에서 `pytest`
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# 스크립트들이 `cd src` 후 실행되는 규약이라(`python -m experiments.…`) config·clear가
# 최상위 모듈이다. 테스트는 루트에서 돌리므로 여기서 같은 경로를 만들어 준다.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture
def needs_data():
    """data/processed/ 가 있어야 하는 테스트용. 없으면 skip."""
    import config as C
    if not (C.SCOPE_DIR / "features.parquet").exists():
        pytest.skip("data/processed/ 없음 — 원본 데이터가 있는 기계에서만 도는 테스트")
    return C
