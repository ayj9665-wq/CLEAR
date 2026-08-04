"""`src/`를 import 경로에 올린다.

실험 스크립트는 `cd src && python -m experiments.X`로 도는 규약이라 CWD가 곧
import 루트다(`config`·`clear`가 그렇게 풀린다). pytest는 저장소 루트에서 도는
편이 자연스러우므로, 그 차이를 여기서만 흡수한다 — 테스트마다 sys.path를 만지면
그 보일러플레이트가 파일 수만큼 늘어난다.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# **torch보다 먼저** config를 읽는다. config.py가 KMP_DUPLICATE_LIB_OK와
# TEMP/JOBLIB_TEMP_FOLDER를 import 전에 세팅하는데(환경 노트 참고), 테스트가
# torch를 먼저 import하면 이 기계에서는 OMP Error #15로 인터프리터가 죽는다 --
# 실제로 이 파일 없이 돌렸을 때 'Fatal Python error: Aborted'가 났다.
# clear/gnn.py가 `import config as C`를 torch 위에 두는 것과 같은 이유다.
import config as _config   # noqa: E402,F401
