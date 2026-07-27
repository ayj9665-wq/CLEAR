"""CLEAR 공용 라이브러리 — 여러 스크립트가 나눠 쓰는 로직의 단일 출처.

원래 이 패키지는 **제약 때문에** 생겼다. 실험 스크립트 이름이 숫자로 시작해서
(`05_train_baseline.py`) import가 불가능했고(`import 05_train_baseline`은 문법
오류), 그래서 공유 로직(데이터 로딩·분할·평가지표)이 스크립트마다 복붙됐다.
번호를 없앤 지금(experiments/__init__.py 참고) 그 제약은 사라졌지만 패키지는
남긴다 — 여기 있는 것들이 실제로 다중 호출자를 갖기 때문이다. 예: clear.metrics는
7곳, clear.data는 6곳, clear.predictions는 7곳이 쓴다. "스크립트 A가 B에서
import" 방식이면 실험들 사이에 방향 있는 사슬이 생기고, 완화 실험 하나를 돌리는
데 baseline의 XGBoost·GridSearchCV import가 딸려온다.

경계선: 여기 두는 것은 **둘 이상이 쓰는 것**이다. 한 스크립트만 쓰는 계산은 그
스크립트에 둔다.

src/에서 `from clear.data import load_xy, get_split` 식으로 쓴다.
"""
