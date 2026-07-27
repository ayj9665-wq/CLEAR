"""CLEAR 실험 스크립트 — 준비된 데이터를 읽어 outputs/에 결과를 쓰는 것들.

`src/`의 01~04(전처리 파이프라인)와 다른 점은 **순서가 없다**는 것이다. 여기
파일들은 전부 features.parquet + edges_*.npy를 읽고 outputs/에 쓴다. 서로
앞뒤가 없다 — diagnose_fairness 없이 mitigate_loss를 돌릴 수 있고,
mitigate_graph는 실패한 가지이지 mitigate_loss의 선행 단계가 아니다.

예전 파일명(`05_train_baseline.py` … `10_fairloss.py`)의 번호는 의존성이 아니라
**집필 순서**를 인코딩하고 있었고, 그게 두 가지 실질적 해를 끼쳤다.
(1) 숫자로 시작하는 이름은 import가 안 되므로(`import 05_train_baseline`은
문법 오류) 스크립트끼리 로직을 나눠 쓸 방법이 없었다 — 공유 코드가 전부
clear/로 밀려나거나, 그마저 안 되면 복붙됐다. (2) 없는 순서를 읽는 사람에게
암시했다.

옛 번호 ↔ 새 이름:
  05 -> train_baseline      06 -> train_gnn         07 -> diagnose_fairness
  08 -> mitigate_threshold  09 -> mitigate_graph    10 -> mitigate_loss
  ablation_sweep -> ablation

그 뒤 저장소를 **그래프 줄기만** 남기도록 좁히면서 05·08과 ablation·eda를
지웠다(커밋 4b128cf 이후). 결과는 전부 outputs/results.csv에 남아 있고 —
family=train의 logreg/xgboost 행, family=mitigate_threshold의 4,004행,
family=ablation의 480행 — 코드는 커밋 96d1dd1에서 되살릴 수 있다.
평면 모델의 test 예측 덤프 4개는 재생성이 불가능하므로 예외적으로 커밋했다
(outputs/predictions/{logreg,xgboost}[_blind].csv). 그래야 diagnose_fairness의
모델 대조표 — 이 프로젝트의 헤드라인 — 를 계속 만들 수 있다.

이제 전부 평범한 모듈이라 `python -m experiments.train_gnn`으로 실행하고
(`-m`이 CWD를 sys.path에 넣으므로 src/에서 실행하면 config·clear가 그대로
잡힌다), 필요하면 서로 import할 수도 있다.

01~04는 src/ 루트에 그대로 뒀다. 그쪽은 순서가 진짜이고(각 단계가 앞 단계
parquet를 읽는다), 하위 디렉터리로 내리면 숫자 때문에 `-m`을 못 써서 파일마다
sys.path 보정 코드를 넣어야 한다 — 이 리팩터가 없애려는 바로 그 종류의 반복이다.
"""
