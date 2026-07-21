"""
CLEAR 프로젝트 공통 설정.

모든 스크립트가 이 파일의 경로·상수를 import 해서 사용한다.
경로는 이 파일 위치(src/)를 기준으로 프로젝트 루트를 자동 계산한다.
"""
import os
from pathlib import Path

# ---- 경로 ----
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
RAW_CSV = ROOT / "dataset" / "kaggle_homicide_Reports_1980_2014.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- joblib/multiprocessing 임시 폴더 (Windows, 비ASCII 사용자명 대응) ----
# 기본 TEMP가 사용자명(예: 한글)을 포함하면, joblib의 resource_tracker가
# 'ascii' 코덱으로 경로를 encode하다 UnicodeEncodeError로 죽는다.
# n_jobs>1(GridSearchCV, 추후 GNN DataLoader 등)을 쓰는 모든 스크립트에
# 영향을 주므로, 이 파일에서 한 번만 ASCII 경로로 리다이렉트한다.
_JOBLIB_TMP = ROOT / ".joblib_tmp"
_JOBLIB_TMP.mkdir(exist_ok=True)
os.environ["TEMP"] = str(_JOBLIB_TMP)
os.environ["TMP"] = str(_JOBLIB_TMP)
os.environ["JOBLIB_TEMP_FOLDER"] = str(_JOBLIB_TMP)

# ---- 타깃 ----
TARGET = "Crime Solved"          # 원본 열 이름
TARGET_BIN = "solved"            # 인코딩된 이진 타깃(1=검거, 0=미해결)

# ---- 타깃 누수(leakage) 열: 반드시 제거 ----
# 미해결 사건에서 90~99% Unknown → 사실상 정답 누설
LEAKAGE_COLS = [
    "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race",
    "Perpetrator Ethnicity", "Perpetrator Count", "Relationship",
]

# ---- 저정보·식별자 열: 학습에서 제외 ----
DROP_COLS = [
    "Record ID", "Agency Code", "Agency Name", "Record Source",
    "Crime Type",   # 98.6% 단일값
    "Incident",     # 관할·월 내 일련번호, 의미 없음
]

# ---- 민감속성(공정성 진단용) ----
SENSITIVE_COLS = ["Victim Race", "Victim Sex"]

# ---- 표본 추출: 논문(Campedelli 2022) 주(州)별 분할 전략 ----
# 논문이 실제로 분석한 3개 주(California·Texas·Michigan) 조합.
# 전국 풀링 대신 이 조합을 쓰는 이유: 주별 검거율 편차가 커서(예: SC 90.8% vs
# NY 54.1%) 전국을 그대로 합치면 인종 격차 분석이 심슨의 역설에 걸릴 위험이
# 있다. 지역·인종 다양성은 확보하면서 논문과 비교 가능한 조합. None 이면 전체 사용.
SAMPLE_STATES = ["California", "Texas", "Michigan"]
# 표본 상한(메모리·학습속도). None 이면 주 전체 사용.
SAMPLE_MAX_ROWS = None

# ---- 특성 공학 ----
AGE_BIN_WIDTH = 5          # 논문: 나이를 5년 구간화 후 One-Hot
AGE_UNKNOWN_CODE = 998     # MAP 미상 코드
CATEGORICAL_COLS = [
    "Victim Sex", "Victim Race", "Victim Ethnicity",
    "Weapon", "Month", "Agency Type", "State",
]
NUMERIC_COLS = ["Victim Count"]   # Perpetrator Count 는 누수라 제외

# ---- 분할·재현성 ----
RANDOM_STATE = 42
TEST_SIZE = 0.30           # 논문: 70/30
CV_FOLDS = 5              # 논문: 5-fold 층화 CV

# ---- 그래프 구성 (04_build_graph.py) ----
GRAPH_DIR = PROCESSED_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# 노드당 신규 이웃 상한(차수 상한). 대칭화 후 실제 차수는 k~2k 사이.
K_NEIGHBORS = 10
# 엣지 후보 3종의 블로킹 키. Agency Code/Name은 01_clean.py에서 이미 제거돼
# City가 가장 세밀한 지리 단위. 같은 블록 안에서는 더 세밀한 유사도 기준이
# 없으므로, 실제 거리 계산 대신 "정확 일치 블로킹 + 블록 내 k개 결정적 선택"
# 방식을 3종 모두에 동일하게 적용한다(자세한 근거는 개발계획서·플랜 참고).
GEO_BLOCK_COLS = ["State", "City"]
TEMPORAL_BLOCK_COLS = ["Year", "Month"]
WEAPON_BLOCK_COLS = ["Weapon", "Victim Sex", "Victim Race"]
