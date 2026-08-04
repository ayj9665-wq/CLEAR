"""
CLEAR 프로젝트 공통 설정.

모든 스크립트가 이 파일의 경로·상수를 import 해서 사용한다.
경로는 이 파일 위치(src/)를 기준으로 프로젝트 루트를 자동 계산한다.
"""
import os
from pathlib import Path

# ---- 분석 스코프 (3개 주 / 전국) ----
# 전국 확장(확장설계서 §10-0)은 파이프라인을 처음부터 다시 돌리는데, 산출물 경로에
# 스코프 개념이 없으면 전국 실행이 3개 주 산출물을 **덮어쓴다**. 단순한 파일 손실이
# 아니라 조용한 오염이 문제다:
#
#   outputs/predictions/*.csv의 row_index는 features.parquet의 **행 위치**이고
#   diagnose_fairness는 그걸로 load_sensitive()를 iloc한다. features를 전국으로
#   갈아끼우면 커밋된 평면 모델 덤프 4개가 엉뚱한 행의 민감속성과 조인되면서도
#   **에러 없이 그럴듯한 표**를 낸다(assert_same_test_set은 덤프끼리만 비교하므로
#   통과한다). 헤드라인 graphsage_geo_blind - xgboost_blind = +0.82가 조용히
#   무효가 되는 경로이며, _assert_blind가 막으려던 것과 같은 종류의 실패다.
#
# 스위치는 환경변수 하나뿐이다. SAMPLE_STATES를 여기서 유도하는 이유가 그것 —
# 스코프와 주 목록을 따로 두면 둘이 어긋난 채로 돌아간다.
DEFAULT_SCOPE = "ca_tx_mi"
SCOPES = {
    "ca_tx_mi": ["California", "Texas", "Michigan"],
    "national": None,                    # None = 주 필터 없음(전국 638,454행)
}
SCOPE = os.environ.get("CLEAR_SCOPE") or DEFAULT_SCOPE
if SCOPE not in SCOPES:
    raise ValueError(
        f"CLEAR_SCOPE={SCOPE!r}는 모르는 스코프다. 가능한 값: {sorted(SCOPES)}. "
        f"오타를 그냥 통과시키면 새 빈 스코프 디렉터리가 생기고, 거기서 나온 결과가 "
        f"어느 표본의 것인지 알 수 없게 된다.")

# ---- 경로 ----
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
RAW_CSV = ROOT / "dataset" / "kaggle_homicide_Reports_1980_2014.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 스코프별 중간 산출물(sample/features/graph)이 사는 곳.
#
# **기본 스코프에서는 PROCESSED_DIR과 같은 디렉터리다** — 3개 주 산출물은 자리를
# 안 옮기므로 마이그레이션이 0이고, 기존 실행 경로가 한 글자도 안 바뀐다.
#
# clean.parquet은 여기 들어오지 않고 PROCESSED_DIR 루트에 남는다: 01_clean.py는
# 주 필터가 없어 산출물이 이미 전국이고 두 스코프가 공유한다. 이것까지 스코프
# 안으로 넣으면 전국 실행이 존재하지도 않는 processed/national/clean.parquet을 찾는다.
SCOPE_DIR = PROCESSED_DIR if SCOPE == DEFAULT_SCOPE else PROCESSED_DIR / SCOPE
SCOPE_DIR.mkdir(parents=True, exist_ok=True)


def scoped_output(name=None):
    """스코프별 결과 파일 경로. name이 없으면 디렉터리 자체.
    기본 스코프면 outputs/ 바로 아래(기존과 동일).

    cold_blocks.csv·fairness_*.csv·edge_homophily.csv 등은 tracked 실험 기록이라
    전국 실행이 덮어쓰면 그대로 손실이다. results.csv만은 예외로 스코프를 안 쪼갠다 --
    long format이고 scope가 params에 들어가므로 한 파일에서 groupby로 갈린다
    ("지표 추가는 열이 아니라 행"의 스코프 버전).
    """
    d = OUTPUT_DIR if SCOPE == DEFAULT_SCOPE else OUTPUT_DIR / SCOPE
    d.mkdir(parents=True, exist_ok=True)
    return d if name is None else d / name


def scope_param():
    """results.csv의 params에 넣을 스코프 조각. **기본 스코프면 빈 dict.**

    GNN_EDGE_MODE가 문자열 "shuffle"이 아니라 None이어야 했던 것과 같은 이유다 --
    기본값이 params에 안 들어가야 기존 행의 JSON이 한 글자도 안 바뀌고, identity
    KEY가 유지되고, 재실행이 중복이 아니라 교체로 남는다. 기본 스코프에서 {"scope":
    "ca_tx_mi"}를 넣으면 과거 행 전부와 KEY가 갈려 results.csv가 두 배로 부푼다.
    """
    return {} if SCOPE == DEFAULT_SCOPE else {"scope": SCOPE}

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

# ---- OpenMP DLL 충돌 회피 (Windows, torch/torch_geometric) ----
# libiomp5md.dll 중복 로드로 torch import 시 OMP Error #15 발생.
# 04_build_graph.py는 torch를 안 써서 미뤘고, experiments/train_gnn.py부터 torch를
# 처음 쓰므로 여기서 처리한다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

# ---- blind 실험에서 X에서 뺄 민감속성 더미 열 ----
# 주의: 03_features.py는 CATEGORICAL_COLS를 전면 원핫하고 그 목록에 Victim
# Race/Sex가 들어 있다. 즉 민감속성은 sens__ 사본과 **별개로** 'Victim Race=Black'
# 같은 더미 열로 X에 남아 있고, load_xy()는 sens__ 사본만 뗀다 — 기본 설정의 세
# 모델은 전부 인종·성별을 직접 보고 학습한다. load_xy(blind=True)가 이 접두어로
# 시작하는 더미를 마저 뺀다.
#
# Victim Ethnicity를 포함하는 이유: Hispanic 여부는 인종의 강한 직접 프록시라,
# 남겨두면 "모델이 인종을 못 본다"는 blind 조건이 성립하지 않는다. 그래프가
# 인종 정보의 우회 경로인지 검정하려면 그래프 아닌 경로를 다 막아야 한다.
SENSITIVE_FEATURE_COLS = ["Victim Race", "Victim Sex", "Victim Ethnicity"]

# ---- 표본 추출: 논문(Campedelli 2022) 주(州)별 분할 전략 ----
# 논문이 실제로 분석한 3개 주(California·Texas·Michigan) 조합.
# 전국 풀링 대신 이 조합을 쓰는 이유: 주별 검거율 편차가 커서(예: SC 90.8% vs
# NY 54.1%) 전국을 그대로 합치면 인종 격차 분석이 심슨의 역설에 걸릴 위험이
# 있다. 지역·인종 다양성은 확보하면서 논문과 비교 가능한 조합. None 이면 전체 사용.
#
# 값은 SCOPE에서 유도한다(위 SCOPES 표) -- 주 목록과 산출물 경로가 한 스위치에
# 묶여 있어야, 전국 데이터를 3개 주 경로에 쓰는 식으로 어긋나지 않는다.
SAMPLE_STATES = SCOPES[SCOPE]
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
GRAPH_DIR = SCOPE_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# 노드당 신규 이웃 상한(차수 상한). 대칭화 후 실제 차수는 k~2k 사이.
# k=5/10/20/30 ablation 결과 k=20이 근소 최선(balanced_accuracy 0.6511 @
# lr=0.01, lr=0.005와 결합 시 0.6525)이라 기본값으로 승격. k=5만 확실히
# 나쁘고 10~30은 평평함 — 데이터가 커지면(예: 전국 확장) 재검토 필요.
K_NEIGHBORS = 20
# 엣지 후보 3종의 블로킹 키. Agency Code/Name은 01_clean.py에서 이미 제거돼
# City가 가장 세밀한 지리 단위. 같은 블록 안에서는 더 세밀한 유사도 기준이
# 없으므로, 실제 거리 계산 대신 "정확 일치 블로킹 + 블록 내 k개 결정적 선택"
# 방식을 3종 모두에 동일하게 적용한다(자세한 근거는 개발계획서·플랜 참고).
GEO_BLOCK_COLS = ["State", "City"]
TEMPORAL_BLOCK_COLS = ["Year", "Month"]
WEAPON_BLOCK_COLS = ["Weapon", "Victim Sex", "Victim Race"]

# ---- GNN 학습 (experiments/train_gnn.py) 기본 하이퍼파라미터 ----
# argparse로 개별 오버라이드 가능(ablation study용). 인자 없이 실행하면
# 이 값들을 그대로 쓴다.
# geo(State+City 블로킹)가 3종 중 유일하게 베이스라인을 4개 지표 모두에서
# 이겨서 임시로 기본값 확정. temporal/weapon은 --edge_type으로 여전히 실행
# 가능 — 데이터가 바뀌면(예: ablation으로 다른 후보가 역전) 재검토.
GNN_DEFAULT_EDGE_TYPE = "geo"
# 그래프 구성 방식. None = 04_build_graph.py의 기본(블록 내 셔플-링 무작위 페어링).
# "rank_onehot"/"rank_ordinal" 등은 04를 --rank로 돌려 만든 유사도 랭킹 그래프.
#
# **기본값이 None이어야 하는 이유**(문자열 "shuffle"이 아니라): clear.results의
# RUN_PARAMS에 edge_mode가 들어가는데 from_run이 None인 키를 params에서 빼므로,
# 기존 셔플 그래프 결과의 params JSON이 한 글자도 안 바뀐다 -> 결과 identity KEY가
# 유지되고 재실행이 '교체'로 남는다. 기본값을 문자열로 두면 옛 행과 params가 갈려
# 재실행이 중복 행을 만든다(통합 직후 78행이 겹쳤던 그 사고).
GNN_EDGE_MODE = None
GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 2
GNN_DROPOUT = 0.3
# one-factor-at-a-time 스윕에서 lr=0.005가 기본값(0.01)보다 근소하게 나아
# 승격(k=20과 결합 시 balanced_accuracy 0.6525, 최고 기록). num_layers=1과
# aggr=max는 확실히 나빠서(그래프 깊이·mean 집계가 중요) 그대로 기본 유지.
GNN_LR = 0.005
GNN_WEIGHT_DECAY = 5e-4
GNN_AGGR = "mean"
GNN_MAX_EPOCHS = 200
GNN_PATIENCE = 20
GNN_VAL_SIZE = 0.15        # train+val 풀 중 val 비율
# seed 반복: split은 RANDOM_STATE로 고정하고 torch seed만 바꿔 GNN 학습 분산을
# 측정한다(GPU scatter 집계 비결정성 + 초기화 분산). 한 config를 이 seed들로
# 돌려 mean±std를 남긴다 — 임계값 의존 지표(F1/Sens/Spec)가 run마다 ±3~4점
# 흔들려서, 단일 run 비교는 신뢰할 수 없기 때문. baseline은 결정적이라 1행.
GNN_SEEDS = [42, 43, 44]

# ---- 미니배치 학습 (NeighborLoader) ----
# 전국(방향 엣지 25.0M)은 full-batch로 GPU에 안 올라간다(레이어당 6.5GB, 2층+역전파
# 약 20GB vs 12.9GB). GraphSAGE는 애초에 이웃 샘플링 미니배치용으로 설계된 모델이라
# 전국 규모에서는 미니배치가 오히려 정통 용법이고, full-batch를 쓰던 것은 표본이
# 작아서였을 뿐이다.
#
# **기본값은 False다.** 발표된 결과 전부가 full-batch에서 나왔고, 미니배치 결과는
# full-batch와 직접 비교할 수 없다 -- 3개 주에서 미니배치를 한 번 재현해 다리를
# 놓아야 두 축이 만난다(확장설계서 §10-2). get_split 공유로 지켜온 비교 가능성
# 규율의 연장이다.
GNN_MINIBATCH = False
# 학습 배치의 seed 노드 수. 크게 잡는 이유는 공정성 벌점이 배치 단위 추정이 되기
# 때문이다(clear.gnn.fairness_penalty) -- 그룹평균이 배치에서 흔들리면 벌점이
# 엉뚱한 방향을 가리킨다. 8192면 전국 White/Black이 배치마다 수천 개씩 들어온다.
GNN_BATCH_SIZE = 8192
# 레이어별 이웃 샘플 수(층수와 길이가 같아야 한다). 차수 상한이 20(대칭화 후 ~40)이라
# [25, 10]이면 1홉은 사실상 전부, 2홉만 자른다. 이 과제에서 그래프의 값어치는
# '관련성'이 아니라 '블록 특성 분포의 비편향 표본'이므로(§2-8-6), 이웃을 일부만
# 뽑아도 그 추정치의 성격은 그대로다 -- 미니배치 손실이 작을 것으로 예상하는 근거.
GNN_NUM_NEIGHBORS = [25, 10]
# 평가(val/test) 배치. 추론은 이웃을 **전부** 쓰므로 배치당 서브그래프가 커진다 --
# 전국 카운티 블록이 크면(LA 44,511행) 4096 seed의 2-hop 합집합이 블록 전체로
# 포화한다. 실측: 4096에서 GPU가 11,880/12,282 MiB(97%)까지 찼다. 배치 크기는
# 추론 결과를 **바꾸지 않으므로**(seed 노드마다 이웃을 전부 쓰는 것은 동일)
# 여유를 위해 낮춘다. RUN_PARAMS에도 없어서 결과 identity와 무관하다.
GNN_EVAL_BATCH_SIZE = 2048
# 배치 안에서 이 수보다 적은 그룹은 공정성 벌점에서 뺀다(그룹평균이 잡음).
GNN_FAIR_MIN_COUNT = 100
# 층 표준화 벌점(mitigate_loss --fair_stratum)의 **(층 x 그룹) 셀** 하한.
# GNN_FAIR_MIN_COUNT가 그룹에 거는 하한이라면 이것은 셀에 거는 하한이다 — 층을
# 쪼개면 셀이 그만큼 작아지므로 하한도 작아야 한다. 하한을 올릴수록 벌점에
# 참여하는 층이 줄어드는 것이 **개입의 정의를 바꾸는** 일이라(층표준화벌점 계획서
# §4-4), 사후에 고르지 않고 격자로 훑는다. 기본값은 격자의 낮은 쪽이다.
GNN_FAIR_MIN_CELL = 20

# ---- 공정성 진단 (experiments/diagnose_fairness.py / clear.fairness) ----
# 'Unknown'은 인구집단이 아니라 기록 누락 코드라 격차(max-min) 계산에서 뺀다.
# 그룹별 지표 표에는 참고용으로 남긴다.
FAIRNESS_UNKNOWN_LABEL = "Unknown"
# 격차를 두 벌로 보고할 때의 최소 표본 기준. 이 표본(CA+TX+MI)에서 인종 그룹은
# White 33,973 / Black 20,718 / Asian·PI 1,431 / Native 178 이라, max-min을
# n=178 그룹이 결정해 버린다 — 개발계획서 Plan B의 "소수 그룹 희소" 리스크가
# 그대로 실현된 상태다. 1000이면 Native만 빠지고 나머지 셋이 남는다.
FAIRNESS_MIN_GROUP_N = 1000
# 부트스트랩 반복수·신뢰수준. 그룹 크기는 데이터가 정한 값이므로 고정하고
# 그룹 '안에서' 복원추출한다(자세한 근거는 clear/fairness.py).
FAIRNESS_BOOTSTRAP_N = 1000
FAIRNESS_CI_PCT = 95
