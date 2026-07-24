"""공용 데이터 로딩·분할 — 05_train_baseline.py와 06_train_gnn.py의 단일 출처.

이전 상태(이 모듈로 통합하기 전):
  - load_xy()가 두 스크립트에 복붙돼 있었고, bool→int8 캐스팅이 baseline
    쪽에만 있어 이미 미세하게 갈라져 있었다.
  - test 분할은 06이 05의 train_test_split을 "같은 n·stratify·random_state로
    다시 호출하면 같은 행이 나온다"는 불변식에 의존해 재현했다. 맞는 말이지만
    둘 중 한쪽 분할 로직이 바뀌면 에러 없이 조용히 어긋난다 — 비교의 최악
    실패 모드다.

이제 두 스크립트가 같은 get_split()을 호출한다. 분할 로직이 바뀌면 양쪽이
함께 바뀌므로 test 집합 동일성이 (문서 규약이 아니라) 코드로 보장된다.
get_split은 결정적 순수 함수라 디스크 캐시 없이도 재현된다 — baseline은
trainval(첫 분할 순서 보존)로, GNN은 train/val/test로 나눠 쓴다.
"""
from collections import namedtuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config as C

# baseline은 trainval+test, GNN은 train/val/test를 쓴다. 한 벌로 둘 다 커버.
Split = namedtuple("Split", ["train", "val", "test", "trainval"])


def load_xy(blind=False):
    """features.parquet → (X, y).

    타깃과 sens__* 열은 X에서 뺀다. **주의**: sens__*를 뺀다고 모델이 민감속성을
    못 보는 게 아니다 — 03_features.py가 CATEGORICAL_COLS(Victim Race/Sex 포함)를
    전면 원핫하므로 'Victim Race=Black' 같은 더미가 X에 그대로 남는다. sens__*는
    진단용 **사본**(원본 라벨)이지 유일한 경로가 아니다. 기본값 blind=False는 이
    상태 그대로이고(= 지금까지의 모든 실험 조건), blind=True가
    config.SENSITIVE_FEATURE_COLS 접두어의 더미를 마저 떼어낸다.

    blind=True는 두 몫을 한다: (1) 가장 단순한 완화기법(fairness through
    unawareness), (2) "그래프가 민감속성의 우회 경로인가"의 검정 조건 — 직접
    경로를 막아야 그래프 경로만 남아 분리 측정이 된다.

    bool 더미는 int8로 캐스팅한다 — XGBoost 입력을 정리하고, GNN은 이후
    float32로 변환하므로 무손실이다.
    """
    df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
    sens_cols = [c for c in df.columns if c.startswith("sens__")]
    y = df[C.TARGET_BIN].values
    X = df.drop(columns=[C.TARGET_BIN] + sens_cols)
    if blind:
        drop = [c for c in X.columns
                if any(c.startswith(f"{p}=") for p in C.SENSITIVE_FEATURE_COLS)]
        X = X.drop(columns=drop)
    X = X.astype({c: "int8" for c in X.columns if X[c].dtype == bool})
    return X, y


def load_sensitive():
    """features.parquet → sens__* 원본(비인코딩) DataFrame, X/y와 같은 행 순서.

    load_xy()가 학습 입력에서 떼어낸 민감속성(sens__Victim Race/Sex)을 그대로
    돌려준다. 공정성 진단(07)에서 test 인덱스로 iloc해 예측과 join하기 위한 것 —
    get_split이 돌려주는 인덱스는 이 DataFrame 행 위치와 그대로 대응한다.
    """
    df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
    sens_cols = [c for c in df.columns if c.startswith("sens__")]
    return df[sens_cols].reset_index(drop=True)


def get_split(y, *, test_size=None, val_size=None, random_state=None):
    """층화 분할 인덱스를 결정적으로 계산. 05·06 공통 출처.

    test는 첫 분할에서만 나오고 val_size와 무관하므로, baseline과 GNN이
    (그리고 --val_size를 바꿔도) 항상 동일한 test 집합을 쓴다 — 비교 가능성이
    구성상 보장된다. trainval은 첫 분할이 돌려준 순서 그대로라, baseline이
    X.iloc[trainval]로 예전(train_test_split 직접 호출)과 비트 단위로 같은
    학습 풀·CV 폴드를 얻는다.

    인자를 안 주면 config 기본값(TEST_SIZE / GNN_VAL_SIZE / RANDOM_STATE).
    """
    test_size = C.TEST_SIZE if test_size is None else test_size
    val_size = C.GNN_VAL_SIZE if val_size is None else val_size
    random_state = C.RANDOM_STATE if random_state is None else random_state

    n = len(y)
    trainval, test = train_test_split(
        np.arange(n), test_size=test_size, stratify=y, random_state=random_state)
    train, val = train_test_split(
        trainval, test_size=val_size, stratify=y[trainval], random_state=random_state)
    return Split(train=train, val=val, test=test, trainval=trainval)
