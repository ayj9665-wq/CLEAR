"""test 노드 예측 덤프의 단일 출처 — 학습 단계와 진단 단계의 인터페이스.

공정성 진단은 모델을 다시 띄우지 않는다. 학습 스크립트가 test 예측을 민감속성과
함께 CSV로 떨어뜨리고, 진단은 그 CSV만 읽는다 — 그래서 GNN 재학습(수 분) 없이
격차 계산을 몇 번이고 반복할 수 있다. 이 모듈이 그 CSV의 형식을 정의한다.

이전에는 이 로직이 clear/gnn.py:dump_test_predictions 안에 GNN 전용으로
(seed 평균을 전제한 채) 들어 있었다. baseline(05)은 seed 반복이 없어 그대로
쓸 수 없었고, 완화 단계(08)는 완화된 예측을 같은 형식으로 다시 써야 한다.
그래서 "seed 평균"(GNN 사정)과 "덤프 형식"(공통)을 분리해, 형식 쪽만 여기 둔다.

열: row_index(=features.parquet 행 위치) / y_true / proba / pred / sens__*.
row_index를 남기는 이유는 서로 다른 모델의 덤프가 같은 test 집합을 가리키는지
동일성 검증(assert_same_test_set)이 가능해야 하기 때문이다 — 그게 깨지면
모델 간 격차 비교가 무의미해진다.
"""
import pandas as pd
import numpy as np

import config as C
from clear.data import load_sensitive

# 덤프는 outputs/predictions/{model}[_{edge}].csv에 모아 둔다.
#
# 예전에는 outputs/ 바로 아래에 predictions_*.csv로 흩어져 있었다. 완화 스윕이
# 설정마다 한 개씩 남기다 보니 42개(79MB)까지 늘어, 실제 실험 기록인 결과 CSV
# 9개(1.5MB)가 파일 목록에서 묻혔다. 전부 gitignore되고 재실행으로 재생성되는
# 중간 산출물이므로 디렉터리 하나로 내린다.
PREDICTIONS_DIRNAME = "predictions"


def predictions_dir():
    """호출 시점 계산 — OUTPUT_DIR을 바꾸면 덤프도 함께 격리된다."""
    return C.OUTPUT_DIR / PREDICTIONS_DIRNAME


def path_for(model, edge_type=None):
    stem = f"{model}_{edge_type}" if edge_type else model
    d = predictions_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.csv"


def dump(path, test_idx, y, proba, threshold=0.5):
    """test 예측 + 민감속성을 CSV로 저장.

    test_idx는 get_split이 돌려준 원본 행 위치라, load_sensitive()(같은 행 순서)를
    그대로 iloc할 수 있다. 임계값은 clear.metrics.evaluate와 동일한 0.5 —
    원장의 지표와 덤프의 pred가 같은 결정 규칙을 쓰게 맞춘 것이다.
    """
    test_idx = np.asarray(test_idx)
    proba = np.asarray(proba)
    sens = load_sensitive().iloc[test_idx].reset_index(drop=True)
    out = pd.DataFrame({
        "row_index": test_idx,
        "y_true": np.asarray(y)[test_idx].astype(int),
        "proba": proba,
        "pred": (proba >= threshold).astype(int),
    })
    out = pd.concat([out, sens], axis=1)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path, out


def load(path):
    return pd.read_csv(path)


def discover(models=None):
    """outputs/predictions/의 덤프를 {라벨: 경로}로 수집(라벨 = 파일명 stem).

    diagnose_fairness가 인자 없이도 "있는 덤프 전부"를 진단할 수 있게 하는 용도.
    models를 주면 그 라벨들로 거른다. 정렬은 파일명 순 — 실행마다 표 순서가
    바뀌지 않게.
    """
    d = predictions_dir()
    found = {p.stem: p for p in sorted(d.glob("*.csv"))} if d.exists() else {}
    # 하위 디렉터리로 옮기기 전 규약(outputs/predictions_*.csv)도 계속 읽는다 —
    # 옮기지 않은 예전 작업 디렉터리에서도 진단이 그대로 돌게.
    for p in sorted(C.OUTPUT_DIR.glob("predictions_*.csv")):
        found.setdefault(p.stem.replace("predictions_", ""), p)
    if models:
        found = {m: found[m] for m in models if m in found}
    return found


def assert_same_test_set(dumps):
    """여러 덤프가 동일한 test 행을 가리키는지 검증(모델 간 비교의 전제).

    clear.data.get_split이 모든 호출부에 같은 test 인덱스를 주므로 정상 상태에선
    항상 통과한다. 그래도 확인하는 이유: 어긋나도 조용히 "그럴듯한" 비교표가
    나오기 때문이다 — 분할 로직이 바뀌었을 때 여기서 걸려야 한다.
    dumps: {라벨: DataFrame}
    """
    ref_label, ref = next(iter(dumps.items()))
    ref_idx = ref["row_index"].values
    for label, df in dumps.items():
        if len(df) != len(ref_idx) or not np.array_equal(df["row_index"].values, ref_idx):
            raise ValueError(
                f"test 집합 불일치: '{label}'({len(df):,}행)와 '{ref_label}'({len(ref_idx):,}행)의 "
                f"row_index가 다르다. 05/06을 같은 clear.data.get_split 아래에서 다시 실행할 것 "
                f"(이 상태의 모델 간 격차 비교는 무의미하다).")
    return ref_idx
