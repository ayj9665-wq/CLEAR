"""공용 평가 지표 — 05_train_baseline.py와 06_train_gnn.py의 단일 출처.

AUC / MCC / F1 / Sensitivity / Specificity 한 벌 + 논문 대조용 Balanced Accuracy /
Precision. 이전에는 이 딕셔너리가 evaluate()(baseline)와 train_one()(GNN)에 각각
복붙돼 있어서, 지표 하나를 바꾸려면 두 곳을 동시에 고쳐야 comparability가
유지됐다(CLAUDE.md가 그 수동 동기화를 문서로 설명하고 있었다). 이제 한 곳이다. 두
스크립트 모두 양성 클래스 확률을 0.5에서 잘라 같은 방식으로 예측 레이블을 만든다.

balanced_accuracy / precision은 선행연구(Campedelli 2022 등)가 이 두 지표만
보고하기 때문에, 문헌과 **같은 축에서** 우리 모델을 대볼 수 있도록 나중에 추가했다.
모델 선택 대표 스칼라는 여전히 MCC(불균형에 강건). 이 둘은 보고·비교용 부가 지표다.
"""
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score, recall_score,
    balanced_accuracy_score, precision_score,
)

# 순서 고정: 출력 로그·CSV 열 순서를 두 스크립트에서 동일하게 유지한다.
# 뒤쪽 둘(balanced_accuracy/precision)은 논문 대조용으로 나중에 추가 → 기존
# 다섯 지표 뒤에 append(원장 CSV 열 정렬 유지).
METRIC_NAMES = ["auc", "mcc", "f1", "sensitivity", "specificity",
                "balanced_accuracy", "precision"]


def evaluate(y_true, y_proba, threshold=0.5):
    """양성 클래스 확률 → 지표 딕셔너리(동일 임계값·동일 지표 체계).

    sensitivity = 양성(검거) 재현율(TPR),
    specificity = recall_score(pos_label=0), 즉 음성(미해결) 재현율(TNR).
    balanced_accuracy = (sensitivity + specificity)/2 — 논문(Campedelli 2022)의
        대표 지표. 불균형에서 다수 클래스에 쏠린 accuracy 대신 두 클래스 재현율 평균.
    precision = 양성(검거) 예측의 정밀도 — 논문이 "미해결을 검거로 오분류하는 것을
        줄이려는" 의도로 함께 보고하는 지표.
    """
    return evaluate_pred(y_true, (y_proba >= threshold).astype(int), y_proba)


def evaluate_pred(y_true, y_pred, y_proba=None):
    """예측 레이블이 이미 정해진 경우의 같은 지표 한 벌.

    완화 단계(08)는 그룹마다 다른 임계값을 적용하므로 "확률 + 단일 임계값"으로
    표현되지 않는다. 그래서 레이블을 직접 받는 입구를 둔다 — 지표 정의는
    evaluate()와 **같은 코드**를 쓴다(정의가 갈라지면 완화 전후 비교가 무의미).

    y_proba를 주면 auc를 함께 낸다. auc는 순위 기반이라 임계값을 어떻게 바꾸든
    변하지 않는다 — 완화 곡선에서 auc가 평평한 것은 버그가 아니라 정의상 당연하다.
    """
    return {
        "auc": roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),               # TPR
        "specificity": recall_score(y_true, y_pred, pos_label=0),  # TNR
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),              # 양성(검거) 정밀도
    }
