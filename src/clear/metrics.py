"""공용 평가 지표 — 05_train_baseline.py와 06_train_gnn.py의 단일 출처.

AUC / MCC / F1 / Sensitivity / Specificity 한 벌. 이전에는 이 딕셔너리가
evaluate()(baseline)와 train_one()(GNN)에 각각 복붙돼 있어서, 지표 하나를
바꾸려면 두 곳을 동시에 고쳐야 comparability가 유지됐다(CLAUDE.md가 그 수동
동기화를 문서로 설명하고 있었다). 이제 한 곳이다. 두 스크립트 모두 양성 클래스
확률을 0.5에서 잘라 같은 방식으로 예측 레이블을 만든다.
"""
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, recall_score

# 순서 고정: 출력 로그·CSV 열 순서를 두 스크립트에서 동일하게 유지한다.
METRIC_NAMES = ["auc", "mcc", "f1", "sensitivity", "specificity"]


def evaluate(y_true, y_proba, threshold=0.5):
    """양성 클래스 확률 → 지표 딕셔너리(동일 임계값·동일 지표 체계).

    sensitivity = 양성(검거) 재현율(TPR),
    specificity = recall_score(pos_label=0), 즉 음성(미해결) 재현율(TNR).
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),               # TPR
        "specificity": recall_score(y_true, y_pred, pos_label=0),  # TNR
    }
