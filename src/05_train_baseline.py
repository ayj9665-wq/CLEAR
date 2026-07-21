"""
05_train_baseline.py — flat 베이스라인 (그래프 없이 노드 특성만)

이 모델은 2주차 GNN 성능을 비교할 '기준선'이다.
"그래프가 실제로 도움이 되는가"를 판정하려면 먼저 그래프를 무시한
평범한 정형 모델의 성능을 확보해야 한다.

논문 기준 반영:
  - 70/30 무작위 층화 분할
  - 5-fold 층화 교차검증(하이퍼파라미터 소규모 그리드)
  - 평가지표: Balanced Accuracy + Precision (+ ROC-AUC, PR-AUC 보조)
  - 클래스 불균형: scale_pos_weight / class_weight

흐름:
  features.parquet 로드 → X / y 분리
   → 70/30 층화 분할
   → LogisticRegression, XGBoost 학습
   → 5-fold CV 로 XGBoost 소규모 튜닝
   → 지표 계산 → outputs/baseline_metrics.csv, 특성 중요도 저장
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
import config as C


def load_xy():
    df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
    sens_cols = [c for c in df.columns if c.startswith("sens__")]
    y = df[C.TARGET_BIN].values
    X = df.drop(columns=[C.TARGET_BIN] + sens_cols)
    # bool 더미 → int
    X = X.astype({c: "int8" for c in X.columns if X[c].dtype == bool})
    return X, y


def evaluate(name, model, X_te, y_te, rows):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    m = {
        "model": name,
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
    }
    print(f"\n=== {name} ===")
    for k, v in m.items():
        if k != "model":
            print(f"  {k:18s}: {v:.4f}")
    print("  confusion matrix [[TN FP][FN TP]]:")
    print("  ", confusion_matrix(y_te, pred).tolist())
    print(classification_report(y_te, pred, target_names=["미해결", "검거"]))
    rows.append(m)


def main():
    X, y = load_xy()
    print(f"[data] X {X.shape}, 검거율 {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    print(f"[split] train {len(y_tr):,} / test {len(y_te):,} (70/30 층화)")

    # 불균형 가중치
    pos = y_tr.sum(); neg = len(y_tr) - pos
    spw = neg / max(pos, 1)

    rows = []

    # 1) 로지스틱 회귀 베이스라인
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(X_tr, y_tr)
    evaluate("LogisticRegression", logit, X_te, y_te, rows)

    # 2) XGBoost + 5-fold CV 소규모 튜닝
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    grid = {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.1, 0.3]}
    xgb = XGBClassifier(
        scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist", random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    gs = GridSearchCV(xgb, grid, scoring="balanced_accuracy", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    print(f"\n[XGB best params] {gs.best_params_}")
    print(f"[XGB CV best balanced_accuracy] {gs.best_score_:.4f}")
    evaluate("XGBoost", gs.best_estimator_, X_te, y_te, rows)

    # 저장: 지표
    out_m = C.OUTPUT_DIR / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(out_m, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out_m}")

    # 저장: XGB 특성 중요도 top 25
    imp = pd.Series(gs.best_estimator_.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(25)
    out_i = C.OUTPUT_DIR / "baseline_xgb_top_features.csv"
    imp.to_csv(out_i, header=["importance"], encoding="utf-8-sig")
    print(f"[save] {out_i}")
    print("\n[XGB 상위 특성]\n", imp.head(10))


if __name__ == "__main__":
    main()
