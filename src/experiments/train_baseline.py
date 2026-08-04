"""
experiments/train_baseline.py — flat 베이스라인 (그래프 없이 노드 특성만)

이 모델은 2주차 GNN 성능을 비교할 '기준선'이다.
"그래프가 실제로 도움이 되는가"를 판정하려면 먼저 그래프를 무시한
평범한 정형 모델의 성능을 확보해야 한다.

논문 기준 반영:
  - 70/30 무작위 층화 분할
  - 5-fold 층화 교차검증(하이퍼파라미터 소규모 그리드)
  - 평가지표: AUC / MCC / F1 / Sensitivity / Specificity — MCC를 대표 지표로
    삼는다(클래스 불균형에 강건한 단일 스칼라, 이전 balanced_accuracy의 역할)
  - 클래스 불균형: scale_pos_weight / class_weight

흐름:
  features.parquet 로드 → X / y 분리
   → 70/30 층화 분할(clear.data.get_split, 06과 동일 test 집합)
   → LogisticRegression, XGBoost 학습
   → 5-fold CV 로 XGBoost 소규모 튜닝
   → 지표 계산 → outputs/metrics.csv(06과 공통 원장)에 append, 특성 중요도 저장
   → 모델별 test 예측을 outputs/predictions_{logreg,xgboost}.csv로 저장

예측 덤프는 06과 같은 형식(clear.predictions)이라 experiments/diagnose_fairness.py가 baseline과
GNN의 공정성 격차를 나란히 비교할 수 있다 — "그래프가 격차를 더 키우는가"를
물으려면 flat 모델 쪽 예측도 있어야 하기 때문. 06과 달리 플래그로 끄지 않는다:
train_baseline는 무인자 실행 규약이고, 덤프 비용이 사실상 0이다.

baseline은 결정적(고정 split·고정 random_state)이라 모델당 1행이다. GNN처럼
seed 반복을 하지 않는 이유: seed로 split을 바꾸면 test 집합이 달라져 06과의
비교 가능성이 깨진다. 그래서 baseline은 고정 test 위의 단일 기준점으로 남긴다.
"""
import argparse
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

import config as C
from clear.data import load_xy, get_split
from clear.metrics import evaluate as compute_metrics
from clear import predictions, results


def evaluate(name, model, X_te, y_te, seed, rows, tag=""):
    """지표는 clear.metrics(06과 공통)로 계산하고, baseline 특유의
    혼동행렬·classification_report 진단만 여기서 추가로 출력한다.
    원장 스키마에 맞춰 timestamp/model/tag/seed를 단 행을 rows에 담는다."""
    proba = model.predict_proba(X_te)[:, 1]
    metric_vals = compute_metrics(y_te, proba)
    print(f"\n=== {name} ===")
    for k, v in metric_vals.items():
        print(f"  {k:18s}: {v:.4f}")
    pred = (proba >= 0.5).astype(int)
    print("  confusion matrix [[TN FP][FN TP]]:")
    print("  ", confusion_matrix(y_te, pred).tolist())
    print(classification_report(y_te, pred, target_names=["미해결", "검거"]))
    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name, "tag": tag, "seed": seed, **metric_vals,
    })
    return proba   # 호출부에서 예측 덤프(07 공정성 진단 입력)로 쓴다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", action="store_true",
                    help="민감속성 더미(config.SENSITIVE_FEATURE_COLS)를 X에서 제외")
    args = ap.parse_args()
    suffix = "_blind" if args.blind else ""

    X, y = load_xy(blind=args.blind)
    print(f"[data] X {X.shape}, 검거율 {y.mean():.1%}"
          + (f"  (blind: {C.SENSITIVE_FEATURE_COLS} 더미 제외)" if args.blind else ""))

    # get_split의 trainval(=예전 train_test_split의 train쪽, 순서 보존)로
    # 학습, test는 experiments/train_gnn.py와 동일 집합. CV는 이 trainval 안에서 돈다.
    split = get_split(y)
    X_tr, y_tr = X.iloc[split.trainval], y[split.trainval]
    X_te, y_te = X.iloc[split.test], y[split.test]
    print(f"[split] train {len(y_tr):,} / test {len(y_te):,} (70/30 층화, 06과 동일 test)")

    # 불균형 가중치
    pos = y_tr.sum(); neg = len(y_tr) - pos
    spw = neg / max(pos, 1)

    rows = []
    seed = C.RANDOM_STATE   # baseline은 이 고정 seed의 단일 기준점

    # 1) 로지스틱 회귀 베이스라인
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(X_tr, y_tr)
    proba_logit = evaluate("logreg", logit, X_te, y_te, seed, rows, tag=suffix.lstrip("_"))

    # 2) XGBoost + 5-fold CV 소규모 튜닝
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    grid = {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.1, 0.3]}
    xgb = XGBClassifier(
        scale_pos_weight=spw, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist", random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    gs = GridSearchCV(xgb, grid, scoring="matthews_corrcoef", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    print(f"\n[XGB best params] {gs.best_params_}")
    print(f"[XGB CV best MCC] {gs.best_score_:.4f}")
    proba_xgb = evaluate("xgboost", gs.best_estimator_, X_te, y_te, seed, rows,
                         tag=suffix.lstrip("_"))

    # 저장: 지표 → 06과 공통 원장(outputs/metrics.csv)에 append
    for r in rows:
        results.write(results.from_run(r, "train", blind=args.blind))
    print(f"\n[save] {results.results_path()} (family=train, {len(rows)}회)")

    # 저장: test 예측 덤프 → experiments/diagnose_fairness.py 입력(06과 동일 형식·동일 test 집합)
    for name, proba in [("logreg", proba_logit), ("xgboost", proba_xgb)]:
        p, _ = predictions.dump(predictions.path_for(name + suffix), split.test, y, proba)
        print(f"[save] {p} (test {len(split.test):,}행)")

    # 저장: XGB 특성 중요도 top 25.
    # sighted/blind를 **한 파일에 blind 열로** 담는다. 예전에는 --blind가 파일을
    # 하나 더 만들었는데(`..._blind.csv`), 조건이 하나 늘 때마다 파일이 하나씩
    # 느는 방식이고 두 조건을 나란히 보려면 매번 두 파일을 join해야 했다.
    imp = pd.Series(gs.best_estimator_.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(25)
    new = pd.DataFrame({"blind": args.blind, "feature": imp.index,
                        "importance": imp.values})
    # 스코프별로 갈라야 한다. 이 스크립트는 스코프 축이 생기기 **전**에 쓰였다가
    # 복원된 것이라 원래는 OUTPUT_DIR 고정이었고, 그대로 두면 전국 실행이 커밋된
    # 3개 주 파일을 덮어쓴다. 지표(results.csv)와 덤프는 이미 스코프 인식이 된다.
    out_i = C.scoped_output("baseline_xgb_top_features.csv")
    if out_i.exists():
        old = pd.read_csv(out_i)
        if "blind" in old.columns:                     # 같은 조건은 최신으로 교체
            new = pd.concat([old[old["blind"] != args.blind], new], ignore_index=True)
    new.sort_values(["blind", "importance"], ascending=[True, False]) \
       .to_csv(out_i, index=False, encoding="utf-8-sig")
    print(f"[save] {out_i} (blind={args.blind})")
    print("\n[XGB 상위 특성]\n", imp.head(10))


if __name__ == "__main__":
    main()
