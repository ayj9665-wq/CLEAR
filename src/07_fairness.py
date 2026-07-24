"""
07_fairness.py — 공정성 진단 (예측 → **진단** 단계, week 3)

06_train_gnn.py가 저장한 test 노드별 예측(outputs/predictions_graphsage_{edge}.csv)을
읽어, 피해자 인종·성별 그룹별로 검거 예측이 **체계적으로 불공평한지**를 측정한다.
이 프로젝트의 두 번째 단계(진단) 산출물 — 포스터의 "인종별 검거율 격차" 그림 재료.

지표(그룹 g별):
  base_rate     = 실제 검거율(y_true 평균)            — 데이터 자체의 격차
  selection_rate= 모델의 검거 예측 비율(pred 평균)     — Demographic Parity 대상
  tpr           = 검거 사건을 맞히는 재현율(sensitivity)
  fpr           = 미해결을 검거로 오판하는 비율        — Equalized Odds 대상(tpr와 함께)
  tnr, accuracy = 참고

공정성 격차(명명된 그룹만, 'Unknown' 제외):
  DP gap        = max-min selection_rate  (기회 동등)
  EO gap(TPR)   = max-min tpr             (검거 사건에서의 동등 대우)
  EO gap(FPR)   = max-min fpr

입력: outputs/predictions_graphsage_{edge}.csv  (06 --dump_predictions, 기본 on)
출력: outputs/fairness_group_metrics.csv         (그룹별 지표 long 포맷)

주의: 아직 **진단만** 한다. 격차를 줄이는 완화(처방, 단계 3)는 다음 스크립트.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config as C

SENS_ATTRS = [f"sens__{a}" for a in C.SENSITIVE_COLS]   # Victim Race, Victim Sex
UNKNOWN = "Unknown"


def _group_row(attr, g, y, p):
    """한 그룹의 혼동행렬 기반 지표. 단일 클래스여도 labels=[0,1]로 2x2 강제."""
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    denom = lambda a, b: (a / (a + b)) if (a + b) > 0 else np.nan
    return {
        "attribute": attr.replace("sens__", ""),
        "group": g,
        "n": len(y),
        "base_rate": y.mean(),            # 실제 검거율
        "selection_rate": p.mean(),       # 예측 검거율(DP)
        "tpr": denom(tp, fn),             # sensitivity
        "fpr": denom(fp, tn),             # 1 - specificity
        "tnr": denom(tn, fp),             # specificity
        "accuracy": (y == p).mean(),
    }


def group_metrics(df, attr):
    rows = [_group_row(attr, g, sub["y_true"].values, sub["pred"].values)
            for g, sub in df.groupby(attr)]
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def gaps(gm):
    """명명된 그룹(Unknown 제외)에서 max-min 격차."""
    named = gm[gm["group"] != UNKNOWN]
    span = lambda col: named[col].max() - named[col].min()
    return {
        "attribute": gm["attribute"].iloc[0],
        "dp_gap(selection_rate)": span("selection_rate"),
        "eo_gap(tpr)": span("tpr"),
        "eo_gap(fpr)": span("fpr"),
        "base_rate_gap": span("base_rate"),   # 데이터 자체 격차(모델 이전)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_type", default=C.GNN_DEFAULT_EDGE_TYPE)
    args = ap.parse_args()

    pred_path = C.OUTPUT_DIR / f"predictions_graphsage_{args.edge_type}.csv"
    if not pred_path.exists():
        sys.exit(f"[에러] {pred_path} 없음 — 먼저 실행: python 06_train_gnn.py "
                 f"--edge_type {args.edge_type}  (예측 덤프가 기본 on)")

    df = pd.read_csv(pred_path)
    print(f"[load] {pred_path.name}  test {len(df):,}행, 실제 검거율 {df['y_true'].mean():.1%}")

    all_gm, all_gaps = [], []
    for attr in SENS_ATTRS:
        if attr not in df.columns:
            print(f"[skip] {attr} 열 없음")
            continue
        gm = group_metrics(df, attr)
        all_gm.append(gm)
        g = gaps(gm)
        all_gaps.append(g)
        name = attr.replace("sens__", "")
        print(f"\n=== {name} ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(gm.to_string(index=False))
        print(f"  → DP gap(선택률) {g['dp_gap(selection_rate)']:.3f} | "
              f"EO gap(TPR) {g['eo_gap(tpr)']:.3f} | EO gap(FPR) {g['eo_gap(fpr)']:.3f} | "
              f"데이터 검거율 격차 {g['base_rate_gap']:.3f}")

    out = pd.concat(all_gm, ignore_index=True)
    out_path = C.OUTPUT_DIR / "fairness_group_metrics.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out_path}")
    print("[해석] EO/DP gap이 클수록 그룹 간 대우 불평등. base_rate_gap은 모델 이전의 "
          "데이터 자체 격차 (모델 격차가 이보다 크면 모델이 격차를 '증폭'한 것).")


if __name__ == "__main__":
    main()
