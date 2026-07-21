"""
03_features.py — 특성 공학 (논문 기준: 전면 One-Hot)

논문(Campedelli 2022)은 모든 변수를 One-Hot 인코딩했고,
피해자 나이는 연속값이 아니라 5년 단위로 구간화 후 One-Hot 했다.

흐름:
  sample.parquet 로드
   → 나이 5년 구간화 → 범주
   → 연도 → Decade 범주
   → 범주형 전면 One-Hot
   → 수치형(Victim Count) 유지
   → 민감속성(원본 값)·타깃 별도 보관
   → data/processed/features.parquet 저장

산출물 features.parquet 구성:
  - X 특성 열 (One-Hot + 수치)
  - solved            : 타깃
  - sens__Victim Race : 공정성 진단용 원본 값
  - sens__Victim Sex  : 공정성 진단용 원본 값
"""
import numpy as np
import pandas as pd
import config as C


def bin_age(df):
    """나이를 5년 구간 범주로. 결측은 'Unknown' 범주."""
    age = df["Victim Age"]
    edges = list(range(0, 105, C.AGE_BIN_WIDTH))
    labels = [f"{lo}-{lo + C.AGE_BIN_WIDTH - 1}" for lo in edges[:-1]]
    binned = pd.cut(age, bins=edges, labels=labels, right=False)
    binned = binned.cat.add_categories(["Unknown"]).fillna("Unknown")
    return binned.astype(str)


def main():
    df = pd.read_parquet(C.PROCESSED_DIR / "sample.parquet")
    print(f"[load] sample {len(df):,}행")

    # 민감속성·타깃 먼저 분리 보관(원본 값 유지)
    keep = pd.DataFrame({C.TARGET_BIN: df[C.TARGET_BIN].values})
    for s in C.SENSITIVE_COLS:
        keep[f"sens__{s}"] = df[s].values

    # 1) 나이 구간화
    df["Age Group"] = bin_age(df)

    # 2) 연도 → Decade
    if "Year" in df.columns:
        df["Decade"] = (df["Year"] // 10 * 10).astype(str)

    # 3) 인코딩 대상 범주형 목록
    cat_cols = [c for c in C.CATEGORICAL_COLS if c in df.columns]
    cat_cols += ["Age Group", "Decade"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # 4) One-Hot
    X_cat = pd.get_dummies(df[cat_cols].astype(str), prefix_sep="=")
    print(f"[onehot] {len(cat_cols)}개 범주형 → {X_cat.shape[1]}개 더미 열")

    # 5) 수치형
    num_cols = [c for c in C.NUMERIC_COLS if c in df.columns]
    X_num = df[num_cols].fillna(0).reset_index(drop=True)

    # 6) 결합
    X = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
    out_df = pd.concat([X, keep.reset_index(drop=True)], axis=1)

    out = C.PROCESSED_DIR / "features.parquet"
    out_df.to_parquet(out, index=False)
    feat_n = X.shape[1]
    print(f"[save] {out}  (샘플 {len(out_df):,} / 특성 {feat_n}열)")


if __name__ == "__main__":
    main()
