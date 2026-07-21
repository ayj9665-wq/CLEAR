"""
02_sample.py — 주(州) 단위 표본 추출

논문(Campedelli 2022)은 전국 모델과 별개로 주(州)별 서브셋을 분석했다.
GNN 학습 부담을 고려해, 기본적으로 California 서브셋(약 12만 행)을 뽑는다.

흐름:
  clean.parquet 로드
   → config.SAMPLE_STATES 로 주 필터
   → (선택) SAMPLE_MAX_ROWS 로 타깃 층화 다운샘플
   → data/processed/sample.parquet 저장
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import config as C


def main():
    df = pd.read_parquet(C.PROCESSED_DIR / "clean.parquet")
    print(f"[load] clean {len(df):,}행")

    # 1) 주(州) 필터
    if C.SAMPLE_STATES:
        df = df[df["State"].isin(C.SAMPLE_STATES)].copy()
        print(f"[filter] 주={C.SAMPLE_STATES} → {len(df):,}행")

    # 2) 층화 다운샘플(옵션)
    if C.SAMPLE_MAX_ROWS and len(df) > C.SAMPLE_MAX_ROWS:
        frac = C.SAMPLE_MAX_ROWS / len(df)
        df, _ = train_test_split(
            df, train_size=frac, stratify=df[C.TARGET_BIN],
            random_state=C.RANDOM_STATE,
        )
        print(f"[subsample] {len(df):,}행 (층화 유지)")

    rate = df[C.TARGET_BIN].mean()
    print(f"[check] 표본 검거율 {rate:.1%}  (분포 유지 확인)")
    print(f"[check] Victim Race:\n{df['Victim Race'].value_counts()}")

    out = C.PROCESSED_DIR / "sample.parquet"
    df.to_parquet(out, index=False)
    print(f"[save] {out}  ({len(df):,}행)")


if __name__ == "__main__":
    main()
