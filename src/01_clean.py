"""
01_clean.py — 데이터 정제

흐름:
  raw CSV 로드
   → 타깃 누수 열 제거(가해자·관계)
   → 저정보·식별자 열 제거
   → 타깃 인코딩(Crime Solved: Yes=1, No=0)
   → 피해자 나이 미상 코드 처리
   → data/processed/clean.parquet 저장

이 단계의 핵심은 '누수 열 제거'다. 미해결 사건은 가해자를 모르므로
가해자 관련 열이 사실상 정답을 누설한다(검증 완료).
"""
import pandas as pd
import config as C


def main():
    print(f"[load] {C.RAW_CSV}")
    df = pd.read_csv(C.RAW_CSV, low_memory=False)
    n0 = len(df)
    print(f"  원본 {n0:,}행 x {df.shape[1]}열")

    # 1) 타깃 인코딩
    df[C.TARGET_BIN] = (df[C.TARGET] == "Yes").astype(int)
    solved_rate = df[C.TARGET_BIN].mean()
    print(f"[target] 검거율 {solved_rate:.1%} / 미해결 {1 - solved_rate:.1%}")

    # 2) 누수 열 제거
    leak = [c for c in C.LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=leak)
    print(f"[drop-leakage] 제거: {leak}")

    # 3) 저정보·식별자 제거 (원본 타깃 열도 이진본으로 대체)
    drop = [c for c in C.DROP_COLS + [C.TARGET] if c in df.columns]
    df = df.drop(columns=drop)
    print(f"[drop-lowinfo] 제거: {drop}")

    # 4) 피해자 나이 미상 처리: 998/999 또는 비현실적 값 → 결측
    if "Victim Age" in df.columns:
        bad = (df["Victim Age"] >= 100) | (df["Victim Age"] == C.AGE_UNKNOWN_CODE)
        print(f"[age] 미상/이상치 {bad.sum():,}건 → NaN 처리")
        df.loc[bad, "Victim Age"] = pd.NA

    out = C.PROCESSED_DIR / "clean.parquet"
    df.to_parquet(out, index=False)
    print(f"[save] {out}  ({len(df):,}행 x {df.shape[1]}열)")
    print(f"[남은 열] {list(df.columns)}")


if __name__ == "__main__":
    main()
