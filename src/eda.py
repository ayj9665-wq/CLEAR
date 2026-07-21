"""
eda.py — 탐색적 데이터 분석

목적: 모델링 전에 (1) 검거율, (2) 인종·성별 분포,
(3) 피해자 인종·성별에 따른 검거율 격차를 사전 확인한다.
이 격차가 바로 CLEAR 프로젝트가 '진단'하려는 대상이다.

산출물:
  outputs/eda_clearance_by_group.csv
  outputs/eda_clearance_by_race.png
  outputs/eda_clearance_by_race_sex.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import config as C


def main():
    df = pd.read_parquet(C.PROCESSED_DIR / "sample.parquet")
    print(f"[load] sample {len(df):,}행 (주={C.SAMPLE_STATES})")

    overall = df[C.TARGET_BIN].mean()
    print(f"\n[전체 검거율] {overall:.1%}")

    # 1) 인종별 검거율
    by_race = df.groupby("Victim Race")[C.TARGET_BIN].agg(["mean", "count"])
    by_race.columns = ["clearance_rate", "n"]
    print("\n[인종별 검거율]\n", by_race)

    # 2) 성별 검거율
    by_sex = df.groupby("Victim Sex")[C.TARGET_BIN].agg(["mean", "count"])
    by_sex.columns = ["clearance_rate", "n"]
    print("\n[성별 검거율]\n", by_sex)

    # 3) 인종 x 성별 교차
    by_rs = df.groupby(["Victim Race", "Victim Sex"])[C.TARGET_BIN].agg(["mean", "count"])
    by_rs.columns = ["clearance_rate", "n"]

    # CSV 저장
    csv_out = C.OUTPUT_DIR / "eda_clearance_by_group.csv"
    by_rs.reset_index().to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n[save] {csv_out}")

    # 그림 1: 인종별 검거율
    ax = by_race.sort_values("clearance_rate")["clearance_rate"].plot(
        kind="barh", figsize=(7, 4), color="#4C72B0")
    ax.axvline(overall, color="crimson", ls="--", label=f"Overall {overall:.1%}")
    ax.set_xlabel("Clearance rate")
    ax.set_title(f"Clearance rate by victim race ({', '.join(C.SAMPLE_STATES)})")
    ax.legend()
    plt.tight_layout()
    f1 = C.OUTPUT_DIR / "eda_clearance_by_race.png"
    plt.savefig(f1, dpi=120); plt.close()
    print(f"[save] {f1}")

    # 그림 2: 인종 x 성별 히트맵 형태(pivot bar)
    piv = by_rs.reset_index().pivot(index="Victim Race", columns="Victim Sex", values="clearance_rate")
    ax = piv.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Clearance rate")
    ax.set_title(f"Clearance rate by race x sex ({', '.join(C.SAMPLE_STATES)})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    f2 = C.OUTPUT_DIR / "eda_clearance_by_race_sex.png"
    plt.savefig(f2, dpi=120); plt.close()
    print(f"[save] {f2}")


if __name__ == "__main__":
    main()
