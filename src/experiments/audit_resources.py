"""§6-4의 4중 교란 중 **수사 인력** 한 항을 밖에서 측정한다.

`detect_cold_blocks`의 z는 카운티 고유 효과를 통째로 담는다 — 모델이 카운티를
특성으로 갖지 않기 때문이다. 그래서 지금까지 보고서와 지도 캡션은 "차별·수사자원·
도시성·기록관행이 분리되지 않는다"까지만 말할 수 있었다. 이 스크립트는 그 넷 중
하나(인력 규모)를 외부 데이터로 붙여 **z가 그것으로 설명되는지** 묻는다.

## 왜 모델에 넣지 않고 밖에서 재는가 — 이 설계의 핵심

자원 변수를 모델 입력에 넣는 것이 자연스러워 보이지만, 그러면 **z의 정의가
무너진다.** 모델은 "1건당 경찰관이 적은 곳은 잘 안 풀린다"를 배워 E_b를 키우고,
디트로이트의 이상은 *문제가 해결돼서가 아니라 모델이 미리 정상이라고 인정해서*
사라진다. 이건 `detect_cold_blocks`가 인종에 대해 이미 한 논증과 같다 — 인종을
보는 모델은 "흑인 피해자니 미해결 기대"를 기준선에 구워 격차를 정상으로 세탁한다.
카운티를 특정하는 변수도 같은 일을 카운티 축에서 한다.

인력은 한 겹 더 위험하다. **내생적**이기 때문이다 — 살인이 많은 곳에 경찰을 더
배치하므로, 인력은 외생적 '투입'이 아니라 "이 카운티가 얼마나 심각한가"의 지표에
가깝다. 모델에 넣으면 통제되는 것이 투입이 아니라 z가 재려던 것 자체가 된다.

따라서 **모델과 z는 건드리지 않고**, 카운티 단위 회귀로 사후 비교만 한다. 기존
지도·`cold_blocks*.csv`·보고서 인용은 전부 그대로 유효하다.

## 측정량

    z_b ~ black_share_b                    (1) 현행 발견
    z_b ~ black_share_b + log(자원_b)      (2)

핵심은 **(1)→(2)에서 black_share 계수가 얼마나 줄어드는가**다. 크게 줄면 인종
상관의 상당 부분이 인력 부족의 대리였다는 뜻이고, 안 줄면 인력으로 설명되지
않는다는 뜻이다. R²보다 계수 감쇠를 헤드라인으로 두는 이유는, 질문이 "z를 무엇이
설명하는가"가 아니라 "인종 상관이 자원 이야기인가"이기 때문이다.

## 자원 지표는 **사건 1건당**이지 인구당이 아니다

인구당(officers per capita)은 두 가지로 잘못된다. (1) 카운티 안에서 시경찰·보안관·
주경찰의 **관할 인구가 겹쳐** 합산하면 이중계산된다(경찰관 수는 고용 기관이 하나라
안 겹친다 — 분자는 멀쩡하고 분모만 부풀려진다). (2) 더 중요하게, 같은 인구당 인력도
살인 부담이 2배면 사건당 역량은 절반이다. 실제로 cold 카운티는 인구당 경찰관이
**더 많고**(2.20 대 warm 1.64) 살인 부담도 1.75배라, 1건당으로 정규화해야 비로소
역량이 26% 적다는 사실이 보인다. 정규화를 안 하면 부호가 뒤집힌다.

원자료 열(`officers`, `homicides`)을 CSV에 그대로 남기므로 다른 정규화도 가능하다.

## 데이터

  - LEOKA (Kaplan 통합본, openICPSR 102180): 기관-연도 선서 경찰관/민간직원 수.
    `ori`가 원자료 `Agency Code`와 그대로 붙는다 — (ORI, 연도) 정확 조인 100.0%
    (638,452/638,454), 1980~2014 전 연도 100%.
  - `Agency Code`는 `01_clean.py`가 떨어뜨리므로 **원본 CSV에서 읽는다.** 행 단위로
    쓰지 않고 (State, City) 카운티 집계에만 쓰므로 파이프라인 산출물과의 정렬 문제는
    없다(원본과 national 스코프는 둘 다 638,454행으로 동일하다).
  - LEAIC(ICPSR 35158)는 **쓰지 않는다.** ORI→카운티 매핑이 필요 없기 때문이고
    (카운티는 우리 `City` 필드가 이미 갖고 있다), 실제로 그 표에는 오류가 있다 —
    PAPEP00(필라델피아 경찰청)이 Indiana County(42063)로 배정돼 있어 그대로 조인하면
    12,848건이 엉뚱한 시골 카운티로 들어간다.

## 이 결과의 강도를 제한하는 것 둘 — 결론과 함께 반드시 인용할 것

  1. **프록시가 뭉뚝하다.** 총 선서 경찰관 수이고 형사 인원이 아니다(LEOKA에 형사
     인원 열이 없다 — `total_detective_*`는 피습 경찰관의 근무형태 분류다).
     "수사 노력을 안 쟀으니 null이 당연하다"는 반론은 옳고, 이 데이터로 막을 수 없다.
  2. **내생성이 감쇠율을 아래로 편향시킨다 — 이 caveat은 결론에 불리한 쪽이다.**
     경찰 인력은 사건 수를 거의 그대로 따라간다(corr(log 경찰관, log 살인) = +0.883,
     탄력성 1.219). 그래서 회귀변수 log(경찰관/살인)의 카운티 간 분산이 경찰관 분산의
     **49%로 압축**돼 있고(SD 0.865 대 1.747), 변동이 없는 변수는 진짜 효과가 있어도
     설명하지 못한다. 역인과도 같은 방향이다 — "자원↑ → 잘 푼다"(음)와 "안 풀린 사건이
     쌓인다 → 인력을 더 붙인다"(양)가 상쇄돼 순효과가 0에 가까워진다. 둘 다 **관측
     감쇠율을 실제보다 작게** 만든다: 진짜 자원 몫은 8%보다 클 수 있고, 따라서 "인력으로
     설명 안 된다"는 주장은 실제보다 강해 보이고 있다. 그래서 결론을 "자원이 원인이
     아니다"가 아니라 **"우리가 측정한 인력 규모로는 설명되지 않는다"**로 좁힌다.

`clear/`에 두지 않은 이유는 호출부가 하나이기 때문이다(저장소 규칙: 2개 이상일 때만
`clear/`로 올린다).

출력: outputs[/{scope}]/resource_audit.csv + results.csv(family=resource_audit)
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear import results

LEOKA_PATH = (C.ROOT / "dataset" / "geo" / "LEOKA_parquet_1960_2024_year"
              / "leoka_yearly_1960_2024.parquet")

# 분자만 쓰는 지표들. 분모는 항상 사건 수 -- 위 docstring의 인구 이중계산 참고.
MEASURES = {
    "officers": "total_employees_officers",
    "employees": "total_employees_total",
}


def load_county_resources():
    """(State, City) -> 경찰관/직원 수 합계 + 사건 수.

    기관-연도는 **우리 데이터에 사건이 있는 것만** 센다. 노출(exposure)과 역량을
    같은 창에서 맞추기 위해서다 -- 사건이 한 건도 없던 기관-연도의 인력을 더하면
    분자만 커지고 분모는 그대로라 1건당 역량이 부풀려진다.
    """
    if not LEOKA_PATH.exists():
        raise SystemExit(
            f"[에러] LEOKA 없음: {LEOKA_PATH}\n"
            f"  openICPSR 102180에서 LEOKA_parquet_1960_2024_year.zip을 받아\n"
            f"  압축을 풀어 dataset/geo/ 에 둘 것(gitignore 됨).")

    raw = pd.read_csv(C.RAW_CSV, dtype={"Agency Code": str},
                      usecols=["Agency Code", "State", "City", "Year"])
    lo = pd.read_parquet(LEOKA_PATH,
                         columns=["ori", "year"] + list(MEASURES.values()))
    lo = lo[lo.year.between(raw.Year.min(), raw.Year.max())]
    lo = lo.drop_duplicates(["ori", "year"]).set_index(["ori", "year"])

    idx = pd.MultiIndex.from_arrays([raw["Agency Code"], raw["Year"]])
    for name, col in MEASURES.items():
        raw[name] = pd.to_numeric(lo[col].reindex(idx).values, errors="coerce")

    # 인력은 기관-연도당 한 번만 (사건 행마다 세면 큰 카운티가 자기 인력을 수천 번 센다).
    per_agency_year = raw.drop_duplicates(["Agency Code", "Year"])
    cap = per_agency_year.groupby(["State", "City"])[list(MEASURES)].sum()
    hom = raw.groupby(["State", "City"]).size().rename("homicides")
    out = cap.join(hom)
    for name in MEASURES:
        out[f"{name}_per_homicide"] = out[name] / out["homicides"]
    joined = raw["officers"].notna().mean()
    print(f"[LEOKA] (ORI, 연도) 조인 {100*joined:.1f}%  "
          f"카운티 {len(out):,}개  기관-연도 {len(per_agency_year):,}")
    return out.reset_index()


def _ols(y, X):
    """절편 포함 OLS. 반환 (계수벡터, R^2). statsmodels 의존성을 만들지 않는다."""
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta, 1.0 - resid.var() / y.var()


def _attenuation(z, race, res):
    """black_share 계수가 자원 통제로 줄어드는 비율. 이 스크립트의 헤드라인."""
    b1, r2_race = _ols(z, race.reshape(-1, 1))
    b2, r2_both = _ols(z, np.column_stack([race, res]))
    _, r2_res = _ols(z, res.reshape(-1, 1))
    att = np.nan if b1[1] == 0 else 1.0 - b2[1] / b1[1]
    return {"race_coef": b1[1], "race_coef_controlled": b2[1],
            "race_attenuation": att, "r2_race": r2_race,
            "r2_resource": r2_res, "r2_both": r2_both,
            "partial_r2_resource": r2_both - r2_race}


def analyse(m, measure, n_boot, seed):
    """카운티 표 -> 통계 dict + 감쇠율 CI."""
    z = m["z"].to_numpy(float)
    race = m["black_share"].to_numpy(float)
    res = np.log(m[f"{measure}_per_homicide"].clip(lower=1e-6).to_numpy(float))
    stats = _attenuation(z, race, res)
    stats["corr_z_resource"] = float(np.corrcoef(z, res)[0, 1])
    stats["corr_z_race"] = float(np.corrcoef(z, race)[0, 1])

    # 카운티 복원추출. 관측 단위가 카운티이므로 카운티를 뽑는다 -- 사건을 뽑으면
    # 큰 카운티가 반복 등장해 z의 카운티 간 분산을 과소평가한다.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(m), len(m))
        try:
            draws.append(_attenuation(z[i], race[i], res[i])["race_attenuation"])
        except np.linalg.LinAlgError:
            continue
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return stats, (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cold_blocks_cv5.csv",
                    help="detect_cold_blocks 출력. 기본은 크로스피팅(대표 지도와 같은 표).")
    ap.add_argument("--min_n", type=int, nargs="+", default=[20, 50, 100],
                    help="블록 최소 표본. 원본 표가 각 floor마다 **검정을 다시 돌린** "
                         "행을 갖고 있으므로 여기서도 floor별로 따로 회귀한다.")
    ap.add_argument("--measures", nargs="+", default=list(MEASURES),
                    choices=list(MEASURES))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="resource_audit.csv")
    args = ap.parse_args()

    res = load_county_resources()
    src = C.scoped_output(args.src)
    if not src.exists():
        raise SystemExit(f"[에러] {src} 없음. 먼저 detect_cold_blocks를 실행할 것.")
    cold = pd.read_csv(src)

    tabs, ledger = [], []
    for min_n in args.min_n:
        c = cold[cold.min_n == min_n]
        if c.empty:
            print(f"[경고] min_n={min_n} 행이 {src.name}에 없다 — 건너뜀")
            continue
        m = c.merge(res, on=["State", "City"], how="inner")
        m = m[m["homicides"] > 0].dropna(subset=["z", "black_share", "officers"])
        if len(m) < 30:
            print(f"[경고] min_n={min_n}: 매칭 카운티 {len(m)}개뿐 — 건너뜀")
            continue

        print(f"\n=== n>={min_n} : 카운티 {len(m):,}개 "
              f"(원표 {len(c):,}개 중 {100*len(m)/len(c):.1f}% 매칭) ===")
        for lab, sub in [("cold", m[m.flag == "cold"]), ("warm", m[m.flag == "warm"]),
                         ("전체", m)]:
            if not len(sub):
                continue
            print(f"  {lab:5s} n={len(sub):5,d}  "
                  f"1건당 경찰관 {sub.officers_per_homicide.median():6.1f}  "
                  f"흑인비중 {sub.black_share.median():.3f}")

        for measure in args.measures:
            st, (lo, hi) = analyse(m, measure, args.n_boot, C.RANDOM_STATE)
            print(f"  [{measure}] "
                  f"R2 race {st['r2_race']:.4f} / 자원 {st['r2_resource']:.4f} / "
                  f"둘 {st['r2_both']:.4f}  (자원 순증분 {st['partial_r2_resource']:+.4f})")
            print(f"             race 계수 {st['race_coef']:+.3f} -> "
                  f"{st['race_coef_controlled']:+.3f}  "
                  f"감쇠 {100*st['race_attenuation']:+.1f}% "
                  f"[{100*lo:+.1f}%, {100*hi:+.1f}%]")
            row = {"min_n": min_n, "measure": measure, "n_counties": len(m),
                   **{k: round(v, 6) for k, v in st.items()},
                   "attenuation_lo": round(lo, 6), "attenuation_hi": round(hi, 6)}
            tabs.append(row)
            ledger += results.rows(
                "resource_audit",
                {k: st[k] for k in ("race_coef", "race_coef_controlled",
                                    "race_attenuation", "partial_r2_resource",
                                    "r2_race", "r2_resource", "r2_both")},
                model="graphsage_fairloss_a100_mb_cv5", tag=measure,
                group_set=f"n>={min_n}",
                params={"min_n": min_n, "measure": measure, "src": args.src,
                        "n_boot": args.n_boot},
                ci={"race_attenuation": (lo, hi)},
                notes={"n_counties": len(m),
                       "corr_z_resource": round(st["corr_z_resource"], 4),
                       "corr_z_race": round(st["corr_z_race"], 4)})

    if not tabs:
        raise SystemExit("[에러] 결과가 비었다.")
    results.write(ledger)
    out = C.scoped_output(args.out)
    pd.DataFrame(tabs).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out}")
    print("\n[해석] 헤드라인은 R²가 아니라 **race 계수 감쇠율**이다 — 질문이 'z를 "
          "무엇이 설명하는가'가 아니라 '인종 상관이 자원 이야기인가'이기 때문이다. "
          "감쇠가 0 근처면 인종 상관은 인력 규모로 설명되지 않는다. "
          "**두 가지를 반드시 함께 인용할 것**: (1) 프록시는 총 경찰관 수이지 형사 "
          "인원이 아니다(LEOKA에 그 열이 없다), (2) 인력이 사건 수에 반응해 배치되는 "
          "내생성 때문에 **관측 감쇠율은 실제보다 작다**(회귀변수의 분산이 경찰관 분산의 "
          "49%로 압축돼 있다). 즉 이 caveat은 결론에 불리한 쪽이다 — 진짜 자원 몫은 8%보다 "
          "클 수 있다. 정확한 문장은 '자원이 원인이 아니다'가 아니라 "
          "**'우리가 측정한 인력 규모로는 설명되지 않는다'**이다. z 자체는 이 스크립트가 "
          "건드리지 않으므로 기존 지도·보고서 인용은 전부 그대로 유효하다.")


if __name__ == "__main__":
    main()
