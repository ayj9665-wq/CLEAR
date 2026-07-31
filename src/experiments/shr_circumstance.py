"""SHR 정황 기록이 얼마나 결과·인종에 의존하는가 — 누수 §8-3의 측정.

카운티차분 계획서 §8이 카운티 차분으로 닫히지 않는 누수 셋을 남겼다. 이 스크립트는
그중 **인종차등 기록관행**(§8-3)을 재고, 동시에 **사건구성**(§8-1)이 이 데이터로는
닫을 수 없음을 확정한다. SHR 정황 조인 계획서 §1(G1 게이트)의 실행이다.

## 왜 정황을 모델에 넣지 않고 이것만 재는가

`Circumstance`는 Kaggle CSV에 없고 SHR 원본에 있다. 그것으로 `E_b`를 조정하면 §8-1의
누수를 닫을 수 있어 보이지만, **기록된 정황은 결과의 하류(downstream)다.** 인과 그림으로:

    R  피해자 인종        C* 진짜 정황(관측 불가)     Y 해결 여부
    C  기록된 정황        D  차별/수사 노력(재려는 것)

    R -> C* -> Y          교란 경로. C*를 통제하면 막힌다 <- 원래 하려던 것
    R -> D  -> Y          목표 경로
    C* -> C <- Y          **C는 충돌부(collider)다**

마지막 줄이 전부다. 아래 §1이 재는 것이 `Y -> C` 화살표의 존재이고, 그것이 있으면
C를 통제하는 순간 충돌부가 열려 C*와 Y 사이에 인과 아닌 연관이 생긴다. R -> C*이므로
그 연관이 R과 Y 사이로 흘러든다. **편향의 크기가 아니라 부호조차 모른다.** 즉 C를
통제하는 것은 C*를 통제하는 것의 근사가 아니라 **다른 연산**이다.

모델 입력에 넣지 않는 이유는 하나 더 있고 그쪽이 실무적으로 결정적이다 — 커밋된 평면
덤프 4개는 현재 특성 집합으로 학습됐고 `train_baseline.py`가 삭제돼 재생성이 불가능하다.
정황을 X에 넣으면 GraphSAGE(정황 있음) 대 XGBoost(정황 없음)이 되어 이 프로젝트의
헤드라인(+0.82 및 짝지은 카운티 표준화 대조)이 영구히 죽는다.

## 재는 것 셋

  1. **결과 의존성**  P(정황 = undetermined | 해결 여부)
     `Y -> C` 화살표가 있는가. 단, 이 값 **단독으로는 모호하다** — 아래 3이 필요하다.
  2. **인종 차등**    같은 카운티 안에서 P(undetermined | 미해결, 인종)의 차이
     `D -> C` 화살표. §8-3의 직접 측정.
  3. **판별 검정**    카운티 검거율과 범주별 기록의 상관
     1의 모호성을 푼다(아래).

## 왜 1 단독으로는 모호하고 3이 필요한가

"미해결에서 undetermined가 많다"는 두 가지로 설명된다. (a) 안 풀려서 정황을 못 적었다
(오염), (b) 원래 정황을 알 수 없는 사건이 안 풀린다(진짜 사건 구성). 횡단면은 둘을
못 가른다.

**흔한 오진 하나를 미리 막아 둔다**: "미해결내 비중 / 해결내 비중"을 오염 지표로 쓰면
안 된다. 정의상

    그 비 = (1-q_c)/q_c x (N_해결/N_미해결)

이라 범주 검거율 q_c의 단조변환일 뿐이고 정보가 0이다(실측으로 소수 둘째 자리까지
일치한다). 관측된 q_c 자체가 오염돼 있으므로 순환이기도 하다.

3이 그것을 푼다. 카운티 간 검거율 차이를 쓰면 **부호가 갈리는** 예측이 나온다:

  - 오염이면 -> 가해자를 알아야 판정되는 범주(other arguments, lovers triangle...)는
    검거율 낮은 카운티에서 미해결 중 비중이 **더 낮다**(undetermined로 더 빠지므로).
  - 진짜 사건 구성이면 -> 저검거 카운티에 강도·갱 살인이 많으므로 중범죄형 범주 비중이
    **더 높다**.

두 힘의 부호가 다르므로, 가해자 의존 범주에 대해서는 단측으로 읽을 수 있다.

## 하지 않는 것

정황으로 `E_b`를 조정해 rho의 하한을 만드는 것(SHR 계획서 §4)은 **하지 않는다.**
위 충돌부 논증 때문이고, 실측이 그 논증을 뒷받침하기 때문이다 — 미해결의 56%가
undetermined이고 그 코딩이 인종에 따라 다르므로, 조정은 "사건 구성"이 아니라
"흑인 피해자 미해결 사건에 정황을 덜 기록했다"를 흡수한다. 그건 설명이 아니라 세탁이고,
`detect_cold_blocks`가 인종에 대해, `audit_resources`가 인력에 대해 이미 거부한 것이다.

따라서 **§8-1(카운티 내부 사건구성)은 이 데이터로 닫을 수 없다**가 이 트랙의 결론이다.
열려 있던 질문에 확정된 답이 붙는 것이지 실패가 아니다.

## 데이터

SHR 원본(Kaplan 통합본, **openICPSR 100699** — LEOKA의 102180과 다른 deposit).
`dataset/geo/shr_1976_2016_csv/`. 조인 키는 `(ori, year, month, incident_number)`이고
원본 Kaggle CSV의 `(Agency Code, Year, Month, Incident)`와 그대로 붙는다 — `Agency Code`는
`01_clean.py`가 떨어뜨리므로 **원본 CSV에서 읽는다**(`audit_resources`와 같은 규약이고,
안전한 이유도 같다 — 결과를 카운티 단위로만 쓴다).

출력: outputs[/{scope}]/shr_recording_{summary,by_county,by_circumstance}.csv
      results.csv (family=shr_recording)
"""
import argparse

import numpy as np
import pandas as pd

import config as C

SHR_PATH = (C.ROOT / "dataset" / "geo" / "shr_1976_2016_csv" / "shr_1976_2016.csv")
KEY = ["ori", "year", "month", "incident"]
CONTRAST = ("Black", "White")

# 가해자를 알아야 판정되는 범주. §3의 판별 검정에서 **부호 예측이 있는 쪽**이다.
# 이 목록은 결과를 보기 전에 고정한다 -- 사후에 고치면 검정이 무의미해진다.
OFFENDER_DEPENDENT = [
    "other arguments", "argument over money or property", "lovers triangle",
    "brawl due to influence of alcohol", "brawl due to influence of narcotics",
    "child killed by babysitter", "felon killed by private citizen",
]
# 현장에서 판정 가능한 중범죄형. 대조군(부호 예측이 반대이거나 없다).
SCENE_DETERMINED = [
    "robbery", "burglary", "rape", "arson", "motor vehicle theft", "larceny",
    "narcotic drug laws", "juvenile gang killings", "gangland killings",
]


def load_joined():
    """Kaggle 행(피해자 단위) + SHR 정황(사건 단위)을 좌측 조인. 행 수는 보존된다."""
    if not SHR_PATH.exists():
        raise SystemExit(
            f"[에러] SHR 없음: {SHR_PATH}\n"
            f"  openICPSR 100699(Kaplan SHR 통합본)에서 받아 dataset/geo/ 에 풀 것.\n"
            f"  1976-2017을 덮는 V7 이상이면 어떤 버전이든 충분하다(gitignore 됨).")
    s = pd.read_csv(SHR_PATH, low_memory=False,
                    dtype={"ori": str, "incident_number": str},
                    usecols=["ori", "year", "month_of_offense", "incident_number",
                             "offender_1_circumstance"])
    s = s.rename(columns={"month_of_offense": "month",
                          "incident_number": "incident",
                          "offender_1_circumstance": "circ"})
    s["month"] = s["month"].str.strip().str.lower()
    # SHR은 사건 단위 와이드 포맷이라 키가 거의 유일하다(실측 중복 28/586,670 = 0.00%).
    # 중복을 남기면 좌측 조인이 Kaggle 행을 늘려 아래 행 보존 검사가 걸린다.
    n_dup = int(s.duplicated(subset=KEY).sum())
    s = s.drop_duplicates(subset=KEY)

    k = pd.read_csv(C.RAW_CSV, dtype={"Agency Code": str, "Incident": str},
                    usecols=["Agency Code", "Year", "Month", "Incident",
                             "Crime Solved", "Victim Race", "State", "City"])
    k = k.rename(columns={"Agency Code": "ori", "Year": "year", "Month": "month",
                          "Incident": "incident"})
    k["month"] = k["month"].str.strip().str.lower()
    if C.SCOPE != C.DEFAULT_SCOPE or C.SAMPLE_STATES:
        k = k[k["State"].isin(C.SAMPLE_STATES)] if C.SAMPLE_STATES else k

    n0 = len(k)
    m = k.merge(s[KEY + ["circ"]], on=KEY, how="left")
    if len(m) != n0:
        raise SystemExit(f"[에러] 행 보존 실패 {n0:,} -> {len(m):,}. 조인 키가 부족하다.")
    m["matched"] = m["circ"].notna()
    m["und"] = m["circ"].str.contains("undetermin", case=False, na=False)
    m["solved"] = m["Crime Solved"] == "Yes"
    print(f"[join] Kaggle {n0:,}행 <- SHR {len(s):,}건 (키 중복 {n_dup} 제거)")
    return m


def gate_match(m):
    """G2 — 매칭률이 인종과 상관되면 새 편향이 들어온다. 통과 기준은 사전등록값."""
    by_r = m.groupby("Victim Race")["matched"].mean() * 100
    overall = 100 * m["matched"].mean()
    gap = abs(by_r.get(CONTRAST[1], np.nan) - by_r.get(CONTRAST[0], np.nan))
    print(f"\n[G2 매칭 편향] 전체 {overall:.2f}%  "
          + "  ".join(f"{g} {v:.2f}%" for g, v in by_r.items()))
    print(f"           White-Black 차이 {gap:.2f}%p  "
          f"-> {'통과' if overall >= 95 and gap < 1.0 else '실패'} "
          f"(기준: 전체>=95%, 차이<1%p)")
    if overall < 95 or gap >= 1.0:
        raise SystemExit("[중단] G2 실패. 매칭 실패가 인종과 상관되면 이 트랙이 "
                         "재려는 편향을 스스로 만들어낸다.")
    return {"match_rate": overall, "match_gap_wb": gap}


def measure_outcome_dependence(m):
    """측정 1 — P(undetermined | 해결 여부). `Y -> C` 화살표.

    **단독으로는 모호하다**(docstring 참고). measure_heterogeneity가 푼다.
    """
    mm = m[m["matched"]]
    by = mm.groupby("solved")["und"].mean() * 100
    tab = (mm.pivot_table(index="solved", columns="Victim Race", values="und",
                          aggfunc="mean") * 100)
    print(f"\n[측정 1] P(정황 = undetermined | 해결 여부)")
    print(f"  해결 {by.get(True, np.nan):.1f}%  vs  미해결 {by.get(False, np.nan):.1f}%"
          f"   차이 {by.get(False, 0) - by.get(True, 0):+.1f}%p")
    cols = [c for c in CONTRAST if c in tab.columns]
    print(tab[cols].round(1).rename_axis("해결").to_string())
    print("  비교 기준 — config.LEAKAGE_COLS의 가해자 열은 미해결에서 90~99%가 Unknown이다.")
    return {"und_solved": float(by.get(True, np.nan)),
            "und_unsolved": float(by.get(False, np.nan))}


def measure_race_gap(m, floors, n_boot, seed):
    """측정 2 — **같은 카운티 안에서** undetermined 코딩의 인종 격차. `D -> C`.

    카운티를 고정하는 이유는 카운티차분 트랙과 같다: 기관 관행이 카운티 절편으로
    작용하는 한 차분에서 소거되고, 남는 것은 인종 상호작용뿐이다.
    CI는 **카운티**를 복원추출한다(관측 단위가 카운티다).
    """
    uns = m[m["matched"] & ~m["solved"] & m["Victim Race"].isin(CONTRAST)]
    pooled = uns.groupby("Victim Race")["und"].mean() * 100
    p_gap = pooled.get(CONTRAST[0], np.nan) - pooled.get(CONTRAST[1], np.nan)
    print(f"\n[측정 2] undetermined 코딩의 인종 격차 ({CONTRAST[0]} - {CONTRAST[1]})")
    print(f"  pooled: {CONTRAST[1]} {pooled.get(CONTRAST[1], np.nan):.1f}%  "
          f"{CONTRAST[0]} {pooled.get(CONTRAST[0], np.nan):.1f}%  차이 {p_gap:+.1f}%p")

    t = (uns.groupby(["State", "City", "Victim Race"])["und"]
            .agg(["mean", "size"]).reset_index())
    w = t.pivot_table(index=["State", "City"], columns="Victim Race",
                      values=["mean", "size"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna().reset_index()

    rows, out = [], {"race_gap_pooled": float(p_gap)}
    rng = np.random.default_rng(seed)
    for floor in floors:
        c = w[(w[f"size_{CONTRAST[0]}"] >= floor)
              & (w[f"size_{CONTRAST[1]}"] >= floor)].copy()
        if len(c) < 30:
            print(f"  [경고] 하한 {floor}: 카운티 {len(c)}개뿐 — 건너뜀")
            continue
        d = (c[f"mean_{CONTRAST[0]}"] - c[f"mean_{CONTRAST[1]}"]).to_numpy() * 100
        wt = (c[f"size_{CONTRAST[0]}"] + c[f"size_{CONTRAST[1]}"]).to_numpy(float)
        draws = [np.average(d[i], weights=wt[i])
                 for i in (rng.integers(0, len(c), len(c)) for _ in range(n_boot))]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        wm = float(np.average(d, weights=wt))
        print(f"  카운티 내부(하한 {floor:2d}): 카운티 {len(c):4d}개  "
              f"{wm:+.1f}%p [{lo:+.1f}, {hi:+.1f}]  (비가중 {d.mean():+.1f}%p)")
        out[f"race_gap_within_county_n{floor}"] = wm
        out[f"race_gap_within_county_n{floor}_lo"] = float(lo)
        out[f"race_gap_within_county_n{floor}_hi"] = float(hi)
        rows.append(c.assign(min_race_n=floor, gap_pp=d, weight=wt))
    return out, (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())


def measure_heterogeneity(m, min_unsolved, n_boot, seed):
    """측정 3 — 카운티 검거율과 정황 기록의 상관. **사전등록한 예측이 빗나갔다.**

    설계 의도는 측정 1의 모호성을 부호로 가르는 것이었다. 범주 c마다
    corr(카운티 검거율, 미해결 사건 중 c의 비중)을 내면,

      오염 가설       -> 가해자 의존 범주는 **양의 상관**
                         (검거율 높은 카운티일수록 정황을 더 알아내 c로 남긴다)
      사건 구성 가설  -> 중범죄형 범주는 **음의 상관**
                         (저검거 카운티에 강도·갱 살인이 실제로 많다)

    **실측은 그렇게 갈리지 않는다**(전국 275개 카운티): 가해자 의존 중앙값 r = +0.007,
    현장 결정 중앙값 +0.106 — 예측과 반대로 후자가 더 크다. 가장 큰 가해자 의존 범주인
    `other arguments`는 -0.149로 오염 예측의 반대편이다. 반면 `narcotic drug laws`
    (-0.296)와 `juvenile gang killings`(-0.241)의 음의 상관은 **진짜 사건 구성**과 깨끗이
    맞는다(저검거 카운티에 마약·갱 살인이 실제로 많다).

    구성비 제약을 의심해 분모에서 undetermined를 빼고 다시 계산했으나 결과는 그대로다
    (가해자 의존 +0.007, 현장 결정 +0.106). 그리고 가장 직접적인 한 숫자인
    corr(검거율, 미해결 중 undetermined 비중)은 **+0.090**으로, 오염 가설이 예측하는
    음수가 아니다.

    **그래서 측정 3은 측정 1을 가르지 못한다.** 그 사실을 지우지 않고 남긴다.

    다만 이것이 결론을 뒤집지는 않는다. 이유 둘:
      1. "circumstances undetermined"는 사건의 속성이 아니라 **수사가 밝히지 못했다는
         진술**이다. 의미상 이미 결과의 하류이고, 카운티 간 상관이 안 나온다고 해서
         그 의미가 바뀌지 않는다.
      2. 결론은 측정 3이 아니라 **측정 2**에 선다. 같은 카운티 안에서 C가 R에 의존하면
         (실측 +4.1%p), C로 통제하는 것은 R을 담은 변수로 통제하는 것이다. 기제가
         Y->C든 D->C든 인종 대비에는 똑같이 무효다.

    범주 분류는 결과를 보기 전에 고정했다(모듈 상단 상수). 빗나간 뒤에도 고치지 않는다.
    """
    mm = m[m["matched"]].copy()
    g = mm.groupby(["State", "City"])
    cl = g["solved"].mean().rename("clearance")
    n_uns = g.apply(lambda d: int((~d["solved"]).sum()), include_groups=False
                    ).rename("n_unsolved")
    base = pd.concat([cl, n_uns], axis=1)
    base = base[base["n_unsolved"] >= min_unsolved]
    uns = mm[~mm["solved"]]
    tot = uns.groupby(["State", "City"]).size()
    share = (uns.groupby(["State", "City", "circ"]).size()
                / tot).rename("share").reset_index()
    print(f"\n[측정 3] 카운티 검거율 x 미해결 정황 구성 (카운티 {len(base):,}개, "
          f"미해결 >= {min_unsolved})")

    # 가장 직접적인 한 숫자. 오염 가설이면 **음수**여야 한다.
    u = (uns[uns["und"]].groupby(["State", "City"]).size() / tot).rename("u")
    ju = base.join(u).fillna({"u": 0.0})
    r_und = float(np.corrcoef(ju["clearance"], ju["u"])[0, 1])
    print(f"  corr(검거율, 미해결 중 undetermined 비중) = {r_und:+.3f}  "
          f"-> 오염 가설의 예측(음수)과 {'일치' if r_und < 0 else '불일치'}")

    rng = np.random.default_rng(seed)
    rows = []
    for circ in OFFENDER_DEPENDENT + SCENE_DETERMINED:
        sub = share[share["circ"] == circ].set_index(["State", "City"])["share"]
        j = base.join(sub, how="left").fillna({"share": 0.0})
        if len(j) < 30:
            continue
        x, y = j["clearance"].to_numpy(), j["share"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        draws = [float(np.corrcoef(x[i], y[i])[0, 1])
                 for i in (rng.integers(0, len(j), len(j)) for _ in range(n_boot))]
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        fam = "가해자의존" if circ in OFFENDER_DEPENDENT else "현장결정"
        rows.append({"circumstance": circ, "family": fam, "n_counties": len(j),
                     "corr_clearance": r, "lo": float(lo), "hi": float(hi)})
    t = pd.DataFrame(rows).sort_values(["family", "corr_clearance"],
                                       ascending=[True, False])
    for fam, sub in t.groupby("family"):
        sig = int(((sub["lo"] > 0) | (sub["hi"] < 0)).sum())
        pos = int((sub["corr_clearance"] > 0).sum())
        print(f"  [{fam}] {len(sub)}개 범주 중 양의 상관 {pos}개, CI가 0 제외 {sig}개  "
              f"(중앙값 r = {sub['corr_clearance'].median():+.3f})")
    print(t[["family", "circumstance", "n_counties", "corr_clearance", "lo", "hi"]]
          .round(3).to_string(index=False))
    med = t.groupby("family")["corr_clearance"].median()
    if med.get("가해자의존", 0) <= med.get("현장결정", 0):
        print("  >>> 사전등록한 부호 분리가 **나오지 않았다**(가해자의존 <= 현장결정). "
              "측정 3은 측정 1을 가르지 못한다 — docstring 참고. 결론은 측정 2에 선다.")
    return t, r_und


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_race_n", type=int, nargs="+", default=[10, 20, 30],
                    help="측정 2에서 카운티당 **두 인종 각각**의 최소 미해결 표본")
    ap.add_argument("--min_unsolved", type=int, default=50,
                    help="측정 3에서 카운티가 가져야 하는 최소 미해결 사건 수")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_prefix", default="shr_recording")
    args = ap.parse_args()

    m = load_joined()
    stats = gate_match(m)
    stats.update(measure_outcome_dependence(m))
    gap_stats, by_county = measure_race_gap(m, args.min_race_n, args.n_boot,
                                            C.RANDOM_STATE)
    stats.update(gap_stats)
    by_circ, r_und = measure_heterogeneity(m, args.min_unsolved, args.n_boot,
                                           C.RANDOM_STATE)
    stats["corr_clearance_undetermined"] = r_und

    from clear import results
    ledger = results.rows(
        "shr_recording",
        {k: v for k, v in stats.items() if isinstance(v, float)
         and not k.endswith(("_lo", "_hi"))},
        model="(none)", tag="circumstance", attribute=C.SENSITIVE_COLS[0],
        group_set=f"{CONTRAST[0]} vs {CONTRAST[1]}",
        params={"min_race_n": ",".join(map(str, args.min_race_n)),
                "min_unsolved": args.min_unsolved, "n_boot": args.n_boot,
                "source": "openICPSR 100699"},
        ci={f"race_gap_within_county_n{f}":
            (stats.get(f"race_gap_within_county_n{f}_lo"),
             stats.get(f"race_gap_within_county_n{f}_hi"))
            for f in args.min_race_n
            if f"race_gap_within_county_n{f}" in stats},
        notes={"n_counties_hetero": int(by_circ["n_counties"].max())
               if len(by_circ) else 0})
    results.write(ledger)
    pd.DataFrame([stats]).to_csv(
        C.scoped_output(f"{args.out_prefix}_summary.csv"), index=False,
        encoding="utf-8-sig")
    if len(by_county):
        by_county.to_csv(C.scoped_output(f"{args.out_prefix}_by_county.csv"),
                         index=False, encoding="utf-8-sig")
    by_circ.to_csv(C.scoped_output(f"{args.out_prefix}_by_circumstance.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"\n[save] {C.scoped_output(args.out_prefix + '_summary.csv')} 외 2개")
    print("\n[해석] 결론은 **측정 2**에 선다 — 같은 카운티 안에서 기록된 정황 C가 "
          "피해자 인종 R에 의존하므로(+4%p대), C로 통제하는 것은 R을 담은 변수로 통제하는 "
          "일이다. 인과 그림에서 C는 충돌부이고, 통제하면 편향의 크기가 아니라 **부호조차** "
          "모르게 된다. 따라서 정황으로 E_b를 조정하는 것은 인종 격차를 설명하는 것이 "
          "아니라 흡수하는 것이며, **카운티차분 §8-1(카운티 내부 사건구성)은 이 데이터로 "
          "닫을 수 없다** — 실패가 아니라 열려 있던 질문에 붙은 확정된 답이다. "
          "측정 2는 동시에 **§8-3(인종차등 기록관행)의 직접 증거**다. "
          "측정 3은 측정 1을 가르려던 검정인데 **사전등록한 부호 분리가 나오지 않았다**; "
          "지우지 않고 남긴다(docstring 참고).")


if __name__ == "__main__":
    main()
