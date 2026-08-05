"""카운티를 고정하고 인종 잔차를 본다 — z의 4채널 중 셋을 산수로 지운다.

카운티차분 계획서 §4·§6의 (B)트랙. `detect_cold_blocks`의 z는 카운티 고유 효과를
통째로 담고(모델이 City를 특성으로 갖지 않는다) 그 안에서 차별·수사자원·도시성·
기록관행이 분리되지 않는다. 이 스크립트는 **공변량을 더하지 않고** 그 넷 중 셋을
지운다.

## 왜 차분이 회귀보다 강한가

네 채널은 전부 카운티 단위 속성이다. 회귀로 통제하려면 넷을 각각 측정해야 하는데,
`audit_resources`가 하나(인력)를 실제로 붙여 보고 확인한 한계가 그것이다 — 프록시가
뭉뚝하고, 내생적이고, 서로 강하게 상관돼 얻은 것은 하한뿐이었다.

차분은 채널을 **측정하지 않는다.** 채널이 카운티 안에서 일정하다는 사실만 쓴다.
웨인 카운티의 형사 수가 몇 명인지 몰라도 되고, 그 값이 카운티 안의 모든 사건에
**같게** 걸린다는 것만 알면 카운티 내부 비교에서 상쇄된다.

    카운티 절편으로 작용하는 채널  -> 소거 (수사자원 / 도시성 / 기록관행)
    인종 x 카운티 상호작용         -> 남음 (차별)

차별만 남는 것은 차별이 강해서가 아니라 **차별만이 상호작용으로 정의되기 때문**이다.
카운티 안의 모든 사건에 똑같이 걸리는 것은 정의상 차별이 아니다.

## 측정량

카운티 b, 인종 g에 대해 detect_cold_blocks와 **같은 정의로**

    O_bg = sum(1-y)      E_bg = sum(1-p_cal)      V_bg = sum q(1-q)
    SMR_bg = O_bg / E_bg

그리고 카운티 내부 인종 대비를 두 형태로 낸다.

    rho_b = log(SMR_Black / SMR_White)     1차. 규모 무관
            Var ~= V_Black/E_Black^2 + V_White/E_White^2   (델타법)
    d_b   = z_Black - z_White               2차. 기존 표와의 연속성

**왜 rho가 1차인가.** z는 sqrt(n)에 비례해 커진다 — 큰 카운티의 작은 격차가 작은
카운티의 큰 격차보다 큰 z 차이를 낼 수 있다. 카운티 공변량과 **상관시키는** 것이
목적이므로 규모 무관 효과크기가 맞다. 크로스피팅 트랙이 발견한 규모 의존성
(z ~ black_share이 +0.290/+0.385/+0.450으로 하한에 따라 움직인다)이 정확히 이
함정의 실례다.

## 왜 alpha=100 blind 모델이어야 하고, 왜 결과가 하한인가

E_bg가 **blind + 인종 완화** 모델에서 온다는 점이 이 설계를 성립시킨다.
sighted 모델을 쓰면 "피해자가 흑인이니 미해결 기대"가 E_bg에 구워져 rho_b는
구조적으로 0이 된다 — 재려는 현상이 기준선에 흡수되는 것이다. 완화 전 blind도
못 쓴다(FPR 인종 증폭 1.82). alpha=100만이 인종 중립적 기대값을 준다.

단 alpha=100의 주내 인종 격차는 여전히 음수이므로 E가 흑인비중 높은 곳에서
과대추정되고 -> O-E가 과소 -> **rho_b는 하한**이다. caveat이 결론에 불리한 쪽이다.

전역 보정 delta는 **한 숫자**라 인종별 차이를 흡수할 수 없다. 그래서 인종-차등
미보정이 잔차에 그대로 남고, 그것이 재려는 격차다 — delta를 인종별로 따로 적합하면
격차가 정의상 0이 된다(clear.predictions.calibrate 참고, §7-4가 이를 검사한다).

## 개별 카운티에 유의성 판정을 붙이지 않는다

질문은 "어느 카운티가 인종 차등적인가"가 아니라 "전국적으로 인종 차등이 있는가"다.
전자는 카운티당 인종별 표본이 수십 건이라 검정력이 없고, 순위를 만들면 D3 목록과
같은 문제(검증 경로 없음 + 잡음 지배)가 재발한다. BH FDR을 쓰지 않는 이유다.

출력: outputs[/{scope}]/county_race_residual{,_summary}.csv
      results.csv (family=county_race_residual)
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear import counties, predictions, results

# 대비할 두 그룹. 전국 test에서 min_n=5000을 넘는 것이 이 둘뿐이고(Asian/PI는 2,940),
# 공정성 벌점(alpha=100)이 누른 그룹집합과 같아야 잔차와 개입이 같은 축에 놓인다.
CONTRAST = ("Black", "White")
_EPS = 1e-9


def load_frame(model, calibrate_mode):
    """예측 덤프 + (State, City) + 보정된 p. detect_cold_blocks와 같은 조인 규약."""
    path = predictions.predictions_dir() / f"{model}.csv"
    if not path.exists():
        raise SystemExit(f"[에러] 덤프 없음: {path}\n"
                         f"  먼저 실행: python -m experiments.crossfit_predictions --minibatch")
    dump = predictions.load(path)
    sample = pd.read_parquet(C.SCOPE_DIR / "sample.parquet")
    df = dump.join(sample[["State", "City"]], on="row_index")
    if df[["State", "City"]].isna().any().any():
        raise ValueError("블록 키에 결측: row_index 정렬이 깨졌다.")
    # 같은 카운티의 옛 이름/새 이름을 합친다(detect_cold_blocks와 같은 정의여야
    # rho와 z가 같은 단위 위에 있다). clear.counties.canonicalize 참조.
    df = counties.canonicalize(df)
    p, cal = predictions.calibrate(dump["proba"].values, dump["y_true"].values,
                                  calibrate_mode)
    df["p_cal"] = p
    return df, cal


def block_table(df, race_col):
    """(State, City, race) -> O/E/V. **모든 인종**을 낸다 — §7-4의 행 보존 검사가
    Unknown 포함 합계를 cold_blocks 표와 대조하기 때문이다."""
    g = df.assign(_q=1.0 - df["p_cal"].values,
                  _u=(df["y_true"].values == 0).astype(float),
                  _v=lambda d: d["_q"] * (1.0 - d["_q"]))
    agg = (g.groupby(["State", "City", race_col], sort=False)
             .agg(n=("_u", "size"), O=("_u", "sum"), E=("_q", "sum"), V=("_v", "sum"))
             .reset_index())
    return agg


def contrast(agg, race_col, min_race_n, haldane):
    """인종별 표 -> 카운티당 한 행(rho, d, 분산). 두 그룹이 모두 하한을 넘는 카운티만."""
    b, w = CONTRAST
    sub = agg[agg[race_col].isin(CONTRAST)]
    wide = sub.pivot_table(index=["State", "City"], columns=race_col,
                           values=["n", "O", "E", "V"], aggfunc="first")
    wide.columns = [f"{a}_{c}" for a, c in wide.columns]
    wide = wide.reset_index().dropna(
        subset=[f"n_{b}", f"n_{w}", f"E_{b}", f"E_{w}"])
    m = wide[(wide[f"n_{b}"] >= min_race_n) & (wide[f"n_{w}"] >= min_race_n)].copy()
    if m.empty:
        return m

    # Haldane-Anscombe: O=0이면 log가 정의되지 않는다. 분자·분모에 0.5를 더하는
    # 표준 보정이고, 안 쓴 버전을 민감도로 함께 낸다(--haldane none).
    adj = 0.5 if haldane else 0.0
    for g in (b, w):
        m[f"smr_{g}"] = (m[f"O_{g}"] + adj) / np.maximum(m[f"E_{g}"] + adj, _EPS)
        # z는 detect_cold_blocks와 같은 식. rho와 달리 규모에 비례한다.
        m[f"z_{g}"] = ((m[f"O_{g}"] - m[f"E_{g}"])
                       / np.sqrt(np.maximum(m[f"V_{g}"], _EPS)))
    m["rho"] = np.log(np.maximum(m[f"smr_{b}"], _EPS)
                      / np.maximum(m[f"smr_{w}"], _EPS))
    # 델타법: Var(log(O/E)) ~= Var(O)/E^2 (E는 모델이 준 상수로 취급).
    m["rho_var"] = (m[f"V_{b}"] / np.maximum(m[f"E_{b}"], _EPS) ** 2
                    + m[f"V_{w}"] / np.maximum(m[f"E_{w}"], _EPS) ** 2)
    m["d_z"] = m[f"z_{b}"] - m[f"z_{w}"]
    m["n_both"] = m[f"n_{b}"] + m[f"n_{w}"]
    m["black_share_both"] = m[f"n_{b}"] / m["n_both"]
    return m.reset_index(drop=True)


def _wmean(x, w):
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(np.asarray(x)[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def summarize(m, extra_corr, n_boot, seed):
    """헤드라인 통계 + 카운티 복원추출 CI.

    관측 단위가 카운티이므로 **카운티를 뽑는다**(audit_resources와 같은 규약).
    사건을 뽑으면 큰 카운티가 반복 등장해 카운티 간 분산을 과소평가한다.
    """
    rho = m["rho"].to_numpy(float)
    iv = 1.0 / np.maximum(m["rho_var"].to_numpy(float), _EPS)   # 역분산 가중
    black = m["black_share_both"].to_numpy(float)

    def _stats(idx):
        r, w, bs = rho[idx], iv[idx], black[idx]
        out = {"rho_wmean": _wmean(r, w), "rho_mean": float(np.nanmean(r)),
               "d_z_mean": float(np.nanmean(m["d_z"].to_numpy(float)[idx]))}
        for name, v in [("black_share", bs), *extra_corr]:
            vv = np.asarray(v)[idx]
            ok = np.isfinite(r) & np.isfinite(vv)
            out[f"corr_rho_{name}"] = (float(np.corrcoef(r[ok], vv[ok])[0, 1])
                                       if ok.sum() > 2 else np.nan)
        return out

    base = _stats(np.arange(len(m)))
    rng = np.random.default_rng(seed)
    draws = [_stats(rng.integers(0, len(m), len(m))) for _ in range(n_boot)]
    ci = {}
    for k in base:
        v = np.array([d[k] for d in draws], dtype=float)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        ci[k] = (float(lo), float(hi))
    return base, ci


def check_row_conservation(agg, src, min_n):
    """§7-4 행 보존 + 보정 단일성. 실패하면 즉시 멈춘다."""
    path = C.scoped_output(src)
    if not path.exists():
        print(f"[검수] {path.name} 없음 - 행 보존 검사 건너뜀")
        return
    cb = pd.read_csv(path)
    cb = cb[(cb.block_key == "county") & (cb.min_n == min_n)]
    if cb.empty:
        print(f"[검수] {src}에 min_n={min_n} 행 없음 - 건너뜀")
        return
    # 모든 인종(Unknown 포함) 합계여야 한다. 두 그룹만 더하면 당연히 안 맞는다.
    tot = (agg.groupby(["State", "City"])[["n", "O", "E"]].sum()
              .rename(columns={"n": "n_all", "O": "O_all", "E": "E_all"})
              .reset_index())
    j = cb.merge(tot, on=["State", "City"], how="inner")
    dn = int((j["n_unsolved"] - j["O_all"]).abs().max())
    de = float((j["expected_unsolved"] - j["E_all"]).abs().max())
    print(f"[검수] 행 보존: 카운티 {len(j):,}개 대조, "
          f"max |O 차이| = {dn}, max |E 차이| = {de:.2e}")
    if dn != 0 or de > 1e-6:
        raise SystemExit("[에러] 인종 분할이 행을 잃거나 중복시켰다(또는 delta를 "
                         "인종별로 적합했다). 계획서 §4-5 실패 조건.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="graphsage_fairloss_a100_mb_cv5",
                    help="예측 덤프 라벨. 기본은 크로스피팅 전수 덤프 — 카운티 안에서 "
                         "다시 인종으로 쪼개므로 test 분할(카운티당 평균 63행)로는 "
                         "검정력이 없다.")
    ap.add_argument("--min_race_n", type=int, nargs="+", default=[10, 20, 30],
                    help="카운티당 **두 인종 각각**의 최소 표본. 여러 개를 주면 각각에 "
                         "대해 따로 낸다 — 하나만 내면 사후에 유리한 하한을 고른 것으로 "
                         "보인다(저장소 규율).")
    ap.add_argument("--calibrate", choices=["shift", "none"], default="shift")
    ap.add_argument("--haldane", choices=["on", "off"], default="on",
                    help="O=0인 카운티의 log를 정의하기 위한 +0.5 보정")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--src", default="cold_blocks_cv5.csv",
                    help="행 보존 검사에 쓸 detect_cold_blocks 표(같은 덤프에서 나온 것)")
    ap.add_argument("--check_min_n", type=int, default=20)
    ap.add_argument("--out", default="county_race_residual.csv")
    args = ap.parse_args()

    race_col = f"sens__{C.SENSITIVE_COLS[0]}"          # sens__Victim Race
    df, cal = load_frame(args.model, args.calibrate)
    print(f"[load] {args.model}: {len(df):,}행, 미해결 {int((df.y_true == 0).sum()):,}건")
    print(f"[calib] delta={cal['delta']:+.4f}  기대 검거 {cal['sum_p_before']:,.0f} -> "
          f"{cal['sum_p_after']:,.0f} (실제 {cal['sum_y']:,.0f})")

    agg = block_table(df, race_col)
    check_row_conservation(agg, args.src, args.check_min_n)

    per_county, summary, ledger = [], [], []
    for min_race_n in args.min_race_n:
        m = contrast(agg, race_col, min_race_n, args.haldane == "on")
        if len(m) < 30:
            print(f"[경고] 인종당 n>={min_race_n}: 카운티 {len(m)}개뿐 - 건너뜀")
            continue
        base, ci = summarize(m, [], args.n_boot, C.RANDOM_STATE)
        m2 = m.assign(min_race_n=min_race_n)
        per_county.append(m2)

        print(f"\n=== 인종당 n>={min_race_n} : 카운티 {len(m):,}개 "
              f"(사건 {int(m['n_both'].sum()):,}건) ===")
        print(f"  rho 역분산가중 평균 {base['rho_wmean']:+.4f} "
              f"[{ci['rho_wmean'][0]:+.4f}, {ci['rho_wmean'][1]:+.4f}]   "
              f"(비가중 {base['rho_mean']:+.4f} "
              f"[{ci['rho_mean'][0]:+.4f}, {ci['rho_mean'][1]:+.4f}])")
        print(f"  d_z 평균 {base['d_z_mean']:+.3f} "
              f"[{ci['d_z_mean'][0]:+.3f}, {ci['d_z_mean'][1]:+.3f}]")
        print(f"  corr(rho, black_share) {base['corr_rho_black_share']:+.3f} "
              f"[{ci['corr_rho_black_share'][0]:+.3f}, "
              f"{ci['corr_rho_black_share'][1]:+.3f}]")
        print(f"  rho>0 카운티 {int((m['rho'] > 0).sum()):,}/{len(m):,} "
              f"({100 * (m['rho'] > 0).mean():.1f}%)")

        row = {"min_race_n": min_race_n, "n_counties": len(m),
               "n_cases": int(m["n_both"].sum()), "haldane": args.haldane,
               **{k: round(v, 6) for k, v in base.items()},
               **{f"{k}_lo": round(v[0], 6) for k, v in ci.items()},
               **{f"{k}_hi": round(v[1], 6) for k, v in ci.items()}}
        summary.append(row)
        ledger += results.rows(
            "county_race_residual", base,
            model=args.model, tag="rho", attribute=C.SENSITIVE_COLS[0],
            blind=True, group_set=f"race_n>={min_race_n}",
            params={"min_race_n": min_race_n, "haldane": args.haldane,
                    "calibrate": args.calibrate, "n_boot": args.n_boot,
                    "contrast": "|".join(CONTRAST)},
            ci=ci,
            notes={"n_counties": len(m), "n_cases": int(m["n_both"].sum()),
                   "delta": round(cal["delta"], 4)})

    if not per_county:
        raise SystemExit("[에러] 어느 하한에서도 카운티가 충분하지 않다.")
    results.write(ledger)
    out = C.scoped_output(args.out)
    pd.concat(per_county, ignore_index=True).to_csv(out, index=False,
                                                    encoding="utf-8-sig")
    sm = C.scoped_output(args.out.replace(".csv", "_summary.csv"))
    pd.DataFrame(summary).to_csv(sm, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out}\n[save] {sm}")
    print("\n[해석] rho > 0 은 **같은 카운티 안에서** 흑인 피해자 사건이 인종 중립적 "
          "기준선 대비 더 많이 미해결로 남는다는 뜻이다. 카운티 절편으로 작용하는 "
          "수사자원·도시성·기록관행은 인종 상호작용 없이 이 값을 만들 수 없으므로, "
          "그 셋의 '절편' 설명은 배제된다. 배제되지 **않는** 것 셋(계획서 §8): "
          "카운티 내부 사건구성(Circumstance가 데이터에 없다), 채널x인종 상호작용"
          "(자원이 인종-차등 트리아지로 작용하는 경우 — 다만 그것도 제도적 차별이다), "
          "인종-차등 기록관행. 그리고 alpha=100의 잔여 편향 때문에 rho는 **하한**이다.")


if __name__ == "__main__":
    main()
