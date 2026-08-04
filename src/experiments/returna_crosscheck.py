"""
experiments/returna_crosscheck.py -- `Crime Solved`를 독립된 두 번째 측정과 대조한다.

계획서: reports/ReturnA교차검증_계획서_CLEAR.md (사전 등록, 롤백 태그 `pre-returna`)

## 무엇을 재는가

`Crime Solved`는 SHR에서 유도한 **하나의** 측정이다. UCR Return A는 **같은 기관이 같은
사건에 대해 낸 독립된 두 번째 측정**이다. 두 측정이 어긋나면 그 어긋남은 차별일 수도
수사자원일 수도 없다 -- 같은 경찰활동을 재고 있으므로 정의상 **기록 문제**다. 그
크기가 `rho = +0.084`(county_race_residual)에 붙는 **오염 상한**이 된다.

이 트랙이 겨냥하는 것은 SHR 정황 트랙이 남긴 물음이다. 같은 카운티 안에서 흑인 피해자
미해결 사건이 `undetermined`로 기록될 확률이 +4.1%p 높은데, 그것을 두 가지로 읽을 수
있고 뜻이 정반대다.

    형제(sibling)  같은 수사 노력 부족이 기록도 검거도 덜하게 만든다 -> rho의 독립 증거
    오염(artifact) 실제로 종결된 사건이 미해결로 **기록**됐다      -> rho의 일부가 가짜

차이는 "`Crime Solved`가 정확한 측정인가"로 환원되고, 그건 두 번째 측정이 있어야만
답할 수 있다.

## 시점 규약 -- 이걸 어기면 정상적인 차이를 오염으로 오독한다

    SHR      사건 1건 단위. 검거는 **스냅샷 시점의 상태 플래그**. 귀속은 발생 연도
    Return A 기관-월 집계. 검거는 **검거가 일어난 달**의 카운트

즉 1990년에 발생해 1992년에 풀린 사건은 SHR에서 1990년의 '검거'이고 Return A에서는
1992년의 '검거'다. **연도별 검거 수를 비교하면 안 된다** -- 누적으로만 본다.
이 파일에 연도별 검거 비교가 생기면 설계 위반이다(계획서 §4-3).

## 이 트랙은 rho를 재계산하지 않는다

`county_race_residual.csv`를 **읽어서** 쓴다. 재계산하면 두 트랙의 rho가 갈릴 수 있고,
이 트랙의 목적은 기존 수치를 바꾸는 것이 아니라 그 **불확실성**을 재는 것이다.
실행 전후로 그 파일이 바이트 동일해야 한다(계획서 §8).

출력: outputs[/{scope}]/returna_{summary,by_county}.csv, results.csv (family=returna)
"""
import argparse
import re

import numpy as np
import pandas as pd

import config as C
from clear import counties, results

GEO_DIR = C.ROOT / "dataset" / "geo"

# Kaplan 통합본의 열 이름은 버전마다 다르다. 코드북을 못 본 채로 이름을 박아 두면
# 조용히 0건이 잡히거나 엉뚱한 열을 읽는다. 그래서 **패턴으로 찾고 무엇을 찾았는지
# 출력한다** -- 못 찾으면 후보 목록을 보여 주고 죽는다.
COLUMN_PATTERNS = {
    "ori": [r"^ori$", r"^ori_?code$", r"^originating"],
    "year": [r"^year$"],
    "actual": [r"act.*murder", r"murder.*actual", r"actual.*mansl"],
    "cleared": [r"clr.*murder", r"murder.*clear", r"clear.*mansl"],
    "months": [r"month.*report", r"report.*month"],
}


def find_source(path=None):
    """Return A 파일을 찾는다. 없으면 **무엇을 받아야 하는지** 정확히 말하고 죽는다."""
    if path:
        p = C.ROOT / path if not str(path).startswith(("/", "C:", "c:")) else path
        cands = [p] if str(p).endswith((".parquet", ".csv", ".zip")) else sorted(
            p.glob("*.parquet")) + sorted(p.glob("*.csv"))
    else:
        cands = [q for pat in ("*return*a*", "*offense*known*", "*clearance*")
                 for q in GEO_DIR.glob(pat)]
        cands = [q for c in cands for q in ([c] if c.is_file()
                 else sorted(c.glob("*.parquet")) + sorted(c.glob("*.csv")))]
    cands = [c for c in cands if c.is_file()]
    if cands:
        return cands[0]

    raise SystemExit(
        "[에러] Return A 자료를 찾지 못했다.\n"
        f"  찾은 곳: {GEO_DIR}\n"
        "  받을 것: openICPSR **100707** - UCR Program Data: Offenses Known and\n"
        "           Clearances by Arrest (Return A), Kaplan 통합본.\n"
        "           SHR(100699)·LEOKA(102180)와 **다른 deposit**이다.\n"
        "  버전   : 1980-2014를 덮으면 충분하다. **연 단위** 파일을 쓸 것 -\n"
        "           월 단위는 시점 문제(위 docstring)를 다루는 데 필요 없고 용량만 크다.\n"
        f"  둘 곳  : {GEO_DIR} (gitignore, 재다운로드 가능. LEOKA·SHR과 같은 취급)\n"
        "  열     : ori / year / 살인 인지 / 살인 검거 / number_of_months_reported\n"
        "           정확한 이름은 버전마다 다르므로 이 스크립트가 패턴으로 찾는다.\n"
        "  경로를 직접 주려면: --src <파일 또는 디렉터리>")


def resolve_columns(cols):
    """실제 열 이름 -> 역할. 못 찾은 역할이 있으면 후보를 보여 주고 죽는다."""
    low = {c.lower(): c for c in cols}
    found, missing = {}, []
    for role, pats in COLUMN_PATTERNS.items():
        hit = next((low[c] for p in pats for c in low if re.search(p, c)), None)
        if hit is None:
            missing.append(role)
        else:
            found[role] = hit
    if missing:
        raise SystemExit(
            f"[에러] Return A에서 역할을 못 찾았다: {', '.join(missing)}\n"
            f"  이 파일의 열 {len(cols)}개 중 앞 40개:\n    "
            + ", ".join(list(cols)[:40])
            + "\n  코드북을 보고 experiments.returna_crosscheck.COLUMN_PATTERNS에 "
              "패턴을 추가할 것.")
    print("[col] " + "  ".join(f"{k}={v}" for k, v in found.items()))
    return found


def load_shr(canonical=True):
    """우리 자료를 (ORI, 연도)로 집계한다. Agency Code는 01_clean이 버리므로 원본에서."""
    raw = pd.read_csv(C.RAW_CSV, dtype={"Agency Code": str},
                      usecols=["Agency Code", "State", "City", "Year", "Crime Solved"])
    n_before = len(raw)
    raw["solved"] = raw["Crime Solved"].astype(str).str.strip().str.lower() == "yes"
    if canonical:
        # cold_blocks / county_race_residual 과 같은 카운티 정의여야 조인이 맞는다.
        raw = counties.canonicalize(raw)
    assert len(raw) == n_before, "정본화가 행을 바꿨다"

    ory = (raw.groupby(["Agency Code", "Year"], sort=False)
              .agg(shr_actual=("solved", "size"), shr_cleared=("solved", "sum"))
              .reset_index())
    county = (raw.drop_duplicates(["Agency Code", "Year"])[["Agency Code", "Year",
                                                            "State", "City"]])
    return raw, ory.merge(county, on=["Agency Code", "Year"], how="left")


def load_returna(path, cols):
    ra = (pd.read_parquet(path) if path.suffix == ".parquet"
          else pd.read_csv(path, low_memory=False))
    c = resolve_columns(ra.columns)
    out = ra[[c["ori"], c["year"], c["actual"], c["cleared"], c["months"]]].copy()
    out.columns = ["ori", "year", "ra_actual", "ra_cleared", "ra_months"]
    out["ori"] = out["ori"].astype(str).str.strip()
    for col in ("ra_actual", "ra_cleared", "ra_months"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # 같은 (ori, year)가 여러 행이면 합친다(월 단위 파일을 준 경우의 방어).
    return out.groupby(["ori", "year"], as_index=False).agg(
        ra_actual=("ra_actual", "sum"), ra_cleared=("ra_cleared", "sum"),
        ra_months=("ra_months", "max"))


def county_tables(joined):
    """(A) 인지 대조 / (B) 검거 대조. **누적으로만** 낸다(시점 규약)."""
    g = (joined.groupby(["State", "City"], sort=False)
                .agg(shr_actual=("shr_actual", "sum"),
                     shr_cleared=("shr_cleared", "sum"),
                     ra_actual=("ra_actual", "sum"),
                     ra_cleared=("ra_cleared", "sum"),
                     agency_years=("ra_actual", "size"))
                .reset_index())
    g["ratio_actual"] = g["shr_actual"] / g["ra_actual"].replace(0, np.nan)
    g["q_shr"] = g["shr_cleared"] / g["shr_actual"].replace(0, np.nan)
    g["q_ra"] = g["ra_cleared"] / g["ra_actual"].replace(0, np.nan)
    g["d"] = g["q_ra"] - g["q_shr"]        # 양수면 SHR이 검거를 적게 잡는다
    return g


def contamination_bound(county, rho_tab):
    """(C) rho의 오염 상한. **점추정이 아니라 구간**을 낸다.

    인종 중립적 오류는 카운티 내부 비교에서 소거되므로(계획서 §3-1), 최악의 경우를
    가정한다 -- 측정된 전체 오차 |d_b|가 **전부 한쪽 인종에 몰려 있다면** rho가
    얼마나 움직이는가. 양방향을 계산해 구간으로 준다.
    """
    m = rho_tab.merge(county[["State", "City", "d", "shr_actual"]],
                      on=["State", "City"], how="inner")
    m = m[np.isfinite(m["d"])].copy()
    eps = 1e-9
    out = {}
    for side, sign in (("black", 1.0), ("white", -1.0)):
        # d>0 이면 SHR이 검거를 적게 잡았다는 뜻 -> 그만큼 O(미해결)가 과대계상.
        shift_b = sign * m["d"].clip(lower=0) * m["n_Black"]
        shift_w = (1 - sign) / 2 * m["d"].clip(lower=0) * m["n_White"]
        ob = np.maximum(m["O_Black"] - shift_b, 0.0)
        ow = np.maximum(m["O_White"] - shift_w, 0.0)
        rho_adj = np.log((ob + 0.5) / np.maximum(m["E_Black"] + 0.5, eps)) - \
                  np.log((ow + 0.5) / np.maximum(m["E_White"] + 0.5, eps))
        w = 1.0 / np.maximum(m["rho_var"], eps)
        out[f"rho_if_all_error_{side}"] = float(np.sum(rho_adj * w) / np.sum(w))
    out["rho_observed"] = float(np.sum(m["rho"] / np.maximum(m["rho_var"], eps))
                                / np.sum(1.0 / np.maximum(m["rho_var"], eps)))
    out["n_counties"] = int(len(m))
    return out, m


def boot_corr(x, y, n_boot, seed):
    """카운티를 복원추출한다(관측 단위가 카운티다 - audit_resources와 같은 규약)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, (np.nan, np.nan), 0
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            draws.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = (np.nanpercentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
    return r, (float(lo), float(hi)), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="Return A 파일 또는 디렉터리(기본: dataset/geo/ 자동 탐색)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=C.RANDOM_STATE)
    ap.add_argument("--check_inputs", action="store_true",
                    help="Return A 없이 **우리 쪽 준비 상태만** 점검한다. 조인 축, "
                         "행 보존, rho 표 존재를 확인하고 끝낸다 - 자료가 도착하기 "
                         "전에 배관이 맞는지 보기 위한 것이다.")
    args = ap.parse_args()

    raw, ory = load_shr()
    print(f"[shr] 원본 {len(raw):,}행 -> (ORI, 연도) {len(ory):,}조합 "
          f"/ ORI {raw['Agency Code'].nunique():,}개 / {raw.Year.min()}-{raw.Year.max()}")
    assert int(ory["shr_actual"].sum()) == len(raw), "집계에서 행이 샜다"

    rho_path = C.scoped_output("county_race_residual.csv")
    if not rho_path.exists():
        raise SystemExit(f"[에러] {rho_path} 없음. 먼저: "
                         f"python -m experiments.county_race_residual")
    rho_tab = pd.read_csv(rho_path)
    rho_tab = rho_tab[rho_tab["min_race_n"] == rho_tab["min_race_n"].min()]
    print(f"[rho] {rho_path.name}: 카운티 {len(rho_tab):,}개 (읽기만 한다 - "
          f"이 트랙은 rho를 재계산하지 않는다)")

    if args.check_inputs:
        print("\n[준비] 우리 쪽 배관은 갖춰졌다. 남은 것은 Return A 파일뿐이다.")
        find_source(args.src)          # 없으면 여기서 안내와 함께 죽는다
        print("[준비] Return A도 있다 - --check_inputs 없이 다시 실행할 것.")
        return

    src = find_source(args.src)
    print(f"[load] Return A: {src}")
    ra = load_returna(src, None)

    # --- G1 조인 ---------------------------------------------------------
    j = ory.merge(ra, left_on=["Agency Code", "Year"], right_on=["ori", "year"],
                  how="left")
    hit = j["ra_actual"].notna()
    rate = float(hit.mean())
    print(f"[G1] (ORI, 연도) 매칭 {int(hit.sum()):,}/{len(j):,} ({rate*100:.1f}%)")
    if rate < 0.95:
        print("  [경고] 95% 미만이다. LEOKA는 같은 축에서 100.0%였으므로 "
              "구현을 의심할 것(계획서 §4-3).")

    # --- G2 커버리지 -----------------------------------------------------
    j["full_year"] = j["ra_months"].fillna(0) >= 12
    print(f"[G2] 12개월 보고 {int(j.full_year.sum()):,} / "
          f"미만·결측 {int((~j.full_year).sum()):,} - 후자는 기록 문제가 아니라 "
          f"자료 결측이라 따로 본다")

    rows, tables = [], {}
    for arm, sub in (("full_year", j[j.full_year]), ("all", j[hit])):
        cty = county_tables(sub.dropna(subset=["State", "City"]))
        tables[arm] = cty
        m = cty.merge(rho_tab[["State", "City", "black_share_both"]],
                      on=["State", "City"], how="inner")
        r_a, ci_a, n_a = boot_corr(m["ratio_actual"], m["black_share_both"],
                                   args.n_boot, args.seed)
        r_d, ci_d, n_d = boot_corr(m["d"], m["black_share_both"],
                                   args.n_boot, args.seed)
        med_ratio = float(np.nanmedian(cty["ratio_actual"]))
        med_d = float(np.nanmedian(cty["d"]))
        print(f"\n=== {arm} : 카운티 {len(cty):,}개 ===")
        print(f"  (A) 인지 비율 SHR/ReturnA 중앙값 {med_ratio:.3f}   "
              f"corr(ratio, black_share) {r_a:+.3f} [{ci_a[0]:+.3f}, {ci_a[1]:+.3f}] (n={n_a})")
        print(f"  (B) 검거율 차 d=q_RA-q_SHR 중앙값 {med_d:+.4f}   "
              f"corr(d, black_share) {r_d:+.3f} [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}] (n={n_d})")
        rows.append({"arm": arm, "n_counties": len(cty),
                     "median_ratio_actual": med_ratio, "median_d": med_d,
                     "corr_ratio_black_share": r_a, "corr_ratio_lo": ci_a[0],
                     "corr_ratio_hi": ci_a[1],
                     "corr_d_black_share": r_d, "corr_d_lo": ci_d[0],
                     "corr_d_hi": ci_d[1]})

    # --- (C) 오염 상한 ---------------------------------------------------
    bound, _ = contamination_bound(tables["full_year"], rho_tab)
    lo = min(bound["rho_if_all_error_black"], bound["rho_if_all_error_white"])
    hi = max(bound["rho_if_all_error_black"], bound["rho_if_all_error_white"])
    print(f"\n=== (C) rho 오염 상한 (카운티 {bound['n_counties']:,}개) ===")
    print(f"  관측 rho {bound['rho_observed']:+.4f}   "
          f"양방향 최악 케이스 [{lo:+.4f}, {hi:+.4f}]")
    if not (lo <= bound["rho_observed"] <= hi):
        print("  [경고] 상한이 관측값을 사이에 두지 않는다 - 계획서 §8의 부호 검사 실패")
    rows[0].update({f"bound_{k}": v for k, v in bound.items()})

    out_s = C.scoped_output("returna_summary.csv")
    pd.DataFrame(rows).to_csv(out_s, index=False, encoding="utf-8-sig")
    out_c = C.scoped_output("returna_by_county.csv")
    tables["full_year"].to_csv(out_c, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out_s}\n[save] {out_c}")

    results.write(results.rows(
        "returna", {"corr_d_black_share": rows[0]["corr_d_black_share"],
                    "median_ratio_actual": rows[0]["median_ratio_actual"],
                    "rho_bound_lo": lo, "rho_bound_hi": hi},
        model="returna", params={"source": src.name, "n_boot": args.n_boot}))
    print(f"[save] {results.results_path()} (family=returna)")


if __name__ == "__main__":
    main()
