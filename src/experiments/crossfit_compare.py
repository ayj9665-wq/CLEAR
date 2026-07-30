"""
experiments/crossfit_compare.py -- 크로스피팅 전후의 z ~ black_share 상관 비교

크로스피팅 계획서 §2-1의 사전 등록된 판정을 실행한다.

    사전 예측: 크로스피팅 후 상관은 현재 +0.381보다 **내려간다**(n>=20).
    근거: 지금 상관이 블록 크기에 의존한다(n>=20 +0.381 / 50 +0.496 / 100 +0.546).
          큰 카운티만 남길수록 강해지므로, 작은 카운티가 들어오면 약해질 것이다.

판정표(§2-1):

    유의하게 하락  -> 상관은 **대도시 카운티의 성질**. 지도 문구를 한정한다
    변화 없음      -> 상관은 **전국적 성질**. 현 주장이 강화된다
    유의하게 상승  -> 예측이 틀렸다. 그 자체로 보고할 결과

## 왜 '매칭 통제'가 필요한가

두 표는 커버리지만 다른 것이 아니다. 세 가지가 한꺼번에 움직인다:

    (1) 블록 집합   853개 -> 1,803개          <- 재려는 것
    (2) 분할        get_split의 test -> 5-fold  <- 부수 변화
    (3) 시드        3시드 평균 -> 1시드         <- 부수 변화

(2)(3)이 상관을 흔들면 (1)의 효과로 오독된다 -- mitigate_graph가 남긴 교훈("개입
하나만 움직여라")이 그대로 적용된다. 그래서 **cv5를 test 표와 같은 카운티 집합으로
제한한** 세 번째 수치를 같이 낸다:

    r_matched - r_test          (2)(3)의 크기. 같은 카운티들이므로 페어드
    r_cv5     - r_matched       (1)의 크기. 커버리지 효과

r_matched가 +0.381 근처면 차이는 커버리지 때문이라고 말할 수 있다. 멀면 그렇게
말할 수 없고, 그때는 시드 3개로 다시 도는 것이 다음 수순이다.

부트스트랩은 **블록을 재표집**한다(카운티가 분석 단위이므로). 페어드 비교는 같은
재표집 draw를 양쪽에 적용한다 -- 독립 CI의 겹침으로 판단하는 것은 같은 표본 위에서
무효라는 clear.fairness의 규율과 같은 이유다.

출력: outputs[/{scope}]/crossfit_compare.csv
"""
import argparse

import numpy as np
import pandas as pd

import config as C


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan


def boot_corr(z, s, n_boot, rng):
    """블록 재표집 상관 분포. 반환 (B,) 배열."""
    n = len(z)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = np.empty(n_boot)
    for i in range(n_boot):
        j = idx[i]
        out[i] = corr(z[j], s[j])
    return out


def ci(v, lo=2.5, hi=97.5):
    return float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))


def load(path, block_key, min_n):
    tab = pd.read_csv(path)
    tab = tab[(tab["block_key"] == block_key) & (tab["min_n"] == min_n)]
    return tab.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_table", default="cold_blocks.csv")
    ap.add_argument("--cv_table", default="cold_blocks_cv5.csv")
    ap.add_argument("--blocks", default="county")
    ap.add_argument("--min_n", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    rng = np.random.default_rng(C.RANDOM_STATE)
    keys = ["State", "City"]
    rows = []

    for min_n in args.min_n:
        t = load(C.scoped_output(args.test_table), args.blocks, min_n)
        c = load(C.scoped_output(args.cv_table), args.blocks, min_n)
        if t.empty or c.empty:
            print(f"[skip] n>={min_n}: 표가 비었다 (test {len(t)} / cv {len(c)})")
            continue

        # 매칭 집합: test 표에 있는 카운티만 남긴 cv5
        t_key = t.set_index(keys)
        m = c.set_index(keys)
        common = m.index.intersection(t_key.index)
        m, t_m = m.loc[common], t_key.loc[common]

        r_test = corr(t["z"].values, t["black_share"].values)
        r_cv = corr(c["z"].values, c["black_share"].values)
        r_mat = corr(m["z"].values, m["black_share"].values)

        b_cv = boot_corr(c["z"].values, c["black_share"].values, args.n_boot, rng)
        # 페어드: 같은 카운티 재표집을 두 표에 동시에 적용한다.
        n = len(common)
        idx = rng.integers(0, n, size=(args.n_boot, n))
        zc, sc = m["z"].values, m["black_share"].values
        zt, st = t_m["z"].values, t_m["black_share"].values
        d_paired = np.array([corr(zc[j], sc[j]) - corr(zt[j], st[j]) for j in idx])

        cv_lo, cv_hi = ci(b_cv)
        d_lo, d_hi = ci(d_paired)
        # 사전 등록 판정: cv5의 CI가 기존 점추정을 품는가
        verdict = ("변화 없음(전국적 성질)" if cv_lo <= r_test <= cv_hi
                   else "유의하게 하락(대도시 성질)" if r_cv < r_test
                   else "유의하게 상승(예측 반증)")

        print(f"\n=== {args.blocks}, n>={min_n} ===")
        print(f"  test  {len(t):>5}블록  r = {r_test:+.3f}")
        print(f"  cv{5}   {len(c):>5}블록  r = {r_cv:+.3f}  [{cv_lo:+.3f}, {cv_hi:+.3f}]")
        print(f"  매칭  {len(m):>5}블록  r = {r_mat:+.3f}   "
              f"(분할·시드 효과 = {r_mat - r_test:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}])")
        print(f"  커버리지 효과 = {r_cv - r_mat:+.3f}")
        print(f"  [판정] {verdict}")

        for lbl, n_b, r, lo, hi in [
                ("test", len(t), r_test, np.nan, np.nan),
                ("cv5", len(c), r_cv, cv_lo, cv_hi),
                ("cv5_matched", len(m), r_mat, np.nan, np.nan)]:
            rows.append({"block_key": args.blocks, "min_n": min_n, "table": lbl,
                         "n_blocks": n_b, "corr_z_black_share": r,
                         "lo": lo, "hi": hi,
                         "n_cold": int((c if lbl.startswith("cv5") else t)["flag"]
                                       .eq("cold").sum()),
                         "split_seed_effect": r_mat - r_test if lbl == "cv5" else np.nan,
                         "split_seed_lo": d_lo if lbl == "cv5" else np.nan,
                         "split_seed_hi": d_hi if lbl == "cv5" else np.nan,
                         "coverage_effect": r_cv - r_mat if lbl == "cv5" else np.nan,
                         "verdict": verdict if lbl == "cv5" else ""})

        # §2-2의 검수 기준: 상위 10개 카운티가 유지되는가(구현 오류 탐지)
        top_t = list(t.nlargest(10, "z").set_index(keys).index)
        top_c = list(c.nlargest(10, "z").set_index(keys).index)
        kept = len(set(top_t) & set(top_c))
        print(f"  [검수] 상위 10개 카운티 유지 {kept}/10"
              + ("" if kept >= 8 else "  [경고] 8 미만 -- 구현을 의심할 것(§2-3)"))

    if rows:
        out = C.scoped_output("crossfit_compare.csv")
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
