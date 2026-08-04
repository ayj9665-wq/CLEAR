"""
experiments/detect_cold_blocks.py -- 이례적으로 안 풀린 사건 묶음 탐지

미해결 사건 클러스터링(확장설계서 트랙 D)의 세 산출물 중 **D2**. 나머지 둘과
방법이 다르므로 섞지 않는다:

    D1 유형화   임베딩 클러스터링        사건 -> 군집
    D2 이상탐지 블록 단위 관측-기대 잔차  블록          <- 이 스크립트
    D3 우선순위 높은 p_hat & 미해결       개별 사건

## 이것이 '연쇄 의심'의 정당한 형태다

데이터에 가해자 ID가 없으므로 "이 N건은 동일범"은 **원리적으로 검증 불가**다.
MAP의 Hargrove 알고리즘이 실제로 하는 것도 동일범 확정이 아니라 **군집 이상 탐지**
-- "같은 지역·수법·피해자 유형의 미해결 사건이 기대보다 이례적으로 몰려 있는가"다.
그건 이 데이터로 할 수 있다.

## 방법: 간접 표준화 (indirect standardization)

블록 b에 대해

    O_b = sum(1 - y_i)              관측 미해결 수
    E_b = sum(1 - p_hat_i)          기대 미해결 수
    SMR = O_b / E_b                 1보다 크면 기대보다 많이 미해결

역학의 SMR(standardized mortality ratio)과 같은 구성이다. Hargrove보다 나은 점 둘:

  1. **사건 구성(case mix)이 통제된다.** "그 카운티에 총기 살인이 많아서 검거율이
     낮다"는 이미 p_hat에 반영돼 E_b로 빠지고, 남는 잔차가 *설명되지 않는* 집중이다.
  2. p_hat이 **blind + 공정성 완화 모델**(mitigate_loss alpha=50)에서 온다.
     이 선택의 뜻을 정확히 해둬야 한다 -- 인종을 기대값에서 **빼주는 것이
     아니라 반대다**. 모델이 인종을 안 쓰므로 인종 관련 격차는 E_b에 흡수되지
     않고 **잔차에 그대로 남는다. 그리고 그것이 목적이다.**

     sighted 모델을 쓰면 "피해자가 흑인이라 안 풀릴 것으로 기대"가 baseline에
     들어가, 인종 격차가 이례성 판정에서 **정상**으로 세탁된다 -- 이 프로젝트가
     재려는 현상 자체가 보이지 않게 된다. 완화 전 blind 모델도 FPR 기준 인종
     증폭이 1.82배라 기대값이 이미 인종에 기울어 있다. alpha=50 모델(FPR 증폭
     0.27)만이 **인종 중립적 기대값**을 준다.

     따라서 올바른 주장은 "인종으로 설명되지 않는 집중"이 아니라
     **"사건 구성으로 설명되지 않는 집중을, 인종 중립적 기준선에 대해 측정한 것"**
     이다. 그 잔차가 인종 구성과 상관되면 그건 잡음이 아니라 **결과**다.

## 전역 보정이 **필수**다 (안 하면 전부 틀린다)

clear.gnn의 손실은 BCEWithLogitsLoss(pos_weight=neg/pos)로 클래스를 재가중한다.
즉 p_hat은 순위는 옳지만 **보정된 확률이 아니다**. 실측하면 test 집합에서
기대 미해결이 관측보다 48% 높다(세 모델 모두 +48~54%). 그대로 쓰면 거의 모든
블록이 "기대보다 적게 미해결"로 나와, 재는 것이 지역 이상이 아니라 전역
miscalibration이 된다.

그래서 로짓에 상수 delta를 더해 **총합을 맞춘 뒤** 잔차를 본다
(sum sigma(logit(p_hat)+delta) = sum y). 파라미터가 하나뿐이고 단조변환이라
**순위가 전혀 안 바뀐다** -- D3 우선순위 목록은 영향을 받지 않는다. 전역 잔차가
0이 되므로 이 검정은 **블록 간 상대 편차**만 본다: 간접 표준화의 정의 그대로다.

## p-value: 모수 부트스트랩

O_b는 서로 다른 p를 가진 베르누이의 합(포아송-이항)이다. 정규근사는 블록이 작거나
p가 극단일 때 나쁘므로, p_hat을 참으로 두고 직접 시뮬레이션한다. 블록 수가 수백
개이므로 **BH FDR 보정 필수** -- 안 하면 5% 유의수준에서 우연히 20개 중 1개가
"이례적"으로 나온다.

## 한계

p_hat이 out-of-sample인 것은 **test 집합뿐**이다(학습 행의 p_hat은 낙관 편향).
따라서 기본 실행은 test 집합 위에서만 돈다 -- 블록당 표본이 그만큼 작다.

**전수로 넓히는 경로는 있다**: experiments/crossfit_predictions.py가 5-fold
cross-fitting으로 모든 행에 out-of-fold p_hat을 만들고, 그 덤프를 --model로 주면
이 스크립트가 그대로 읽는다(--out으로 표를 분리할 것 -- cold_blocks.csv는 test
분할 표이고 지도·보고서가 그것을 인용한다). 전국에서 카운티 853 -> 1,803개가 됐고,
그때 z ~ black_share가 +0.381 -> +0.290으로 내려갔다 -- 상관의 강도가 카운티 규모에
의존한다는 뜻이다(확장설계서 §10-4-1). 3개 주에서는 하지 않았다: test 기준 143개
블록이 이미 카운티의 97.1%를 덮어 얻을 것이 없다.

**결정적 한계: 모델은 카운티를 모른다.** City는 특성이 아니라 블로킹 키일 뿐이라
(config.CATEGORICAL_COLS에 없다) 그래프를 통해서만 간접적으로 들어간다. 그래서
E_b에는 카운티 고유 효과가 거의 없고, z는 사실상 **"이 카운티가 주(州)·사건구성
기준선에서 얼마나 벗어나는가"**를 잰다. 그 이탈이 **사건구성** 때문인지, 차별
때문인지, 수사 자원 때문인지, 기록 관행 때문인지는 **이 설계로 분리되지 않는다.**
(도시성은 이 넷과 나란한 채널이 아니라 **공통원인**이다 — 목격자 협조·사건당 부하·
기관 규모·낯선사람 살인 비중을 담는 그릇이고, 동시에 거주지 분리를 통해 차별 경로의
매개자이기도 하다. 회귀변수로 넣으면 새로운 방식으로 해석 불가능해지므로 층화
변수로만 쓴다. Agency Type이 그것을 일부 대리한다.) z와 인종 구성의 상관을 인과로
읽으면 안 된다.

**그리고 그 상관이 어느 층에 있는지는 이제 측정됐다**(experiments/county_race_residual.py,
카운티차분 계획서 §12-4). 카운티를 고정하고 인종별로 같은 잔차를 내면
log(SMR_Black/SMR_White) = **+0.083 [+0.056, +0.128]** (역분산 가중, 카운티 651개)로
0이 아니지만, 그 값과 카운티 흑인비중의 상관은 **0**이다(−0.015 [−0.108, +0.105]).
즉 **z ~ black_share = +0.290은 전부 카운티 수준 절편 효과**다 — 흑인 비중이 높은
카운티는 잔차가 나쁘지만 그 카운티 **안에서** 흑인 피해자가 특별히 더 나쁘지는 않고,
카운티 내부 격차는 어디에나 균일하게 존재한다. 위 네 채널이 분리되지 않는다는 말은
**절편 성분에 대해서만** 유효하다.

출력: outputs/cold_blocks.csv
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear import counties, predictions, results

# 블록 키. county는 지도(카운티 코로플레스)와 직결되고, hargrove는 MAP 알고리즘의
# 원래 키(지리 + 수법 + 피해자 성별)에 가장 가깝다.
BLOCK_KEYS = {
    "county": ["State", "City"],
    "county_weapon": ["State", "City", "Weapon"],
    "hargrove": ["State", "City", "Weapon", "Victim Sex"],
}

_EPS = 1e-9

# 전역 보정은 clear.predictions로 옮겼다 -- 보정 대상이 그 모듈이 정의한 proba 열이고
# 호출부가 둘이 됐다(county_race_residual). 이름을 남겨 두는 이유는 이 파일의 설명
# (§"전역 보정이 필수다")이 계속 calibrate를 가리키기 때문이다.
calibrate = predictions.calibrate


def bh_fdr(p):
    """Benjamini-Hochberg q-value. 블록이 수백 개라 다중검정 보정 없이는 무의미하다."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def block_pvalue(q_unsolved, observed, n_boot, rng, chunk=500):
    """O_b의 양측 p-value를 모수 부트스트랩으로.

    q_unsolved: 이 블록 각 사건의 미해결 확률(=1-p_hat). 이를 참으로 두고
    sum Bernoulli(q)를 n_boot회 시뮬레이션한다(포아송-이항의 정확한 표본).
    큰 블록(LA는 test에만 1만 행대)에서 (n_boot, n) 행렬이 커지므로 chunk로 나눈다.

    양측: 2 * min(P(S>=O), P(S<=O)), 1로 절단. (count+1)/(n_boot+1) 보정으로
    p=0이 나오지 않게 한다 -- 부트스트랩 해상도를 넘는 확신을 표시하지 않기 위함.
    """
    ge = le = 0
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        sim = (rng.random((b, len(q_unsolved))) < q_unsolved).sum(axis=1)
        ge += int((sim >= observed).sum())
        le += int((sim <= observed).sum())
    p_ge = (ge + 1) / (n_boot + 1)
    p_le = (le + 1) / (n_boot + 1)
    return float(min(1.0, 2 * min(p_ge, p_le)))


def analyse(df, key_cols, min_n, n_boot, priority_q, seed):
    """블록별 관측-기대 잔차 표. df는 예측 + 블록 키 + 민감속성이 붙은 프레임."""
    rng = np.random.default_rng(seed)
    unsolved_p = 1.0 - df["p_cal"].values
    is_unsolved = (df["y_true"].values == 0)
    # '모델은 풀릴 만하다고 본 미해결 사건' = D3 우선순위 후보.
    # 임계값은 미해결 사건들의 p_hat 분포 상위 분위로 잡는다.
    thr = np.quantile(df.loc[is_unsolved, "p_cal"].values, 1 - priority_q)
    is_priority = is_unsolved & (df["p_cal"].values >= thr)

    race = df["sens__Victim Race"].values
    rows = []
    for key, idx in df.groupby(key_cols, sort=False).indices.items():
        if len(idx) < min_n:
            continue
        q = unsolved_p[idx]
        O = int(is_unsolved[idx].sum())
        E = float(q.sum())
        var = float((q * (1 - q)).sum())
        z = (O - E) / np.sqrt(max(var, _EPS))
        p = block_pvalue(q, O, n_boot, rng)
        r = race[idx]
        rows.append({
            **dict(zip(key_cols, key if isinstance(key, tuple) else (key,))),
            "n": len(idx),
            "n_unsolved": O,
            "expected_unsolved": E,
            "smr": O / max(E, _EPS),
            "z": z,
            "p_value": p,
            "n_priority": int(is_priority[idx].sum()),
            # 공정성 게이트: 이례 블록이 특정 인종에 몰리면 그 자체가 결과다.
            "white_share": float((r == "White").mean()),
            "black_share": float((r == "Black").mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = bh_fdr(out["p_value"].values)
    out["flag"] = np.where(out["q_value"] >= 0.05, "",
                           np.where(out["z"] > 0, "cold", "warm"))
    return out.sort_values("z", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="graphsage_fairloss_a50",
                    help="outputs/predictions/의 덤프 라벨. 기본은 공정성 완화를 "
                         "거친 모델 -- 완화 전 모델의 p_hat은 인종 격차를 1.8배 "
                         "증폭하므로 이례성 판정이 인종에 기운다.")
    ap.add_argument("--blocks", nargs="*", default=["county"],
                    choices=list(BLOCK_KEYS))
    ap.add_argument("--min_n", type=int, nargs="+", default=[20],
                    help="블록 최소 표본(여러 개 가능: --min_n 20 50 100). 작을수록 "
                         "검정력이 없다(test 기준 n>=20이면 카운티 143개 = test의 97.1%%). "
                         "여러 개를 주면 각각에 대해 **검정을 다시 돌린다** — BH FDR의 "
                         "q값이 검정 개수에 의존하므로, 한 번 돌리고 n으로 거르는 것은 "
                         "다른 결과다. 웹 지도의 최소표본 슬라이더가 이 행들을 쓴다.")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--calibrate", choices=["shift", "none"], default="shift")
    ap.add_argument("--out", default="cold_blocks.csv",
                    help="블록 표 파일명(스코프 디렉터리 안). 크로스피팅 덤프처럼 "
                         "**다른 분할**에서 나온 결과는 여기를 바꿔 파일을 분리한다 "
                         "-- 기본 파일은 get_split의 test 기준이고 지도·보고서 60곳이 "
                         "그 표를 인용한다. 두 분할을 한 파일에 섞으면 어느 행이 어느 "
                         "분할인지 알 수 없게 된다.")
    ap.add_argument("--priority_q", type=float, default=0.10,
                    help="미해결 사건 중 p_hat 상위 이 비율을 '재수사 우선순위'로 센다")
    args = ap.parse_args()

    path = predictions.predictions_dir() / f"{args.model}.csv"
    if not path.exists():
        raise SystemExit(f"[에러] 덤프 없음: {path}\n"
                         f"  먼저 실행: python -m experiments.mitigate_loss --alphas 50")
    dump = predictions.load(path)
    sample = pd.read_parquet(C.SCOPE_DIR / "sample.parquet")

    # row_index는 features/sample.parquet의 행 위치라 그대로 join된다.
    keys = sorted({c for b in args.blocks for c in BLOCK_KEYS[b]})
    df = dump.join(sample[keys], on="row_index")
    if df[keys].isna().any().any():
        raise ValueError("블록 키에 결측: row_index 정렬이 깨졌다.")

    # 같은 카운티의 옛 이름/새 이름을 합친다. 이걸 안 하면 한 카운티가 두 블록으로
    # 나뉘어 **두 번 검정되고**, BH FDR도 중복 단위가 든 집합에서 계산된다.
    # 별칭은 오랫동안 지도(그리는 단계)에만 적용돼 있었다 -- clear.counties 참조.
    if {"State", "City"} <= set(keys):
        df = counties.canonicalize(df)

    p_cal, cal = calibrate(dump["proba"].values, dump["y_true"].values, args.calibrate)
    df["p_cal"] = p_cal
    print(f"[load] {args.model}: test {len(df):,}행, 미해결 {int((df.y_true == 0).sum()):,}건")
    print(f"[calib] delta={cal['delta']:+.4f}  기대 검거 {cal['sum_p_before']:,.0f} -> "
          f"{cal['sum_p_after']:,.0f} (실제 {cal['sum_y']:,.0f})")
    if args.calibrate == "none":
        print("  [경고] 보정 없이는 전역 miscalibration이 모든 블록의 잔차를 "
              "같은 방향으로 민다.")

    all_rows = []
    # min_n마다 **검정을 다시** 돌린다. 한 번 돌리고 n으로 거르는 것과 다르다 --
    # BH FDR의 q값이 동시에 검정한 블록 수에 의존하기 때문이다.
    for min_n, b in [(m, b) for m in args.min_n for b in args.blocks]:
        tab = analyse(df, BLOCK_KEYS[b], min_n, args.n_boot,
                      args.priority_q, C.RANDOM_STATE)
        if tab.empty:
            print(f"[{b}] n>={min_n} 블록 없음")
            continue
        tab.insert(0, "block_key", b)
        tab.insert(1, "min_n", min_n)
        all_rows.append(tab)

        n_cold = int((tab["flag"] == "cold").sum())
        n_warm = int((tab["flag"] == "warm").sum())
        # 기대값이 인종 중립적이므로 잔차와 인종 구성의 상관은 흡수되지 않고 남는다.
        # 이 프로젝트의 윤리 게이트상 **항상** 함께 보고한다 -- 이 표가 재수사
        # 우선순위로 쓰이면 그 자체로 자원 배분이기 때문이다.
        corr = float(np.corrcoef(tab["z"], tab["black_share"])[0, 1]) \
            if len(tab) > 2 else np.nan
        print(f"\n=== {b} ({BLOCK_KEYS[b]}, n>={min_n}) ===")
        print(f"블록 {len(tab)}개, FDR 5% 유의: cold {n_cold}개 / warm {n_warm}개")
        print(f"[공정성] z ~ black_share 상관 {corr:+.3f}   "
              f"cold 평균 흑인비중 {tab.loc[tab.flag == 'cold', 'black_share'].mean():.3f} / "
              f"warm {tab.loc[tab.flag == 'warm', 'black_share'].mean():.3f} / "
              f"전체 {tab['black_share'].mean():.3f}")
        show = ["n", "n_unsolved", "expected_unsolved", "smr", "z", "q_value",
                "n_priority", "black_share"]
        head = tab[tab["flag"] == "cold"].head(10)
        if len(head):
            print("[가장 이례적으로 미해결이 몰린 블록]")
            print(head[BLOCK_KEYS[b] + show].round(3).to_string(index=False))
        # params의 min_n은 **스칼라**여야 한다. args.min_n(리스트)을 그대로 넣으면
        # 기존 행과 params JSON이 갈려 재실행이 교체가 아니라 중복이 된다.
        results.write(results.rows(
            "cold_blocks",
            {"n_significant_blocks": n_cold, "n_eval": len(tab)},
            model=args.model, tag=b, group_set=f"n>={min_n}",
            params={"block_key": b, "min_n": min_n, "n_boot": args.n_boot,
                    "calibrate": args.calibrate},
            notes={"delta": round(cal["delta"], 4), "n_warm": n_warm,
                   "z_black_share_corr": round(corr, 3)}))

    if all_rows:
        out = C.scoped_output(args.out)
        pd.concat(all_rows, ignore_index=True).to_csv(
            out, index=False, encoding="utf-8-sig")
        print(f"\n[save] {out}")
        print("[해석] smr>1(z>0) = 사건 구성으로 설명되지 않는 미해결 집중을 "
              "**인종 중립적 기준선**에 대해 잰 값이다(인종을 빼준 것이 아니다). "
              "동일범 판정이 아니며, 가해자 ID가 없어 그 주장은 불가능하다. "
              "모델이 카운티를 특성으로 갖지 않으므로 z는 카운티 고유 효과 전체를 "
              "담는다 - 차별/자원/도시성/기록관행이 분리되지 않는다. "
              "q_value는 BH FDR 보정 후 값이고 n>=min_n 블록만 포함한다.")


if __name__ == "__main__":
    main()
