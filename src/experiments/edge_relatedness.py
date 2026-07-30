"""
experiments/edge_relatedness.py -- 엣지가 정말 '관련 있는' 사건을 잇고 있는가

동기: 04_build_graph.py는 블록(같은 State+City / Year+Month / Weapon+Sex+Race)
안에서 **셔플-링 무작위 페어링**으로 엣지를 만든다. 근거는 "블록 안에는 더 세밀한
유사도 신호가 없다"는 판단이었는데, 그 전제는 한 번도 검정된 적이 없다.

검정할 외부 라벨이 하나 있다 -- **가해자 열**이다. Perpetrator Sex/Race/Age와
Relationship은 미해결 사건에서 90~99%가 Unknown이라 모델 입력에서는 누수로
배제되지만(config.LEAKAGE_COLS), 한 번도 입력에 들어간 적이 없으므로 **평가
라벨로는 완벽히 정당하다**. 두 사건의 가해자 프로필이 같다는 것은 "관련성"의
불완전하지만 객관적인 대리값이다.

주의: 이건 **동일범 판정이 아니다.** 데이터에 가해자 ID가 없으므로 "같은 프로필"은
"같은 범인"을 뜻하지 않는다. 재는 것은 "엣지가 서로 닮은 사건을 잇는가"이지
"엣지가 같은 범인의 사건을 잇는가"가 아니다.

## 무엇을 재는가 -- 세 층으로 분해한다

한 엣지의 프로필 일치율을 그냥 보면 아무 뜻이 없다. 블로킹 자체가 이미 사건을
비슷하게 만들기 때문이다. 그래서 셋으로 나눈다:

    profile_match       실제 엣지의 가해자 프로필 일치율
    null_global         전역 무작위 페어링의 기대 일치율 (엣지 끝점 가중)
    null_within_block   같은 블록 안에서 무작위로 이었을 때의 기대 일치율

    block_lift = null_within_block - null_global   <- 블록이 잡아낸 관련도
    edge_lift  = profile_match - null_within_block <- 엣지 페어링 자체의 기여

셔플-링이 편향 없는 무작위 페어링이라면 **edge_lift는 0이어야 한다**. 0이 나오면
그것은 실패가 아니라 확인이다 -- "관련도는 전적으로 블록이 담당하고 개별 엣지는
블록의 무작위 표본"이라는 구조가 측정으로 확정되며, 그때 엣지 중요도 분석의
단위는 개별 엣지가 아니라 블록/속성이어야 한다는 결론이 따라온다.

## oracle -- 블로킹 설계를 바꿀 근거가 있는가

    oracle_match  블록 안에서 특성 유사도 상위 k개로 이었을 때의 일치율
    oracle_lift   oracle_match - (같은 부분표본에서의 무작위 페어링 기대값)

oracle_lift가 유의하게 양수면 "블록 안에 랭킹할 신호가 없다"는 전제가 틀린 것이고,
04_build_graph.py를 유사도 랭킹으로 업그레이드할 근거가 된다.

**oracle은 반드시 두 벌로 낸다 -- sighted와 blind.** 유사도를 인종·성별을 포함한
전체 특성으로 재면(sighted), 유사도 랭킹은 곧 같은 인종끼리 잇는 것이 되어
geo 엣지의 인종 동종성(assortativity 0.174)을 더 키운다. 즉 sighted에서만
oracle_lift가 양수라면 그 업그레이드 경로는 **정확도를 공정성으로 사는 것**이라
채택하면 안 된다. blind에서도 양수여야 순수한 개선이다. 이 저장소가 완화 실험마다
대조군을 붙이는 것과 같은 규율이다.

## 한계

미해결-미해결 엣지는 검증할 수 없다(가해자 미상). 대상은 **양 끝이 모두 solved이고
프로필이 모두 알려진 엣지**뿐이므로, 여기서 나온 값은 그래프 전체가 아니라 그
부분집합의 성질이다.

출력: outputs/edge_relatedness.csv
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear import similarity
from clear.data import load_xy
from clear.graph import load_edges

# 후보별 블로킹 키. 04_build_graph.py의 candidates와 같은 출처를 봐야
# "엣지가 블록 안에 있다"는 검사가 의미를 갖는다.
BLOCK_COLS = {
    "geo": C.GEO_BLOCK_COLS,
    "temporal": C.TEMPORAL_BLOCK_COLS,
    "weapon": C.WEAPON_BLOCK_COLS,
}

# 가해자 프로필 정의. 굵기를 달리한 세 벌을 함께 내는 이유: 세밀한 정의는 일치가
# 드물어 신호가 묻히고, 거친 정의는 우연 일치가 많아 null이 높아진다. 셋의 결론이
# 같은 방향이면 정의 선택에 결과가 좌우되지 않는다는 뜻이다.
PROFILE_DEFS = {
    "perp_sex_race_age": ["Perpetrator Sex", "Perpetrator Race", "_perp_age_bin"],
    "perp_sex_race": ["Perpetrator Sex", "Perpetrator Race"],
    "relationship": ["Relationship"],
}

UNKNOWN = "Unknown"


def load_perpetrator():
    """raw CSV에서 가해자 열을 복원해 sample.parquet 행 순서에 맞춘다.

    01_clean.py가 이 열들을 지우기 전의 원본을 다시 읽는 것이므로, 정렬이
    위치 기준으로 맞는다는 보장이 필요하다. 성립 근거:
      - 01_clean.py는 행을 지우거나 섞지 않는다(열만 지운다).
      - 02_sample.py는 State 필터만 하고 순서를 보존한다(.copy(), 정렬 없음).
      - SAMPLE_MAX_ROWS가 None이라 층화 다운샘플(순서를 섞는다)이 돌지 않는다.

    이 중 하나라도 깨지면 정렬이 조용히 어긋나고, 그러면 "엣지가 관련 사건을
    잇는가"라는 측정 전체가 무의미해진다 -- 에러 없이 그럴듯한 표가 나오는
    최악의 실패 모드다. 그래서 세 가지를 전부 assert한다.
    """
    if C.SAMPLE_MAX_ROWS is not None:
        raise ValueError(
            f"config.SAMPLE_MAX_ROWS={C.SAMPLE_MAX_ROWS}. 층화 다운샘플이 켜져 있으면 "
            f"02_sample.py가 행 순서를 섞으므로 raw CSV와 위치 기준 정렬이 성립하지 않는다. "
            f"이 스크립트는 SAMPLE_MAX_ROWS=None 에서만 유효하다.")

    raw = pd.read_csv(C.RAW_CSV, low_memory=False)
    if C.SAMPLE_STATES:
        raw = raw[raw["State"].isin(C.SAMPLE_STATES)]
    raw = raw.reset_index(drop=True)

    sample = pd.read_parquet(C.SCOPE_DIR / "sample.parquet")
    if len(raw) != len(sample):
        raise ValueError(
            f"행 수 불일치: raw 필터 결과 {len(raw):,} vs sample.parquet {len(sample):,}. "
            f"01/02의 필터 조건이 바뀌었는지 확인할 것.")

    solved_raw = (raw[C.TARGET] == "Yes").astype(int).values
    if not np.array_equal(solved_raw, sample[C.TARGET_BIN].values):
        raise ValueError(
            "타깃 벡터 불일치: raw의 Crime Solved와 sample.parquet의 solved가 행 단위로 "
            "다르다. 위치 기준 정렬이 깨졌다.")
    for col in ["Year", "Month", "State", "City"]:
        if not (raw[col].values == sample[col].values).all():
            raise ValueError(f"'{col}' 열 불일치: 위치 기준 정렬이 깨졌다.")

    perp = raw[C.LEAKAGE_COLS].copy()
    # 나이는 5년 구간(config.AGE_BIN_WIDTH)으로 묶는다. 0은 미상 코드라 결측 처리 --
    # 피해자 나이의 998과 같은 역할이고, 구간화 전에 빼지 않으면 '0-4세 가해자'라는
    # 대형 가짜 그룹이 생긴다.
    age = pd.to_numeric(perp["Perpetrator Age"], errors="coerce")
    bad = age.isna() | (age <= 0) | (age >= 100)
    perp["_perp_age_bin"] = np.where(bad, UNKNOWN,
                                     (age // C.AGE_BIN_WIDTH).astype("Int64").astype(str))
    return perp


def profile_codes(perp, cols):
    """가해자 프로필 -> (정수 코드, 알려짐 마스크).

    구성 열 중 하나라도 Unknown/결측이면 그 사건은 프로필을 모르는 것으로 본다.
    부분적으로 아는 프로필을 일치 판정에 넣으면 'Unknown끼리 일치'가 신호로
    잡히는데, 그건 관련성이 아니라 기록 누락의 공유다.
    """
    sub = perp[cols].astype(str)
    known = np.ones(len(sub), dtype=bool)
    for c in cols:
        # .values 필수: Series를 그대로 &= 하면 이후 node_ok가 Series로 남고,
        # node_ok[edge_idx]가 라벨 중복(한 노드가 여러 엣지에 등장)을 가진 Series가
        # 되어 다음 & 연산이 인덱스 정렬로 교차곱을 만든다(수천만 행 폭발).
        known &= ~(sub[c].isin([UNKNOWN, "nan", "<NA>", ""]).values)
    joined = sub.agg("|".join, axis=1)
    codes = pd.Categorical(joined).codes.astype(np.int64)
    return codes, np.asarray(known)


def fold_pairs(edges):
    """대칭 저장된 (2,E) -> 무향 쌍 (2,E/2). 같은 쌍을 두 번 세지 않기 위함."""
    return edges[:, edges[0] < edges[1]]


def within_block_null(codes, block_ids, node_ok, edge_weight, n_blocks, n_codes):
    """블록 안에서 무작위로 이었을 때의 기대 일치율.

    블록 b의 적격 노드 중 프로필 g가 n_bg개면, 그 블록에서 무작위 한 쌍이 일치할
    확률은 sum_g n_bg(n_bg-1) / n_b(n_b-1) 이다. 이를 블록별로 구해 **그 블록이
    실제로 가진 적격 엣지 수**로 가중 평균한다 -- 실제 엣지 분포와 같은 가중이라야
    profile_match와 나란히 놓고 뺄 수 있다.

    (node, code) 평탄 인덱스에 대한 bincount 한 번으로 전 블록을 동시에 센다.
    """
    ok = np.flatnonzero(node_ok)
    flat = block_ids[ok] * n_codes + codes[ok]
    cnt = np.bincount(flat, minlength=n_blocks * n_codes).reshape(n_blocks, n_codes)
    n_b = cnt.sum(axis=1)
    same = (cnt * (cnt - 1)).sum(axis=1)
    tot = n_b * (n_b - 1)
    valid = (tot > 0) & (edge_weight > 0)
    if not valid.any():
        return np.nan
    null_b = same[valid] / tot[valid]
    w = edge_weight[valid]
    return float((null_b * w).sum() / w.sum())


def oracle_lift(Xn, codes, node_ok, block_ids, n_blocks, k, cap, seed):
    """블록 안에서 '특성 유사도 상위 k'로 이었다면 프로필 일치가 얼마나 올랐을까.

    각 적격 노드마다 같은 블록의 가장 닮은 k개를 이웃으로 삼는다(셔플-링이 무작위
    k개를 삼는 자리에 유사도 상위 k개를 넣은 것). 비교 대상은 전역 값이 아니라
    **같은 부분표본에서의 무작위 페어링 기대값**이다 -- 큰 블록을 cap으로 줄이면
    블록 가중이 달라지므로, 블록 안에서 짝지어 빼야 그 차이가 상쇄된다.

    반환: (oracle_match, oracle_null, lift, n_pairs, n_blocks_used)
    """
    rng = np.random.default_rng(seed)
    num_m = num_n = 0.0
    tot_pairs = 0
    used = 0
    for b in range(n_blocks):
        idx = np.flatnonzero((block_ids == b) & node_ok)
        if len(idx) < k + 2:                 # 이웃 k개를 뽑을 수 없는 블록은 건너뛴다
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        cb = codes[idx]
        # 04_build_graph.py --rank과 **같은 함수**를 쓴다. 따로 구현하면 여기서 잰
        # 이득과 실제로 만든 그래프가 조용히 갈린다(동점 처리만 달라도 갈린다).
        nb = similarity.block_topk(Xn, idx, k, seed + b)
        match = float((cb[:, None] == cb[nb]).mean())

        # 같은 부분표본의 무작위 페어링 기대값
        c = np.bincount(cb)
        n = len(cb)
        null = float((c * (c - 1)).sum() / (n * (n - 1)))

        pairs = len(idx) * k
        num_m += match * pairs
        num_n += null * pairs
        tot_pairs += pairs
        used += 1
    if tot_pairs == 0:
        return np.nan, np.nan, np.nan, 0, 0
    m, nl = num_m / tot_pairs, num_n / tot_pairs
    return m, nl, m - nl, tot_pairs, used


def measure(edge_type, k, perp, solved, sample, Z_blind, Z_sighted, args):
    edges = load_edges(edge_type, k, args.edge_mode)
    pairs = fold_pairs(edges)

    block_cols = BLOCK_COLS[edge_type]
    block_ids, block_vals = pd.factorize(
        pd.MultiIndex.from_frame(sample[block_cols]), sort=False)
    n_blocks = len(block_vals)

    rows, bake = [], []
    for pname in args.profiles:
        codes, known = profile_codes(perp, PROFILE_DEFS[pname])
        n_codes = int(codes.max()) + 1
        node_ok = known & (solved == 1)

        i, j = pairs[0], pairs[1]
        elig = node_ok[i] & node_ok[j]
        pi, pj = i[elig], j[elig]
        if len(pi) == 0:
            print(f"  [{edge_type}/{pname}] 적격 엣지 0개 - 건너뜀")
            continue

        # 엣지는 구성상 블록 안에만 있어야 한다. 어긋나면 블로킹 키가 04와
        # 달라졌다는 뜻이고, 그러면 within-block null이 다른 것을 재게 된다.
        if not (block_ids[pi] == block_ids[pj]).all():
            raise ValueError(
                f"'{edge_type}' 엣지가 블록을 넘는다. config의 {block_cols}가 "
                f"04_build_graph.py가 그래프를 만들 때와 다르다.")

        match = float((codes[pi] == codes[pj]).mean())

        endp = np.concatenate([pi, pj])
        q = np.bincount(codes[endp], minlength=n_codes) / len(endp)
        null_g = float((q ** 2).sum())

        w = np.bincount(block_ids[pi], minlength=n_blocks)
        null_b = within_block_null(codes, block_ids, node_ok, w, n_blocks, n_codes)

        row = {
            "edge_type": edge_type, "k": k,
            "edge_mode": args.edge_mode or "shuffle", "profile": pname,
            "n_pairs": int(pairs.shape[1]),
            "n_pairs_eligible": int(len(pi)),
            "eligible_frac": len(pi) / pairs.shape[1],
            "n_profiles": n_codes,
            "profile_match": match,
            "null_global": null_g,
            "null_within_block": null_b,
            "block_lift": null_b - null_g,
            "edge_lift": match - null_b,
        }

        # 유사도 대결: 같은 블록·같은 라벨 위에서 인코딩만 바꿔 oracle을 잰다.
        # 공간은 blind 고정 -- sighted 랭킹은 인종 동종성을 키우므로 채택 후보가
        # 아니고(§4), 참조용 한 벌만 onehot에서 낸다.
        for space, Zs in [("blind", Z_blind), ("sighted", Z_sighted)]:
            for scheme, Z in (Zs or {}).items():
                om, on, ol, npair, nblk = oracle_lift(
                    Z, codes, node_ok, block_ids, n_blocks,
                    args.oracle_k, args.oracle_cap, C.RANDOM_STATE)
                bake.append({
                    "edge_type": edge_type, "k": k, "profile": pname,
                    "space": space, "similarity": scheme,
                    "oracle_null": on, "oracle_match": om, "oracle_lift": ol,
                    "n_oracle_pairs": npair, "n_oracle_blocks": nblk,
                })
                if scheme == "onehot":   # 기존 표의 열을 그대로 유지
                    row[f"oracle_match_{space}"] = om
                    row[f"oracle_null_{space}"] = on
                    row[f"oracle_lift_{space}"] = ol
                    row["n_oracle_pairs"] = npair
                    row["n_oracle_blocks"] = nblk
                print(f"    [{space}/{scheme:12s}] oracle {on:.4f} -> {om:.4f}  "
                      f"lift {ol:+.4f}")

        rows.append(row)
        print(f"  [{edge_type}/{pname}] 적격 {len(pi):,}쌍  "
              f"일치 {match:.4f}  블록null {null_b:.4f}  "
              f"block_lift {null_b - null_g:+.4f}  edge_lift {match - null_b:+.4f}")
    return rows, bake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--edge_types", nargs="*", default=["geo", "temporal", "weapon"])
    ap.add_argument("--profiles", nargs="*", default=list(PROFILE_DEFS),
                    choices=list(PROFILE_DEFS))
    ap.add_argument("--oracle_k", type=int, default=None,
                    help="oracle이 고를 이웃 수(기본: --k와 동일)")
    ap.add_argument("--oracle_cap", type=int, default=2000,
                    help="oracle 계산 시 블록당 표본 상한(블록 내 전체 쌍은 O(n^2))")
    ap.add_argument("--no_oracle", action="store_true",
                    help="oracle 생략(빠른 실행). block_lift/edge_lift만 낸다.")
    ap.add_argument("--edge_mode", default=None,
                    help="검정할 그래프의 구성 방식(기본 None=셔플-링). "
                         "'rank_onehot' 등을 주면 04_build_graph.py --rank이 만든 "
                         "랭킹 그래프를 검정한다 -- edge_lift가 0이 아니라 oracle "
                         "수준으로 올라와야 구현이 의도대로 된 것이다.")
    ap.add_argument("--similarity", nargs="*", default=["onehot"],
                    choices=similarity.SCHEMES,
                    help="oracle이 쓸 유사도 인코딩. 여러 개를 주면 대결 결과를 "
                         "outputs/edge_relatedness_similarity.csv에 남긴다.")
    args = ap.parse_args()
    if args.oracle_k is None:
        args.oracle_k = args.k

    perp = load_perpetrator()
    sample = pd.read_parquet(C.SCOPE_DIR / "sample.parquet")
    solved = sample[C.TARGET_BIN].values
    print(f"[load] {len(sample):,}행 (정렬 검증 통과), solved {int(solved.sum()):,}건, "
          f"k={args.k}")

    for pname in args.profiles:
        _, known = profile_codes(perp, PROFILE_DEFS[pname])
        cov = (known & (solved == 1)).sum() / max(int(solved.sum()), 1)
        print(f"  [profile:{pname}] solved 중 프로필 확인 가능 {cov:.1%}")

    Z_blind = Z_sighted = None
    if not args.no_oracle:
        Xb, _ = load_xy(blind=True)
        Xs, _ = load_xy(blind=False)
        Z_blind = {s: similarity.build(Xb, s) for s in args.similarity}
        # sighted는 참조용 한 벌만(onehot). 랭킹 후보가 아니라 '거절한 선택지'다.
        Z_sighted = {"onehot": similarity.build(Xs, "onehot")}
        shapes = ", ".join(f"{s}->{Z_blind[s].shape[1]}열" for s in args.similarity)
        print(f"[oracle] blind X {Xb.shape[1]}열 [{shapes}], sighted {Xs.shape[1]}열, "
              f"이웃 {args.oracle_k}개, 블록 상한 {args.oracle_cap:,}")

    rows, bake = [], []
    for et in args.edge_types:
        print(f"\n=== {et} ===")
        r, b = measure(et, args.k, perp, solved, sample, Z_blind, Z_sighted, args)
        rows += r
        bake += b

    df = pd.DataFrame(rows)
    out = C.scoped_output("edge_relatedness.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    if bake:
        bdf = pd.DataFrame(bake)
        bout = C.scoped_output("edge_relatedness_similarity.csv")
        bdf.to_csv(bout, index=False, encoding="utf-8-sig")
        if len(args.similarity) > 1:
            print("\n=== 유사도 인코딩 대결 (blind, oracle_lift) ===")
            piv = (bdf[bdf.space == "blind"]
                   .pivot_table(index=["profile", "edge_type"], columns="similarity",
                                values="oracle_lift", aggfunc="first")
                   .reindex(columns=[s for s in similarity.SCHEMES
                                     if s in args.similarity]))
            print(piv.round(4).to_string())
        print(f"[save] {bout}")

    for pname, sub in df.groupby("profile"):
        print(f"\n=== profile = {pname} ===")
        cols = ["edge_type", "profile_match", "null_within_block", "null_global",
                "block_lift", "edge_lift"]
        if not args.no_oracle:
            cols += ["oracle_lift_blind", "oracle_lift_sighted"]
        print(sub[cols].round(4).to_string(index=False))

    print(f"\n[save] {out}")
    print("[해석] block_lift > 0 이면 블로킹 키가 관련도를 잡아낸 것이다. "
          "edge_lift ~ 0 이면 블록 안 페어링은 무작위 표본이라는 뜻이고, "
          "그때 엣지 중요도의 단위는 개별 엣지가 아니라 블록/속성이어야 한다. "
          "oracle_lift가 blind에서도 양수라야 유사도 랭킹 업그레이드가 "
          "공정성 대가 없는 순수 개선이 된다.")


if __name__ == "__main__":
    main()
