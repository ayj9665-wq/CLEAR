"""
experiments/edge_homophily.py — 엣지가 민감속성의 프록시인지 측정 (번호 없는 유틸, eda.py 티어)

동기: experiments/diagnose_fairness.py가 GraphSAGE(geo)의 공정성 격차 증폭이 XGBoost보다 크다는
결과를 냈다(성별 +0.11배, 짝지은 부트스트랩 기준 유의). 그렇다면 **왜** 그런가.
가설: 노드 특성이 아니라 **엣지 자체가 민감속성과 상관**돼 있어서, 메시지 전달이
이웃의 인종·성별 정보를 노드 표현으로 끌어온다.

이 가설은 검증 가능한 예측을 낳는다 — 엣지 후보별 동종성(homophily) 순서가
증폭비 순서와 맞아야 한다. 특히 weapon 후보는 config.WEAPON_BLOCK_COLS가
Victim Race/Sex를 블로킹 키로 쓰므로 **구성상 동종성 1.0**이어야 한다(이 값이
1.0로 안 나오면 그래프 구성 쪽 버그다 — 일종의 sanity check).

측정 지표(엣지 후보 x 민감속성):
  homophily        같은 그룹끼리 이어진 엣지 비율
  homophily_null   그룹 크기만 유지하고 무작위로 이었을 때의 기대값 = sum_g p_g^2
  assortativity    (homophily - null) / (1 - null). 0=무작위, 1=완전 분리.
                   동종성 자체는 그룹이 편중되면 그냥 높게 나오므로, 순서 비교는
                   이 표준화 값으로 해야 한다.
  neighbor_recovery 이웃들의 다수결로 자기 그룹을 맞히는 정확도 —
                   "메시지 전달이 민감속성을 복원할 수 있는가"의 직접 측정
  majority_baseline 최대 그룹 비율(위 정확도를 이것과 비교해야 의미가 있다)

출력: outputs/edge_homophily.csv

주의: 이건 진단이지 처방이 아니다. 여기서 동종성이 확인되면, 민감속성 열을
빼는 것만으로는 부족하다는 뜻이 된다(그래프가 우회 경로를 준다).
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear.data import load_sensitive
from clear.graph import load_edges

ATTRS = [f"sens__{a}" for a in C.SENSITIVE_COLS]


def _codes(series):
    """라벨 → 0..G-1 정수 코드. bincount 기반 집계를 쓰기 위함."""
    cat = pd.Categorical(series)
    return cat.codes.astype(np.int64), list(cat.categories)


def neighbor_recovery(edges, codes, n_groups):
    """이웃 다수결로 자기 그룹을 맞히는 정확도 + 이웃이 있는 노드 비율.

    엣지 배열은 대칭화돼 있으므로 (src=행0)만 보면 각 노드의 이웃이 모두 모인다.
    노드별 이웃 그룹 카운트는 (node, group) 평탄 인덱스에 대한 bincount로 한 번에 센다.
    """
    n = len(codes)
    src, dst = edges[0], edges[1]
    flat = src * n_groups + codes[dst]
    counts = np.bincount(flat, minlength=n * n_groups).reshape(n, n_groups)
    has_nbr = counts.sum(axis=1) > 0
    pred = counts.argmax(axis=1)
    acc = float((pred[has_nbr] == codes[has_nbr]).mean()) if has_nbr.any() else np.nan
    return acc, float(has_nbr.mean())


def measure(edge_type, k, sens):
    edges = load_edges(edge_type, k)
    rows = []
    for attr in ATTRS:
        codes, cats = _codes(sens[attr])
        g = len(cats)
        same = codes[edges[0]] == codes[edges[1]]
        h = float(same.mean())
        p = np.bincount(codes, minlength=g) / len(codes)
        h0 = float((p ** 2).sum())
        acc, cov = neighbor_recovery(edges, codes, g)
        rows.append({
            "edge_type": edge_type, "k": k,
            "attribute": attr.replace("sens__", ""),
            "n_edges_directed": int(edges.shape[1]),
            "homophily": h,
            "homophily_null": h0,
            # null이 1에 붙으면(그룹 하나가 거의 전부) 표준화가 불안정 → nan
            "assortativity": (h - h0) / (1 - h0) if h0 < 1 - 1e-12 else np.nan,
            "neighbor_recovery": acc,
            "majority_baseline": float(p.max()),
            "recovery_lift": acc - float(p.max()),
            "node_coverage": cov,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--edge_types", nargs="*", default=["geo", "temporal", "weapon"])
    args = ap.parse_args()

    sens = load_sensitive()
    print(f"[load] 노드 {len(sens):,}개, 엣지 후보 {args.edge_types}, k={args.k}")

    rows = []
    for et in args.edge_types:
        rows += measure(et, args.k, sens)
        print(f"  [{et}] 측정 완료")

    df = pd.DataFrame(rows)
    out = C.OUTPUT_DIR / "edge_homophily.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    for attr, sub in df.groupby("attribute"):
        print(f"\n=== {attr} ===")
        show = sub[["edge_type", "homophily", "homophily_null", "assortativity",
                    "neighbor_recovery", "majority_baseline", "recovery_lift"]]
        print(show.round(4).to_string(index=False))

    print(f"\n[save] {out}")
    print("[해석] assortativity가 0보다 크게 높으면 엣지가 민감속성과 상관된 것 "
          "= 메시지 전달이 민감속성 정보를 실어 나른다. recovery_lift가 양수면 "
          "이웃만 보고도 자기 그룹을 다수결보다 잘 맞힌다는 뜻 "
          "(민감속성 열을 빼도 그래프가 우회 경로를 준다).")


if __name__ == "__main__":
    main()
