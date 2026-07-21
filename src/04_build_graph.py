"""
04_build_graph.py — 엣지 후보 3종 구성 + 비교표

개발계획서가 제시한 3가지 엣지 후보(지리·시간·수법 유사)를 각각 독립적으로
구성해 평균 차수·연결 성분 등을 비교한다. 최종 그래프 선택과 GNN 학습
(06_train_gnn.py)은 이 비교표를 보고 다음 단계에서 결정한다.

핵심 설계: "특성 유사도 top-k" 대신 "정확 일치 블로킹 + 블록 내 k개 결정적
선택"을 사용한다. City가 이미 가장 세밀한 지리 단위이고(Agency Code/Name은
01_clean.py에서 제거됨), 같은 블록 안에서는 더 세밀한 유사도 기준이 없기
때문이다. 블록이 클 때(예: LA 4만+행) 전체 쌍을 계산하지 않도록 셔플-링
방식으로 O(n·k)에 처리한다. 블록별 시드는 Python hash()가 아니라 crc32로
계산해 프로세스 재실행 간 재현성을 보장한다.

흐름:
  sample.parquet(블로킹 키) + features.parquet(정렬 확인용) 로드
   → 후보 3종 각각: 블로킹 → 블록별 엣지 생성 → 대칭화
   → data/processed/graph/edges_{geo,temporal,weapon}_k{k}.npy 저장
   → 후보별 평균 차수·고립 노드·연결 성분 계산
   → outputs/graph_edge_comparison.csv에 append

--k로 차수 상한을 바꿔가며 재실행 가능(ablation). 06_train_gnn.py의
--k_neighbors가 여기서 만든 파일명과 맞물려 있으니 같이 바꿔야 한다.
"""
import argparse
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config as C


def _block_edges(idx, k, seed):
    """idx: 한 블록에 속한 전역 행 위치. (2,E) 방향 쌍 반환."""
    n = len(idx)
    if n <= 1:
        return np.empty((2, 0), dtype=np.int64)
    if n <= k + 1:
        i, j = np.triu_indices(n, k=1)          # 소규모 블록: 완전 연결
        return np.stack([idx[i], idx[j]])
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    shuffled = idx[order]
    src = np.repeat(shuffled, k)
    offsets = np.tile(np.arange(1, k + 1), n)
    dst = shuffled[(np.repeat(np.arange(n), k) + offsets) % n]  # 셔플 순서에서 다음 k개(원형)
    return np.stack([src, dst])


def _symmetrize(directed):
    """방향 쌍을 무방향(양방향 모두 저장, PyG 포맷)으로. self-loop·중복 제거."""
    if directed.shape[1] == 0:
        return directed
    both = np.concatenate([directed, directed[::-1]], axis=1)
    both = both[:, both[0] != both[1]]
    pairs = np.unique(both.T, axis=0)
    return pairs.T.astype(np.int64)


def build_block_graph(df, block_cols, k, base_seed):
    parts = []
    for block_key, sub in df.groupby(block_cols, sort=False):
        seed = (base_seed + zlib.crc32(str(block_key).encode())) % (2 ** 31)
        parts.append(_block_edges(sub.index.to_numpy(), k, seed))
    directed = np.concatenate(parts, axis=1) if parts else np.empty((2, 0), dtype=np.int64)
    return _symmetrize(directed)


def graph_stats(edges, n_nodes):
    """edges: 대칭화된(양방향 모두 저장) (2,E) 배열이므로 edges.shape[1]은 무방향 엣지 수의 2배."""
    e_directed = edges.shape[1]
    deg = np.zeros(n_nodes, dtype=np.int64)
    if e_directed > 0:
        np.add.at(deg, edges[0], 1)
    adj = coo_matrix((np.ones(e_directed), (edges[0], edges[1])), shape=(n_nodes, n_nodes))
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    return {
        "n_edges": e_directed // 2,
        "avg_degree": deg.mean(),
        "isolated_nodes": int((deg == 0).sum()),
        "isolated_pct": (deg == 0).mean() * 100,
        "n_components": int(n_components),
        "largest_component_pct": comp_sizes.max() / n_nodes * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=C.K_NEIGHBORS,
                         help="블록당 노드 차수 상한 (ablation용, 기본 config.K_NEIGHBORS)")
    args = parser.parse_args()
    k = args.k

    sample_df = pd.read_parquet(C.PROCESSED_DIR / "sample.parquet")
    feat_df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
    assert len(sample_df) == len(feat_df), "행 수 불일치: sample vs features"
    assert (sample_df[C.TARGET_BIN].values == feat_df[C.TARGET_BIN].values).all(), \
        "행 순서 불일치: sample.parquet과 features.parquet이 위치 기준으로 정렬돼 있지 않음"
    n = len(sample_df)
    print(f"[load] sample {n:,}행, features {n:,}행 (정렬 확인 완료), k={k}")

    candidates = {
        "geo": C.GEO_BLOCK_COLS,
        "temporal": C.TEMPORAL_BLOCK_COLS,
        "weapon": C.WEAPON_BLOCK_COLS,
    }

    rows = []
    for name, block_cols in candidates.items():
        sizes = sample_df.groupby(block_cols, sort=False).size()
        print(f"[block:{name}] {block_cols} {len(sizes)}개 블록, "
              f"최대 {sizes.max():,}행, 최소 {sizes.min()}행")

        edges = build_block_graph(sample_df, block_cols, k, C.RANDOM_STATE)
        stats = graph_stats(edges, n)
        print(f"[edges:{name}] 무방향 {stats['n_edges']:,}개, 평균 차수 {stats['avg_degree']:.2f}")
        print(f"[stats:{name}] 고립노드 {stats['isolated_nodes']:,}개({stats['isolated_pct']:.2f}%), "
              f"연결성분 {stats['n_components']:,}개, 최대성분 {stats['largest_component_pct']:.1f}%")

        out = C.GRAPH_DIR / f"edges_{name}_k{k}.npy"
        np.save(out, edges)
        print(f"[save] {out}")

        rows.append({"candidate": name, "k": k, **stats})

    table = pd.DataFrame(rows).set_index("candidate")
    out_csv = C.OUTPUT_DIR / "graph_edge_comparison.csv"
    table.to_csv(out_csv, mode="a", header=not out_csv.exists(), encoding="utf-8-sig")
    print(f"\n[save] {out_csv} (append)")
    print(table)


if __name__ == "__main__":
    main()
