"""
04_build_graph.py — 엣지 후보 3종 구성 + 비교표

개발계획서가 제시한 3가지 엣지 후보(지리·시간·수법 유사)를 각각 독립적으로
구성해 평균 차수·연결 성분 등을 비교한다. 최종 그래프 선택과 GNN 학습
(experiments/train_gnn.py)은 이 비교표를 보고 다음 단계에서 결정한다.

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

--k로 차수 상한을 바꿔가며 재실행 가능(ablation). experiments/train_gnn.py의
--k_neighbors가 여기서 만든 파일명과 맞물려 있으니 같이 바꿔야 한다.
"""
import argparse
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config as C
from clear import similarity
from clear.data import load_xy
from clear.graph import edges_path


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


def build_block_graph(df, block_cols, k, base_seed, Z=None):
    """블록별 엣지 생성 후 대칭화.

    Z가 None이면 셔플-링(무작위 k개). Z를 주면 **유사도 상위 k개**로 잇는다
    (clear.similarity.block_topk). 어느 쪽이든 블록별 시드는 crc32로 계산해
    프로세스 재실행 간 재현성을 유지한다 -- Python hash()는 실행마다 달라진다.
    """
    parts = []
    for block_key, sub in df.groupby(block_cols, sort=False):
        idx = sub.index.to_numpy()
        seed = (base_seed + zlib.crc32(str(block_key).encode())) % (2 ** 31)
        if Z is None:
            parts.append(_block_edges(idx, k, seed))
        elif len(idx) > 1:
            nb = similarity.block_topk(Z, idx, k, seed)
            src = np.repeat(idx, nb.shape[1])
            parts.append(np.stack([src, idx[nb.ravel()]]))
    directed = np.concatenate(parts, axis=1) if parts else np.empty((2, 0), dtype=np.int64)
    return _symmetrize(directed)


def degree_matched_control(df, block_cols, edges, n_nodes, base_seed):
    """랭킹 그래프와 **차수 분포가 같은** 블록 내 무작위 그래프(대조군).

    왜 필요한가: 셔플-링은 준정규 차수(모든 노드가 정확히 k개를 고른다)지만 랭킹은
    **허브를 만든다** -- '전형적인' 사건은 많은 노드의 top-k에 들고 특이한 사건은
    아무에게도 안 든다. 그래서 랭킹 그래프의 정확도가 달라져도 "랭킹 때문인가
    차수 분포 때문인가"를 가를 수 없다. mitigate_graph에서 무작위 대조군 없이는
    아무것도 증명되지 않았던 것과 같은 문제다.

    구현은 블록 내 stub matching(configuration model): 각 노드의 차수만큼 stub을
    만들고 블록 안에서 무작위로 짝지은 뒤, 자기루프·중복은 버린다(그만큼 차수가
    근사가 되므로 실제 차수 분포를 결과에 함께 보고한다).
    """
    deg = np.zeros(n_nodes, dtype=np.int64)
    if edges.shape[1]:
        np.add.at(deg, edges[0], 1)
    parts = []
    for block_key, sub in df.groupby(block_cols, sort=False):
        idx = sub.index.to_numpy()
        d = deg[idx]
        if d.sum() < 2:
            continue
        rng = np.random.default_rng(
            (base_seed + zlib.crc32(("ctrl" + str(block_key)).encode())) % (2 ** 31))
        stubs = np.repeat(idx, d)
        rng.shuffle(stubs)
        half = (len(stubs) // 2) * 2
        parts.append(stubs[:half].reshape(2, -1, order="F"))
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
        # 셔플-링은 준정규(std가 작다)지만 랭킹은 허브를 만든다. 두 방식의 정확도
        # 차이를 해석하려면 차수 분포가 얼마나 달라졌는지를 함께 봐야 한다.
        "degree_std": float(deg.std()),
        "degree_max": int(deg.max()),
        "isolated_nodes": int((deg == 0).sum()),
        "isolated_pct": (deg == 0).mean() * 100,
        "n_components": int(n_components),
        "largest_component_pct": comp_sizes.max() / n_nodes * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=C.K_NEIGHBORS,
                         help="블록당 노드 차수 상한 (ablation용, 기본 config.K_NEIGHBORS)")
    parser.add_argument("--rank", choices=similarity.SCHEMES, default=None,
                         help="블록 내 셔플-링 대신 blind 특성 유사도 상위 k로 잇는다. "
                              "인코딩 대결 결과 geo는 onehot, temporal은 ordinal이 낫다 "
                              "(outputs/edge_relatedness_similarity.csv).")
    parser.add_argument("--control", action="store_true",
                         help="--rank과 함께: 차수 분포를 맞춘 무작위 그래프도 만든다 "
                              "(랭킹 효과와 차수 효과를 가르는 대조군)")
    parser.add_argument("--candidates", nargs="*",
                         default=["geo", "temporal", "weapon"],
                         help="구성할 엣지 후보")
    args = parser.parse_args()
    k = args.k
    mode = f"rank_{args.rank}" if args.rank else None

    sample_df = pd.read_parquet(C.PROCESSED_DIR / "sample.parquet")
    feat_df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
    assert len(sample_df) == len(feat_df), "행 수 불일치: sample vs features"
    assert (sample_df[C.TARGET_BIN].values == feat_df[C.TARGET_BIN].values).all(), \
        "행 순서 불일치: sample.parquet과 features.parquet이 위치 기준으로 정렬돼 있지 않음"
    n = len(sample_df)
    print(f"[load] sample {n:,}행, features {n:,}행 (정렬 확인 완료), k={k}")

    all_candidates = {
        "geo": C.GEO_BLOCK_COLS,
        "temporal": C.TEMPORAL_BLOCK_COLS,
        "weapon": C.WEAPON_BLOCK_COLS,
    }
    candidates = {c: all_candidates[c] for c in args.candidates}

    Z = None
    if args.rank:
        # 랭킹은 반드시 blind 특성으로 한다. sighted 유사도의 oracle 이득은 blind의
        # 6배지만(geo 0.1163 vs 0.0186) 그 대부분이 '같은 인종끼리 잇기'라, 이
        # 프로젝트가 규명한 인종 누출 경로를 넓히는 대가로 사는 정확도다.
        X, _ = load_xy(blind=True)
        Z = similarity.build(X, args.rank)
        print(f"[rank] {similarity.describe(X, args.rank)}  (blind 특성)")

    rows = []
    for name, block_cols in candidates.items():
        sizes = sample_df.groupby(block_cols, sort=False).size()
        print(f"[block:{name}] {block_cols} {len(sizes)}개 블록, "
              f"최대 {sizes.max():,}행, 최소 {sizes.min()}행")

        arms = [(mode, build_block_graph(sample_df, block_cols, k, C.RANDOM_STATE, Z))]
        if args.rank and args.control:
            arms.append((f"{mode}_control",
                         degree_matched_control(sample_df, block_cols, arms[0][1],
                                                n, C.RANDOM_STATE)))

        for arm_mode, edges in arms:
            stats = graph_stats(edges, n)
            label = f"{name}/{arm_mode or 'shuffle'}"
            print(f"[edges:{label}] 무방향 {stats['n_edges']:,}개, "
                  f"평균 차수 {stats['avg_degree']:.2f} (std {stats['degree_std']:.2f}, "
                  f"최대 {stats['degree_max']:,})")
            print(f"[stats:{label}] 고립노드 {stats['isolated_nodes']:,}개"
                  f"({stats['isolated_pct']:.2f}%), 연결성분 {stats['n_components']:,}개, "
                  f"최대성분 {stats['largest_component_pct']:.1f}%")

            out = edges_path(name, k, arm_mode)
            np.save(out, edges)
            print(f"[save] {out}")
            rows.append({"candidate": name, "k": k,
                         "mode": arm_mode or "shuffle", **stats})

    table = pd.DataFrame(rows)
    out_csv = C.OUTPUT_DIR / "graph_edge_comparison.csv"
    # 열이 늘었으므로(mode/degree_*) append가 아니라 **읽고 합쳐 다시 쓴다**.
    # 헤더보다 필드가 많은 행을 append하면 pandas가 그 파일을 아예 못 읽게 된다 --
    # 옛 metrics.csv가 정확히 그렇게 망가졌다(CLAUDE.md의 ParserError 사례).
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if "mode" not in old.columns:
            old["mode"] = "shuffle"          # 이전 실행은 전부 셔플-링이었다
        table = pd.concat([old, table], ignore_index=True)
    cols = ["candidate", "mode", "k"] + [c for c in table.columns
                                         if c not in ("candidate", "mode", "k")]
    table = table[cols]
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out_csv}")
    print(table.tail(len(rows)).to_string(index=False))


if __name__ == "__main__":
    main()
