"""엣지 로딩·조합 — 04_build_graph.py가 저장한 .npy를 읽어 edge_index를 만든다.

이전에는 geo_temporal(합집합) 같은 "엣지 대수"가 experiments/train_gnn.py:build_data
안에 섞여 있었다 — 엣지 구성 로직이 04(생성)와 06(조합) 두 곳에 나뉘어 있던
셈이다. 조합 로직을 여기로 모아 04가 만든 산출물을 읽는 한 곳으로 둔다.
"""
import numpy as np

import config as C


def edges_path(name, k, mode=None):
    """엣지 파일 경로. mode가 None이면 기존 셔플-링 그래프(파일명 무변경).

    **mode는 edge_type과 별도 축이어야 한다.** build_edge_index가 edge_type의 '_'를
    합집합 구분자로 쓰기 때문에(geo_temporal = geo ∪ temporal), 랭킹 그래프를
    'geo_rank' 같은 새 edge_type 이름으로 넣으면 "geo와 rank의 합집합"으로 파싱된다.
    그래서 이름이 아니라 파일명 접미사로 구분한다.
    """
    stem = f"edges_{name}_k{k}" + (f"_{mode}" if mode else "")
    return C.GRAPH_DIR / f"{stem}.npy"


def load_edges(name, k, mode=None):
    """단일 후보(geo/temporal/weapon)의 대칭화된 (2,E) 엣지 배열."""
    return np.load(edges_path(name, k, mode))


def build_edge_index(edge_type, k, mode=None):
    """edge_type에 해당하는 (2,E) 엣지 배열.

    'a_b' 꼴이면 a·b 엣지의 합집합(중복 제거) — 예: geo_temporal은 geo의 강한
    신호 + temporal의 균일한 연결성이 보완되는지 보는 실험용 조합. 단일 후보는
    이름에 '_'가 없으므로(geo/temporal/weapon) 이 분기로 구분된다.

    mode는 그래프 구성 방식(None=셔플-링, rank_*=유사도 랭킹)이며 합집합의 모든
    항에 동일하게 적용된다.

    반환 배열은 **반드시 C 연속**이다. 이유가 사소하지 않다: 옛 04_build_graph.py는
    `np.unique(pairs, axis=0).T.astype(...)`로 만들었는데 `astype`의 기본 order='K'가
    전치 순서를 보존해 저장된 .npy가 **Fortran 순서**였다(합집합 분기의 `.T`도 같다).
    full-batch 경로는 torch가 stride를 알아서 처리해 이게 드러나지 않았지만,
    NeighborLoader가 쓰는 pyg-lib의 index_sort는 연속 입력을 요구해
    "Input should be contiguous"로 죽는다. 저장 형식이 아니라 **로딩 지점에서**
    보장하는 이유는 이미 저장된 파일들을 그대로 쓰기 위해서다.
    """
    if "_" in edge_type:
        arrs = [load_edges(part, k, mode) for part in edge_type.split("_")]
        out = np.unique(np.concatenate(arrs, axis=1).T, axis=0).T.astype(np.int64)
    else:
        out = load_edges(edge_type, k, mode)
    return np.ascontiguousarray(out)
