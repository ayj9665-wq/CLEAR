"""엣지 로딩·조합 — 04_build_graph.py가 저장한 .npy를 읽어 edge_index를 만든다.

이전에는 geo_temporal(합집합) 같은 "엣지 대수"가 experiments/train_gnn.py:build_data
안에 섞여 있었다 — 엣지 구성 로직이 04(생성)와 06(조합) 두 곳에 나뉘어 있던
셈이다. 조합 로직을 여기로 모아 04가 만든 산출물을 읽는 한 곳으로 둔다.
"""
import numpy as np

import config as C


def load_edges(name, k):
    """단일 후보(geo/temporal/weapon)의 대칭화된 (2,E) 엣지 배열."""
    return np.load(C.GRAPH_DIR / f"edges_{name}_k{k}.npy")


def build_edge_index(edge_type, k):
    """edge_type에 해당하는 (2,E) 엣지 배열.

    'a_b' 꼴이면 a·b 엣지의 합집합(중복 제거) — 예: geo_temporal은 geo의 강한
    신호 + temporal의 균일한 연결성이 보완되는지 보는 실험용 조합. 단일 후보는
    이름에 '_'가 없으므로(geo/temporal/weapon) 이 분기로 구분된다.
    """
    if "_" in edge_type:
        arrs = [load_edges(part, k) for part in edge_type.split("_")]
        return np.unique(np.concatenate(arrs, axis=1).T, axis=0).T.astype(np.int64)
    return load_edges(edge_type, k)
