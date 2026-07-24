"""엣지 수준 완화 — 민감속성 동종(homophilous) 엣지를 떨어뜨린 그래프를 만든다.

진단(edge_homophily.py + 07)이 특정한 병변은 **geo 엣지의 인종 동종성**이다.
후처리(08)가 결과를 사후 보정한다면, 여기서는 **원인을 직접 건드린다** — 같은
인종끼리 이어진 엣지를 확률 p로 제거해 메시지 전달이 인종 정보를 덜 나르게 한다
(FairDrop 계열).

이 모듈이 처방이자 **메커니즘의 최종 검정**인 이유: §5-5의 geo vs temporal 비교는
동종성 외에 블록 크기·밀도도 함께 달랐다. 같은 geo 엣지 안에서 동종 엣지만 골라
빼면 **동종성만 단독으로 조작**한 것이 된다.

**대조군이 없으면 아무것도 증명되지 않는다.** 동종 엣지를 빼면 엣지 수도 준다.
따라서 격차가 줄어도 그게 "동종성이 빠져서"인지 "그래프가 희박해져서"인지 구분되지
않는다. 그래서 항상 **같은 개수만큼 무작위로 뺀 그래프**를 짝으로 돌린다. 동종
제거에서만 격차가 줄고 무작위 제거에서는 안 줄어야 동종성이 원인이다.

**대칭성 주의**: 04_build_graph.py의 엣지는 (i,j)와 (j,i)가 모두 든 대칭 배열이다.
한 방향만 떨어뜨리면 그래프가 방향성을 갖게 돼 메시지 전달이 비대칭해진다. 그래서
무향 쌍(i<j)으로 접어서 제거를 결정하고 다시 대칭화한다.

**배포 시 주의**: 여기서 만든 그래프는 학습·추론에 모두 쓰인다. 즉 새 사건의 엣지를
정할 때도 인종을 알아야 한다. "배포 시 민감속성이 불필요"하려면 학습 때만 떨어뜨리고
추론은 원본 그래프로 하는 변형이어야 하며, 그건 별개의 실험이다.
"""
import numpy as np

import config as C


def to_pairs(edges):
    """대칭 (2,E) 배열 → 무향 쌍 (2,E/2). 자기루프는 버린다."""
    src, dst = edges[0], edges[1]
    keep = src < dst
    return edges[:, keep]


def symmetrize(pairs):
    """무향 쌍 → 양방향 (2,2E) 배열. PyG edge_index가 기대하는 형식."""
    return np.concatenate([pairs, pairs[::-1]], axis=1).astype(np.int64)


def homophily(pairs, codes):
    """무향 쌍 중 같은 그룹끼리 이어진 비율."""
    return float((codes[pairs[0]] == codes[pairs[1]]).mean()) if pairs.shape[1] else np.nan


def drop_homophilous(pairs, codes, p, seed=None):
    """같은 그룹끼리 이어진 쌍을 확률 p로 제거. 이종 쌍은 건드리지 않는다."""
    rng = np.random.default_rng(C.RANDOM_STATE if seed is None else seed)
    same = codes[pairs[0]] == codes[pairs[1]]
    drop = same & (rng.random(pairs.shape[1]) < p)
    return pairs[:, ~drop]


def drop_random(pairs, n_keep, seed=None):
    """그룹과 무관하게 무작위로 n_keep개만 남긴다(희박화 대조군)."""
    rng = np.random.default_rng(C.RANDOM_STATE if seed is None else seed)
    n = pairs.shape[1]
    if n_keep >= n:
        return pairs
    idx = rng.choice(n, size=n_keep, replace=False)
    idx.sort()                     # 열 순서 보존(결정성·디버깅 편의)
    return pairs[:, idx]


def build(edges, codes, p, mode, seed=None):
    """수정된 대칭 edge_index + 진단용 통계.

    mode='homophily' 는 동종 쌍을 확률 p로 제거하고, mode='random' 은 그 결과와
    **같은 개수**를 무작위로 남긴다 — 두 팔의 엣지 수가 같아야 비교가 성립한다.
    """
    pairs = to_pairs(edges)
    dropped = drop_homophilous(pairs, codes, p, seed=seed)
    if mode == "homophily":
        out = dropped
    elif mode == "random":
        out = drop_random(pairs, dropped.shape[1], seed=seed)
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")
    return symmetrize(out), {
        "mode": mode, "p": p,
        "n_pairs": int(out.shape[1]),
        "n_pairs_full": int(pairs.shape[1]),
        "kept_frac": out.shape[1] / pairs.shape[1] if pairs.shape[1] else np.nan,
        "homophily": homophily(out, codes),
        "homophily_full": homophily(pairs, codes),
    }
