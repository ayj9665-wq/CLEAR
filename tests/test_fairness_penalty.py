"""층 표준화 공정성 벌점 — 계획서 §8 검수표를 테스트로 옮긴 것.

G1(층 1개면 pooled와 수치적으로 일치)이 이 파일의 존재 이유다. 층 벌점은 pooled
벌점의 일반화이므로, 층을 하나로 두면 정의상 같은 값이 나와야 한다. 다르면 그것은
개입의 성질이 아니라 **구현 오류**이고, 그 구분을 실험 결과를 보고 하려 하면 이미
늦다 — alpha 격자를 하룻밤 돌린 뒤에 "층 벌점이 pooled와 다르게 움직인다"는 관찰이
버그인지 발견인지 알 수 없게 된다.

나머지 넷(고정 가중 / 재정규화 / 셀 하한 / 제외 코드)은 계획서 §4-3·§4-4가 정의에
포함시킨 규칙이라 같은 이유로 여기 있다.

float64로 검사한다 — 학습은 float32로 돌지만, 등가성을 재는 자리에서 부동소수
오차와 구현 차이를 섞으면 G1이 무엇을 통과시켰는지 모르게 된다.
"""
import pytest

torch = pytest.importorskip("torch")

from clear.gnn import fairness_penalty, _pooled_penalty   # noqa: E402


def _t(x, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype)


def _codes(x):
    return torch.tensor(x, dtype=torch.long)


# 그룹 A/B가 섞인 한 벌. 값은 임의지만 그룹평균이 서로 다르도록 잡았다.
P = _t([0.10, 0.20, 0.30, 0.40, 0.55, 0.65, 0.75, 0.85, 0.15, 0.95])
C = _codes([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])


# --- G1: 층 1개 = pooled -----------------------------------------------------

@pytest.mark.parametrize("min_count", [0, 3])
def test_g1_single_stratum_equals_pooled(min_count):
    """계획서 G1. 층을 1개로 두면 벌점 값이 현행 pooled 벌점과 일치한다."""
    pooled = _pooled_penalty(P, C, min_count)
    strat = fairness_penalty(P, C, min_count, strata=torch.zeros_like(C),
                             min_cell=min_count)
    assert strat.item() == pytest.approx(pooled.item(), abs=1e-12)


def test_g1_holds_when_a_group_is_below_the_floor():
    """하한 미만 그룹이 있어도 등가여야 한다 — pooled는 그 그룹을 기여에서만 빼고
    분모(전체 평균·가중치 분모)에는 남긴다. 층 판이 그 규약을 안 따르면 여기서 갈린다."""
    codes = _codes([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])   # 그룹 2는 n=2
    pooled = _pooled_penalty(P, codes, 3)
    strat = fairness_penalty(P, codes, 0, strata=torch.zeros_like(codes), min_cell=3)
    assert strat.item() == pytest.approx(pooled.item(), abs=1e-12)


def test_strata_none_is_the_old_path():
    """strata를 안 주면 예전 함수 그대로 — 기존 결과 재현이 여기 걸려 있다."""
    assert fairness_penalty(P, C, 0).item() == _pooled_penalty(P, C, 0).item()


# --- 층이 둘 이상일 때의 정의 -------------------------------------------------

def _expected(pairs, w):
    """층별 (그룹0 값들, 그룹1 값들)과 층가중 -> Σ w_s var_s / Σ w_s.

    두 그룹이면 층 안 분산은 w_a·w_b·(평균차)²다.
    """
    num = den = 0.0
    for (a, b), ws in zip(pairs, w):
        na, nb = len(a), len(b)
        wa, wb = na / (na + nb), nb / (na + nb)
        d = sum(a) / na - sum(b) / nb
        num += ws * wa * wb * d * d
        den += ws
    return num / den


def test_two_strata_matches_the_written_definition():
    p = _t([0.1, 0.3, 0.7, 0.9,      0.2, 0.4, 0.4, 0.6])
    c = _codes([0, 0, 1, 1,          0, 0, 1, 1])
    s = _codes([0, 0, 0, 0,          1, 1, 1, 1])
    w = _t([0.75, 0.25])
    got = fairness_penalty(p, c, 0, strata=s, weights=w, min_cell=1)
    want = _expected([([0.1, 0.3], [0.7, 0.9]), ([0.2, 0.4], [0.4, 0.6])], [0.75, 0.25])
    assert got.item() == pytest.approx(want, abs=1e-12)


def test_weights_are_fixed_not_read_off_the_batch():
    """계획서 §4-3. 배치 구성이 달라져도 층가중은 넘겨준 값이다.

    같은 층별 격차를 유지한 채 층 1의 표본만 3배로 늘린다. 배치에서 가중을 재면
    값이 바뀌고, 고정 가중이면 안 바뀐다."""
    w = _t([0.75, 0.25])
    small_p = _t([0.1, 0.3, 0.7, 0.9,  0.2, 0.4, 0.4, 0.6])
    small_c = _codes([0, 0, 1, 1,      0, 0, 1, 1])
    small_s = _codes([0, 0, 0, 0,      1, 1, 1, 1])
    big_p = torch.cat([small_p, small_p[4:], small_p[4:]])
    big_c = torch.cat([small_c, small_c[4:], small_c[4:]])
    big_s = torch.cat([small_s, small_s[4:], small_s[4:]])

    a = fairness_penalty(small_p, small_c, 0, strata=small_s, weights=w, min_cell=1)
    b = fairness_penalty(big_p, big_c, 0, strata=big_s, weights=w, min_cell=1)
    assert a.item() == pytest.approx(b.item(), abs=1e-12)

    # 대조: 고정 가중을 안 주면(배치에서 계산) 값이 실제로 달라진다 — 위 검사가
    # 우연히 통과한 것이 아님을 보인다.
    a0 = fairness_penalty(small_p, small_c, 0, strata=small_s, min_cell=1)
    b0 = fairness_penalty(big_p, big_c, 0, strata=big_s, min_cell=1)
    assert a0.item() != pytest.approx(b0.item(), abs=1e-6)


def test_absent_stratum_is_dropped_and_weights_renormalize():
    """배치에 없는 층은 빼고 남은 층에 재정규화한다. 0으로 두면 관측하지 않은 것을
    0으로 관측한 것처럼 다루게 된다(standardized_gap_row와 같은 규칙)."""
    p = _t([0.1, 0.3, 0.7, 0.9])
    c = _codes([0, 0, 1, 1])
    s = _codes([0, 0, 0, 0])                 # 층 1은 배치에 없다
    w = _t([0.25, 0.75])
    got = fairness_penalty(p, c, 0, strata=s, weights=w, min_cell=1)
    want = _expected([([0.1, 0.3], [0.7, 0.9])], [0.25])   # = 층 0의 값 그대로
    assert got.item() == pytest.approx(want, abs=1e-12)


def test_cell_floor_drops_the_cell_and_then_the_stratum():
    """계획서 §4-4. 하한 미만 셀은 기여하지 않고, 남은 그룹이 2개 미만이면 그 층이 빠진다."""
    p = _t([0.1, 0.3, 0.7, 0.9,      0.2, 0.4, 0.4, 0.9])
    c = _codes([0, 0, 1, 1,          0, 0, 0, 1])   # 층 1의 그룹 1은 n=1
    s = _codes([0, 0, 0, 0,          1, 1, 1, 1])
    w = _t([0.5, 0.5])
    got = fairness_penalty(p, c, 0, strata=s, weights=w, min_cell=2)
    want = _expected([([0.1, 0.3], [0.7, 0.9])], [0.5])
    assert got.item() == pytest.approx(want, abs=1e-12)

    # 모든 층이 탈락하면 0을 돌려준다(gradient가 붙은 채로).
    dead = fairness_penalty(p, c, 0, strata=s, weights=w, min_cell=5)
    assert dead.item() == 0.0


def test_negative_codes_and_strata_are_excluded():
    """-1은 벌점에서 빠진다 — 코드 쪽은 Unknown·소수 그룹, 층 쪽은 층 라벨이 없는 행."""
    p = _t([0.1, 0.3, 0.7, 0.9, 0.99, 0.01])
    c = _codes([0, 0, 1, 1, -1, 0])
    s = _codes([0, 0, 0, 0, 0, -1])
    base = fairness_penalty(p[:4], c[:4], 0, strata=s[:4], min_cell=1)
    got = fairness_penalty(p, c, 0, strata=s, min_cell=1)
    assert got.item() == pytest.approx(base.item(), abs=1e-12)


def test_penalty_is_differentiable_through_the_strata_path():
    """벌점이 실제로 확률을 민다 — index_add를 거쳐도 gradient가 살아 있어야 한다."""
    p = _t([0.1, 0.3, 0.7, 0.9, 0.2, 0.4, 0.4, 0.6]).requires_grad_(True)
    c = _codes([0, 0, 1, 1, 0, 0, 1, 1])
    s = _codes([0, 0, 0, 0, 1, 1, 1, 1])
    fairness_penalty(p, c, 0, strata=s, weights=_t([0.5, 0.5]), min_cell=1).backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def test_diag_counts_participating_strata():
    """검토 3-5: 고정 가중을 넘겼다는 것만으로는 실효 표준화 인구를 알 수 없다.
    셀 하한 때문에 빠진 층이 몇 개인지를 세어 둔다."""
    p = _t([0.1, 0.3, 0.7, 0.9, 0.2, 0.9])
    c = _codes([0, 0, 1, 1, 0, 1])
    s = _codes([0, 0, 0, 0, 1, 1])      # 층 1은 셀이 각각 1개
    diag = {}
    fairness_penalty(p, c, 0, strata=s, min_cell=2, diag=diag)
    assert diag["steps"] == 1
    assert diag["strata_used"] == 1     # 층 1은 하한에 걸려 빠진다
    assert set(diag["w_seen"]) == {0}
