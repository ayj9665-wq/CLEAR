"""블록 내 사건 유사도의 특성 표현 -- '어느 20개를 이웃으로 삼을까'의 정렬 기준.

## 왜 별도 모듈인가

04_build_graph.py는 블록 안에서 셔플-링 무작위 페어링으로 이웃을 고른다. 그 근거는
"블록 안에는 랭킹할 신호가 없다"였는데, experiments/edge_relatedness.py가 가해자
프로필을 외부 라벨로 써서 그 전제를 반증했다(oracle_lift > 0). 그렇다면 랭킹으로
바꿀 수 있는데, **무엇으로 랭킹하느냐**가 남는다. 이 모듈이 그 후보들을 담고,
edge_relatedness의 oracle이 후보를 고른 뒤 04가 같은 정의로 그래프를 만든다 --
두 곳이 같은 함수를 불러야 "잰 유사도"와 "만든 그래프"가 갈리지 않는다.

## 원핫 코사인이 여기서 퇴화하는 이유 (측정 근거)

03_features.py는 drop_first 없이 전면 원핫하고 결측도 'Unknown' 범주로 남기므로,
**모든 범주형 필드가 행마다 정확히 1개의 1**을 갖는다(검증됨). blind X의 필드는
6개(Weapon 16 / Month 12 / Age Group 21 / Agency Type 5 / State 3 / Decade 4 더미)
+ 수치 Victim Count. 따라서 행 L2 노름이 거의 상수(sqrt(6)~sqrt(7))이고,

    코사인 유사도 = (일치한 필드 수) / (필드 수)

가 된다. nominal 데이터의 정식 유사도(simple matching coefficient)이므로 틀린 것은
아니지만, geo 블록 안에서는 State가 상수라 변하는 필드가 5개뿐 -> **유사도가
{0, .2, .4, .6, .8, 1} 6단계 계단함수**가 된다. 랭킹이 아니라 거친 버킷팅이다.
실제로 최대 블록(LA 44,511행)에서 **노드의 50.2%가 자기와 완전히 동일한 행들만으로
이웃 20개를 채울 수 있다** -- 그 노드들에게 '유사도 랭킹'은 완전일치 집합 안의
셔플-링과 같다.

원핫이 버리는 것 셋:
  1. 순서   Age Group(21구간)·Decade는 순서형인데 '25-29 vs 30-34'와
            '25-29 vs 95-99'가 똑같이 멀다.
  2. 주기   Month는 12월과 1월이 인접인데 원핫에서는 최대로 멀다.
  3. 정보량 Handgun 일치(표본의 50.9%)와 Poison 일치(0.07%)가 같은 무게다.

## 이 모듈의 처방

  ordinal  Age Group·Decade·Victim Count를 **사분원 위 2차원**으로,
           Month를 **원 위 2차원**으로 embedding한다. 내적이
           cos(pi*(t_i-t_j)/2) / cos(2pi*(m_i-m_j)/12)가 되어 '가까울수록 크다'가
           내적 한 번으로 계산된다(BLAS 유지). 각 블록의 노름 기여가 1로 고정되므로
           원핫 필드 하나와 척도가 같다.
  idf      원핫 열 c를 sqrt(w_c), w_c = -log(p_c)로 가중한다. 내적이
           sum_c w_c * 1[일치]가 되어 희귀 범주 일치가 제대로 평가된다. 순서/주기
           필드는 그 필드의 엔트로피 H_f로 진폭을 맞춰(sqrt(H_f)) 같은 정보 척도에
           놓는다.

## 정규화 규약 -- 코사인이 아니라 **내적**을 쓴다

가중이 균일하지 않으면 L2 정규화가 해롭다: 희귀 범주를 가진 행은 노름이 커져
코사인이 그 행을 **깎아내린다**(원하는 것의 반대). 질의 행 i를 고정하면 i의 노름은
후보 전체에 걸린 상수라 순위를 바꾸지 않으므로, 정규화 없이 내적으로 순위를 매기는
것이 정확히 'sum_c w_c * 일치'를 랭킹하는 것이 된다.

예외는 'onehot' 하나다. 이미 발표된 표(outputs/edge_relatedness.csv)가 minmax+L2
코사인으로 측정됐으므로 재현을 위해 그 경로를 그대로 보존한다. 위에서 보았듯 행
노름이 거의 상수라 코사인과 내적의 순위 차이는 무시할 수준이다.

## 블록 내 상수 필드

제거할 필요가 없다. 질의 행을 고정하면 상수 필드는 모든 후보에게 같은 값을 더하므로
argsort를 바꾸지 않는다(내적 규약이라서 성립한다 -- 코사인이었다면 정규화가 섞어서
제거가 필요했다).
"""
import numpy as np
import pandas as pd

import config as C

SCHEMES = ["onehot", "onehot_idf", "ordinal", "ordinal_idf"]

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

UNKNOWN = "Unknown"
# 수치 열의 상한(이 값 이상은 같게 본다). Victim Count는 중앙값 0, 최대 10의
# 롱테일이라 상한 없이 척도화하면 극단값 한둘이 전 구간을 압축한다.
NUMERIC_CLIP = {"Victim Count": 5.0}
_EPS = 1e-12


def _split_fields(X):
    """열 이름 'Field=Category' -> {필드: [열…]}. '='가 없으면 수치 열."""
    fields, numeric = {}, []
    for c in X.columns:
        if "=" in c:
            fields.setdefault(c.split("=", 1)[0], []).append(c)
        else:
            numeric.append(c)
    return fields, numeric


def _ordinal_value(field, cat):
    """순서형 범주 -> 실수(등간격). 미상이면 None."""
    if cat == UNKNOWN:
        return None
    if field == "Age Group":
        return float(cat.split("-")[0]) / C.AGE_BIN_WIDTH
    if field == "Decade":
        return float(cat)
    return None


ORDINAL_FIELDS = ("Age Group", "Decade")
CYCLIC_FIELDS = ("Month",)


def _minmax(Xv):
    """이진 아닌 열만 [0,1]로. 한 열(Victim Count)이 유사도를 지배하는 것을 막는다."""
    mn, mx = Xv.min(axis=0), Xv.max(axis=0)
    need = (mn < 0) | (mx > 1)
    if need.any():
        span = np.where(mx > mn, mx - mn, 1.0)
        Xv[:, need] = ((Xv - mn) / span)[:, need]
    return Xv


def _l2(Xv):
    return Xv / np.maximum(np.linalg.norm(Xv, axis=1, keepdims=True), _EPS)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _arc(t, amp):
    """t in [0,1] -> 사분원 위 2차원. 내적 = amp^2 * cos(pi/2 * (t_i - t_j)).

    같으면 최대(amp^2), 멀어질수록 단조 감소하고 양 끝이 직교(0)한다. 노름 기여가
    t와 무관하게 amp로 고정되므로 원핫 필드 하나와 척도가 같다.
    """
    th = (np.pi / 2.0) * t
    return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32) * amp


def _circle(m, period, amp):
    """m in [0,period) -> 원 위 2차원. 내적 = amp^2 * cos(2pi*(m_i-m_j)/period).

    주기형(Month) 전용 -- 12월과 1월이 인접하게 나온다.
    """
    th = 2.0 * np.pi * m / period
    return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32) * amp


def build(X, scheme="onehot"):
    """특성 DataFrame -> Z. 'Z @ Z.T 상위 k'가 곧 그 scheme의 이웃 랭킹이 된다.

    반환 Z는 float32이며, 블록 내에서 행 부분집합을 잘라 써도 의미가 보존된다
    (열 척도가 전역 빈도로만 정해지고 행에 의존하지 않는다).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"알 수 없는 scheme: {scheme} (가능: {SCHEMES})")
    if scheme == "onehot":
        # 레거시 경로: 발표된 표를 그대로 재현한다(minmax + L2 코사인).
        return _l2(_minmax(X.values.astype(np.float32)))

    idf = scheme.endswith("_idf")
    ordinal = scheme.startswith("ordinal")
    fields, numeric = _split_fields(X)
    blocks = []

    for field, cols in fields.items():
        B = X[cols].values.astype(np.float32)
        p = B.mean(axis=0)                       # 원핫이므로 열 평균 = 범주 빈도
        H = _entropy(p)
        cats = [c.split("=", 1)[1] for c in cols]

        if ordinal and field in CYCLIC_FIELDS:
            idx = {c: i for i, c in enumerate(MONTH_ORDER)}
            m = np.array([idx.get(c, 0) for c in cats], dtype=np.float32)[B.argmax(1)]
            blocks.append(_circle(m, len(MONTH_ORDER), np.sqrt(H) if idf else 1.0))
            continue

        if ordinal and field in ORDINAL_FIELDS:
            vals = [_ordinal_value(field, c) for c in cats]
            known_cat = np.array([v is not None for v in vals])
            num = np.array([0.0 if v is None else v for v in vals], dtype=np.float32)
            pick = B.argmax(1)
            raw, known = num[pick], known_cat[pick]
            lo, hi = raw[known].min(), raw[known].max() if known.any() else (0.0, 1.0)
            t = (raw - lo) / max(hi - lo, _EPS)
            arc = _arc(t, np.sqrt(H) if idf else 1.0)
            # 미상 행은 원호 기여를 0으로 죽이고(아무 나이와도 안 닮음) 별도
            # 지시열로 미상끼리만 일치시킨다. 임의의 값을 채우면 그 근처 나이와
            # 가짜로 닮게 된다.
            arc[~known] = 0.0
            unk = (~known).astype(np.float32)
            w_unk = np.sqrt(-np.log(max(unk.mean(), _EPS))) if idf else 1.0
            blocks.append(arc)
            blocks.append((unk * w_unk).reshape(-1, 1).astype(np.float32))
            continue

        # 명목형: 열마다 sqrt(w_c)를 곱하면 내적이 곧 w_c * 1[일치]가 된다.
        w = np.sqrt(-np.log(np.maximum(p, _EPS))) if idf else np.ones_like(p)
        blocks.append(B * w)

    for c in numeric:
        v = X[c].values.astype(np.float32)
        cap = NUMERIC_CLIP.get(c)
        v = np.clip(v, 0, cap) if cap else v
        span = max(float(v.max() - v.min()), _EPS)
        t = (v - v.min()) / span
        if ordinal:
            blocks.append(_arc(t, 1.0))          # 수치도 순서형으로 취급
        else:
            blocks.append(t.reshape(-1, 1))

    return np.ascontiguousarray(np.hstack(blocks), dtype=np.float32)


def block_topk(Z, idx, k, seed, chunk=2000):
    """블록 idx 안에서 각 노드의 유사도 상위 k 이웃. (len(idx), k) 위치 배열 반환.

    반환값은 **idx 안에서의 위치**(0..len(idx)-1)다 -- 전역 행 번호로 바꾸려면
    호출부가 idx[...]로 매핑한다.

    ## 왜 청크인가

    유사도 행렬은 블록 크기의 제곱이다. 최대 geo 블록(LA 44,511행)이면
    44,511^2 * 4B = 7.9GB로 한 번에 못 만든다. 행을 chunk개씩 잘라
    (chunk, n) 행렬만 만들고 top-k를 뽑은 뒤 버린다 -- LA 기준 chunk=2000이면
    356MB. 전체 연산량은 그대로(geo 약 370 GFLOP)이고 BLAS가 처리하므로
    CPU에서 십수 초다. GPU가 필요 없어 04_build_graph.py를 torch-free로 유지한다.

    ## 왜 동점 처리가 필수인가

    이 데이터의 유사도는 동점이 지배적이다. blind 특성 벡터가 190,326행 중
    34,189개(18.0%)뿐이고, LA 블록에서는 노드의 50.2%가 자기와 완전히 동일한
    행들만으로 이웃 20개를 채울 수 있다. 즉 'top-k'의 경계는 거의 항상 동점
    구간 안에 있다.

    argpartition에 그냥 맡기면 동점 중 **낮은 인덱스**가 뽑히는 경향이 있는데,
    행 순서는 원본 CSV 순서(대략 관할·연도순)라서 그게 같은 관할·같은 시기
    이웃으로 치우친다 -- 무작위가 아니라 **숨은 편향**이다. 그래서 seed 고정
    난수 키로 동점을 명시적으로 흔든다.

    구현: float32 BLAS의 오차(~1e-6 상대)가 '진짜 동점'을 미세하게 갈라놓으므로,
    먼저 1e-5 격자로 양자화해 동점을 동점으로 만든 뒤 1e-6 크기의 난수 키를
    더한다. 격자보다 작은 차이는 어차피 의미가 없다.
    """
    n = len(idx)
    kk = int(min(k, n - 1))
    if kk <= 0:
        return np.empty((n, 0), dtype=np.int64)
    Zb = np.ascontiguousarray(Z[idx])
    # 점수 척도: 자기 자신과의 내적 최대값. 양자화 격자를 척도에 맞추기 위함.
    scale = float(np.maximum((Zb * Zb).sum(axis=1).max(), _EPS))
    jitter = np.random.default_rng(seed).random(n).astype(np.float32) * 1e-6

    out = np.empty((n, kk), dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = Zb[s:e] @ Zb.T
        np.round(S / scale, 5, out=S)
        S += jitter[None, :]
        S[np.arange(e - s), np.arange(s, e)] = -np.inf   # 자기 자신 제외
        out[s:e] = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    return out


def describe(X, scheme):
    """Z의 모양과 필드 구성을 한 줄로. 콘솔 확인용."""
    Z = build(X, scheme)
    fields, numeric = _split_fields(X)
    return f"{scheme}: Z {Z.shape}  (필드 {len(fields)}개 + 수치 {len(numeric)}개)"
