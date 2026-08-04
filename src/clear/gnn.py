"""GraphSAGE 모델·학습 로직 — train_gnn과 완화 스윕(clear.sweep)의 공통 출처.

이전에는 이 로직이 전부 학습 스크립트(당시 이름 `06_train_gnn.py`)에 있었고,
하이퍼파라미터 스윕은 숫자로 시작하는 그 파일을 import할 수 없어(문법 오류)
subprocess로 Python을 config마다 다시 띄웠다 — 매번 torch를 재import하고 parquet를
다시 읽었다. 학습 로직을 importable한 여기로 옮기면 호출부가
`from clear.gnn import ...`로 in-process 호출해 데이터를 한 번만 로드한다.
experiments/train_gnn.py는 얇은 CLI 래퍼가 된다.

(스크립트 번호는 그 뒤 없앴고 — experiments/__init__.py 참고 — 하이퍼파라미터
스윕 스크립트 자체는 탐색이 끝난 뒤 지웠다. 우승값은 config.py에 승격돼 있고
결과는 outputs/results.csv의 family=ablation에 남아 있다. 이 모듈은 지금도
train_gnn·sweep이 공유한다.)

seed 반복 설계: split은 config.RANDOM_STATE로 고정하고(=baseline과 동일 test
집합, 비교 가능성 유지) torch seed만 바꿔 학습 분산을 측정한다. GPU scatter
집계의 비결정성 때문에 같은 seed로도 run마다 흔들리므로, 여러 seed의 mean±std로
"마진이 노이즈를 넘는가"를 판정한다.
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config as C   # torch import 전에 필요 (KMP_DUPLICATE_LIB_OK 등 env 설정)
from clear.data import get_split
from clear.graph import build_edge_index
from clear.metrics import evaluate
from clear import results

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import matthews_corrcoef   # val 조기 종료 스칼라(지표 한 벌은 clear.metrics)


class GraphSAGE(nn.Module):
    """x -> SAGEConv -> ReLU -> Dropout -> ... -> SAGEConv -> Linear head -> 로짓(N,1)."""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout, aggr):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_dim, aggr=aggr)])
        self.convs.extend(
            SAGEConv(hidden_dim, hidden_dim, aggr=aggr) for _ in range(num_layers - 1)
        )
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


def default_hp():
    """config.py 기본값의 하이퍼파라미터 한 벌. train_one에 그대로 **전개된다.

    clear.sweep(mitigate_graph·mitigate_loss)이 쓴다 — "기본값에서 출발"이 전제라,
    따로 적어두면 config를 바꿨을 때 한쪽만 따라가 두 실험이 말없이 다른 기준점을
    쓰게 된다. experiments/train_gnn.py는 이걸 안 쓴다(그쪽은 argparse 값이
    출처이고, 기본값은 add_argument의 default가 이미 config다).

    grad_clip은 뺐다 — train_one의 기본값(0.0)이 곧 '안 씀'이고, 10만 노출한다.
    """
    return dict(
        hidden_dim=C.GNN_HIDDEN_DIM, num_layers=C.GNN_NUM_LAYERS,
        dropout=C.GNN_DROPOUT, lr=C.GNN_LR, weight_decay=C.GNN_WEIGHT_DECAY,
        aggr=C.GNN_AGGR, max_epochs=C.GNN_MAX_EPOCHS, patience=C.GNN_PATIENCE,
        val_size=C.GNN_VAL_SIZE,
        minibatch=C.GNN_MINIBATCH, batch_size=C.GNN_BATCH_SIZE,
        num_neighbors=C.GNN_NUM_NEIGHBORS, eval_batch_size=C.GNN_EVAL_BATCH_SIZE,
        fair_min_count=C.GNN_FAIR_MIN_COUNT,
    )


def parse_seeds(s):
    """'42,43,44' -> [42, 43, 44]. 06과 clear.sweep(mitigate_graph·mitigate_loss)이 공유하는 argparse type=.
    seed 규약(split은 고정, torch seed만 변주)이 이 모듈의 소관이라 여기 둔다."""
    return [int(x) for x in str(s).split(",") if x != ""]


def build_data(X, edge_type, k, device, mode=None, minibatch=False):
    """그래프 Data 하나. minibatch면 **CPU에 둔다** — NeighborLoader가 CPU에서
    샘플링하고 배치만 device로 올리는 구조라, 전체 그래프를 GPU에 올리면 애초에
    미니배치를 쓰는 이유(전국 25M 엣지가 full-batch로 안 올라감)가 없어진다."""
    edges = build_edge_index(edge_type, k, mode)
    x = torch.tensor(X.values.astype(np.float32))
    edge_index = torch.from_numpy(edges).long()
    data = Data(x=x, edge_index=edge_index)
    return data if minibatch else data.to(device)


def _loader(data, input_idx, num_neighbors, batch_size, shuffle):
    """NeighborLoader 한 개. import를 함수 안에 두는 이유는 full-batch 경로가
    torch_geometric.loader를 건드리지 않게 하기 위함이다(기존 실행 경로 불변)."""
    from torch_geometric.loader import NeighborLoader
    return NeighborLoader(data, num_neighbors=list(num_neighbors),
                          batch_size=batch_size,
                          input_nodes=input_idx.detach().cpu(),
                          shuffle=shuffle, num_workers=0)


@torch.no_grad()
def _infer(model, loader, device):
    """loader가 가리키는 노드의 로짓. **loader는 재사용해야 한다** — 아래 참고.

    이웃을 샘플링하지 않고 전부 쓴다(호출부가 num_neighbors=-1로 만든다). 학습은
    fanout을 잘라도 되지만 추론은 안 된다: 여기서 나온 proba가 그대로 덤프되어
    diagnose_fairness의 모든 격차, detect_cold_blocks의 E_b, 지도의 색까지 흘러가므로
    샘플링 노이즈를 실으면 그 전부가 재현 불가능해진다. 조기 종료 신호(val MCC)도
    같은 이유로 전체 이웃을 쓴다.

    shuffle=False라 반환 순서는 input_nodes 순서 그대로다(호출부가 그렇게 쓴다).
    """
    model.eval()
    out = []
    for batch in loader:
        batch = batch.to(device)
        # NeighborLoader는 seed 노드를 배치 앞쪽 batch_size개에 놓는다.
        out.append(model(batch.x, batch.edge_index).squeeze(-1)[:batch.batch_size].cpu())
    return torch.cat(out) if out else torch.empty(0)


def prepare(X, y, val_size, device):
    """split + device 텐서 준비. 호출부가 공유(로드·분할 한 번).
    split은 config 기본 random_state로 고정 — 모든 seed·config가 동일 test 집합."""
    split = get_split(y, val_size=val_size)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    train_t = torch.tensor(split.train, dtype=torch.long, device=device)
    val_t = torch.tensor(split.val, dtype=torch.long, device=device)
    test_t = torch.tensor(split.test, dtype=torch.long, device=device)
    return y_t, train_t, val_t, test_t


def _group_gap(pred, codes_np):
    """val 노드의 그룹 간 선택률 max-min. codes_np가 None이면 nan(계산 안 함).

    diagnose_fairness이 재는 selection_rate 격차와 같은 정의를, 조기 종료가 쓸 수 있게 val에서
    계산한 것이다. -1(대상 외)은 제외한다.
    """
    if codes_np is None:
        return np.nan
    rates = [pred[codes_np == g].mean() for g in np.unique(codes_np) if g >= 0
             and (codes_np == g).any()]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def _pooled_penalty(proba, codes, min_count):
    """층 없이 학습 노드(=배치) 전체에서 잰 그룹평균 분산. 원래의 fairness_penalty 본문."""
    keep = codes >= 0
    if not torch.any(keep):
        return proba.sum() * 0.0
    p, c = proba[keep], codes[keep]
    overall = p.mean()
    pen = proba.sum() * 0.0
    n_groups = 0
    for g in torch.unique(c):
        m = c == g
        n_g = int(m.sum())
        if n_g < min_count:
            continue
        n_groups += 1
        w = n_g / c.numel()
        pen = pen + w * (p[m].mean() - overall) ** 2
    return pen if n_groups >= 2 else proba.sum() * 0.0


def _stratified_penalty(proba, codes, strata, weights, min_cell, diag):
    """층 안에서 그룹평균 분산을 재고 **고정 층가중**으로 평균한다.

    `standardized_gap_row`의 직접 표준화와 같은 구조다 — 측정이 "층 안의 비율을 층
    가중으로 평균"하므로 개입도 "층 안의 그룹평균 분산을 층 가중으로 평균"한다.

    셀(층 x 그룹) 하나하나가 아니라 (S, G) 행렬을 한 번에 만들어 계산한다. 층이 51개면
    파이썬 루프로도 되지만 이건 학습 스텝마다 도는 자리라, index_add 두 번으로 끝내는
    편이 스텝 비용을 눈에 띄게 하지 않는다. cnt는 상수(grad 없음)이고 합계 tot만
    proba로부터 미분되므로, 벌점의 gradient 경로는 루프판과 같다.

    층 1개 + min_cell=min_count이면 **_pooled_penalty와 수치적으로 같아야 한다**
    (계획서 G1). 그래서 층 안의 평균·가중치 분모를 pooled와 같은 규약으로 둔다 —
    p̄_s는 그 층의 **남은 노드 전체** 평균이고 w_{g|s}의 분모도 같다. 하한 미만 셀은
    분모에는 남고 기여만 빠진다(pooled에서 min_count 미만 그룹이 그랬듯이).
    """
    keep = (codes >= 0) & (strata >= 0)
    if not torch.any(keep):
        return proba.sum() * 0.0
    p, c, s = proba[keep], codes[keep], strata[keep]

    suid, sinv = torch.unique(s, return_inverse=True)
    guid, ginv = torch.unique(c, return_inverse=True)
    S, G = suid.numel(), guid.numel()
    if G < 2:
        return proba.sum() * 0.0

    flat = sinv * G + ginv
    ones = torch.ones_like(p, dtype=p.dtype)
    cnt = torch.zeros(S * G, dtype=p.dtype, device=p.device).index_add(0, flat, ones).detach()
    tot = torch.zeros(S * G, dtype=p.dtype, device=p.device).index_add(0, flat, p)
    cnt, tot = cnt.view(S, G), tot.view(S, G)

    n_s = cnt.sum(1)                                   # 층별 남은 노드 수(상수)
    ok_cell = cnt >= max(min_cell, 1)
    ok_s = ok_cell.sum(1) >= 2                         # 그룹이 2개 미만이면 그 층을 뺀다
    if not bool(ok_s.any()):
        return proba.sum() * 0.0

    mean = tot / cnt.clamp(min=1)                      # 셀 평균(빈 셀은 0, 가중이 0이라 무해)
    pbar = (tot.sum(1) / n_s.clamp(min=1)).unsqueeze(1)
    w_cell = cnt / n_s.clamp(min=1).unsqueeze(1)
    var_s = (ok_cell * w_cell * (mean - pbar) ** 2).sum(1)

    # 층가중은 **배치가 아니라 학습 노드 전체에서 한 번 계산해 고정**한 값이다
    # (계획서 §4-3). 배치에서 계산하면 스텝마다 표준화 대상 인구가 달라져 벌점의
    # 목표가 흔들린다. weights=None은 테스트/full-batch용 경로다.
    w_s = weights[suid].to(p.dtype) if weights is not None else n_s / n_s.sum()
    w_s = w_s * ok_s                                   # 빠진 층은 가중 0
    w_tot = w_s.sum()
    if diag is not None:
        diag["steps"] = diag.get("steps", 0) + 1
        diag["strata_used"] = diag.get("strata_used", 0) + int(ok_s.sum())
        diag["w_kept"] = diag.get("w_kept", 0.0) + float(w_tot)
        seen = diag.setdefault("w_seen", {})
        for i, sid in enumerate(suid.tolist()):
            if bool(ok_s[i]):
                seen[sid] = seen.get(sid, 0) + 1
    # 배치에 없는 층은 빼고 남은 층에 대해 재정규화한다 — 빼지 않고 0으로 두면
    # 관측하지 않은 것을 0으로 관측한 것처럼 다루게 된다(standardized_gap_row와 동일).
    return (w_s * var_s).sum() / w_tot


def fairness_penalty(proba, codes, min_count=0, *, strata=None, weights=None,
                     min_cell=0, diag=None):
    """그룹 평균 예측확률의 (크기가중) 분산 — Demographic Parity의 미분가능 대리항.

    codes: 노드별 그룹 인덱스, -1은 벌점에서 제외(Unknown 등). 그룹이 둘이면
    이 값은 두 그룹 평균 차이의 제곱에 비례하므로, "선택률 격차를 줄여라"를
    그대로 손실에 넣은 것이 된다. 하드 예측(임계값)은 미분이 안 되므로 시그모이드
    확률의 평균을 쓴다 — diagnose_fairness이 재는 selection_rate의 매끄러운 대리값이다.

    벌점은 **학습 노드에서만** 계산한다(호출부가 train 인덱스를 넘긴다). 민감속성이
    학습 시점에만 필요하고 추론 시점엔 불필요하다는 게 이 방식의 요점이다 —
    후처리(mitigate_threshold)가 배포 시 인종을 알아야 하는 제약을 피한다.

    미니배치 학습에서는 이 값이 **전체 학습 노드가 아니라 배치의** 그룹평균 분산이
    된다. 추정량이 바뀌는 것이므로 두 가지가 따라온다. (1) 배치에 어떤 그룹이 너무
    적게 들어오면 그 그룹 평균이 잡음이라 벌점이 엉뚱한 방향을 가리킨다 →
    min_count 미만인 그룹은 뺀다(남은 그룹이 2개 미만이면 벌점 0). (2) 배치 그룹평균
    분산은 각 그룹평균의 표본분산만큼 **위로 편향**돼 있다(대략 Σ w_g σ²/n_g). 배치를
    키우면 줄어드는 양이지만 0은 아니므로, **alpha 값 자체는 full-batch 결과에서
    이전되지 않는다** — 전국 확장에서 alpha를 다시 훑는 이유 중 하나다.

    strata를 주면 **층 표준화 벌점**이 된다(층표준화벌점 계획서). 재는 쪽이 pooled ->
    주 표준화 -> 카운티 표준화로 세 번 내려가는 동안 누르는 쪽은 pooled로 남아 있었고,
    그 어긋남이 실측된 증상 셋(통과하는 alpha 없음 / 판정 지표에 비단조 / 최적점 이동)의
    원인이다. strata=None이면 이 함수는 예전 그대로 동작한다 — 기존 결과 재현이
    그것에 걸려 있고, 계획서 G1이 "층 1개면 pooled와 수치적으로 일치"를 검사한다.

    strata   노드별 층 인덱스(-1은 벌점에서 제외). codes와 같은 길이.
    weights  층 인덱스로 색인하는 **고정** 층가중(학습 노드 전체에서 한 번 계산).
    min_cell (층 x 그룹) 셀 하한. min_count가 그룹에 거는 하한과 다른 축이다.
    diag     주면 스텝별 참여 층 수·유지 가중치 질량을 누적한다(§4-3 검수용).
    """
    if strata is None:
        return _pooled_penalty(proba, codes, min_count)
    if min_count > 0:
        # 그룹 하한은 층을 쪼개기 **전에** 건다. 층 안에서 이미 작아진 셀에 그룹용
        # 하한(100)을 그대로 걸면 층 하한과 두 번 걸리는 셈이 된다.
        keep = codes >= 0
        gid, gcnt = torch.unique(codes[keep], return_counts=True)
        small = gid[gcnt < min_count]
        if small.numel():
            codes = torch.where(torch.isin(codes, small),
                                torch.full_like(codes, -1), codes)
    return _stratified_penalty(proba, codes, strata, weights, min_cell, diag)


def train_one(edge_type, data, y, train_idx, val_idx, test_idx, *, seed, k_neighbors,
              hidden_dim, num_layers, dropout, lr, weight_decay, aggr,
              max_epochs, patience, val_size, tag, device, edge_mode=None,
              fair_alpha=0.0, fair_codes=None, fair_beta=0.0, grad_clip=0.0,
              minibatch=False, batch_size=None, num_neighbors=None,
              eval_batch_size=None, fair_min_count=0,
              fair_strata=None, fair_stratum=None, fair_weights=None,
              fair_min_cell=0):
    torch.manual_seed(seed)
    model = GraphSAGE(data.x.shape[1], hidden_dim, num_layers, dropout, aggr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_train = y[train_idx]
    pos = y_train.sum().item()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 조기 종료 스칼라. fair_beta=0이면 예전처럼 val MCC만 본다.
    # fair_beta>0이면 val MCC - beta * (val 선택률 격차)로 고른다 — 손실에 공정성
    # 벌점을 걸어놓고 모델 선택은 MCC만 보면 alpha가 클 때 둘이 서로 싸운다
    # (실제로 alpha>=200에서 곡선이 뒤집혔다). 선택 기준을 목표와 맞추는 것이다.
    val_codes_np = (fair_codes[val_idx].cpu().numpy()
                    if (fair_beta > 0 and fair_codes is not None) else None)

    # 그룹 최소 인원 floor는 **미니배치에서만** 건다. 그 floor가 존재하는 이유가
    # 배치 표집 잡음이므로, 전체 학습 노드를 한 번에 보는 full-batch에서는 걸 이유가
    # 없다 -- 그리고 걸지 않아야 full-batch 경로가 이 변경 전과 비트 단위로 같다.
    min_count = fair_min_count if minibatch else 0

    # 층 참여 진단(계획서 §4-3 검수 · 검토 3-5). 셀 하한 때문에 배치마다 어떤 층이
    # 빠지는지가 개입의 실효 표준화 인구를 정하므로, 고정 가중 w_s를 넘겼다는 것만으로는
    # 충분하지 않다 -- 실제로 몇 개 층이 몇 번 들어왔는지를 세어 학습 끝에 보고한다.
    fair_diag = {} if fair_strata is not None else None

    def _step(out, seed_idx):
        """로짓 + 그 로짓에 대응하는 노드 인덱스 -> 손실. 두 경로가 공유한다."""
        loss = criterion(out, y[seed_idx])
        if fair_alpha > 0 and fair_codes is not None:
            loss = loss + fair_alpha * fairness_penalty(
                torch.sigmoid(out), fair_codes[seed_idx], min_count,
                strata=None if fair_strata is None else fair_strata[seed_idx],
                weights=fair_weights, min_cell=fair_min_cell, diag=fair_diag)
        return loss

    def _backward(loss):
        loss.backward()
        if grad_clip and grad_clip > 0:
            # 높은 fair_alpha에서 벌점 gradient가 BCE를 압도해 스텝이 튀는 것을 막는다
            # (§10-2의 최적화 불안정). fair_alpha=0/grad_clip=0이면 기존 동작 그대로.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # 로더는 **세 개 다 여기서 한 번만** 만든다. NeighborLoader 생성은 edge_index로
    # CSC 인덱스를 새로 짜는 일이라, val 로더를 에폭마다 만들면 그 비용을 매 에폭
    # 다시 치르고 메모리도 계속 붙든다 -- 실제로 3개 주 실행에서 상주 메모리가
    # 1.8GB -> 5.3GB로 단조 증가했다. 전국은 노드가 3.4배라 그대로면 호스트가 죽는다.
    if minibatch:
        full_fanout = [-1] * num_layers
        train_loader = _loader(data, train_idx, num_neighbors, batch_size, True)
        val_loader = _loader(data, val_idx, full_fanout, eval_batch_size, False)
        test_loader = _loader(data, test_idx, full_fanout, eval_batch_size, False)
    else:
        train_loader = val_loader = test_loader = None

    def _train_epoch():
        model.train()
        if not minibatch:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index).squeeze(-1)[train_idx]
            loss = _step(out, train_idx)
            _backward(loss)
            return loss.item()
        total, nb = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # seed 노드는 배치 앞쪽 batch_size개다. n_id는 원본 노드 번호라
            # y·fair_codes(둘 다 전체 길이)를 그대로 인덱싱할 수 있다.
            out = model(batch.x, batch.edge_index).squeeze(-1)[:batch.batch_size]
            loss = _step(out, batch.n_id[:batch.batch_size])
            _backward(loss)
            total, nb = total + loss.item(), nb + 1
        return total / max(nb, 1)

    def _logits(idx, loader):
        """평가용 로짓. 미니배치 경로에서도 **이웃을 전부** 쓴다(_infer 참고)."""
        if minibatch:
            return _infer(model, loader, device)
        model.eval()
        with torch.no_grad():
            return model(data.x, data.edge_index).squeeze(-1)[idx].cpu()

    best_score, best_val_mcc, best_gap = -np.inf, -1.0, np.nan
    best_epoch, best_state, no_improve = 0, None, 0
    t0 = time.time()
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        loss_val = _train_epoch()

        val_proba = torch.sigmoid(_logits(val_idx, val_loader)).numpy()
        val_pred = (val_proba >= 0.5).astype(int)
        val_mcc = matthews_corrcoef(y[val_idx].cpu().numpy().astype(int), val_pred)

        val_gap = _group_gap(val_pred, val_codes_np)
        score = val_mcc if val_codes_np is None else val_mcc - fair_beta * val_gap

        improved = score > best_score
        if improved:
            best_score, best_val_mcc, best_gap, best_epoch = score, val_mcc, val_gap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if improved or epoch % 20 == 0:
            mark = "  *best*" if improved else ""
            extra = "" if val_codes_np is None else f"  val_gap {val_gap:.4f}"
            print(f"[train:{edge_type}/seed{seed}] epoch {epoch:>3d}  loss {loss_val:.4f}  "
                  f"val_mcc {val_mcc:.4f}{extra}{mark}")

        if no_improve >= patience:
            break

    train_seconds = time.time() - t0
    strata_per_step = n_strata_seen = np.nan
    if fair_diag and fair_diag.get("steps"):
        strata_per_step = fair_diag["strata_used"] / fair_diag["steps"]
        n_strata_seen = len(fair_diag.get("w_seen", {}))
        w_kept = fair_diag["w_kept"] / fair_diag["steps"]
        # 콘솔은 이 기계에서 cp949다. em-dash를 쓰면 여기서 UnicodeEncodeError로
        # 죽는다(이 저장소가 --verify_clone 성공 메시지에서 이미 한 번 당했다).
        print(f"[fair:{fair_stratum}] 스텝당 참여 층 {strata_per_step:.1f}개 "
              f"(전체 {n_strata_seen}개가 한 번 이상), 유지 가중치 질량 평균 "
              f"{w_kept:.3f} (재정규화 전). 낮을수록 실효 표준화 인구가 "
              f"대형 층으로 쏠린다.")
    gap_note = "" if val_codes_np is None else f"  best_val_gap={best_gap:.4f}"
    print(f"[stop:{edge_type}/seed{seed}] best_epoch={best_epoch}  best_val_mcc={best_val_mcc:.4f}"
          f"{gap_note}  ({train_seconds:.1f}s, {epoch}epoch)")

    model.load_state_dict(best_state)
    test_proba = torch.sigmoid(_logits(test_idx, test_loader)).numpy()
    y_test = y[test_idx].cpu().numpy().astype(int)

    metric_vals = evaluate(y_test, test_proba)   # 커밋된 평면 모델 덤프와 동일 지표·임계값
    print(f"[eval:{edge_type}/seed{seed}] " + "  ".join(f"{k}={v:.4f}" for k, v in metric_vals.items()))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "graphsage", "tag": tag, "seed": seed,
        # edge_mode=None(셔플-링)이면 results의 from_run이 params에서 빼므로 기존
        # 결과 행의 identity KEY가 그대로 유지된다(config.GNN_EDGE_MODE 주석 참고).
        "edge_type": edge_type, "edge_mode": edge_mode, "k_neighbors": k_neighbors,
        "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
        "lr": lr, "weight_decay": weight_decay, "aggr": aggr,
        "max_epochs": max_epochs, "patience": patience, "val_size": val_size,
        "fair_alpha": fair_alpha, "fair_beta": fair_beta,
        # 층 벌점을 안 쓰면 셋 다 None -> results.from_run이 params에서 빼므로 기존
        # 행의 params JSON이 안 바뀐다(edge_mode·minibatch와 같은 규약).
        "fair_stratum": fair_stratum,
        "fair_min_cell": fair_min_cell if fair_strata is not None else None,
        "fair_strata_per_step": strata_per_step,
        # full-batch면 셋 다 None -> results.from_run이 params에서 빼므로 기존 행의
        # params JSON이 안 바뀐다(edge_mode와 같은 규약).
        "minibatch": True if minibatch else None,
        "batch_size": batch_size if minibatch else None,
        "num_neighbors": list(num_neighbors) if minibatch else None,
        "best_val_gap": best_gap,
        "best_epoch": best_epoch, "train_seconds": round(train_seconds, 1),
        "n_params": sum(p.numel() for p in model.parameters()),
        **metric_vals,
        # 공정성 진단(diagnose_fairness)용 원자료. '_' 접두 키는 clear.results가
        # canonical 지표 목록만 골라 담으므로 results.csv엔 안 들어간다.
        "_test_proba": test_proba,
        # 체크포인트 저장용. '_' 접두라 results.csv에는 안 들어간다.
        "_model": model,
    }


def save_checkpoint(row, path, feature_cols, blind, scope):
    """학습된 모델을 **되살리는 데 필요한 맥락 전부**와 함께 저장한다.

    state_dict만 저장하면 못 쓴다. 셋이 더 필요하다.

    1. **하이퍼파라미터** — 모델을 같은 모양으로 다시 만들어야 로드가 된다.
    2. **특성 열 이름 순서** — 03_features.py의 원-핫 순서는 데이터에 의존하므로,
       다른 기계에서 파이프라인을 다시 돌리면 열 순서가 달라질 수 있다. 그러면
       가중치가 엉뚱한 특성에 붙는데 **오류는 안 나고 지표만 그럴듯하게 나온다**.
       load_checkpoint가 이 목록을 대조해 불일치하면 죽는다.
    3. **calibrated=False** — clear.gnn은 pos_weight로 학습해서 예측이 순위는 맞지만
       확률로는 치우쳐 있다(전국 기대 미제가 관측보다 +47.9%). detect_cold_blocks가
       로짓 이동 델타로 보정한 뒤에야 쓴다. 받아서 확률로 바로 쓰는 사람이 반드시
       생기므로 파일에 적어 두고 로더가 경고한다.
    """
    import torch

    model = row["_model"]
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "hyper": {k: row[k] for k in
                  ("hidden_dim", "num_layers", "dropout", "aggr")},
        "feature_cols": list(feature_cols),
        "in_dim": len(feature_cols),
        "blind": bool(blind),
        "scope": scope,
        "edge_type": row["edge_type"], "edge_mode": row["edge_mode"],
        "k_neighbors": row["k_neighbors"], "seed": row["seed"],
        "tag": row["tag"], "fair_alpha": row.get("fair_alpha"),
        "calibrated": False,
        "n_params": row["n_params"],
        "test_metrics": {k: v for k, v in row.items()
                         if isinstance(v, float) and not k.startswith("_")},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[ckpt] {path.name}  ({path.stat().st_size/1024:.0f} KB, "
          f"파라미터 {row['n_params']:,}개)")
    return path


def load_checkpoint(path, feature_cols=None):
    """체크포인트를 읽어 (model, meta)를 돌려준다.

    feature_cols를 주면 **순서까지 대조하고 다르면 죽는다.** 조용히 통과시키면
    가중치가 다른 특성에 붙은 채로 그럴듯한 지표가 나온다 -- 이 저장소가 --blind
    무음 no-op에서 이미 한 번 당한 종류다.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if feature_cols is not None and list(feature_cols) != ckpt["feature_cols"]:
        a, b = list(feature_cols), ckpt["feature_cols"]
        diff = next((i for i, (x, y_) in enumerate(zip(a, b)) if x != y_), None)
        raise SystemExit(
            f"[에러] 특성 열이 체크포인트와 다르다 "
            f"(현재 {len(a)}개 / 저장 {len(b)}개"
            + (f", 첫 불일치 {diff}번: {a[diff]!r} != {b[diff]!r}" if diff is not None
               else "")
            + "). 가중치가 엉뚱한 특성에 붙으므로 로드를 중단한다. "
              "03_features.py를 같은 스코프·같은 blind 조건으로 다시 만들 것.")
    h = ckpt["hyper"]
    model = GraphSAGE(ckpt["in_dim"], h["hidden_dim"], h["num_layers"],
                      h["dropout"], h["aggr"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if not ckpt.get("calibrated", False):
        print("[ckpt] 주의: 이 모델의 출력은 **보정되지 않은 점수**다. 순위는 맞지만 "
              "확률로 쓰려면 로짓 이동 보정이 필요하다(detect_cold_blocks 참고).")
    return model, ckpt


def dump_test_predictions(rows, test_idx, y, path):
    """seed별 test_proba를 평균해 test 노드별 예측을 CSV로 저장(공정성 진단 07용).

    GNN 사정(seed 평균)만 여기서 처리하고, 덤프 형식 자체는 clear.predictions가
    정의한다 — 평면 모델의 단일 결정적 예측과 완화된 예측이 같은 형식을 써야
    diagnose_fairness가 셋을 나란히 진단할 수 있기 때문.

    rows: train_eval 반환(각 dict에 '_test_proba'). test_idx: 원본 행 위치(np array,
    features.parquet 행에 대응 → load_sensitive()와 join 가능). 모든 seed가 동일 test
    집합이므로 proba를 평균해 seed 노이즈를 줄인 단일 예측을 남긴다.
    """
    from clear import predictions

    probas = np.stack([r["_test_proba"] for r in rows])   # (n_seeds, n_test)
    return predictions.dump(path, test_idx, y, probas.mean(axis=0))


def train_eval(edge_type, k_neighbors, hp, seeds, tag, X, y_t, train_t, val_t, test_t,
               device, family=None, data=None, attribute=None, blind=None,
               edge_mode=None, **extra):
    """한 config를 여러 seed로 학습. edge 그래프는 한 번만 로드해 재사용.
    seed별 결과를 outputs/results.csv에 기록(family를 주면)하고 행 리스트를 반환한다.

    data       이미 만든 그래프를 쓰려면 넘긴다. mitigate_graph가 동종 엣지를 제거한
               그래프를, mitigate_loss가 alpha 격자 내내 재사용할 그래프를 이렇게
               넘긴다. None이면 여기서 edge_type/k_neighbors로 로드한다(train_gnn의 경로).
    family     결과 기록 계열(train / mitigate_graph / mitigate_loss).
               None이면 기록하지 않는다.
    extra      train_one에 그대로 전달(mitigate_loss의 fair_alpha/fair_codes/fair_beta).

    data·extra가 없던 시절 mitigate_graph·mitigate_loss는 이 함수를 못 쓰고 seed
    루프를 각자 재구현했다 — "공용 함수가 정작 자기 호출부를 못 덮는" 상태였다.
    """
    minibatch = bool(hp.get("minibatch"))
    if data is None:
        data = build_data(X, edge_type, k_neighbors, device, edge_mode, minibatch)
        print(f"[graph:{edge_type}/{edge_mode or 'shuffle'}] "
              f"엣지 {data.edge_index.shape[1]:,}개(방향), k={k_neighbors}"
              + ("  (CPU 상주, NeighborLoader)" if minibatch else ""))
    elif minibatch and data.x.is_cuda:
        # 호출부가 미리 만든 그래프를 넘겼는데(mitigate_loss가 그렇게 한다) GPU에
        # 올라와 있으면 미니배치를 쓰는 의미가 없다. seed 루프 앞에서 한 번만 내린다.
        data = data.cpu()
    rows, recorded = [], []
    for seed in seeds:
        row = train_one(edge_type, data, y_t, train_t, val_t, test_t,
                        seed=seed, k_neighbors=k_neighbors, tag=tag, device=device,
                        edge_mode=edge_mode, **hp, **extra)
        if family is not None:
            recorded += results.from_run(row, family, attribute=attribute, blind=blind)
        rows.append(row)
    if recorded:
        results.write(recorded)
    return rows
