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


def fairness_penalty(proba, codes, min_count=0):
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
    """
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


def train_one(edge_type, data, y, train_idx, val_idx, test_idx, *, seed, k_neighbors,
              hidden_dim, num_layers, dropout, lr, weight_decay, aggr,
              max_epochs, patience, val_size, tag, device, edge_mode=None,
              fair_alpha=0.0, fair_codes=None, fair_beta=0.0, grad_clip=0.0,
              minibatch=False, batch_size=None, num_neighbors=None,
              eval_batch_size=None, fair_min_count=0):
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

    def _step(out, seed_idx):
        """로짓 + 그 로짓에 대응하는 노드 인덱스 -> 손실. 두 경로가 공유한다."""
        loss = criterion(out, y[seed_idx])
        if fair_alpha > 0 and fair_codes is not None:
            loss = loss + fair_alpha * fairness_penalty(
                torch.sigmoid(out), fair_codes[seed_idx], min_count)
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
    }


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
