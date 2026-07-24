"""GraphSAGE 모델·학습 로직 — 06_train_gnn.py와 ablation_sweep.py의 공통 출처.

이전에는 이 로직이 전부 06_train_gnn.py에 있었고, ablation_sweep.py는 숫자로
시작하는 그 파일을 import할 수 없어(문법 오류) subprocess로 Python을 config마다
다시 띄웠다 — 매번 torch를 재import하고 parquet를 다시 읽었다. 학습 로직을
importable한 여기로 옮기면 ablation이 `from clear.gnn import ...`로 in-process
호출해 데이터를 한 번만 로드한다. 06_train_gnn.py는 얇은 CLI 래퍼가 된다.

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


def build_data(X, edge_type, k, device):
    edges = build_edge_index(edge_type, k)
    x = torch.tensor(X.values.astype(np.float32))
    edge_index = torch.from_numpy(edges).long()
    return Data(x=x, edge_index=edge_index).to(device)


def prepare(X, y, val_size, device):
    """split + device 텐서 준비. main·ablation이 공유(로드·분할 한 번).
    split은 config 기본 random_state로 고정 — 모든 seed·config가 동일 test 집합."""
    split = get_split(y, val_size=val_size)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    train_t = torch.tensor(split.train, dtype=torch.long, device=device)
    val_t = torch.tensor(split.val, dtype=torch.long, device=device)
    test_t = torch.tensor(split.test, dtype=torch.long, device=device)
    return y_t, train_t, val_t, test_t


def fairness_penalty(proba, codes):
    """그룹 평균 예측확률의 (크기가중) 분산 — Demographic Parity의 미분가능 대리항.

    codes: 노드별 그룹 인덱스, -1은 벌점에서 제외(Unknown 등). 그룹이 둘이면
    이 값은 두 그룹 평균 차이의 제곱에 비례하므로, "선택률 격차를 줄여라"를
    그대로 손실에 넣은 것이 된다. 하드 예측(임계값)은 미분이 안 되므로 시그모이드
    확률의 평균을 쓴다 — 07이 재는 selection_rate의 매끄러운 대리값이다.

    벌점은 **학습 노드에서만** 계산한다(호출부가 train 인덱스를 넘긴다). 민감속성이
    학습 시점에만 필요하고 추론 시점엔 불필요하다는 게 이 방식의 요점이다 —
    후처리(08)가 배포 시 인종을 알아야 하는 제약을 피한다.
    """
    keep = codes >= 0
    if not torch.any(keep):
        return proba.sum() * 0.0
    p, c = proba[keep], codes[keep]
    overall = p.mean()
    pen = proba.sum() * 0.0
    for g in torch.unique(c):
        m = c == g
        w = m.sum().to(p.dtype) / c.numel()
        pen = pen + w * (p[m].mean() - overall) ** 2
    return pen


def train_one(edge_type, data, y, train_idx, val_idx, test_idx, *, seed, k_neighbors,
              hidden_dim, num_layers, dropout, lr, weight_decay, aggr,
              max_epochs, patience, val_size, tag, device,
              fair_alpha=0.0, fair_codes=None):
    torch.manual_seed(seed)
    model = GraphSAGE(data.x.shape[1], hidden_dim, num_layers, dropout, aggr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_train = y[train_idx]
    pos = y_train.sum().item()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_mcc, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    t0 = time.time()
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze(-1)
        loss = criterion(out[train_idx], y[train_idx])
        if fair_alpha > 0 and fair_codes is not None:
            loss = loss + fair_alpha * fairness_penalty(
                torch.sigmoid(out[train_idx]), fair_codes[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index).squeeze(-1)
            val_proba = torch.sigmoid(out[val_idx]).cpu().numpy()
            val_pred = (val_proba >= 0.5).astype(int)
            val_mcc = matthews_corrcoef(y[val_idx].cpu().numpy().astype(int), val_pred)

        improved = val_mcc > best_val_mcc
        if improved:
            best_val_mcc, best_epoch = val_mcc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if improved or epoch % 20 == 0:
            mark = "  *best*" if improved else ""
            print(f"[train:{edge_type}/seed{seed}] epoch {epoch:>3d}  loss {loss.item():.4f}  "
                  f"val_mcc {val_mcc:.4f}{mark}")

        if no_improve >= patience:
            break

    train_seconds = time.time() - t0
    print(f"[stop:{edge_type}/seed{seed}] best_epoch={best_epoch}  best_val_mcc={best_val_mcc:.4f}  "
          f"({train_seconds:.1f}s, {epoch}epoch)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze(-1)
        test_proba = torch.sigmoid(out[test_idx]).cpu().numpy()
    y_test = y[test_idx].cpu().numpy().astype(int)

    metric_vals = evaluate(y_test, test_proba)   # 05와 동일 지표·동일 0.5 임계값
    print(f"[eval:{edge_type}/seed{seed}] " + "  ".join(f"{k}={v:.4f}" for k, v in metric_vals.items()))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "graphsage", "tag": tag, "seed": seed,
        "edge_type": edge_type, "k_neighbors": k_neighbors,
        "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
        "lr": lr, "weight_decay": weight_decay, "aggr": aggr,
        "max_epochs": max_epochs, "patience": patience, "val_size": val_size,
        "fair_alpha": fair_alpha,
        "best_epoch": best_epoch, "train_seconds": round(train_seconds, 1),
        "n_params": sum(p.numel() for p in model.parameters()),
        **metric_vals,
        # 공정성 진단(07)용 원자료. '_' 접두 키는 ledger.append의 reindex(columns=
        # LEDGER_COLS)에서 자동으로 떨어져 원장 CSV엔 안 들어간다.
        "_test_proba": test_proba,
    }


def dump_test_predictions(rows, test_idx, y, path):
    """seed별 test_proba를 평균해 test 노드별 예측을 CSV로 저장(공정성 진단 07용).

    GNN 사정(seed 평균)만 여기서 처리하고, 덤프 형식 자체는 clear.predictions가
    정의한다 — 05_train_baseline.py(단일 결정적 예측)와 08(완화된 예측)이 같은
    형식을 써야 07이 셋을 나란히 진단할 수 있기 때문.

    rows: train_eval 반환(각 dict에 '_test_proba'). test_idx: 원본 행 위치(np array,
    features.parquet 행에 대응 → load_sensitive()와 join 가능). 모든 seed가 동일 test
    집합이므로 proba를 평균해 seed 노이즈를 줄인 단일 예측을 남긴다.
    """
    from clear import predictions

    probas = np.stack([r["_test_proba"] for r in rows])   # (n_seeds, n_test)
    return predictions.dump(path, test_idx, y, probas.mean(axis=0))


def train_eval(edge_type, k_neighbors, hp, seeds, tag, X, y_t, train_t, val_t, test_t,
               device, ledger=None):
    """한 config를 여러 seed로 학습. edge 그래프는 한 번만 로드해 재사용.
    각 seed 행을 ledger에 append(넘겨주면)하고 행 리스트를 반환한다."""
    data = build_data(X, edge_type, k_neighbors, device)
    print(f"[graph:{edge_type}] 엣지 {data.edge_index.shape[1]:,}개(방향), k={k_neighbors}")
    rows = []
    for seed in seeds:
        row = train_one(edge_type, data, y_t, train_t, val_t, test_t,
                        seed=seed, k_neighbors=k_neighbors, tag=tag, device=device, **hp)
        if ledger is not None:
            ledger.append(row)
        rows.append(row)
    return rows
