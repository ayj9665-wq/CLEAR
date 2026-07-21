"""
06_train_gnn.py — GraphSAGE 학습 (엣지 후보 3종) + ablation 준비

04_build_graph.py가 만든 geo/temporal/weapon 3개 후보 그래프 각각으로
GraphSAGE를 학습해 flat 베이스라인(05_train_baseline.py, balanced_accuracy
0.6456)과 비교한다. 인자 없이 실행하면 config.py 기본 하이퍼파라미터로 3종을
전부 학습하고, 이후 하이퍼파라미터를 바꿔가며 반복 실행(ablation)할 수 있게
전부 argparse로 오버라이드 가능하다. 결과는 outputs/gnn_metrics.csv에
누적(append)된다 — baseline_metrics.csv와 달리 매번 덮어쓰지 않는다.

흐름:
  features.parquet 로드(X/y) → 베이스라인과 동일한 test 분할 재현
   (train_test_split은 n·stratify·random_state에만 의존하므로 X 내용과
   무관하게 05_train_baseline.py와 같은 test 행 집합이 나옴)
   → train 풀에서 val 추가 분리(조기 종료용)
   → 후보별: edges_{edge_type}.npy 로드 → GraphSAGE 학습(val balanced_accuracy
     기준 조기 종료) → test 평가(베이스라인과 동일 지표·동일 0.5 임계값)
   → outputs/gnn_metrics.csv에 한 줄씩 append
"""
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as C   # torch import 전에 필요 (KMP_DUPLICATE_LIB_OK 등 env 설정)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, roc_auc_score, average_precision_score,
)


def load_xy():
    """05_train_baseline.py:load_xy()와 동일 패턴."""
    df = pd.read_parquet(C.PROCESSED_DIR / "features.parquet")
    sens_cols = [c for c in df.columns if c.startswith("sens__")]
    y = df[C.TARGET_BIN].values
    X = df.drop(columns=[C.TARGET_BIN] + sens_cols)
    return X, y


def make_splits(y, test_size, val_size, random_state):
    """test_idx는 05_train_baseline.py의 train_test_split과 동일 인자라
    같은 행 집합이 나온다(배정은 n·stratify·random_state에만 의존, X 무관)."""
    n = len(y)
    trainval_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_size, stratify=y, random_state=random_state)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_size, stratify=y[trainval_idx], random_state=random_state)
    return train_idx, val_idx, test_idx


def build_data(X, edge_type, device):
    edges = np.load(C.GRAPH_DIR / f"edges_{edge_type}.npy")
    x = torch.tensor(X.values.astype(np.float32))
    edge_index = torch.from_numpy(edges).long()
    return Data(x=x, edge_index=edge_index).to(device)


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


def train_one(edge_type, data, y, train_idx, val_idx, test_idx, *,
              hidden_dim, num_layers, dropout, lr, weight_decay, aggr,
              max_epochs, patience, val_size, tag, device):
    torch.manual_seed(C.RANDOM_STATE)
    model = GraphSAGE(data.x.shape[1], hidden_dim, num_layers, dropout, aggr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_train = y[train_idx]
    pos = y_train.sum().item()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_ba, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    t0 = time.time()
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze(-1)
        loss = criterion(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index).squeeze(-1)
            val_proba = torch.sigmoid(out[val_idx]).cpu().numpy()
            val_pred = (val_proba >= 0.5).astype(int)
            val_ba = balanced_accuracy_score(y[val_idx].cpu().numpy().astype(int), val_pred)

        improved = val_ba > best_val_ba
        if improved:
            best_val_ba, best_epoch = val_ba, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if improved or epoch % 20 == 0:
            mark = "  *best*" if improved else ""
            print(f"[train:{edge_type}] epoch {epoch:>3d}  loss {loss.item():.4f}  "
                  f"val_ba {val_ba:.4f}{mark}")

        if no_improve >= patience:
            break

    train_seconds = time.time() - t0
    print(f"[stop:{edge_type}] best_epoch={best_epoch}  best_val_ba={best_val_ba:.4f}  "
          f"({train_seconds:.1f}s, {epoch}epoch)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze(-1)
        test_proba = torch.sigmoid(out[test_idx]).cpu().numpy()
    test_pred = (test_proba >= 0.5).astype(int)
    y_test = y[test_idx].cpu().numpy().astype(int)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
    }
    print(f"[eval:{edge_type}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "edge_type": edge_type, "tag": tag,
        "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout,
        "lr": lr, "weight_decay": weight_decay, "aggr": aggr,
        "max_epochs": max_epochs, "patience": patience, "val_size": val_size,
        "best_epoch": best_epoch, "train_seconds": round(train_seconds, 1),
        "n_params": sum(p.numel() for p in model.parameters()),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_type", choices=["geo", "temporal", "weapon", "all"], default="all")
    parser.add_argument("--hidden_dim", type=int, default=C.GNN_HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=C.GNN_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=C.GNN_DROPOUT)
    parser.add_argument("--lr", type=float, default=C.GNN_LR)
    parser.add_argument("--weight_decay", type=float, default=C.GNN_WEIGHT_DECAY)
    parser.add_argument("--aggr", default=C.GNN_AGGR)
    parser.add_argument("--max_epochs", type=int, default=C.GNN_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=C.GNN_PATIENCE)
    parser.add_argument("--val_size", type=float, default=C.GNN_VAL_SIZE)
    parser.add_argument("--tag", default="", help="ablation 결과 CSV에서 구분할 자유 라벨")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_xy()
    print(f"[load] features {X.shape}, 검거율 {y.mean():.1%}, device={device}")

    train_idx, val_idx, test_idx = make_splits(y, C.TEST_SIZE, args.val_size, C.RANDOM_STATE)
    print(f"[split] train {len(train_idx):,} / val {len(val_idx):,} / test {len(test_idx):,}")

    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    edge_types = ["geo", "temporal", "weapon"] if args.edge_type == "all" else [args.edge_type]
    out_csv = C.OUTPUT_DIR / "gnn_metrics.csv"

    for edge_type in edge_types:
        data = build_data(X, edge_type, device)
        print(f"[graph:{edge_type}] 엣지 {data.edge_index.shape[1]:,}개(방향)")

        row = train_one(
            edge_type, data, y_t, train_idx_t, val_idx_t, test_idx_t,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
            lr=args.lr, weight_decay=args.weight_decay, aggr=args.aggr,
            max_epochs=args.max_epochs, patience=args.patience,
            val_size=args.val_size, tag=args.tag, device=device,
        )
        pd.DataFrame([row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False, encoding="utf-8-sig")
        print(f"[save] {out_csv} (append)")


if __name__ == "__main__":
    main()
