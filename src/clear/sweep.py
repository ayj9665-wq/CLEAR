"""완화 스윕 하네스 — experiments/mitigate_graph.py와 experiments/mitigate_loss.py의 공통 골격.

두 스크립트는 같은 실험 루프다: **노브를 하나 훑으면서, 각 점마다 GNN을 seed
반복 학습하고, test 예측을 덤프하고, diagnose_fairness과 같은 정의로 격차를 다시 재서,
정확도-공정성 한 행을 남긴다.** 다른 것은 개입 방식뿐이다 — mitigate_graph는 그래프를
바꾸고(`data` 인자), 10은 손실을 바꾼다(`fair_*` 인자).

그래서 루프는 여기 한 벌만 두고 개입만 호출부에 남긴다. 이전에는 argparse 6개,
셋업 10줄, seed 루프, 재진단 6줄, 지표 집계 7개, 격차 열 6개가 두 파일에 그대로
복붙돼 있었다 — diagnose_fairness의 격차 정의나 지표 집합이 바뀌면 두 곳을 같이 고쳐야 했고,
한쪽만 고치면 두 완화기법의 곡선이 조용히 다른 축에 놓인다(비교 불가).

네 번째 완화기법을 붙일 때 필요한 것은 tag와 개입 인자뿐이다.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import config as C   # torch import 전에 필요 (KMP_DUPLICATE_LIB_OK 등 env 설정)
from clear.data import load_xy, load_sensitive
from clear import fairness as F, gnn, predictions, results

import torch

# 트레이드오프 표에 싣는 정확도 지표 — clear.metrics.evaluate가 내는 7종과 같다.
ACC_METRICS = ["auc", "mcc", "f1", "sensitivity", "specificity",
               "balanced_accuracy", "precision"]


def add_common_args(ap):
    """mitigate_graph·mitigate_loss이 공유하는 인자. 원래 두 파일에 같은 기본값으로 중복돼 있었다."""
    ap.add_argument("--attr", default="Victim Race",
                    help="완화·격차 측정 대상 민감속성")
    ap.add_argument("--edge_type", default=C.GNN_DEFAULT_EDGE_TYPE)
    ap.add_argument("--k_neighbors", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--seeds", type=gnn.parse_seeds, default=C.GNN_SEEDS,
                    help="쉼표구분 torch seed. split은 고정(=train_baseline·06과 동일 test 집합).")
    ap.add_argument("--min_n", type=int, default=5000,
                    help="완화·격차 대상 그룹의 최소 표본수(기본 5000 = 인종은 White/Black)")
    ap.add_argument("--sighted", action="store_true",
                    help="민감속성 열을 X에 남긴 채 실행(기본은 blind). "
                         "인종이 X에 있으면 모델이 직접 읽으므로 완화 효과가 가려진다(보고서 §5-1).")
    return ap


@dataclass
class Setup:
    """스윕 한 번의 고정 재료 — 노브를 어떻게 돌리든 안 바뀌는 것들."""
    args: Any
    X: Any
    y: Any
    sens: Any
    attr: str
    attr_col: str
    blind: bool
    seeds: list
    min_n: int
    device: Any
    y_t: Any
    train_t: Any
    val_t: Any
    test_t: Any
    test_idx: Any
    hp: dict = field(default_factory=dict)


def setup(args):
    """인자 → 데이터 로드 · 분할 · 하이퍼파라미터. mitigate_graph·mitigate_loss이 동일하게 하던 일.

    하이퍼파라미터는 config 기본값 고정이다(06과 달리 CLI 오버라이드를 두지
    않는다) — 스윕에서 재는 것은 완화 세기지 하이퍼파라미터가 아니므로, 여기서
    둘을 같이 움직이면 곡선의 원인을 못 가른다.
    """
    blind = not args.sighted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_xy(blind=blind)
    sens = load_sensitive()
    y_t, train_t, val_t, test_t = gnn.prepare(X, y, C.GNN_VAL_SIZE, device)

    # grad_clip은 10만 노출한다. train_one의 기본값과 같으므로 09에서도 무해하다.
    hp = {**gnn.default_hp(), "grad_clip": getattr(args, "grad_clip", 0.0)}

    return Setup(args=args, X=X, y=y, sens=sens, attr=args.attr,
                 attr_col=f"sens__{args.attr}", blind=blind, seeds=args.seeds,
                 min_n=args.min_n, device=device, y_t=y_t, train_t=train_t,
                 val_t=val_t, test_t=test_t, test_idx=test_t.cpu().numpy(), hp=hp)


def run_point(su, tag, data, family, params, notes=None, **train_kwargs):
    """스윕 한 점: seed 반복 학습 → 예측 덤프 → 격차 재진단 → 결과 기록.

    data          이 점에서 쓸 그래프(mitigate_graph는 동종 엣지를 뺀 것,
                  mitigate_loss는 원본 그대로).
    family        결과 계열(mitigate_graph / mitigate_loss).
    params        이 점에서 정한 노브 dict(mode·p 또는 alpha·beta·grad_clip).
    notes         실행이 만들어낸 파생값(그래프 통계 등). identity 아님.
    train_kwargs  clear.gnn.train_one에 그대로 전달(fair_alpha/fair_codes/fair_beta).

    outputs/results.csv에 두 종류를 남긴다 — seed별 원자료 행(seed=N)과, seed를
    평균해 격차·CI까지 붙인 요약 행(seed 비움). 예전에는 전자가 metrics.csv로,
    후자가 실험별 *_tradeoff.csv로 갈라져 이름 규약까지 달랐다.

    반환 (core, seed_rows) — core는 콘솔 출력·호출부 판단용 canonical 지표 dict.
    """
    seed_rows = gnn.train_eval(su.args.edge_type, su.args.k_neighbors, su.hp,
                               su.seeds, tag, su.X, su.y_t, su.train_t, su.val_t,
                               su.test_t, su.device, family=family, data=data,
                               attribute=su.attr, blind=su.blind, **train_kwargs)

    path = predictions.path_for("graphsage", tag)
    _, pred_df = predictions.dump(path, su.test_idx, su.y,
                                  np.stack([r["_test_proba"] for r in seed_rows]).mean(0))

    # diagnose_fairness와 같은 정의(clear.fairness)로 계산해야 완화 전후가 같은 축에 놓인다.
    # 방금 쓴 CSV를 다시 읽지 않고 dump가 돌려준 프레임을 그대로 쓴다(수 MB 재파싱 회피).
    cnt, _ = F.joint_counts({"_": pred_df}, su.attr_col)
    groups = F.select_groups(cnt, su.min_n)
    pt, st = F.gap_samples(cnt, F.bootstrap_joint(cnt), groups)
    group_set = f"named_n>={su.min_n}"
    gap = F.gap_row(st, pt, groups, group_set)

    core = {k: float(np.mean([r[k] for r in seed_rows])) for k in ACC_METRICS}
    for m in ["base_rate_gap", "selection_rate_gap", "selection_rate_amplification",
              "tpr_gap"]:
        core[m] = gap[m]
    if any(pd.notna(r.get("best_val_gap")) for r in seed_rows):
        core["best_val_gap"] = float(np.mean([r["best_val_gap"] for r in seed_rows]))

    ci = {m: (gap[f"{m}_lo"], gap[f"{m}_hi"]) for m in
          ["base_rate_gap", "selection_rate_gap", "selection_rate_amplification"]
          if f"{m}_lo" in gap}
    std = ({"mcc": float(np.std([r["mcc"] for r in seed_rows], ddof=1))}
           if len(su.seeds) > 1 else {})

    results.write(results.rows(
        family, core, model="graphsage", tag=tag, seed=None, attribute=su.attr,
        blind=su.blind, group_set=group_set, params=params, notes=notes,
        ci=ci, std=std))

    print(f"[{tag}] MCC {core['mcc']:.4f}  격차 {core['selection_rate_gap']:.4f}  "
          f"증폭비 {core['selection_rate_amplification']:.2f} "
          f"[{gap['selection_rate_amplification_lo']:.2f}, "
          f"{gap['selection_rate_amplification_hi']:.2f}]")
    return core, seed_rows
