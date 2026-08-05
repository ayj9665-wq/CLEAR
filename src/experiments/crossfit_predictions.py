"""
experiments/crossfit_predictions.py -- 5-fold cross-fitting으로 전체 행의 out-of-fold p_hat

detect_cold_blocks는 지금 **test 집합 위에서만** 돈다. 편향은 없지만(test의 p_hat은
진짜 out-of-sample) 커버리지가 절반이다 -- 전국 카운티 3,042개 중 853개(28%)만
판정되고 나머지는 지도에서 빗금이다. 더 중요한 것은 그 853개가 **크기로 선별된
부분집합**이라는 점이다:

    n >= 20   853 블록   z ~ black_share = +0.381
    n >= 50   418 블록                     +0.496
    n >= 100  248 블록                     +0.546

큰 카운티만 남길수록 상관이 강해진다. 그러면 지도의 핵심 서사인 +0.381이
"미국의 성질"인지 "대도시 카운티의 성질"인지 지금 데이터로는 못 가른다.
크로스피팅이 들여보내는 것이 정확히 그 작은 카운티들이므로, 이 트랙은
**정확성 수정이 아니라 일반화 범위를 확정하는 작업**이다.

## 무엇을 쪼개는가 -- 행이 아니라 노드

평면 모델의 크로스피팅은 행을 5등분하면 끝이지만 GNN은 그래프 위에서 학습하므로
폴드를 나눈다고 그래프가 나뉘지 않는다. 두 선택지 중 **transductive**를 쓴다:
그래프는 전체를 유지하고 **손실만** 해당 폴드의 learn 노드에서 계산한다.

이유는 현재의 train/val/test 분할이 **이미** transductive이기 때문이다 --
clear.gnn은 전체 그래프로 forward하고 train_idx에서만 손실을 건다. 크로스피팅은
그 train_idx를 5번 바꾸는 것 이상이어서는 안 된다. inductive(폴드마다 held-out
노드를 그래프에서 제거)로 바꾸면 커버리지 효과와 레짐 변경 효과가 섞여 원인을
못 가른다 -- mitigate_graph에서 이미 치른 수업료다.

그래서 이 스크립트는 clear.gnn을 **손대지 않는다**. train_one이 이미
train_idx/val_idx/test_idx를 인자로 받으므로 폴드를 만들어 넘기는 얇은 루프다.
clear/에도 아무것도 안 넣는다 -- 호출부가 하나뿐이라 "2곳 이상" 규칙에 걸린다.

**정확한 표현은 "out-of-fold 라벨 예측"이다.** transductive이므로 held-out 노드의
*특성*은 메시지 전달을 통해 학습에 참여한다(라벨은 아니다). 준지도 그래프 학습의
표준 설정이고 라벨 누수가 아니지만, "완전한 out-of-sample"이라고 쓰지 않는다.

## 개입은 고정한다

alpha=100 / blind / 미니배치 -- 현재 전국 산출물과 같은 조건이다(확장설계서 §10-3의
게이트 통과값). 커버리지 효과를 재는 실험에서 완화 세기를 같이 움직이면 원인을
못 가른다.

그 "같은 조건"에는 **벌점 대상 그룹집합**도 포함되는데, 여기가 조용히 갈리는
지점이다. mitigate_loss는 대상 그룹을 **canonical test 분할에서** 센다(그쪽의
run_point가 test 덤프로 격차를 재기 때문). 전국 인종 분포에서 Asian/PI는
test 2,940명 / 전체 9,890명이라 min_n=5000 기준으로 **한쪽은 제외, 한쪽은 포함**
이다. 크로스피팅은 전체 행을 평가하므로 순진하게 다시 세면 벌점이 2그룹이 아니라
3그룹을 누르게 되고, 그러면 "두 그룹이면 이 벌점은 선택률 격차의 제곱"이라는
근거가 깨지면서 alpha=100이 **다른 개입**이 된다. 그래서 대상 그룹은 a100 실행과
동일하게 test 분할 기준으로 고정한다.

## 기존 test 결과를 대체하지 않는다

크로스피팅 폴드는 get_split의 test와 **다른 분할**이다. 따라서 결과는 별도 축으로
남긴다 -- 덤프는 `..._cv5` 접미사, 블록 표는 detect_cold_blocks --out으로 파일 분리.
get_split 자체는 건드리지 않는다(커밋된 평면 모델 덤프와 60곳의 인용이 걸린
기준선이다).

출력: outputs/predictions/{scope}/graphsage_fairloss_a{alpha}_mb_cv{k}.csv
      outputs/results.csv (family=crossfit, 폴드별 정확도)
"""
import argparse

import numpy as np

import config as C   # torch import 전에 필요 (KMP_DUPLICATE_LIB_OK 등 env 설정)
from clear.data import load_xy, load_sensitive, get_split
from clear import gnn, predictions, results

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


def fold_indices(y, n_folds, val_size, random_state):
    """(learn_train, learn_val, held_out) 폴드 목록.

    층화는 solved로 한다 -- get_split과 같은 기준이라 폴드별 클래스 비율이 전체와
    같다. learn 안에서 val을 떼는 비율도 GNN_VAL_SIZE 그대로여서, 각 폴드의 학습
    레짐이 기존 단일 분할과 같은 모양이 된다(train 68% / val 12% / held-out 20%,
    기존은 59.5/10.5/30).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    out = []
    for learn, held in skf.split(np.zeros(len(y)), y):
        tr, va = train_test_split(learn, test_size=val_size, stratify=y[learn],
                                  random_state=random_state)
        out.append((tr, va, held))
    return out


def penalty_codes(sens, attr_col, y, min_n, device):
    """벌점 대상 그룹 코드. **canonical test 분할에서 센다** -- 위 docstring 참고.

    mitigate_loss와 같은 규칙(Unknown 제외, count >= min_n)이라 alpha=100 실행과
    같은 그룹집합이 나온다. 여기서 전체 행으로 세면 전국에서 Asian/PI가 들어와
    개입 자체가 달라진다.
    """
    test_idx = get_split(y).test
    counts = sens.iloc[test_idx][attr_col].value_counts()
    targets = sorted(g for g in counts.index
                     if g != C.FAIRNESS_UNKNOWN_LABEL and counts[g] >= min_n)
    idx_of = {g: i for i, g in enumerate(targets)}
    codes = sens[attr_col].map(lambda g: idx_of.get(g, -1)).values.astype(np.int64)
    return torch.tensor(codes, device=device), targets, codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=100.0,
                    help="손실 벌점 세기. 기본값은 전국 게이트 통과값(§10-3). "
                         "여기서 바꾸면 커버리지 효과와 완화 세기가 섞인다.")
    ap.add_argument("--attr", default="Victim Race")
    ap.add_argument("--min_n", type=int, default=5000)
    ap.add_argument("--edge_type", default=C.GNN_DEFAULT_EDGE_TYPE)
    ap.add_argument("--k_neighbors", type=int, default=C.K_NEIGHBORS)
    ap.add_argument("--seed", type=int, default=42,
                    help="폴드당 torch seed **1개**. 크로스피팅이 재는 것은 시드 "
                         "분산이 아니라 커버리지이고, 시드 3개면 비용이 3배다. "
                         "폴드 간 MCC 이탈 검사로 폴드 하나가 튀는 것은 잡는다.")
    ap.add_argument("--sighted", action="store_true",
                    help="민감속성 열을 X에 남긴다(기본은 blind = 전국 산출물 조건)")
    ap.add_argument("--minibatch", action="store_true", default=C.GNN_MINIBATCH,
                    help="NeighborLoader 미니배치(전국 규모에 필수)")
    ap.add_argument("--fair_stratum", default="none",
                    help="층 표준화 벌점의 층(none이면 pooled 벌점). **산출물 모델과 "
                         "같은 값을 줘야 한다** -- cv5 덤프는 test 분할 모델과 같은 "
                         "개입이어야 cold_blocks·rho·지도가 한 모델을 가리킨다.")
    ap.add_argument("--fair_min_cell", type=int, default=C.GNN_FAIR_MIN_CELL,
                    help="(층 x 그룹) 셀 하한. --fair_stratum이 none이면 무시된다.")
    args = ap.parse_args()

    blind = not args.sighted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_xy(blind=blind)
    sens = load_sensitive()
    attr_col = f"sens__{args.attr}"
    fair_codes, targets, codes_np = penalty_codes(
        sens, attr_col, y, args.min_n, device)
    print(f"[load] X {X.shape} (blind={blind}), 벌점 대상 그룹 {targets} "
          f"/ 제외 {int((codes_np < 0).sum()):,}행, device={device}")

    hp = {**gnn.default_hp(), "minibatch": args.minibatch}
    data = gnn.build_data(X, args.edge_type, args.k_neighbors, device,
                          minibatch=args.minibatch)
    print(f"[graph:{args.edge_type}] 엣지 {data.edge_index.shape[1]:,}개(방향)"
          + ("  (CPU 상주, NeighborLoader)" if args.minibatch else ""))

    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    folds = fold_indices(y, args.folds, hp["val_size"], C.RANDOM_STATE)

    # 층가중은 canonical train 분할에서 고정한다 -- 폴드마다 다시 계산하면 폴드별로
    # 개입이 달라져 out-of-fold 예측이 한 모델의 것이 아니게 된다(gnn.penalty_strata).
    fair_strata, fair_weights, _ = gnn.penalty_strata(
        args.fair_stratum, codes_np, get_split(y).train, device,
        min_cell=args.fair_min_cell,
        batch_size=hp["batch_size"] if args.minibatch else None)

    tag = (f"fairloss_a{args.alpha:g}"
           + gnn.strat_tag(args.fair_stratum, args.fair_min_cell)
           + ("_mb" if args.minibatch else "") + f"_cv{args.folds}")

    # 폴드 커버리지 검수를 **학습 전에** 한다 -- 80분을 태우고 나서 폴드가 겹쳤다는
    # 것을 알면 그 80분이 통째로 낭비다.
    covered = np.concatenate([h for _, _, h in folds])
    if len(covered) != len(y) or len(np.unique(covered)) != len(y):
        raise SystemExit(f"[에러] 폴드 커버리지 불량: held-out 합집합 "
                         f"{len(np.unique(covered)):,} / 중복 {len(covered) - len(np.unique(covered)):,} "
                         f"(전체 {len(y):,})")
    print(f"[folds] {args.folds}개 폴드, held-out 합집합 {len(y):,}행 · 중복 0 "
          f"(learn {len(folds[0][0]):,} + val {len(folds[0][1]):,} / "
          f"held {len(folds[0][2]):,})")

    proba = np.full(len(y), np.nan)
    recorded, mccs = [], []
    for i, (tr, va, held) in enumerate(folds):
        print(f"\n=== fold {i + 1}/{args.folds} ===")
        row = gnn.train_one(
            args.edge_type, data, y_t,
            torch.tensor(tr, dtype=torch.long, device=device),
            torch.tensor(va, dtype=torch.long, device=device),
            torch.tensor(held, dtype=torch.long, device=device),
            seed=args.seed, k_neighbors=args.k_neighbors, tag=tag, device=device,
            fair_alpha=args.alpha, fair_codes=fair_codes,
            fair_strata=fair_strata, fair_stratum=(
                None if fair_strata is None else args.fair_stratum),
            fair_weights=fair_weights, fair_min_cell=args.fair_min_cell, **hp)
        # _infer/_logits 모두 input_nodes 순서를 보존하므로 held 순서 그대로 꽂힌다.
        proba[held] = row["_test_proba"]
        mccs.append(row["mcc"])

        # 폴드는 params에 넣는다 -- identity의 일부라 폴드 하나를 재실행하면 그
        # 폴드 행만 교체된다(다른 폴드 행은 안 건드린다).
        metrics = {m: float(row[m]) for m in results.ACCURACY}
        metrics.update({m: float(row[m]) for m in ["best_epoch", "train_seconds"]})
        recorded += results.rows(
            "crossfit", metrics, model="graphsage", tag=tag, seed=args.seed,
            attribute=args.attr, blind=blind,
            # None 키는 **빼고** 넘긴다. results.rows는 from_run과 달리 None을
            # 걸러주지 않으므로(그쪽만 필터가 있다) 그대로 두면 params JSON에
            # "fair_stratum": null이 실려, 층 벌점을 안 쓰는 기존 cv5 행과 KEY가
            # 달라지고 재실행이 교체가 아니라 중복이 된다.
            params={k: v for k, v in {
                "fold": i, "n_folds": args.folds, "split": f"cv{args.folds}",
                "alpha": args.alpha, "edge_type": args.edge_type,
                "k_neighbors": args.k_neighbors,
                "minibatch": True if args.minibatch else None,
                "fair_stratum": None if fair_strata is None else args.fair_stratum,
                "fair_min_cell": None if fair_strata is None else args.fair_min_cell,
            }.items() if v is not None})
    results.write(recorded)

    if np.isnan(proba).any():
        raise SystemExit(f"[에러] out-of-fold 예측 결측 {int(np.isnan(proba).sum()):,}행")

    # --- 검수 -----------------------------------------------------------------
    # 폴드 균질성: 폴드 하나만 수렴에 실패하면 그 폴드의 블록만 조용히 틀린다.
    mccs = np.array(mccs)
    sd = mccs.std(ddof=1)
    dev = np.abs(mccs - mccs.mean()).max() / max(sd, 1e-12)
    print(f"\n[검수] 폴드별 MCC {np.round(mccs, 4).tolist()}  "
          f"mean {mccs.mean():.4f} ± {sd:.4f}  최대이탈 {dev:.2f}σ"
          + ("  [경고] 2σ 초과" if dev > 2 else ""))

    # 전역 보정: **여기서 실제 검거율과 맞는지 보면 안 된다.** clear.gnn의 손실은
    # BCEWithLogitsLoss(pos_weight=neg/pos)라 p_hat은 순위만 옳고 보정된 확률이
    # 아니다 -- test 분할 a100 모델도 검거율을 70.2% 대신 53.3%로 낸다(-16.9%p).
    # 그러니 기준은 "실제와 일치"가 아니라 **"기존 test 분할 모델과 같은 정도로
    # 어긋나는가"**다. 크게 다르면 폴드 학습이 다른 지점에 수렴했다는 뜻이다.
    # (실제 보정은 detect_cold_blocks의 로짓 시프트 delta가 한다.)
    print(f"[검수] out-of-fold 예측 검거율 {proba.mean():.4f} "
          f"(실제 {y.mean():.4f}, test분할 a100 모델 0.5328) -- "
          f"pos_weight 재가중 때문에 실제와 어긋나는 것이 정상이고, "
          f"비교 대상은 0.5328이다")

    all_idx = np.arange(len(y))
    path = predictions.path_for("graphsage", tag)
    predictions.dump(path, all_idx, y, proba)
    print(f"\n[save] {path}  ({len(y):,}행 = 전체 표본의 out-of-fold 라벨 예측)")
    print("[다음] python -m experiments.detect_cold_blocks "
          f"--model graphsage_{tag} --min_n 20 50 100 --out cold_blocks_cv{args.folds}.csv")


if __name__ == "__main__":
    main()
