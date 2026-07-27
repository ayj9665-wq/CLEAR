"""실험 결과의 단일 스키마 — outputs/results.csv (long format).

## 왜 통합했나

이전에는 "설정 하나를 돌렸더니 지표가 이렇게 나왔다"는 **같은 내용**이 네 파일에
서로 다른 스키마로 쌓였다:

    metrics.csv              mcc      / 하이퍼파라미터가 개별 열
    fairgraph_tradeoff.csv   acc_mcc  / mode·p가 개별 열 / dp_gap
    fairloss_tradeoff.csv    acc_mcc  / alpha·beta·grad_clip이 개별 열 / dp_gap
    mitigation_tradeoff.csv  acc_mcc  / criterion·lambda가 개별 열 / dp_gap

같은 양에 이름이 둘씩 있었다(`mcc` vs `acc_mcc`, `selection_rate_gap` vs
`dp_gap`). 그리고 "설정 열"이 실험마다 달라서 wide 스키마를 유지하려면 매번
땜질이 필요했다 — 원장의 고정 스키마 reindex, fairloss의 merge-on-key,
Balanced Accuracy/Precision을 나중에 추가하며 과거 행을 대수로 역산한 backfill이
전부 그 세금이었다.

long format이면 지표를 추가해도 **행이 늘 뿐 기존 행은 그대로**다. 설정은
params(JSON)에 들어가므로 실험마다 다른 노브를 가져도 스키마가 안 바뀐다.

## 스키마

    family      train | ablation | mitigate_threshold | mitigate_graph | mitigate_loss
    timestamp   ISO8601
    model       logreg | xgboost | graphsage
    tag         자유 라벨(blind, fairloss_a50_b1, ...)
    seed        torch seed. 시드 반복이 없는 결과(baseline·완화 요약)는 비어 있음
    attribute   민감속성(Victim Race/Sex). 정확도만 있는 행은 비어 있음
    blind       민감속성 열을 X에서 뺐는지
    group_set   격차를 잰 그룹집합(named_n>=5000 등)
    params      이 실행에서 **정한** 노브·하이퍼파라미터(JSON). identity의 일부
    notes       실행이 **만들어낸** 파생값(적합된 임계값·그래프 통계 등, JSON).
                identity 아님 — 재실행이 교체가 되도록
    metric      아래 canonical 이름 하나
    value/lo/hi 점추정과 부트스트랩 CI(없으면 lo/hi 비움)
    std         시드 반복 표준편차(없으면 비움)

지표 이름은 **한 벌만** 쓴다(METRICS). `acc_` 접두어와 `dp_` 별칭은 없앴다 —
격차는 무엇을 균등화하려 했든 `selection_rate_gap`이다.
"""
import json
from datetime import datetime

import pandas as pd

import config as C

# ---- canonical 지표 이름 ------------------------------------------------------
ACCURACY = ["auc", "mcc", "f1", "sensitivity", "specificity",
            "balanced_accuracy", "precision"]
FAIRNESS = ["base_rate_gap", "selection_rate_gap", "selection_rate_amplification",
            "tpr_gap", "tpr_amplification", "fpr_gap", "fpr_amplification"]
RUNINFO = ["best_epoch", "train_seconds", "n_params", "best_val_gap", "n_eval"]
METRICS = ACCURACY + FAIRNESS + RUNINFO

# 옛 파일들이 쓰던 별칭 -> canonical. 마이그레이션과, 혹시 남은 호출부를 위해.
ALIASES = {
    **{f"acc_{m}": m for m in ACCURACY},
    "dp_gap": "selection_rate_gap",
    "dp_amplification": "selection_rate_amplification",
    "dp_amplification_lo": "selection_rate_amplification_lo",
    "dp_amplification_hi": "selection_rate_amplification_hi",
    "acc_mcc_std": "mcc_std",
}

COLUMNS = ["family", "timestamp", "model", "tag", "seed", "attribute", "blind",
           "group_set", "params", "notes", "metric", "value", "lo", "hi", "std"]

# 한 결과를 유일하게 식별하는 키. 같은 키가 다시 들어오면 **교체**한다
# (덮어쓰면 다른 곡선이 지워지고, 그냥 append하면 재실행분이 중복된다 —
#  fairloss가 merge-on-key로 손수 하던 일을 여기서 일반화했다).
#
# notes는 일부러 뺐다. params에는 **정한 노브**만(alpha, lambda, mode, p...),
# notes에는 **실행이 만들어낸 값**을(적합된 임계값, 그래프 통계...) 담는다.
# 둘을 같이 키에 넣었더니 같은 설정을 재실행해도 파생값 표기가 조금만 달라지면
# 교체가 아니라 중복이 됐다 — 실제로 통합 직후 재실행 한 번에 78행이 겹쳤다.
KEY = ["family", "model", "tag", "seed", "attribute", "blind", "group_set",
       "params", "metric"]


def _plain(v):
    """numpy 스칼라 -> 파이썬 스칼라.

    안 하면 json.dumps가 default=str로 떨어뜨려 0.5가 "0.5"(문자열)로 직렬화되고,
    같은 노브인데 CSV에서 되읽은 값(실수 0.5)과 표기가 갈린다 -> KEY 불일치 ->
    재실행이 교체가 아니라 중복이 된다.
    """
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def _dump(d):
    return json.dumps({k: _plain(v) for k, v in (d or {}).items()},
                      sort_keys=True, ensure_ascii=False, default=str)


# 수치 열. dtype을 명시해 두지 않으면 값이 전부 비어 있는 열(대부분의 행에서
# lo/hi/std가 그렇다) 때문에 concat이 dtype을 추론하며 FutureWarning을 낸다.
NUMERIC = ["value", "lo", "hi", "std"]


def _numeric(df):
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _key_frame(df):
    """KEY 열을 비교 가능한 문자열로 정규화.

    None(메모리)과 NaN(CSV 왕복 후)은 같은 '값 없음'인데 astype(str)에서
    'None'/'nan'으로 갈린다. fillna("")로 먼저 합쳐야 교체가 동작한다.
    """
    return df[KEY].astype(object).where(df[KEY].notna(), "").astype(str)


def results_path():
    """호출 시점에 계산 — config.OUTPUT_DIR을 바꾸면 결과도 함께 격리된다."""
    return C.OUTPUT_DIR / "results.csv"


def rows(family, metrics, *, model=None, tag=None, seed=None, attribute=None,
         blind=None, group_set=None, params=None, notes=None, ci=None, std=None,
         timestamp=None):
    """(맥락 + {지표: 값}) -> long 행 리스트.

    metrics  {canonical 이름: 값}. 별칭(acc_mcc 등)을 줘도 canonical로 바꿔 넣는다.
    ci       {지표: (lo, hi)}
    std      {지표: 표준편차}
    params   이 실행에서 **정한** 노브 dict(identity의 일부).
    notes    실행이 **만들어낸** 파생값 dict(적합된 임계값, 그래프 통계 등).
             identity에 들어가지 않으므로 재실행 시 교체를 방해하지 않는다.
    """
    ci, std = ci or {}, std or {}
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    params_s, notes_s = _dump(params), _dump(notes)
    out = []
    for name, value in metrics.items():
        name = ALIASES.get(name, name)
        lo, hi = ci.get(name, (None, None))
        out.append({
            "family": family, "timestamp": ts, "model": model, "tag": tag,
            "seed": seed, "attribute": attribute, "blind": blind,
            "group_set": group_set, "params": params_s, "notes": notes_s,
            "metric": name,
            "value": value, "lo": lo, "hi": hi, "std": std.get(name),
        })
    return out


def write(new_rows, path=None):
    """행들을 results.csv에 병합 저장. KEY가 같은 기존 행은 최신 실행으로 교체."""
    path = results_path() if path is None else path
    df = _numeric(pd.DataFrame(new_rows).reindex(columns=COLUMNS))
    if path.exists():
        old = _numeric(pd.read_csv(path).reindex(columns=COLUMNS))
        merged = pd.MultiIndex.from_frame(_key_frame(df))
        keep = ~pd.MultiIndex.from_frame(_key_frame(old)).isin(merged)
        if keep.any():
            df = pd.concat([old[keep], df], ignore_index=True)
    df = df.sort_values(["family", "model", "tag", "attribute", "metric", "seed"],
                        na_position="first").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# 학습 한 번이 남기는 노브/부가 측정치. clear.gnn.train_one 반환 dict의 키와 맞춘다.
RUN_PARAMS = ["edge_type", "k_neighbors", "hidden_dim", "num_layers", "dropout",
              "lr", "weight_decay", "aggr", "max_epochs", "patience", "val_size",
              "fair_alpha", "fair_beta", "grad_clip"]


def from_run(row, family, *, attribute=None, blind=None, group_set=None):
    """학습 한 번의 결과 dict -> long 행. (clear.gnn.train_one / baseline 공통)

    옛 clear.ledger.append가 하던 일이다. 차이는 고정 스키마로 reindex하지
    않는다는 것 — 그 reindex가 필요했던 이유(행마다 열이 다름)를 long format이
    없앴다. 실제로 옛 원장은 열이 두 번 늘어나며 헤더-행 길이가 어긋나 pandas가
    읽지 못하는 상태까지 갔다. 그때의 위치 기반 복구는 커밋 96d1dd1의
    migrate_results.py에 남아 있다(변환 완료 후 삭제).
    """
    metrics = {m: float(row[m]) for m in ACCURACY + RUNINFO
               if row.get(m) is not None and not isinstance(row.get(m), str)
               and pd.notna(row.get(m))}
    return rows(family, metrics,
                model=row.get("model"), tag=row.get("tag") or None,
                seed=row.get("seed"), attribute=attribute, blind=blind,
                group_set=group_set,
                params={k: row[k] for k in RUN_PARAMS if row.get(k) is not None},
                timestamp=row.get("timestamp"))


def from_wide(d, family, param_keys, *, note_keys=(), model=None, group_set=None):
    """`acc_*` / `dp_*` 형태의 wide 결과 행 하나 -> long 행.

    옛 *_tradeoff.csv의 열 이름 규약을 canonical로 옮긴다. 완화 스크립트
    이 규칙이 두 벌이면 마이그레이션한 과거 결과와 새로 만든 결과가 조용히
    다른 이름을 갖게 된다. 지금은 호출부가 없지만(mitigate_threshold를 지웠다)
    results.csv에 그 형식으로 들어간 4,004행이 남아 있으므로 규칙은 유지한다.
    """
    num, ci, std = {}, {}, {}
    for m in ACCURACY:
        if pd.notna(d.get(f"acc_{m}")):
            num[m] = float(d[f"acc_{m}"])
    if pd.notna(d.get("acc_mcc_std")):
        std["mcc"] = float(d["acc_mcc_std"])
    for old in ["dp_gap", "dp_amplification", "tpr_gap", "tpr_amplification",
                "base_rate_gap"]:
        if pd.notna(d.get(old)):
            new = ALIASES.get(old, old)
            num[new] = float(d[old])
            lo, hi = d.get(f"{old}_lo"), d.get(f"{old}_hi")
            if pd.notna(lo) and pd.notna(hi):
                ci[new] = (float(lo), float(hi))
    for m in ["best_val_gap", "n_eval"]:
        if pd.notna(d.get(m)):
            num[m] = float(d[m])
    return rows(family, num,
                model=d.get("model") or model, tag=d.get("tag"), seed=None,
                attribute=d.get("attribute"), blind=d.get("blind"),
                group_set=group_set,
                params={k: d[k] for k in param_keys
                        if k in d and pd.notna(d[k]) and d[k] != ""},
                notes={k: d[k] for k in note_keys
                       if k in d and pd.notna(d[k]) and d[k] != ""},
                ci=ci, std=std)


def read(family=None, metric=None, path=None):
    """results.csv 읽기(+ 선택 필터). params는 dict로 되돌려 놓는다."""
    path = results_path() if path is None else path
    df = pd.read_csv(path)
    if family is not None:
        df = df[df["family"].isin([family] if isinstance(family, str) else family)]
    if metric is not None:
        df = df[df["metric"].isin([metric] if isinstance(metric, str) else metric)]
    return df.reset_index(drop=True)


def param(df, name, cast=float):
    """params JSON에서 노브 하나를 열로 꺼낸다. 곡선 그릴 때 x축을 만드는 용도."""
    def _get(s):
        v = json.loads(s).get(name)
        if v is None:
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return v
    return df["params"].map(_get)


def wide(df, index, metrics=None):
    """long -> wide 피벗(사람이 읽는 표·플롯용). index는 행을 이루는 열 목록."""
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
    # dropna=False: index 열에 결측이 있어도 행을 버리지 않는다. tag가 없는 결과
    # (완화 후처리처럼 점을 params로만 구분하는 계열)가 통째로 사라지는 것을 막는다.
    return (df.pivot_table(index=index, columns="metric", values="value",
                           aggfunc="first", dropna=False)
              .reset_index()
              .rename_axis(columns=None))


def summarize(df, group_cols, metrics=None, sort_by="mcc"):
    """시드 반복 mean/std/n. 원장의 summarize()를 long 스키마로 옮긴 것.

    반환 열: group_cols + n + {지표}_mean/{지표}_std — 옛 형식과 같아서
    옛 ablation 요약과 같은 형식이다.
    """
    metrics = metrics or ACCURACY
    df = df[df["metric"].isin(metrics)]
    g = (df.groupby(list(group_cols) + ["metric"], dropna=False)["value"]
           .agg(["mean", "std", "count"]).reset_index())
    out = g.pivot_table(index=list(group_cols), columns="metric",
                        values=["mean", "std", "count"], aggfunc="first")
    out.columns = [f"{m}_{stat}" if stat != "count" else f"{m}_n"
                   for stat, m in out.columns]
    out = out.reset_index()
    n_col = f"{sort_by}_n"
    if n_col in out.columns:
        out = out.rename(columns={n_col: "n"})
    out = out[[c for c in out.columns if not c.endswith("_n")]]
    return out.sort_values(f"{sort_by}_mean", ascending=False)
