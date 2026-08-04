"""results.csv의 identity 규약 — 재실행이 **교체**여야 하고 중복이면 안 된다.

겨냥하는 사고: 적합된 임계값을 identity 키에 넣었더니 재실행 한 번이 78개의 중복
행을 만들었다. `params`(정한 노브)는 identity이고 `notes`(실행이 만든 값)는 아니다,
가 그 뒤에 세운 규약이다. 이 파일은 그 규약이 살아 있는지만 본다.
"""
import json

import pandas as pd

from clear import results as R


def _row(**kw):
    return R.rows("train", {"mcc": kw.pop("mcc", 0.30)}, **kw)


def test_same_params_replace_not_append(tmp_path):
    """같은 설정을 다시 쓰면 행이 늘지 않고 값만 바뀐다."""
    p = tmp_path / "results.csv"
    R.write(_row(model="graphsage", params={"edge_type": "geo"}, mcc=0.30), p)
    R.write(_row(model="graphsage", params={"edge_type": "geo"}, mcc=0.31), p)

    df = pd.read_csv(p)
    assert len(df) == 1, f"재실행이 교체가 아니라 추가됐다 ({len(df)}행)"
    assert df["value"].iloc[0] == 0.31, "교체는 됐는데 값이 최신이 아니다"


def test_different_params_are_different_rows(tmp_path):
    """params가 다르면 서로 다른 실행이다 — 덮어쓰면 안 된다."""
    p = tmp_path / "results.csv"
    R.write(_row(model="graphsage", params={"edge_type": "geo"}), p)
    R.write(_row(model="graphsage", params={"edge_type": "temporal"}), p)
    assert len(pd.read_csv(p)) == 2


def test_notes_are_not_part_of_identity(tmp_path):
    """notes만 다른 재실행은 **교체**여야 한다. 이것이 78행 중복의 직접 원인이었다."""
    p = tmp_path / "results.csv"
    R.write(_row(model="m", params={"lr": 0.005}, notes={"threshold": 0.500}), p)
    R.write(_row(model="m", params={"lr": 0.005}, notes={"threshold": 0.517}), p)

    df = pd.read_csv(p)
    assert len(df) == 1, ("notes가 identity에 들어갔다 — 적합된 값이 바뀔 때마다 "
                          f"행이 쌓인다 ({len(df)}행)")


def test_seed_is_part_of_identity(tmp_path):
    """시드별로 한 줄씩 남아야 한다(집계는 읽는 쪽이 한다)."""
    p = tmp_path / "results.csv"
    R.write(_row(model="graphsage", seed=42, params={"edge_type": "geo"}), p)
    R.write(_row(model="graphsage", seed=43, params={"edge_type": "geo"}), p)
    assert len(pd.read_csv(p)) == 2


def test_from_run_drops_none_valued_params(tmp_path):
    """None인 노브는 params JSON에 실리지 않는다 — 드롭은 `from_run`이 한다.

    `GNN_EDGE_MODE=None` / `scope_param()` 규약이다. full-batch 실행에서
    `edge_mode`/`minibatch`/`batch_size`가 전부 None이고, 그것이 params에 실리면
    **기존 행의 params JSON이 달라져 identity가 깨진다** — 재실행이 교체가 아니라
    중복이 된다. 학습기가 부르는 경로가 `from_run`이므로 거기서 검증한다
    (`rows()`에 직접 None을 넘기는 호출부는 없다).
    """
    p = tmp_path / "results.csv"
    run = {"model": "graphsage", "seed": 42, "mcc": 0.30,
           "edge_type": "geo", "edge_mode": None, "minibatch": None}
    R.write(R.from_run(run, "train"), p)

    got = json.loads(pd.read_csv(p)["params"].iloc[0])
    assert got == {"edge_type": "geo"}, f"None 키가 params에 실렸다: {got}"


def test_from_run_none_params_keep_the_row_replaceable(tmp_path):
    """위 규약의 목적 자체를 검증한다: None이 섞여도 재실행이 교체여야 한다."""
    p = tmp_path / "results.csv"
    base = {"model": "graphsage", "seed": 42, "edge_type": "geo"}
    R.write(R.from_run({**base, "mcc": 0.30, "edge_mode": None}, "train"), p)
    R.write(R.from_run({**base, "mcc": 0.31, "edge_mode": None}, "train"), p)

    df = pd.read_csv(p)
    assert len(df) == 1 and df["value"].iloc[0] == 0.31


def test_new_metric_is_a_new_row(tmp_path):
    """long format의 핵심 규약 — 새 지표는 새 행이지 새 열이 아니다."""
    p = tmp_path / "results.csv"
    R.write(R.rows("train", {"mcc": 0.3, "auc": 0.7}, model="m",
                   params={"edge_type": "geo"}), p)
    df = pd.read_csv(p)
    assert len(df) == 2
    assert set(df["metric"]) == {"mcc", "auc"}
