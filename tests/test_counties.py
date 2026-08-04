"""카운티 정본화 — 같은 카운티가 두 번 세어지면 안 된다.

겨냥하는 사고(2026-08-04): `clear.counties.ALIASES`가 **그리는 단계에서만** 적용돼
플로리다 `Dade`(9,054건)와 `Miami-Dade`(523건)가 별개 블록으로 검정됐다. 523건짜리
쪽이 z=+6.86으로 cold 판정을 받았고(합치면 SMR 1.408 -> 1.035), 지도는
`drop_duplicates('fips')`로 하나를 조용히 버렸다. 화면의 수를 CSV와 대조해서야
드러났다.

여기서 보는 것은 두 가지다. 정본화가 실제로 합치는가, 그리고 합치지 않은 표가
지도로 넘어갈 때 **시끄럽게 죽는가**.
"""
import pandas as pd
import pytest

from clear import counties as CT


def _frame(rows):
    return pd.DataFrame(rows, columns=["State", "City"])


def test_renamed_county_folds_into_one():
    """Dade와 Miami-Dade는 같은 카운티다(1997년 개명)."""
    df = _frame([("Florida", "Dade"), ("Florida", "Miami-Dade"),
                 ("Florida", "Broward")])
    out = CT.canonicalize(df, verbose=False)

    assert len(out) == len(df), "행이 사라지거나 늘었다"
    assert out["City"].nunique() == 2, f"합쳐지지 않았다: {sorted(out.City.unique())}"
    assert set(out["City"]) == {"Miami-Dade", "Broward"}


def test_absorbed_independent_city_folds():
    """Clifton Forge는 2001년 독립시 지위를 반납하고 Alleghany에 편입됐다."""
    out = CT.canonicalize(_frame([("Virginia", "Clifton Forge"),
                                  ("Virginia", "Alleghany")]), verbose=False)
    assert set(out["City"]) == {"Alleghany"}


def test_distinct_counties_are_not_merged():
    """독립시와 주변 카운티는 **다른 카운티**다. 과잉 병합이 더 위험하다."""
    df = _frame([("Maryland", "Baltimore"), ("Maryland", "Baltimore city"),
                 ("Missouri", "St. Louis"), ("Missouri", "St. Louis city")])
    out = CT.canonicalize(df, verbose=False)
    assert out.drop_duplicates().shape[0] == 4, (
        "서로 다른 카운티가 합쳐졌다 — 독립시를 주변 카운티에 흡수하면 "
        f"사건이 엉뚱한 곳으로 간다: {out.drop_duplicates().values.tolist()}")


def test_untouched_names_survive_verbatim():
    """별칭에 없는 이름은 표기가 그대로여야 한다(다른 표와 조인되기 때문)."""
    df = _frame([("Michigan", "Wayne"), ("Texas", "Harris"),
                 ("California", "Los Angeles")])
    out = CT.canonicalize(df, verbose=False)
    assert out.equals(df)


def test_unmappable_placeholder_is_left_alone():
    """`Repressed`는 카운티명이 아니라 비식별 표시다. 합치지도, 지우지도 않는다."""
    df = _frame([("Maryland", "Repressed"), ("Idaho", "Repressed")])
    out = CT.canonicalize(df, verbose=False)
    assert list(out["City"]) == ["Repressed", "Repressed"]
    assert out.drop_duplicates().shape[0] == 2, "주가 다르면 다른 행이다"


def test_duplicate_fips_raises_instead_of_dropping():
    """한 FIPS에 행이 둘이면 **죽어야 한다.**

    조용한 `drop_duplicates`가 Dade 결함을 석 달 살렸다. 조용한 유실을 시끄러운
    실패로 바꾼 것이 수정의 절반이므로, 그 절반이 살아 있는지 본다.
    """
    tab = pd.DataFrame({"State": ["Florida", "Florida"],
                        "City": ["Dade", "Miami-Dade"],
                        "fips": ["12086", "12086"], "n": [9054, 523]})
    with pytest.raises(SystemExit) as e:
        CT.assert_one_row_per_fips(tab, where="테스트")
    assert "12086" in str(e.value)


def test_unique_fips_passes_and_nan_is_ignored():
    """FIPS가 유일하면 통과한다. 조인 실패(NaN)는 중복 판정 대상이 아니다."""
    tab = pd.DataFrame({"State": ["Florida", "Texas", "Maryland"],
                        "City": ["Miami-Dade", "Harris", "Repressed"],
                        "fips": ["12086", "48201", None]})
    CT.assert_one_row_per_fips(tab)      # 예외가 없어야 한다


def test_normalize_folds_diacritics_and_suffix():
    """Census는 'Doña Ana County', 우리 데이터는 'Dona Ana'로 쓴다."""
    assert CT.normalize("Doña Ana County") == CT.normalize("Dona Ana")
    assert CT.normalize("St. Louis") == CT.normalize("St Louis")
