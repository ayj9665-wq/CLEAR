"""카운티 지도 재료 -- FIPS 조인 · 경계 도형 · Albers 투영 · 발산형 색.

## 왜 필요한가

이 데이터의 `City` 열은 사실상 **카운티**다. 상위 값이 Cook·Wayne·Harris·Dade·
Maricopa·Fulton 등 카운티명이고, State+City 조합이 3,042개로 미국 카운티 수(3,143)와
거의 일치한다 -- SHR 관할(agency) 단위가 카운티이기 때문이다. 따라서 지도의 정석
형태는 도시 점 지도가 아니라 **카운티 코로플레스**다.

## 왜 geopandas를 안 쓰나

geopandas는 GDAL/GEOS 바이너리 의존성이 있어 Windows에서 설치가 무겁고 깨지기 쉽다.
필요한 것은 (1) FIPS 조인, (2) 다각형 좌표, (3) 등적 투영뿐이고 셋 다 json + numpy로
충분하다. 같은 이유로 지도 라이브러리도 안 쓴다 -- 코로플레스는 결국 다각형에 색을
칠하는 것이다. 이 선택은 트랙 E(전국 웹 지도)와도 이어진다: 브라우저에 3MB짜리 차트
라이브러리를 인라인할 이유가 없고, 투영은 여기 파이썬에서 끝내고 웹은 SVG path만
받으면 된다.

## 참조 데이터

둘 다 공개 자료이며 `dataset/geo/`에 캐시한다(gitignore, 재다운로드 가능).

  national_county2020.txt   Census. (주 약칭, 카운티명) -> 5자리 FIPS
  geojson-counties-fips.json  FIPS로 키가 붙은 카운티 경계 3,221개

## 이름 조인의 함정

  독립시 vs 동명 카운티   'Baltimore city'(24510) vs 'Baltimore'(24005)
  버지니아 독립시 38개    전부 별도 FIPS
  개명                    'Dade' -> 현 'Miami-Dade'(1997)
  비-카운티 단위          루이지애나 'Orleans' 패리시, 알래스카 borough

따라서 접미사 정규화에서 **' city'는 떼지 않는다** -- 떼면 Baltimore city와 Baltimore
County가 충돌한다. 우리 데이터가 'Baltimore city'로 적고 있으므로 그대로 두면 맞는다.
조인 실패는 조용히 넘기지 않고 목록으로 돌려준다(match_report).
"""
import json
import urllib.request

import numpy as np
import pandas as pd

import config as C

GEO_DIR = C.ROOT / "dataset" / "geo"
COUNTY_FIPS_URL = ("https://www2.census.gov/geo/docs/reference/codes2020/"
                   "national_county2020.txt")
COUNTY_GEOJSON_URL = ("https://raw.githubusercontent.com/plotly/datasets/master/"
                      "geojson-counties-fips.json")

STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# 떼어낼 접미사. ' city'는 **의도적으로 없다**(독립시 구분이 사라진다).
_SUFFIXES = (" county", " parish", " borough", " census area", " municipality",
             " city and borough", " municipio")

# 데이터의 옛 이름 -> 현재 Census 이름. 1980-2014 기록이라 개명이 섞여 있다.
ALIASES = {
    ("Florida", "dade"): "miami-dade",
    ("Alaska", "prince of wales outer ketchikan"): "prince of wales-hyder",
    ("South Dakota", "shannon"): "oglala lakota",
}


def _cache(url, name):
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    path = GEO_DIR / name
    if not path.exists():
        print(f"[fetch] {url}")
        urllib.request.urlretrieve(url, path)
    return path


def normalize(name):
    """카운티명 정규화: 소문자, 구두점 제거, 접미사 제거, 공백 압축."""
    s = str(name).strip().lower().replace(".", "").replace("'", "")
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return " ".join(s.split())


def fips_table():
    """Census 코드 파일 -> DataFrame[state, county_norm, fips]."""
    path = _cache(COUNTY_FIPS_URL, "national_county2020.txt")
    df = pd.read_csv(path, sep="|", dtype=str)
    df["state"] = df["STATE"].map(STATE_ABBREV)
    df["fips"] = df["STATEFP"] + df["COUNTYFP"]
    df["county_norm"] = df["COUNTYNAME"].map(normalize)
    return df.dropna(subset=["state"])[["state", "county_norm", "fips",
                                        "COUNTYNAME"]]


def join_fips(df, state_col="State", city_col="City"):
    """(주, 카운티명) -> FIPS 조인. 반환 (조인된 df, 미매칭 DataFrame).

    미매칭을 **조용히 버리지 않는다** -- 지도에서 사라진 카운티는 눈에 띄지 않으므로,
    호출부가 매칭률을 보고 별칭을 보탤 수 있어야 한다.
    """
    ref = fips_table()
    out = df.copy()
    out["_state"] = out[state_col].astype(str)
    out["_norm"] = out[city_col].map(normalize)
    key = list(zip(out["_state"], out["_norm"]))
    out["_norm"] = [ALIASES.get(k, k[1]) for k in key]

    m = ref.drop_duplicates(["state", "county_norm"]).set_index(
        ["state", "county_norm"])["fips"]
    out["fips"] = pd.MultiIndex.from_arrays(
        [out["_state"], out["_norm"]]).map(m)
    missing = out[out["fips"].isna()][[state_col, city_col]].drop_duplicates()
    return out.drop(columns=["_state", "_norm"]), missing


# ---- 경계 도형 ---------------------------------------------------------------

def load_geometry(fips=None):
    """FIPS -> [ (lon, lat) 배열 ... ] (링 목록). fips를 주면 그 집합만 남긴다."""
    path = _cache(COUNTY_GEOJSON_URL, "geojson-counties-fips.json")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    want = set(fips) if fips is not None else None
    out = {}
    for feat in gj["features"]:
        fid = feat.get("id")
        if want is not None and fid not in want:
            continue
        geom = feat["geometry"]
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        rings = []
        for poly in polys:
            for ring in poly:                 # [0]=외곽, 이후는 구멍
                rings.append(np.asarray(ring, dtype=np.float64))
        out[fid] = rings
    return out


def albers(lon, lat, lat1=29.5, lat2=45.5, lon0=-96.0, lat0=37.5):
    """Albers 등적 원뿔 투영(CONUS 표준 파라미터). 단위구 기준 -- 지도 축척은 임의.

    등적(equal-area)이라야 코로플레스가 정직하다. 등각 투영은 고위도 카운티의
    면적을 부풀려 '큰 카운티가 중요해 보이는' 착시를 만든다.
    """
    p1, p2, l0, p0 = map(np.radians, (lat1, lat2, lon0, lat0))
    phi, lam = np.radians(lat), np.radians(lon)
    n = (np.sin(p1) + np.sin(p2)) / 2.0
    Cc = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(np.maximum(Cc - 2 * n * np.sin(phi), 0)) / n
    rho0 = np.sqrt(max(Cc - 2 * n * np.sin(p0), 0)) / n
    theta = n * (lam - l0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


# ---- 색: 발산형 (라이트 모드 전용) -------------------------------------------
#
# 이 프로젝트의 지도는 **라이트 모드 하나에만** 의도적으로 고정한다(확장설계서 §6-6).
# 그래서 팔레트는 라이트 표면(#fcfcfb)에 대해서만 성립하면 된다.
#
# 파랑 팔은 문서화된 순차 램프를 그대로 쓰고, 빨강 팔은 **각 파랑 단계와 OKLab
# 밝기(L)를 맞춰** 생성한다. 눈으로 고르지 않는 이유는 발산형의 요건이 '양 팔의
# 밝기 대칭'이기 때문이다 -- 한쪽이 밝으면 그쪽이 약해 보여서 0을 중심으로 한
# 대칭적 읽기가 깨진다. 중립 중점은 회색이어야 한다(색조를 두면 세 번째 범주로 읽힌다).

BLUE_ARM = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # 100…700
NEUTRAL = "#f0efec"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
HAIRLINE, AXIS = "#e1e0d9", "#c3c2b7"
RED_ANCHOR = "#e34948"          # 문서화된 categorical red -- 빨강 팔의 색조 기준

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def hex_to_oklab(h):
    rgb = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    lms = _M1 @ _srgb_to_lin(rgb)
    return _M2 @ np.cbrt(lms)


def oklab_to_hex(lab):
    lms = np.linalg.solve(_M2, lab) ** 3
    rgb = _lin_to_srgb(np.linalg.solve(_M1, lms))
    v = np.clip(np.round(rgb * 255), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(v)


def _in_gamut(lab):
    lms = np.linalg.solve(_M2, lab) ** 3
    rgb = _lin_to_srgb(np.linalg.solve(_M1, lms))
    return bool(np.all(rgb >= -1e-6) and np.all(rgb <= 1 + 1e-6))


def red_arm(blue_arm=None, anchor=RED_ANCHOR, chroma_scale=0.85):
    """파랑 팔의 각 단계와 밝기가 같은 빨강 단계들.

    각 밝기에서 색역 경계 채도를 이분법으로 찾은 뒤 chroma_scale만큼만 쓴다.
    경계를 그대로 쓰면 중간 밝기에서 형광에 가까운 빨강(#fb002b)이 나오는데,
    살인사건 지도에서 그 강도는 데이터가 아니라 색이 경보를 울리는 꼴이 된다.
    밝기는 그대로이므로 발산형의 요건(양 팔 대칭)은 유지된다.
    """
    blue_arm = blue_arm or BLUE_ARM
    _, aa, ba = hex_to_oklab(anchor)
    chroma = np.hypot(aa, ba)
    hue = np.arctan2(ba, aa)
    out = []
    for h in blue_arm:
        L = hex_to_oklab(h)[0]
        lo, hi = 0.0, chroma * 1.6           # 색역 경계를 이분법으로 찾는다
        for _ in range(40):
            mid = (lo + hi) / 2
            lab = np.array([L, mid * np.cos(hue), mid * np.sin(hue)])
            if _in_gamut(lab):
                lo = mid
            else:
                hi = mid
        c = lo * chroma_scale
        out.append(oklab_to_hex(np.array([L, c * np.cos(hue), c * np.sin(hue)])))
    return out


def arms(n_steps=None):
    """(파랑 단계, 빨강 단계) -- 5단계 램프에서 n_steps개를 **균등 샘플링**한다.

    앞에서부터 n_steps개를 자르면(arm[:3]) 램프의 밝은 쪽만 쓰게 되어 가장 강한
    구간조차 연한 색이 된다 -- 실제로 그렇게 그렸다가 최대 z=11.4인 카운티가
    중간 빨강으로 나왔다. 양 끝(가장 밝은 단계와 가장 어두운 단계)을 항상 포함해야
    구간 수와 무관하게 대비가 유지된다.
    """
    blue, red = BLUE_ARM, red_arm()
    if not n_steps or n_steps >= len(blue):
        return blue, red
    idx = np.round(np.linspace(0, len(blue) - 1, n_steps)).astype(int)
    return [blue[i] for i in idx], [red[i] for i in idx]


def diverging_colors(values, vmax=None, n_steps=None):
    """z 값 -> 색 목록. 0에 앵커 고정, 양 팔 단계 수 동일, 중점은 중립 회색.

    vmax를 주지 않으면 |z|의 최대로 잡는다. 양 팔을 같은 스케일로 자르므로
    부호가 대칭적으로 읽힌다.
    """
    blue, red = arms(n_steps)
    n_steps = len(blue)
    v = np.asarray(values, dtype=np.float64)
    vmax = float(vmax if vmax is not None else np.nanmax(np.abs(v)))
    edges = np.linspace(0, vmax, n_steps + 1)[1:]        # 각 팔의 상한들
    out = []
    for x in v:
        if not np.isfinite(x):
            out.append(NEUTRAL)
            continue
        arm = red if x > 0 else blue
        i = int(np.searchsorted(edges, abs(x)))
        out.append(NEUTRAL if abs(x) <= edges[0] * 0.25
                   else arm[min(i, n_steps - 1)])
    return out, vmax
