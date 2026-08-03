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
import unicodedata
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
#
# **긴 것부터** 봐야 한다. 원래는 선언 순서대로 첫 일치에서 멈췄는데, ' borough'가
# ' city and borough'보다 앞에 있어서 'Juneau City and Borough'가 'juneau city'로
# 잘렸다(알래스카 전역이 이렇게 미매칭됐다). 정렬로 강제한다.
_SUFFIXES = tuple(sorted(
    (" county", " parish", " borough", " census area", " municipality",
     " city and borough", " municipio"),
    key=len, reverse=True))

# 데이터의 주 이름 오기. 'Rhodes Island'는 전국 표본에서 로드아일랜드 전체를
# 미매칭으로 만든다(1,203건).
STATE_ALIASES = {"Rhodes Island": "Rhode Island"}

# 데이터의 옛 이름 -> 현재 Census 이름. 1980-2014 기록이라 개명이 섞여 있다.
#
# **키는 normalize()를 통과한 형태여야 한다.** 알래스카 항목이 원래 공백 표기라
# (normalize는 하이픈을 보존한다) 한 번도 일치하지 않는 죽은 별칭이었다 -- 3개 주
# 표본에는 알래스카가 없어서 드러나지 않았다.
ALIASES = {
    ("Florida", "dade"): "miami-dade",
    ("South Dakota", "shannon"): "oglala lakota",
    # 알래스카 census area 개편(2008~2015). 데이터는 개편 전 이름이라 분할 후
    # 인구가 많은 쪽으로 보낸다 -- 정확한 복원이 불가능하므로 근사임을 명시한다.
    ("Alaska", "prince of wales-outer ketchikan"): "prince of wales-hyder",
    ("Alaska", "wrangell-petersburg"): "wrangell",
    ("Alaska", "skagway-hoonah-angoon"): "hoonah-angoon",
    # 네바다 카슨시티는 데이터가 'Carson City city'로 중복 표기한다.
    ("Nevada", "carson city city"): "carson city",
    # 2001년 독립시 지위를 반납하고 Alleghany 카운티로 편입.
    ("Virginia", "clifton forge"): "alleghany",
}

# 'Repressed'는 카운티명이 아니라 **비식별 처리 표시**다(MAP이 소표본 관할을 가린
# 것). 조인 실패가 아니라 애초에 지도에 올릴 수 없는 행이므로, 미매칭 목록에서
# 빼서 진짜 조인 문제와 섞이지 않게 한다.
NON_COUNTY = {"repressed"}


def _cache(url, name):
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    path = GEO_DIR / name
    if not path.exists():
        print(f"[fetch] {url}")
        urllib.request.urlretrieve(url, path)
    return path


def normalize(name):
    """카운티명 정규화: 발음부호 제거, 소문자, 구두점 제거, 접미사 제거, 공백 압축.

    발음부호를 접는 이유: Census는 'Doña Ana County'로 쓰고 우리 데이터는
    'Dona Ana'로 쓴다. NFKD로 분해한 뒤 결합문자를 버리면 둘 다 'dona ana'가 된다.
    """
    s = unicodedata.normalize("NFKD", str(name).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", "").replace("'", "")
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
    out["_state"] = out[state_col].astype(str).map(lambda s: STATE_ALIASES.get(s, s))
    out["_norm"] = out[city_col].map(normalize)
    key = list(zip(out["_state"], out["_norm"]))
    out["_norm"] = [ALIASES.get(k, k[1]) for k in key]

    m = ref.drop_duplicates(["state", "county_norm"]).set_index(
        ["state", "county_norm"])["fips"]

    def _lookup(names):
        return pd.MultiIndex.from_arrays([out["_state"], names]).map(m)

    out["fips"] = _lookup(out["_norm"])

    # 폴백 두 벌. **둘 다 1차 조회가 실패한 행에만** 적용한다 -- 먼저 시도하면
    # 애매한 이름의 판정을 바꿔버린다. 예: 버지니아 'Richmond'는 'Richmond County'와
    # 'Richmond city'가 **둘 다 존재**하는데, 1차에서 County로 붙는 현행 동작을
    # 유지해야 과거 결과가 재현된다.
    for fallback in (
        # 붙여쓰기: Census 'DeKalb'/'LaPorte'/'DeSoto' vs 데이터 'De Kalb'/'La Porte'
        lambda s: s.str.replace(" ", "", regex=False),
        # 독립시: Census 'Norfolk city' vs 데이터 'Norfolk'. 버지니아 38개 독립시가
        # 전부 여기 걸린다(원래 카운티가 폐지돼 동명 카운티 자체가 없다).
        lambda s: s + " city",
    ):
        miss = out["fips"].isna()
        if not miss.any():
            break
        out.loc[miss, "fips"] = _lookup(fallback(out["_norm"]))[miss]

    # 비식별 표시는 '조인 못 한 카운티'가 아니라 '카운티가 아닌 값'이다.
    missing = out[out["fips"].isna() & ~out["_norm"].isin(NON_COUNTY)]
    missing = missing[[state_col, city_col]].drop_duplicates()
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
    """Albers 등적 원뿔 투영(기본값은 CONUS 표준 파라미터). 단위구 기준 -- 축척은 임의.

    등적(equal-area)이라야 코로플레스가 정직하다. 등각 투영은 고위도 카운티의
    면적을 부풀려 '큰 카운티가 중요해 보이는' 착시를 만든다.
    """
    p1, p2, l0, p0 = map(np.radians, (lat1, lat2, lon0, lat0))
    phi, lam = np.radians(lat), np.radians(lon)
    n = (np.sin(p1) + np.sin(p2)) / 2.0
    Cc = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(np.maximum(Cc - 2 * n * np.sin(phi), 0)) / n
    rho0 = np.sqrt(max(Cc - 2 * n * np.sin(p0), 0)) / n
    # 경도차를 (-pi, pi]로 감는다. 안 감으면 날짜변경선을 넘는 도형이 지구를 한 바퀴
    # 돌아 반대쪽으로 날아간다 -- 알류샨 열도(경도가 -180과 +180을 오간다)가 정확히
    # 그 경우이고, 감기 전 알래스카의 y 범위가 0.36~2.33까지 벌어졌다. CONUS는
    # 애초에 날짜변경선과 무관하므로 이 변경의 영향을 받지 않는다.
    dlam = (lam - l0 + np.pi) % (2 * np.pi) - np.pi
    theta = n * dlam
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


# ---- Albers USA 합성(전국 지도) ---------------------------------------------
#
# CONUS 파라미터 하나로는 알래스카·하와이가 화면 밖으로 나간다. 표준 해법은
# **부분마다 다른 Albers를 쓰고 결과를 이동·축소해 한 화면에 배치**하는 것이다
# (d3.geoAlbersUsa와 같은 구성).
#
# 정직성에 대해 분명히 해둘 것: **합성 전체는 등적이 아니다.** 각 조각 안에서는
# 등적이지만 알래스카는 0.35배로 줄여 놓았으므로, 알래스카 카운티의 화면 면적을
# CONUS 카운티와 비교하면 안 된다. 지도에 이 배율을 표기한다(§6-9의 오독 방지와
# 같은 성격 -- 색 이전에 좌표부터 오해를 부른다).
#
# 분기는 **점이 아니라 도형 단위**로 해야 한다. 점마다 판정하면 경계에 걸친
# 다각형이 두 투영으로 찢어진다. 그래서 FIPS 앞 2자리(주 코드)로 나눈다.
ALASKA_FIPS, HAWAII_FIPS = "02", "15"

_USA_PARTS = {
    # 조각별 (Albers 파라미터, 축척, 이동). 이동량은 CONUS를 투영해 본 실측
    # 경계에 맞춰 잡는다(_place_parts 참고).
    ALASKA_FIPS: dict(params=(55.0, 65.0, -154.0, 50.0), scale=0.35),
    HAWAII_FIPS: dict(params=(8.0, 18.0, -157.0, 13.0), scale=1.0),
    None:        dict(params=(29.5, 45.5, -96.0, 37.5), scale=1.0),
}

# 조각을 CONUS 아래-왼쪽에 놓기 위한 평행이동(단위구 좌표).
#
# 눈으로 고른 값이 아니라 **실측 범위에서 역산**했다(각 조각을 자기 투영으로 그린 뒤
# 경계를 잰 값):
#   CONUS x[-0.369, 0.353] y[-0.210, 0.246]
#   AK    x[-0.339, 0.233] y[ 0.065, 0.372]   (0.35배 축소 후 x[-0.119,0.082] y[0.023,0.130])
#   HI    x[-0.058, 0.036] y[ 0.104, 0.161]
# **y는 북쪽으로 증가**하므로(아래 albers 주석) "CONUS 아래"는 y < -0.210이다. AK의
# 위 끝을 y=-0.24, HI의 위 끝을 y=-0.25에 두고 x_min을 각각 -0.36, -0.13에 맞춘 값이다.
#
# **하와이는 CONUS와 같은 축척(1.0)이라 면적 비교가 성립한다. 줄인 것은 알래스카뿐**
# (0.35배)이므로, 오독 방지 문구는 알래스카만 지목하면 된다.
_USA_OFFSET = {ALASKA_FIPS: (-0.241, -0.370), HAWAII_FIPS: (-0.072, -0.411),
               None: (0.0, 0.0)}


def albers_usa(lon, lat, fips):
    """Albers USA 합성. fips는 **도형 하나의** 5자리 코드(앞 2자리만 본다).

    반환은 albers()와 같은 단위구 스케일이며 **y는 북쪽으로 증가**한다(위도가
    높을수록 rho가 작아지고 y = rho0 - rho·cos θ가 커진다). 실측: Seattle y=+0.220
    vs Miami y=-0.185. 따라서 matplotlib에 그릴 때 **y를 뒤집으면 안 된다** --
    experiments/map_figures.py가 albers()를 그대로 쓰는 것과 같은 규약이다.
    """
    key = str(fips)[:2]
    part = _USA_PARTS.get(key if key in _USA_PARTS else None)
    lat1, lat2, lon0, lat0 = part["params"]
    x, y = albers(lon, lat, lat1, lat2, lon0, lat0)
    dx, dy = _USA_OFFSET.get(key if key in _USA_OFFSET else None)
    return x * part["scale"] + dx, y * part["scale"] + dy


# ---- 웹 지도용: 단순화 · SVG path 직렬화 --------------------------------------
#
# 코로플레스는 결국 <path>에 fill을 칠하는 것이므로 브라우저에는 좌표만 넘기면
# 된다(확장설계서 §6-3). 어려운 쪽(투영·단순화)은 여기 파이썬에서 끝낸다.

def simplify(ring, tol):
    """Douglas-Peucker. ring: (N,2), tol: 투영 좌표 단위 허용오차.

    **재귀가 아니라 스택**으로 돈다 -- 카운티 외곽선은 점이 수천 개라 재귀로 짜면
    파이썬 기본 재귀한도에 걸린다.

    주의: 다각형마다 **독립적으로** 단순화하므로 인접 카운티의 공유 경계가 서로
    다르게 줄어 틈(sliver)이 생긴다. 위상을 보존하는 단순화는 훨씬 복잡하고, 이
    지도에서는 얇은 흰 stroke로 경계를 그리므로 그 틈이 stroke 아래로 숨는다.
    허용오차를 키울 때는 **렌더해서 확인**할 것(§6-3-1의 교훈).
    """
    n = len(ring)
    if n < 3:
        return ring
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = ring[j] - ring[i]
        L = np.hypot(*seg)
        pts = ring[i + 1:j] - ring[i]
        # 선분 길이가 0이면(닫힌 고리의 시작=끝) 점까지의 거리로 대체한다.
        d = (np.abs(pts[:, 0] * seg[1] - pts[:, 1] * seg[0]) / L if L > 0
             else np.hypot(pts[:, 0], pts[:, 1]))
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return ring[keep]


def svg_path(rings, scale, ox, oy, tol=0.0, ndigits=1):
    """링 목록 -> SVG path 문자열 하나(서브패스 여러 개, 각각 Z로 닫는다).

    좌표는 (v*scale + offset)으로 viewBox에 맞춘 뒤 ndigits로 **양자화**한다.
    단순화 다음으로 페이로드를 가장 크게 줄이는 것이 이 반올림이다 -- 좌표
    문자열이 '123.456789'에서 '123.5'로 줄어든다.

    y는 albers가 북쪽으로 증가하게 주는데 SVG는 아래로 증가하므로 **여기서 뒤집는다**
    (호출부가 또 뒤집지 않도록 이 함수가 유일한 뒤집기 지점이다).
    """
    out = []
    for ring in rings:
        r = simplify(ring, tol) if tol > 0 else ring
        if len(r) < 3:
            continue
        xs = np.round(r[:, 0] * scale + ox, ndigits)
        ys = np.round(-r[:, 1] * scale + oy, ndigits)
        pts = [f"{x:g},{y:g}" for x, y in zip(xs, ys)]
        out.append("M" + "L".join(pts) + "Z")
    return "".join(out)


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


def gray_arm(blue_arm=None, l_floor=0.66):
    """파랑 팔을 대신하는 무채색 팔. **밝기를 그대로 두지 않고 압축한다.**

    포트폴리오 페이지(experiments/build_story_page.py)는 "붉은색은 격차 경보 한
    가지 뜻으로만 쓴다"는 규칙을 두므로, 발산형의 반대편 팔에서 색상을 뺀다.

    처음에는 red_arm처럼 **밝기를 정확히 맞춰**(chroma만 0으로) 만들었다. 산술은
    옳았지만 그려 놓고 보니 규칙이 뒤집혔다 -- 밝은 표면 위에서 같은 밝기라면
    **무채색이 유채색보다 대비가 세다.** 가장 어두운 파랑 단계의 L=0.338을 그대로
    회색으로 옮기면 #4a4a4a가 나오는데, 흰 카운티들 사이에서 이게 같은 밝기의
    빨강보다 훨씬 강하게 튄다. 실제 지도에서 캘리포니아·네바다의 warm 카운티가
    화면을 지배하고 정작 cold 카운티가 옅어 보였다.

    그래서 밝기 범위를 [L_max, l_floor]로 **아핀 압축**한다. 단계 간 상대 간격은
    보존되므로 순서와 단조성은 그대로이고, 팔 전체가 조용해질 뿐이다. 발산형의
    "양 팔 밝기 대칭"을 의도적으로 깨는 것이며, 그 대가로 warm 쪽의 크기 비교가
    어려워진다 -- 정확한 값은 표에서 읽는다.

    렌더해서 눈으로 보기 전에는 안 보이는 종류의 문제였다(map_figures의 버그 셋과
    같은 계열). 검증기는 색을 재지만 위계가 뒤집혔는지는 재지 않는다.
    """
    ls = [hex_to_oklab(h)[0] for h in (blue_arm or BLUE_ARM)]
    lo, hi = min(ls), max(ls)
    span = (hi - lo) or 1.0
    return [oklab_to_hex(np.array([hi - (hi - L) * (hi - l_floor) / span, 0.0, 0.0]))
            for L in ls]


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
