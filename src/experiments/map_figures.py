"""
experiments/map_figures.py -- 카운티 코로플레스 (확장설계서 트랙 E, 1단계)

detect_cold_blocks가 낸 블록별 관측-기대 잔차를 지도로 그린다. 표본이 CA+TX+MI
3개 주뿐이므로 여기서는 **정적 PNG**(포스터용)이고, 전국(카운티 3,042개)은
호버·줌이 필수라 웹으로 간다 -- 그때 이 스크립트가 만든 FIPS 조인과 투영을 그대로
재사용한다.

## 왜 잔차를 칠하는가 (원시 검거율이 아니라)

원시 검거율 지도는 "남부·시골이 높고 대도시가 낮다"만 보여준다 -- 이미 알려진
사실이고 새 정보가 0이다. 게다가 주별 검거율이 34%(DC)~93%(ND)로 갈려 전국을
그대로 칠하면 심슨의 역설에 걸린다. 잔차는 "모델이 기대한 것 대비"라 주·사건구성
차이가 이미 p_hat에 흡수되므로, **잔차 지도가 심슨 역설 대응이자 유일하게 새로운
정보**다. 원시 검거율은 참고 패널로만 둔다.

## 그림

  fig1  주별 3패널 카운티 코로플레스, 색 = 표준화 잔차 z (발산형)
  fig2  z vs 흑인 피해자 비중 산점도 -- 이 분석의 중심 결과이자 윤리 게이트

## 색 규약 (라이트 모드 전용, 확장설계서 §6-6)

발산형 blue<->red, 중립 중점은 **회색**, 0에 앵커 고정, 양 팔 단계 수 동일.
빨강 팔은 파랑 팔과 OKLab 밝기를 맞춰 생성한다(clear.counties.red_arm) -- 한쪽이
밝으면 그쪽이 약해 보여 대칭적 읽기가 깨진다. 무지개 팔레트는 쓰지 않고, 중점에
색조를 두지 않는다(세 번째 범주로 읽힌다).

**표본이 적은 카운티는 램프의 옅은 색으로 칠하지 않는다.** 그건 '값이 0에 가깝다'는
다른 주장이 된다. 빗금 + 범례 항목으로 '판정 제외'임을 명시한다.

matplotlib 출력 텍스트는 **영문만** -- 기본 폰트에 한글 글리프가 없어 저장 시
빈 네모로 렌더된다.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

import config as C
from clear import counties as CT

# 주 이름 -> STATEFP. 지도에 그릴 '전체' 카운티(판정 제외분 포함)를 뽑기 위함.
STATEFP = {"California": "06", "Texas": "48", "Michigan": "26"}
# 주 이름 -> 약칭. State[:2]로 자르면 Texas가 "TE"가 된다(실제로 그렇게 나왔다).
ABBREV = {v: k for k, v in CT.STATE_ABBREV.items()}


def rings_to_path(rings):
    """[(lon,lat) 배열, ...] -> 투영된 matplotlib Path (구멍 포함)."""
    verts, codes = [], []
    for ring in rings:
        x, y = CT.albers(ring[:, 0], ring[:, 1])
        pts = np.column_stack([x, y])
        verts.append(pts)
        codes.append([Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY])
    return Path(np.concatenate(verts), np.concatenate(codes))


def color_for(row, edges, blue, red):
    """카운티 하나의 채움색.

    **FDR 유의한 블록만 색을 받는다.** 143개 카운티의 z를 전부 칠하면 유의하지 않은
    123개(대부분 노이즈)가 신호처럼 읽힌다 -- 이 프로젝트가 F1/Precision을 시드
    노이즈 안이라는 이유로 주장에서 뺀 것과 같은 규율이다. 판정했으나 유의하지 않은
    카운티는 중립 회색이고, 아예 판정 못 한 카운티(n<min_n)는 호출부가 빗금으로
    구분한다.
    """
    # 주의: CSV 왕복에서 빈 문자열은 NaN으로 읽힌다. `== ""`만 보면 판정이 조용히
    # 뒤집혀 유의하지 않은 카운티까지 전부 칠해진다(실제로 한 번 그렇게 나왔다).
    if pd.isna(row["flag"]) or row["flag"] == "":
        return CT.NEUTRAL
    arm = red if row["z"] > 0 else blue
    return arm[min(int(np.searchsorted(edges, abs(row["z"]))), len(edges) - 1)]


def draw_state(ax, state, tab, geom, all_fips, edges, blue, red, label_top=0):
    """한 주의 카운티를 칠한다. tab: 그 주의 판정 결과."""
    cmap = {r["fips"]: color_for(r, edges, blue, red) for _, r in tab.iterrows()}
    centro = {}

    xs, ys = [], []
    for fips in all_fips:
        rings = geom.get(fips)
        if rings is None:
            continue
        path = rings_to_path(rings)
        if fips in cmap:
            ax.add_patch(PathPatch(path, facecolor=cmap[fips],
                                   edgecolor=CT.SURFACE, linewidth=0.35))
        else:
            # 판정 제외(n < min_n). 램프 색이 아니라 빗금 -- '0에 가깝다'가 아니라
            # '재지 않았다'는 뜻이어야 한다.
            ax.add_patch(PathPatch(path, facecolor=CT.SURFACE, hatch="////",
                                   edgecolor=CT.AXIS, linewidth=0.35))
        v = path.vertices
        centro[fips] = v.mean(axis=0)
        xs.append(v[:, 0]); ys.append(v[:, 1])

    x = np.concatenate(xs); y = np.concatenate(ys)
    pad = 0.02 * max(np.ptp(x), np.ptp(y))
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(state, color=CT.INK, fontsize=11, pad=4)

    # 유의한 과잉 미해결 카운티만 직접 라벨. 전부 달면 읽을 수 없다.
    # 인접 카운티(SF/Alameda)가 겹치므로 순위마다 세로로 어긋나게 놓는다.
    span = np.ptp(y)
    for rank, (_, r) in enumerate(
            tab[tab["flag"] == "cold"].nlargest(label_top, "z").iterrows()):
        cx, cy = centro.get(r["fips"], (None, None))
        if cx is None:
            continue
        dy = span * (0.035 if rank % 2 == 0 else -0.045)
        ax.annotate(r["City"], (cx, cy), xytext=(cx, cy + dy),
                    fontsize=7.5, color=CT.INK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=CT.MUTED, lw=0.6),
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=CT.SURFACE,
                              edgecolor="none", alpha=0.85))


def _swatch(color, label, edge=None):
    return Line2D([], [], marker="s", linestyle="", markersize=11, label=label,
                  markerfacecolor=color, markeredgecolor=edge or CT.SURFACE)


def fig_map(tab, out_path, min_n, vmax=None, n_steps=3):
    ref = CT.fips_table()
    states = [s for s in STATEFP if (tab["State"] == s).any()]
    all_fips = {s: sorted(ref.loc[ref["fips"].str[:2] == STATEFP[s], "fips"])
                for s in states}
    geom = CT.load_geometry({f for v in all_fips.values() for f in v})

    vmax = float(vmax if vmax is not None else np.abs(tab["z"]).max())
    edges = np.linspace(0, vmax, n_steps + 1)[1:]
    blue, red = CT.arms(n_steps)
    # 패널 폭을 각 주의 투영 가로폭에 비례시켜 축척이 왜곡되지 않게 한다.
    widths = []
    for s in states:
        pts = np.concatenate([np.concatenate(geom[f]) for f in all_fips[s]
                              if f in geom])
        gx, _ = CT.albers(pts[:, 0], pts[:, 1])
        widths.append(np.ptp(gx))
    fig, axes = plt.subplots(1, len(states), figsize=(13, 5.6),
                             gridspec_kw={"width_ratios": widths})
    fig.patch.set_facecolor(CT.SURFACE)
    for ax, s in zip(np.atleast_1d(axes), states):
        ax.set_facecolor(CT.SURFACE)
        draw_state(ax, s, tab[tab["State"] == s], geom, all_fips[s], edges,
                   blue, red, label_top=3)

    n_cold = int((tab["flag"] == "cold").sum())
    n_warm = int((tab["flag"] == "warm").sum())
    fig.suptitle("Counties with more unsolved homicides than the model expects",
                 color=CT.INK, fontsize=14, y=0.99)
    fig.text(0.5, 0.935,
             f"Standardized residual z, observed minus expected unsolved cases, "
             f"against a race-neutral baseline. Only blocks significant at "
             f"FDR 5% are coloured ({n_cold} excess, {n_warm} below).",
             ha="center", color=CT.INK2, fontsize=9)
    fig.text(0.5, 0.905,
             "The model has no county identity, so z absorbs every county-level "
             "effect at once - policing, resourcing, urbanicity, recording. "
             "Not causal.",
             ha="center", color=CT.MUTED, fontsize=8)

    def _band(i):
        """구간 i의 경계 문구. searchsorted(edges, |z|)와 정확히 맞춰야 한다 --
        어긋나면 그림이 말하는 값과 범례가 다른 값을 가리킨다."""
        lo = 0.0 if i == 0 else edges[i - 1]
        return (f"{lo:.0f}-{edges[i]:.0f}" if i < n_steps - 1
                else f"over {lo:.0f}")

    handles = (
        [_swatch(red[i], f"excess unsolved,  z {_band(i)}")
         for i in range(n_steps - 1, -1, -1)]
        + [_swatch(CT.NEUTRAL, "assessed, not significant"),
           _swatch(CT.SURFACE, f"not assessed (n < {min_n})", edge=CT.AXIS)]
        + [_swatch(blue[i], f"fewer unsolved,  |z| {_band(i)}")
           for i in range(n_steps - 1, -1, -1)])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, labelcolor=CT.INK2, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0.11, 1, 0.885])
    fig.savefig(out_path, dpi=200, facecolor=CT.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return vmax


def fig_race(tab, out_path):
    """z vs 흑인 피해자 비중. 이 분석의 중심 결과이자 윤리 게이트."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.patch.set_facecolor(CT.SURFACE); ax.set_facecolor(CT.SURFACE)

    colors, vmax = CT.diverging_colors(tab["z"].values)
    ax.axhline(0, color=CT.AXIS, linewidth=1, zorder=1)
    ax.scatter(tab["black_share"], tab["z"],
               s=np.sqrt(tab["n"]) * 1.6, c=colors,
               edgecolors=CT.SURFACE, linewidths=0.6, zorder=3)

    r = np.corrcoef(tab["black_share"], tab["z"])[0, 1]
    ax.set_xlabel("Share of Black victims in county", color=CT.INK2, fontsize=10)
    ax.set_ylabel("Standardized residual z", color=CT.INK2, fontsize=10)
    ax.set_title(f"Unexplained unsolved cases vs. victim race composition  "
                 f"(r = {r:+.2f})", color=CT.INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(CT.AXIS)
    ax.tick_params(colors=CT.MUTED, labelsize=9)
    ax.grid(axis="y", color=CT.HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # 상위 이례 카운티만 직접 라벨 -- 모든 점에 이름을 달면 읽을 수 없다.
    for _, row in tab.nlargest(4, "z").iterrows():
        ax.annotate(f"{row['City']}, {ABBREV.get(row['State'], row['State'])}",
                    (row["black_share"], row["z"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=8, color=CT.INK2)
    fig.text(0.01, 0.005, "Marker area ~ county sample size. "
             "Baseline is race-neutral, so this correlation is not absorbed - "
             "but it is not causal either (the model has no county identity).",
             fontsize=7.5, color=CT.MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=200, facecolor=CT.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_key", default="county")
    ap.add_argument("--min_n", type=int, default=20,
                    help="범례 문구용(detect_cold_blocks의 --min_n과 맞출 것)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="색 스케일 상한(기본 |z|의 최대). 여러 그림의 축을 맞출 때")
    args = ap.parse_args()

    src = C.OUTPUT_DIR / "cold_blocks.csv"
    if not src.exists():
        raise SystemExit(f"[에러] {src} 없음. 먼저: "
                         f"python -m experiments.detect_cold_blocks")
    tab = pd.read_csv(src)
    tab = tab[tab["block_key"] == args.block_key].copy()

    tab, missing = CT.join_fips(tab)
    matched = tab["fips"].notna()
    print(f"[join] FIPS 매칭 {int(matched.sum())}/{len(tab)} "
          f"({matched.mean() * 100:.1f}%)")
    if len(missing):
        print(f"[경고] 미매칭 {len(missing)}건 - clear.counties.ALIASES에 추가 필요:")
        print(missing.to_string(index=False))
    tab = tab[matched]

    p1 = C.OUTPUT_DIR / "map_fig1_county_residual.png"
    vmax = fig_map(tab, p1, args.min_n, args.vmax)
    print(f"[save] {p1}  (카운티 {len(tab)}개, 색 스케일 |z| <= {vmax:.1f})")

    p2 = C.OUTPUT_DIR / "map_fig2_residual_vs_race.png"
    fig_race(tab, p2)
    print(f"[save] {p2}")


if __name__ == "__main__":
    main()
