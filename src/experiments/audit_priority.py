"""D3 우선순위 목록의 공정성 감사 — 목록은 만들되 내보내지 않는다.

확장설계서 §5-5(D3)는 "미해결 사건 중 p_hat 상위 N"을 재수사 우선순위로 정의하고,
§5-6은 그 목록이 **자원 배분 문서**이므로 완화 모델 위에서만 뽑으라는 게이트를 걸었다.
그 게이트에는 검증되지 않은 가정이 하나 들어 있다 — **모델의 격차를 완화하면 그
모델에서 파생된 목록도 따라서 공정해진다**는 것. 이 스크립트가 그 가정을 검정한다.

## 하나의 양으로 둘을 잰다

    lift_g(tau) = P(p_hat >= tau | y=0, g) / P(p_hat >= tau | y=0)

미해결 사건만 놓고, 임계 tau 이상으로 뽑히는 비율의 그룹 대 전체 비. 두 지점이
같은 곡선 위에 있다는 것이 이 감사의 핵심이다:

  - tau = 0.5          -> 이는 그룹별 FPR의 비다. 손실 벌점(mitigate_loss)이
                          누르는 지점이자 fairness_gaps.csv의 fpr_gap이 사는 곳.
  - tau = 상위 q 분위   -> 우선순위 목록이 실제로 뽑는 지점(q=0.005면 상위 0.5%).

벌점은 작동점 근방의 격차를 누르는데 목록은 분포의 **꼬리**를 자른다. 두 지점이
갈리면 "모델은 공정한데 목록은 아니다"가 성립한다. 그래서 곡선으로 본다.

lift는 비율이라 1이 중립이고 스코프·모델 간 비교가 되는 반면, 차(share_top -
share_pop)는 그룹 크기에 끌린다. 다만 **lift는 가치중립이 아니다** — 이 목록의
'공정한' 목표 구성이 미해결 모집단 구성이라는 판단이 들어 있고, 그건 최종보고서
§8-9가 demographic parity에 대해 적어 둔 것과 같은 종류의 **가치 선택**이다.
반대 입장(목록은 해결 가능성 순이어야 하고 구성은 결과일 뿐)도 성립한다.
그래서 이 표는 판정이 아니라 **병기해야 할 진단**이다.

## p_hat 보정이 필요 없는 이유

detect_cold_blocks는 로짓 이동(delta)으로 p_hat을 보정한다 — 기대 미해결 수를
합으로 쓰기 때문이다. 여기서는 **순위만** 쓰고 tau도 분위수로 잡으므로 단조변환에
불변이고, 보정 여부가 결과를 바꾸지 않는다. 예외는 tau=0.5 지점 하나뿐이라
(절대 임계값이므로) 그 행만 덤프의 원래 척도에서 읽는다는 뜻이다 — 덤프의 pred
열, 즉 원장의 지표와 같은 결정 규칙이다.

## 이 스크립트가 하지 않는 것

**사건 단위 목록을 파일로 쓰지 않는다.** 검증 경로가 없고(재수사로 풀렸는지
확인할 열이 데이터에 없다), 모델이 카운티를 특성으로 갖지 않아 "해결 가능성 높은
사건"과 "저검거 카운티의 사건"이 분리되지 않으며, 개별 사건 단위 출력은 이
저장소가 엣지에 대해 이미 내린 결론("단위는 블록이어야 한다")과 같은 이유로
잡음이 지배한다. 감사 통계만 남긴다.

출력: outputs[/{scope}]/priority_audit.csv + priority_audit_lift.png
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from clear import predictions, results

# 기본 대조군. a0 = 완화 없음(blind), a100 = 전국 게이트가 고른 완화 강도.
# 둘은 **같은 test 분할**이라 곡선을 겹쳐 읽을 수 있다.
DEFAULT_MODELS = ["graphsage_fairloss_a0_mb", "graphsage_fairloss_a100_mb"]

# 상위 분위. 0.10은 확장설계서 §5-4의 priority_q 기본값이고, 아래로 내려갈수록
# 실제 재수사 명단에 가깝다(전국 미해결 19만 기준 0.005 = 951건).
DEFAULT_QS = [0.005, 0.01, 0.025, 0.05, 0.10]

FIXED_TAU = 0.5    # 원장/덤프 pred와 같은 결정 규칙. 곡선의 '작동점' 앵커.


def _cells(sub, tau_mask):
    """그룹별 (뽑힘, 안 뽑힘) 2칸 카운트. 그룹 내 복원추출 = 이 2칸 multinomial."""
    top = tau_mask.astype(int)
    return {g: np.array([int(top[sub.values == g].sum()),
                         int((sub.values == g).sum() - top[sub.values == g].sum())],
                        dtype=np.int64)
            for g in pd.unique(sub)}


def _lift(counts_by_group, groups):
    """점추정 lift. counts: {g: [n_top, n_rest]} -> {g: lift}."""
    n_top = sum(counts_by_group[g][0] for g in groups)
    n_all = sum(counts_by_group[g].sum() for g in groups)
    base = n_top / n_all if n_all else np.nan
    out = {}
    for g in groups:
        n_g = counts_by_group[g].sum()
        rate = counts_by_group[g][0] / n_g if n_g else np.nan
        out[g] = rate / base if base else np.nan
    return out, base


def _boot_lift(counts_by_group, groups, n_boot, seed):
    """그룹 내 복원추출 부트스트랩 -> {g: (lo, hi)}.

    clear.fairness와 같은 규약이다: 그룹 크기는 데이터가 정한 것이므로 표집하지
    않고, 그룹 **안**에서만 다시 뽑는다. 칸이 2개뿐이라 이항이지만 multinomial로
    써 두면 칸을 늘릴 때 그대로 확장된다.

    분모(전체 뽑힘률)도 같은 draw에서 다시 계산한다 -- 분자와 분모가 같은 표본을
    공유하므로 상관이 유지되고, 그래야 lift의 CI가 과대해지지 않는다.
    """
    rng = np.random.default_rng(seed)
    draws = {}
    for g in groups:
        c = counts_by_group[g]
        n = int(c.sum())
        draws[g] = (rng.multinomial(n, c / n, size=n_boot) if n > 0
                    else np.zeros((n_boot, 2), dtype=np.int64))
    n_all = sum(int(counts_by_group[g].sum()) for g in groups)
    top_all = sum(draws[g][:, 0] for g in groups)
    base = top_all / n_all
    out = {}
    for g in groups:
        n_g = int(counts_by_group[g].sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            samp = (draws[g][:, 0] / n_g) / base
        lo, hi = np.nanpercentile(samp, [2.5, 97.5])
        out[g] = (float(lo), float(hi))
    return out


def audit_one(dump, attr, qs, min_n, n_boot, seed):
    """한 덤프 × 한 민감속성 -> 긴 형식 표(q 지점마다 그룹별 lift)."""
    uns = dump[dump["y_true"] == 0]
    col = uns[f"sens__{attr}"]
    # Unknown 제외 + 그룹 최소 표본. clear.fairness의 group_set 규약과 같다 --
    # Unknown은 기록 아티팩트이지 인구집단이 아니고, 소수 그룹은 max-min류
    # 통계에서 상향 편향을 만든다.
    named = col[col != C.FAIRNESS_UNKNOWN_LABEL]
    sizes = named.value_counts()
    groups = sorted(sizes[sizes >= min_n].index)
    if len(groups) < 2:
        return pd.DataFrame()
    keep = named.index[named.isin(groups)]
    sub, proba = col.loc[keep], uns["proba"].loc[keep]

    rows = []
    # 작동점(tau=0.5) 앵커 + 상위 분위들. 앵커의 q는 관측된 전체 FPR이다.
    points = [("threshold_0.5", FIXED_TAU, float((proba >= FIXED_TAU).mean()))]
    for q in qs:
        tau = float(np.quantile(proba, 1 - q))
        points.append((f"top_{q:g}", tau, q))

    for label, tau, q in points:
        counts = _cells(sub, (proba >= tau).values)
        lifts, base = _lift(counts, groups)
        cis = _boot_lift(counts, groups, n_boot, seed)
        for g in groups:
            n_g = int(counts[g].sum())
            rows.append({
                "attribute": attr, "point": label, "q": round(q, 6),
                "threshold": round(tau, 6), "group": g,
                "n_unsolved": n_g, "n_selected": int(counts[g][0]),
                "share_pop": n_g / sum(int(counts[x].sum()) for x in groups),
                "share_sel": (counts[g][0] / sum(int(counts[x][0]) for x in groups)
                              if sum(int(counts[x][0]) for x in groups) else np.nan),
                "lift": lifts[g], "lo": cis[g][0], "hi": cis[g][1],
                "significant": not (cis[g][0] <= 1.0 <= cis[g][1]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure. English-only labels: the default matplotlib font has no Hangul glyphs
# (see CLAUDE.md environment notes).
# ---------------------------------------------------------------------------
def figure(tab, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 1.0,
    })

    attrs = [a for a in ("Victim Race", "Victim Sex") if a in set(tab.attribute)]
    models = list(dict.fromkeys(tab.model))
    fig, axes = plt.subplots(1, len(attrs), figsize=(5.6 * len(attrs), 4.8),
                             dpi=200, squeeze=False)
    colors = {m: c for m, c in zip(models, [ORANGE, BLUE, "#7a7a7a"])}

    for ax, attr in zip(axes[0], attrs):
        t = tab[(tab.attribute == attr) & (tab.point != "threshold_0.5")]
        # 가장 크게 기우는 그룹 하나만 그린다 -- 곡선이 5개면 아무것도 안 보인다.
        focus = (t.groupby("group")["lift"].max() - 1).abs().idxmax()
        for m in models:
            s = t[(t.model == m) & (t.group == focus)].sort_values("q")
            ax.plot(s["q"] * 100, s["lift"], "-o", ms=4.5, lw=1.8,
                    color=colors[m], label=m.replace("graphsage_fairloss_", ""))
            ax.fill_between(s["q"] * 100, s["lo"], s["hi"],
                            color=colors[m], alpha=0.13, lw=0)
            # 작동점 앵커: 벌점이 실제로 누르는 지점.
            a = tab[(tab.model == m) & (tab.attribute == attr) &
                    (tab.group == focus) & (tab.point == "threshold_0.5")]
            if len(a):
                ax.axhline(float(a["lift"].iloc[0]), color=colors[m],
                           ls=":", lw=1.3, alpha=0.85)
        ax.axhline(1.0, color=INK2, lw=1.2)
        # 로그 축이라야 상위 0.5%~10%가 고르게 퍼진다. 기본 10^0/10^1 눈금은
        # "목록 크기 몇 %"를 읽을 수 없으므로 실제 q를 눈금으로 쓴다.
        ax.set_xscale("log")
        ticks = sorted({q * 100 for q in t["q"].unique()})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}%" for v in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Priority list size (% of unsolved cases)")
        ax.set_ylabel(f"Share lift of '{focus}' vs. unsolved population")
        ax.set_title(f"{attr} — {focus}", loc="left", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)

    # 제목은 측정된 것만 말한다. 벌점은 Victim Race에만 걸려 있고, 결과가
    # 속성별로 갈렸다 -- 걸린 쪽은 꼬리까지 평평해지고, 안 걸린 쪽은 작동점만
    # 조금 나아지며 꼬리는 오히려 나빠진다. "완화는 목록을 못 고친다"는 한 줄로
    # 묶으면 왼쪽 패널과 어긋난다.
    fig.suptitle("The penalty reaches the priority list only for the attribute "
                 "it targets\npenalty is on Victim Race   ·   solid = list at "
                 "top q%   ·   dotted = same lift at the 0.5 operating point"
                 "   ·   1.0 = proportional",
                 x=0.005, y=0.995, va="top", ha="left", fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="덤프 라벨. 기본은 완화 전(a0) vs 게이트가 고른 완화(a100) "
                         "대조 -- 같은 test 분할이라 곡선을 겹쳐 읽을 수 있다.")
    ap.add_argument("--attrs", nargs="+", default=["Victim Race", "Victim Sex"])
    ap.add_argument("--qs", type=float, nargs="+", default=DEFAULT_QS,
                    help="목록 크기(미해결 대비 비율). 작을수록 실제 명단에 가깝다.")
    ap.add_argument("--min_n", type=int, default=C.FAIRNESS_MIN_GROUP_N,
                    help="그룹 최소 표본. 미해결 사건만 세므로 전체 표본 기준 "
                         "floor보다 실질적으로 엄격하다.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--out", default="priority_audit.csv")
    args = ap.parse_args()

    all_tabs = []
    for model in args.models:
        path = predictions.predictions_dir() / f"{model}.csv"
        if not path.exists():
            avail = ", ".join(sorted(predictions.discover())) or "(없음)"
            raise SystemExit(f"[에러] 덤프 없음: {path}\n  가용: {avail}\n"
                             f"  전국이면: CLEAR_SCOPE=national python -m "
                             f"experiments.mitigate_loss --minibatch --alphas 0 100")
        dump = predictions.load(path)
        n_uns = int((dump["y_true"] == 0).sum())
        print(f"\n=== {model} ===  {len(dump):,}행, 미해결 {n_uns:,}건")
        for attr in args.attrs:
            tab = audit_one(dump, attr, args.qs, args.min_n, args.n_boot,
                            C.RANDOM_STATE)
            if tab.empty:
                print(f"  [{attr}] n>={args.min_n} 그룹이 2개 미만 — 건너뜀")
                continue
            tab.insert(0, "model", model)
            all_tabs.append(tab)

            anchor = tab[tab.point == "threshold_0.5"]
            tight = tab[tab.point == f"top_{min(args.qs):g}"]
            worst = tight.loc[(tight["lift"] - 1).abs().idxmax()]
            a = anchor[anchor.group == worst["group"]].iloc[0]
            print(f"  [{attr}] 최대 편향 그룹 '{worst['group']}': "
                  f"작동점(0.5) lift {a['lift']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}] "
                  f"-> 상위 {min(args.qs)*100:g}% lift {worst['lift']:.2f} "
                  f"[{worst['lo']:.2f}, {worst['hi']:.2f}]")
            print(tab[tab.point.isin(["threshold_0.5", f"top_{min(args.qs):g}"])]
                  [["point", "group", "n_unsolved", "share_pop", "share_sel",
                    "lift", "lo", "hi"]].round(3).to_string(index=False))

            # 원장에는 그룹별 lift와 그 실행의 최대 편향을 남긴다. group은 스키마에
            # 열이 없으므로 params에 넣는다 -- identity의 일부라 재실행이 교체가 된다.
            rows = []
            for _, r in tab.iterrows():
                rows.append(results.rows(
                    "priority_audit", {"priority_lift": float(r["lift"])},
                    model=model, tag=r["point"], attribute=attr, blind=True,
                    group_set=f"n>={args.min_n}",
                    params={"group": r["group"], "q": float(r["q"]),
                            "n_boot": args.n_boot},
                    ci={"priority_lift": (float(r["lo"]), float(r["hi"]))},
                    notes={"threshold": float(r["threshold"]),
                           "share_pop": round(float(r["share_pop"]), 4),
                           "share_sel": round(float(r["share_sel"]), 4)})[0])
            results.write(rows)

    if not all_tabs:
        raise SystemExit("[에러] 표가 비었다.")
    out_tab = pd.concat(all_tabs, ignore_index=True)
    out = C.scoped_output(args.out)
    out_tab.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[save] {out}")

    if not args.no_figure:
        fig_path = figure(out_tab, C.scoped_output(
            args.out.replace(".csv", "_lift.png")))
        print(f"[save] {fig_path}")

    print("\n[해석] lift = (그룹의 목록 내 비중) / (그룹의 미해결 모집단 내 비중). "
          "1이면 비례, >1이면 그 그룹이 우선순위 목록에 과대대표된다. "
          "작동점(0.5)과 꼬리(상위 q%)가 갈리면 '모델은 완화됐는데 목록은 아니다'가 "
          "성립한다 -- 손실 벌점은 작동점 근방의 격차를 누르고 목록은 꼬리를 자르기 "
          "때문이다. **이 표는 판정이 아니라 병기 자료다**: 목록의 '공정한' 구성이 "
          "미해결 모집단 구성이라는 것은 가치 선택이고(최종보고서 §8-9와 같은 종류), "
          "해결 가능성 순 정렬의 결과로 보는 반대 입장도 성립한다. "
          "개별 사건 목록은 검증 경로가 없어 산출물로 만들지 않는다.")


if __name__ == "__main__":
    main()
