"""
make_figures.py
---------------------------------------------------------------
보고서용 그림 5장을 다시 그린다. (모델을 다시 학습하지 않음)

입력 : tci_model_full.py 가 저장한 결과 파일
       model_comparison.csv, feature_importance.csv, tci_result.csv, tci_player.csv
출력 : figures/fig1 ~ fig5 (.png)

디자인 원칙
  - 흰 배경, 회색 계열 + 강조색 1개(남색)만 사용
  - 강조색은 '최종 모델', 'TCI', '챌린저'처럼 보고서에서 말하고 싶은 대상에만 사용
  - 라인 구분은 색이 아니라 패널 제목으로 한다
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
FINAL = "MLP"

# ---- 색 ----
ACCENT = "#1F3A5F"      # 남색 (강조)
ACCENT_LIGHT = "#8FA7C4"
GRAY = "#B8B8B8"        # 비교 대상
GRAY_DARK = "#6B6B6B"   # 보조 선
INK = "#222222"         # 글자
GRID = "#E6E6E6"

# ---- 글꼴 (Windows: 맑은 고딕 / 그 외: Noto Sans CJK, 나눔고딕) ----
for f in font_manager.findSystemFonts():
    low = f.lower()
    if "malgun" in low or "notosanscjk" in low or "nanumgothic" in low:
        font_manager.fontManager.addfont(f)
names = {f.name for f in font_manager.fontManager.ttflist}
for n in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if n in names:
        plt.rcParams["font.family"] = n
        break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY_DARK, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "figure.titlesize": 12.5, "figure.titleweight": "bold",
    "axes.axisbelow": True,   # 격자선을 막대 뒤로
})

MODEL_KR = {"LinearRegression": "선형 회귀", "Ridge": "릿지", "Lasso": "라쏘",
            "DecisionTree": "결정 나무", "RandomForest": "랜덤 포레스트",
            "XGBoost": "XGBoost", "LightGBM": "LightGBM", "MLP": "MLP"}
FEAT_KR = {"minionDiff": "미니언", "dmgDoneDiff": "가한 피해", "dmgTakenDiff": "받은 피해",
           "soloKillDiff": "솔로킬", "deathDiff": "데스", "jungleMinionDiff": "정글 몬스터",
           "epicMonsterDiff": "에픽 몬스터", "wardPlacedDiff": "와드 설치", "plateDiff": "방패 파괴",
           "roamDiff": "로밍 시간", "adcDmgDoneDiff": "ADC 가한 피해", "duoTakedownDiff": "듀오 킬관여",
           "duoDeathDiff": "듀오 데스", "duoWardKillDiff": "듀오 와드 제거", "minute": "경과 시간"}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {FIG}/{name}")


# =====================================================================
# 그림 1 : 라인별 모델 성능 (R²)
# =====================================================================
def fig1():
    res = pd.read_csv("model_comparison.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
    for ax, lane in zip(axes, LANES):
        s = res[res.lane == lane].sort_values("R2")
        colors = [ACCENT if m == FINAL else GRAY for m in s.model]
        ax.barh([MODEL_KR[m] for m in s.model], s.R2, color=colors, height=0.62)
        lo = s.R2.min()
        ax.set_xlim(lo - 0.04, s.R2.max() + 0.025)
        for i, v in enumerate(s.R2):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5,
                    color=ACCENT if s.model.iloc[i] == FINAL else GRAY_DARK,
                    fontweight="bold" if s.model.iloc[i] == FINAL else "normal")
        ax.set_title(lane)
        ax.set_xlabel("결정계수 R² (높을수록 정확)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("그림 1. 라인별 모델 성능 비교 (매치 단위 5-fold 교차검증)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig1_model_rmse.png")


# =====================================================================
# 그림 2 : 실제 타겟 vs 예측 TCI (밀도)
# =====================================================================
def fig2(tci):
    cmap = LinearSegmentedColormap.from_list("navy", ["#F4F6F9", ACCENT_LIGHT, ACCENT])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, lane in zip(axes, LANES):
        d = tci[tci.lane == lane]
        y, p = d.y.values, d.TCI.values
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        lim = np.percentile(np.abs(np.r_[y, p]), 99.5) * 1.1
        ax.hexbin(y, p, gridsize=45, extent=(-lim, lim, -lim, lim), cmap=cmap,
                  bins="log", mincnt=1, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color=GRAY_DARK, lw=0.9, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{lane}   R² = {r2:.3f}")
        ax.set_xlabel("실제 타겟 y")
        ax.set_ylabel("예측 TCI")
        ax.grid(False)
    fig.suptitle(f"그림 2. 실제 타겟과 예측 TCI의 비교 ({FINAL}, 학습에 쓰지 않은 경기 기준 · 색이 진할수록 데이터가 많음)",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig2_pred_vs_true.png")


# =====================================================================
# 그림 3 : 피처 순열 중요도
# =====================================================================
def fig3():
    imp = pd.read_csv("feature_importance.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, lane in zip(axes, LANES):
        s = imp[imp.lane == lane].sort_values("importance")
        top = s.importance.max()
        colors = [ACCENT if v >= 0.25 * top else GRAY for v in s.importance]
        ax.barh([FEAT_KR.get(f, f) for f in s.feature], s.importance, color=colors, height=0.62)
        for i, v in enumerate(s.importance):
            ax.text(v + top * 0.02, i, f"{v:.3f}", va="center", fontsize=7.5, color=GRAY_DARK)
        ax.set_xlim(0, top * 1.25)
        ax.set_title(lane)
        ax.set_xlabel("피처를 섞었을 때 RMSE 증가량")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"그림 3. 라인별 피처 중요도 ({FINAL}, 순열 중요도)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig3_importance.png")


# =====================================================================
# 그림 4 : 챌린저 vs 비챌린저 경기 평균 TCI
# =====================================================================
def fig4():
    pl = pd.read_csv("tci_player.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, lane in zip(axes, LANES):
        s = pl[pl.lane == lane]
        a, b = s[s.challenger == 0].meanTCI, s[s.challenger == 1].meanTCI
        bp = ax.boxplot([a, b], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 1.6},
                        whiskerprops={"color": GRAY_DARK}, capprops={"color": GRAY_DARK},
                        boxprops={"linewidth": 0})
        for patch, c in zip(bp["boxes"], [GRAY, ACCENT]):
            patch.set_facecolor(c)
        for x, v, c in [(1, a.mean(), GRAY_DARK), (2, b.mean(), ACCENT)]:
            ax.scatter([x], [v], marker="D", s=22, color="white", edgecolor=c, zorder=3)
            ax.text(x + 0.3, v, f"평균 {v:+.3f}", va="center", fontsize=7.5, color=c)
        d = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
        ax.axhline(0, color=GRAY_DARK, lw=0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"비챌린저\n(n={len(a):,})", f"챌린저\n(n={len(b):,})"])
        ax.set_title(f"{lane}   효과 크기 d = {d:.2f}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("경기 평균 TCI")
    fig.suptitle("그림 4. 챌린저와 비챌린저의 경기 평균 TCI 분포 (상자: 중간 50%, ◇: 평균)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig4_challenger.png")


# =====================================================================
# 그림 5 : 경기 하나의 라인별 TCI 곡선
# =====================================================================
def fig5(tci, match_id=None):
    blue = tci[tci.side == "blue"]
    mid = match_id or blue.matchId.iloc[0]
    one = blue[blue.matchId == mid]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, lane in zip(axes, LANES):
        s = one[one.lane == lane].sort_values("minute")
        ax.axhline(0, color=GRAY_DARK, lw=0.8)
        ax.fill_between(s.minute, 0, s.TCI, where=s.TCI >= 0, color=ACCENT, alpha=0.12, lw=0)
        ax.fill_between(s.minute, 0, s.TCI, where=s.TCI < 0, color=GRAY, alpha=0.35, lw=0)
        ax.plot(s.minute, s.y, color=GRAY_DARK, lw=1.3, ls="--", label="실제 타겟 y")
        ax.plot(s.minute, s.TCI, color=ACCENT, lw=2.0, label="예측 TCI")
        ax.set_title(lane)
        ax.set_xlabel("경기 시간 (분)")
    axes[0].set_ylabel("블루 관점 (+ 우세 / - 열세)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    result = "블루 승리" if one.win.iloc[0] == 1 else "레드 승리"
    fig.suptitle(f"그림 5. 경기 {mid}의 라인별 TCI 곡선 ({result})", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_tci_curve.png")


def main():
    print("결과 파일 읽는 중... (tci_result.csv 가 커서 1~2분 걸릴 수 있음)")
    tci = pd.read_csv("tci_result.csv", usecols=["matchId", "lane", "side", "minute", "win", "y", "TCI"])
    fig1()
    fig2(tci)
    fig3()
    fig4()
    fig5(tci)
    print("완료")


if __name__ == "__main__":
    main()