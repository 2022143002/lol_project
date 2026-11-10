"""
tci_model.py
---------------------------------------------------------------
[3장] 라인별 TCI 모델링  +  [4장] TCI 활용

입력 : data_process_target.py 가 만든 processed_data.pkl
       (라인별 표 : 블루 관점 피처 x, 타겟 y, 승패, 챌린저 여부)

1) 학습 데이터 구성
   - 한 행 = (매치, 라인, 진영, t분)
   - 입력 x = 해당 라인의 누적 격차 피처 5개 + 경과 시간 t
   - 출력 y = 2.1절의 정규화 타겟
   - 0분(모두 0)과 마지막 프레임(1분 미만 잔여 구간)은 제외
   - 블루/레드 관점 행을 모두 사용하되, 같은 매치는 항상 같은 fold 에 넣는다

2) 모델 비교 (손실함수 = MSE)
   선형 : LinearRegression, Ridge, Lasso
   트리 : DecisionTree, RandomForest, XGBoost, LightGBM
   신경망 : MLP
   평가 : 매치 단위 5-fold 교차검증(GroupKFold) 의 out-of-fold 예측으로 MSE, RMSE, MAE, R2

3) 최종 모델 해석 : 순열 중요도(permutation importance)

4) TCI 활용
   - TCI_t = 최종 모델의 out-of-fold 예측값 (학습에 쓰지 않은 매치에 대한 예측)
   - 소환사별 평균 TCI : 챌린저 vs 비챌린저 비교, 승리/패배 경기 비교
   - 경기별 TCI 곡선 시각화
"""
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import mannwhitneyu

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
SEED = 42
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
LANE_KR = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "BOTTOM": "BOTTOM"}
LANE_COLOR = {"TOP": "#2a78d6", "JUNGLE": "#eb6834", "MIDDLE": "#1baf7a", "BOTTOM": "#6250d6"}
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

# ---- 그림 공통 설정 (Windows: 맑은 고딕 / Linux: Noto Sans CJK) ----
for f in font_manager.findSystemFonts():
    low = f.lower()
    if "malgun" in low or "notosanscjk" in low or "nanumgothic" in low:
        font_manager.fontManager.addfont(f)
available = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if name in available:
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "axes.edgecolor": "#8a8984"})


# =====================================================================
# 1) 학습 데이터 구성
# =====================================================================
def build_table(tables):
    """블루 관점 표 → 블루 + 레드(부호 반전) 행. 0분과 마지막 프레임은 제외"""
    out = []
    for lane in LANES:
        df = tables[lane]
        df = df[(df.minute > 0) & (df.isLast == 0)]
        feats = feature_names(tables, lane)
        base = df[["matchId", "minute"] + feats[:-1] + ["y"]]
        blue = base.assign(lane=lane, side="blue", win=df.blueWin.values, challenger=df.blueChallenger.values)
        red = base.copy()
        red[feats[:-1] + ["y"]] = -red[feats[:-1] + ["y"]]
        red = red.assign(lane=lane, side="red", win=1 - df.blueWin.values, challenger=df.redChallenger.values)
        out += [blue, red]
    return pd.concat(out, ignore_index=True)


def feature_names(tables, lane):
    df = tables[lane]
    return [c for c in df.columns if c.endswith("Diff") and not c.startswith("T_")] + ["minute"]


# =====================================================================
# 2) 모델 후보 + 하이퍼파라미터 튜닝
# ---------------------------------------------------------------------
# 모델마다 성능에 큰 영향을 주는 설정값의 후보(GRIDS)를 정해 두고,
# 튜닝용 표본(TUNE_MATCHES 경기)에서 매치 단위 3-fold 교차검증으로 RMSE 가 가장 낮은 값을 고른다.
# 고른 설정으로 전체 데이터 5-fold 교차검증을 수행해 모델을 비교한다.
# =====================================================================
TUNE_MATCHES = 3000
TUNE_FOLDS = 3

GRIDS = {
    "LinearRegression": [{}],
    "Ridge": [{"alpha": a} for a in [0.1, 1, 10, 100]],
    "Lasso": [{"alpha": a} for a in [0.0001, 0.001, 0.01, 0.1]],
    "DecisionTree": [{"max_depth": d} for d in [4, 6, 8, 10, 12]],
    "RandomForest": [{"max_depth": d} for d in [8, 12, 16]],
    "XGBoost": [{"max_depth": d, "learning_rate": lr} for d in [3, 5, 7] for lr in [0.05, 0.1]],
    "LightGBM": [{"num_leaves": n, "learning_rate": lr} for n in [15, 31, 63] for lr in [0.05, 0.1]],
    "MLP": [{"hidden_layer_sizes": h, "alpha": a}
            for h in [(16,), (32, 16), (64, 32), (64, 32, 16)] for a in [1e-4, 1e-3]],
}
BEST = {}   # BEST[lane][model] = 튜닝으로 고른 설정


def build_model(name, p):
    if name == "LinearRegression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "Lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=p["alpha"], max_iter=10000))
    if name == "DecisionTree":
        return DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=20, random_state=SEED)
    if name == "RandomForest":
        # 데이터가 많아 각 나무는 10% 표본으로 학습 (학습 시간 단축)
        return RandomForestRegressor(n_estimators=100, max_depth=p["max_depth"], min_samples_leaf=20,
                                     max_samples=0.1, random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=300, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                            random_state=SEED, n_jobs=-1, verbosity=0)
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=300, num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                             random_state=SEED, verbose=-1)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=p["hidden_layer_sizes"], alpha=p["alpha"], batch_size=1024,
            learning_rate_init=1e-3, max_iter=200, early_stopping=True, random_state=SEED))
    raise ValueError(name)


def get_models(lane):
    return {name: build_model(name, BEST[lane][name]) for name in GRIDS}


def tune(df, cols, lane):
    """튜닝용 표본 경기에서 모델별 최적 설정 선택. 반환: 튜닝 결과 행 리스트"""
    rng = np.random.default_rng(SEED)
    ids = df["matchId"].unique()
    pick = set(rng.choice(ids, min(TUNE_MATCHES, len(ids)), replace=False))
    sub = df[df["matchId"].isin(pick)]
    X, y, groups = sub[cols].values, sub["y"].values, sub["matchId"].values
    folds = list(GroupKFold(n_splits=TUNE_FOLDS).split(X, y, groups))
    BEST[lane], rows = {}, []
    for name, grid in GRIDS.items():
        t0, best = time.time(), None
        for p in grid:
            pred = np.zeros(len(sub))
            for tr, te in folds:
                pred[te] = build_model(name, p).fit(X[tr], y[tr]).predict(X[te])
            rmse = np.sqrt(mean_squared_error(y, pred))
            rows.append({"lane": lane, "model": name, "params": str(p), "RMSE": rmse})
            if best is None or rmse < best[1]:
                best = (p, rmse)
        BEST[lane][name] = best[0]
        print(f"    {name:<16} 최적 {best[0]}  RMSE={best[1]:.4f}  ({time.time() - t0:.0f}초)")
    return rows


def cross_validate(df, cols, lane, n_splits=5):
    """매치 단위 GroupKFold. 반환: 모델별 out-of-fold 예측"""
    X, y, groups = df[cols].values, df["y"].values, df["matchId"].values
    oof = {name: np.zeros(len(df)) for name in GRIDS}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), 1):
        print(f"  fold {k}/{n_splits}")
        for name, model in get_models(lane).items():
            t0 = time.time()
            model.fit(X[tr], y[tr])
            oof[name][te] = model.predict(X[te])
            print(f"    {name:<16} {time.time() - t0:6.1f}초")
    return oof


def scores(y, pred):
    mse = mean_squared_error(y, pred)
    return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred)}


# =====================================================================
# main
# =====================================================================
def main():
    tables = pd.read_pickle("processed_data.pkl")["tables"]
    table = build_table(tables)
    print(f"매치 {table.matchId.nunique():,}건 / 학습 행 {len(table):,}개 (라인당 {len(table)//4:,}개)")

    # ---------- 2) 모델 비교 ----------
    result_rows, oof_all, tune_rows = [], {}, []
    for lane in LANES:
        df = table[table.lane == lane].reset_index(drop=True)
        cols = feature_names(tables, lane)
        print(f"\n[{lane}] 하이퍼파라미터 튜닝 중... (표본 {TUNE_MATCHES:,}경기, {TUNE_FOLDS}-fold)")
        tune_rows += tune(df, cols, lane)
        print(f"[{lane}] 전체 데이터 교차검증 중...")
        oof = cross_validate(df, cols, lane)
        oof_all[lane] = (df, cols, oof)
        for name, pred in oof.items():
            result_rows.append({"lane": lane, "model": name, **scores(df["y"].values, pred)})
    res = pd.DataFrame(result_rows)
    res.to_csv("model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tune_rows).to_csv("tuning_result.csv", index=False, encoding="utf-8-sig")
    print("\n[튜닝으로 선택한 설정]")
    for name in GRIDS:
        print(f"  {name:<16} " + " | ".join(f"{lane}: {BEST[lane][name]}" for lane in LANES))

    model_order = list(GRIDS)
    for metric in ["RMSE", "R2"]:
        tab = res.pivot(index="model", columns="lane", values=metric).loc[model_order, LANES]
        tab["평균"] = tab.mean(axis=1)
        print(f"\n[모델별 {metric}]  (매치 단위 5-fold 교차검증)")
        print(tab.round(3).to_string())

    rmse_mean = res.groupby("model")["RMSE"].mean().sort_values()
    final = rmse_mean.index[0]
    print(f"\n최종 선정 모델 (평균 RMSE 최소): {final}")

    # 그림 1: 모델별 RMSE
    fig, ax = plt.subplots(figsize=(11, 4.2))
    order = rmse_mean.index.tolist()
    xs, w = np.arange(len(order)), 0.2
    for i, lane in enumerate(LANES):
        vals = res[res.lane == lane].set_index("model").loc[order, "RMSE"]
        ax.bar(xs + (i - 1.5) * w, vals, width=w - 0.02, color=LANE_COLOR[lane], label=lane)
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("RMSE (낮을수록 정확)"); ax.grid(axis="x", visible=False)
    ax.set_title("라인별 모델 성능 비교 (매치 단위 5-fold 교차검증)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_model_rmse.png", dpi=150); plt.close(fig)

    # 그림 2: 최종 모델 실제 vs 예측
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, lane in zip(axes, LANES):
        df, cols, oof = oof_all[lane]
        p = oof[final]
        r2 = r2_score(df["y"], p)
        idx = np.random.default_rng(SEED).choice(len(df), min(len(df), 20000), replace=False)  # 점이 많아 2만 개만 표시
        ax.scatter(df["y"].values[idx], p[idx], s=6, alpha=0.35, color=LANE_COLOR[lane], edgecolors="none")
        lim = max(abs(df["y"]).max(), abs(p).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#8a8984", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{lane}  (R²={r2:.2f})")
        ax.set_xlabel("실제 y"); ax.set_ylabel("예측 TCI")
    fig.suptitle(f"실제 타겟 vs 예측값 - {final}", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_pred_vs_true.png", dpi=150); plt.close(fig)

    # ---------- 3) 최종 모델 해석 : 순열 중요도 ----------
    imp_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, lane in zip(axes, LANES):
        df, cols, _ = oof_all[lane]
        model = get_models(lane)[final].fit(df[cols].values, df["y"].values)
        if hasattr(model, "steps") and hasattr(model[-1], "coef_"):   # 선형 모델이면 표준화 계수 출력
            print(f"  {lane} 표준화 계수: " + ", ".join(f"{c}={w:+.3f}" for c, w in zip(cols, model[-1].coef_)))
        sample = df.sample(min(len(df), 200000), random_state=SEED)      # 계산 시간 단축용 표본
        pi = permutation_importance(model, sample[cols].values, sample["y"].values, n_repeats=10,
                                    random_state=SEED, scoring="neg_root_mean_squared_error")
        imp = pd.Series(pi.importances_mean, index=cols).sort_values()
        for k, v in imp.items():
            imp_rows.append({"lane": lane, "feature": k, "importance": v})
        ax.barh(imp.index, imp.values, color=LANE_COLOR[lane], height=0.6)
        ax.set_title(lane); ax.set_xlabel("섞었을 때 RMSE 증가량"); ax.grid(axis="y", visible=False)
    fig.suptitle(f"피처 순열 중요도 - {final}", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_importance.png", dpi=150); plt.close(fig)
    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv("feature_importance.csv", index=False, encoding="utf-8-sig")
    print("\n[순열 중요도 (RMSE 증가량)]")
    for lane in LANES:
        s = imp_df[imp_df.lane == lane].sort_values("importance", ascending=False)
        print(f"  {lane:<7}: " + ", ".join(f"{r.feature}={r.importance:.3f}" for r in s.itertuples()))

    # ---------- 4) TCI 활용 ----------
    tci = []
    for lane in LANES:
        df, cols, oof = oof_all[lane]
        df = df.copy()
        df["TCI"] = oof[final]
        tci.append(df)
    tci = pd.concat(tci)
    tci.to_csv("tci_result.csv", index=False, encoding="utf-8-sig")

    # (a) TCI 의 승패 예측력 (15분 / 종료 직전) - 라인, 팀
    print("\n[TCI 부호의 승패 일치율 (블루 관점)]")
    blue = tci[tci.side == "blue"]
    last = blue.groupby(["matchId", "lane"]).minute.transform("max") == blue.minute
    rows = []
    for label, sub in [("10분", blue[blue.minute == 10]), ("15분", blue[blue.minute == 15]),
                       ("20분", blue[blue.minute == 20]), ("종료 직전", blue[last])]:
        r = {"시점": label}
        for lane in LANES:
            s = sub[sub.lane == lane]
            r[lane] = ((s.TCI > 0) == (s.win == 1)).mean()
        team = sub.groupby("matchId").agg(TCI=("TCI", "sum"), win=("win", "first"))
        r["TEAM"] = ((team.TCI > 0) == (team.win == 1)).mean()
        rows.append(r)
    win_tab = pd.DataFrame(rows).set_index("시점")
    win_tab.to_csv("tci_win_accuracy.csv", encoding="utf-8-sig")
    print(win_tab.round(3).to_string())

    # (b) 소환사(라인 x 진영 x 매치)별 평균 TCI
    player = tci.groupby(["matchId", "lane", "side"]).agg(
        meanTCI=("TCI", "mean"), win=("win", "first"), challenger=("challenger", "first")).reset_index()
    player.to_csv("tci_player.csv", index=False, encoding="utf-8-sig")

    print("\n[챌린저 vs 비챌린저 평균 TCI]")
    comp = player.groupby(["lane", "challenger"]).meanTCI.agg(["mean", "count"]).unstack("challenger")
    comp.columns = [f"{a}_{'챌린저' if b else '비챌린저'}" for a, b in comp.columns]
    comp = comp.reindex(LANES)
    print(comp.round(3).to_string())
    allc = player.groupby("challenger").meanTCI.agg(["mean", "std", "count"])
    print(allc.round(3).to_string())
    u = mannwhitneyu(player[player.challenger == 1].meanTCI, player[player.challenger == 0].meanTCI)
    print(f"맨-휘트니 U 검정 : U={u.statistic:.1f}, p={u.pvalue:.4f}")
    comp.to_csv("tci_challenger.csv", encoding="utf-8-sig")

    print("\n[승리 vs 패배 경기 평균 TCI]")
    wl = player.groupby(["lane", "win"]).meanTCI.mean().unstack("win").reindex(LANES)
    wl.columns = ["패배", "승리"]
    print(wl.round(3).to_string())
    lost_pos = player[(player.win == 0) & (player.meanTCI > 0)]
    print(f"패배했지만 평균 TCI > 0 인 소환사 : {len(lost_pos)} / {int((player.win == 0).sum())}")

    # 그림 4: 챌린저 vs 비챌린저 분포
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, lane in zip(axes, LANES):
        s = player[player.lane == lane]
        data = [s[s.challenger == 0].meanTCI, s[s.challenger == 1].meanTCI]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops={"color": "#0b0b0b"})
        for patch, c in zip(bp["boxes"], ["#c3c2b7", LANE_COLOR[lane]]):
            patch.set_facecolor(c)
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"비챌린저\n(n={len(data[0])})", f"챌린저\n(n={len(data[1])})"])
        ax.axhline(0, color="#8a8984", lw=1); ax.set_title(lane); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("경기 평균 TCI")
    fig.suptitle("챌린저 vs 비챌린저 경기 평균 TCI", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_challenger.png", dpi=150); plt.close(fig)

    # 그림 5: 경기 하나의 라인별 TCI 곡선 (블루 관점)
    mid = blue.matchId.iloc[0]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, lane in zip(axes, LANES):
        s = blue[(blue.matchId == mid) & (blue.lane == lane)]
        ax.axhline(0, color="#8a8984", lw=1)
        ax.plot(s.minute, s.y, color="#8a8984", lw=2, ls="--", label="실제 y")
        ax.plot(s.minute, s.TCI, color=LANE_COLOR[lane], lw=2, label="TCI (예측)")
        ax.set_title(lane); ax.set_xlabel("경기 시간(분)")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("블루 관점 (+ 우세)")
    result = "블루 승" if blue[blue.matchId == mid].win.iloc[0] else "레드 승"
    fig.suptitle(f"매치 {mid} 라인별 TCI 곡선 ({result})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_tci_curve.png", dpi=150); plt.close(fig)

    print(f"\n그림 저장: {FIG}/fig1~fig5")


if __name__ == "__main__":
    main()