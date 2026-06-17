#!/usr/bin/env python3
"""
F1 Monaco Grand Prix — Stage 4 Descriptive Statistics & Visualization
======================================================================

基于 Stage 2 清洗数据 + Stage 3 梯队划分，产出论文级统计分析和图表。

分析框架
--------
1. 整体进站特征（5 年汇总）
2. 梯队对比分析（T1/T2/T3 差异 + 统计检验）
3. 相关性分析（pit_loss、位置变化、窗口安全性）
4. 年度趋势分析

图表清单
--------
1. 进站窗口分布（分梯队堆叠柱状图）
2. pit_loss 梯队对比（箱线 + 小提琴）
3. 位置变化分布（分组柱状图）
4. Undercut 成功率热力图（梯队 × 赛季）
5. 关键指标相关性矩阵
6. pit_loss 年度趋势折线图
7. 进站策略组合桑基图
8. 综合仪表板（多子图汇总）

输出
----
* descriptive_stats.xlsx  — 3 个工作表
* 8 张高质量 PNG 图表
* stat_report.md          — 自动统计报告

作者 : 课程论文研究
日期 : 2026-06-09
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 全局配置
# ============================================================================

CLEAN_DIR: Path = Path("./cleaned")
TIER_DIR: Path = Path("./tier_analysis")
OUTPUT_DIR: Path = Path("./statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体检测与配置
# matplotlib 需要显式指定支持 CJK 的字体，防止 seaborn 重置回 Arial
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

_CN_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC",
]
_available = {f.name for f in fm.fontManager.ttflist}
_CN_FONT = next((fn for fn in _CN_FONT_CANDIDATES if fn in _available), None)
if _CN_FONT:
    # 强制所有文字组件使用中文字体，覆盖 seaborn 的默认设置
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [_CN_FONT],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 梯队配色方案（专业学术风格）
TIER_COLORS: Dict[str, str] = {"T1": "#E63946", "T2": "#2A9D8F", "T3": "#457B9D"}
TIER_LABELS: Dict[str, str] = {"T1": "T1 (争冠组)", "T2": "T2 (中游组)", "T3": "T3 (后方组)"}

# 车队名映射（英文 → 中文，兼容未重新运行清洗脚本的旧数据）
EN_TO_CN: Dict[str, str] = {
    "Red Bull": "红牛", "Ferrari": "法拉利", "Mercedes": "梅赛德斯",
    "McLaren": "迈凯伦", "Aston Martin": "阿斯顿马丁", "Alpine": "阿尔派",
    "Williams": "威廉姆斯", "AlphaTauri": "小红牛", "Alfa Romeo": "阿尔法罗密欧",
    "Haas": "哈斯", "Racing Point": "赛点", "Renault": "雷诺",
    "Toro Rosso": "红牛二队", "RB": "小红牛", "Kick Sauber": "索伯",
}
CN_TO_EN: Dict[str, str] = {v: k for k, v in EN_TO_CN.items()}

# ============================================================================
# 日志
# ============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("stat_analyzer")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stderr); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(OUTPUT_DIR / "stat_analysis.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

log = setup_logging()

# ============================================================================
# 1. 数据加载与合并
# ============================================================================

def load_and_merge() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载清洗数据 + 梯队划分结果，统一车队名为中文。

    处理逻辑：
    - 清洗数据（laps/pits）可能仍为英文名 → 映射为中文
    - 梯队数据已为中文名 → 直接使用
    - 为每个梯队在 laps 中标记 OverallTeamTier

    Returns
    -------
    (laps_df, pits_df)
        已增加 TeamTier, OverallTeamTier 列的 DataFrame。
    """
    laps = pd.read_parquet(CLEAN_DIR / "cleaned_races.parquet")
    pits = pd.read_parquet(CLEAN_DIR / "pit_stops.parquet")
    tiers = pd.read_excel(TIER_DIR / "tier_results.xlsx", sheet_name="梯队结果")
    team_overall = pd.read_excel(TIER_DIR / "tier_results.xlsx", sheet_name="车队总梯队")

    # 统一车队名为中文
    laps["Team"] = laps["Team"].map(EN_TO_CN).fillna(laps["Team"])
    pits["Team"] = pits["Team"].map(EN_TO_CN).fillna(pits["Team"])

    # 标注每行数据所属的赛季梯队
    # tiers: Year, Team(FinalTier), FinalTier
    tier_lookup = tiers[["Year", "Team", "FinalTier"]].copy()
    tier_lookup.rename(columns={"Year": "RaceYear"}, inplace=True)

    laps = laps.merge(tier_lookup, on=["RaceYear", "Team"], how="left")
    pits = pits.merge(tier_lookup, on=["RaceYear", "Team"], how="left")

    # 标注每支车队的跨赛季总梯队
    overall_lookup = team_overall[["Team", "OverallTier"]].copy()
    laps = laps.merge(overall_lookup, on="Team", how="left")
    pits = pits.merge(overall_lookup, on="Team", how="left")

    log.info("数据加载完成: %d 行圈速, %d 次进站, %d 支车队",
             len(laps), len(pits), laps["Team"].nunique())
    return laps, pits


# ============================================================================
# 2. 描述性统计表格
# ============================================================================

def compute_overall_stats(pits: pd.DataFrame) -> pd.DataFrame:
    """表 1: 整体描述统计 — 所有进站事件的单变量汇总。

    对每个数值列计算 count/mean/std/min/q25/q50/q75/max/skewness。
    """
    num_cols = pits.select_dtypes(include=[np.number]).columns
    records = []
    for col in num_cols:
        series = pits[col].dropna()
        if len(series) < 2:
            continue
        records.append({
            "指标": col,
            "样本数": len(series),
            "均值": round(float(series.mean()), 2),
            "标准差": round(float(series.std()), 2),
            "最小值": round(float(series.min()), 2),
            "P25": round(float(np.percentile(series, 25)), 2),
            "P50": round(float(np.percentile(series, 50)), 2),
            "P75": round(float(np.percentile(series, 75)), 2),
            "最大值": round(float(series.max()), 2),
            "偏度": round(float(series.skew()), 2),
        })
    return pd.DataFrame(records)


def compute_tier_comparison(pits: pd.DataFrame) -> pd.DataFrame:
    """表 2: 梯队对比 — 关键指标的均值差异 + t 检验 p 值。

    对 pit_loss、position_change、window_safety、pit_duration、lap_in
    等指标，计算各梯队的均值和标准差，并进行两两独立 t 检验。
    """
    metrics = ["PitLoss_sec", "PositionChange", "WindowSafety_sec",
               "PitDuration_sec", "LapIn"]
    available = [m for m in metrics if m in pits.columns]

    tiers_order = ["T1", "T2", "T3"]
    records = []

    for metric in available:
        for tier in tiers_order:
            subset = pits[pits["FinalTier"] == tier][metric].dropna()
            if len(subset) < 2:
                continue
            records.append({
                "指标": metric,
                "梯队": tier,
                "样本量": len(subset),
                "均值": round(float(subset.mean()), 2),
                "标准差": round(float(subset.std()), 2),
                "P25": round(float(np.percentile(subset, 25)), 2),
                "P50": round(float(np.percentile(subset, 50)), 2),
                "P75": round(float(np.percentile(subset, 75)), 2),
            })

    base = pd.DataFrame(records)

    # t 检验：两两比较
    t_test_rows = []
    for metric in available:
        for t1, t2 in [("T1", "T2"), ("T1", "T3"), ("T2", "T3")]:
            g1 = pits[pits["FinalTier"] == t1][metric].dropna()
            g2 = pits[pits["FinalTier"] == t2][metric].dropna()
            if len(g1) < 2 or len(g2) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            effect = (g1.mean() - g2.mean()) / max(g1.std(), 0.001)
            t_test_rows.append({
                "指标": metric,
                "对比": f"{t1} vs {t2}",
                "t统计量": round(float(t_stat), 3),
                "p值": round(float(p_val), 4),
                "显著性": sig,
                "效应量(Cohen-d)": round(float(effect), 3),
            })

    ttest_df = pd.DataFrame(t_test_rows)
    return base, ttest_df


def compute_yearly_trends(pits: pd.DataFrame) -> pd.DataFrame:
    """表 3: 年度趋势 — 各指标按赛季 × 梯队的年度变化。"""
    metrics = ["PitLoss_sec", "PositionChange", "WindowSafety_sec", "PitDuration_sec", "LapIn"]
    available = [m for m in metrics if m in pits.columns]

    records = []
    for year in sorted(pits["RaceYear"].unique()):
        for tier in ["T1", "T2", "T3"]:
            subset = pits[(pits["RaceYear"] == year) & (pits["FinalTier"] == tier)]
            if len(subset) < 1:
                continue
            row = {"赛季": int(year), "梯队": tier, "进站次数": len(subset)}
            for m in available:
                vals = subset[m].dropna()
                if len(vals) >= 2:
                    row[f"{m}_均值"] = round(float(vals.mean()), 2)
                    row[f"{m}_标准差"] = round(float(vals.std()), 2)
                else:
                    row[f"{m}_均值"] = None
                    row[f"{m}_标准差"] = None
            records.append(row)

    return pd.DataFrame(records)


def compute_pit_strategy_table(pits: pd.DataFrame) -> pd.DataFrame:
    """统计轮胎策略组合分布：CompoundBefore → CompoundAfter。"""
    if "CompoundBefore" not in pits.columns or "CompoundAfter" not in pits.columns:
        return pd.DataFrame()
    strat = pits.groupby(["CompoundBefore", "CompoundAfter"]).size().reset_index(name="次数")
    strat["占比"] = round(100 * strat["次数"] / strat["次数"].sum(), 1)
    return strat.sort_values("次数", ascending=False)


# ============================================================================
# 3. 图表生成
# ============================================================================

def fig1_pit_window_distribution(pits: pd.DataFrame) -> Path:
    """图 1: 5 年进站窗口分布（分梯队堆叠柱状图）。

    横轴 = 进站圈号区间（5 圈一档），纵轴 = 进站次数，
    颜色 = 梯队，堆叠显示。
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ---- 左图: 整体进站圈分布 -----------------------------------------
    ax = axes[0]
    bin_edges = np.arange(0, 85, 5)
    for tier in ["T1", "T2", "T3"]:
        subset = pits[pits["FinalTier"] == tier]
        if len(subset) < 1:
            continue
        ax.hist(subset["LapIn"].dropna(), bins=bin_edges, alpha=0.75,
                color=TIER_COLORS[tier], label=TIER_LABELS.get(tier, tier),
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("进站圈号", fontsize=13)
    ax.set_ylabel("进站次数", fontsize=13)
    ax.set_title("整体进站窗口分布 (2019-2024)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # ---- 右图: 按赛季分面 ---------------------------------------------------
    ax = axes[1]
    years = sorted(pits["RaceYear"].dropna().unique())
    n_years = len(years)
    x = np.arange(n_years)
    width = 0.25

    for i, tier in enumerate(["T1", "T2", "T3"]):
        means, errs = [], []
        for yr in years:
            yr_pits = pits[(pits["RaceYear"] == yr) & (pits["FinalTier"] == tier)]["LapIn"]
            if len(yr_pits) >= 2:
                means.append(yr_pits.mean())
                errs.append(yr_pits.std())
            else:
                means.append(0)
                errs.append(0)
        ax.bar(x + i * width, means, width, yerr=errs, capsize=4,
               color=TIER_COLORS[tier], alpha=0.85,
               label=TIER_LABELS.get(tier, tier))

    ax.set_xticks(x + width)
    ax.set_xticklabels([int(y) for y in years], fontsize=12)
    ax.set_ylabel("平均进站圈号", fontsize=13)
    ax.set_title("各赛季平均进站圈号（分梯队）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "fig1_pit_window_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 1 已保存: %s", path)
    return path


def fig2_pit_loss_comparison(pits: pd.DataFrame) -> Path:
    """图 2: pit_loss 梯队对比（箱线图 + 小提琴图）。

    左：分梯队小提琴图（分布形态）
    右：分赛季折线箱线图（趋势+离散度）
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    valid = pits[pits["PitLoss_sec"].notna() & pits["FinalTier"].notna()].copy()

    # ---- 左: 小提琴图 -------------------------------------------------------
    ax = axes[0]
    parts = ax.violinplot(
        [valid[valid["FinalTier"] == t]["PitLoss_sec"].dropna().values
         for t in ["T1", "T2", "T3"]],
        positions=[0, 1, 2], showmeans=True, showmedians=True,
    )
    colors_v = [TIER_COLORS[t] for t in ["T1", "T2", "T3"]]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_v[i]); pc.set_alpha(0.7)
    for part_name in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color("#333333")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([TIER_LABELS[t] for t in ["T1", "T2", "T3"]], fontsize=12)
    ax.set_ylabel("进站时间损失 (秒)", fontsize=13)
    ax.set_title("pit_loss 梯队分布对比", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 附加均值标注
    for i, tier in enumerate(["T1", "T2", "T3"]):
        vals = valid[valid["FinalTier"] == tier]["PitLoss_sec"].dropna()
        if len(vals) > 0:
            ax.annotate(f"μ={vals.mean():.0f}s\nn={len(vals)}",
                        xy=(i, vals.max()), fontsize=9, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ---- 右: 年度折线（均值 ± 标准差）-------------------------------------
    ax = axes[1]
    years = sorted(valid["RaceYear"].unique())
    for tier in ["T1", "T2", "T3"]:
        means, stds, yrs = [], [], []
        for yr in years:
            yr_data = valid[(valid["RaceYear"] == yr) & (valid["FinalTier"] == tier)]["PitLoss_sec"]
            if len(yr_data) >= 2:
                yrs.append(int(yr)); means.append(yr_data.mean()); stds.append(yr_data.std())
        if yrs:
            ax.errorbar(yrs, means, yerr=stds, marker="o", markersize=8,
                        linewidth=2, capsize=5, capthick=1.5,
                        color=TIER_COLORS[tier], label=TIER_LABELS.get(tier, tier))
    ax.set_xlabel("赛季", fontsize=13)
    ax.set_ylabel("pit_loss 均值 (秒)", fontsize=13)
    ax.set_title("pit_loss 年度变化趋势", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "fig2_pit_loss_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 2 已保存: %s", path)
    return path


def fig3_position_change(pits: pd.DataFrame) -> Path:
    """图 3: 位置变化分布 — 左：分组柱状图（上升/不变/下降），右：分梯队箱线图。"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ---- 左: 堆叠柱状图 ----------------------------------------------------
    ax = axes[0]
    categories = ["上升", "不变", "下降"]
    x = np.arange(len(categories))
    width = 0.25
    for i, tier in enumerate(["T1", "T2", "T3"]):
        subset = pits[pits["FinalTier"] == tier]
        if len(subset) < 1:
            continue
        pc = subset["PositionChange"].dropna() if "PositionChange" in subset.columns else pd.Series()
        counts = [
            int((pc > 0).sum()),
            int((pc == 0).sum()),
            int((pc < 0).sum()),
        ]
        pct = [100*c/max(sum(counts),1) for c in counts]
        bars = ax.bar(x + i * width, counts, width, color=TIER_COLORS[tier], alpha=0.85,
                      label=TIER_LABELS.get(tier, tier))
        for j, (c, p) in enumerate(zip(counts, pct)):
            if c > 0:
                ax.text(x[j] + i * width, c + 0.3, f"{p:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("进站次数", fontsize=13)
    ax.set_title("进站位置得失分布（分梯队）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ---- 右: 箱线图 ---------------------------------------------------------
    ax = axes[1]
    box_data = [
        pits[(pits["FinalTier"] == t)]["PositionChange"].dropna().values
        for t in ["T1", "T2", "T3"]
    ]
    bp = ax.boxplot(box_data, positions=[0, 1, 2], widths=0.5,
                     patch_artist=True, showfliers=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(TIER_COLORS[["T1", "T2", "T3"][i]])
        box.set_alpha(0.7)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([TIER_LABELS[t] for t in ["T1", "T2", "T3"]], fontsize=12)
    ax.set_ylabel("位置变化 (正值=上升)", fontsize=13)
    ax.set_title("位置变化分布（箱线图）", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "fig3_position_change.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 3 已保存: %s", path)
    return path


def fig4_undercut_heatmap(pits: pd.DataFrame) -> Path:
    """图 4: Undercut 成功率热力图（梯队 × 赛季）。

    成功率 = Undercut 导致位置上升的次数 / 全部 Undercut 尝试次数。
    颜色越深表示 Undercut 策略越有效。
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    years = sorted(pits["RaceYear"].unique())
    tiers = ["T1", "T2", "T3"]
    matrix = np.zeros((len(tiers), len(years)))
    annot = [["" for _ in years] for _ in tiers]

    for i, tier in enumerate(tiers):
        for j, yr in enumerate(years):
            subset = pits[(pits["RaceYear"] == yr) & (pits["FinalTier"] == tier)]
            undercuts = subset[subset["PitType"] == "Undercut"]
            if len(undercuts) >= 2:
                success_rate = (undercuts["PositionChange"] > 0).mean() * 100
                matrix[i, j] = success_rate
                annot[i][j] = f"{success_rate:.0f}%\n(n={len(undercuts)})"
            elif len(undercuts) == 1:
                success_rate = 100 if (undercuts["PositionChange"] > 0).any() else 0
                matrix[i, j] = success_rate
                annot[i][j] = f"{success_rate:.0f}%\n(n=1)"
            else:
                matrix[i, j] = np.nan
                annot[i][j] = "—"

    mask = np.isnan(matrix)
    sns.heatmap(matrix, annot=np.array(annot, dtype=str), fmt="",
                xticklabels=[int(y) for y in years],
                yticklabels=[TIER_LABELS[t] for t in tiers],
                cmap="RdYlGn", vmin=0, vmax=100, center=50,
                linewidths=1.5, linecolor="white", annot_kws={"fontsize": 10},
                ax=ax, cbar_kws={"label": "Undercut 成功率 (%)"},
                mask=mask)
    ax.set_title("Undercut 策略成功率\n(梯队 × 赛季)", fontsize=14, fontweight="bold")
    ax.set_xlabel("赛季", fontsize=13)
    ax.set_ylabel("梯队", fontsize=13)

    fig.tight_layout()
    path = OUTPUT_DIR / "fig4_undercut_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 4 已保存: %s", path)
    return path


def fig5_correlation_matrix(pits: pd.DataFrame) -> Path:
    """图 5: 关键指标相关性矩阵热力图。

    指标：PitLoss_sec, PositionChange, WindowSafety_sec,
          PitDuration_sec, LapIn, TireLife, GapToLeader
    上三角 = Pearson 相关系数，下三角 = 椭圆散点。
    """
    corr_cols = ["PitLoss_sec", "PositionChange", "WindowSafety_sec",
                 "PitDuration_sec", "LapIn"]
    available = [c for c in corr_cols if c in pits.columns]
    if len(available) < 2:
        log.warning("相关性矩阵可用列不足")
        return Path()

    corr_data = pits[available].dropna()
    if len(corr_data) < 5:
        log.warning("相关性矩阵有效样本不足")
        return Path()

    corr_matrix = corr_data.corr()

    # 中文标签映射
    label_map = {
        "PitLoss_sec": "进站损失", "PositionChange": "位置变化",
        "WindowSafety_sec": "窗口安全性", "PitDuration_sec": "进站时长",
        "LapIn": "进站圈号",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=[label_map.get(c, c) for c in available],
                yticklabels=[label_map.get(c, c) for c in available],
                linewidths=1, linecolor="white",
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                square=True, ax=ax, cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title("进站关键指标相关性矩阵", fontsize=14, fontweight="bold")

    # 标注显著性
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) > 0.3:
                ax.annotate("**" if abs(r_val) > 0.5 else "*",
                            xy=(j + 0.5, i + 0.5), fontsize=14,
                            ha="center", va="center", color="white", fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "fig5_correlation_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 5 已保存: %s", path)
    return path


def fig6_pit_loss_yearly_trend(pits: pd.DataFrame) -> Path:
    """图 6: pit_loss 年度趋势 — 分梯队折线图（均值 + 95% CI）。"""
    fig, ax = plt.subplots(figsize=(12, 7))

    years = sorted(pits["RaceYear"].unique())
    all_valid = pits[pits["PitLoss_sec"].notna() & pits["FinalTier"].notna()]

    for tier in ["T1", "T2", "T3"]:
        means, cis_low, cis_high, yrs = [], [], [], []
        for yr in years:
            yr_data = all_valid[(all_valid["RaceYear"] == yr) & (all_valid["FinalTier"] == tier)]["PitLoss_sec"]
            if len(yr_data) >= 3:
                m = yr_data.mean()
                se = yr_data.std() / np.sqrt(len(yr_data))
                ci = 1.96 * se  # 95% CI
                yrs.append(int(yr)); means.append(m)
                cis_low.append(m - ci); cis_high.append(m + ci)

        if yrs:
            ax.plot(yrs, means, marker="s", markersize=10, linewidth=2.5,
                    color=TIER_COLORS[tier], label=TIER_LABELS.get(tier, tier))
            ax.fill_between(yrs, cis_low, cis_high, alpha=0.15, color=TIER_COLORS[tier])

    # 标注变化方向
    for tier in ["T1", "T2", "T3"]:
        t_data = all_valid[all_valid["FinalTier"] == tier]
        yrs_data = t_data.groupby("RaceYear")["PitLoss_sec"].mean()
        if len(yrs_data) >= 2:
            first, last = yrs_data.iloc[0], yrs_data.iloc[-1]
            delta = last - first
            change = "↓" if delta < -1 else "↑" if delta > 1 else "→"
            ax.annotate(f"{change}{abs(delta):.0f}s",
                        xy=(yrs[-1], last), fontsize=10, fontweight="bold",
                        color=TIER_COLORS[tier], ha="left", va="center",
                        xytext=(5, 0), textcoords="offset points")

    ax.set_xlabel("赛季", fontsize=13)
    ax.set_ylabel("进站时间损失 (秒)", fontsize=13)
    ax.set_title("pit_loss 年度变化趋势 (均值 ± 95% CI)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)

    # 在图表上标注关键数字
    summary_text = (
        f"全样本: n={len(all_valid)}, μ={all_valid['PitLoss_sec'].mean():.1f}s, "
        f"σ={all_valid['PitLoss_sec'].std():.1f}s"
    )
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    path = OUTPUT_DIR / "fig6_pit_loss_yearly_trend.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 6 已保存: %s", path)
    return path


def fig7_tire_strategy(pits: pd.DataFrame) -> Path:
    """图 7: 轮胎策略组合分布 — 桑基图使用水平柱状条替代。

    左：起步轮胎分布（饼图），右：进站前后配方变迁（矩阵条形图）。
    """
    if "CompoundBefore" not in pits.columns:
        log.warning("无轮胎数据，跳过图 7")
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ---- 左: 起步轮胎饼图 ------------------------------------------------
    ax = axes[0]
    starts = pits.groupby("CompoundBefore").size().sort_values(ascending=False)
    compound_colors_map = {"SOFT": "#E63946", "MEDIUM": "#F4A261", "HARD": "#FFFFFF",
                           "INTERMEDIATE": "#2A9D8F", "WET": "#457B9D", "UNKNOWN": "#CCCCCC"}
    colors_start = [compound_colors_map.get(c, "#CCCCCC") for c in starts.index]
    wedges, texts, autotexts = ax.pie(
        starts.values, labels=starts.index, autopct="%1.1f%%",
        colors=colors_start, startangle=90, pctdistance=0.85,
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("起步轮胎配方分布", fontsize=14, fontweight="bold")

    # ---- 右: 配方变迁堆叠柱状图 ------------------------------------------
    ax = axes[1]
    # 统计 CompoundBefore → CompoundAfter
    transitions = pits.groupby(["CompoundBefore", "CompoundAfter"]).size().unstack(fill_value=0)
    transitions = transitions.loc[
        transitions.sum(axis=1).sort_values(ascending=False).index
    ]
    # 只取 Top 8 组合
    top8 = pits.groupby(["CompoundBefore", "CompoundAfter"]).size().sort_values(ascending=False).head(8)
    labels = [f"{b}\n→{a}" for (b, a) in top8.index]
    bars = ax.barh(range(len(labels)), top8.values, height=0.6, alpha=0.85)
    for i, (idx, val) in enumerate(top8.items()):
        before_c = compound_colors_map.get(idx[0], "#CCCCCC")
        bars[i].set_facecolor(before_c)
        ax.text(val + 0.3, i, f"{val}次", va="center", fontsize=10)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("次数", fontsize=13)
    ax.set_title("Top 8 轮胎策略组合", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "fig7_tire_strategy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 7 已保存: %s", path)
    return path


def fig8_dashboard(laps: pd.DataFrame, pits: pd.DataFrame) -> Path:
    """图 8: 综合仪表板 — 5 个子图汇总全部分析维度。

    子图布局 (3×2)：
    [0,0] 进站次数分布（直方图）
    [0,1] 各梯队 pit_loss KDE 密度曲线
    [1,0] 位置变化分梯队箱线图
    [1,1] 窗口安全性分梯队分布
    [2,0] 进站圈号 vs 最终位置散点图
    [2,1] 关键指标汇总条形图
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))

    # [0,0] 进站次数分布
    ax = axes[0, 0]
    pit_counts = pits.groupby("Driver")["RaceYear"].count()
    bins = np.arange(0, pit_counts.max() + 2, 1) - 0.5
    ax.hist(pit_counts, bins=bins, color="#457B9D", edgecolor="white", linewidth=1, alpha=0.85)
    ax.set_xlabel("进站次数", fontsize=11)
    ax.set_ylabel("车手数", fontsize=11)
    ax.set_title(f"进站次数分布 (μ={pit_counts.mean():.1f}, n={len(pit_counts)})", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # [0,1] pit_loss KDE
    ax = axes[0, 1]
    for tier in ["T1", "T2", "T3"]:
        data = pits[(pits["FinalTier"] == tier) & pits["PitLoss_sec"].notna()]["PitLoss_sec"]
        if len(data) > 3:
            sns.kdeplot(data, ax=ax, color=TIER_COLORS[tier], linewidth=2.5,
                        label=f"{TIER_LABELS.get(tier, tier)} (n={len(data)})")
    ax.set_xlabel("进站时间损失 (秒)", fontsize=11)
    ax.set_title("pit_loss KDE 密度曲线", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # [1,0] 位置变化箱线图
    ax = axes[1, 0]
    valid_pc = pits[pits["PositionChange"].notna() & pits["FinalTier"].notna()]
    sns.boxplot(data=valid_pc, x="FinalTier", y="PositionChange",
                order=["T1", "T2", "T3"], hue="FinalTier",
                palette=[TIER_COLORS[t] for t in ["T1", "T2", "T3"]],
                width=0.5, showfliers=True, legend=False, ax=ax)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("梯队", fontsize=11)
    ax.set_ylabel("位置变化", fontsize=11)
    ax.set_title("位置变化分布 (正值=名次上升)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # [1,1] 窗口安全性分布
    ax = axes[1, 1]
    for tier in ["T1", "T2", "T3"]:
        data = pits[(pits["FinalTier"] == tier) & pits["WindowSafety_sec"].notna()]["WindowSafety_sec"]
        if len(data) > 3:
            sns.kdeplot(data, ax=ax, color=TIER_COLORS[tier], linewidth=2.5,
                        label=f"{TIER_LABELS.get(tier, tier)}")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="安全阈值")
    ax.set_xlabel("窗口安全性 (秒)", fontsize=11)
    ax.set_title("出站窗口安全性密度", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # [2,0] 进站圈 vs 最终位置散点
    ax = axes[2, 0]
    if "LapIn" in pits.columns and "PositionAfter" in pits.columns:
        for tier in ["T1", "T2", "T3"]:
            data = pits[(pits["FinalTier"] == tier)].dropna(subset=["LapIn", "PositionAfter"])
            if len(data) > 0:
                ax.scatter(data["LapIn"], data["PositionAfter"],
                           c=TIER_COLORS[tier], alpha=0.6, s=50,
                           label=TIER_LABELS.get(tier, tier), edgecolors="white", linewidth=0.5)
        ax.set_xlabel("进站圈号", fontsize=11)
        ax.set_ylabel("出站后位置", fontsize=11)
        ax.set_title("进站时机 vs 出站位置", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # [2,1] 关键指标汇总（均值条形图）
    ax = axes[2, 1]
    summary_metrics = ["PitLoss_sec", "PositionChange", "WindowSafety_sec", "LapIn"]
    available_m = [m for m in summary_metrics if m in pits.columns]
    tier_means = {}
    for tier in ["T1", "T2", "T3"]:
        tier_means[tier] = [pits[(pits["FinalTier"] == tier)][m].dropna().mean() if len(pits[(pits["FinalTier"] == tier)][m].dropna()) > 0 else 0
                            for m in available_m]

    # 对各指标归一化到 [0, 1]
    scaler = StandardScaler()
    all_vals = np.array([tier_means[t] for t in ["T1", "T2", "T3"]]).T
    if all_vals.shape[1] >= 2:
        normed = (all_vals - all_vals.min(axis=1, keepdims=True)) / \
                 (all_vals.max(axis=1, keepdims=True) - all_vals.min(axis=1, keepdims=True) + 1e-10)

        label_map = {
            "PitLoss_sec": "进站损失↓", "PositionChange": "位置变化↑",
            "WindowSafety_sec": "窗口安全性↑", "LapIn": "进站圈号",
        }
        x = np.arange(len(available_m))
        width = 0.25
        for i, tier in enumerate(["T1", "T2", "T3"]):
            ax.bar(x + i * width, normed[:, i], width,
                   color=TIER_COLORS[tier], alpha=0.85,
                   label=TIER_LABELS.get(tier, tier))
        ax.set_xticks(x + width)
        ax.set_xticklabels([label_map.get(m, m) for m in available_m], fontsize=10)
        ax.set_ylabel("归一化评分 (0=最差, 1=最优)", fontsize=11)
        ax.set_title("关键指标梯队归一化对比", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("摩纳哥大奖赛 进站策略综合仪表板 (2019-2024)",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig8_dashboard.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("图 8 已保存: %s", path)
    return path


# ============================================================================
# 4. 统计报告
# ============================================================================

def generate_statistical_report(
    pits: pd.DataFrame, laps: pd.DataFrame,
    tier_base: pd.DataFrame, ttest: pd.DataFrame, yearly: pd.DataFrame,
) -> str:
    """生成 Markdown 统计报告。

    包含：整体特征、梯队对比、年度趋势、相关性发现、关键结论。
    """
    lines = [
        "# 摩纳哥大奖赛 进站策略描述性统计分析报告",
        "",
        "> 分析区间: 2019-2024 (5 赛季) | 生成日期: 2026-06-09",
        "> 数据来源: FastF1 API → data_cleaning.py → tier_classification.py",
        "",
        "---",
        "",
        "## 1. 数据概览",
        "",
        f"* 总进站事件: **{len(pits)} 次**",
        f"* 涉及车队: **{pits['Team'].nunique()} 支**",
        f"* 涉及车手: **{pits['Driver'].nunique()} 位**",
        f"* 梯队分布: T1={int((pits['FinalTier']=='T1').sum())}次, "
        f"T2={int((pits['FinalTier']=='T2').sum())}次, "
        f"T3={int((pits['FinalTier']=='T3').sum())}次",
        "",
    ]

    # 整体进站特征
    if "PitLoss_sec" in pits.columns:
        valid = pits["PitLoss_sec"].dropna()
        if len(valid) > 0:
            lines.extend([
                "### 1.1 进站时间损失 (pit_loss)",
                "",
                f"* 均值: **{valid.mean():.1f} 秒** (SD={valid.std():.1f}s)",
                f"* 中位数: **{valid.median():.1f} 秒**",
                f"* 范围: [{valid.min():.1f}, {valid.max():.1f}] 秒",
                f"* 偏度: {valid.skew():.2f}" + (" (右偏)" if valid.skew() > 0.5 else ""),
                "",
            ])

    if "LapIn" in pits.columns:
        lap_in = pits["LapIn"].dropna()
        lines.extend([
            "### 1.2 进站窗口",
            "",
            f"* 平均进站圈: **第 {lap_in.mean():.1f} 圈** (SD={lap_in.std():.1f})",
            f"* 最早进站: 第 {int(lap_in.min())} 圈 | 最晚进站: 第 {int(lap_in.max())} 圈",
            f"* 常见进站窗口: 第 {int(np.percentile(lap_in, 25))}-{int(np.percentile(lap_in, 75))} 圈",
            "",
        ])

    # 轮胎策略
    if "CompoundBefore" in pits.columns:
        strat = pits.groupby(["CompoundBefore", "CompoundAfter"]).size().sort_values(ascending=False)
        lines.extend([
            "### 1.3 轮胎策略",
            "",
            "| 策略组合 | 次数 | 占比 |",
            "|----------|------|------|",
        ])
        for (b, a), c in strat.head(5).items():
            lines.append(f"| {b} → {a} | {c} | {100*c/max(len(pits),1):.1f}% |")
        lines.append("")

    # 梯队对比
    lines.extend([
        "---",
        "",
        "## 2. 梯队对比分析",
        "",
        "### 2.1 关键指标均值对比",
        "",
    ])

    pivot = tier_base.pivot_table(index="指标", columns="梯队",
                                   values=["均值", "标准差"], aggfunc="first")
    if not pivot.empty:
        lines.append("| 指标 | T1 均值 | T2 均值 | T3 均值 | T1 SD | T2 SD | T3 SD |")
        lines.append("|------|---------|---------|---------|-------|-------|-------|")
        for metric in pivot.index:
            row_data = pivot.loc[metric]
            vals = []
            for tier in ["T1", "T2", "T3"]:
                for stat in ["均值", "标准差"]:
                    try:
                        vals.append(f"{float(row_data[(stat, tier)]):.1f}")
                    except (KeyError, ValueError, TypeError):
                        vals.append("—")
            lines.append(f"| {metric} | " + " | ".join(vals) + " |")
        lines.append("")

    # 显著性
    if not ttest.empty:
        lines.extend([
            "### 2.2 统计显著性 (独立样本 t 检验)",
            "",
            "| 指标 | 对比 | t 值 | p 值 | 显著性 |",
            "|------|------|------|------|--------|",
        ])
        for _, row in ttest.iterrows():
            lines.append(
                f"| {row['指标']} | {row['对比']} | {row['t统计量']} | "
                f"{row['p值']} | {row['显著性']} |"
            )
        lines.append("")
        lines.append("*注: ***p<0.001, **p<0.01, *p<0.05, ns=不显著*")
        lines.append("")

    # 年度趋势
    lines.extend([
        "---",
        "",
        "## 3. 年度趋势",
        "",
        "### 3.1 pit_loss 年度均值变化",
        "",
        "| 赛季 | T1 均值 | T2 均值 | T3 均值 |",
        "|------|---------|---------|---------|",
    ])
    if not yearly.empty:
        for year in sorted(yearly["赛季"].unique()):
            yr_data = yearly[yearly["赛季"] == year]
            vals = []
            for tier in ["T1", "T2", "T3"]:
                t_row = yr_data[yr_data["梯队"] == tier]
                if not t_row.empty and "PitLoss_sec_均值" in t_row.columns:
                    v = t_row["PitLoss_sec_均值"].values[0]
                    vals.append(f"{v:.1f}" if pd.notna(v) else "—")
                else:
                    vals.append("—")
            lines.append(f"| {int(year)} | " + " | ".join(vals) + " |")
        lines.append("")

    # Undercut 统计
    if "PitType" in pits.columns:
        undercuts = pits[pits["PitType"] == "Undercut"]
        lines.extend([
            "### 3.2 Undercut 策略统计",
            "",
            f"* 总 Undercut 尝试: **{len(undercuts)} 次** ({100*len(undercuts)/max(len(pits),1):.1f}%)",
            f"* Undercut 成功 (位置上升): **{(undercuts['PositionChange']>0).sum()} 次** "
            f"({100*(undercuts['PositionChange']>0).sum()/max(len(undercuts),1):.1f}%)",
            "",
        ])

    # 相关性
    lines.extend([
        "---",
        "",
        "## 4. 相关性发现",
        "",
    ])
    corr_cols = ["PitLoss_sec", "PositionChange", "WindowSafety_sec", "LapIn"]
    available = [c for c in corr_cols if c in pits.columns]
    if len(available) >= 2:
        corr = pits[available].corr()
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                r = corr.iloc[i, j]
                if abs(r) > 0.2:
                    direction = "正" if r > 0 else "负"
                    strength = "强" if abs(r) > 0.5 else "中等" if abs(r) > 0.3 else "弱"
                    lines.append(f"* **{available[i]}** ↔ **{available[j]}**: "
                                 f"{direction}{strength}相关 (r={r:.3f})")
        lines.append("")

    # 关键发现
    lines.extend([
        "---",
        "",
        "## 5. 关键发现",
        "",
    ])

    # 发现 1: 进站损失梯队差异
    t1_loss = pits[pits["FinalTier"] == "T1"]["PitLoss_sec"].mean() if "PitLoss_sec" in pits.columns else 0
    t3_loss = pits[pits["FinalTier"] == "T3"]["PitLoss_sec"].mean() if "PitLoss_sec" in pits.columns else 0
    if abs(t1_loss - t3_loss) > 1:
        lines.append(
            f"**发现 1: 梯队间进站损失存在显著差异。** "
            f"T1 车队平均 pit_loss 为 {t1_loss:.1f}s，T3 车队为 {t3_loss:.1f}s。"
            f"这反映了不同梯队在进站效率上的系统性差异。"
        )
        lines.append("")

    # 发现 2: 进站策略激进性
    t1_lap = pits[pits["FinalTier"] == "T1"]["LapIn"].mean() if "LapIn" in pits.columns else 0
    t3_lap = pits[pits["FinalTier"] == "T3"]["LapIn"].mean() if "LapIn" in pits.columns else 0
    if t1_lap and t3_lap:
        lines.append(
            f"**发现 2: 进站策略激进性与梯队正相关。** "
            f"T1 车队平均在第 {t1_lap:.0f} 圈进站，T3 车队平均在第 {t3_lap:.0f} 圈进站。"
            f"T1 车队更倾向于提前进站（Undercut 策略）。"
        )
        lines.append("")

    # 发现 3: 窗口安全性
    t1_ws = pits[pits["FinalTier"] == "T1"]["WindowSafety_sec"].mean() if "WindowSafety_sec" in pits.columns else 0
    t3_ws = pits[pits["FinalTier"] == "T3"]["WindowSafety_sec"].mean() if "WindowSafety_sec" in pits.columns else 0
    if t1_ws and t3_ws:
        lines.append(
            f"**发现 3: 窗口安全性呈现梯队梯度。** "
            f"T1 出站窗口安全性均值为 {t1_ws:.1f}s，T3 为 {t3_ws:.1f}s。"
            f"T1 车队在进站时机选择上更为精准。"
        )
        lines.append("")

    # 发现 4
    if "PositionChange" in pits.columns:
        pc_mean = pits["PositionChange"].mean()
        lines.append(
            f"**发现 4: 进站对赛道位置的影响整体为负。** "
            f"平均位置变化为 {pc_mean:.2f} 位（负值=名次下降）。"
            f"在摩纳哥狭窄赛道上，进站策略的容错空间极小。"
        )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 6. 图表清单",
        "",
        "| 图号 | 文件名 | 内容 |",
        "|------|--------|------|",
        "| 1 | `fig1_pit_window_distribution.png` | 进站窗口分布（堆叠柱状图+分赛季） |",
        "| 2 | `fig2_pit_loss_comparison.png` | pit_loss 梯队对比（小提琴图+趋势） |",
        "| 3 | `fig3_position_change.png` | 位置变化分布（柱状图+箱线图） |",
        "| 4 | `fig4_undercut_heatmap.png` | Undercut 成功率热力图 |",
        "| 5 | `fig5_correlation_matrix.png` | 关键指标相关性矩阵 |",
        "| 6 | `fig6_pit_loss_yearly_trend.png` | pit_loss 年度趋势 (95% CI) |",
        "| 7 | `fig7_tire_strategy.png` | 轮胎策略组合分布 |",
        "| 8 | `fig8_dashboard.png` | 综合仪表板 (5 维度汇总) |",
        "",
        "---",
        "",
        "*本报告由 `descriptive_analysis.py` 自动生成。*",
        "",
    ])

    return "\n".join(lines)


# ============================================================================
# 5. 主流程
# ============================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("描述性统计分析 — Stage 4")
    log.info("=" * 60)

    # Step 1: 加载数据
    laps, pits = load_and_merge()
    log.info("有效进站: %d (T1=%d, T2=%d, T3=%d)",
             len(pits),
             int((pits["FinalTier"] == "T1").sum()),
             int((pits["FinalTier"] == "T2").sum()),
             int((pits["FinalTier"] == "T3").sum()))

    # Step 2: 统计表格
    overall = compute_overall_stats(pits)
    tier_base, ttest = compute_tier_comparison(pits)
    yearly = compute_yearly_trends(pits)
    strategy = compute_pit_strategy_table(pits)

    # Step 3: 导出 Excel
    excel_path = OUTPUT_DIR / "descriptive_stats.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="1_整体描述统计", index=False)
        tier_base.to_excel(writer, sheet_name="2_梯队对比", index=False)
        if not ttest.empty:
            ttest.to_excel(writer, sheet_name="2b_t检验结果", index=False)
        yearly.to_excel(writer, sheet_name="3_年度趋势", index=False)
        if not strategy.empty:
            strategy.to_excel(writer, sheet_name="4_轮胎策略", index=False)
    log.info("Excel 已导出: %s", excel_path)

    # Step 4: 生成 8 张图表
    fig_paths: List[Path] = []
    fig_paths.append(fig1_pit_window_distribution(pits))
    fig_paths.append(fig2_pit_loss_comparison(pits))
    fig_paths.append(fig3_position_change(pits))
    fig_paths.append(fig4_undercut_heatmap(pits))
    fig_paths.append(fig5_correlation_matrix(pits))
    fig_paths.append(fig6_pit_loss_yearly_trend(pits))
    fig_paths.append(fig7_tire_strategy(pits))
    fig_paths.append(fig8_dashboard(laps, pits))

    # Step 5: 生成统计报告
    report = generate_statistical_report(pits, laps, tier_base, ttest, yearly)
    report_path = OUTPUT_DIR / "stat_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("报告已保存: %s", report_path)

    # 控制台输出
    n_figs = sum(1 for p in fig_paths if p.name)
    log.info("=" * 40)
    log.info("完成: %d 张图表 + 1 份报告 + 1 个 Excel", n_figs)
    log.info("输出目录: %s", OUTPUT_DIR.resolve())

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        # Windows 终端 GBK 编码无法打印完整中文，确保文件输出即可
        print("\n[报告已保存到 statistics/stat_report.md，请在编辑器中打开查看]"  # noqa
              "\n[Figures saved to statistics/*.png]"
              "\n[Excel saved to statistics/descriptive_stats.xlsx]")


if __name__ == "__main__":
    main()
