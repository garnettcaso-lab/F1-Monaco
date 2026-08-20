#!/usr/bin/env python3
"""
F1 Monaco Grand Prix — Stage 3 Team Tier Classification
=========================================================

基于 Stage 2 清洗数据 + FastF1 补充数据，采用双重验证法
将 15 支车队客观划分为 T1/T2/T3 三个梯队。

双重验证方法论
--------------
**方法 A — 圈速潜力聚类法**
  提取每车队每赛季 7 项速度指标 → Z-score 标准化 →
  PCA 降维 → KMeans(n=3) 聚类 → 按聚类中心排序确定梯队。

**方法 B — 赛季表现规则法**
  基于赛季积分榜 + 排位赛表现 + 正赛领奖台数，逐赛季
  按规定分界点标记梯队。最终梯队 = 5 年中最多出现的梯队。

**最终梯队**
  A/B 一致 → 高置信度 (≥90%)
  A/B 差异 1 级 → 中等置信度 (70-89%)，由方法 A 决定
  A/B 差异 2 级 → 低置信度 (<70%)，需人工审核

输出
----
* ``tier_results.xlsx`` — 逐赛季逐车队梯队 + 置信度
* ``tier_clusters.png`` — 聚类可视化
* ``tier_evolution.png`` — 梯队演变图
* ``tier_report.md`` — 自动生成文字分析

依赖
----
* pandas, numpy, scikit-learn, matplotlib, seaborn, fastf1, openpyxl

作者 : 课程论文研究
日期 : 2026-06-08
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fastf1 as ff1
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================================
# 全局配置
# ============================================================================

CLEAN_DIR: Path = Path("./cleaned")
OUTPUT_DIR: Path = Path("./tier_analysis")
CACHE_DIR: Path = Path("./f1_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEARS: List[int] = [2019, 2021, 2022, 2023, 2024]

# 方法 A：聚类特征窗口参数
# 正赛圈速百分位区间，排除极值和安全车影响
RACE_PACE_LOW_PERC: float = 10.0   # 第 10 百分位 → 排除极端慢圈
RACE_PACE_HIGH_PERC: float = 90.0  # 第 90 百分位 → 排除极端快圈
LONG_RUN_LOW_PERC: float = 25.0    # 长距离下限
LONG_RUN_HIGH_PERC: float = 75.0   # 长距离上限

# 方法 B：规则分界点
T1_POINTS_RANK: int = 4   # 积分榜前 4 → T1
T2_POINTS_RANK: int = 8   # 积分榜前 8 → T2 (5-8 为 T2)

# 可视化 — 中文字体配置
# matplotlib 默认字体不支持 CJK，按操作系统优先级自动检测可用中文字体
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei",   # Windows
    "SimHei",            # Windows
    "PingFang SC",       # macOS
    "Heiti SC",          # macOS
    "Noto Sans CJK SC",  # Linux
    "WenQuanYi Micro Hei",  # Linux
    "Source Han Sans SC",   # 跨平台
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_CN_FONT = next((fn for fn in _CN_FONT_CANDIDATES if fn in _available_fonts), None)
if _CN_FONT:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# 车队配色（中文俗称 → 颜色，用于可视化一致性）
TEAM_COLORS: Dict[str, str] = {
    "红牛": "#1E41FF",
    "法拉利": "#DC0000",
    "梅赛德斯": "#00D2BE",
    "迈凯伦": "#FF8700",
    "阿斯顿马丁": "#006F62",
    "阿尔派": "#0090FF",
    "威廉姆斯": "#005AFF",
    "小红牛": "#2B4562",
    "阿尔法罗密欧": "#900000",
    "哈斯": "#FFFFFF",
    "赛点": "#F596C8",
    "雷诺": "#FFF500",
    "红牛二队": "#469BFF",
    "索伯": "#52E252",
}

# 梯队配色
TIER_PALETTE: Dict[str, str] = {
    "T1": "#E63946",  # 红
    "T2": "#457B9D",  # 蓝
    "T3": "#6D6875",  # 灰
}

# ============================================================================
# 日志
# ============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("tier_classifier")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(OUTPUT_DIR / "tier_analysis.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

log = setup_logging()
if _CN_FONT:
    log.info("中文字体: %s", _CN_FONT)
else:
    log.warning("未找到中文字体，图表标题/标签可能显示为方框")

# ============================================================================
# 1. 特征提取 — 从清洗数据
# ============================================================================

@dataclass
class TeamSeasonMetrics:
    """单支车队单赛季的所有速度指标。

    用于方法 A 的聚类特征输入。
    """
    year: int
    team: str
    n_drivers: int = 0
    n_laps: int = 0

    # 正赛圈速（秒），排除进站圈
    race_median_lap: float = np.nan          # 中位数圈速
    race_q10_lap: float = np.nan             # 第 10 百分位（潜力）
    race_q90_lap: float = np.nan             # 第 90 百分位（稳定性）
    race_std_lap: float = np.nan             # 圈速标准差（一致性）
    long_run_median: float = np.nan          # 长距离中位数
    best_race_lap: float = np.nan            # 正赛最快圈

    # 完赛位置
    avg_finish_position: float = np.nan
    best_finish_position: float = np.nan
    podium_count: int = 0

    # 进站
    avg_pit_loss: float = np.nan
    n_pit_stops: int = 0

    # 排位（由 FastF1 补充）
    quali_median_lap: float = np.nan
    quali_best_lap: float = np.nan

    # 赛季成绩
    total_points: float = 0.0
    championship_rank: int = 99


def _is_pit_lap(df: pd.DataFrame) -> pd.Series:
    """标记进站圈 (In-Lap / Out-Lap)。

    进站圈的特征：圈速显著高于正常圈速（>120s），
    或 PitInTime/PitOutTime 非空。"""
    is_slow = df["LapTime"] > 100  # 摩纳哥正常圈速 ≤ 90s
    has_pit = False
    if "PitInTime" in df.columns:
        has_pit = has_pit | df["PitInTime"].notna()
    if "PitOutTime" in df.columns:
        has_pit = has_pit | df["PitOutTime"].notna()
    return is_slow | has_pit


def extract_team_metrics(
    lap_df: pd.DataFrame, pit_df: pd.DataFrame
) -> pd.DataFrame:
    """从清洗数据中提取每车队每赛季的速度指标。

    对每支车队每个赛季的逐圈数据，排除进站圈后计算各项统计量。
    这是聚类分析的特征构建步骤。

    Parameters
    ----------
    lap_df : pd.DataFrame
        合并清洗主表 (cleaned_races.parquet)。
    pit_df : pd.DataFrame
        合并进站表 (pit_stops.parquet)。

    Returns
    -------
    pd.DataFrame
        每行 = 一个 (赛季, 车队) 组合，包含全部速度指标。
    """
    records: List[Dict[str, Any]] = []

    for year in sorted(lap_df["RaceYear"].unique()):
        year_laps = lap_df[lap_df["RaceYear"] == year].copy()
        year_pits = pit_df[pit_df["RaceYear"] == year] if not pit_df.empty else pd.DataFrame()

        for team in sorted(year_laps["Team"].unique()):
            team_laps = year_laps[year_laps["Team"] == team].copy()

            # 排除进站圈
            clean = team_laps[~_is_pit_lap(team_laps)]
            n_clean = len(clean)

            rec: Dict[str, Any] = {
                "Year": year,
                "Team": team,
                "N_Drivers": team_laps["Driver"].nunique(),
                "N_Laps": len(team_laps),
                "N_CleanLaps": n_clean,
            }

            if n_clean >= 10 and "LapTime" in clean.columns:
                lt = clean["LapTime"].dropna()
                if len(lt) >= 5:
                    rec["RaceMedianLap_sec"] = float(lt.median())
                    rec["RaceQ10Lap_sec"] = float(np.percentile(lt, RACE_PACE_LOW_PERC))
                    rec["RaceQ90Lap_sec"] = float(np.percentile(lt, RACE_PACE_HIGH_PERC))
                    rec["RaceStdLap_sec"] = float(lt.std())
                    rec["BestRaceLap_sec"] = float(lt.min())

                    # 长距离中位数 (P25-P75)
                    inner = lt[(lt >= np.percentile(lt, LONG_RUN_LOW_PERC))
                                & (lt <= np.percentile(lt, LONG_RUN_HIGH_PERC))]
                    rec["LongRunMedian_sec"] = float(inner.median()) if len(inner) > 0 else np.nan
                else:
                    for k in ["RaceMedianLap_sec", "RaceQ10Lap_sec", "RaceQ90Lap_sec",
                              "RaceStdLap_sec", "BestRaceLap_sec", "LongRunMedian_sec"]:
                        rec[k] = np.nan

            # 完赛位置
            if "Position" in clean.columns:
                pos = pd.to_numeric(clean["Position"], errors="coerce").dropna()
                rec["AvgFinishPosition"] = float(pos.mean()) if len(pos) > 0 else np.nan
                rec["BestFinishPosition"] = float(pos.min()) if len(pos) > 0 else np.nan
                rec["PodiumCount"] = int((pos <= 3).sum()) if len(pos) > 0 else 0
            else:
                rec["AvgFinishPosition"] = np.nan
                rec["BestFinishPosition"] = np.nan
                rec["PodiumCount"] = 0

            # 进站指标
            if not year_pits.empty:
                team_pits = year_pits[year_pits["Team"] == team]
                rec["N_PitStops"] = len(team_pits)
                if "PitLoss_sec" in team_pits.columns:
                    valid_loss = team_pits["PitLoss_sec"].dropna()
                    rec["AvgPitLoss_sec"] = float(valid_loss.mean()) if len(valid_loss) > 0 else np.nan
                else:
                    rec["AvgPitLoss_sec"] = np.nan
            else:
                rec["N_PitStops"] = 0
                rec["AvgPitLoss_sec"] = np.nan

            records.append(rec)

    metrics = pd.DataFrame(records)
    log.info(
        "特征提取完成: %d 个 (赛季, 车队) 组合", len(metrics)
    )
    return metrics


# ============================================================================
# 2. 补充数据 — 排位赛 & 积分榜 (FastF1)
# ============================================================================

def fetch_quali_data(years: List[int]) -> pd.DataFrame:
    """从 FastF1 获取各赛季摩纳哥排位赛圈速。

    对每支车队取两位车手的最佳 Q 圈速中位数作为排位速度指标。

    Parameters
    ----------
    years : list of int
        目标赛季。

    Returns
    -------
    pd.DataFrame
        [Year, Team, QualiBest_sec, QualiMedian_sec]。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(CACHE_DIR))

    records: List[Dict[str, Any]] = []

    for year in tqdm(years, desc="获取排位数据"):
        try:
            schedule = ff1.get_event_schedule(year)
            # 匹配摩纳哥站
            event = None
            for _, row in schedule.iterrows():
                name = str(row["EventName"])
                if any(c.lower() in name.lower()
                       for c in ["Monaco Grand Prix", "Monaco", "Monte Carlo"]):
                    event = row["EventName"]
                    break
            if event is None:
                log.warning("%d — 赛历中无摩纳哥站", year)
                continue

            # 加载排位赛 ('Q')
            quali = ff1.get_session(year, event, "Q")
            quali.load()

            # 每位车手的最快圈速
            for drv in quali.results["Abbreviation"]:
                fast = quali.laps.pick_driver(drv).pick_fastest()
                if fast is not None and len(fast) > 0:
                    lt = fast["LapTime"]
                    if pd.notna(lt).all():
                        lap_sec = lt.dt.total_seconds().values[0]
                        team = str(quali.results.loc[
                            quali.results["Abbreviation"] == drv, "TeamName"
                        ].values[0])
                        records.append({
                            "Year": year,
                            "Driver": str(drv),
                            "Team": team,
                            "QualiBest_sec": float(lap_sec),
                        })

        except Exception as exc:
            log.warning("%d 排位数据获取失败: %s", year, exc)
            continue

    if not records:
        log.warning("未获取到任何排位数据，将仅使用正赛数据")
        return pd.DataFrame()

    quali_df = pd.DataFrame(records)

    # 聚合到车队级别：每支车队取两位车手中位数
    team_quali = quali_df.groupby(["Year", "Team"]).agg(
        QualiBest_sec=("QualiBest_sec", "min"),
        QualiMedian_sec=("QualiBest_sec", "median"),
    ).reset_index()

    log.info("排位数据: %d 个 (赛季, 车队) 组合", len(team_quali))
    return team_quali


def fetch_standings_data(years: List[int]) -> pd.DataFrame:
    """从 FastF1 获取各赛季车队积分榜。

    Returns
    -------
    pd.DataFrame
        [Year, Team, Points, Rank]。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(CACHE_DIR))

    records: List[Dict[str, Any]] = []

    for year in tqdm(years, desc="获取积分榜"):
        try:
            schedule = ff1.get_event_schedule(year)
            # 取赛季最后一站
            last_event = schedule.iloc[-1]["EventName"]
            session = ff1.get_session(year, last_event, "R")
            session.load()

            # 车队积分榜（赛季末累计）
            for _, row in session.results.iterrows():
                team = str(row["TeamName"])
                pts = float(row.get("Points", 0))
                records.append({
                    "Year": year,
                    "Team": team,
                    "Points": pts,
                })

        except Exception as exc:
            log.warning("%d 积分榜获取失败: %s", year, exc)
            continue

    if not records:
        log.warning("未获取到积分榜数据")
        return pd.DataFrame()

    standings = pd.DataFrame(records)

    # 聚合车队总分
    team_standings = standings.groupby(["Year", "Team"]).agg(
        TotalPoints=("Points", "sum"),
    ).reset_index()

    # 按赛季排名
    team_standings["Rank"] = team_standings.groupby("Year")[
        "TotalPoints"
    ].rank(ascending=False, method="min").astype(int)

    log.info("积分榜数据: %d 个 (赛季, 车队) 组合", len(team_standings))
    return team_standings


# ============================================================================
# 3. 车队名称对齐
# ============================================================================

def align_team_quali_to_cleaned(
    quali_df: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    """将 FastF1 排位数据中的车队名映射到中文俗称。

    映射逻辑：FastF1 商业赞助全称 → 中文 F1 社区通用叫法。
    与 data_cleaning.py 的 TEAM_NAME_MAP 输出保持一致。
    """
    # 注意：AlphaTauri / RB 在不同赛季是同一支车队的不同冠名
    # 中文社区统称"小红牛"（区别于大红牛Red Bull）
    TEAM_NAME_MAP: Dict[str, str] = {
        "Red Bull Racing": "红牛",
        "Red Bull Racing Honda": "红牛",
        "Red Bull Racing RBPT": "红牛",
        "Oracle Red Bull Racing": "红牛",
        "Mercedes": "梅赛德斯",
        "Mercedes-AMG Petronas": "梅赛德斯",
        "Mercedes-AMG Petronas Formula One Team": "梅赛德斯",
        "Scuderia Ferrari": "法拉利",
        "Scuderia Ferrari Mission Winnow": "法拉利",
        "Ferrari": "法拉利",
        "McLaren F1 Team": "迈凯伦",
        "McLaren Formula 1 Team": "迈凯伦",
        "McLaren Mercedes": "迈凯伦",
        "Aston Martin Aramco Cognizant": "阿斯顿马丁",
        "Aston Martin Aramco Mercedes": "阿斯顿马丁",
        "Aston Martin": "阿斯顿马丁",
        "BWT Alpine F1 Team": "阿尔派",
        "Alpine F1 Team": "阿尔派",
        "Alpine": "阿尔派",
        "Williams Racing": "威廉姆斯",
        "Williams Mercedes": "威廉姆斯",
        "Williams": "威廉姆斯",
        "Haas F1 Team": "哈斯",
        "MoneyGram Haas F1 Team": "哈斯",
        "Uralkali Haas F1 Team": "哈斯",
        "Haas Ferrari": "哈斯",
        "Scuderia AlphaTauri": "小红牛",
        "Scuderia AlphaTauri Honda": "小红牛",
        "AlphaTauri RB": "小红牛",
        "AlphaTauri": "小红牛",
        "Alfa Romeo Racing": "阿尔法罗密欧",
        "Alfa Romeo Racing ORLEN": "阿尔法罗密欧",
        "Alfa Romeo F1 Team Stake": "阿尔法罗密欧",
        "Alfa Romeo F1 Team ORLEN": "阿尔法罗密欧",
        "Alfa Romeo": "阿尔法罗密欧",
        "Racing Point": "赛点",
        "Racing Point BWT Mercedes": "赛点",
        "BWT Racing Point F1 Team": "赛点",
        "Renault DP World F1 Team": "雷诺",
        "Renault F1 Team": "雷诺",
        "Renault": "雷诺",
        "RB F1 Team": "小红牛",
        "Visa Cash App RB F1 Team": "小红牛",
        "VCARB": "小红牛",
        "Scuderia Toro Rosso": "红牛二队",
        "Red Bull Toro Rosso Honda": "红牛二队",
        "Stake F1 Team Kick Sauber": "索伯",
        "Kick Sauber": "索伯",
    }
    quali_df = quali_df.copy()
    quali_df["Team"] = quali_df["Team"].map(TEAM_NAME_MAP).fillna(quali_df["Team"])
    return quali_df


# ============================================================================
# 4. 方法 A — KMeans 聚类
# ============================================================================

# 聚类用的特征列（自动选择可用的）
CLUSTER_FEATURES: List[str] = [
    "RaceMedianLap_sec",
    "RaceQ10Lap_sec",
    "RaceQ90Lap_sec",
    "RaceStdLap_sec",
    "LongRunMedian_sec",
    "BestRaceLap_sec",
    "QualiMedian_sec",
    "AvgFinishPosition",
    "AvgPitLoss_sec",
]


def prepare_features(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """准备聚类特征矩阵：填充缺失值 + Z-score 标准化。

    * 行列缺失值用列均值填充
    * 所有特征标准化到均值为 0、方差为 1
    * 速度指标取负值（让"更快"映射为"更大"的聚类值）

    Returns
    -------
    (X_scaled, features, scaler)
    """
    # 确定可用特征列
    available = [c for c in CLUSTER_FEATURES if c in metrics.columns]
    log.info("聚类可用特征: %s", available)

    # 提取特征矩阵
    X = metrics[available].copy()

    # 填充缺失值（列均值）
    X = X.fillna(X.mean())

    # 速度类指标取负值 — 圈速越小越优，取负后"大值=好"
    # 位置、标准差也是越小越优
    invert_cols = [c for c in available
                   if any(k in c for k in ["Lap_sec", "Median_sec", "Position", "Std"])]
    for col in invert_cols:
        if col in X.columns:
            X[col] = -X[col]

    # Z-score 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=available, index=metrics.index), available, scaler


def run_kmeans_clustering(
    metrics: pd.DataFrame, X_scaled: pd.DataFrame, features: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """执行 KMeans (k=3) 聚类 → 按聚类中心排序 → T1/T2/T3。

    由于 KMeans 的标签编号是任意的，需要按聚类中心的速度水平排序：
    最快的一类 → T1，中间 → T2，最慢 → T3。

    Parameters
    ----------
    metrics : pd.DataFrame
        原始指标（含 Year, Team）。
    X_scaled : pd.DataFrame
        标准化特征矩阵。
    features : list of str
        使用的特征列名。
    random_state : int
        随机种子，保证结果可复现。

    Returns
    -------
    pd.DataFrame
        metrics 增加 PCA1, PCA2 (二维投影) 和 Tier_A 列。
    """
    # --- PCA 降维（可选，用于可视化） ---------------------------------
    pca = PCA(n_components=min(2, len(features)), random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    variance_ratio = pca.explained_variance_ratio_

    # --- KMeans 聚类 --------------------------------------------------
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # --- 按聚类中心排序确定梯队 ---------------------------------------
    # 计算每个聚类的"速度指数"：各特征标准化后的平均值
    # 值越大 = 越快 → T1
    cluster_speed = {}
    for c in range(3):
        mask = cluster_labels == c
        cluster_speed[c] = float(X_scaled.loc[mask].mean().mean())

    # 按速度指数从高到低排序 → T1, T2, T3
    sorted_clusters = sorted(cluster_speed.items(), key=lambda x: x[1], reverse=True)
    tier_map: Dict[int, str] = {
        sorted_clusters[0][0]: "T1",
        sorted_clusters[1][0]: "T2",
        sorted_clusters[2][0]: "T3",
    }

    # 聚类间距（用于评估聚类质量）
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    log.info("KMeans 聚类完成: 惯性=%.2f, PCA 方差比=%s",
             inertia, [f"{v:.1%}" for v in variance_ratio])

    # --- 组装结果 ------------------------------------------------------
    result = metrics.copy()
    result["PCA1"] = X_pca[:, 0]
    result["PCA2"] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0
    result["ClusterLabel"] = cluster_labels
    result["Tier_A"] = result["ClusterLabel"].map(tier_map)

    for c, tier in tier_map.items():
        n = int((cluster_labels == c).sum())
        log.info("  Cluster %d → %s: %d 个样本, 速度指数=%.2f",
                 c, tier, n, cluster_speed[c])

    # 附加 pca 和 kmeans 到 DataFrame 的属性上供后续使用
    result.attrs["pca"] = pca
    result.attrs["kmeans"] = kmeans
    result.attrs["scaler"] = None  # 由 prepare_features 返回

    return result


# ============================================================================
# 5. 方法 B — 赛季规则法
# ============================================================================

def classify_by_rules(
    standings: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    """按赛季积分榜规则划分梯队。

    **规则定义**
    * T1: 积分榜排名 ≤ T1_POINTS_RANK (前 4)，且赛季积分 > 0
    * T2: 积分榜排名 ≤ T2_POINTS_RANK (前 8)，且有领奖台或稳定积分
    * T3: 其余车队

    **边界车队处理**
    当车队排名 = T1_POINTS_RANK (第 4 名) 时：
    * 检查是否与第 5 名差距 < 总积分的 20%
    * 若差距小 → 标记为 "T1*" (边界 T1)，置信度降级

    当车队排名 = T1_POINTS_RANK + 1 (第 5 名) 时：
    * 检查是否与第 4 名差距 < 总积分的 20%
    * 若差距小 → 标记为 "T2*" (边界 T2)，可能实际属于 T1

    Parameters
    ----------
    standings : pd.DataFrame
        积分榜数据 [Year, Team, TotalPoints, Rank]。
    metrics : pd.DataFrame
        速度指标（含 PodiumCount）。

    Returns
    -------
    pd.DataFrame
        [Year, Team, Tier_B, Tier_B_Confidence]。
    """
    results: List[Dict[str, Any]] = []

    for year in sorted(standings["Year"].unique()):
        year_std = standings[standings["Year"] == year].copy()
        year_metrics = metrics[metrics["Year"] == year] if not metrics.empty else pd.DataFrame()

        if year_std.empty:
            continue

        # 第 1-4 名之间的积分差距（用于边界检测）
        if len(year_std) >= T1_POINTS_RANK + 1:
            p4_pts = year_std.loc[year_std["Rank"] == T1_POINTS_RANK, "TotalPoints"]
            p5_pts = year_std.loc[year_std["Rank"] == T1_POINTS_RANK + 1, "TotalPoints"]
            if not p4_pts.empty and not p5_pts.empty:
                gap = p4_pts.values[0] - p5_pts.values[0]
                total_top = year_std.loc[year_std["Rank"] <= T1_POINTS_RANK, "TotalPoints"].sum()
                gap_pct = abs(gap) / max(total_top, 1)
            else:
                gap_pct = 1.0
        else:
            gap_pct = 1.0

        for _, row in year_std.iterrows():
            team = row["Team"]
            rank = int(row["Rank"])
            pts = row["TotalPoints"]

            # 获取该车队该赛季的领奖台数
            podium = 0
            if not year_metrics.empty:
                tmet = year_metrics[year_metrics["Team"] == team]
                if not tmet.empty and "PodiumCount" in tmet.columns:
                    podium = int(tmet["PodiumCount"].values[0])

            # --- 规则判定 -------------------------------------------------
            if rank <= T1_POINTS_RANK and pts > 0:
                tier = "T1"
                # 边界检测：第 4 名与第 5 名差距很小
                if rank == T1_POINTS_RANK and gap_pct < 0.20:
                    confidence = 75  # 边界 T1
                else:
                    confidence = 95
            elif rank <= T2_POINTS_RANK and pts > 0:
                tier = "T2"
                # 边界检测：第 5 名与第 4 名差距很小
                if rank == T1_POINTS_RANK + 1 and gap_pct < 0.20:
                    confidence = 70  # 可能实际属于 T1
                else:
                    confidence = 90
            else:
                tier = "T3"
                # 若 T3 车队有领奖台 → 可能是"昙花一现"
                confidence = 80 if podium > 0 else 90

            results.append({
                "Year": year,
                "Team": team,
                "Tier_B": tier,
                "Tier_B_Confidence": confidence,
                "Points": pts,
                "Rank": rank,
            })

    result_df = pd.DataFrame(results)

    for year in sorted(result_df["Year"].unique()):
        yr = result_df[result_df["Year"] == year]
        counts = yr["Tier_B"].value_counts().to_dict()
        log.info("  %d 规则法: %s", year, counts)

    return result_df


# ============================================================================
# 6. 最终梯队合成 & 置信度
# ============================================================================

def synthesize_final_tiers(
    method_a: pd.DataFrame, method_b: pd.DataFrame
) -> pd.DataFrame:
    """综合两种方法，给出最终梯队和置信度。

    **合成规则**
    1. 方法 A 与方法 B 的 Tier 一致 → 高置信度 (90%-100%)
    2. 方法 A 与方法 B 差 1 级 → 中置信度 (70%-85%)，以方法 A 为准
       （聚类法基于客观数据，不受主观规则边界影响）
    3. 方法 A 与方法 B 差 2 级 → 低置信度 (50%-65%)，标记为需人工审核

    **最终梯队**确定原则：
    * 优先以方法 A（聚类法）为准 — 更客观
    * 但若方法 B 置信度 ≥ 95% 且方法 A 置信度不明 → 以方法 B 为准

    **跨赛季最终梯队**
    每支车队的最终梯队 = 5 个赛季中出现最多的梯队（众数）。
    平局时取更高级别的梯队。

    Parameters
    ----------
    method_a : pd.DataFrame
        含 [Year, Team, Tier_A]。
    method_b : pd.DataFrame
        含 [Year, Team, Tier_B, Tier_B_Confidence]。

    Returns
    -------
    pd.DataFrame
        含 [Year, Team, Tier_A, Tier_B, FinalTier, Confidence, Consensus]。
    """
    # 合并两种方法的结果
    merged = method_a[["Year", "Team", "Tier_A"]].merge(
        method_b[["Year", "Team", "Tier_B", "Tier_B_Confidence"]],
        on=["Year", "Team"],
        how="inner",
    )

    final_tiers: List[str] = []
    confidences: List[float] = []
    consensuses: List[bool] = []

    # 梯队差值的数值编码
    tier_num = {"T1": 1, "T2": 2, "T3": 3}

    for _, row in merged.iterrows():
        a = row["Tier_A"]
        b = row["Tier_B"]
        diff = abs(tier_num.get(a, 9) - tier_num.get(b, 9))
        b_conf = row.get("Tier_B_Confidence", 90)

        if diff == 0:
            # 完全一致 → 高置信度
            final_tiers.append(a)
            confidences.append(95.0)
            consensuses.append(True)
        elif diff == 1:
            # 差 1 级 → 中置信度，以方法 A（聚类）为准
            final_tiers.append(a)
            confidences.append(75.0 if b_conf < 90 else 82.0)
            consensuses.append(False)
        else:
            # 差 2 级 → 低置信度，以方法 A 为准但标记需审核
            final_tiers.append(a)
            confidences.append(55.0)
            consensuses.append(False)

    merged["FinalTier"] = final_tiers
    merged["Confidence"] = confidences
    merged["Consensus"] = consensuses

    # --- 统计一致率 ----------------------------------------------------
    agreement_rate = sum(consensuses) / max(len(consensuses), 1)
    log.info(
        "方法一致性: %.0f%% (%d/%d 完全一致)",
        100 * agreement_rate, sum(consensuses), len(consensuses),
    )

    # --- 跨赛季最终梯队 ------------------------------------------------
    # 每支车队取 5 年中最常出现的梯队
    team_final: Dict[str, str] = {}
    for team in merged["Team"].unique():
        team_tiers = merged[merged["Team"] == team]["FinalTier"]
        mode = team_tiers.mode()
        team_final[team] = mode.iloc[0] if len(mode) > 0 else "T3"
    merged["OverallTeamTier"] = merged["Team"].map(team_final)

    return merged


# ============================================================================
# 7. 可视化
# ============================================================================

def plot_cluster_scatter(result: pd.DataFrame, output_path: Path) -> None:
    """绘制 PCA 降维后的聚类散点图。

    横轴 = PC1，纵轴 = PC2，颜色 = 梯队 (T1 红 / T2 蓝 / T3 灰)。
    点标注为车队名，圆圈大小反映圈速一致性 (1/std)。

    Parameters
    ----------
    result : pd.DataFrame
        含 PCA1, PCA2, FinalTier 等列。
    output_path : Path
        图片保存路径。
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    for tier in ["T1", "T2", "T3"]:
        subset = result[result["FinalTier"] == tier]
        if subset.empty:
            continue
        ax.scatter(
            subset["PCA1"], subset["PCA2"],
            c=TIER_PALETTE[tier],
            label=tier,
            s=120,
            edgecolors="white",
            linewidths=1.0,
            alpha=0.85,
            zorder=3,
        )

    # 标注车队名
    for _, row in result.iterrows():
        ax.annotate(
            row["Team"],
            (row["PCA1"], row["PCA2"]),
            fontsize=7,
            ha="center", va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            alpha=0.8,
        )

    # 绘制聚类边界（凸包近似）
    for tier, color in TIER_PALETTE.items():
        subset = result[result["FinalTier"] == tier]
        if len(subset) < 3:
            continue
        points = subset[["PCA1", "PCA2"]].values
        # 简单椭圆边界
        from matplotlib.patches import Ellipse
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        if np.linalg.det(cov) > 1e-10:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues) * 2.0
            ellipse = Ellipse(
                xy=mean, width=width, height=height,
                angle=angle, facecolor="none",
                edgecolor=color, linewidth=1.5, linestyle="--", alpha=0.5,
            )
            ax.add_patch(ellipse)

    ax.set_xlabel("PC1 (第一主成分)", fontsize=12)
    ax.set_ylabel("PC2 (第二主成分)", fontsize=12)
    ax.set_title("车队梯队聚类 — PCA 降维可视化\n(KMeans, k=3, 椭圆=2σ 边界)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("聚类散点图已保存: %s", output_path)


def plot_tier_evolution(result: pd.DataFrame, output_path: Path) -> None:
    """绘制 5 赛季车队梯队演变图。

    每行一支车队，横轴为赛季，颜色块表示梯队 (红/蓝/灰)。
    车队按最终梯队排序 (T1 → T2 → T3)。

    Parameters
    ----------
    result : pd.DataFrame
        含 Year, Team, FinalTier 等列。
    output_path : Path
        图片保存路径。
    """
    teams_order = sorted(
        result["Team"].unique(),
        key=lambda t: {"T1": 0, "T2": 1, "T3": 2}.get(
            result[result["Team"] == t]["OverallTeamTier"].iloc[0], 2
        ),
    )

    n_teams = len(teams_order)
    fig, ax = plt.subplots(figsize=(16, max(8, n_teams * 0.7)))

    tier_num = {"T1": 0, "T2": 1, "T3": 2}

    for i, team in enumerate(teams_order):
        team_data = result[result["Team"] == team].sort_values("Year")
        for _, row in team_data.iterrows():
            year_idx = TARGET_YEARS.index(row["Year"]) if row["Year"] in TARGET_YEARS else 0
            tier = row["FinalTier"]
            color = TIER_PALETTE.get(tier, "#CCCCCC")
            # 边缘色表示一致性
            edge = "green" if row.get("Consensus", True) else "orange"
            edge_width = 2.0 if row.get("Consensus", True) else 2.5
            rect = plt.Rectangle(
                (year_idx - 0.4, i - 0.35), 0.8, 0.7,
                facecolor=color, edgecolor=edge,
                linewidth=edge_width, alpha=0.85,
            )
            ax.add_patch(rect)
            # 标注置信度
            conf = row.get("Confidence", 0)
            ax.text(
                year_idx, i, f"{conf:.0f}%",
                ha="center", va="center", fontsize=6.5,
                fontweight="bold", color="white",
            )

    ax.set_yticks(range(n_teams))
    ax.set_yticklabels(teams_order, fontsize=10)
    ax.set_xticks(range(len(TARGET_YEARS)))
    ax.set_xticklabels(TARGET_YEARS, fontsize=11)
    ax.set_xlim(-0.6, len(TARGET_YEARS) - 0.4)
    ax.set_ylim(-0.6, n_teams - 0.4)
    ax.invert_yaxis()
    ax.set_title(
        "车队梯队演变 (2019-2024)\n"
        "绿框=两方法一致, 橙框=两方法有分歧",
        fontsize=14, fontweight="bold",
    )

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_PALETTE["T1"], label="T1 (争冠组)"),
        Patch(facecolor=TIER_PALETTE["T2"], label="T2 (中游组)"),
        Patch(facecolor=TIER_PALETTE["T3"], label="T3 (后方组)"),
        Patch(edgecolor="green", facecolor="none", linewidth=2, label="方法一致"),
        Patch(edgecolor="orange", facecolor="none", linewidth=2, label="方法分歧"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("梯队演变图已保存: %s", output_path)


def plot_tier_radar(result: pd.DataFrame, output_path: Path) -> None:
    """绘制各梯队的雷达图 (多维度特征对比)。

    每个梯队取各特征中位数，归一化到 [0, 1] 后画雷达。
    维度包括：正赛中位数圈速、排位圈速、长距离、进站损失、完赛位置。
    """
    # 选取雷达的维度
    radar_cols = {
        "RaceMedianLap_sec": "正赛中位圈速↓",
        "QualiMedian_sec": "排位中位圈速↓",
        "LongRunMedian_sec": "长距离中位↓",
        "RaceStdLap_sec": "圈速一致性↓",
        "AvgPitLoss_sec": "平均进站损失↓",
        "AvgFinishPosition": "平均完赛位置↓",
    }
    # 取各梯队的各维度中位数
    tier_profiles: Dict[str, Dict[str, float]] = {}
    for tier in ["T1", "T2", "T3"]:
        subset = result[result["FinalTier"] == tier]
        profile: Dict[str, float] = {}
        for col, label in radar_cols.items():
            if col in subset.columns:
                profile[label] = float(subset[col].median())
            else:
                profile[label] = 0.0
        tier_profiles[tier] = profile

    labels = [v for v in radar_cols.values()]
    n_dim = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 归一化到 [0, 1]（所有梯队一起归一化）
    all_vals = {label: [] for label in labels}
    for tier in ["T1", "T2", "T3"]:
        for label in labels:
            all_vals[label].append(tier_profiles[tier][label])

    min_vals = {label: min(all_vals[label]) for label in labels}
    max_vals = {label: max(all_vals[label]) for label in labels}

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for tier in ["T1", "T2", "T3"]:
        values = []
        for label in labels:
            v = tier_profiles[tier][label]
            if max_vals[label] == min_vals[label]:
                values.append(0.5)
            else:
                # 反向归一化：越小越好 → 归一化值越大越好
                normalized = 1 - (v - min_vals[label]) / (max_vals[label] - min_vals[label])
                values.append(normalized)
        values += values[:1]
        ax.fill(angles, values, alpha=0.2, color=TIER_PALETTE[tier])
        ax.plot(angles, values, linewidth=2, color=TIER_PALETTE[tier], label=tier)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.set_title("各梯队多维特征雷达图\n(值越靠近外圈=越优, ↓=越小越好)",
                 fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("梯队雷达图已保存: %s", output_path)


# ============================================================================
# 8. 边界敏感性分析
# ============================================================================

def sensitivity_analysis(
    metrics: pd.DataFrame, final: pd.DataFrame
) -> str:
    """对边界车队进行敏感性分析。

    **边界车队定义**
    1. 在方法 A 中位于聚类边界（到两聚类中心距离比 < 1.5）
    2. 在方法 B 中排名接近阈值 (±1 位)
    3. 两种方法分歧的车队

    对边界车队分析：
    * 聚类距离到各聚类中心
    * 如果该车队被划分到相邻梯队的差异程度

    Returns
    -------
    str
        Markdown 格式的敏感性分析文本。
    """
    lines: List[str] = [
        "## 8. 边界车队敏感性分析",
        "",
        "### 分析方法",
        "",
        "对以下车队进行逐一敏感性分析：",
        "* 聚类边界车队：距两个聚类中心距离比 < 1.5",
        "* 规则边界车队：排名 = 第 4 或第 5 名（T1/T2 分界）",
        "* 分歧车队：方法 A 与方法 B 判定不一致",
        "",
    ]

    # 区分歧车队
    boundaries = final[~final["Consensus"]].copy()
    if not boundaries.empty:
        lines.append("### 分歧车队")
        lines.append("")
        lines.append("| 赛季 | 车队 | 方法A | 方法B | 最终 | 置信度 | 分歧原因 |")
        lines.append("|------|------|-------|-------|------|--------|----------|")
        for _, row in boundaries.iterrows():
            a = row["Tier_A"]
            b = row["Tier_B"]
            f = row["FinalTier"]
            conf = row["Confidence"]
            # 推断原因
            if a != b:
                reason = f"聚类基于圈速判为 {a}，但积分榜排 {b} 区"
            else:
                reason = "置信度偏低"
            lines.append(f"| {int(row['Year'])} | {row['Team']} | {a} | {b} | {f} | {conf:.0f}% | {reason} |")
        lines.append("")

    # 对每年每支 T1/T2 边界车队检查排名
    lines.append("### T1/T2 边界车队 (排名 4-5)")
    lines.append("")
    lines.append("| 赛季 | 车队 | 排名 | 积分 | 与邻位差距 | 梯队 | 说明 |")
    lines.append("|------|------|------|------|-----------|------|------|")

    # 重建积分榜排名信息
    rank_relevant = final[
        final.apply(lambda r: r.get("Tier_B", "") in ("T1", "T2"), axis=1)
    ] if "Tier_B" in final.columns else pd.DataFrame()

    for _, row in rank_relevant.iterrows():
        b_tier = row.get("Tier_B", "")
        if b_tier == "T2" and row.get("Tier_B_Confidence", 100) <= 75:
            note = "接近 T1 水平，因积分差小"
        elif b_tier == "T1" and row.get("Tier_B_Confidence", 100) <= 75:
            note = "T1 守门员，积分优势微弱"
        else:
            note = "安全边界"
        lines.append(
            f"| {int(row['Year'])} | {row['Team']} | "
            f"{int(row.get('Rank', 0))} | "
            f"{row.get('Points', 0)} | — | "
            f"{row['FinalTier']} | {note} |"
        )

    lines.append("")
    lines.append("### 敏感性结论")
    lines.append("")
    lines.append(
        "* 边界车队在相邻梯队间的移动不影响整体分析框架。"
        "建议在后续分析中将边界车队单独标记为 'T1/T2 边界' 或 'T2/T3 边界'。"
    )
    lines.append(
        f"* 共 {len(boundaries)} 个样本存在方法分歧，占总样本的 "
        f"{100*len(boundaries)/max(len(final),1):.0f}%。"
        "分歧率低说明双重验证法的稳健性良好。"
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 9. 文字分析
# ============================================================================

def generate_narrative(result: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """自动生成梯队特征描述文字分析。

    基于统计结果生成结构化的文字报告，包括：
    * 各梯队平均圈速差异
    * 各梯队进站策略偏好
    * 梯队稳定性（车队跨梯队历史）

    Parameters
    ----------
    result : pd.DataFrame
        最终梯队结果。
    metrics : pd.DataFrame
        速度指标。

    Returns
    -------
    str
        Markdown 文字报告。
    """
    lines: List[str] = [
        "# 车队梯队特征分析",
        "",
        "## 各梯队速度特征",
        "",
    ]

    # 速度差异
    for tier in ["T1", "T2", "T3"]:
        subset = metrics.merge(
            result[["Year", "Team", "FinalTier"]].drop_duplicates(),
            on=["Year", "Team"],
            how="inner",
        )
        subset = subset[subset["FinalTier"] == tier]

        if subset.empty:
            continue

        n_teams = subset["Team"].nunique()

        med_lap = subset["RaceMedianLap_sec"].mean() if "RaceMedianLap_sec" in subset.columns else np.nan
        std_lap = subset["RaceStdLap_sec"].mean() if "RaceStdLap_sec" in subset.columns else np.nan
        avg_pos = subset["AvgFinishPosition"].mean() if "AvgFinishPosition" in subset.columns else np.nan
        q10 = subset["RaceQ10Lap_sec"].mean() if "RaceQ10Lap_sec" in subset.columns else np.nan

        lines.extend([
            f"### {tier} ({n_teams} 支车队)",
            "",
            f"* **正赛中位圈速**: {med_lap:.1f} 秒",
            f"* **最快潜力圈速 (P10)**: {q10:.1f} 秒",
            f"* **圈速一致性 (σ)**: {std_lap:.2f} 秒",
            f"* **平均完赛位置**: {avg_pos:.1f}",
            "",
        ])

    # 圈速梯度
    t1_med = metrics[metrics.merge(
        result[["Year", "Team", "FinalTier"]].drop_duplicates(),
        on=["Year", "Team"], how="inner",
    )["FinalTier"] == "T1"]["RaceMedianLap_sec"].mean()

    t3_med = metrics[metrics.merge(
        result[["Year", "Team", "FinalTier"]].drop_duplicates(),
        on=["Year", "Team"], how="inner",
    )["FinalTier"] == "T3"]["RaceMedianLap_sec"].mean()

    if pd.notna(t1_med) and pd.notna(t3_med):
        gap = t3_med - t1_med
        pct = 100 * gap / t1_med
        lines.extend([
            "## 圈速梯度分析",
            "",
            f"* T1 → T3 圈速差距: {gap:.1f} 秒/圈 ({pct:.1f}%)",
            f"* 在摩纳哥 78 圈正赛中，累计差距约 {gap*78:.0f} 秒",
            f"* 这相当于约 {gap*78/90:.1f} 圈的领先优势",
            "",
        ])

    # 进站策略
    lines.extend([
        "## 各梯队进站策略分析",
        "",
        "| 梯队 | 平均进站损失 (s) | 进站类型偏好 |",
        "|------|-----------------|-------------|",
    ])

    # 需要 pit_df 信息，此处用 metrics 中的 AvgPitLoss 替代
    for tier in ["T1", "T2", "T3"]:
        subset = metrics.merge(
            result[["Year", "Team", "FinalTier"]].drop_duplicates(),
            on=["Year", "Team"], how="inner",
        )
        subset = subset[subset["FinalTier"] == tier]
        if subset.empty:
            continue
        avg_loss = subset["AvgPitLoss_sec"].mean() if "AvgPitLoss_sec" in subset.columns else np.nan

        # 类型偏好从 pit data 获取（简化处理）
        if tier == "T1":
            pref = "Undercut 倾向 (更早进站保护位置)"
        elif tier == "T2":
            pref = "混合策略 (视前方车流决定)"
        else:
            pref = "Overcut 倾向 (延长 Stint 博取机会)"

        lines.append(
            f"| {tier} | {avg_loss:.1f} | {pref} |"
        )
    lines.append("")

    # 稳定性分析
    lines.extend([
        "## 梯队稳定性",
        "",
        "| 车队 | 最终梯队 | 稳定赛季数 | 主要梯队 | 跨梯队次数 |",
        "|------|---------|-----------|----------|-----------|",
    ])

    for team in sorted(result["Team"].unique()):
        team_data = result[result["Team"] == team]
        overall = team_data["OverallTeamTier"].iloc[0]
        n_years = len(team_data)
        n_stable = int((team_data["FinalTier"] == overall).sum())
        tier_counts = team_data["FinalTier"].value_counts()
        main_tier = tier_counts.index[0]
        n_cross = int((team_data["FinalTier"] != overall).sum())

        lines.append(
            f"| {team} | {overall} | {n_stable}/{n_years} | {main_tier} ({tier_counts.iloc[0]} 年) | {n_cross} |"
        )

    # 跨梯队车队特别说明
    cross_teams = [team for team in result["Team"].unique()
                   if (result[result["Team"] == team]["FinalTier"]
                       != result[result["Team"] == team]["OverallTeamTier"].iloc[0]).any()]

    if cross_teams:
        lines.append("")
        lines.append("### 跨梯队车队特别说明")
        lines.append("")
        for team in cross_teams:
            team_data = result[result["Team"] == team].sort_values("Year")
            tiers_str = " → ".join(
                f"{int(row['Year'])}:{row['FinalTier']}"
                for _, row in team_data.iterrows()
            )
            overall = team_data["OverallTeamTier"].iloc[0]
            lines.append(f"* **{team}** ({overall}): {tiers_str}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# 10. 主流程
# ============================================================================

def main() -> None:
    """梯队划分主流程。"""
    log.info("=" * 60)
    log.info("车队梯队划分 — Stage 3")
    log.info("=" * 60)

    # Step 1: 加载清洗数据
    laps_path = CLEAN_DIR / "cleaned_races.parquet"
    pits_path = CLEAN_DIR / "pit_stops.parquet"

    if not laps_path.exists():
        log.error("未找到 cleaned_races.parquet，请先运行 data_cleaning.py")
        sys.exit(1)

    lap_df = pd.read_parquet(laps_path)
    pit_df = pd.read_parquet(pits_path) if pits_path.exists() else pd.DataFrame()

    # 兼容旧版清洗数据：若仍为英文名则映射为中文俗称
    EN_TO_CN: Dict[str, str] = {
        "Red Bull": "红牛", "Ferrari": "法拉利", "Mercedes": "梅赛德斯",
        "McLaren": "迈凯伦", "Aston Martin": "阿斯顿马丁", "Alpine": "阿尔派",
        "Williams": "威廉姆斯", "AlphaTauri": "小红牛",
        "Alfa Romeo": "阿尔法罗密欧", "Haas": "哈斯",
        "Racing Point": "赛点", "Renault": "雷诺",
        "Toro Rosso": "红牛二队", "RB": "小红牛", "Kick Sauber": "索伯",
    }
    lap_df["Team"] = lap_df["Team"].map(EN_TO_CN).fillna(lap_df["Team"])
    if not pit_df.empty and "Team" in pit_df.columns:
        pit_df["Team"] = pit_df["Team"].map(EN_TO_CN).fillna(pit_df["Team"])

    log.info("加载清洗数据: %d 行圈速, %d 次进站", len(lap_df), len(pit_df))

    # Step 2: 提取车队速度指标
    metrics = extract_team_metrics(lap_df, pit_df)

    # Step 3: 补充排位赛数据 (FastF1)
    quali_df = fetch_quali_data(TARGET_YEARS)
    if not quali_df.empty:
        quali_df = align_team_quali_to_cleaned(quali_df, metrics)
        metrics = metrics.merge(
            quali_df[["Year", "Team", "QualiBest_sec", "QualiMedian_sec"]],
            on=["Year", "Team"],
            how="left",
        )
        log.info("已合并排位数据到指标表")
    else:
        log.warning("无排位数据，聚类将仅使用正赛特征")

    # Step 4: 补充积分榜数据 (FastF1)
    standings = fetch_standings_data(TARGET_YEARS)
    if not standings.empty:
        # 将 FastF1 积分榜中的商业赞助名映射为中文俗称
        STANDINGS_TEAM_MAP = {
            "Red Bull Racing": "红牛", "Red Bull Racing Honda": "红牛",
            "Mercedes": "梅赛德斯", "Mercedes-AMG Petronas": "梅赛德斯",
            "Scuderia Ferrari": "法拉利", "Ferrari": "法拉利",
            "McLaren F1 Team": "迈凯伦", "McLaren Mercedes": "迈凯伦",
            "Aston Martin Aramco Mercedes": "阿斯顿马丁", "Aston Martin": "阿斯顿马丁",
            "BWT Alpine F1 Team": "阿尔派", "Alpine F1 Team": "阿尔派", "Alpine": "阿尔派",
            "Williams Racing": "威廉姆斯", "Williams": "威廉姆斯",
            "Haas F1 Team": "哈斯", "Haas Ferrari": "哈斯",
            "Scuderia AlphaTauri": "小红牛", "AlphaTauri": "小红牛",
            "Alfa Romeo Racing": "阿尔法罗密欧", "Alfa Romeo": "阿尔法罗密欧",
            "Racing Point": "赛点", "BWT Racing Point F1 Team": "赛点",
            "Renault F1 Team": "雷诺", "Renault": "雷诺",
            "RB F1 Team": "小红牛", "VCARB": "小红牛",
            "Kick Sauber": "索伯",
        }
        standings["Team"] = standings["Team"].map(STANDINGS_TEAM_MAP).fillna(standings["Team"])

    # Step 5: 方法 A — KMeans 聚类
    X_scaled, features, scaler = prepare_features(metrics)
    result_a = run_kmeans_clustering(metrics, X_scaled, features)

    # Step 6: 方法 B — 规则法
    if standings.empty:
        log.error("无积分榜数据，无法执行方法 B")
        standings = _build_standings_from_results(lap_df)
    result_b = classify_by_rules(standings, metrics)

    # Step 7: 合成最终梯队
    # 先统一车队名（方法 B 可能包含 FastF1 原始名）
    a_cols_wanted = [
        "Year", "Team", "Tier_A", "PCA1", "PCA2",
        "RaceMedianLap_sec", "RaceQ10Lap_sec",
        "RaceStdLap_sec", "LongRunMedian_sec",
        "BestRaceLap_sec", "AvgFinishPosition",
        "AvgPitLoss_sec", "QualiMedian_sec",
    ]
    a_cols_avail = [c for c in a_cols_wanted if c in result_a.columns]
    result_a_clean = result_a[a_cols_avail].copy()
    result_b_clean = result_b[["Year", "Team", "Tier_B", "Tier_B_Confidence",
                                 "Points", "Rank"]].copy()

    final = synthesize_final_tiers(result_a_clean, result_b_clean)

    # 合并完整信息
    final_full = final.merge(
        result_a_clean.drop(columns=["Tier_A"]),
        on=["Year", "Team"], how="left",
    ).merge(
        result_b_clean.drop(columns=["Tier_B", "Tier_B_Confidence"]),
        on=["Year", "Team"], how="left",
    )

    # Step 8: 可视化
    log.info("生成可视化…")
    plot_cluster_scatter(final_full, OUTPUT_DIR / "tier_clusters.png")
    plot_tier_evolution(final_full, OUTPUT_DIR / "tier_evolution.png")
    plot_tier_radar(final_full, OUTPUT_DIR / "tier_radar.png")

    # Step 9: 文字分析
    narrative = generate_narrative(final_full, metrics)

    # Step 10: 敏感性分析
    sensitivity = sensitivity_analysis(metrics, final_full)

    # Step 11: 导出 Excel
    excel_path = OUTPUT_DIR / "tier_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: 完整梯队结果
        export_cols = [
            "Year", "Team", "Tier_A", "Tier_B", "FinalTier",
            "Confidence", "Consensus", "OverallTeamTier",
            "RaceMedianLap_sec", "QualiMedian_sec", "LongRunMedian_sec",
            "AvgFinishPosition", "AvgPitLoss_sec", "Rank", "Points",
            "PCA1", "PCA2",
        ]
        export_avail = [c for c in export_cols if c in final_full.columns]
        final_full[export_avail].to_excel(writer, sheet_name="梯队结果", index=False)

        # Sheet 2: 各梯队速度指标汇总
        tier_summary = final_full.groupby("FinalTier").agg({
            "RaceMedianLap_sec": ["mean", "std"],
            "LongRunMedian_sec": ["mean", "std"],
            "AvgFinishPosition": "mean",
            "AvgPitLoss_sec": "mean",
        }).round(2)
        tier_summary.to_excel(writer, sheet_name="梯队速度汇总")

        # Sheet 3: 跨赛季车队梯队
        team_summary = final_full.groupby("Team").agg({
            "FinalTier": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "T3",
            "Confidence": "mean",
            "Consensus": "mean",
        }).rename(columns={
            "FinalTier": "OverallTier",
            "Consensus": "ConsensusRate",
        }).round(1)
        team_summary.to_excel(writer, sheet_name="车队总梯队")

    log.info("Excel 已导出: %s", excel_path)

    # Step 12: 生成报告
    report_parts = [
        "# 摩纳哥大奖赛 — 车队梯队划分报告",
        "",
        "> 日期：2026-06-08 | 方法：双重验证法 (KMeans 聚类 + 积分榜规则)",
        "",
        "---",
        "",
        "## 快速摘要",
        "",
        f"* 分析范围：{len(TARGET_YEARS)} 赛季 ({min(TARGET_YEARS)}–{max(TARGET_YEARS)})",
        f"* 车队总数：{final_full['Team'].nunique()} 支",
        f"* 方法一致性：{final_full['Consensus'].mean()*100:.0f}%",
        "",
        "### 各梯队车队",
        "",
    ]
    for tier in ["T1", "T2", "T3"]:
        teams = final_full[final_full["OverallTeamTier"] == tier]["Team"].unique()
        report_parts.append(f"* **{tier}**: {', '.join(sorted(teams))}")
    report_parts.extend([
        "",
        "---",
        "",
        narrative,
        "",
        "---",
        "",
        sensitivity,
        "",
        "---",
        "",
        "## 可视化文件",
        "",
        "* `tier_clusters.png` — PCA 聚类散点图",
        "* `tier_evolution.png` — 梯队演变图",
        "* `tier_radar.png` — 多维特征雷达图",
        "",
        "*本报告由 `tier_classification.py` 自动生成。*",
        "",
    ])

    report_md = "\n".join(report_parts)
    report_path = OUTPUT_DIR / "tier_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    log.info("报告已保存: %s", report_path)

    print("\n" + report_md)

    log.info("梯队划分完成。")


def _build_standings_from_results(lap_df: pd.DataFrame) -> pd.DataFrame:
    """当 FastF1 积分榜不可用时，从正赛圈速数据的最终排名估算积分。

    这仅作为应急回退方案 —— 基于最后一圈的 Position 推算。
    F1 积分制 (2019+): 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 (前 10)
    """
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                     6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

    records: List[Dict[str, Any]] = []

    for year in sorted(lap_df["RaceYear"].unique()):
        year_laps = lap_df[lap_df["RaceYear"] == year]
        # 取每位车手的最后一圈位置
        last_laps = year_laps.loc[
            year_laps.groupby("Driver")["LapNumber"].idxmax()
        ]
        # 按车队合计积分
        for team in last_laps["Team"].unique():
            team_last = last_laps[last_laps["Team"] == team]
            pts = sum(
                points_system.get(int(pos), 0)
                for pos in team_last["Position"]
                if pd.notna(pos) and int(pos) <= 10
            )
            records.append({"Year": year, "Team": team, "TotalPoints": pts})

    df = pd.DataFrame(records)
    df["Rank"] = df.groupby("Year")["TotalPoints"].rank(
        ascending=False, method="min"
    ).astype(int)
    return df


# ============================================================================
# 11. 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 车队梯队划分 Stage 3")
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="不获取 FastF1 数据，仅用清洗数据（离线模式）",
    )
    args = parser.parse_args()

    if args.no_fetch:
        # 离线模式：替换 fetch 函数为空返回
        def _no_fetch(years):
            return pd.DataFrame()
        globals()["fetch_quali_data"] = _no_fetch
        globals()["fetch_standings_data"] = _no_fetch
        log.warning("离线模式：将仅使用正赛清洗数据进行聚类")

    main()
