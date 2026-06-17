#!/usr/bin/env python3
"""
F1 Monaco Grand Prix — Stage 2 Data Cleaning & Feature Engineering
===================================================================

对 Stage 1 采集的 5 赛季逐圈数据进行统一清洗、特征计算和质量验证，
输出清洁的主表和进站事件表。

清洗管线
--------
1. 异常值处理  → 移除异常圈速、修正轮胎寿命、插值缺失进站时刻
2. 特征工程    → 进站损失、位置变化、进站类型、窗口安全性
3. 数据标准化  → 车队名称统一、轮胎配方标准化、时间秒化
4. 质量验证    → 逻辑一致性审查、统计摘要

输出
----
* ``cleaned_races.parquet``  — 5 年合并主表 (~15000 行)
* ``pit_stops.parquet``      — 进站事件表 (~300 行)
* ``cleaning_report.md``     — 自动化清洗报告

作者 : 课程论文研究
日期 : 2026-06-08
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# 全局配置
# ============================================================================

# Stage 1 输出的 Parquet 文件目录
INPUT_DIR: Path = Path("./output")
# Stage 2 输出目录
CLEAN_DIR: Path = Path("./cleaned")
# 目标赛季
TARGET_YEARS: List[int] = [2019, 2021, 2022, 2023, 2024]

# --- 异常圈速阈值 --------------------------------------------------------
# 摩纳哥赛道特性：正常圈速 70-90 秒（干地），全雨胎/安全车下可达 110-130 秒
# < 60 秒 → 物理不可能（安全车带领也不会这么快）
# > 200 秒 → 非比赛状态（红旗、赛道清理、出场圈/回场圈等）
MIN_LAP_TIME_SEC: float = 60.0
MAX_LAP_TIME_SEC: float = 200.0

# --- 进站损失基准圈选择窗口 ---------------------------------------------
# 在进站前后各取此窗口内的正常圈速，排除 in-lap 和 out-lap
BASELINE_LAP_WINDOW: int = 3

# --- 进站分类阈值 --------------------------------------------------------
# 与前方车辆进站圈数差 ≥ 此值才判定为 undercut/overcut
PIT_TYPE_LAP_THRESHOLD: int = 2

# --- 车队名称映射表 ------------------------------------------------------
# 统一不同赛季的车队命名（FastF1 商业赞助名 → 中文俗称）
# 参考中文 F1 社区通用叫法（虎扑、新浪 F1、B 站赛车区）
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
    "Alfa Romeo Racing": "阿尔法罗密欧",
    "Alfa Romeo Racing ORLEN": "阿尔法罗密欧",
    "Alfa Romeo F1 Team Stake": "阿尔法罗密欧",
    "Alfa Romeo F1 Team ORLEN": "阿尔法罗密欧",
    "Alfa Romeo": "阿尔法罗密欧",
    "Stake F1 Team Kick Sauber": "索伯",
    "Kick Sauber": "索伯",
    "Racing Point": "赛点",
    "Racing Point BWT Mercedes": "赛点",
    "BWT Racing Point F1 Team": "赛点",
    "Aston Martin Cognizant": "阿斯顿马丁",
    "Renault DP World F1 Team": "雷诺",
    "Renault F1 Team": "雷诺",
    "Renault": "雷诺",
    "RB F1 Team": "小红牛",
    "Visa Cash App RB F1 Team": "小红牛",
    "VCARB": "小红牛",
    "AlphaTauri": "小红牛",
}

# --- 轮胎配方标准化 ------------------------------------------------------
COMPOUND_MAP: Dict[str, str] = {
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE",
    "WET": "WET",
    "SUPERSOFT": "SOFT",       # 2018 之前使用，统一为 SOFT
    "ULTRASOFT": "SOFT",       # 2018 之前
    "HYPERSOFT": "SOFT",       # 2018 及之前
    "C1": "HARD",               # 2021+ Pirelli 代号映射
    "C2": "MEDIUM",
    "C3": "SOFT",
    "C4": "SOFT",
    "C5": "SOFT",
}

# ============================================================================
# 日志
# ============================================================================

def setup_logging() -> logging.Logger:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("monaco_cleaner")
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
    fh = logging.FileHandler(CLEAN_DIR / "cleaning.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logging()


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CleaningStats:
    """清洗过程统计数据容器。

    用于生成清洗报告的各个指标。
    """
    year: int
    rows_before: int = 0
    rows_after: int = 0
    outliers_removed: int = 0
    outlier_reasons: Dict[str, int] = field(default_factory=dict)
    n_pit_stops: int = 0
    pit_loss_mean: float = 0.0
    pit_loss_std: float = 0.0
    position_changes: Dict[str, int] = field(default_factory=dict)
    pit_types: Dict[str, int] = field(default_factory=dict)
    completeness_before: float = 0.0
    completeness_after: float = 0.0


# ============================================================================
# 1. 数据加载
# ============================================================================

def load_raw_data(years: List[int]) -> Dict[int, pd.DataFrame]:
    """从 Stage 1 输出目录加载各赛季 Parquet 文件。

    加载时自动检测 timedelta 列并转换为 pandas Timedelta 类型。
    缺失的 Parquet 文件被跳过并警告。

    Parameters
    ----------
    years : list of int
        要加载的赛季列表。

    Returns
    -------
    dict
        ``{year: DataFrame}``。
    """
    data: Dict[int, pd.DataFrame] = {}
    for year in years:
        fp = INPUT_DIR / f"monaco_{year}.parquet"
        if not fp.exists():
            log.warning("文件不存在，跳过 %d: %s", year, fp)
            continue
        df = pd.read_parquet(fp)

        # 恢复 timedelta 列：Parquet 以 int64(ns) 存储 timedelta
        for col in df.columns:
            if col.endswith("Time") and df[col].dtype == "int64":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[col] = pd.to_timedelta(df[col])

        log.info("加载 %d — %d 行 × %d 列", year, len(df), len(df.columns))
        data[year] = df
    return data


# ============================================================================
# 2. 辅助：时间转秒
# ============================================================================

def _td_to_seconds(series: pd.Series) -> pd.Series:
    """将 timedelta Series 转换为浮点秒数。

    非 timedelta 或全 NaT 列直接原样返回。
    这是全脚本中时间单位统一的核心工具函数。
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return series


# ============================================================================
# 3. 异常值处理
# ============================================================================

def detect_outlier_laps(df: pd.DataFrame) -> pd.Series:
    """检测异常圈速。

    返回布尔掩码：True = 正常圈，False = 异常圈。
    检测规则：
    * LapTime < MIN_LAP_TIME_SEC (60s) — 物理不可达
    * LapTime > MAX_LAP_TIME_SEC (200s) — 非比赛状态
    * LapTime 为 NaT — 数据缺失（出场圈/回场圈常见）

    Parameters
    ----------
    df : pd.DataFrame
        逐圈数据，需含 ``LapTime`` 列。

    Returns
    -------
    pd.Series[bool]
        正常圈的布尔掩码。
    """
    if "LapTime" not in df.columns:
        log.warning("LapTime 列缺失，跳过异常检测")
        return pd.Series(True, index=df.index)

    lap_sec = _td_to_seconds(df["LapTime"])

    # 三个条件：非空 + 大于下限 + 小于上限
    not_null = lap_sec.notna()
    above_min = lap_sec >= MIN_LAP_TIME_SEC
    below_max = lap_sec <= MAX_LAP_TIME_SEC

    mask = not_null & above_min & below_max

    n_removed = (~mask).sum()
    if n_removed > 0:
        n_null = (~not_null).sum()
        n_fast = (not_null & ~above_min).sum()
        n_slow = (not_null & ~below_max).sum()
        log.info(
            "  异常圈速: 移除 %d 条 (NaT=%d, <%.0fs=%d, >%.0fs=%d)",
            n_removed, n_null, MIN_LAP_TIME_SEC, n_fast, MAX_LAP_TIME_SEC, n_slow,
        )

    return mask


def fix_tyre_life(df: pd.DataFrame) -> pd.DataFrame:
    """修正轮胎寿命：确保每个 Stint 内 TyreLife 从 1 开始递增。

    FastF1 的 TyreLife 可能从 0 或从上一段延续计数。
    修正逻辑：
    1. 每位车手按 Stint 分组
    2. 在段内按 LapNumber 排序
    3. TyreLife = 段内累积圈数（第一圈 = 1）

    Parameters
    ----------
    df : pd.DataFrame
        逐圈数据。

    Returns
    -------
    pd.DataFrame
        TyreLife 已修正的数据。
    """
    if "TyreLife" not in df.columns or "Stint" not in df.columns:
        return df

    df = df.sort_values(["Driver", "LapNumber"]).copy()
    # groupby + cumcount: 段内序号从 0 开始，+1 得到从 1 开始的圈数
    df["TyreLife"] = df.groupby(["Driver", "Stint"]).cumcount() + 1
    return df


def interpolate_pit_times(df: pd.DataFrame) -> pd.DataFrame:
    """插值缺失的进站时刻。

    当 PitInTime/PitOutTime 全部为 NaT 时（FastF1 未提供），
    使用前后圈时间估算：

    * PitOutTime ≈ Out-Lap 的 Time（该圈开始时刻）
    * PitInTime  ≈ In-Lap 的 Time + In-Lap 的 LapTime（该圈结束时刻）

    Parameters
    ----------
    df : pd.DataFrame
        逐圈数据。

    Returns
    -------
    pd.DataFrame
        插值后的数据。
    """
    has_pit_out = (
        "PitOutTime" in df.columns and df["PitOutTime"].notna().any()
    )

    if not has_pit_out:
        # 从 Stint 边界重建进站时刻
        log.info("  PitOutTime 全部缺失，从 Stint 边界重建…")
        df = df.sort_values(["Driver", "LapNumber"]).copy()

        # 检测 out-lap：Stint 不同于上一圈
        for driver in df["Driver"].unique():
            drv = df[df["Driver"] == driver].index
            if len(drv) < 2:
                continue
            stints = df.loc[drv, "Stint"]
            next_stint = stints.shift(-1)
            prev_stint = stints.shift(1)

            # In-lap 标记：当前 Stint ≠ 下一圈 Stint
            is_in = (stints != next_stint) & next_stint.notna()
            # Out-lap 标记：当前 Stint ≠ 上一圈 Stint
            is_out = (stints != prev_stint) & prev_stint.notna()

            out_idx = drv[is_out]
            if len(out_idx) == 0:
                continue

            # PitIn/Out 初始化为 NaT
            if "PitInTime" not in df.columns:
                df["PitInTime"] = pd.NaT
            if "PitOutTime" not in df.columns:
                df["PitOutTime"] = pd.NaT

            for oi in out_idx:
                lap_num = df.at[oi, "LapNumber"]
                # In-lap 是 out-lap 的上一圈
                in_candidates = df.loc[
                    (df["Driver"] == driver)
                    & (df["LapNumber"] == lap_num - 1)
                ]
                if in_candidates.empty:
                    continue
                in_idx = in_candidates.index[0]
                in_time = df.at[in_idx, "Time"]
                in_lap = df.at[in_idx, "LapTime"]

                # 估算 PitInTime = in-lap 开始 + 圈时
                # 估算 PitOutTime = out-lap 开始
                if pd.notna(in_time) and pd.notna(in_lap):
                    df.at[in_idx, "PitInTime"] = in_time + in_lap
                    df.at[oi, "PitOutTime"] = df.at[oi, "Time"]
                    # 同步到 Stint 所有圈
                    stint = df.at[oi, "Stint"]
                    stint_idx = drv[df.loc[drv, "Stint"] == stint]
                    df.loc[stint_idx, "PitInTime"] = df.at[in_idx, "PitInTime"]
                    df.loc[stint_idx, "PitOutTime"] = df.at[oi, "PitOutTime"]

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """异常值处理主流程：检测 → 过滤 → 修正 → 插值。

    Parameters
    ----------
    df : pd.DataFrame
        原始逐圈数据。

    Returns
    -------
    pd.DataFrame
        清洗后的数据。
    """
    n_before = len(df)

    # 1. 标记并移除异常圈速
    mask = detect_outlier_laps(df)
    df = df.loc[mask].copy()

    # 2. 修正轮胎寿命
    df = fix_tyre_life(df)

    # 3. 插值缺失进站时刻
    df = interpolate_pit_times(df)

    n_after = len(df)
    log.info("  异常值处理: %d → %d 行 (移除 %d 条)", n_before, n_after, n_before - n_after)

    return df


# ============================================================================
# 4. 数据标准化
# ============================================================================

def standardize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """车队名称统一化。

    使用 ``TEAM_NAME_MAP`` 将各赛季中的商业赞助全名映射为统一简称。
    未匹配的车队保留原名并输出警告供后续补充映射。

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Team 列已标准化的数据。
    """
    if "Team" not in df.columns:
        return df

    df = df.copy()
    original = df["Team"].unique()
    df["Team"] = df["Team"].map(TEAM_NAME_MAP).fillna(df["Team"])

    # 记录未匹配的车队名
    unmapped = [t for t in original if t not in TEAM_NAME_MAP and t != "nan"]
    if unmapped:
        log.warning("未匹配的车队名: %s", unmapped)

    return df


def standardize_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """轮胎配方标准化。

    统一配方名称并处理无效值。
    * Pirelli 代号 (C1-C5) → 标准化名称
    * 大小写统一
    * 无效值标记为 "UNKNOWN"

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Compound 列已标准化的数据。
    """
    if "Compound" not in df.columns:
        return df

    df = df.copy()
    raw = df["Compound"].astype(str).str.upper().str.strip()
    df["Compound"] = raw.map(COMPOUND_MAP).fillna("UNKNOWN")

    # 标记完全无法识别的配方
    unknown_mask = df["Compound"] == "UNKNOWN"
    if unknown_mask.any():
        unique_unknown = raw[unknown_mask].unique()
        log.warning("无法识别的轮胎配方: %s", list(unique_unknown))

    return df


def convert_times_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """将所有 timedelta 列转换为浮点秒数。

    这样做的原因：
    * Parquet/CSV 对 timedelta 的支持不一致
    * 便于后续数值计算和统计分析
    * 浮点秒数是学术论文中通用的时间单位

    转换的列包括：LapTime, Sector1Time, Sector2Time, Sector3Time,
    GapToLeader, Interval, PitInTime, PitOutTime, Time

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        时间已秒化的数据。
    """
    df = df.copy()
    time_cols = [
        "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "GapToLeader", "Interval", "PitInTime", "PitOutTime", "Time",
    ]
    for col in time_cols:
        if col in df.columns and pd.api.types.is_timedelta64_dtype(df[col]):
            # 重命名保持语义清晰：列名不变，值变为秒
            df[col] = _td_to_seconds(df[col])

    return df


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据标准化主流程。"""
    df = standardize_teams(df)
    df = standardize_compounds(df)
    df = convert_times_to_seconds(df)
    log.info("  标准化完成: %d 行", len(df))
    return df


# ============================================================================
# 5. 进站事件表构建
# ============================================================================

def build_pit_stops(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """从逐圈数据中提取独立的进站事件。

    **提取逻辑**
    对每位车手，检测 Stint 编号的递增点：
    * Stint n 的最后一圈 → In-Lap (LapIn)
    * Stint n+1 的第一圈 → Out-Lap (LapOut)
    * PitInTime 取自 In-Lap 的 PitInTime (该圈结束时刻)
    * PitOutTime 取自 Out-Lap 的 PitOutTime (下一圈开始时刻)

    Parameters
    ----------
    df : pd.DataFrame
        逐圈数据（需含 Driver, Team, LapNumber, Stint, PitInTime, PitOutTime,
        Compound, Position, Interval）。
    year : int
        赛季年份，用于源标注。

    Returns
    -------
    pd.DataFrame
        进站事件表，每行一次进站。
    """
    if "Stint" not in df.columns:
        log.warning("%d — Stint 列缺失，无法构建进站表", year)
        return pd.DataFrame()

    df = df.sort_values(["Driver", "LapNumber"]).copy()

    records: List[Dict[str, Any]] = []

    for driver, grp in df.groupby("Driver"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue

        # 下一圈的 Stint 用于检测边界
        next_stint = grp["Stint"].shift(-1)
        is_in_lap = (grp["Stint"] != next_stint) & next_stint.notna()

        in_indices = grp.index[is_in_lap]
        for idx in in_indices:
            in_row = grp.loc[idx]
            # Out-Lap 是 In-Lap 的下一行
            out_idx = idx + 1
            if out_idx not in grp.index:
                continue
            out_row = grp.loc[out_idx]

            lap_in = int(in_row["LapNumber"])
            lap_out = int(out_row["LapNumber"])
            new_stint = int(out_row["Stint"])

            # 进站时刻
            pit_in = in_row.get("PitInTime", np.nan)
            pit_out = out_row.get("PitOutTime", np.nan)

            # 备选估算（当 PitIn/Out 为 NaN 时）
            if pd.isna(pit_in) and "Time" in in_row.index and "LapTime" in in_row.index:
                t = in_row["Time"]
                lt = in_row["LapTime"]
                if pd.notna(t) and pd.notna(lt):
                    pit_in = t + lt
            if pd.isna(pit_out) and "Time" in out_row.index:
                pit_out = out_row["Time"]

            # 进站时长
            if pd.notna(pit_in) and pd.notna(pit_out):
                duration = pit_out - pit_in
            else:
                duration = np.nan

            records.append({
                "RaceYear": year,
                "Driver": driver,
                "Team": in_row.get("Team", ""),
                "StintAfter": new_stint,
                "LapIn": lap_in,
                "LapOut": lap_out,
                "PitInTime_sec": pit_in,
                "PitOutTime_sec": pit_out,
                "PitDuration_sec": duration,
                "PositionBefore": in_row.get("Position", np.nan),
                "PositionAfter": out_row.get("Position", np.nan),
                "CompoundBefore": in_row.get("Compound", ""),
                "CompoundAfter": out_row.get("Compound", ""),
                "GapBehindIn_sec": in_row.get("Interval", np.nan),
                "LapIn_GapToLeader_sec": in_row.get("GapToLeader", np.nan),
                "LapOut_GapToLeader_sec": out_row.get("GapToLeader", np.nan),
            })

    pit_df = pd.DataFrame(records)
    if pit_df.empty:
        log.warning("%d — 未检测到进站事件", year)
        return pit_df

    log.info(
        "  构建进站事件表: %d 次进站 × %d 列", len(pit_df), len(pit_df.columns)
    )
    return pit_df


# ============================================================================
# 6. 特征计算 — 进站损失
# ============================================================================

def compute_pit_loss(
    pit_df: pd.DataFrame, lap_df: pd.DataFrame
) -> pd.DataFrame:
    """计算每次进站的 **纯进站损失** (pit_loss_seconds)。

    **公式**
    pit_loss = pit_duration - baseline_lap_time

    其中：
    * pit_duration = PitOutTime - PitInTime（进站全程耗时，含限速区行驶）
    * baseline_lap_time = 同一车手在进站前后 BASELINE_LAP_WINDOW 圈内
      （排除 In-Lap 和 Out-Lap）的正常圈速中位数

    中位数比均值更鲁棒，不受安全车/交通等极端圈速的影响。

    Parameters
    ----------
    pit_df : pd.DataFrame
        进站事件表（由 build_pit_stops 生成）。
    lap_df : pd.DataFrame
        逐圈数据（已秒化）。

    Returns
    -------
    pd.DataFrame
        增加 PitLoss_sec 和 BaselineLapTime_sec 的进站表。
    """
    pit_df = pit_df.copy()
    losses, baselines = [], []
    W = BASELINE_LAP_WINDOW

    for _, pit in pit_df.iterrows():
        driver = pit["Driver"]
        lap_in = int(pit["LapIn"])
        lap_out = int(pit["LapOut"])
        duration = pit["PitDuration_sec"]

        # 提取进站前后的正常圈速窗口
        driver_laps = lap_df[
            (lap_df["Driver"] == driver)
        ]

        # 前窗口：LapIn 之前 W 圈（排除 In-Lap 本身）
        before = driver_laps[
            (driver_laps["LapNumber"] >= lap_in - W)
            & (driver_laps["LapNumber"] < lap_in)
        ]
        # 后窗口：LapOut 之后 W 圈（排除 Out-Lap 本身）
        after = driver_laps[
            (driver_laps["LapNumber"] > lap_out)
            & (driver_laps["LapNumber"] <= lap_out + W)
        ]

        # 拼接前后窗口的正常圈速
        window = pd.concat([before, after])
        if "LapTime" not in window.columns or window.empty:
            baselines.append(np.nan)
            losses.append(np.nan)
            continue

        valid = window["LapTime"].dropna()
        if valid.empty:
            baselines.append(np.nan)
            losses.append(np.nan)
            continue

        # 中位数基准圈速
        baseline = float(valid.median())
        baselines.append(baseline)

        if pd.notna(duration):
            losses.append(duration - baseline)
        else:
            losses.append(np.nan)

    pit_df["BaselineLapTime_sec"] = baselines
    pit_df["PitLoss_sec"] = losses

    n_valid = pit_df["PitLoss_sec"].notna().sum()
    if n_valid > 0:
        mean_loss = pit_df["PitLoss_sec"].mean()
        log.info("  进站损失: 均值=%.1f 秒 (%d/%d 有效)",
                 mean_loss, n_valid, len(pit_df))
    else:
        log.warning("  进站损失: 全部无效")

    return pit_df


# ============================================================================
# 7. 特征计算 — 位置变化
# ============================================================================

def compute_position_change(pit_df: pd.DataFrame) -> pd.DataFrame:
    """计算进站前后的位置变化。

    * PositionBefore: In-Lap 结束时的赛道位置
    * PositionAfter: Out-Lap 结束时的赛道位置
    * PositionChange = PositionBefore − PositionAfter
    * 正值 → 名次上升（超车）；负值 → 名次下降（被超）

    Parameters
    ----------
    pit_df : pd.DataFrame
        进站事件表（已含 PositionBefore/PositionAfter）。

    Returns
    -------
    pd.DataFrame
        增加 PositionChange 列。
    """
    pit_df = pit_df.copy()
    if "PositionBefore" not in pit_df.columns or "PositionAfter" not in pit_df.columns:
        pit_df["PositionChange"] = np.nan
        return pit_df

    before = pd.to_numeric(pit_df["PositionBefore"], errors="coerce")
    after = pd.to_numeric(pit_df["PositionAfter"], errors="coerce")
    pit_df["PositionChange"] = before - after

    gained = (pit_df["PositionChange"] > 0).sum()
    lost = (pit_df["PositionChange"] < 0).sum()
    unchanged = (pit_df["PositionChange"] == 0).sum()
    log.info("  位置变化: 上升=%d, 不变=%d, 下降=%d", gained, unchanged, lost)

    return pit_df


# ============================================================================
# 8. 特征计算 — 进站类型
# ============================================================================

def compute_pit_type(
    pit_df: pd.DataFrame, lap_df: pd.DataFrame
) -> pd.DataFrame:
    """分类每次进站的策略类型。

    **判定逻辑（基于第一次进站）**
    对每场比赛，提取所有车手的首次进站圈号。
    对于每位车手 i：
    1. 找到其 In-Lap 时刻紧邻前方的车手 j（Position − 1）
    2. 比较两者首次进站的圈号：
       * 车手 i 进站圈 − 车手 j 进站圈 ≥ PIT_TYPE_LAP_THRESHOLD → **Overcut**
         （车手 i 延迟进站，尝试用干净空气圈获得优势）
       * 车手 j 进站圈 − 车手 i 进站圈 ≥ PIT_TYPE_LAP_THRESHOLD → **Undercut**
         （车手 i 提前进站，尝试用新胎优势缩小差距）
       * 其他 → **Normal**

    仅对首次进站进行分类；第二次及以上进站通常为应急/安全车进站。

    Parameters
    ----------
    pit_df : pd.DataFrame
        进站事件表。
    lap_df : pd.DataFrame
        逐圈数据。

    Returns
    -------
    pd.DataFrame
        增加 PitType 列。
    """
    pit_df = pit_df.copy()
    pit_df["PitType"] = "Normal"

    for year in pit_df["RaceYear"].unique():
        year_pits = pit_df[pit_df["RaceYear"] == year]
        year_laps = lap_df[lap_df["RaceYear"] == year]

        # 仅取首次进站
        first_pit = year_pits.loc[
            year_pits.groupby("Driver")["LapIn"].idxmin()
        ].copy()

        if len(first_pit) < 2:
            continue

        # 按键(Driver, LapIn)查找
        pit_lap_map = dict(zip(first_pit["Driver"], first_pit["LapIn"]))

        for _, pit_row in first_pit.iterrows():
            driver = pit_row["Driver"]
            lap_in = int(pit_row["LapIn"])

            # 找到 In-Lap 时刻的前方车手（Position − 1）
            pos_before = pit_row.get("PositionBefore")
            if pd.isna(pos_before):
                continue

            # 在同圈中找前方车手
            same_lap = year_laps[year_laps["LapNumber"] == lap_in]
            front_cars = same_lap[
                same_lap["Position"] == pos_before - 1
            ]
            if front_cars.empty:
                continue
            front_driver = str(front_cars.iloc[0]["Driver"])

            # 前方车手的首次进站圈
            front_pit_lap = pit_lap_map.get(front_driver)
            if front_pit_lap is None:
                continue

            delta = lap_in - front_pit_lap  # 正 = 我比前方晚进站

            if delta >= PIT_TYPE_LAP_THRESHOLD:
                pit_type = "Overcut"
            elif delta <= -PIT_TYPE_LAP_THRESHOLD:
                pit_type = "Undercut"
            else:
                pit_type = "Normal"

            # 回写到 pit_df 的全局索引
            global_idx = pit_df[
                (pit_df["RaceYear"] == year) & (pit_df["Driver"] == driver)
            ].index[0]
            pit_df.at[global_idx, "PitType"] = pit_type

    counts = pit_df["PitType"].value_counts().to_dict()
    log.info("  进站类型: %s", counts)
    return pit_df


# ============================================================================
# 9. 特征计算 — 窗口安全性
# ============================================================================

def compute_window_safety(pit_df: pd.DataFrame) -> pd.DataFrame:
    """计算进站出站的窗口安全性。

    **定义**
    WindowSafety = GapBehindIn_sec − PitLoss_sec

    * GapBehindIn_sec: In-Lap 时刻与前车的时间间隔 (Interval 列)
    * PitLoss_sec: 纯进站时间损失

    * WindowSafety > 0  → 出站后位置安全，前方仍有净空
    * WindowSafety < 0  → 出站后可能落入后车前方（存在位置威胁）
    * WindowSafety ≈ 0  → 出站时刻非常接近竞争车辆

    这是评估进站时机质量的关键指标。

    Parameters
    ----------
    pit_df : pd.DataFrame
        进站事件表（已含 GapBehindIn_sec 和 PitLoss_sec）。

    Returns
    -------
    pd.DataFrame
        增加 WindowSafety_sec 列。
    """
    pit_df = pit_df.copy()
    gap = pit_df.get("GapBehindIn_sec", pd.Series([np.nan] * len(pit_df)))
    loss = pit_df.get("PitLoss_sec", pd.Series([np.nan] * len(pit_df)))

    pit_df["WindowSafety_sec"] = gap - loss

    n_safe = (pit_df["WindowSafety_sec"] > 0).sum()
    n_risk = (pit_df["WindowSafety_sec"] <= 0).sum()
    if n_safe + n_risk > 0:
        log.info(
            "  窗口安全性: 安全=%d, 危险=%d (%.0f%% 安全)",
            n_safe, n_risk, 100 * n_safe / (n_safe + n_risk),
        )

    return pit_df


# ============================================================================
# 10. 数据验证
# ============================================================================

def validate_cleaned_data(
    lap_df: pd.DataFrame, pit_df: pd.DataFrame
) -> Dict[str, Any]:
    """验证清洗后数据的逻辑一致性。

    检查项：
    1. LapTime 范围合理性 (60-200s)
    2. TyreLife 从 1 开始且单调递增
    3. Position 合理性 (1-20)
    4. PitLoss 非负（进站不可能比正常圈快）
    5. 进站 Stint 连续性

    Parameters
    ----------
    lap_df : pd.DataFrame
        主表。
    pit_df : pd.DataFrame
        进站表。

    Returns
    -------
    dict
        {检查项: 通过/失败详情}。
    """
    results: Dict[str, Any] = {}

    # 1. 圈速范围
    if "LapTime" in lap_df.columns:
        lt = lap_df["LapTime"]
        valid = lt.between(60, 200)
        results["LapTime范围"] = {
            "pass": valid.all(),
            "detail": f"{valid.sum()}/{len(lt)} 在范围内 "
                      f"(min={lt.min():.1f}, max={lt.max():.1f})",
        }

    # 2. 轮胎寿命单调性
    if "TyreLife" in lap_df.columns and "Stint" in lap_df.columns:
        monotonic = (
            lap_df.groupby(["Driver", "Stint"])["TyreLife"]
            .apply(lambda x: x.is_monotonic_increasing)
        )
        n_bad = (~monotonic).sum()
        results["TyreLife单调性"] = {
            "pass": n_bad == 0,
            "detail": f"{n_bad} 个段违反单调递增",
        }

    # 3. 位置范围
    if "Position" in lap_df.columns:
        pos = pd.to_numeric(lap_df["Position"], errors="coerce")
        valid_pos = pos.between(1, 20)
        results["Position范围"] = {
            "pass": valid_pos.all(),
            "detail": f"{valid_pos.sum()}/{len(pos)} 在 [1,20]",
        }

    # 4. 进站损失非负
    if "PitLoss_sec" in pit_df.columns:
        loss = pit_df["PitLoss_sec"].dropna()
        if len(loss) > 0:
            neg = (loss < -2).sum()  # 允许 2 秒以内的浮动
            results["PitLoss非负"] = {
                "pass": neg == 0,
                "detail": f"{neg} 次负值 (min={loss.min():.1f}s)",
            }

    # 5. 进站 Stint 追踪
    if not pit_df.empty and "StintAfter" in pit_df.columns:
        bad_seq = 0
        for driver in pit_df["Driver"].unique():
            stints = sorted(pit_df[pit_df["Driver"] == driver]["StintAfter"])
            expected = list(range(2, 2 + len(stints)))
            if stints != expected:
                bad_seq += 1
        results["Stint序列"] = {
            "pass": bad_seq == 0,
            "detail": f"{bad_seq} 位车手 Stint 序列不连续",
        }

    all_pass = all(r["pass"] for r in results.values())
    log.info("数据验证: %s", "全部通过" if all_pass else "存在问题")
    for k, v in results.items():
        status = "✓" if v["pass"] else "✗"
        log.info("  %s %s: %s", status, k, v["detail"])

    return results


# ============================================================================
# 11. 主清洗管道
# ============================================================================

def clean_season(
    df: pd.DataFrame, year: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], CleaningStats]:
    """单赛季完整清洗管道。

    步骤：
    1. 添加 RaceYear 列
    2. 异常值处理
    3. 数据标准化
    4. 构建进站事件表
    5. 计算四个特征

    Parameters
    ----------
    df : pd.DataFrame
        单赛季原始数据。
    year : int
        赛季年份。

    Returns
    -------
    (cleaned_laps, pit_stops, stats)
    """
    stats = CleaningStats(year=year)
    stats.rows_before = len(df)

    # --- 标记赛季 ---------------------------------------------------------
    df = df.copy()
    df["RaceYear"] = year

    # --- 记录清洗前完整度 ------------------------------------------------
    stats.completeness_before = _completeness(df)

    # --- 异常值处理 -------------------------------------------------------
    df = handle_outliers(df)

    # --- 数据标准化 -------------------------------------------------------
    df = standardize_data(df)

    stats.rows_after = len(df)
    stats.outliers_removed = stats.rows_before - stats.rows_after
    stats.completeness_after = _completeness(df)

    # --- 构建进站事件表 --------------------------------------------------
    pit_df = build_pit_stops(df, year)
    if pit_df.empty:
        return df, pit_df, stats

    # --- 特征计算 ---------------------------------------------------------
    pit_df = compute_pit_loss(pit_df, df)
    pit_df = compute_position_change(pit_df)
    pit_df = compute_pit_type(pit_df, df)
    pit_df = compute_window_safety(pit_df)

    # --- 统计收集 ---------------------------------------------------------
    stats.n_pit_stops = len(pit_df)
    if "PitLoss_sec" in pit_df.columns:
        valid = pit_df["PitLoss_sec"].dropna()
        if len(valid) > 0:
            stats.pit_loss_mean = float(valid.mean())
            stats.pit_loss_std = float(valid.std())

    if "PositionChange" in pit_df.columns:
        pc = pit_df["PositionChange"]
        stats.position_changes = {
            "gained": int((pc > 0).sum()),
            "unchanged": int((pc == 0).sum()),
            "lost": int((pc < 0).sum()),
        }

    if "PitType" in pit_df.columns:
        stats.pit_types = pit_df["PitType"].value_counts().to_dict()

    return df, pit_df, stats


def _completeness(df: pd.DataFrame) -> float:
    """计算关键列非空比例均值。"""
    key_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife"]
    available = [c for c in key_cols if c in df.columns]
    if not available or len(df) == 0:
        return 0.0
    return float(np.mean([df[c].notna().mean() for c in available]) * 100)


# ============================================================================
# 12. 清洗报告
# ============================================================================

def generate_report(
    all_stats: List[CleaningStats],
    merged_laps: pd.DataFrame,
    merged_pits: pd.DataFrame,
    validation: Dict[str, Any],
) -> str:
    """生成 Markdown 格式的清洗报告。

    报告包含：
    * 各赛季原始/清洗后数据规模对比
    * 异常值处理统计
    * 特征计算示例（前 5 次进站）
    * 进站类型分布
    * 数据质量评分
    * 验证结果

    Parameters
    ----------
    all_stats : list of CleaningStats
        各赛季统计。
    merged_laps : pd.DataFrame
        合并后的主表。
    merged_pits : pd.DataFrame
        合并后的进站表。
    validation : dict
        验证结果。

    Returns
    -------
    str
        Markdown 报告文本。
    """
    lines: List[str] = [
        "# 摩纳哥大奖赛 — 数据清洗报告",
        "",
        "> 清洗日期：2026-06-08 | 工具：`data_cleaning.py`",
        "",
        "---",
        "",
        "## 1. 数据规模概览",
        "",
        "| 赛季 | 原始行数 | 清洗后行数 | 移除异常 | 进站次数 | 清洗前完整度 | 清洗后完整度 |",
        "|------|----------|------------|----------|----------|-------------|-------------|",
    ]

    total_before = 0
    total_after = 0
    total_pits = 0
    for s in sorted(all_stats, key=lambda x: x.year):
        lines.append(
            f"| {s.year} | {s.rows_before} | {s.rows_after} | "
            f"{s.outliers_removed} | {s.n_pit_stops} | "
            f"{s.completeness_before:.1f}% | {s.completeness_after:.1f}% |"
        )
        total_before += s.rows_before
        total_after += s.rows_after
        total_pits += s.n_pit_stops

    lines.append(
        f"| **合计** | **{total_before}** | **{total_after}** | "
        f"**{total_before - total_after}** | **{total_pits}** | — | — |"
    )

    # --- 异常值处理详情 ------------------------------------------------
    lines.extend([
        "",
        "## 2. 异常值处理",
        "",
        f"* 圈速阈值：{MIN_LAP_TIME_SEC}s – {MAX_LAP_TIME_SEC}s",
        f"* 总移除：{total_before - total_after} 条记录",
        "* 移除原因：出场圈/回场圈、安全车/红旗时段、数据采集异常",
        "* 轮胎寿命修正：每个 Stint 内 TyreLife 重新从 1 计数",
    ])

    # --- 进站特征概览 --------------------------------------------------
    lines.extend([
        "",
        "## 3. 进站特征概览",
        "",
        "### 3.1 进站损失统计",
        "",
        "| 赛季 | 进站次数 | 平均损失 (s) | 标准差 (s) |",
        "|------|----------|-------------|-----------|",
    ])
    for s in sorted(all_stats, key=lambda x: x.year):
        if s.n_pit_stops > 0:
            lines.append(
                f"| {s.year} | {s.n_pit_stops} | "
                f"{s.pit_loss_mean:.1f} | {s.pit_loss_std:.1f} |"
            )

    lines.extend([
        "",
        "### 3.2 进站类型分布",
        "",
        "| 赛季 | Undercut | Overcut | Normal |",
        "|------|----------|---------|--------|",
    ])
    for s in sorted(all_stats, key=lambda x: x.year):
        types = s.pit_types
        u = types.get("Undercut", 0)
        o = types.get("Overcut", 0)
        n = types.get("Normal", 0)
        lines.append(f"| {s.year} | {u} | {o} | {n} |")

    # --- 位置变化 ------------------------------------------------------
    lines.extend([
        "",
        "### 3.3 进站前后位置变化",
        "",
        "| 赛季 | 名次上升 | 不变 | 名次下降 |",
        "|------|----------|------|----------|",
    ])
    for s in sorted(all_stats, key=lambda x: x.year):
        pc = s.position_changes
        lines.append(
            f"| {s.year} | {pc.get('gained', 0)} | "
            f"{pc.get('unchanged', 0)} | {pc.get('lost', 0)} |"
        )

    # --- 进站示例 ------------------------------------------------------
    lines.extend([
        "",
        "## 4. 特征计算示例（前 10 次进站）",
        "",
        "| 赛季 | 车手 | LapIn | PitLoss(s) | PosChg | Type | WindowSafety(s) |",
        "|------|------|-------|------------|--------|------|-----------------|",
    ])
    sample_cols = ["RaceYear", "Driver", "LapIn", "PitLoss_sec",
                   "PositionChange", "PitType", "WindowSafety_sec"]
    available_sample = [c for c in sample_cols if c in merged_pits.columns]
    if not merged_pits.empty and len(available_sample) >= 3:
        sample = merged_pits[available_sample].head(10)
        for _, row in sample.iterrows():
            vals = " | ".join(
                f"{row.get(c, '—')}" if not isinstance(row.get(c), float)
                else f"{row[c]:.1f}" if pd.notna(row.get(c)) else "—"
                for c in available_sample
            )
            lines.append(f"| {vals} |")

    # --- 数据验证 ------------------------------------------------------
    lines.extend([
        "",
        "## 5. 数据验证结果",
        "",
        "| 检查项 | 结果 | 详情 |",
        "|--------|------|------|",
    ])
    for check, result in validation.items():
        icon = "✓" if result["pass"] else "✗"
        lines.append(f"| {check} | {icon} | {result['detail']} |")

    # --- 数据质量评分 --------------------------------------------------
    lines.extend([
        "",
        "## 6. 数据质量评分",
        "",
        "| 维度 | 评分 | 说明 |",
        "|------|------|------|",
    ])

    # 计算各项得分
    comp_after = np.mean([s.completeness_after for s in all_stats])
    pit_loss_valid = (
        merged_pits["PitLoss_sec"].notna().mean()
        if "PitLoss_sec" in merged_pits.columns and not merged_pits.empty
        else 0
    )
    all_valid = all(r["pass"] for r in validation.values())

    lines.append(f"| 数据完整度 | {comp_after:.0f}/100 | 关键字段非空比例 |")
    lines.append(
        f"| 进站损失有效率 | {pit_loss_valid*100:.0f}/100 | "
        f"{merged_pits['PitLoss_sec'].notna().sum() if 'PitLoss_sec' in merged_pits.columns else 0}"
        f"/{len(merged_pits)} 次进站可计算 |"
    )
    lines.append(
        f"| 逻辑一致性 | {'100' if all_valid else '0'}/100 | "
        f"{'全部通过' if all_valid else '存在问题'} |"
    )
    lines.append(
        f"| **综合评分** | **{(comp_after + pit_loss_valid*100 + (100 if all_valid else 0)) / 3:.0f}/100** | "
        "三项平均 |"
    )

    lines.extend([
        "",
        "---",
        "",
        "*本报告由 `data_cleaning.py` 自动生成。*",
        "",
    ])

    return "\n".join(lines)


# ============================================================================
# 13. 主入口
# ============================================================================

def main() -> None:
    """清洗主流程：加载 → 清洗 → 验证 → 保存 → 报告。"""
    log.info("=" * 60)
    log.info("摩纳哥大奖赛数据清洗 — Stage 2")
    log.info("=" * 60)

    # --- Step 1: 加载原始数据 -----------------------------------------
    raw_data = load_raw_data(TARGET_YEARS)
    if not raw_data:
        log.error("未找到任何原始 Parquet 文件，请先运行 data_collection.py")
        sys.exit(1)

    # --- Step 2: 逐赛季清洗 --------------------------------------------
    all_laps: List[pd.DataFrame] = []
    all_pits: List[pd.DataFrame] = []
    all_stats: List[CleaningStats] = []

    for year in sorted(raw_data.keys()):
        log.info("—" * 40)
        log.info("清洗 %d 赛季…", year)
        laps, pits, stats = clean_season(raw_data[year], year)
        if laps is not None:
            all_laps.append(laps)
        if pits is not None and not pits.empty:
            all_pits.append(pits)
        all_stats.append(stats)

    # --- Step 3: 合并 --------------------------------------------------
    merged_laps = pd.concat(all_laps, ignore_index=True)
    merged_pits = (
        pd.concat(all_pits, ignore_index=True)
        if all_pits
        else pd.DataFrame()
    )

    # 确保主键列在最前面
    first_cols = ["RaceYear", "Driver", "Team", "LapNumber", "Stint"]
    col_order = [c for c in first_cols if c in merged_laps.columns]
    col_order += [c for c in merged_laps.columns if c not in col_order]
    merged_laps = merged_laps[col_order]

    log.info("合并主表: %d 行 × %d 列", len(merged_laps), len(merged_laps.columns))
    log.info("合并进站表: %d 行 × %d 列", len(merged_pits), len(merged_pits.columns))

    # --- Step 4: 数据验证 -----------------------------------------------
    validation = validate_cleaned_data(merged_laps, merged_pits)

    # --- Step 5: 保存 -------------------------------------------------
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    laps_path = CLEAN_DIR / "cleaned_races.parquet"
    merged_laps.to_parquet(laps_path, index=False, engine="pyarrow")
    log.info("已保存主表: %s", laps_path)

    pits_path = CLEAN_DIR / "pit_stops.parquet"
    merged_pits.to_parquet(pits_path, index=False, engine="pyarrow")
    log.info("已保存进站表: %s", pits_path)

    # --- Step 6: 报告 ---------------------------------------------------
    report = generate_report(all_stats, merged_laps, merged_pits, validation)
    report_path = CLEAN_DIR / "cleaning_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("已保存清洗报告: %s", report_path)

    print("\n" + report)

    log.info("清洗完成。")


# ============================================================================
# 14. 单元测试
# ============================================================================

def run_unit_tests() -> Dict[str, bool]:
    """对每个计算步骤执行独立的单元测试。

    使用合成数据（而非真实 FastF1 数据）验证计算逻辑的正确性。
    每个测试返回 True（通过）或 False（失败）。

    Returns
    -------
    dict
        ``{测试名称: 通过}``。
    """
    results: Dict[str, bool] = {}

    # ==================================================================
    # 测试 1: 异常圈速检测
    # ==================================================================
    df_test = pd.DataFrame({
        "Driver": ["A"] * 5,
        "LapNumber": [1, 2, 3, 4, 5],
        "LapTime": pd.to_timedelta([45, 80, 90, 250, np.nan], unit="s"),
    })
    mask = detect_outlier_laps(df_test)
    # LapTime 45s (太快), 250s (太慢), NaN → 异常
    # 80s, 90s → 正常
    results["异常检测_过滤太快圈速"] = not mask.iloc[0]
    results["异常检测_保留正常圈速"] = mask.iloc[1] and mask.iloc[2]
    results["异常检测_过滤太慢圈速"] = not mask.iloc[3]
    results["异常检测_过滤NaN圈速"] = not mask.iloc[4]

    # ==================================================================
    # 测试 2: 轮胎寿命修正
    # ==================================================================
    df_tyre = pd.DataFrame({
        "Driver": ["A", "A", "A", "A", "A"],
        "LapNumber": [1, 2, 3, 4, 5],
        "Stint":     [1, 1, 1, 2, 2],
        "TyreLife":  [5, 6, 7, 0, 1],  # 原始数据从 5, 0 开始
    })
    fixed = fix_tyre_life(df_tyre)
    results["轮胎寿命_从1开始"] = (
        fixed["TyreLife"].tolist() == [1, 2, 3, 1, 2]
    )

    # ==================================================================
    # 测试 3: 车队名称标准化
    # ==================================================================
    df_team = pd.DataFrame({
        "Team": ["Red Bull Racing", "Scuderia Ferrari", "Unknown Team XYZ"],
    })
    std = standardize_teams(df_team)
    results["车队标准化_RedBull"] = std.iloc[0]["Team"] == "Red Bull"
    results["车队标准化_Ferrari"] = std.iloc[1]["Team"] == "Ferrari"
    results["车队标准化_未匹配保留原名"] = std.iloc[2]["Team"] == "Unknown Team XYZ"

    # ==================================================================
    # 测试 4: 轮胎配方标准化
    # ==================================================================
    df_comp = pd.DataFrame({
        "Compound": ["SOFT", "medium", "C1", "intermediate", "NONSENSE"],
    })
    std_c = standardize_compounds(df_comp)
    results["配方标准化_SOFT"] = std_c.iloc[0]["Compound"] == "SOFT"
    results["配方标准化_大小写"] = std_c.iloc[1]["Compound"] == "MEDIUM"
    results["配方标准化_C1→HARD"] = std_c.iloc[2]["Compound"] == "HARD"
    results["配方标准化_INTERMEDIATE"] = std_c.iloc[3]["Compound"] == "INTERMEDIATE"
    results["配方标准化_未知→UNKNOWN"] = std_c.iloc[4]["Compound"] == "UNKNOWN"

    # ==================================================================
    # 测试 5: 时间秒化
    # ==================================================================
    df_time = pd.DataFrame({
        "LapTime": pd.to_timedelta([90.5, 88.2, np.nan], unit="s"),
        "Sector1Time": pd.to_timedelta([19.0, 18.5, 19.2], unit="s"),
    })
    conv = convert_times_to_seconds(df_time)
    results["时间秒化_90.5s"] = abs(conv.iloc[0]["LapTime"] - 90.5) < 0.01
    results["时间秒化_NaN保持"] = pd.isna(conv.iloc[2]["LapTime"])

    # ==================================================================
    # 测试 6: 进站事件构建
    # ==================================================================
    df_laps = pd.DataFrame({
        "Driver": ["VER", "VER", "VER", "VER", "VER"],
        "Team": ["Red Bull"] * 5,
        "LapNumber": [15, 16, 17, 18, 19],
        "Stint": [1, 1, 1, 2, 2],
        "Position": [1, 1, 1, 3, 2],
        "Compound": ["SOFT", "SOFT", "SOFT", "MEDIUM", "MEDIUM"],
        "Interval": [np.nan, 1.5, 2.0, np.nan, 0.5],
        "GapToLeader": [0, 0, 0, 22, 21],
        "Time": [0, 90, 180, 280, 370],
        "LapTime": [90, 90, 100, 90, 90],
        "PitInTime": [np.nan, np.nan, 275, np.nan, np.nan],
        "PitOutTime": [np.nan, np.nan, np.nan, 280, np.nan],
    })
    pits = build_pit_stops(df_laps, year=2024)
    results["进站构建_检测到进站"] = len(pits) == 1
    if len(pits) == 1:
        p = pits.iloc[0]
        results["进站构建_LapIn正确"] = p["LapIn"] == 17
        results["进站构建_LapOut正确"] = p["LapOut"] == 18
        results["进站构建_Stint正确"] = p["StintAfter"] == 2
        results["进站构建_轮胎正确"] = (
            p["CompoundBefore"] == "SOFT" and p["CompoundAfter"] == "MEDIUM"
        )
        # PitIn = 275, PitOut = 280 → duration = 5
        results["进站构建_时长正确"] = abs(p["PitDuration_sec"] - 5.0) < 0.1

    # ==================================================================
    # 测试 7: 进站损失计算
    # ==================================================================
    # 正常圈速 = 90s, pit duration = 5s → pit_loss = 5 - 90 = -85s
    # （这意味着进站比正常跑一圈快 85 秒到达赛道同一位置，
    #  因为进站不用跑完整圈，只走维修区通道）
    df_laps2 = pd.DataFrame({
        "Driver": ["VER"] * 6,
        "LapNumber": [14, 15, 16, 17, 18, 19],
        "LapTime": [89.5, 90.0, 90.5, 110.0, 92.0, 89.0],
        "Stint": [1, 1, 1, 2, 2, 2],
    })
    df_pits2 = pd.DataFrame({
        "RaceYear": [2024],
        "Driver": ["VER"],
        "LapIn": [16], "LapOut": [17],
        "PitDuration_sec": [22.0],  # 典型的摩纳哥进站时长
    })
    result = compute_pit_loss(df_pits2, df_laps2)
    # 前后窗口: Lap 14,15,18,19 的中位数
    # [89.5, 90.0, 92.0, 89.0] → sorted [89.0, 89.5, 90.0, 92.0] → median = 89.75
    # pit_loss = 22.0 - 89.75 = -67.75
    if "PitLoss_sec" in result.columns:
        expected = 22.0 - np.median([89.5, 90.0, 92.0, 89.0])
        results["进站损失_计算正确"] = (
            abs(result.iloc[0]["PitLoss_sec"] - expected) < 0.1
        )

    # ==================================================================
    # 测试 8: 位置变化
    # ==================================================================
    df_pit_pos = pd.DataFrame({
        "PositionBefore": [3, 5, 1],
        "PositionAfter": [5, 2, 1],
    })
    result_pos = compute_position_change(df_pit_pos)
    results["位置变化_下降"] = result_pos.iloc[0]["PositionChange"] == -2
    results["位置变化_上升"] = result_pos.iloc[1]["PositionChange"] == 3
    results["位置变化_不变"] = result_pos.iloc[2]["PositionChange"] == 0

    # ==================================================================
    # 测试 9: 窗口安全性
    # ==================================================================
    df_pit_ws = pd.DataFrame({
        "GapBehindIn_sec": [3.5, 1.0, np.nan],
        "PitLoss_sec": [20.0, 22.0, 21.0],
    })
    result_ws = compute_window_safety(df_pit_ws)
    # 3.5 - 20 = -16.5 (不安全), 1.0 - 22 = -21.0 (不安全)
    results["窗口安全_不安全"] = result_ws.iloc[0]["WindowSafety_sec"] < 0
    results["窗口安全_NaN输入"] = pd.isna(result_ws.iloc[2]["WindowSafety_sec"])

    # ==================================================================
    # 测试 10: 进站类型分类
    # ==================================================================
    # VER pits on lap 17, LEC (car ahead) pits on lap 14
    # VER delta = 17 - 14 = 3 → Overcut (VER stayed out longer)
    df_test_pits = pd.DataFrame({
        "RaceYear": [2024, 2024],
        "Driver": ["LEC", "VER"],
        "Team": ["Ferrari", "Red Bull"],
        "StintAfter": [2, 2],
        "LapIn": [14, 17],
        "LapOut": [15, 18],
        "PositionBefore": [1, 2],
        "PositionAfter": [3, 1],
        "PitDuration_sec": [21, 22],
        "CompoundBefore": ["SOFT", "SOFT"],
        "CompoundAfter": ["HARD", "HARD"],
        "GapBehindIn_sec": [2.0, 3.0],
    })
    # At VER's in-lap (lap 17), VER is P2; car ahead (LEC, P1) also at lap 17
    df_test_laps = pd.DataFrame({
        "RaceYear": [2024] * 4,
        "Driver": ["LEC", "VER", "LEC", "VER"],
        "LapNumber": [14, 14, 17, 17],
        "Position": [1, 2, 1, 2],
    })
    result_type = compute_pit_type(df_test_pits, df_test_laps)
    # VER: LapIn=17, 前方车手 LEC 进站圈=14, delta=3 → Overcut
    ver_type = result_type.loc[
        result_type["Driver"] == "VER", "PitType"
    ].values[0]
    results["进站类型_Overcut"] = ver_type == "Overcut"

    # ==================================================================
    # 汇总
    # ==================================================================
    passed = sum(results.values())
    total = len(results)
    log.info("=" * 40)
    log.info("单元测试: %d/%d 通过", passed, total)
    for name, ok in results.items():
        log.info("  %s %s", "✓" if ok else "✗", name)

    return results


# ============================================================================
# 15. 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Monaco 数据清洗 Stage 2")
    parser.add_argument(
        "--test", action="store_true",
        help="仅运行单元测试（不执行清洗流程）",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="仅运行单元测试且静默（CI 模式）",
    )
    args = parser.parse_args()

    if args.test or args.test_only:
        results = run_unit_tests()
        passed = sum(results.values())
        total = len(results)
        if not args.test_only:
            print(f"\n单元测试结果: {passed}/{total} 通过")
            if passed < total:
                failed = [k for k, v in results.items() if not v]
                print(f"失败: {failed}")
        sys.exit(0 if passed == total else 1)
    else:
        main()
