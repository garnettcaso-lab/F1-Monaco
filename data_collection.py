#!/usr/bin/env python3
"""
F1 Monaco Grand Prix — Multi-Season Lap-Level Data Collection
===============================================================

Purpose
-------
面向课程论文研究，通过 **FastF1 v3.x** API 采集 2019、2021、2022、2023、2024
五个赛季的摩纳哥大奖赛正赛逐圈数据，输出结构化 Parquet 文件及汇总统计表。

核心功能
--------
1. ``fetch_monaco_data(years)`` — 主采集管道，加载会话 → 提取圈速 →
   计算差距 → 检测进站 → 合并导出。
2. 自动跳过 2020 年（摩纳哥站因 COVID-19 取消）。
3. 内置指数退避重试 & 缓存机制，避免重复下载。
4. 每个赛季独立输出 ``monaco_<year>.parquet`` + 汇总 Markdown 表格。

依赖
----
* Python ≥ 3.9
* fastf1 ≥ 3.0
* pandas, numpy, tqdm, pyarrow (Parquet 写入)

作者 : 课程论文研究
日期 : 2026-06-08
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fastf1 as ff1
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# 全局配置 — 可按需修改路径
# ============================================================================

# FastF1 缓存目录，避免重复下载同一赛季数据
CACHE_DIR: Path = Path("./f1_cache")
# 最终输出目录（Parquet 文件 + 汇总表 + 日志）
OUTPUT_DIR: Path = Path("./output")

# 默认目标赛季（2020 摩纳哥站取消）
DEFAULT_YEARS: List[int] = [2019, 2021, 2022, 2023, 2024]

# 摩纳哥大奖赛在 FastF1 中的事件名称候选列表
# 不同赛季可能使用不同命名，逐一匹配
MONACO_EVENT_CANDIDATES: List[str] = [
    "Monaco Grand Prix",
    "Monaco",
    "Grand Prix de Monaco",
    "Monte Carlo",
]

# 数据完整度计算所使用的关键字段
# 这些字段对分析至关重要，任一缺失都会降低完整度指标
COMPLETENESS_KEY_COLUMNS: List[str] = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Compound",
    "TyreLife",
]

# 网络重试参数
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 2.0  # 指数退避的底数（秒）


# ============================================================================
# 日志系统
# ============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """初始化双通道日志：控制台 + 文件。

    日志文件写入 ``OUTPUT_DIR/collection.log``，保留每次运行的完整审计跟踪，
    方便排查数据缺失或 API 异常。

    Parameters
    ----------
    level : int
        控制台日志级别（文件始终记录 DEBUG 级别）。

    Returns
    -------
    logging.Logger
        配置好的全局 logger 实例。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("monaco_collector")
    logger.setLevel(logging.DEBUG)  # logger 本身捕获所有级别

    # 防止重复添加 handler（多次 import 场景）
    if logger.handlers:
        return logger

    # 日志格式：时间 + 级别 + 消息
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # --- 控制台 handler ---------------------------------------------------
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # --- 文件 handler -----------------------------------------------------
    log_path = OUTPUT_DIR / "collection.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件保留完整调试信息
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("日志系统初始化完成 — 文件位于 %s", log_path)
    return logger


# 模块级 logger，所有函数通过 ``log.xxx()`` 输出
log = setup_logging()


# ============================================================================
# 工具函数
# ============================================================================

def _backoff_delay(attempt: int) -> float:
    """计算第 *attempt* 次重试的等待时间。

    使用指数退避 + 随机抖动，避免多个并发请求同时重试造成"惊群效应"。

    Parameters
    ----------
    attempt : int
        当前重试次数（从 1 开始）。

    Returns
    -------
    float
        等待秒数 = ``2^attempt + random(0, 0.5)``。
    """
    return (RETRY_BACKOFF_BASE ** attempt) + np.random.uniform(0, 0.5)


def retry_call(
    func,
    *args,
    max_retries: int = MAX_RETRIES,
    description: str = "",
    **kwargs,
) -> Any:
    """带指数退避的函数调用重试包装器。

    用于包装 FastF1 API 调用，处理瞬时网络故障。
    每次失败后等待时间翻倍，最多重试 *max_retries* 次。

    Parameters
    ----------
    func : callable
        要调用的函数。
    max_retries : int
        最大重试次数（含首次尝试）。
    description : str
        可读描述，用于日志输出。

    Returns
    -------
    Any
        *func(*args, **kwargs)* 的返回值。

    Raises
    ------
    Exception
        所有重试均失败时抛出最后一次的异常。
    """
    label = description or func.__name__
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                log.error(
                    "%s — 重试 %d 次后仍失败: %s", label, max_retries, exc
                )
                raise
            wait = _backoff_delay(attempt)
            log.warning(
                "%s — 第 %d/%d 次尝试失败 (%s)，%.1f 秒后重试…",
                label, attempt, max_retries, exc, wait,
            )
            time.sleep(wait)

    # 逻辑上不可达，但保持类型检查器满意
    raise RuntimeError(f"Unexpected retry exhaustion: {label}") from last_exc


# ============================================================================
# 赛事发现与会话加载
# ============================================================================

def _find_monaco_event(year: int) -> Optional[pd.Series]:
    """在赛季赛历中定位摩纳哥大奖赛。

    遍历 FastF1 赛历，按 ``MONACO_EVENT_CANDIDATES`` 中的名称子串匹配。
    2020 年摩纳哥不在赛历中，返回 ``None``。

    Parameters
    ----------
    year : int
        赛季年份。

    Returns
    -------
    pd.Series or None
        匹配到的事件行（包含 EventName 等字段），或 None。
    """
    try:
        # FastF1 v3.x: get_event_schedule 返回可迭代的 EventSchedule 对象
        schedule = retry_call(
            ff1.get_event_schedule, year, description=f"获取 {year} 赛历"
        )
    except Exception:
        log.exception("无法加载 %d 赛季赛历", year)
        return None

    # 遍历赛历中的每一站
    for _, event_row in schedule.iterrows():
        event_name: str = str(event_row["EventName"])
        for candidate in MONACO_EVENT_CANDIDATES:
            if candidate.lower() in event_name.lower():
                log.info("  ✓ 找到摩纳哥站: '%s' (第 %s 站)",
                         event_name, event_row.get("RoundNumber", "?"))
                return event_row

    log.warning("%d — 赛历中未找到摩纳哥站", year)
    return None


def load_monaco_session(year: int) -> Optional[ff1.core.Session]:
    """加载指定年份摩纳哥大奖赛的正赛会话。

    完整的加载流程：发现赛事 → 创建会话对象 → 调用 ``session.load()``
    拉取圈速 & 遥测数据。全程带重试保护。

    Parameters
    ----------
    year : int
        赛季年份（2020 将直接返回 None）。

    Returns
    -------
    ff1.core.Session or None
        加载完成的 Session 对象，或 None（赛事取消 / 加载失败）。
    """
    # 2020 年摩纳哥站因 COVID-19 取消
    if year == 2020:
        log.warning("%d — 摩纳哥大奖赛因 COVID-19 取消，跳过", year)
        return None

    # Step 1: 在赛历中定位摩纳哥站
    event_row = _find_monaco_event(year)
    if event_row is None:
        return None

    event_name: str = str(event_row["EventName"])

    # Step 2: 获取正赛会话对象（'R' = Race）
    # FastF1 v3.x 支持传入事件名称或轮次编号
    session = ff1.get_session(year, event_name, "R")
    log.info("%d — 开始加载正赛数据 (Event: %s)…", year, event_name)

    # Step 3: 加载数据（带重试）
    try:
        retry_call(session.load, description=f"加载 {year} 正赛数据")
    except Exception:
        log.exception("%d — 会话加载失败", year)
        return None

    log.info("%d — 正赛数据加载完成", year)
    return session


# ============================================================================
# 圈速数据提取与清洗
# ============================================================================

def _build_driver_name_map(session: ff1.core.Session) -> Dict[str, str]:
    """从会话结果构建车手缩写 → 全名的映射字典。

    FastF1 的圈速数据使用三字母缩写（如 VER、HAM），输出中需要完整姓名。
    此函数遍历 ``session.results`` 构建映射表。

    Parameters
    ----------
    session : ff1.core.Session
        已加载的会话。

    Returns
    -------
    dict
        ``{缩写: 全名}`` 映射。
    """
    results = session.results
    name_map: Dict[str, str] = {}

    for _, row in results.iterrows():
        abbr: str = str(row["Abbreviation"])

        # 优先使用 FullName 字段（FastF1 v3.1+）
        if "FullName" in results.columns and pd.notna(row.get("FullName")):
            name_map[abbr] = str(row["FullName"])
        else:
            # 回退：拼接 FirstName + LastName
            first = str(row.get("FirstName", ""))
            last = str(row.get("LastName", ""))
            name_map[abbr] = f"{first} {last}".strip()

    return name_map


def extract_lap_data(session: ff1.core.Session) -> pd.DataFrame:
    """从已加载的会话中提取并清洗圈速数据。

    处理步骤：
    1. 从 ``session.laps`` 获取逐圈 DataFrame
    2. 将车手缩写映射为全名
    3. 选择目标列（存在性检查，兼容不同 FastF1 版本）
    4. 按车手和圈号排序

    Parameters
    ----------
    session : ff1.core.Session
        已加载的正赛会话。

    Returns
    -------
    pd.DataFrame
        清洗后的圈速数据，包含以下核心列：
        Driver, Team, LapNumber, LapTime, Position,
        Sector1Time, Sector2Time, Sector3Time,
        Compound, TyreLife, Stint, Time
    """
    # --- 获取原始圈速数据 ------------------------------------------------
    # session.laps 是一个带缓存的属性，返回 pandas DataFrame
    laps: pd.DataFrame = session.laps.copy()

    # --- 车手名称映射 ----------------------------------------------------
    name_map = _build_driver_name_map(session)
    # Driver 列存储的是缩写，替换为全名
    laps["Driver"] = laps["Driver"].map(name_map).fillna(laps["Driver"])

    # --- 目标列清单 ------------------------------------------------------
    # 列出我们需要的所有列；实际只保留存在的那些
    desired_columns = [
        "Driver",          # 车手全名
        "Team",            # 车队名称
        "LapNumber",       # 圈号（从 1 开始）
        "LapTime",         # 单圈时间（timedelta）
        "Position",        # 冲线位置（该圈结束时）
        "Sector1Time",     # 第一段计时
        "Sector2Time",     # 第二段计时
        "Sector3Time",     # 第三段计时
        "Compound",        # 轮胎配方（SOFT/MEDIUM/HARD/INTERMEDIATE/WET）
        "TyreLife",        # 轮胎已使用圈数
        "Stint",           # 当前进站段编号（从 1 开始）
        "Time",            # 会话时间戳（timedelta，该圈开始时）
        "LapStartTime",    # 备选列名（部分 FastF1 版本使用）
        "PitInTime",       # 进站时刻（如果 FastF1 提供）
        "PitOutTime",      # 出站时刻（如果 FastF1 提供）
    ]

    # 交集筛选：只保留 laps 中实际存在的列
    available = [c for c in desired_columns if c in laps.columns]
    laps = laps[available].copy()

    # --- 时间列统一 ------------------------------------------------------
    # 确保有 "Time" 列用于后续差距计算
    if "Time" not in laps.columns and "LapStartTime" in laps.columns:
        laps["Time"] = laps["LapStartTime"]
        log.info("  使用 LapStartTime 作为 Time 列")

    # --- 排序 ------------------------------------------------------------
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    n_drivers = laps["Driver"].nunique()
    n_laps = len(laps)
    log.info("  圈速数据提取完成: %d 位车手, %d 条圈速记录", n_drivers, n_laps)

    return laps


# ============================================================================
# 差距计算
# ============================================================================

def calculate_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """计算每圈的 **GapToLeader**（与领先者差距）和 **Interval**（与前车间距）。

    **计算逻辑（逐圈）**
    1. 对每位车手，累加其所有已完成圈的 LapTime 得到 **累计比赛时间**。
    2. 在同一圈内，累计时间最小的车手即为该圈的"领先者"。
    3. ``GapToLeader = 车手累计时间 − 领先者累计时间``。
    4. 在同一圈内按累计时间升序排列，相邻车手的时间差即为 ``Interval``。
       领先者的 Interval 为 NaN（前方无车）。

    **关于 Lap 1**
    第一圈的 Interval 为发车格位间距的近似值——这是 F1 计时标准做法，
    并非赛道上的实时间距，使用时需注明。

    Parameters
    ----------
    laps : pd.DataFrame
        圈速数据（需含 Driver, LapNumber, LapTime 列）。

    Returns
    -------
    pd.DataFrame
        增加 GapToLeader 和 Interval 列的 DataFrame。
    """
    df = laps.copy()

    # --- 确保 LapTime 为 timedelta 类型 --------------------------------
    if "LapTime" not in df.columns:
        log.warning("LapTime 列缺失，无法计算差距")
        df["GapToLeader"] = pd.NaT
        df["Interval"] = pd.NaT
        return df

    if not pd.api.types.is_timedelta64_dtype(df["LapTime"]):
        df["LapTime"] = pd.to_timedelta(df["LapTime"])

    # --- 累计比赛时间 ---------------------------------------------------
    # 每位车手从 Lap 1 到当前圈的 LapTime 累加
    df = df.sort_values(["Driver", "LapNumber"])
    df["_cum_time"] = df.groupby("Driver")["LapTime"].transform(
        lambda x: x.cumsum()
    )

    # --- 与领先者的差距 -------------------------------------------------
    # 每圈的最小累计时间 = 该圈领先者的累计时间
    min_per_lap = df.groupby("LapNumber")["_cum_time"].transform("min")
    df["GapToLeader"] = df["_cum_time"] - min_per_lap

    # --- 与前车的间隔 ---------------------------------------------------
    # 在同圈内按累计时间升序，相邻车手差即为间隔
    df = df.sort_values(["LapNumber", "_cum_time"])
    prev_cum = df.groupby("LapNumber")["_cum_time"].shift(1)
    df["Interval"] = df["_cum_time"] - prev_cum
    # 每圈领先者（第一条记录）的 Interval 正确为 NaT

    # --- 清理临时列 -----------------------------------------------------
    df = df.drop(columns=["_cum_time"])

    log.info("  GapToLeader & Interval 计算完成")
    return df


# ============================================================================
# 进站检测
# ============================================================================

def detect_pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    """从圈速数据推断进站事件。

    **检测规则**
    当车手的 ``Stint`` 编号递增时，认为发生了一次进站。
    进站发生在上一段（Stint n）的最后一圈与当前段（Stint n+1）的第一圈之间。

    **时间估算（当 FastF1 未提供精确时间时）**
    * ``PitInTime``  = In-Lap 开始时间 + In-Lap 圈时（≈ 进入维修区通道的时刻）
    * ``PitOutTime`` = Out-Lap 开始时间（≈ 离开维修区通道的时刻）

    此方法的误差量级约为维修区通道通行时间（摩纳哥约 18-20 秒），
    对于学术分析是可接受的。

    注意：如果 FastF1 已在圈速数据中提供了 PitInTime/PitOutTime，
    则优先使用 API 提供的精确值。

    Parameters
    ----------
    laps : pd.DataFrame
        圈速数据（需含 Driver, LapNumber, Stint, Time, LapTime 列）。

    Returns
    -------
    pd.DataFrame
        进站事件表，字段：
        Driver, Stint, LapIn, LapOut, PitInTime, PitOutTime,
        CompoundBefore, CompoundAfter
    """
    # --- 优先检查 FastF1 是否已提供进站时间 -----------------------------
    has_pit_cols = (
        "PitInTime" in laps.columns and "PitOutTime" in laps.columns
    )
    f1_pit_available = False
    if has_pit_cols:
        n_pit_in = laps["PitInTime"].notna().sum()
        n_pit_out = laps["PitOutTime"].notna().sum()
        if n_pit_in > 0 and n_pit_out > 0:
            f1_pit_available = True
            log.info("  FastF1 已提供 PitInTime/PitOutTime，直接使用")

    # --- 如果 FastF1 未提供，从 Stint 转换推断 ---------------------------
    if not f1_pit_available:
        if "Stint" not in laps.columns:
            log.warning("  Stint 列缺失，无法检测进站")
            return pd.DataFrame(
                columns=["Driver", "Stint", "LapIn", "LapOut",
                         "PitInTime", "PitOutTime",
                         "CompoundBefore", "CompoundAfter"]
            )

        if "Time" not in laps.columns:
            log.warning("  Time 列缺失，无法估算进站时刻")
            return pd.DataFrame(
                columns=["Driver", "Stint", "LapIn", "LapOut",
                         "PitInTime", "PitOutTime",
                         "CompoundBefore", "CompoundAfter"]
            )

        log.info("  从 Stint 转换推断进站事件…")
        return _detect_pits_from_stint(laps)

    # --- 从 FastF1 直接提取进站信息 ------------------------------------
    return _extract_pits_from_f1(laps)


def _detect_pits_from_stint(laps: pd.DataFrame) -> pd.DataFrame:
    """通过 Stint 编号变化推断进站事件（回退方案）。

    对每位车手，检测 Stint 编号从 n 跳到 n+1 的位置，
    提取相邻两段间的进站信息。
    """
    df = laps.sort_values(["Driver", "LapNumber"]).copy()

    # 标记每段的第一圈 (out-lap) 和最后一圈 (in-lap)
    # NextStint: 下一圈的 Stint；PrevStint: 上一圈的 Stint
    df["_next_stint"] = df.groupby("Driver")["Stint"].shift(-1)
    df["_prev_stint"] = df.groupby("Driver")["Stint"].shift(1)

    # out-lap: Stint 不同于上一圈的 Stint（且不是第一圈数据）
    is_out_lap = (df["Stint"] != df["_prev_stint"]) & df["_prev_stint"].notna()
    # in-lap: Stint 不同于下一圈的 Stint（且不是最后一圈数据）
    is_in_lap = (df["Stint"] != df["_next_stint"]) & df["_next_stint"].notna()

    pit_records: List[Dict[str, Any]] = []

    # 遍历所有 out-lap（每次进站后驶出维修区的第一圈）
    out_laps = df[is_out_lap]
    for _, out_row in out_laps.iterrows():
        driver = out_row["Driver"]
        new_stint = int(out_row["Stint"])        # 新 Stint 编号
        out_time = out_row["Time"]               # out-lap 开始时刻
        lap_out = int(out_row["LapNumber"])      # out-lap 圈号
        compound_after = out_row.get("Compound", None)

        # 查找对应的 in-lap：同一车手 Stint = new_stint - 1 的最后一圈
        in_candidates = df[
            (df["Driver"] == driver)
            & (df["Stint"] == new_stint - 1)
            & is_in_lap
        ]
        if in_candidates.empty:
            # 理论不应发生，但做防御性处理
            pit_in_time = pd.NaT
            lap_in = np.nan
            compound_before = None
        else:
            in_row = in_candidates.iloc[-1]       # in-lap
            in_start = in_row["Time"]             # in-lap 开始时刻
            in_lap_time = in_row["LapTime"]       # in-lap 圈时
            # PitInTime = in-lap 开始时刻 + in-lap 圈时
            if pd.notna(in_start) and pd.notna(in_lap_time):
                pit_in_time = in_start + in_lap_time
            else:
                pit_in_time = pd.NaT
            lap_in = int(in_row["LapNumber"])
            compound_before = in_row.get("Compound", None)

        pit_records.append({
            "Driver": driver,
            "Stint": new_stint,                    # 进站后所在的 Stint
            "LapIn": lap_in,                       # 进站前最后一圈
            "LapOut": lap_out,                     # 进站后第一圈
            "PitInTime": pit_in_time,              # 估算进站时刻
            "PitOutTime": out_time,                # 估算出站时刻
            "CompoundBefore": compound_before,     # 进站前轮胎
            "CompoundAfter": compound_after,       # 进站后轮胎
        })

    pit_df = pd.DataFrame(pit_records)

    if pit_df.empty:
        log.info("  未检测到进站事件")
    else:
        n_pits = len(pit_df)
        n_drv = pit_df["Driver"].nunique()
        log.info("  检测到 %d 次进站 (%d 位车手)", n_pits, n_drv)

    return pit_df


def _extract_pits_from_f1(laps: pd.DataFrame) -> pd.DataFrame:
    """从 FastF1 原生 PitInTime/PitOutTime 字段提取进站信息。

    识别条件：PitInTime 非空的圈 = 该圈为 in-lap。
    """
    df = laps.sort_values(["Driver", "LapNumber"]).copy()
    pit_records: List[Dict[str, Any]] = []

    for driver, grp in df.groupby("Driver"):
        # 筛选有进站记录的圈
        pit_laps = grp[grp["PitInTime"].notna() | grp["PitOutTime"].notna()]
        for _, row in pit_laps.iterrows():
            stint_val = row.get("Stint", np.nan)
            pit_records.append({
                "Driver": driver,
                "Stint": int(stint_val) if pd.notna(stint_val) else np.nan,
                "LapIn": row.get("LapNumber", np.nan),
                "LapOut": row.get("LapNumber", np.nan),
                "PitInTime": row.get("PitInTime", pd.NaT),
                "PitOutTime": row.get("PitOutTime", pd.NaT),
                "CompoundBefore": row.get("Compound", None),
                "CompoundAfter": row.get("Compound", None),
            })

    pit_df = pd.DataFrame(pit_records)
    n_pits = len(pit_df)
    log.info("  从 FastF1 提取 %d 条进站记录", n_pits)
    return pit_df


# ============================================================================
# 圈速与进站数据合并
# ============================================================================

def merge_lap_pit_data(laps: pd.DataFrame, pits: pd.DataFrame) -> pd.DataFrame:
    """将进站上下文信息合并到逐圈数据中。

    每一项圈速记录关联其所属 Stint 的进站信息。
    Stint 1 的圈速对应首次发车（无进站），PitInTime/PitOutTime 为 NaT。

    Parameters
    ----------
    laps : pd.DataFrame
        圈速数据。
    pits : pd.DataFrame
        进站事件数据。

    Returns
    -------
    pd.DataFrame
        增加 PitInTime、PitOutTime 列的圈速数据。
    """
    df = laps.copy()

    # 确保目标列存在，初始化为 NaT
    for col in ["PitInTime", "PitOutTime"]:
        if col not in df.columns:
            df[col] = pd.NaT

    if pits.empty:
        return df

    # --- 构建查找表: (Driver, Stint) → {PitInTime, PitOutTime} ---------
    # 每个 Stint（≥2）对应其开始前的进站时刻
    pit_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for _, pit_row in pits.iterrows():
        driver = str(pit_row["Driver"])
        stint = int(pit_row["Stint"]) if pd.notna(pit_row["Stint"]) else 0
        if stint <= 0:
            continue
        pit_lookup[(driver, stint)] = {
            "PitInTime": pit_row.get("PitInTime", pd.NaT),
            "PitOutTime": pit_row.get("PitOutTime", pd.NaT),
        }

    # --- 逐行注释 --------------------------------------------------------
    pit_in_list: List[Any] = []
    pit_out_list: List[Any] = []

    for _, lap_row in df.iterrows():
        driver = str(lap_row["Driver"])
        stint = int(lap_row["Stint"]) if pd.notna(lap_row.get("Stint")) else 1
        info = pit_lookup.get((driver, stint), {})
        pit_in_list.append(info.get("PitInTime", pd.NaT))
        pit_out_list.append(info.get("PitOutTime", pd.NaT))

    df["PitInTime"] = pit_in_list
    df["PitOutTime"] = pit_out_list

    return df


# ============================================================================
# 数据完整度
# ============================================================================

def compute_completeness(df: pd.DataFrame) -> float:
    """计算关键分析字段的非空比例（百分比）。

    对 ``COMPLETENESS_KEY_COLUMNS`` 中的每个字段计算非空比例，
    然后取平均值作为整体完整度指标。

    100% 表示所有关键字段在所有行中都有值；
    < 100% 表示存在数据缺口（如某赛季缺少扇区时间或轮胎数据）。

    Parameters
    ----------
    df : pd.DataFrame
        待评估的圈速数据。

    Returns
    -------
    float
        整体数据完整度百分比 [0, 100]。
    """
    # 只检查 DataFrame 中实际存在的列
    available_cols = [c for c in COMPLETENESS_KEY_COLUMNS if c in df.columns]
    if not available_cols:
        return 0.0

    ratios: List[float] = []
    total = len(df)
    if total == 0:
        return 0.0

    for col in available_cols:
        non_null = df[col].notna().sum()
        ratios.append(non_null / total)

    return float(np.mean(ratios) * 100)


# ============================================================================
# 主采集管道
# ============================================================================

def fetch_monaco_data(years: List[int]) -> Dict[int, pd.DataFrame]:
    """**主函数** — 采集指定年份的摩纳哥大奖赛正赛数据。

    对每一年执行完整的采集管道：
    加载会话 → 提取圈速 → 计算差距 → 检测进站 → 合并 → 质量评估

    Parameters
    ----------
    years : list of int
        要采集的赛季列表，如 [2019, 2021, 2022, 2023, 2024]。
        2020 会自动跳过（摩纳哥站取消）。

    Returns
    -------
    dict
        ``{year: DataFrame}`` — 每个赛季对应一个完整的逐圈 DataFrame。
        键按年份升序排列。

    Examples
    --------
    >>> data = fetch_monaco_data([2019, 2021])
    >>> df_2019 = data[2019]
    >>> df_2019.columns
    Index(['Driver', 'Team', 'LapNumber', 'LapTime', 'Position', 'Sector1Time',
           'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife', 'Stint', 'Time',
           'LapStartTime', 'PitInTime', 'PitOutTime', 'GapToLeader', 'Interval'],
          dtype='object')
    """
    result: Dict[int, pd.DataFrame] = {}

    # --- 启用 FastF1 缓存 ------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(CACHE_DIR))
    log.info("FastF1 缓存目录: %s (缓存已启用)", CACHE_DIR.resolve())

    # --- 逐赛季采集 -------------------------------------------------------
    # 外层进度条：赛季级别
    pbar_years = tqdm(years, desc="赛季进度", unit="season")

    for year in pbar_years:
        pbar_years.set_postfix_str(f"处理 {year}")

        # ---- 加载会话 ---------------------------------------------------
        session = load_monaco_session(year)
        if session is None:
            log.warning("%d — 跳过（无可用数据）", year)
            continue

        # ---- 四步处理管线（内层进度条）----------------------------------
        with tqdm(
            total=4, desc=f"  {year} 处理步骤", leave=False
        ) as step_bar:

            # Step 1: 提取圈速
            step_bar.set_description_str("  提取圈速")
            laps = extract_lap_data(session)
            step_bar.update(1)

            # Step 2: 计算差距指标
            step_bar.set_description_str("  计算差距")
            laps = calculate_gaps(laps)
            step_bar.update(1)

            # Step 3: 检测进站事件
            step_bar.set_description_str("  检测进站")
            pits = detect_pit_stops(laps)
            step_bar.update(1)

            # Step 4: 合并进站信息到圈速数据
            step_bar.set_description_str("  合并数据")
            laps = merge_lap_pit_data(laps, pits)
            step_bar.update(1)

        # ---- 质量检查 ---------------------------------------------------
        n_drivers = laps["Driver"].nunique() if "Driver" in laps.columns else 0
        n_laps = len(laps)
        completeness = compute_completeness(laps)

        log.info(
            "%d — 采集完成: %d 位车手, %d 条圈速, 完整度 %.1f%%",
            year, n_drivers, n_laps, completeness,
        )

        # 完整性不足时告警
        if completeness < 70.0:
            log.warning(
                "%d — 数据完整度仅 %.1f%%，请检查 FastF1 缓存或网络状况",
                year, completeness,
            )

        # 进站时间全部缺失时告警
        if "PitOutTime" in laps.columns:
            n_pit_out = laps["PitOutTime"].notna().sum()
            if n_pit_out == 0:
                log.warning(
                    "%d — PitInTime/PitOutTime 全部为 NaT，"
                    "此赛季可能不支持进站时刻数据", year,
                )

        result[year] = laps

    # --- 汇总 -----------------------------------------------------------
    n_collected = len(result)
    log.info("=" * 50)
    log.info("采集完毕: 成功获取 %d/%d 个赛季的数据", n_collected, len(years))
    for yr in sorted(result.keys()):
        log.info("  %d — %d 行", yr, len(result[yr]))

    return result


# ============================================================================
# 输出
# ============================================================================

def save_parquet_files(data: Dict[int, pd.DataFrame]) -> List[Path]:
    """将每个赛季的 DataFrame 保存为 Parquet 文件。

    Parquet 格式保留列类型和元数据，适合后续 pandas 直接读取分析。

    Parameters
    ----------
    data : dict
        ``{year: DataFrame}`` 映射。

    Returns
    -------
    list of Path
        已保存的文件路径列表。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for year in sorted(data.keys()):
        df = data[year]
        file_path = OUTPUT_DIR / f"monaco_{year}.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")
        saved.append(file_path)
        log.info("已保存: %s (%d 行 × %d 列)",
                 file_path, len(df), len(df.columns))

    return saved


def generate_summary_markdown(data: Dict[int, pd.DataFrame]) -> str:
    """生成数据采集汇总统计的 Markdown 表格。

    每行包含一个赛季的：
    * 车手数量
    * 总圈数记录
    * 进站次数（按 PitInTime 非空计数）
    * 数据完整度百分比

    Parameters
    ----------
    data : dict
        ``{year: DataFrame}`` 映射。

    Returns
    -------
    str
        Markdown 格式的汇总文本，可直接写入 .md 文件。
    """
    lines: List[str] = [
        "# 摩纳哥大奖赛 — 数据采集汇总统计",
        "",
        "> 数据来源：FastF1 API  "
        "> 采集日期：2026-06-08  "
        "> 说明：2020 年摩纳哥大奖赛因 COVID-19 疫情取消，不在采集范围内。",
        "",
        "## 各赛季概况",
        "",
        "| 赛季 | 车手数 | 总圈数 | 进站次数 | 数据完整度 (%) |",
        "|------|--------|--------|----------|----------------|",
    ]

    for year in sorted(data.keys()):
        df = data[year]

        # 车手数量（Driver 列唯一值计数）
        n_drivers = df["Driver"].nunique() if "Driver" in df.columns else 0

        # 总圈数（DataFrame 行数）
        n_laps = len(df)

        # 进站次数 — 统计 PitInTime 非空的行数
        # PitInTime 在合并后标注了每圈对应的进站时刻
        if "PitInTime" in df.columns:
            # 每次进站可能在多圈中重复出现（同一 Stint 的所有圈）
            # 因此统计唯一的 (Driver, PitInTime) 组合
            pit_mask = df["PitInTime"].notna()
            n_pits = df.loc[pit_mask, ["Driver", "PitInTime"]].drop_duplicates().shape[0]
        elif "Stint" in df.columns:
            # 回退方案：统计 Stint > 1 的 out-lap 数量
            out_laps = df.groupby("Driver")["Stint"].diff().fillna(0)
            n_pits = int((out_laps > 0).sum())
        else:
            n_pits = 0

        # 数据完整度
        completeness = compute_completeness(df)

        lines.append(
            f"| {year} | {n_drivers} | {n_laps} | {n_pits} | {completeness:.1f} |"
        )

    # --- 数据字典 --------------------------------------------------------
    lines.extend([
        "",
        "## 字段说明",
        "",
        "| 字段 | 类型 | 说明 |",
        "|------|------|------|",
        "| Driver | str | 车手全名 |",
        "| Team | str | 车队名称 |",
        "| LapNumber | int | 圈号（从 1 开始） |",
        "| LapTime | timedelta | 单圈时间 |",
        "| Position | int | 该圈结束时的赛道位置 |",
        "| Sector1Time | timedelta | 第一计时段时间 |",
        "| Sector2Time | timedelta | 第二计时段时间 |",
        "| Sector3Time | timedelta | 第三计时段时间 |",
        "| Compound | str | 轮胎配方 (SOFT/MEDIUM/HARD/INTERMEDIATE/WET) |",
        "| TyreLife | float | 轮胎已使用圈数 |",
        "| Stint | int | 进站段编号（从 1 开始） |",
        "| Time | timedelta | 该圈开始时的会话时间 |",
        "| GapToLeader | timedelta | 与领先者的累计时间差 |",
        "| Interval | timedelta | 与前车的累计时间差 |",
        "| PitInTime | timedelta | 最近一次进站的入场时刻（NaT=无进站） |",
        "| PitOutTime | timedelta | 最近一次进站的出场时刻（NaT=无进站） |",
        "",
        "---",
        "",
        "*本文件由 `data_collection.py` 自动生成。*",
        "",
    ])

    return "\n".join(lines)


# ============================================================================
# 入口
# ============================================================================

def main() -> None:
    """完整的数据采集 → 保存 → 汇总流程入口。

    执行步骤：
    1. 启用 FastF1 缓存
    2. 采集五个赛季的摩纳哥正赛数据
    3. 保存为 Parquet 文件
    4. 生成 Markdown 汇总表并打印
    """
    log.info("=" * 60)
    log.info("摩纳哥大奖赛数据采集 — 开始运行")
    log.info("目标赛季: %s", DEFAULT_YEARS)
    log.info("=" * 60)

    # ---- 第一步：数据采集 -----------------------------------------------
    data = fetch_monaco_data(DEFAULT_YEARS)

    if not data:
        log.error("未采集到任何数据，终止运行。请检查网络连接和 FastF1 安装。")
        sys.exit(1)

    # ---- 第二步：保存 Parquet 文件 -------------------------------------
    saved_files = save_parquet_files(data)
    print(f"\n已保存 {len(saved_files)} 个 Parquet 文件到 {OUTPUT_DIR.resolve()}/")

    # ---- 第三步：生成汇总表 ---------------------------------------------
    summary_md = generate_summary_markdown(data)

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    log.info("汇总表已保存至 %s", summary_path)

    # 打印汇总表到控制台
    print("\n" + summary_md)

    log.info("全部完成。")


if __name__ == "__main__":
    main()
