// ============================================================
// F1 Monaco Strategy Simulator — Type Definitions
// Based on: 基于多赛季数据的F1摩纳哥大奖赛进站策略梯队差异研究
// ============================================================

export type Tier = 'T1' | 'T2' | 'T3';
export type TireCompound = 'SOFT' | 'MEDIUM' | 'HARD';
export type RaceEvent = 'NORMAL' | 'SAFETY_CAR' | 'VSC' | 'RAIN';

/** 单个车队的参数配置 */
export interface TeamConfig {
  count: number;        // 该梯队赛车数量
  baseLapTime: number;  // 基准圈速（秒）
  pitLoss: number;      // 进站损失时间（秒）
  mistakeRate: number;  // 进站失误概率（0-1）
}

/** 模拟全局参数 */
export interface SimulationParams {
  teams: {
    t1: TeamConfig;
    t2: TeamConfig;
    t3: TeamConfig;
  };
  strategy: {
    firstStopLap: number;      // 首次进站圈数
    secondStopLap?: number;    // 二次进站圈数（可选，undefined表示一停）
    tireDegradation: number;   // 轮胎磨损率（秒/圈）
    compound: TireCompound;    // 首发轮胎配方
  };
  randomEvents: {
    safetyCarProbability: number;  // 安全车触发概率（每场比赛）
    vscProbability: number;        // 虚拟安全车概率（每场比赛）
    safetyCarDuration: number;     // 安全车持续圈数
    vscDuration: number;           // VSC持续圈数
  };
  simulationCount: number;  // 蒙特卡洛迭代次数
  totalLaps: number;        // 总圈数（摩纳哥=70）
}

/** 单辆赛车状态 */
export interface CarState {
  id: number;
  tier: Tier;
  baseLapTime: number;
  pitLoss: number;
  mistakeRate: number;
  // 动态状态
  totalTime: number;        // 累计总时间
  tireAge: number;          // 当前轮胎已使用圈数
  position: number;         // 当前名次
  gapToLeader: number;      // 与领先者的时间差
  stopsCompleted: number;   // 已完成进站次数
  hasMistake: boolean;      // 本次进站是否有失误
}

/** 单圈记录 */
export interface LapRecord {
  lap: number;
  carId: number;
  tier: Tier;
  lapTime: number;
  totalTime: number;
  tireAge: number;
  event: RaceEvent;
  isPitLap: boolean;
  position: number;
}

/** 单次模拟结果 */
export interface SingleRaceResult {
  finishingOrder: CarState[];  // 按完赛顺序排列
  lapRecords: LapRecord[];     // 逐圈记录（用于动画回放）
  safetyCarLaps: number[];     // 发生安全车的圈数
  vscLaps: number[];           // 发生VSC的圈数
}

/** 聚合统计结果 */
export interface AggregatedResult {
  // 胜率分布
  winRates: Record<Tier, number>;
  // 平均完赛名次
  avgFinishingPositions: Record<Tier, number>;
  //  podium率 (前3)
  podiumRates: Record<Tier, number>;
  // 名次分布热力图数据
  positionDistribution: {
    tier: Tier;
    startingPos: number;
    finishingPos: number;
    probability: number;
  }[];
  // 各梯队的名次分布统计
  tierPositionDistribution: Record<Tier, { position: number; probability: number }[]>;
  // 模拟场次统计
  totalSimulations: number;
  // 安全车触发次数统计
  safetyCarCount: number;
  vscCount: number;
  // 最佳/最差策略收益（T2车队）
  t2StrategyGain: number;  // 最优策略vs最差策略的名次差异
}

/** 敏感性分析参数 */
export interface SensitivityParams {
  paramName: string;
  paramKey: string;
  values: number[];
  t1WinRates: number[];
}

/** 默认模拟参数（基于论文数据） */
export const DEFAULT_PARAMS: SimulationParams = {
  teams: {
    t1: { count: 3, baseLapTime: 75.0, pitLoss: 20.0, mistakeRate: 0.05 },
    t2: { count: 3, baseLapTime: 75.45, pitLoss: 21.5, mistakeRate: 0.10 },
    t3: { count: 4, baseLapTime: 75.85, pitLoss: 23.0, mistakeRate: 0.15 },
  },
  strategy: {
    firstStopLap: 22,
    tireDegradation: 0.12,
    compound: 'SOFT',
  },
  randomEvents: {
    safetyCarProbability: 0.30,
    vscProbability: 0.20,
    safetyCarDuration: 4,
    vscDuration: 3,
  },
  simulationCount: 1000,
  totalLaps: 70,
};

/** F1风格色彩体系 */
export const TIER_COLORS: Record<Tier, string> = {
  T1: '#DC0000',  // 法拉利红
  T2: '#00D2BE',  // 梅赛德斯银/青
  T3: '#005AFF',  // 威廉姆斯蓝
};

export const TIER_NAMES: Record<Tier, string> = {
  T1: '争冠组 (T1)',
  T2: '中游组 (T2)',
  T3: '后方组 (T3)',
};

export const COMPOUND_COLORS: Record<TireCompound, string> = {
  SOFT: '#FF3333',
  MEDIUM: '#FFCC00',
  HARD: '#FFFFFF',
};

// ============================================================
// Undercut 策略预测器 — 类型定义
// 基于逻辑回归模型 (Logistic Regression)
// ============================================================

/** Undercut 预测输入参数 */
export interface UndercutParams {
  gapToAhead: number;              // 与前车的时间差（秒）
  pitStopLap: number;             // 执行进站圈数
  lapsRemainingAfterStop: number; // 进站后剩余圈数
  competitorTireAge: number;      // 对手当前轮胎年龄（圈数）
  tireDegradation: number;        // 轮胎磨损率（秒/圈）
  pitLoss: number;                // 进站损失时间（秒）
  safetyCarProbability: number;   // 安全车触发概率
  trackPosition: number;          // 当前赛道位置（1-20）
}

/** Undercut 因素贡献 */
export interface UndercutFactor {
  name: string;
  contribution: number;  // z-score 贡献值（正=有利，负=不利）
  value: string;         // 当前值显示
  description: string;   // 影响说明
}

/** Undercut 预测结果 */
export interface UndercutResult {
  probability: number;          // 成功概率 (0-1)
  zScore: number;                // 逻辑回归线性组合 z 值
  confidence: 'low' | 'medium' | 'high';
  factors: UndercutFactor[];
  recommendation: string;
  recommendationLevel: 'danger' | 'caution' | 'consider' | 'recommend';
}

/** Undercut 默认参数（摩纳哥典型场景：P3 车手试图 Undercut P2） */
export const DEFAULT_UNDERCUT_PARAMS: UndercutParams = {
  gapToAhead: 2.0,
  pitStopLap: 18,
  lapsRemainingAfterStop: 52,  // 70 - 18
  competitorTireAge: 18,
  tireDegradation: 0.12,
  pitLoss: 20.0,
  safetyCarProbability: 0.30,
  trackPosition: 3,
};

/** 逻辑回归系数（基于 F1 策略研究与蒙特卡洛仿真标定） */
export const UNDERCUT_COEFFICIENTS = {
  intercept: 1.2,
  gapToAhead: -0.15,
  lapsRemaining: 0.04,
  competitorTireAge: 0.08,
  tireDegradation: 4.0,
  pitLoss: -0.12,
  safetyCarProb: -0.8,
  trackPosition: -0.02,
};
