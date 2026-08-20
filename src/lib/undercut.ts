// ============================================================
// Undercut 策略预测器 — 逻辑回归引擎
// Based on: 基于多赛季数据的F1摩纳哥大奖赛进站策略梯队差异研究
// ============================================================

import type { UndercutParams, UndercutResult, UndercutFactor } from '@/types/simulation';
import { UNDERCUT_COEFFICIENTS } from '@/types/simulation';

/**
 * 逻辑回归预测 Undercut 成功概率
 *
 * 模型: P(success) = 1 / (1 + e^(-z))
 * 其中 z = β0 + Σ βi * xi
 *
 * 特征变量:
 *   x1 = gapToAhead          (与前车差距, 秒)     β1 = -0.15  (差距越大越难)
 *   x2 = lapsRemaining        (剩余圈数)          β2 = +0.04  (圈数越多越有利)
 *   x3 = competitorTireAge   (对手轮胎年龄, 圈)   β3 = +0.08  (对手轮胎越旧越有利)
 *   x4 = tireDegradation     (轮胎磨损率, 秒/圈)  β4 = +4.00  (高磨损放大新胎优势)
 *   x5 = pitLoss             (进站损失, 秒)       β5 = -0.12  (损失越大越难)
 *   x6 = safetyCarProb       (安全车概率)         β6 = -0.80  (安全车neutralize策略)
 *   x7 = trackPosition       (赛道位置)           β7 = -0.02  (后方位置更难执行)
 */
export function predictUndercut(params: UndercutParams): UndercutResult {
  const c = UNDERCUT_COEFFICIENTS;

  const contributions = {
    gap: c.gapToAhead * params.gapToAhead,
    laps: c.lapsRemaining * params.lapsRemainingAfterStop,
    tireAge: c.competitorTireAge * params.competitorTireAge,
    deg: c.tireDegradation * params.tireDegradation,
    pit: c.pitLoss * params.pitLoss,
    sc: c.safetyCarProb * params.safetyCarProbability,
    pos: c.trackPosition * params.trackPosition,
  };

  const z =
    c.intercept +
    contributions.gap +
    contributions.laps +
    contributions.tireAge +
    contributions.deg +
    contributions.pit +
    contributions.sc +
    contributions.pos;

  const probability = 1 / (1 + Math.exp(-z));

  const factors: UndercutFactor[] = [
    {
      name: '对手轮胎年龄',
      contribution: contributions.tireAge,
      value: `${params.competitorTireAge} 圈`,
      description: '对手轮胎越旧，新胎圈速优势越明显',
    },
    {
      name: '剩余圈数',
      contribution: contributions.laps,
      value: `${params.lapsRemainingAfterStop} 圈`,
      description: '更多剩余圈数 = 更多时间利用新胎优势追回差距',
    },
    {
      name: '轮胎磨损率',
      contribution: contributions.deg,
      value: `${params.tireDegradation.toFixed(2)} 秒/圈`,
      description: '高磨损率放大新旧轮胎的圈速差异',
    },
    {
      name: '与前车差距',
      contribution: contributions.gap,
      value: `${params.gapToAhead.toFixed(1)} 秒`,
      description: '差距越小，需要弥补的时间越少',
    },
    {
      name: '进站损失',
      contribution: contributions.pit,
      value: `${params.pitLoss.toFixed(1)} 秒`,
      description: '进站损失越大，出站后需要追回的时间越多',
    },
    {
      name: '安全车风险',
      contribution: contributions.sc,
      value: `${(params.safetyCarProbability * 100).toFixed(0)}%`,
      description: '安全车可能 neutralize 策略优势或改变比赛节奏',
    },
    {
      name: '赛道位置',
      contribution: contributions.pos,
      value: `P${params.trackPosition}`,
      description: '靠前位置执行 Undercut 交通风险更低',
    },
  ].sort((a, b) => b.contribution - a.contribution);

  const confidence: UndercutResult['confidence'] =
    probability > 0.65 ? 'high' : probability > 0.4 ? 'medium' : 'low';

  let recommendation: string;
  let recommendationLevel: UndercutResult['recommendationLevel'];

  if (probability > 0.7) {
    recommendation = '强烈建议执行 Undercut — 当前条件非常有利，新胎优势足以弥补进站损失并完成超越';
    recommendationLevel = 'recommend';
  } else if (probability > 0.5) {
    recommendation = '可以考虑 Undercut — 成功概率适中，需密切关注对手进站时机与出站交通状况';
    recommendationLevel = 'consider';
  } else if (probability > 0.3) {
    recommendation = 'Undercut 风险较高 — 建议等待对手先进站（Overcut）或延长 stint 寻找窗口';
    recommendationLevel = 'caution';
  } else {
    recommendation = '不建议 Undercut — 当前条件不利，进站损失难以弥补，应坚持原策略或等待安全车';
    recommendationLevel = 'danger';
  }

  return { probability, zScore: z, confidence, factors, recommendation, recommendationLevel };
}

/** 敏感性分析：改变单个参数，观察成功率变化 */
export function undercutSensitivity(
  baseParams: UndercutParams,
  paramKey: keyof UndercutParams,
  range: number[]
): { value: number; probability: number }[] {
  return range.map((value) => {
    const params = { ...baseParams, [paramKey]: value };
    const result = predictUndercut(params);
    return { value, probability: result.probability };
  });
}

/** 敏感性分析配置 */
export const UNDERCUT_SENSITIVITY_CONFIGS = [
  {
    name: '与前车差距',
    paramKey: 'gapToAhead' as keyof UndercutParams,
    range: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    unit: '秒',
  },
  {
    name: '对手轮胎年龄',
    paramKey: 'competitorTireAge' as keyof UndercutParams,
    range: [5, 10, 15, 20, 25, 30, 35],
    unit: '圈',
  },
  {
    name: '进站损失',
    paramKey: 'pitLoss' as keyof UndercutParams,
    range: [16, 18, 20, 22, 24, 26, 28],
    unit: '秒',
  },
  {
    name: '轮胎磨损率',
    paramKey: 'tireDegradation' as keyof UndercutParams,
    range: [0, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30, 0.40],
    unit: '秒/圈',
  },
];
