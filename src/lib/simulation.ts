// ============================================================
// F1 Monaco Strategy Simulator — Monte Carlo Engine
// ============================================================

import type {
  SimulationParams,
  CarState,
  LapRecord,
  SingleRaceResult,
  AggregatedResult,
  Tier,
  RaceEvent,
} from '@/types/simulation';

// ─── Helpers ─────────────────────────────────────────────────

function rand() {
  return Math.random();
}

function randInt(min: number, max: number) {
  return Math.floor(rand() * (max - min + 1)) + min;
}

/** 生成安全车触发圈数（可能多个） */
function generateSafetyCarLaps(probability: number, duration: number, totalLaps: number): number[] {
  if (rand() >= probability) return [];
  const laps: number[] = [];
  const startLap = randInt(5, totalLaps - duration - 5);
  for (let i = 0; i < duration; i++) {
    laps.push(startLap + i);
  }
  return laps;
}

/** 生成VSC触发圈数 */
function generateVSCLaps(probability: number, duration: number, totalLaps: number, safetyLaps: number[]): number[] {
  if (rand() >= probability) return [];
  const excluded = new Set(safetyLaps);
  let attempts = 0;
  while (attempts < 20) {
    const startLap = randInt(5, totalLaps - duration - 5);
    let overlap = false;
    for (let i = 0; i < duration; i++) {
      if (excluded.has(startLap + i)) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      const laps: number[] = [];
      for (let i = 0; i < duration; i++) laps.push(startLap + i);
      return laps;
    }
    attempts++;
  }
  return [];
}

// ─── Single Race Simulation ──────────────────────────────────

export function simulateSingleRace(params: SimulationParams): SingleRaceResult {
  const { teams, strategy, randomEvents, totalLaps } = params;

  const cars: CarState[] = [];
  let id = 0;
  (['T1', 'T2', 'T3'] as Tier[]).forEach((tier) => {
    const config = teams[tier.toLowerCase() as 't1' | 't2' | 't3'];
    for (let i = 0; i < config.count; i++) {
      cars.push({
        id: id++,
        tier,
        baseLapTime: config.baseLapTime,
        pitLoss: config.pitLoss,
        mistakeRate: config.mistakeRate,
        totalTime: 0,
        tireAge: 0,
        position: 0,
        gapToLeader: 0,
        stopsCompleted: 0,
        hasMistake: false,
      });
    }
  });

  const safetyCarLaps = generateSafetyCarLaps(
    randomEvents.safetyCarProbability,
    randomEvents.safetyCarDuration,
    totalLaps
  );
  const vscLaps = generateVSCLaps(
    randomEvents.vscProbability,
    randomEvents.vscDuration,
    totalLaps,
    safetyCarLaps
  );
  const safetySet = new Set(safetyCarLaps);
  const vscSet = new Set(vscLaps);

  const lapRecords: LapRecord[] = [];

  for (let lap = 1; lap <= totalLaps; lap++) {
    let event: RaceEvent = 'NORMAL';

    if (safetySet.has(lap)) {
      event = 'SAFETY_CAR';
    } else if (vscSet.has(lap)) {
      event = 'VSC';
    }

    for (const car of cars) {
      let lapTime: number;
      let isPitLap = false;

      if (event === 'SAFETY_CAR') {
        lapTime = 95.0;
      } else if (event === 'VSC') {
        lapTime = 85.0;
      } else {
        const tirePenalty = strategy.tireDegradation * car.tireAge;
        lapTime = car.baseLapTime + tirePenalty;
        car.tireAge++;

        const shouldPit =
          (lap === strategy.firstStopLap && car.stopsCompleted === 0) ||
          (lap === strategy.secondStopLap && car.stopsCompleted === 1);

        if (shouldPit) {
          isPitLap = true;
          let actualPitLoss = car.pitLoss;

          if (rand() < car.mistakeRate) {
            actualPitLoss += randInt(2, 8);
            car.hasMistake = true;
          }

          lapTime += actualPitLoss;
          car.tireAge = 0;
          car.stopsCompleted++;
        }
      }

      car.totalTime += lapTime;

      lapRecords.push({
        lap,
        carId: car.id,
        tier: car.tier,
        lapTime,
        totalTime: car.totalTime,
        tireAge: car.tireAge,
        event,
        isPitLap,
        position: 0,
      });
    }

    const sorted = [...cars].sort((a, b) => a.totalTime - b.totalTime);
    sorted.forEach((car, idx) => {
      car.position = idx + 1;
      car.gapToLeader = car.totalTime - sorted[0].totalTime;
    });

    for (const lr of lapRecords) {
      if (lr.lap === lap) {
        const car = cars.find((c) => c.id === lr.carId)!;
        lr.position = car.position;
      }
    }
  }

  const finishingOrder = [...cars].sort((a, b) => a.totalTime - b.totalTime);

  return {
    finishingOrder,
    lapRecords,
    safetyCarLaps,
    vscLaps,
  };
}

// ─── Monte Carlo Aggregation ─────────────────────────────────

export function runMonteCarlo(params: SimulationParams): AggregatedResult {
  const tierCounts = {
    T1: params.teams.t1.count,
    T2: params.teams.t2.count,
    T3: params.teams.t3.count,
  };
  const totalCars = tierCounts.T1 + tierCounts.T2 + tierCounts.T3;

  let t1Wins = 0;
  let t2Wins = 0;
  let t3Wins = 0;
  let t1Podiums = 0;
  let t2Podiums = 0;
  let t3Podiums = 0;

  const tierTotalPositions: Record<Tier, number> = { T1: 0, T2: 0, T3: 0 };
  const tierPositionCounts: Record<Tier, number[]> = {
    T1: Array(totalCars + 1).fill(0),
    T2: Array(totalCars + 1).fill(0),
    T3: Array(totalCars + 1).fill(0),
  };

  let safetyCarTotal = 0;
  let vscTotal = 0;

  const t2FinishingPositions: number[] = [];

  for (let i = 0; i < params.simulationCount; i++) {
    const result = simulateSingleRace(params);

    const winner = result.finishingOrder[0];
    if (winner.tier === 'T1') t1Wins++;
    else if (winner.tier === 'T2') t2Wins++;
    else t3Wins++;

    result.finishingOrder.slice(0, 3).forEach((car) => {
      if (car.tier === 'T1') t1Podiums++;
      else if (car.tier === 'T2') t2Podiums++;
      else t3Podiums++;
    });

    result.finishingOrder.forEach((car) => {
      tierTotalPositions[car.tier] += car.position;
      tierPositionCounts[car.tier][car.position]++;
      if (car.tier === 'T2') {
        t2FinishingPositions.push(car.position);
      }
    });

    if (result.safetyCarLaps.length > 0) safetyCarTotal++;
    if (result.vscLaps.length > 0) vscTotal++;
  }

  const n = params.simulationCount;

  const t2Min = t2FinishingPositions.length > 0 ? Math.min(...t2FinishingPositions) : 0;
  const t2Max = t2FinishingPositions.length > 0 ? Math.max(...t2FinishingPositions) : 0;
  const t2Gain = t2Max - t2Min;

  const tierPositionDistribution: AggregatedResult['tierPositionDistribution'] = {
    T1: [],
    T2: [],
    T3: [],
  };
  (['T1', 'T2', 'T3'] as Tier[]).forEach((tier) => {
    for (let pos = 1; pos <= totalCars; pos++) {
      const count = tierPositionCounts[tier][pos];
      if (count > 0) {
        tierPositionDistribution[tier].push({
          position: pos,
          probability: count / n / tierCounts[tier],
        });
      }
    }
  });

  const totalCarsPerTier = tierCounts;

  return {
    winRates: {
      T1: t1Wins / n,
      T2: t2Wins / n,
      T3: t3Wins / n,
    },
    avgFinishingPositions: {
      T1: tierTotalPositions.T1 / (n * totalCarsPerTier.T1),
      T2: tierTotalPositions.T2 / (n * totalCarsPerTier.T2),
      T3: tierTotalPositions.T3 / (n * totalCarsPerTier.T3),
    },
    podiumRates: {
      T1: t1Podiums / (n * totalCarsPerTier.T1),
      T2: t2Podiums / (n * totalCarsPerTier.T2),
      T3: t3Podiums / (n * totalCarsPerTier.T3),
    },
    positionDistribution: [],
    tierPositionDistribution,
    totalSimulations: n,
    safetyCarCount: safetyCarTotal,
    vscCount: vscTotal,
    t2StrategyGain: t2Gain,
  };
}

// ─── Sensitivity Analysis ────────────────────────────────────

export interface SensitivityPoint {
  value: number;
  t1WinRate: number;
}

export function runSensitivityAnalysis(
  baseParams: SimulationParams,
  paramPath: string,
  valueRange: number[],
  onProgress?: (done: number, total: number) => void
): SensitivityPoint[] {
  const results: SensitivityPoint[] = [];

  for (let i = 0; i < valueRange.length; i++) {
    const value = valueRange[i];
    const params = JSON.parse(JSON.stringify(baseParams)) as SimulationParams;

    const pathParts = paramPath.split('.');
    let target: any = params;
    for (let j = 0; j < pathParts.length - 1; j++) {
      target = target[pathParts[j]];
    }
    target[pathParts[pathParts.length - 1]] = value;

    const result = runMonteCarlo(params);
    results.push({
      value,
      t1WinRate: result.winRates.T1,
    });

    if (onProgress) {
      onProgress(i + 1, valueRange.length);
    }
  }

  return results;
}

export const SENSITIVITY_CONFIGS = [
  {
    name: 'T1-T3 圈速差',
    paramPath: 'teams.t3.baseLapTime',
    range: [75.0, 75.2, 75.4, 75.6, 75.85, 76.0, 76.3, 76.6, 76.9, 77.2],
    baseValue: 75.85,
    unit: '秒',
  },
  {
    name: 'T1 进站损失',
    paramPath: 'teams.t1.pitLoss',
    range: [15, 17, 19, 20, 22, 24, 26, 28, 30],
    baseValue: 20,
    unit: '秒',
  },
  {
    name: '安全车概率',
    paramPath: 'randomEvents.safetyCarProbability',
    range: [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    baseValue: 0.3,
    unit: '',
  },
  {
    name: '轮胎磨损率',
    paramPath: 'strategy.tireDegradation',
    range: [0, 0.05, 0.08, 0.12, 0.15, 0.18, 0.25, 0.35, 0.5],
    baseValue: 0.12,
    unit: '秒/圈',
  },
  {
    name: 'T3 进站失误率',
    paramPath: 'teams.t3.mistakeRate',
    range: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    baseValue: 0.15,
    unit: '',
  },
];
