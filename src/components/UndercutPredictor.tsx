import { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import {
  Target,
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { predictUndercut, undercutSensitivity, UNDERCUT_SENSITIVITY_CONFIGS } from '@/lib/undercut';
import type { UndercutParams } from '@/types/simulation';
import { DEFAULT_UNDERCUT_PARAMS, UNDERCUT_COEFFICIENTS } from '@/types/simulation';

const TOTAL_LAPS = 70;

// ─── 概率颜色 ─────────────────────────────────────────────
function getProbabilityColor(p: number): string {
  if (p > 0.7) return '#00D2BE';
  if (p > 0.5) return '#FF8700';
  if (p > 0.3) return '#FFCC00';
  return '#DC0000';
}

function getProbabilityLabel(p: number): string {
  if (p > 0.7) return '高成功率';
  if (p > 0.5) return '中等概率';
  if (p > 0.3) return '低概率';
  return '极低概率';
}

// ─── SVG 概率仪表盘 ───────────────────────────────────────
function ProbabilityGauge({ probability }: { probability: number }) {
  const percentage = probability * 100;
  const color = getProbabilityColor(probability);
  const radius = 85;
  const circumference = Math.PI * radius;
  const offset = circumference * (1 - percentage / 100);

  return (
    <div className="relative w-full flex flex-col items-center">
      <svg width="200" height="110" viewBox="0 0 200 110">
        {/* 背景刻度 */}
        <path
          d="M 15 100 A 85 85 0 0 1 185 100"
          fill="none"
          stroke="#2a2a2a"
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* 进度弧 */}
        <motion.path
          d="M 15 100 A 85 85 0 0 1 185 100"
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={false}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          style={{ filter: `drop-shadow(0 0 6px ${color}66)` }}
        />
        {/* 刻度标记 */}
        {[0, 25, 50, 75, 100].map((tick) => {
          const angle = Math.PI - (tick / 100) * Math.PI;
          const x1 = 100 + Math.cos(angle) * 72;
          const y1 = 100 - Math.sin(angle) * 72;
          const x2 = 100 + Math.cos(angle) * 62;
          const y2 = 100 - Math.sin(angle) * 62;
          return (
            <line key={tick} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#555" strokeWidth="1.5" />
          );
        })}
      </svg>
      <div className="absolute top-[55px] flex flex-col items-center">
        <motion.span
          key={Math.round(percentage)}
          initial={{ scale: 0.8, opacity: 0.5 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="text-4xl font-bold"
          style={{ color }}
        >
          {percentage.toFixed(0)}%
        </motion.span>
        <span className="text-[10px] text-muted-foreground uppercase tracking-widest mt-0.5">
          {getProbabilityLabel(probability)}
        </span>
      </div>
    </div>
  );
}

// ─── 因素贡献条 ────────────────────────────────────────────
function FactorBar({
  name,
  contribution,
  value,
  description,
}: {
  name: string;
  contribution: number;
  value: string;
  description: string;
}) {
  const isPositive = contribution >= 0;
  const maxAbs = 3.0;
  const width = Math.min(Math.abs(contribution) / maxAbs * 100, 100);
  const barColor = isPositive ? '#00D2BE' : '#FF4444';

  return (
    <div className="group">
      <div className="flex items-center justify-between text-xs mb-1">
        <div className="flex items-center gap-1.5">
          {isPositive ? (
            <TrendingUp className="w-3 h-3 text-teal-400" />
          ) : (
            <TrendingDown className="w-3 h-3 text-red-400" />
          )}
          <span className="font-medium">{name}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">{value}</span>
          <span
            className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
              isPositive ? 'bg-teal-500/15 text-teal-400' : 'bg-red-500/15 text-red-400'
            }`}
          >
            {isPositive ? '+' : ''}
            {contribution.toFixed(2)}
          </span>
        </div>
      </div>
      <div className="relative h-2 bg-muted/40 rounded-full overflow-hidden">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border/60" />
        <motion.div
          className={`absolute top-0 bottom-0 rounded-full ${isPositive ? '' : ''}`}
          style={{
            backgroundColor: barColor,
            [isPositive ? 'left' : 'right']: '50%',
          } as React.CSSProperties}
          initial={{ width: 0 }}
          animate={{ width: `${width / 2}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {description}
      </p>
    </div>
  );
}

// ─── 输入控件 ──────────────────────────────────────────────
interface SliderInputProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
}

function SliderInput({ label, value, onChange, min, max, step, format }: SliderInputProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <span className="text-sm font-mono text-foreground">{format(value)}</span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  );
}

// ─── 主组件 ────────────────────────────────────────────────
export default function UndercutPredictor() {
  const [params, setParams] = useState<UndercutParams>(DEFAULT_UNDERCUT_PARAMS);
  const [sensitivityIndex, setSensitivityIndex] = useState(0);
  const [showModelDetails, setShowModelDetails] = useState(false);

  const result = useMemo(() => predictUndercut(params), [params]);

  const sensitivityConfig = UNDERCUT_SENSITIVITY_CONFIGS[sensitivityIndex];
  const sensitivityData = useMemo(
    () => undercutSensitivity(params, sensitivityConfig.paramKey, sensitivityConfig.range),
    [params, sensitivityConfig]
  );

  const updateParam = useCallback((key: keyof UndercutParams, value: number) => {
    setParams((prev) => {
      const next = { ...prev, [key]: value };
      if (key === 'pitStopLap') {
        next.lapsRemainingAfterStop = TOTAL_LAPS - value;
      }
      return next;
    });
  }, []);

  const reset = useCallback(() => setParams(DEFAULT_UNDERCUT_PARAMS), []);

  const recoColor = {
    recommend: '#00D2BE',
    consider: '#FF8700',
    caution: '#FFCC00',
    danger: '#DC0000',
  }[result.recommendationLevel];

  const recoIcon = {
    recommend: CheckCircle2,
    consider: Target,
    caution: AlertTriangle,
    danger: AlertTriangle,
  }[result.recommendationLevel];

  const RecoIcon = recoIcon;

  return (
    <Card className="bg-card/50 backdrop-blur border-border/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Target className="w-5 h-5 text-accent" />
            Undercut 策略预测器
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={reset} className="text-muted-foreground">
            <RotateCcw className="w-3.5 h-3.5 mr-1" />
            重置
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          基于逻辑回归模型，输入比赛实时参数预测 Undercut 成功概率
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 主体：左输入 + 右结果 */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* 左侧：输入参数 */}
          <div className="lg:col-span-5 space-y-3 p-3 rounded-lg bg-background/40 border border-border/30">
            <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              比赛参数
            </h3>
            <SliderInput
              label="与前车差距"
              value={params.gapToAhead}
              onChange={(v) => updateParam('gapToAhead', v)}
              min={0.3}
              max={5.0}
              step={0.1}
              format={(v) => `${v.toFixed(1)} 秒`}
            />
            <SliderInput
              label="执行进站圈"
              value={params.pitStopLap}
              onChange={(v) => updateParam('pitStopLap', v)}
              min={10}
              max={40}
              step={1}
              format={(v) => `第 ${v} 圈`}
            />
            <SliderInput
              label="对手轮胎年龄"
              value={params.competitorTireAge}
              onChange={(v) => updateParam('competitorTireAge', v)}
              min={5}
              max={40}
              step={1}
              format={(v) => `${v} 圈`}
            />
            <SliderInput
              label="轮胎磨损率"
              value={params.tireDegradation}
              onChange={(v) => updateParam('tireDegradation', v)}
              min={0}
              max={0.4}
              step={0.01}
              format={(v) => `${v.toFixed(2)} 秒/圈`}
            />
            <SliderInput
              label="进站损失"
              value={params.pitLoss}
              onChange={(v) => updateParam('pitLoss', v)}
              min={16}
              max={28}
              step={0.5}
              format={(v) => `${v.toFixed(1)} 秒`}
            />
            <SliderInput
              label="安全车概率"
              value={params.safetyCarProbability}
              onChange={(v) => updateParam('safetyCarProbability', v)}
              min={0}
              max={1}
              step={0.05}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <SliderInput
              label="当前赛道位置"
              value={params.trackPosition}
              onChange={(v) => updateParam('trackPosition', v)}
              min={1}
              max={20}
              step={1}
              format={(v) => `P${v}`}
            />
            <div className="text-[10px] text-muted-foreground pt-1 border-t border-border/20">
              进站后剩余圈数：{params.lapsRemainingAfterStop} 圈（自动计算）
            </div>
          </div>

          {/* 右侧：结果 */}
          <div className="lg:col-span-7 space-y-4">
            {/* 概率仪表盘 */}
            <div className="flex flex-col items-center pt-2">
              <ProbabilityGauge probability={result.probability} />
            </div>

            {/* 策略建议 */}
            <motion.div
              className="rounded-lg p-3 border"
              style={{
                backgroundColor: `${recoColor}11`,
                borderColor: `${recoColor}44`,
              }}
              initial={false}
              animate={{ opacity: 1 }}
              key={result.recommendationLevel}
            >
              <div className="flex items-start gap-2">
                <RecoIcon className="w-4 h-4 mt-0.5 shrink-0" style={{ color: recoColor }} />
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider mb-0.5" style={{ color: recoColor }}>
                    策略建议
                  </p>
                  <p className="text-sm text-foreground leading-relaxed">{result.recommendation}</p>
                </div>
              </div>
            </motion.div>

            {/* 因素分析 */}
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  因素贡献分析
                </h4>
                <span className="text-[10px] text-muted-foreground">z-score 分解</span>
              </div>
              {result.factors.map((factor) => (
                <FactorBar key={factor.name} {...factor} />
              ))}
            </div>
          </div>
        </div>

        {/* 敏感性分析图表 */}
        <div className="pt-2 border-t border-border/30">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              敏感性分析
            </h4>
            <div className="flex gap-1 flex-wrap">
              {UNDERCUT_SENSITIVITY_CONFIGS.map((config, idx) => (
                <button
                  key={config.name}
                  className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
                    sensitivityIndex === idx
                      ? 'bg-accent/20 text-accent border-accent/40'
                      : 'bg-muted/30 text-muted-foreground border-border/30 hover:bg-muted/50'
                  }`}
                  onClick={() => setSensitivityIndex(idx)}
                >
                  {config.name}
                </button>
              ))}
            </div>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sensitivityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="value"
                  stroke="#666"
                  fontSize={11}
                  tickFormatter={(v) =>
                    sensitivityConfig.paramKey === 'tireDegradation'
                      ? v.toFixed(2)
                      : sensitivityConfig.paramKey === 'gapToAhead'
                        ? v.toFixed(1)
                        : v.toString()
                  }
                >
                </XAxis>
                <YAxis
                  stroke="#666"
                  fontSize={12}
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                  formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                  labelFormatter={(v) =>
                    `${sensitivityConfig.name}: ${
                      sensitivityConfig.paramKey === 'tireDegradation'
                        ? v.toFixed(2)
                        : sensitivityConfig.paramKey === 'gapToAhead'
                          ? v.toFixed(1)
                          : v
                    }${sensitivityConfig.unit}`
                  }
                />
                <ReferenceLine
                  y={0.5}
                  stroke="#555"
                  strokeDasharray="2 2"
                  label={{ value: '50%', fill: '#555', fontSize: 9 }}
                />
                <Line
                  type="monotone"
                  dataKey="probability"
                  stroke="#FF8700"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#FF8700' }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">
            当前 {sensitivityConfig.name}：{
              sensitivityConfig.paramKey === 'tireDegradation'
                ? (params[sensitivityConfig.paramKey] as number).toFixed(2)
                : sensitivityConfig.paramKey === 'gapToAhead'
                  ? (params[sensitivityConfig.paramKey] as number).toFixed(1)
                  : (params[sensitivityConfig.paramKey] as number).toString()
            }
            {sensitivityConfig.unit}
          </p>
        </div>

        {/* 模型详情（可折叠） */}
        <div className="pt-2 border-t border-border/30">
          <button
            className="w-full flex items-center justify-between text-xs text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowModelDetails(!showModelDetails)}
          >
            <span className="flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5" />
              模型详情：逻辑回归公式与系数
            </span>
            {showModelDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          <AnimatePresence>
            {showModelDetails && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="pt-3 space-y-2 text-xs text-muted-foreground">
                  <div className="font-mono p-3 rounded-lg bg-background/60 border border-border/30 text-[11px] leading-relaxed">
                    <span className="text-accent">P</span>(success) = 1 / (1 + e
                    <sup>-z</sup>)
                    <br />
                    <br />
                    <span className="text-accent">z</span> = {UNDERCUT_COEFFICIENTS.intercept}
                    {' '}
                    {UNDERCUT_COEFFICIENTS.gapToAhead >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.gapToAhead}×gap
                    {' '}
                    {UNDERCUT_COEFFICIENTS.lapsRemaining >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.lapsRemaining}×lapsRemaining
                    {' '}
                    {UNDERCUT_COEFFICIENTS.competitorTireAge >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.competitorTireAge}×compTireAge
                    {' '}
                    {UNDERCUT_COEFFICIENTS.tireDegradation >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.tireDegradation}×tireDeg
                    {' '}
                    {UNDERCUT_COEFFICIENTS.pitLoss >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.pitLoss}×pitLoss
                    {' '}
                    {UNDERCUT_COEFFICIENTS.safetyCarProb >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.safetyCarProb}×safetyCarProb
                    {' '}
                    {UNDERCUT_COEFFICIENTS.trackPosition >= 0 ? '+' : ''}
                    {UNDERCUT_COEFFICIENTS.trackPosition}×trackPos
                  </div>
                  <p>
                    当前 z = <span className="font-mono text-foreground">{result.zScore.toFixed(3)}</span>
                    ，对应 P ={' '}
                    <span className="font-mono" style={{ color: getProbabilityColor(result.probability) }}>
                      {(result.probability * 100).toFixed(1)}%
                    </span>
                  </p>
                  <p className="text-[10px] leading-relaxed">
                    系数基于 F1 摩纳哥大奖赛多赛季数据与蒙特卡洛仿真（10,000 次）标定。
                    正系数表示该因素有利于 Undercut 成功，负系数表示不利。
                    模型假设 Undercut 成功定义为：执行进站后在新胎 stint 内超越前车。
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
}
