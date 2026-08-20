import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SimulatorPanel from '@/components/SimulatorPanel';
import ResultsPanel from '@/components/ResultsPanel';
import RaceReplay from '@/components/RaceReplay';
import UndercutPredictor from '@/components/UndercutPredictor';
import { runMonteCarlo, runSensitivityAnalysis, SENSITIVITY_CONFIGS, type SensitivityPoint } from '@/lib/simulation';
import type { SimulationParams, AggregatedResult } from '@/types/simulation';
import { DEFAULT_PARAMS } from '@/types/simulation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
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
import { Activity, GitBranch, FlaskConical, ChevronDown, ChevronUp } from 'lucide-react';

// ─── Sensitivity Panel ───────────────────────────────────────

function SensitivityPanel({ params }: { params: SimulationParams }) {
  const [results, setResults] = useState<Record<string, SensitivityPoint[]>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);

  const runAnalysis = useCallback(async (config: typeof SENSITIVITY_CONFIGS[0]) => {
    setRunning(config.name);
    setProgress(0);
    const pts = await new Promise<SensitivityPoint[]>((resolve) => {
      setTimeout(() => {
        const res = runSensitivityAnalysis(params, config.paramPath, config.range, (d, t) => {
          setProgress(Math.round((d / t) * 100));
        });
        resolve(res);
      }, 50);
    });
    setResults((prev) => ({ ...prev, [config.name]: pts }));
    setRunning(null);
  }, [params]);

  return (
    <Card className="bg-card/50 border-border/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-accent" />
          敏感性分析
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          探索各参数对 T1 胜率的影响。论文结论：梯队圈速差 &gt; 进站损失 &gt; 安全车概率 &gt; 轮胎磨损率 &gt; 进站失误率
        </p>

        {SENSITIVITY_CONFIGS.map((config) => {
          const data = results[config.name];
          const isRunning = running === config.name;
          const isExpanded = expanded === config.name;

          return (
            <div key={config.name} className="border border-border/30 rounded-lg overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-3 hover:bg-muted/30 transition-colors cursor-pointer"
                onClick={() => setExpanded(isExpanded ? null : config.name)}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{config.name}</span>
                  {data && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">
                      已完成
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {!data && !isRunning && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        runAnalysis(config);
                      }}
                    >
                      分析
                    </Button>
                  )}
                  {isRunning && (
                    <span className="text-xs text-muted-foreground">{progress}%</span>
                  )}
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </div>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: 'auto' }}
                    exit={{ height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="p-3 pt-0">
                      {isRunning && <Progress value={progress} className="mb-2" />}
                      {data ? (
                        <div className="h-48">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                              <XAxis
                                dataKey="value"
                                stroke="#666"
                                fontSize={11}
                                tickFormatter={(v) => `${v}${config.unit}`}
                              />
                              <YAxis
                                stroke="#666"
                                fontSize={12}
                                domain={[0, 1]}
                                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                              />
                              <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                                formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                                labelFormatter={(v) => `${v}${config.unit}`}
                              />
                              <ReferenceLine
                                x={config.baseValue}
                                stroke="#FF8700"
                                strokeDasharray="5 5"
                                label={{ value: '基准', fill: '#FF8700', fontSize: 10 }}
                              />
                              <Line
                                type="monotone"
                                dataKey="t1WinRate"
                                stroke="#FF8700"
                                strokeWidth={2}
                                dot={{ r: 3 }}
                                activeDot={{ r: 5 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div className="h-24 flex items-center justify-center text-xs text-muted-foreground">
                          点击「分析」运行敏感性分析
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

// ─── Main Home Page ──────────────────────────────────────────

export default function Home() {
  const [params, setParams] = useState<SimulationParams>(DEFAULT_PARAMS);
  const [result, setResult] = useState<AggregatedResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleRun = useCallback(() => {
    setIsRunning(true);
    setProgress(0);

    const batchSize = Math.min(100, params.simulationCount);
    let completed = 0;

    const runBatch = () => {
      const start = performance.now();
      while (completed < params.simulationCount && performance.now() - start < 50) {
        runMonteCarlo({ ...params, simulationCount: Math.min(batchSize, params.simulationCount - completed) });
        completed += Math.min(batchSize, params.simulationCount - completed);
        setProgress(Math.round((completed / params.simulationCount) * 100));

        if (completed >= params.simulationCount) {
          const finalResult = runMonteCarlo(params);
          setResult(finalResult);
          setIsRunning(false);
          return;
        }
      }
      requestAnimationFrame(runBatch);
    };

    requestAnimationFrame(runBatch);
  }, [params]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <FlaskConical className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-bold leading-tight">F1 摩纳哥策略模拟器</h1>
              <p className="text-[10px] text-muted-foreground">
                基于《基于多赛季数据的F1摩纳哥大奖赛进站策略梯队差异研究》
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Activity className="w-4 h-4" />
            <span>蒙特卡洛引擎 v1.0</span>
          </div>
        </div>
      </header>

      {/* Progress Bar */}
      <AnimatePresence>
        {isRunning && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="sticky top-[60px] z-40 bg-background/80 backdrop-blur"
          >
            <div className="max-w-7xl mx-auto px-4 py-2">
              <div className="flex items-center gap-2">
                <Progress value={progress} className="flex-1" />
                <span className="text-xs font-mono w-12 text-right">{progress}%</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Simulator Controls */}
          <div className="lg:col-span-4 space-y-4">
            <SimulatorPanel
              params={params}
              onParamsChange={setParams}
              onRun={handleRun}
              isRunning={isRunning}
            />
          </div>

          {/* Right: Results */}
          <div className="lg:col-span-8 space-y-4">
            {/* 论文核心发现卡片 */}
            {!result && (
              <Card className="bg-gradient-to-br from-primary/10 to-accent/5 border-primary/20">
                <CardContent className="p-5">
                  <h2 className="text-lg font-bold mb-2">论文核心发现</h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-t1 mt-2 shrink-0" />
                      <p>T1 车队胜率 <strong>99.6%</strong>，单圈速度优势累积约60秒，碾压性决定名次</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-t2 mt-2 shrink-0" />
                      <p>进站策略对 <strong>T2 车队</strong>最关键，可提升约 1.5 个名次</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-t3 mt-2 shrink-0" />
                      <p>T3 进站窗口平均比 T1 晚 <strong>18 圈</strong>，偏向 Gamble Strategy</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-accent mt-2 shrink-0" />
                      <p>安全车期间所有车同步减速，相对差距几乎不变 (Δ&lt;0.01位)</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <ResultsPanel result={result} />

            {/* 比赛回放 */}
            <RaceReplay params={params} />

            {/* Undercut 预测器 */}
            <UndercutPredictor />

            {/* 敏感性分析 */}
            {result && <SensitivityPanel params={params} />}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/30 mt-8 py-4 text-center text-xs text-muted-foreground">
        <p>数据来源：FastF1 API (2019-2024) | 研究方法：蒙特卡洛模拟 (10,000次) | 论文：基于多赛季数据的F1摩纳哥大奖赛进站策略梯队差异研究</p>
      </footer>
    </div>
  );
}
