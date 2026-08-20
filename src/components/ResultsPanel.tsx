import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { Trophy, TrendingUp, Shield, Zap } from 'lucide-react';
import type { AggregatedResult } from '@/types/simulation';
import { TIER_COLORS, TIER_NAMES } from '@/types/simulation';

interface ResultsPanelProps {
  result: AggregatedResult | null;
}

function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <Card className="bg-card/50 border-border/50">
      <CardContent className="pt-4 pb-3 px-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-wider">{title}</p>
            <p className="text-2xl font-bold mt-1" style={{ color }}>{value}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
          </div>
          <div className="p-2 rounded-lg bg-background" style={{ color }}>
            <Icon className="w-5 h-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ResultsPanel({ result }: ResultsPanelProps) {
  const winRateData = useMemo(() => {
    if (!result) return [];
    return [
      { name: TIER_NAMES.T1, value: result.winRates.T1, color: TIER_COLORS.T1 },
      { name: TIER_NAMES.T2, value: result.winRates.T2, color: TIER_COLORS.T2 },
      { name: TIER_NAMES.T3, value: result.winRates.T3, color: TIER_COLORS.T3 },
    ];
  }, [result]);

  const positionData = useMemo(() => {
    if (!result) return [];
    const tiers = ['T1', 'T2', 'T3'] as const;
    // 获取所有可能的位置
    const allPositions = new Set<number>();
    tiers.forEach((t) => {
      result.tierPositionDistribution[t].forEach((d) => allPositions.add(d.position));
    });
    const sortedPositions = Array.from(allPositions).sort((a, b) => a - b);

    return sortedPositions.map((pos) => {
      const row: Record<string, any> = { position: `${pos}` };
      tiers.forEach((t) => {
        const data = result.tierPositionDistribution[t].find((d) => d.position === pos);
        row[t] = data ? data.probability * 100 : 0;
      });
      return row;
    });
  }, [result]);

  const avgPosData = useMemo(() => {
    if (!result) return [];
    return [
      { name: 'T1', value: result.avgFinishingPositions.T1, color: TIER_COLORS.T1 },
      { name: 'T2', value: result.avgFinishingPositions.T2, color: TIER_COLORS.T2 },
      { name: 'T3', value: result.avgFinishingPositions.T3, color: TIER_COLORS.T3 },
    ];
  }, [result]);

  if (!result) {
    return (
      <div className="flex items-center justify-center h-96 text-muted-foreground">
        <div className="text-center space-y-2">
          <Zap className="w-12 h-12 mx-auto opacity-30" />
          <p>点击「运行蒙特卡洛模拟」查看结果</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* 统计卡片 */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          title="T1 胜率"
          value={`${(result.winRates.T1 * 100).toFixed(1)}%`}
          subtitle={`${result.totalSimulations} 场模拟`}
          icon={Trophy}
          color={TIER_COLORS.T1}
        />
        <StatCard
          title="T1 平均名次"
          value={result.avgFinishingPositions.T1.toFixed(2)}
          subtitle="完赛位置"
          icon={TrendingUp}
          color={TIER_COLORS.T1}
        />
        <StatCard
          title="T2 Podium率"
          value={`${(result.podiumRates.T2 * 100).toFixed(1)}%`}
          subtitle="进入前3概率"
          icon={Shield}
          color={TIER_COLORS.T2}
        />
        <StatCard
          title="T2 策略收益"
          value={`+${result.t2StrategyGain.toFixed(1)}`}
          subtitle="最优vs最差名次差"
          icon={Zap}
          color="#FF8700"
        />
      </div>

      {/* 胜率分布 */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">胜率分布</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={winRateData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${(value * 100).toFixed(1)}%`}
                >
                  {winRateData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          {/* 进度条 */}
          <div className="space-y-2 mt-2">
            {winRateData.map((d) => (
              <div key={d.name} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span style={{ color: d.color }}>{d.name}</span>
                  <span className="text-muted-foreground">{(d.value * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${d.value * 100}%`, backgroundColor: d.color }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 平均完赛名次 */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">平均完赛名次</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={avgPosData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" domain={[0, 12]} stroke="#666" fontSize={12} />
                <YAxis dataKey="name" type="category" stroke="#888" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                  formatter={(v: number) => v.toFixed(2)}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {avgPosData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* 名次分布热力图 */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">名次分布概率 (%)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={positionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="position" stroke="#666" fontSize={11} />
                <YAxis stroke="#666" fontSize={12} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                  formatter={(v: number) => `${v.toFixed(1)}%`}
                />
                <Legend />
                <Bar dataKey="T1" fill={TIER_COLORS.T1} radius={[2, 2, 0, 0]} />
                <Bar dataKey="T2" fill={TIER_COLORS.T2} radius={[2, 2, 0, 0]} />
                <Bar dataKey="T3" fill={TIER_COLORS.T3} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* 随机事件统计 */}
      <Card className="bg-card/50 border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">随机事件统计</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-muted-foreground">安全车触发场次</p>
              <p className="text-xl font-bold">{result.safetyCarCount} <span className="text-sm font-normal text-muted-foreground">/ {result.totalSimulations}</span></p>
              <p className="text-xs text-muted-foreground">{((result.safetyCarCount / result.totalSimulations) * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">VSC触发场次</p>
              <p className="text-xl font-bold">{result.vscCount} <span className="text-sm font-normal text-muted-foreground">/ {result.totalSimulations}</span></p>
              <p className="text-xs text-muted-foreground">{((result.vscCount / result.totalSimulations) * 100).toFixed(1)}%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
