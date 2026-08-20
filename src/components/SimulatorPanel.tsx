import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Play, RotateCcw, Settings2 } from 'lucide-react';
import type { SimulationParams, Tier } from '@/types/simulation';
import { DEFAULT_PARAMS, TIER_COLORS, TIER_NAMES } from '@/types/simulation';

interface SimulatorPanelProps {
  params: SimulationParams;
  onParamsChange: (params: SimulationParams) => void;
  onRun: () => void;
  isRunning: boolean;
}

function TierConfigSection({
  tier,
  config,
  onChange,
}: {
  tier: Tier;
  config: SimulationParams['teams']['t1'];
  onChange: (c: SimulationParams['teams']['t1']) => void;
}) {
  const color = TIER_COLORS[tier];
  const name = TIER_NAMES[tier];

  return (
    <div className="space-y-3 p-3 rounded-lg border" style={{ borderColor: `${color}30` }}>
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
        <h4 className="font-semibold text-sm">{name}</h4>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">赛车数量</Label>
          <input
            type="number"
            min={1}
            max={8}
            value={config.count}
            onChange={(e) => onChange({ ...config, count: Number(e.target.value) })}
            className="w-full px-2 py-1 text-sm bg-background border rounded"
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">基准圈速 (秒)</Label>
          <input
            type="number"
            step={0.01}
            min={70}
            max={85}
            value={config.baseLapTime}
            onChange={(e) => onChange({ ...config, baseLapTime: Number(e.target.value) })}
            className="w-full px-2 py-1 text-sm bg-background border rounded"
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">进站损失 (秒)</Label>
          <input
            type="number"
            step={0.5}
            min={15}
            max={30}
            value={config.pitLoss}
            onChange={(e) => onChange({ ...config, pitLoss: Number(e.target.value) })}
            className="w-full px-2 py-1 text-sm bg-background border rounded"
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">失误率</Label>
          <input
            type="number"
            step={0.01}
            min={0}
            max={0.3}
            value={config.mistakeRate}
            onChange={(e) => onChange({ ...config, mistakeRate: Number(e.target.value) })}
            className="w-full px-2 py-1 text-sm bg-background border rounded"
          />
        </div>
      </div>
    </div>
  );
}

export default function SimulatorPanel({ params, onParamsChange, onRun, isRunning }: SimulatorPanelProps) {
  const [twoStop, setTwoStop] = useState(false);

  const updateTeam = useCallback(
    (tier: 't1' | 't2' | 't3', config: SimulationParams['teams']['t1']) => {
      onParamsChange({ ...params, teams: { ...params.teams, [tier]: config } });
    },
    [params, onParamsChange]
  );

  const updateStrategy = useCallback(
    (key: keyof SimulationParams['strategy'], value: any) => {
      onParamsChange({
        ...params,
        strategy: { ...params.strategy, [key]: value },
      });
    },
    [params, onParamsChange]
  );

  const updateRandomEvents = useCallback(
    (key: keyof SimulationParams['randomEvents'], value: any) => {
      onParamsChange({
        ...params,
        randomEvents: { ...params.randomEvents, [key]: value },
      });
    },
    [params, onParamsChange]
  );

  const resetToDefault = useCallback(() => {
    onParamsChange(DEFAULT_PARAMS);
    setTwoStop(false);
  }, [onParamsChange]);

  return (
    <Card className="bg-card/50 backdrop-blur border-border/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Settings2 className="w-5 h-5 text-primary" />
            策略参数配置
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={resetToDefault} className="text-muted-foreground">
            <RotateCcw className="w-4 h-4 mr-1" />
            重置
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 车队配置 */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">车队梯队</h3>
          <TierConfigSection tier="T1" config={params.teams.t1} onChange={(c) => updateTeam('t1', c)} />
          <TierConfigSection tier="T2" config={params.teams.t2} onChange={(c) => updateTeam('t2', c)} />
          <TierConfigSection tier="T3" config={params.teams.t3} onChange={(c) => updateTeam('t3', c)} />
        </div>

        {/* 策略配置 */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">进站策略</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">一停窗口 (圈)</Label>
              <div className="flex items-center gap-2">
                <Slider
                  value={[params.strategy.firstStopLap]}
                  onValueChange={([v]) => updateStrategy('firstStopLap', v)}
                  min={10}
                  max={40}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-8 text-right">{params.strategy.firstStopLap}</span>
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-muted-foreground">二停窗口</Label>
                <Switch checked={twoStop} onCheckedChange={setTwoStop} />
              </div>
              {twoStop ? (
                <div className="flex items-center gap-2">
                  <Slider
                    value={[params.strategy.secondStopLap ?? 42]}
                    onValueChange={([v]) => updateStrategy('secondStopLap', v)}
                    min={25}
                    max={60}
                    step={1}
                    className="flex-1"
                  />
                  <span className="text-sm font-mono w-8 text-right">{params.strategy.secondStopLap ?? 42}</span>
                </div>
              ) : (
                <div className="h-9 flex items-center text-xs text-muted-foreground">一停策略</div>
              )}
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">轮胎磨损率 (秒/圈)</Label>
              <div className="flex items-center gap-2">
                <Slider
                  value={[params.strategy.tireDegradation]}
                  onValueChange={([v]) => updateStrategy('tireDegradation', v)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-10 text-right">{params.strategy.tireDegradation.toFixed(2)}</span>
              </div>
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">首发轮胎</Label>
              <Select
                value={params.strategy.compound}
                onValueChange={(v) => updateStrategy('compound', v)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="SOFT">软胎 (Soft)</SelectItem>
                  <SelectItem value="MEDIUM">中性胎 (Medium)</SelectItem>
                  <SelectItem value="HARD">硬胎 (Hard)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* 随机事件 */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">随机事件</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">安全车概率</Label>
              <div className="flex items-center gap-2">
                <Slider
                  value={[params.randomEvents.safetyCarProbability]}
                  onValueChange={([v]) => updateRandomEvents('safetyCarProbability', v)}
                  min={0}
                  max={1}
                  step={0.05}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{(params.randomEvents.safetyCarProbability * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">VSC概率</Label>
              <div className="flex items-center gap-2">
                <Slider
                  value={[params.randomEvents.vscProbability]}
                  onValueChange={([v]) => updateRandomEvents('vscProbability', v)}
                  min={0}
                  max={1}
                  step={0.05}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{(params.randomEvents.vscProbability * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* 模拟设置 */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">模拟设置</h3>
          <div className="flex items-center gap-3">
            <div className="flex-1 space-y-1">
              <Label className="text-xs text-muted-foreground">模拟次数</Label>
              <div className="flex items-center gap-2">
                <Slider
                  value={[params.simulationCount]}
                  onValueChange={([v]) => onParamsChange({ ...params, simulationCount: v })}
                  min={100}
                  max={5000}
                  step={100}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-12 text-right">{params.simulationCount}</span>
              </div>
            </div>
          </div>
        </div>

        <Button
          onClick={onRun}
          disabled={isRunning}
          className="w-full h-11 text-base font-semibold bg-primary hover:bg-primary/90"
        >
          <Play className="w-5 h-5 mr-2" />
          {isRunning ? '模拟运行中...' : '运行蒙特卡洛模拟'}
        </Button>
      </CardContent>
    </Card>
  );
}
