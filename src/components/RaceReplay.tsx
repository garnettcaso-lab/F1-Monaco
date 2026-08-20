import { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, Pause, RotateCcw, SkipForward, SkipBack } from 'lucide-react';
import { simulateSingleRace } from '@/lib/simulation';
import type { SimulationParams, LapRecord, Tier } from '@/types/simulation';
import { TIER_COLORS } from '@/types/simulation';

interface RaceReplayProps {
  params: SimulationParams;
}

interface FrameData {
  lap: number;
  cars: { id: number; tier: Tier; position: number; gapToLeader: number; totalTime: number; isPitLap: boolean }[];
}

export default function RaceReplay({ params }: RaceReplayProps) {
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [currentLap, setCurrentLap] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const generateRace = useCallback(() => {
    const result = simulateSingleRace(params);
    // 按圈分组
    const lapMap = new Map<number, LapRecord[]>();
    result.lapRecords.forEach((lr) => {
      if (!lapMap.has(lr.lap)) lapMap.set(lr.lap, []);
      lapMap.get(lr.lap)!.push(lr);
    });

    const newFrames: FrameData[] = [];
    for (let lap = 0; lap <= params.totalLaps; lap++) {
      if (lap === 0) {
        // 起跑
        const carCount = params.teams.t1.count + params.teams.t2.count + params.teams.t3.count;
        newFrames.push({
          lap: 0,
          cars: Array.from({ length: carCount }, (_, i) => ({
            id: i,
            tier: i < params.teams.t1.count ? 'T1' : i < params.teams.t1.count + params.teams.t2.count ? 'T2' : 'T3',
            position: i + 1,
            gapToLeader: 0,
            totalTime: 0,
            isPitLap: false,
          })),
        });
      } else {
        const records = lapMap.get(lap) || [];
        const sorted = [...records].sort((a, b) => a.position - b.position);
        newFrames.push({
          lap,
          cars: sorted.map((r) => ({
            id: r.carId,
            tier: r.tier,
            position: r.position,
            gapToLeader: r.totalTime - (sorted[0]?.totalTime ?? 0),
            totalTime: r.totalTime,
            isPitLap: r.isPitLap,
          })),
        });
      }
    }
    setFrames(newFrames);
    setCurrentLap(0);
    setHasGenerated(true);
    setIsPlaying(false);
  }, [params]);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentLap((prev) => {
          if (prev >= params.totalLaps) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 300);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, params.totalLaps]);

  const frame = frames[currentLap];
  const totalCars = params.teams.t1.count + params.teams.t2.count + params.teams.t3.count;

  return (
    <Card className="bg-card/50 border-border/50">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">虚拟比赛回放</CardTitle>
          <Button size="sm" variant="outline" onClick={generateRace} disabled={isPlaying}>
            <RotateCcw className="w-4 h-4 mr-1" />
            生成新比赛
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {!hasGenerated ? (
          <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">
            点击「生成新比赛」开始回放
          </div>
        ) : (
          <>
            {/* 控制栏 */}
            <div className="flex items-center gap-2">
              <Button size="icon" variant="ghost" onClick={() => setCurrentLap(Math.max(0, currentLap - 5))}>
                <SkipBack className="w-4 h-4" />
              </Button>
              <Button size="icon" variant={isPlaying ? 'default' : 'outline'} onClick={() => setIsPlaying(!isPlaying)}>
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </Button>
              <Button size="icon" variant="ghost" onClick={() => setCurrentLap(Math.min(params.totalLaps, currentLap + 5))}>
                <SkipForward className="w-4 h-4" />
              </Button>
              <div className="flex-1 mx-2">
                <input
                  type="range"
                  min={0}
                  max={params.totalLaps}
                  value={currentLap}
                  onChange={(e) => setCurrentLap(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <span className="text-sm font-mono w-16 text-right">
                第 {currentLap}/{params.totalLaps} 圈
              </span>
            </div>

            {/* 赛道可视化 - 简化版 */}
            <div className="relative h-64 bg-background/50 rounded-lg border border-border/30 p-3 overflow-hidden">
              {/* 赛道背景 - 用椭圆表示摩纳哥赛道 */}
              <svg viewBox="0 0 400 220" className="w-full h-full opacity-20">
                {/* 简化的摩纳哥赛道轮廓 */}
                <path
                  d="M 200 20 
                     C 280 20, 340 50, 350 100
                     C 360 150, 320 190, 260 195
                     C 200 200, 140 195, 100 180
                     C 50 160, 40 120, 50 90
                     C 60 50, 120 20, 200 20 Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="8"
                  strokeLinecap="round"
                />
              </svg>

              {/* 赛车位置 - 用水平条表示 */}
              <div className="absolute inset-0 p-4 flex flex-col justify-center gap-1">
                {frame?.cars.map((car, idx) => {
                  const progress = totalCars > 1 ? ((totalCars - car.position) / (totalCars - 1)) * 85 + 5 : 50;
                  const color = TIER_COLORS[car.tier];
                  return (
                    <div
                      key={car.id}
                      className="flex items-center gap-2 transition-all duration-200"
                      style={{ opacity: 1 - idx * 0.05 }}
                    >
                      <span className="text-xs font-mono w-5 text-right text-muted-foreground">
                        {car.position}
                      </span>
                      <div className="flex-1 h-5 relative bg-muted/30 rounded-full overflow-hidden">
                        <div
                          className="absolute top-0 left-0 h-full rounded-full transition-all duration-200 flex items-center px-2"
                          style={{
                            width: `${progress}%`,
                            backgroundColor: `${color}80`,
                            borderLeft: `3px solid ${color}`,
                          }}
                        >
                          <span className="text-[10px] font-bold text-white whitespace-nowrap">
                            {car.tier}-{car.id + 1}
                            {car.isPitLap && ' 🛠'}
                          </span>
                        </div>
                      </div>
                      <span className="text-[10px] font-mono w-14 text-right text-muted-foreground">
                        {car.gapToLeader > 0 ? `+${car.gapToLeader.toFixed(1)}s` : 'LEADER'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 图例 */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground justify-center">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-t1" />
                <span>争冠组</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-t2" />
                <span>中游组</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-t3" />
                <span>后方组</span>
              </div>
              <span className="text-[10px]">🛠 = 进站</span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
