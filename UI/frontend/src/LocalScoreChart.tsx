import { Chart } from './Charts';
import type { Analysis } from './local-flight-types';

type ScorePoint = [number, number];

export function scoreChartData(analysis: Analysis) {
  const scoring = analysis.scoring;
  if (!scoring?.scores.length || scoring.scores.length !== scoring.time_us.length) return;

  const threshold =
    analysis.decision?.status === 'unavailable' ? undefined : analysis.decision?.threshold;
  const points: ScorePoint[] = [];
  const exceeded: ScorePoint[] = [];
  const lines: ScorePoint[][] = [];
  let previousSegment = -1;
  let previousTime = -Infinity;

  for (const [i, score] of scoring.scores.entries()) {
    const time = scoring.time_us[i];
    if (!Number.isFinite(time) || !Number.isFinite(score)) {
      previousSegment = -1;
      continue;
    }
    const point: ScorePoint = [time / 1e6, score];
    points.push(point);
    if (threshold !== undefined && Number.isFinite(threshold) && score > threshold) {
      exceeded.push(point);
    }
    const segment = analysis.metrics.series?.findIndex((s) => s.time_us.includes(time)) ?? -1;
    if (segment >= 0) {
      if (segment !== previousSegment || time <= previousTime) lines.push([]);
      lines[lines.length - 1].push(point);
    }
    previousSegment = segment;
    previousTime = time;
  }

  return { points, exceeded, lines, threshold: Number.isFinite(threshold) ? threshold : undefined };
}

export default function LocalScoreChart({
  analysis,
  sourceLabel,
  clockLabel,
}: {
  analysis: Analysis;
  sourceLabel: string;
  clockLabel: string;
}) {
  const data = scoreChartData(analysis);
  if (!data?.points.length)
    return <p>점수와 시점 자료가 없거나 일치하지 않아 그래프를 표시할 수 없습니다.</p>;

  const { points, exceeded, lines, threshold } = data;
  return (
    <section className="local-score-chart" aria-label={`${analysis.model_id} 이상 점수 그래프`}>
      <h3>이상 점수와 임계값</h3>
      <p className="local-model-name">
        {analysis.model_id} · {sourceLabel}
      </p>
      <div className="local-score-legend">
        <span className="score">━ 이상 점수</span>
        <span className="threshold">┄ 임계값 {threshold?.toFixed(5) ?? '없음'}</span>
        <span className="exceeded">
          ● 기준 초과{' '}
          {threshold === undefined ? '판정 불가' : `${exceeded.length} / ${points.length}시점`}
        </span>
      </div>
      <Chart
        height={320}
        label={`${analysis.model_id}. ${clockLabel}. 이상 점수 ${points.length}개, 임계값 ${threshold ?? '없음'}, 기준 초과 ${exceeded.length}시점.`}
        option={{
          animation: false,
          textStyle: { fontFamily: 'sans-serif', fontSize: 12 },
          grid: { left: 46, right: 20, top: 36, bottom: 52 },
          tooltip: { trigger: 'axis', valueFormatter: (v: number) => v.toFixed(5) },
          xAxis: {
            type: 'value',
            name: clockLabel,
            nameLocation: 'middle',
            nameGap: 30,
            min: (extent: { min: number }) => Math.floor(extent.min / 10) * 10 - 10,
            axisLabel: { hideOverlap: true },
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value',
            name: '이상 점수',
            splitLine: { lineStyle: { type: 'dashed', color: '#e4eaf0' } },
          },
          series: [
            ...lines.map((line) => ({
              name: '이상 점수',
              type: 'line',
              data: line,
              showSymbol: false,
              smooth: false,
              connectNulls: false,
              lineStyle: { color: '#208778', width: 2 },
              tooltip: { show: false },
            })),
            {
              name: '이상 점수',
              type: 'scatter',
              data: points,
              symbolSize: 4,
              itemStyle: { color: '#208778' },
              markLine:
                threshold === undefined
                  ? undefined
                  : {
                      silent: true,
                      symbol: 'none',
                      lineStyle: { color: '#b86c23', type: 'dashed', width: 2 },
                      label: {
                        formatter: `임계값 ${threshold.toFixed(3)}`,
                        position: 'insideEndBottom',
                        color: '#925416',
                      },
                      data: [{ yAxis: threshold }],
                    },
            },
            {
              name: '기준 초과',
              type: 'scatter',
              data: exceeded,
              symbolSize: 10,
              z: 5,
              itemStyle: { color: '#cc3749', borderColor: '#fff', borderWidth: 1 },
              label: {
                show: true,
                position: 'top',
                color: '#b52b3c',
                formatter: (p: { value: number[] }) => `${p.value[0].toFixed(1)}초`,
              },
              labelLayout: { hideOverlap: true },
            },
          ],
        }}
      />
      <p className="muted">
        빨간 점은 점수가 임계값을 초과한 시점입니다. 경보 유지·해제 조건이 적용된 경보 이력 및 실제
        고장 구간과 구분합니다.
      </p>
    </section>
  );
}
