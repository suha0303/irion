import { useEffect, useMemo, useRef } from 'react';
import * as echarts from 'echarts/core';
import { LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, MarkLineComponent, MarkAreaComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';
import type { Detection, Point } from './types';
echarts.use([LineChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, MarkAreaComponent, CanvasRenderer]);
export const motorColors = ['#079d8e', '#5878e0', '#d49536', '#9e6dc7'];

function Chart({ option, height, label }: { option: EChartsCoreOption; height: number; label: string }) {
  const element = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.EChartsType | null>(null);
  useEffect(() => {
    const chart = echarts.init(element.current!); instance.current = chart;
    const observer = new ResizeObserver(() => chart.resize()); observer.observe(element.current!);
    return () => { observer.disconnect(); chart.dispose(); instance.current = null; };
  }, []);
  useEffect(() => { instance.current?.setOption(option, { notMerge: true, lazyUpdate: false }); }, [option]);
  return <div ref={element} className="chart-canvas" style={{ height }} role="img" aria-label={label} />;
}

export function TimeChart({ points, cursor, signals, colors, height = 220, unit, event }: {
  points: Point[]; cursor: number; signals: { key: string; name: string }[]; colors: string[];
  height?: number; unit: string; event?: Detection | null;
}) {
  const option = useMemo(() => {
    const past = points.filter(p => p.t_ms <= cursor);
    return {
      animation: false, textStyle: { fontFamily: 'inherit', fontSize: 11 },
      grid: { top: 26, left: 53, right: 22, bottom: 30 },
      tooltip: { trigger: 'axis', backgroundColor: '#14243a', borderWidth: 0, textStyle: { color: '#fff', fontSize: 11 }, valueFormatter: (v: unknown) => v == null ? '—' : `${Number(v).toFixed(2)} ${unit}` },
      xAxis: { type: 'value', min: Math.max(0, cursor / 1000 - 60), max: Math.max(5, cursor / 1000), axisLabel: { color: '#8a95a5', formatter: (v: number) => `${v.toFixed(0)}s` }, axisLine: { lineStyle: { color: '#e6ebf1' } }, axisTick: { show: false }, splitLine: { show: false } },
      yAxis: { type: 'value', scale: true, name: unit, nameTextStyle: { color: '#8a95a5', align: 'right', padding: [0, 0, 0, -6] }, axisLabel: { color: '#8a95a5', formatter: (v: number) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : Number(v.toFixed(1)) }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } },
      series: signals.map((signal, index) => ({
        name: signal.name, type: 'line', showSymbol: false, connectNulls: false, smooth: false,
        data: past.map(p => [p.t_ms / 1000, p.values[signal.key]]),
        lineStyle: { color: colors[index], width: 2 }, itemStyle: { color: colors[index] },
        areaStyle: signals.length === 1 ? { color: colors[index], opacity: .045 } : undefined,
        markLine: index === 0 ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#b2becd', width: 1 }, data: [{ xAxis: cursor / 1000 }] } : undefined,
        markArea: index === 0 && event && event.decided_at_ms <= cursor ? { silent: true, itemStyle: { color: '#f5ad4520' }, data: [[{ xAxis: event.start_ms / 1000 }, { xAxis: event.end_ms / 1000 }]] } : undefined,
      })),
    };
  }, [points, cursor, signals, colors, unit, event]);
  return <Chart option={option} height={height} label={`${signals.map(s => s.name).join(', ')} 시계열. 단위 ${unit}. 재생 시각까지 표시.`} />;
}

export function PathChart({ points, cursor }: { points: Point[]; cursor: number }) {
  const option = useMemo(() => {
    const past = points.filter(p => p.t_ms <= cursor);
    const data = past.map(p => [p.values.east_m, p.values.north_m]);
    const last = data.filter(p => p[0] != null && p[1] != null).at(-1);
    const min = (extent: { min: number }) => Math.min(-1, extent.min - 1);
    const max = (extent: { max: number }) => Math.max(1, extent.max + 1);
    return { animation: false, grid: { top: 18, left: 45, right: 26, bottom: 37 }, tooltip: { trigger: 'item' },
      xAxis: { type: 'value', name: '동 (m)', nameLocation: 'middle', nameGap: 24, min, max, axisLabel: { color: '#93a0b1' }, splitLine: { lineStyle: { color: '#eaf0f5', type: 'dashed' } } },
      yAxis: { type: 'value', name: '북 (m)', min, max, axisLabel: { color: '#93a0b1' }, splitLine: { lineStyle: { color: '#eaf0f5', type: 'dashed' } } },
      series: [{ type: 'line', data, showSymbol: false, connectNulls: false, lineStyle: { color: '#18a998', width: 2 }, itemStyle: { color: '#18a998' } },
        { type: 'scatter', data: last ? [last] : [], symbolSize: 11, itemStyle: { color: '#079d8e', borderColor: '#fff', borderWidth: 3 } }],
    };
  }, [points, cursor]);
  return <Chart option={option} height={206} label="현재 재생 시점까지의 로컬 동·북 궤적. GPS 지도가 아닙니다." />;
}
