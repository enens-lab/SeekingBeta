import { MouseEvent, useEffect, useMemo, useState } from 'react';
import { performance, TrackRecordCurveResponse, TrackRecordResponse } from '../api/client';
import { trackEvent } from '../lib/analytics';

type TrackModel = 'lstm_5d' | 'lstm_jackpot';

type LoadedData = {
  curve: TrackRecordCurveResponse | null;
  summary: TrackRecordResponse | null;
};

type ChartPoint = {
  date: string;
  x: number;
  modelY: number;
  benchmarkY: number | null;
  modelValue: number;
  benchmarkValue: number | null;
  modelReturn: number | null;
  benchmarkReturn: number | null;
};

type ChartState = {
  width: number;
  height: number;
  padLeft: number;
  padRight: number;
  padTop: number;
  padBottom: number;
  xSpan: number;
  ySpan: number;
  min: number;
  max: number;
  modelPath: string;
  benchmarkPath: string;
  areaPath: string;
  startDate: string;
  midDate: string;
  endDate: string;
  points: ChartPoint[];
};

const MODEL_OPTIONS: Array<{ value: TrackModel; label: string }> = [
  { value: 'lstm_5d', label: 'Core 5-Day' },
  { value: 'lstm_jackpot', label: 'Jackpot 20-Day' },
];

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatMoney(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value);
}

function buildPath(xs: number[], ys: Array<number | null>): string {
  let path = '';
  let drawing = false;
  for (let i = 0; i < xs.length; i += 1) {
    const y = ys[i];
    if (y === null || Number.isNaN(y)) {
      drawing = false;
      continue;
    }
    if (!drawing) {
      path += `M ${xs[i].toFixed(2)} ${y.toFixed(2)} `;
      drawing = true;
    } else {
      path += `L ${xs[i].toFixed(2)} ${y.toFixed(2)} `;
    }
  }
  return path.trim();
}

function PerformanceComparison() {
  const [selectedModel, setSelectedModel] = useState<TrackModel>('lstm_5d');
  const [data, setData] = useState<LoadedData>({ curve: null, summary: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const REFRESH_INTERVAL_MS = 5 * 60 * 1000;

  const handleModelSelect = (model: TrackModel) => {
    if (model !== selectedModel) {
      trackEvent('track_record_model_switch', { model });
    }
    setSelectedModel(model);
  };

  useEffect(() => {
    let cancelled = false;
    const load = async (showSpinner: boolean) => {
      if (showSpinner) {
        setLoading(true);
      }
      setError(null);
      try {
        const [curve, summary] = await Promise.all([
          performance.getTrackRecordCurve(selectedModel).catch(() => null),
          performance.getTrackRecord(selectedModel).catch(() => null),
        ]);
        if (!cancelled) {
          setData({ curve, summary });
          setHoverIndex(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load performance data');
        }
      } finally {
        if (!cancelled && showSpinner) {
          setLoading(false);
        }
      }
    };
    void load(true);
    const refreshId = window.setInterval(() => {
      void load(false);
    }, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(refreshId);
    };
  }, [selectedModel]);

  const chart = useMemo<ChartState | null>(() => {
    const series = data.curve?.series ?? [];
    if (!data.curve?.available || series.length < 2) {
      return null;
    }

    const width = 940;
    const height = 360;
    const padLeft = 54;
    const padRight = 20;
    const padTop = 18;
    const padBottom = 36;
    const xSpan = width - padLeft - padRight;
    const ySpan = height - padTop - padBottom;

    const values: number[] = [];
    series.forEach((point) => {
      values.push(point.model_value);
      if (point.benchmark_value !== null && point.benchmark_value !== undefined) {
        values.push(point.benchmark_value);
      }
    });
    if (values.length === 0) {
      return null;
    }

    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      const bump = Math.max(1, Math.abs(min) * 0.05);
      min -= bump;
      max += bump;
    }
    const toY = (value: number) => padTop + ((max - value) / (max - min)) * ySpan;

    const xs = series.map((_, idx) => {
      if (series.length === 1) {
        return padLeft + xSpan / 2;
      }
      return padLeft + (idx / (series.length - 1)) * xSpan;
    });

    const modelYs = series.map((point) => toY(point.model_value));
    const benchmarkYs = series.map((point) => {
      if (point.benchmark_value === null || point.benchmark_value === undefined) {
        return null;
      }
      return toY(point.benchmark_value);
    });

    const modelPath = buildPath(xs, modelYs);
    const benchmarkPath = buildPath(xs, benchmarkYs);
    const baseY = height - padBottom;
    const areaPath = `${`M ${xs[0].toFixed(2)} ${baseY.toFixed(2)} `}${modelPath.slice(
      1,
    )} L ${xs[xs.length - 1].toFixed(2)} ${baseY.toFixed(2)} Z`;

    const startValue =
      data.curve.start_value && data.curve.start_value > 0 ? data.curve.start_value : series[0].model_value;

    const points: ChartPoint[] = series.map((point, idx) => {
      const benchmarkValue =
        point.benchmark_value === null || point.benchmark_value === undefined ? null : point.benchmark_value;
      return {
        date: point.date,
        x: xs[idx],
        modelY: modelYs[idx],
        benchmarkY: benchmarkYs[idx],
        modelValue: point.model_value,
        benchmarkValue,
        modelReturn: startValue > 0 ? point.model_value / startValue - 1.0 : null,
        benchmarkReturn: benchmarkValue !== null && startValue > 0 ? benchmarkValue / startValue - 1.0 : null,
      };
    });

    return {
      width,
      height,
      padLeft,
      padRight,
      padTop,
      padBottom,
      xSpan,
      ySpan,
      min,
      max,
      modelPath,
      benchmarkPath,
      areaPath,
      startDate: series[0].date,
      midDate: series[Math.floor(series.length / 2)].date,
      endDate: series[series.length - 1].date,
      points,
    };
  }, [data.curve]);

  if (loading) {
    return (
      <section className="performance-showcase">
        <div className="performance-showcase-inner">
          <div className="performance-loading">Loading the scorecard...</div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="performance-showcase">
        <div className="performance-showcase-inner">
          <div className="performance-error">{error}</div>
        </div>
      </section>
    );
  }

  if (!data.curve?.available || !chart) {
    return (
      <section className="performance-showcase">
        <div className="performance-showcase-inner">
          <div className="performance-empty">
            {data.curve?.message || 'The scorecard will appear here once backtest data is available.'}
          </div>
        </div>
      </section>
    );
  }

  const startValue = data.curve.start_value;
  const endValue = data.curve.end_value;
  const benchmarkEndValue = data.curve.benchmark_end_value;
  const modelReturn =
    startValue && endValue ? endValue / startValue - 1.0 : data.summary?.summary?.total_return_net ?? null;
  const benchmarkReturn =
    startValue && benchmarkEndValue
      ? benchmarkEndValue / startValue - 1.0
      : data.summary?.summary?.benchmark_return ?? null;
  const alpha =
    modelReturn !== null && modelReturn !== undefined && benchmarkReturn !== null && benchmarkReturn !== undefined
      ? modelReturn - benchmarkReturn
      : null;

  const yTicks = [0, 1, 2, 3, 4].map((idx) => {
    const value = chart.max - ((chart.max - chart.min) * idx) / 4;
    const y = chart.padTop + (chart.ySpan * idx) / 4;
    return { value, y };
  });

  const activePoint =
    hoverIndex !== null && hoverIndex >= 0 && hoverIndex < chart.points.length ? chart.points[hoverIndex] : null;
  const tooltipLeftPct = activePoint
    ? Math.max(6, Math.min(94, (activePoint.x / chart.width) * 100))
    : 0;

  const handleChartMove = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || chart.points.length < 2) {
      return;
    }
    const mouseX = ((event.clientX - rect.left) / rect.width) * chart.width;
    const clampedX = Math.max(chart.padLeft, Math.min(chart.width - chart.padRight, mouseX));
    const ratio = (clampedX - chart.padLeft) / chart.xSpan;
    const idx = Math.round(ratio * (chart.points.length - 1));
    setHoverIndex(Math.max(0, Math.min(chart.points.length - 1, idx)));
  };

  return (
    <section className="performance-showcase" id="performance">
      <div className="performance-showcase-inner">
        <div className="performance-header">
          <h2>Check the scorecard before you trust the board.</h2>
          <p>If the model cannot outperform a simple benchmark, it does not deserve your attention.</p>
        </div>

        <div className="performance-model-tabs" role="tablist" aria-label="Track record model">
          {MODEL_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              role="tab"
              aria-selected={selectedModel === option.value}
              className={`performance-model-tab${selectedModel === option.value ? ' active' : ''}`}
              onClick={() => handleModelSelect(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>

        <div className="performance-metrics-row">
          <div className="performance-chip">
            <span className="label">Model Return</span>
            <strong>{formatPct(modelReturn)}</strong>
          </div>
          <div className="performance-chip">
            <span className="label">SPY Return</span>
            <strong>{formatPct(benchmarkReturn)}</strong>
          </div>
          <div className="performance-chip">
            <span className="label">Alpha vs SPY</span>
            <strong className={alpha !== null && alpha >= 0 ? 'positive' : 'negative'}>{formatPct(alpha)}</strong>
          </div>
          <div className="performance-chip">
            <span className="label">Latest Equity</span>
            <strong>{formatMoney(endValue)}</strong>
          </div>
        </div>

        <div className="performance-chart-wrap">
          <svg
            className="performance-chart"
            viewBox={`0 0 ${chart.width} ${chart.height}`}
            role="img"
            aria-label="AI model performance versus S&P 500"
            onMouseMove={handleChartMove}
            onMouseLeave={() => setHoverIndex(null)}
          >
            <rect x="0" y="0" width={chart.width} height={chart.height} fill="transparent" />

            {yTicks.map((tick) => (
              <g key={`tick-${tick.y.toFixed(2)}`}>
                <line
                  x1={chart.padLeft}
                  x2={chart.width - chart.padRight}
                  y1={tick.y}
                  y2={tick.y}
                  className="chart-grid-line"
                />
                <text x={chart.padLeft - 8} y={tick.y + 4} className="chart-y-label">
                  {formatMoney(tick.value)}
                </text>
              </g>
            ))}

            <path d={chart.areaPath} className="chart-model-area" />
            <path d={chart.benchmarkPath} className="chart-benchmark-line" />
            <path d={chart.modelPath} className="chart-model-line" />

            {activePoint && (
              <>
                <line
                  x1={activePoint.x}
                  x2={activePoint.x}
                  y1={chart.padTop}
                  y2={chart.height - chart.padBottom}
                  className="chart-hover-line"
                />
                <circle cx={activePoint.x} cy={activePoint.modelY} r="4.5" className="chart-hover-point chart-hover-model" />
                {activePoint.benchmarkY !== null && (
                  <circle
                    cx={activePoint.x}
                    cy={activePoint.benchmarkY}
                    r="4"
                    className="chart-hover-point chart-hover-benchmark"
                  />
                )}
              </>
            )}

            <text x={chart.padLeft} y={chart.height - 10} className="chart-x-label">
              {chart.startDate}
            </text>
            <text
              x={chart.padLeft + chart.xSpan / 2}
              y={chart.height - 10}
              className="chart-x-label"
              textAnchor="middle"
            >
              {chart.midDate}
            </text>
            <text x={chart.width - chart.padRight} y={chart.height - 10} className="chart-x-label" textAnchor="end">
              {chart.endDate}
            </text>
          </svg>

          {activePoint && (
            <div className="performance-tooltip" style={{ left: `${tooltipLeftPct}%` }}>
              <div className="tooltip-date">{activePoint.date}</div>
              <div className="tooltip-row">
                <span>{data.curve.model_label}</span>
                <strong>{formatMoney(activePoint.modelValue)}</strong>
                <em>{formatPct(activePoint.modelReturn)}</em>
              </div>
              <div className="tooltip-row">
                <span>{data.curve.benchmark_label}</span>
                <strong>{formatMoney(activePoint.benchmarkValue)}</strong>
                <em>{formatPct(activePoint.benchmarkReturn)}</em>
              </div>
            </div>
          )}

          <div className="performance-legend">
            <span className="legend-item">
              <i className="legend-swatch legend-model" />
              {data.curve.model_label}
            </span>
            <span className="legend-item">
              <i className="legend-swatch legend-benchmark" />
              {data.curve.benchmark_label}
            </span>
          </div>
        </div>

        {data.summary?.summary && (
          <div className="performance-footnote">
            <span>Hit rate: {formatPct(data.summary.summary.hit_rate)}</span>
            <span>Sharpe: {data.summary.summary.sharpe_ratio?.toFixed(2) ?? '--'}</span>
            <span>Max drawdown: {formatPct(data.summary.summary.max_drawdown)}</span>
            <span>Trades: {data.summary.summary.sample_size}</span>
          </div>
        )}
      </div>
    </section>
  );
}

export default PerformanceComparison;
