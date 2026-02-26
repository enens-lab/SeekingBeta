import { useEffect, useMemo, useState } from 'react';
import {
  performance,
  TrackRecordCurveResponse,
  TrackRecordResponse,
} from '../api/client';

type LoadedData = {
  curve: TrackRecordCurveResponse | null;
  summary: TrackRecordResponse | null;
};

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
  const [data, setData] = useState<LoadedData>({ curve: null, summary: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [curve, summary] = await Promise.all([
          performance.getTrackRecordCurve().catch(() => null),
          performance.getTrackRecord().catch(() => null),
        ]);

        if (!cancelled) {
          setData({ curve, summary });
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load performance data');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const chart = useMemo(() => {
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
    const areaPath = `${`M ${xs[0].toFixed(2)} ${baseY.toFixed(2)} `}${modelPath.slice(1)} L ${xs[xs.length - 1].toFixed(2)} ${baseY.toFixed(2)} Z`;

    const startDate = series[0].date;
    const midDate = series[Math.floor(series.length / 2)].date;
    const endDate = series[series.length - 1].date;

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
      xs,
      modelPath,
      benchmarkPath,
      areaPath,
      startDate,
      midDate,
      endDate,
    };
  }, [data.curve]);

  if (loading) {
    return (
      <section className="performance-showcase">
        <div className="performance-showcase-inner">
          <div className="performance-loading">Loading performance track record...</div>
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
            {data.curve?.message || 'Track record data will appear here once backtest data is loaded.'}
          </div>
        </div>
      </section>
    );
  }

  const startValue = data.curve.start_value;
  const endValue = data.curve.end_value;
  const benchmarkEndValue = data.curve.benchmark_end_value;
  const modelReturn =
    startValue && endValue ? (endValue / startValue) - 1.0 : data.summary?.summary?.total_return_net ?? null;
  const benchmarkReturn =
    startValue && benchmarkEndValue
      ? (benchmarkEndValue / startValue) - 1.0
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

  return (
    <section className="performance-showcase" id="performance">
      <div className="performance-showcase-inner">
        <div className="performance-header">
          <h2>Proof of Performance</h2>
          <p>Public track record benchmarked against the S&amp;P 500 (SPY).</p>
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

            <text x={chart.padLeft} y={chart.height - 10} className="chart-x-label">
              {chart.startDate}
            </text>
            <text x={chart.padLeft + chart.xSpan / 2} y={chart.height - 10} className="chart-x-label" textAnchor="middle">
              {chart.midDate}
            </text>
            <text x={chart.width - chart.padRight} y={chart.height - 10} className="chart-x-label" textAnchor="end">
              {chart.endDate}
            </text>
          </svg>

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
