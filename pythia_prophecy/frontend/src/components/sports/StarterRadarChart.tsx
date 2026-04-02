import { useMemo, type CSSProperties } from 'react';
import type { SportsRadarMetric } from '../../api/client';
import './SportsVisuals.css';

type StarterRadarChartProps = {
  title: string;
  subtitle?: string | null;
  metrics?: SportsRadarMetric[] | null;
  accentColor?: string | null;
};

function polarToCartesian(cx: number, cy: number, radius: number, angle: number) {
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  };
}

function StarterRadarChart({
  title,
  subtitle,
  metrics,
  accentColor,
}: StarterRadarChartProps) {
  const safeMetrics = metrics?.filter((metric) => Number.isFinite(metric.value)) ?? [];

  const geometry = useMemo(() => {
    const cx = 120;
    const cy = 120;
    const radius = 82;
    const axes = safeMetrics.length || 6;
    const startingAngle = -Math.PI / 2;

    const axisPoints = safeMetrics.map((metric, index) => {
      const angle = startingAngle + (index / axes) * Math.PI * 2;
      const outer = polarToCartesian(cx, cy, radius, angle);
      const valuePoint = polarToCartesian(cx, cy, radius * Math.max(0, Math.min(metric.value, 100)) / 100, angle);
      const labelPoint = polarToCartesian(cx, cy, radius + 20, angle);
      return { metric, angle, outer, valuePoint, labelPoint };
    });

    const rings = [0.25, 0.5, 0.75, 1.0].map((ratio) =>
      axisPoints
        .map((point) => polarToCartesian(cx, cy, radius * ratio, point.angle))
        .map((point) => `${point.x},${point.y}`)
        .join(' ')
    );

    const dataPolygon = axisPoints.map((point) => `${point.valuePoint.x},${point.valuePoint.y}`).join(' ');
    return { axisPoints, rings, dataPolygon };
  }, [safeMetrics]);

  const style = useMemo(
    () =>
      ({
        '--radar-accent': accentColor || '#4d99ff',
      }) as CSSProperties,
    [accentColor]
  );

  return (
    <div className="starter-radar" style={style}>
      <div className="starter-radar-header">
        <div>
          <h4 className="starter-radar-title">{title}</h4>
          {subtitle ? <p className="starter-radar-subtitle">{subtitle}</p> : null}
        </div>
      </div>

      {safeMetrics.length === 0 ? (
        <div className="starter-radar-empty">Probable starter pending. Radar fills in once a pregame starter profile is available.</div>
      ) : (
        <>
          <svg viewBox="0 0 240 240" className="starter-radar-graphic" role="img" aria-label={`${title} starter radar`}>
            {geometry.rings.map((points, index) => (
              <polygon
                key={`ring-${index}`}
                points={points}
                fill="none"
                stroke="rgba(255,255,255,0.08)"
                strokeWidth="1"
              />
            ))}
            {geometry.axisPoints.map((point) => (
              <line
                key={`axis-${point.metric.label}`}
                x1="120"
                y1="120"
                x2={point.outer.x}
                y2={point.outer.y}
                stroke="rgba(255,255,255,0.08)"
                strokeWidth="1"
              />
            ))}
            <polygon
              points={geometry.dataPolygon}
              fill="var(--radar-accent)"
              fillOpacity="0.18"
              stroke="var(--radar-accent)"
              strokeWidth="2"
            />
            {geometry.axisPoints.map((point) => (
              <circle
                key={`dot-${point.metric.label}`}
                cx={point.valuePoint.x}
                cy={point.valuePoint.y}
                r="3"
                fill="var(--radar-accent)"
              />
            ))}
          </svg>

          <ul className="starter-radar-labels">
            {safeMetrics.map((metric) => (
              <li key={metric.label}>
                <span>{metric.label}</span>
                <strong>{metric.value.toFixed(0)}</strong>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default StarterRadarChart;
