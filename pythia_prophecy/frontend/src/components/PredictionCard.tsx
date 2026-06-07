import { KeyboardEvent, MouseEvent, useEffect, useMemo, useState } from 'react';
import { Prediction, PredictionAttribution, predictions } from '../api/client';
import { trackEvent } from '../lib/analytics';

interface PredictionCardProps {
  prediction: Prediction;
  modelName?: string;
  onClick?: (ticker: string) => void;
}

type LstmModelName = 'lstm_5d' | 'lstm_jackpot';

const MODEL_SIGNAL_DISCLAIMER =
  'Model view only: this is an automated probability estimate for research. It is not a recommendation, order, or personalized advice.';

function formatSignalLabel(signal?: string | null): string {
  const normalized = (signal || '').toLowerCase();
  if (normalized === 'buy' || normalized === 'strong_buy') return 'Bullish';
  if (normalized === 'sell' || normalized === 'avoid') return 'Bearish';
  return 'Neutral';
}

function formatSignalClass(signal?: string | null): 'buy' | 'sell' | 'hold' {
  const normalized = (signal || '').toLowerCase();
  if (normalized === 'buy' || normalized === 'strong_buy') return 'buy';
  if (normalized === 'sell' || normalized === 'avoid') return 'sell';
  return 'hold';
}

function buildTopDrivers(probability: string, probValue: number, horizon: string): string[] {
  const directionalSummary =
    probValue >= 0.55
      ? 'Recent trend and volatility inputs lean bullish.'
      : probValue <= 0.45
      ? 'Recent trend is weaker and downside pressure is elevated.'
      : 'Recent momentum is mixed with no strong directional edge.';

  return [
    `Estimated upside probability is ${probability}% over the ${horizon} horizon.`,
    directionalSummary,
  ];
}

function prettifyFeatureName(feature: string): string {
  return feature
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function friendlyFeatureName(feature: string): string {
  const normalized = feature.toLowerCase().replace(/[^a-z0-9]/g, '');
  const labels: Record<string, string> = {
    bollingerupper: 'Upper Bollinger band',
    bollingerlower: 'Lower Bollinger band',
    yesterdayclose: 'Previous close',
    volume: 'Volume',
    rsi: 'RSI momentum',
    macd: 'MACD momentum',
    numarticles: 'News volume',
  };
  return labels[normalized] ?? prettifyFeatureName(feature);
}

function buildAttributionTopDrivers(
  attribution: PredictionAttribution | null | undefined,
  probValue: number,
  fallback: string[]
): string[] {
  if (!attribution || typeof attribution !== 'object') {
    return fallback;
  }

  const rawDrivers = Array.isArray(attribution.top_drivers)
    ? attribution.top_drivers
    : [];

  const drivers = rawDrivers.filter((d: any) => d && typeof d.feature === 'string').slice(0, 3);
  if (drivers.length === 0) {
    return fallback;
  }

  const positiveCount = drivers.filter(
    (d: any) => String(d.direction || '').toLowerCase() !== 'negative'
  ).length;
  const negativeCount = drivers.length - positiveCount;
  const biasLine =
    positiveCount > negativeCount
      ? 'Model lean: mildly bullish.'
      : negativeCount > positiveCount
      ? 'Model lean: mildly bearish.'
      : probValue >= 0.5
      ? 'Model lean: slightly bullish.'
      : 'Model lean: mixed to neutral.';

  const keyFactors = drivers
    .map((d: any) => friendlyFeatureName(String(d.feature)))
    .filter((name: string, index: number, arr: string[]) => arr.indexOf(name) === index)
    .slice(0, 3);

  return [
    biasLine,
    `Key factors: ${keyFactors.join(', ')}.`,
    'This read is approximate and can shift as fresh data comes in.',
  ];
}

function normalizeLstmModelName(_model: string | null | undefined): LstmModelName | null {
  // lstm_5d/lstm_jackpot are now torch options models with no integrated-gradients
  // attribution (it was keras/TF-specific). Skip the attribution fetch so the card
  // uses its trend/volatility fallback drivers rather than mismatched keras reasoning.
  return null;
}

function safeNumber(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }
  return value;
}

function PredictionCard({ prediction, modelName, onClick }: PredictionCardProps) {
  const { ticker, signal, prob_up, last_close, horizon } = prediction as any;
  const [isFlipped, setIsFlipped] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [attribution, setAttribution] = useState<PredictionAttribution | null>(
    prediction.attribution ?? null
  );
  const [attributionLoading, setAttributionLoading] = useState(false);
  const [attributionError, setAttributionError] = useState<string | null>(null);

  const resolvedModelName = useMemo(
    () => normalizeLstmModelName(modelName ?? (prediction as any).model),
    [modelName, prediction]
  );

  useEffect(() => {
    setAttribution(prediction.attribution ?? null);
    setAttributionLoading(false);
    setAttributionError(null);
  }, [prediction]);

  const signalClass = formatSignalClass(signal);
  const signalLabel = formatSignalLabel(signal);
  const probUpValue = safeNumber(prob_up);
  const lastCloseValue = safeNumber(last_close);
  const probValue = probUpValue ?? 0;
  const probability = (probValue * 100).toFixed(1);
  const probClass = probValue >= 0.55 ? 'high' : probValue >= 0.45 ? 'medium' : 'low';
  const price = lastCloseValue !== null ? lastCloseValue.toFixed(2) : '—';
  const horizonLabel = horizon || '1d';
  const topDrivers = buildAttributionTopDrivers(
    attribution,
    probValue,
    buildTopDrivers(probability, probValue, horizonLabel)
  );

  const handleFlip = () => {
    trackEvent('prediction_card_flip', {
      ticker,
      model: resolvedModelName ?? 'unknown',
      side: isFlipped ? 'front' : 'back',
    });
    trackEvent('ticker_interaction', {
      ticker,
      action: 'card_flip',
      surface: onClick ? 'dashboard' : 'landing',
    });
    setIsFlipped((prev) => {
      const next = !prev;
      if (next && !attribution && !attributionLoading && resolvedModelName && ticker) {
        setAttributionLoading(true);
        setAttributionError(null);
        predictions
          .getAttribution(resolvedModelName, ticker, 3)
          .then((response) => {
            if (response?.attribution) {
              setAttribution(response.attribution);
            } else {
              setAttributionError('Attribution details are temporarily unavailable.');
            }
          })
          .catch(() => {
            setAttributionError('Attribution details are temporarily unavailable.');
          })
          .finally(() => {
            setAttributionLoading(false);
          });
      }
      return next;
    });
    setShowInfo(false);
  };

  const handleKeyFlip = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleFlip();
    }
  };

  const handleInfoClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setShowInfo((prev) => !prev);
  };

  const handleOpenCompany = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    trackEvent('company_profile_open', { ticker, source: 'prediction_card' });
    trackEvent('ticker_interaction', {
      ticker,
      action: 'company_profile_open',
      surface: onClick ? 'dashboard' : 'landing',
    });
    if (onClick) {
      onClick(ticker);
    }
  };

  return (
    <div
      className={`prediction-card${onClick ? ' clickable' : ''}${isFlipped ? ' is-flipped' : ''}`}
      onClick={handleFlip}
      onKeyDown={handleKeyFlip}
      role="button"
      tabIndex={0}
      aria-label={`${ticker} model card. Press to ${isFlipped ? 'show front' : 'show top drivers'}.`}
      aria-pressed={isFlipped}
    >
      <div className="prediction-card-inner">
        <div className="prediction-card-face prediction-card-front">
          <div className="prediction-header">
            <span className="prediction-ticker">{ticker}</span>
            <div className="prediction-signal-stack">
              <div className="prediction-signal-row">
                <span className={`prediction-signal ${signalClass}`}>{signalLabel}</span>
                <button
                  type="button"
                  className="prediction-info-btn"
                  onClick={handleInfoClick}
                  aria-label="More info about model signal"
                  aria-expanded={showInfo}
                >
                  i
                </button>
              </div>
              {showInfo && <div className="prediction-signal-popover">{MODEL_SIGNAL_DISCLAIMER}</div>}
            </div>
          </div>
          <div className="prediction-stats">
            <div className="prediction-stat">
              <span className="prediction-stat-label">Last Close</span>
              <span className="prediction-stat-value">${price}</span>
            </div>
            <div className="prediction-stat">
              <span className="prediction-stat-label">Horizon</span>
              <span className="prediction-stat-value">{horizonLabel}</span>
            </div>
          </div>
          <div className="prediction-probability">
            <div className="prediction-stat">
              <span className="prediction-stat-label">Upside Odds</span>
              <span className="prediction-stat-value">{probability}%</span>
            </div>
            <div className="probability-bar-bg">
              <div className={`probability-bar ${probClass}`} style={{ width: `${probability}%` }} />
            </div>
          </div>

          <div className="prediction-card-actions">
            {onClick && (
              <button
                type="button"
                className="prediction-company-btn"
                onClick={handleOpenCompany}
                aria-label={`Open ${ticker} company profile`}
              >
                Company Profile
              </button>
            )}
            <span className="prediction-flip-hint">Flip for the model read</span>
          </div>
        </div>

        <div className="prediction-card-face prediction-card-back">
          <div className="prediction-back-header">
            <span className="prediction-ticker">{ticker}</span>
            <span className="prediction-back-title">Why the board leans this way</span>
          </div>
          <ul className="prediction-drivers-list">
            {topDrivers.map((driver, index) => (
              <li key={`${ticker}-driver-${index}`}>{driver}</li>
            ))}
          </ul>
          {attributionLoading && (
            <div className="prediction-drivers-loading">Pulling the latest factor summary...</div>
          )}
          {attributionError && (
            <div className="prediction-drivers-loading">Using a fallback summary for now.</div>
          )}

          <div className="prediction-card-actions">
            {onClick && (
              <button
                type="button"
                className="prediction-company-btn"
                onClick={handleOpenCompany}
                aria-label={`Open ${ticker} company profile`}
              >
                Company Profile
              </button>
            )}
            <span className="prediction-flip-hint">Flip back to the live card</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PredictionCard;
