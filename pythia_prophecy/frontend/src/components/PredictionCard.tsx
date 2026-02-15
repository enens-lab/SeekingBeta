import { Prediction } from '../api/client';

interface PredictionCardProps {
  prediction: Prediction;
  onClick?: (ticker: string) => void;
}

function PredictionCard({ prediction, onClick }: PredictionCardProps) {
  const { ticker, signal, prob_up, last_close, horizon, modelDisplayName, model } = prediction as any;

  const signalClass = signal?.toLowerCase() || 'hold';
  const probability = prob_up !== null ? (prob_up * 100).toFixed(1) : '0.0';
  const probValue = prob_up ?? 0;
  const probClass = probValue >= 0.55 ? 'high' : probValue >= 0.45 ? 'medium' : 'low';
  const price = last_close?.toFixed(2) || '—';
  const displayModel = modelDisplayName || model || 'Unknown Model';

  return (
    <div
      className={`prediction-card${onClick ? ' clickable' : ''}`}
      onClick={() => onClick && onClick(ticker)}
    >
      <div className="prediction-header">
        <span className="prediction-ticker">{ticker}</span>
        <span className={`prediction-signal ${signalClass}`}>
          {signal || 'Hold'}
        </span>
      </div>
      <div className="prediction-model-badge">
        {displayModel}
      </div>
      <div className="prediction-stats">
        <div className="prediction-stat">
          <span className="prediction-stat-label">Last Price</span>
          <span className="prediction-stat-value">${price}</span>
        </div>
        <div className="prediction-stat">
          <span className="prediction-stat-label">Horizon</span>
          <span className="prediction-stat-value">{horizon || '1d'}</span>
        </div>
      </div>
      <div className="prediction-probability">
        <div className="prediction-stat">
          <span className="prediction-stat-label">Upside Probability</span>
          <span className="prediction-stat-value">{probability}%</span>
        </div>
        <div className="probability-bar-bg">
          <div
            className={`probability-bar ${probClass}`}
            style={{ width: `${probability}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export default PredictionCard;
