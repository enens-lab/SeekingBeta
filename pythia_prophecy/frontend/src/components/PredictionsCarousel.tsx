import { useState, useEffect, useCallback } from 'react';
import PredictionCard from './PredictionCard';
import { Prediction } from '../api/client';
import { trackEvent } from '../lib/analytics';

const ChevronLeft = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M15 18l-6-6 6-6" />
  </svg>
);

const ChevronRight = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M9 18l6-6-6-6" />
  </svg>
);

const InfoIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4M12 8h.01" />
  </svg>
);

// Model configurations with descriptions
const MODELS = [
  {
    name: 'lstm_5d',
    displayName: 'LSTM 5-Day',
    endpoint: '/predict/lstm_5d',
    description:
      'Short-horizon model focused on 5-day moves.'
  },
  {
    name: 'lstm_jackpot',
    displayName: 'LSTM Jackpot',
    endpoint: '/predict/lstm_jackpot',
    description:
      'Higher-volatility model focused on larger 20-day moves.'
  },
];

// Magnificent 7 stocks
const MAGNIFICENT_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'];

interface ModelRow {
  model: typeof MODELS[0];
  predictions: Prediction[];
  loading: boolean;
  error: boolean;
}

interface HomepageBatchRow {
  model: string;
  predictions: any[];
}

interface HomepageBatchResponse {
  available: boolean;
  rows: HomepageBatchRow[];
}

function PredictionsCarousel() {
  const [modelRows, setModelRows] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [carouselStates, setCarouselStates] = useState<{ [key: string]: number }>({});
  const [activeInfoModal, setActiveInfoModal] = useState<string | null>(null);

  const normalizePrediction = useCallback((model: typeof MODELS[0], ticker: string, data: any): Prediction => {
    const rawProbUp =
      typeof data.prob_up === 'number'
        ? data.prob_up
        : typeof data.probability === 'number'
        ? data.probability / 100
        : null;

    const probUp =
      typeof rawProbUp === 'number' && Number.isFinite(rawProbUp)
        ? rawProbUp
        : null;

    return {
      ...data,
      ticker,
      prob_up: probUp,
      predicted_return: data.predicted_return ?? null,
      last_close: data.last_close ?? null,
      model: model.name,
      modelDisplayName: model.displayName,
    };
  }, []);

  const fetchHomepageBatch = useCallback(async (): Promise<ModelRow[] | null> => {
    const res = await fetch('/predict/homepage');
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const payload = await res.json() as HomepageBatchResponse;
    if (!payload?.rows || !Array.isArray(payload.rows)) {
      throw new Error('Invalid homepage payload');
    }

    const rowsByModel = new Map(payload.rows.map((row) => [row.model, row]));
    const baseRows = MODELS.map((model) => {
      const row = rowsByModel.get(model.name);
      const predictions = Array.isArray(row?.predictions)
        ? row!.predictions.map((item) => normalizePrediction(model, item.ticker || '', item))
        : [];
      return {
        model,
        predictions,
        loading: false,
        error: predictions.length === 0,
      };
    });

    return baseRows;
  }, [normalizePrediction]);

  const fetchAllPredictions = useCallback(async () => {
    setLoading(true);

    try {
      const batchRows = await fetchHomepageBatch();
      if (batchRows) {
        setModelRows(batchRows);
      }
    } catch (batchError) {
      console.warn('Batch homepage prediction fetch failed; serving empty state', batchError);
      setModelRows(
        MODELS.map((model) => ({
          model,
          predictions: [],
          loading: false,
          error: true,
        }))
      );
    }
    setLoading(false);
  }, [fetchHomepageBatch]);

  useEffect(() => {
    fetchAllPredictions();
  }, [fetchAllPredictions]);

  const displayedRows = showAll ? modelRows : modelRows.slice(0, 2);

  const goToPrev = (modelName: string) => {
    trackEvent('carousel_slide', { model: modelName, direction: 'prev' });
    setCarouselStates((prev) => ({
      ...prev,
      [modelName]: Math.max(0, (prev[modelName] || 0) - 1),
    }));
  };

  const goToNext = (modelName: string, maxSlide: number) => {
    trackEvent('carousel_slide', { model: modelName, direction: 'next' });
    setCarouselStates((prev) => ({
      ...prev,
      [modelName]: Math.min(maxSlide - 1, (prev[modelName] || 0) + 1),
    }));
  };

  if (loading) {
    return (
      <section className="predictions-section" id="predictions">
        <div className="section-header">
          <h2 className="section-title">Today&apos;s Model Ratings</h2>
          <p className="section-subtitle">
            Compare two models across the Magnificent 7.
          </p>
        </div>
        <div className="predictions-loading">
          <div className="spinner" />
          <p>Loading today&apos;s model signals...</p>
        </div>
      </section>
    );
  }

  return (
    <section className="predictions-section" id="predictions">
      <div className="section-header">
        <h2 className="section-title">Today&apos;s Model Ratings</h2>
        <p className="section-subtitle">
          Compare two models across the Magnificent 7.
        </p>
      </div>

      <div className="model-rows-container">
        {displayedRows.map((row) => {
          const currentSlide = carouselStates[row.model.name] || 0;
          const cardsPerView = 4;
          const totalSlides = Math.max(1, Math.ceil(row.predictions.length / cardsPerView));
          const offsetPercentage = currentSlide * 100;

          return (
            <div key={row.model.name} className="model-row">
              <div className="model-row-header">
                <div className="model-row-title-container">
                  <h3 className="model-row-title">{row.model.displayName}</h3>
                  <button
                    className="model-info-btn"
                    onClick={() => {
                      trackEvent('model_info_open', { model: row.model.name });
                      setActiveInfoModal(row.model.name);
                    }}
                    aria-label={`Info about ${row.model.displayName}`}
                    title={`More details about ${row.model.displayName}`}
                  >
                    <InfoIcon />
                  </button>
                </div>
                <span className="model-row-count">
                  {row.predictions.length} of {MAGNIFICENT_7.length} stocks
                </span>
              </div>

              {row.error ? (
                <div className="model-row-error">
                  <p>Failed to load predictions for this model</p>
                </div>
              ) : (
                <div className="carousel-container">
                  <button
                    className="carousel-btn carousel-btn-prev"
                    onClick={() => goToPrev(row.model.name)}
                    disabled={currentSlide === 0}
                    aria-label="Previous"
                  >
                    <ChevronLeft />
                  </button>

                  <div className="carousel-viewport">
                    <div
                      className="carousel-track"
                      style={{ transform: `translateX(-${offsetPercentage}%)` }}
                    >
                      {row.predictions.map((prediction) => (
                        <PredictionCard
                          key={`${row.model.name}-${prediction.ticker}`}
                          prediction={prediction}
                          modelName={row.model.name}
                        />
                      ))}
                    </div>
                  </div>

                  <button
                    className="carousel-btn carousel-btn-next"
                    onClick={() => goToNext(row.model.name, totalSlides)}
                    disabled={currentSlide >= totalSlides - 1}
                    aria-label="Next"
                  >
                    <ChevronRight />
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!showAll && modelRows.length > 2 && (
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            className="btn btn-primary"
            onClick={() => {
              trackEvent('carousel_show_more_models');
              setShowAll(true);
            }}
          >
            View More Models ({modelRows.length - 2} more)
          </button>
        </div>
      )}

      {showAll && (
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            className="btn btn-outline"
            onClick={() => {
              trackEvent('carousel_show_less_models');
              setShowAll(false);
            }}
          >
            Show Less
          </button>
        </div>
      )}

      {activeInfoModal && (
        <div className="modal-overlay" onClick={() => setActiveInfoModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="modal-close"
              onClick={() => setActiveInfoModal(null)}
              aria-label="Close modal"
            >
              ×
            </button>
            {MODELS.find((m) => m.name === activeInfoModal) && (
              <>
                <h2 className="modal-title">
                  {MODELS.find((m) => m.name === activeInfoModal)?.displayName}
                </h2>
                <p className="modal-description">
                  {MODELS.find((m) => m.name === activeInfoModal)?.description}
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

export default PredictionsCarousel;
