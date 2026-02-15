import { useState, useEffect, useCallback } from 'react';
import PredictionCard from './PredictionCard';
import { Prediction } from '../api/client';

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

// Model configurations
const MODELS = [
  { name: 'gradient_boosting', displayName: 'Gradient Boosting', endpoint: '/predict' },
  { name: 'lstm_5d', displayName: 'LSTM 5-Day', endpoint: '/predict/lstm_5d' },
  { name: 'lstm_jackpot', displayName: 'LSTM Jackpot', endpoint: '/predict/lstm_jackpot' },
  { name: 'random_forest', displayName: 'Random Forest', endpoint: '/predict' },
  { name: 'linear_regression', displayName: 'Linear Regression', endpoint: '/predict' },
  { name: 'lstm', displayName: 'LSTM Classic', endpoint: '/predict' },
];

// Magnificent 7 stocks
const MAGNIFICENT_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'];

interface ModelRow {
  model: typeof MODELS[0];
  predictions: Prediction[];
  loading: boolean;
  error: boolean;
}

function PredictionsCarousel() {
  const [modelRows, setModelRows] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [carouselStates, setCarouselStates] = useState<{ [key: string]: number }>({});

  const fetchPredictionsForModel = useCallback(async (model: typeof MODELS[0]): Promise<ModelRow> => {
    try {
      const results = await Promise.allSettled(
        MAGNIFICENT_7.map(async (ticker) => {
          let url = '';
          if (model.endpoint === '/predict') {
            // Standard models: fetch from prophecy backend
            url = `/predict/${ticker}?horizon=1d&model=${model.name}`;
          } else {
            // LSTM models: fetch via prophecy backend proxy (e.g., /predict/lstm_5d/AAPL)
            url = `${model.endpoint}/${ticker}`;
          }

          const res = await fetch(url);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();

          return {
            ...data,
            ticker,
            model: model.name,
            modelDisplayName: model.displayName,
          };
        })
      );

      const predictions = results
        .filter((r): r is PromiseFulfilledResult<Prediction> => r.status === 'fulfilled')
        .map((r) => r.value);

      return {
        model,
        predictions,
        loading: false,
        error: predictions.length === 0,
      };
    } catch (err) {
      console.error(`Failed to fetch predictions for ${model.name}:`, err);
      return {
        model,
        predictions: [],
        loading: false,
        error: true,
      };
    }
  }, []);

  const fetchAllPredictions = useCallback(async () => {
    setLoading(true);

    const rows = await Promise.all(
      MODELS.map((model) => fetchPredictionsForModel(model))
    );

    setModelRows(rows);
    setLoading(false);
  }, [fetchPredictionsForModel]);

  useEffect(() => {
    fetchAllPredictions();
  }, [fetchAllPredictions]);

  const displayedRows = showAll ? modelRows : modelRows.slice(0, 2);

  const goToPrev = (modelName: string) => {
    setCarouselStates((prev) => ({
      ...prev,
      [modelName]: Math.max(0, (prev[modelName] || 0) - 1),
    }));
  };

  const goToNext = (modelName: string, maxSlide: number) => {
    setCarouselStates((prev) => ({
      ...prev,
      [modelName]: Math.min(maxSlide - 1, (prev[modelName] || 0) + 1),
    }));
  };

  if (loading) {
    return (
      <section className="predictions-section" id="predictions">
        <div className="section-header">
          <h2 className="section-title">Daily Prediction Signals</h2>
          <p className="section-subtitle">
            Comparing 6 AI models across the Magnificent 7 stocks
          </p>
        </div>
        <div className="predictions-loading">
          <div className="spinner" />
          <p>Loading predictions from 6 models...</p>
        </div>
      </section>
    );
  }

  return (
    <section className="predictions-section" id="predictions">
      <div className="section-header">
        <h2 className="section-title">Daily Prediction Signals</h2>
        <p className="section-subtitle">
          6 AI models analyzing the Magnificent 7 stocks - Each row shows one model's view
        </p>
      </div>

      <div className="model-rows-container">
        {displayedRows.map((row) => {
          const currentSlide = carouselStates[row.model.name] || 0;
          const cardsPerView = 4; // Adjust based on screen size
          const totalSlides = Math.max(1, Math.ceil(row.predictions.length / cardsPerView));
          const cardWidth = 280;
          const gap = 20;
          const offset = currentSlide * cardsPerView * (cardWidth + gap);

          return (
            <div key={row.model.name} className="model-row">
              <div className="model-row-header">
                <h3 className="model-row-title">{row.model.displayName}</h3>
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
                      style={{ transform: `translateX(-${offset}px)` }}
                    >
                      {row.predictions.map((prediction) => (
                        <PredictionCard
                          key={`${row.model.name}-${prediction.ticker}`}
                          prediction={prediction}
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
            onClick={() => setShowAll(true)}
          >
            View More Models ({modelRows.length - 2} more)
          </button>
        </div>
      )}

      {showAll && (
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            className="btn btn-outline"
            onClick={() => setShowAll(false)}
          >
            Show Less
          </button>
        </div>
      )}
    </section>
  );
}

export default PredictionsCarousel;
