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

function PredictionsCarousel() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [cardsPerView, setCardsPerView] = useState(4);
  const [showAll, setShowAll] = useState(false);
  const [ticker, setTicker] = useState('AAPL'); // Default ticker

  const updateCardsPerView = useCallback(() => {
    const width = window.innerWidth;
    if (width < 640) {
      setCardsPerView(1);
    } else if (width < 900) {
      setCardsPerView(2);
    } else if (width < 1100) {
      setCardsPerView(3);
    } else {
      setCardsPerView(4);
    }
  }, []);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch predictions from all 6 models for the same ticker
      const results = await Promise.allSettled(
        MODELS.map(async (model) => {
          let url = '';
          if (model.endpoint === '/predict') {
            // Standard models use /predict/{ticker}?model=...
            url = `/predict/${ticker}?horizon=1d&model=${model.name}`;
          } else {
            // LSTM 5d and Jackpot have their own endpoints
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

      const successfulPredictions = results
        .filter((r): r is PromiseFulfilledResult<Prediction> => r.status === 'fulfilled')
        .map((r) => r.value);

      if (successfulPredictions.length === 0) {
        throw new Error('No predictions available');
      }

      setPredictions(successfulPredictions);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch predictions:', err);
      setError(err instanceof Error ? err.message : 'Failed to load predictions');
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    updateCardsPerView();
    fetchPredictions();

    window.addEventListener('resize', updateCardsPerView);
    return () => window.removeEventListener('resize', updateCardsPerView);
  }, [updateCardsPerView, fetchPredictions]);

  // Reset slide when cards per view changes
  useEffect(() => {
    setCurrentSlide(0);
  }, [cardsPerView]);

  // Filter predictions based on showAll state
  const displayedPredictions = showAll ? predictions : predictions.slice(0, 2);
  const totalSlides = Math.max(1, Math.ceil(displayedPredictions.length / cardsPerView));

  const goToPrev = () => {
    if (currentSlide > 0) {
      setCurrentSlide(currentSlide - 1);
    }
  };

  const goToNext = () => {
    if (currentSlide < totalSlides - 1) {
      setCurrentSlide(currentSlide + 1);
    }
  };

  const cardWidth = 280;
  const gap = 20;
  const offset = currentSlide * cardsPerView * (cardWidth + gap);

  if (loading) {
    return (
      <section className="predictions-section" id="predictions">
        <div className="section-header">
          <h2 className="section-title">Daily Prediction Signals</h2>
          <p className="section-subtitle">
            Multi-model predictions for {ticker} - See how different AI models analyze the same stock
          </p>
        </div>
        <div className="predictions-loading">
          <div className="spinner" />
          <p>Loading predictions from 6 models...</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="predictions-section" id="predictions">
        <div className="section-header">
          <h2 className="section-title">Daily Prediction Signals</h2>
          <p className="section-subtitle">
            Multi-model predictions for {ticker} - See how different AI models analyze the same stock
          </p>
        </div>
        <div className="predictions-error">
          <p>Unable to load predictions. Please try again later.</p>
          <button className="btn btn-outline" onClick={fetchPredictions}>
            Retry
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="predictions-section" id="predictions">
      <div className="section-header">
        <h2 className="section-title">Daily Prediction Signals</h2>
        <p className="section-subtitle">
          Comparing 6 AI models on {ticker} - Each model brings unique insights
        </p>
      </div>

      <div className="carousel-container">
        <button
          className="carousel-btn carousel-btn-prev"
          onClick={goToPrev}
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
            {displayedPredictions.map((prediction, index) => (
              <PredictionCard
                key={`${prediction.ticker}-${prediction.model || index}`}
                prediction={prediction}
              />
            ))}
          </div>
        </div>

        <button
          className="carousel-btn carousel-btn-next"
          onClick={goToNext}
          disabled={currentSlide >= totalSlides - 1}
          aria-label="Next"
        >
          <ChevronRight />
        </button>
      </div>

      <div className="carousel-dots">
        {Array.from({ length: totalSlides }, (_, i) => (
          <button
            key={i}
            className={`carousel-dot ${i === currentSlide ? 'active' : ''}`}
            onClick={() => setCurrentSlide(i)}
            aria-label={`Go to slide ${i + 1}`}
          />
        ))}
      </div>

      {!showAll && predictions.length > 2 && (
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            className="btn btn-primary"
            onClick={() => {
              setShowAll(true);
              setCurrentSlide(0);
            }}
          >
            View More Models ({predictions.length - 2} more)
          </button>
        </div>
      )}

      {showAll && (
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <button
            className="btn btn-outline"
            onClick={() => {
              setShowAll(false);
              setCurrentSlide(0);
            }}
          >
            Show Less
          </button>
        </div>
      )}
    </section>
  );
}

export default PredictionsCarousel;
