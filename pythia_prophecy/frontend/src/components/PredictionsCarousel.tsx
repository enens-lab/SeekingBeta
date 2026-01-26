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

function PredictionsCarousel() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [cardsPerView, setCardsPerView] = useState(4);

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
      // Default universe (free tier)
      let universe = ['AAPL', 'MSFT', 'GOOGL'];

      // Try to fetch universe from API (respects tier limits)
      try {
        const universeRes = await fetch('/api/universe');
        if (universeRes.ok) {
          const data = await universeRes.json();
          if (data.universe) {
            universe = data.universe;
          }
        }
      } catch {
        console.warn('Could not fetch universe, using default');
      }

      // Fetch predictions for each ticker
      const results = await Promise.allSettled(
        universe.map((ticker) =>
          fetch(`/predict/${ticker}?horizon=1d`)
            .then((res) => {
              if (!res.ok) throw new Error(`HTTP ${res.status}`);
              return res.json();
            })
            .then((data) => ({ ...data, ticker }))
        )
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
  }, []);

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

  const totalSlides = Math.max(1, Math.ceil(predictions.length / cardsPerView));

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
          <p className="section-subtitle">Real-time ML predictions for stocks in our universe</p>
        </div>
        <div className="predictions-loading">
          <div className="spinner" />
          <p>Loading predictions...</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="predictions-section" id="predictions">
        <div className="section-header">
          <h2 className="section-title">Daily Prediction Signals</h2>
          <p className="section-subtitle">Real-time ML predictions for stocks in our universe</p>
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
        <p className="section-subtitle">Real-time ML predictions for stocks in our universe</p>
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
            {predictions.map((prediction, index) => (
              <PredictionCard key={prediction.ticker || index} prediction={prediction} />
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
    </section>
  );
}

export default PredictionsCarousel;
