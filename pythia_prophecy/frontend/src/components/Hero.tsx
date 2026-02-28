import { Link } from 'react-router-dom';

function Hero() {
  const scrollToFeatures = () => {
    const element = document.getElementById('features');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="hero">
      <div className="hero-content">
        <h1 className="hero-title">Model-Driven Stock Signals for Modern Investors</h1>
        <p className="hero-subtitle">
          SeekingBeta converts market data into clear Bullish, Neutral, and Bearish ratings with
          probability context and track-record transparency, so you can cut through noise faster.
        </p>
        <div className="hero-cta">
          <Link to="/signup" className="btn btn-primary btn-lg">
            Start Free
          </Link>
          <button className="btn btn-outline btn-lg" onClick={scrollToFeatures}>
            See How It Works
          </button>
        </div>
        <p className="hero-disclaimer">
          For informational use only. Not investment advice.
        </p>
      </div>
    </section>
  );
}

export default Hero;
