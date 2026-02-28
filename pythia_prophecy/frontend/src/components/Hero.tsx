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
        <h1 className="hero-title">Learn market behavior with model-based stock ratings</h1>
        <p className="hero-subtitle">
          SeekingBeta translates complex market data into clear Bullish, Neutral, and Bearish
          model views so you can study scenarios faster and make your own informed decisions.
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
          Built for education and reference. Not investment advice.
        </p>
      </div>
    </section>
  );
}

export default Hero;
