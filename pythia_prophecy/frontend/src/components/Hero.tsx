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
        <h1 className="hero-title">Stock signals made simple for everyday investors</h1>
        <p className="hero-subtitle">
          SeekingBeta turns complex market data into clear Buy, Hold, and Sell guidance.
          No finance jargon required, just a clean second opinion before you place a trade.
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
          Built for learning and reference. Always do your own due diligence.
        </p>
      </div>
    </section>
  );
}

export default Hero;
