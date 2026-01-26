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
        <h1 className="hero-title">AI-Powered Stock Signal Intelligence</h1>
        <p className="hero-subtitle">
          Harness machine learning to analyze market trends across multiple timeframes.
          Get actionable buy, hold, and sell signals for stocks in our curated universe.
        </p>
        <div className="hero-cta">
          <Link to="/signup" className="btn btn-primary btn-lg">
            Start Free Trial
          </Link>
          <button className="btn btn-outline btn-lg" onClick={scrollToFeatures}>
            Learn More
          </button>
        </div>
        <p className="hero-disclaimer">
          Educational purposes only. Not financial advice.
        </p>
      </div>
    </section>
  );
}

export default Hero;
