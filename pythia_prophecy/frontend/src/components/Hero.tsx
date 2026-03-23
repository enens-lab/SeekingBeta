import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function Hero() {
  const scrollToFeatures = () => {
    trackEvent('hero_cta_click', { cta: 'see_how_it_works' });
    const element = document.getElementById('features');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="hero">
      <div className="hero-content">
        <h1 className="hero-title">AI Prediction Markets for Finance & Sports</h1>
        <p className="hero-subtitle">
          Explore educational probability signals across stock markets, PGA tournaments, and major sports. 
          Our deep learning models are trained on vast historical datasets to rank outcomes with precision.
        </p>
        <div className="hero-cta">
          <Link
            to="/signup"
            className="btn btn-primary btn-lg"
            onClick={() => trackEvent('hero_cta_click', { cta: 'start_free' })}
          >
            Start Exploring
          </Link>
          <button className="btn btn-outline btn-lg" onClick={scrollToFeatures}>
            How It Works
          </button>
        </div>
        <p className="hero-disclaimer">
          For educational and informational use only. Not financial or betting advice.
        </p>
      </div>
    </section>
  );
}

export default Hero;
