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
        <h1 className="hero-title">Prediction boards for stocks and sports</h1>
        <p className="hero-subtitle">
          Scan model-generated probabilities across equities, golf, and tennis in one clean research interface.
          SeekingBeta.AI brings the speed and clarity of a prediction board without betting, trading, or event contracts.
        </p>
        <div className="hero-cta">
          <Link
            to="/signup"
            className="btn btn-primary btn-lg"
            onClick={() => trackEvent('hero_cta_click', { cta: 'start_free' })}
          >
            Start Free
          </Link>
          <button className="btn btn-outline btn-lg" onClick={scrollToFeatures}>
            See How It Works
          </button>
        </div>
        <p className="hero-disclaimer">
          Educational probabilities only. No wagering, no settlement layer, and no financial or betting advice.
        </p>
      </div>
    </section>
  );
}

export default Hero;
