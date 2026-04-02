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
        <h1 className="hero-title">The fastest way to scan stock and sports probabilities.</h1>
        <p className="hero-subtitle">
          SeekingBeta.AI gives self-directed investors one place to follow stock boards, golf outrights, tennis draws,
          and same-day MLB matchups, with context and track record built in.
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
          Research only. No brokerage, no wagering, and no event-contract settlement layer.
        </p>
      </div>
    </section>
  );
}

export default Hero;
