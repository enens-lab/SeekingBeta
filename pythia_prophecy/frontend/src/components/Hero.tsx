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
        <h1 className="hero-title">Every pick published, timestamped, and graded.</h1>
        <p className="hero-subtitle">
          AI model signals for stocks, golf, tennis, basketball, and baseball — with the full
          track record public, losses included. Audit us before you rely on us.
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
          For research only. We do not place trades or bets.
        </p>
      </div>
    </section>
  );
}

export default Hero;
