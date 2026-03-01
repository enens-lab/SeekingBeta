import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Build a watchlist and get model context in minutes.</h2>
        <p>SeekingBeta gives you model-rated signals, probability bands, and benchmarked performance in one workflow.</p>
        <Link
          to="/signup"
          className="btn btn-primary btn-lg"
          onClick={() => trackEvent('cta_click', { section: 'final_cta', cta: 'create_free_account' })}
        >
          Create Free Account
        </Link>
      </div>
    </section>
  );
}

export default CallToAction;
