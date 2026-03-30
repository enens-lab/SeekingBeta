import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Open the boards for free.</h2>
        <p>Build a watchlist, compare probabilities, and see how the models behave before you ever upgrade.</p>
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
