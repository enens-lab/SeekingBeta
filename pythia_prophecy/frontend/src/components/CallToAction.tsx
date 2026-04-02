import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Start with the live board.</h2>
        <p>Create a free account to follow stocks and sports, build a watchlist, and inspect the scorecard before you upgrade.</p>
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
