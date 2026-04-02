import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Try it for free.</h2>
        <p>See the live boards, build a watchlist, and pay only if you want more.</p>
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
