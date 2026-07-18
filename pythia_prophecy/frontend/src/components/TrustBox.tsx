import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

/**
 * "What we are / what we aren't" — the honest-positioning box.
 * Market research (2026-07): the deepest buyer objection in both the stock-signal
 * and sports-pick markets is record trustworthiness, not pick quality. Saying the
 * quiet parts out loud (same models every tier, losses shown, no locks) is the
 * differentiator touts structurally cannot copy.
 */
function TrustBox() {
  return (
    <section className="trust-box" id="what-we-are">
      <div className="trust-box-inner">
        <h2 className="trust-box-title">Straight answers before you trust a model</h2>
        <div className="trust-box-columns">
          <div className="trust-box-col">
            <h3 className="trust-box-col-title trust-box-col-title--are">What we are</h3>
            <ul className="trust-box-list">
              <li>Model-generated research, published on a fixed schedule.</li>
              <li>
                The same signals for every subscriber — paid tiers see{' '}
                <strong>more of them, sooner</strong>, never better ones.
              </li>
              <li>
                A public, graded track record —{' '}
                <Link
                  to="/track-record"
                  onClick={() => trackEvent('trust_box_click', { destination: 'track_record' })}
                >
                  losses included
                </Link>
                .
              </li>
            </ul>
          </div>
          <div className="trust-box-col">
            <h3 className="trust-box-col-title trust-box-col-title--arent">What we aren&apos;t</h3>
            <ul className="trust-box-list">
              <li>Not a broker or a sportsbook — we never place trades or take bets.</li>
              <li>Not personalized financial advice.</li>
              <li>
                Not &ldquo;locks.&rdquo; Anyone promising 70% winners is selling something —
                here&apos;s{' '}
                <Link
                  to="/methodology"
                  onClick={() => trackEvent('trust_box_click', { destination: 'methodology' })}
                >
                  how we actually measure
                </Link>
                .
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

export default TrustBox;
