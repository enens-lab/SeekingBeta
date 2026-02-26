import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { tiers as tiersApi } from '../api/client';

interface TierData {
  tier: string;
  name: string;
  price: number;
  stocks_limit: number;
  timeframes: string[];
  features: string[];
}

function Pricing() {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuth();
  const [tiers, setTiers] = useState<TierData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    tiersApi.getAll()
      .then((data) => setTiers(data as unknown as TierData[]))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleSelectTier = (tier: string) => {
    if (isAuthenticated) {
      // TODO: Handle upgrade flow
      alert('Upgrade flow coming soon!');
    } else {
      navigate(`/signup?tier=${tier}`);
    }
  };

  const CheckIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );

  return (
    <div className="pricing-page">
      <header className="site-header">
        <div className="header-content">
          <Link to="/" className="logo">
            <span className="logo-icon">β</span>
            <span className="logo-text">SeekingBeta</span>
          </Link>
          <nav className="nav">
            <Link to="/#predictions" className="nav-link">Signals</Link>
            <Link to="/#performance" className="nav-link">Track Record</Link>
            <Link to="/#features" className="nav-link">How It Works</Link>
            <Link to="/pricing" className="nav-link">Pricing</Link>
          </nav>
          <div className="header-actions">
            {isAuthenticated ? (
              <Link to="/dashboard" className="btn btn-primary">Dashboard</Link>
            ) : (
              <>
                <Link to="/login" className="btn btn-ghost">Log In</Link>
                <Link to="/signup" className="btn btn-primary">Get Started</Link>
              </>
            )}
          </div>
        </div>
      </header>

      <main>
        <section className="pricing-hero">
          <h1>Simple pricing for everyday investors</h1>
          <p>Choose the level of guidance that fits your goals</p>
        </section>

        <section className="pricing-cards">
          {loading ? (
            <div className="pricing-loading">
              <div className="spinner" />
              <p>Loading plans...</p>
            </div>
          ) : (
            <div className="pricing-grid">
              {tiers.map((tier) => (
                <div
                  key={tier.tier}
                  className={`pricing-card ${tier.tier === 'basic' ? 'popular' : ''}`}
                >
                  {tier.tier === 'basic' && (
                    <div className="popular-badge">Most Popular</div>
                  )}
                  <div className="pricing-card-header">
                    <h2>{tier.name}</h2>
                    <div className="pricing-amount">
                      {tier.price === 0 ? (
                        <span className="price">Free</span>
                      ) : (
                        <>
                          <span className="currency">$</span>
                          <span className="price">{tier.price}</span>
                          <span className="period">/month</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="pricing-card-body">
                    <div className="pricing-highlights">
                      <div className="highlight">
                        <strong>{tier.stocks_limit === -1 ? 'Unlimited' : tier.stocks_limit}</strong>
                        <span>stocks</span>
                      </div>
                      <div className="highlight">
                        <strong>{tier.timeframes.length}</strong>
                        <span>timeframe{tier.timeframes.length > 1 ? 's' : ''}</span>
                      </div>
                    </div>

                    <div className="pricing-timeframes">
                      <span className="label">Timeframes:</span>
                      <span className="values">{tier.timeframes.join(', ')}</span>
                    </div>

                    <ul className="pricing-features">
                      {tier.features.map((feature, index) => (
                        <li key={index}>
                          <CheckIcon />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="pricing-card-footer">
                    <button
                      className={`btn btn-block ${tier.tier === 'basic' ? 'btn-primary' : 'btn-outline'}`}
                      onClick={() => handleSelectTier(tier.tier)}
                      disabled={isAuthenticated && user?.tier === tier.tier}
                    >
                      {isAuthenticated && user?.tier === tier.tier
                        ? 'Current Plan'
                        : tier.price === 0
                        ? 'Get Started Free'
                        : 'Choose Plan'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="pricing-faq">
          <h2>Frequently Asked Questions</h2>
          <div className="faq-grid">
            <div className="faq-item">
              <h3>Can I change my plan later?</h3>
              <p>Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately.</p>
            </div>
            <div className="faq-item">
              <h3>What payment methods do you accept?</h3>
              <p>We accept all major credit cards. Enterprise plans can be paid via invoice.</p>
            </div>
            <div className="faq-item">
              <h3>Is there a free trial for paid plans?</h3>
              <p>Yes, all paid plans come with a 14-day free trial. No credit card required to start.</p>
            </div>
            <div className="faq-item">
              <h3>What happens if I exceed my stock limit?</h3>
              <p>You'll be prompted to upgrade to a higher tier to access additional stocks.</p>
            </div>
          </div>
        </section>

        <section className="pricing-cta">
          <h2>Ready to trade with more confidence?</h2>
          <p>Join investors using SeekingBeta as a daily decision support layer</p>
          <Link to="/signup" className="btn btn-primary btn-lg">
            Start Free
          </Link>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="logo-icon">β</span>
            <span className="logo-text">SeekingBeta</span>
          </div>
          <p className="footer-disclaimer">
            SeekingBeta is for educational and research purposes only.
            Past performance does not guarantee future results.
          </p>
          <p className="footer-copyright">
            &copy; {new Date().getFullYear()} SeekingBeta. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default Pricing;
