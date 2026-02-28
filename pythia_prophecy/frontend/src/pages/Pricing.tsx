import { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { tiers as tiersApi, billing as billingApi, BillingStatus } from '../api/client';
import { useToast } from '../components/Toast';
import ThemeToggle from '../components/ThemeToggle';

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
  const [searchParams, setSearchParams] = useSearchParams();
  const { isAuthenticated, isVerified, user } = useAuth();
  const toast = useToast();

  const [tiers, setTiers] = useState<TierData[]>([]);
  const [loading, setLoading] = useState(true);
  const [billingStatus, setBillingStatus] = useState<BillingStatus | null>(null);
  const [checkoutTier, setCheckoutTier] = useState<string | null>(null);
  const [portalLoading, setPortalLoading] = useState(false);

  useEffect(() => {
    tiersApi
      .getAll()
      .then((data) => setTiers(data as unknown as TierData[]))
      .catch((err) => {
        console.error(err);
        toast.error('Failed to load pricing plans');
      })
      .finally(() => setLoading(false));
  }, [toast]);

  useEffect(() => {
    if (!isAuthenticated) {
      setBillingStatus(null);
      return;
    }

    billingApi
      .getStatus()
      .then(setBillingStatus)
      .catch((err) => {
        console.error(err);
      });
  }, [isAuthenticated]);

  useEffect(() => {
    const checkout = searchParams.get('checkout');
    if (!checkout) {
      return;
    }

    if (checkout === 'success') {
      toast.success('Checkout complete. Subscription is syncing now.');
      billingApi.getStatus().then(setBillingStatus).catch(console.error);
    } else if (checkout === 'cancelled') {
      toast.error('Checkout canceled. No changes were made.');
    } else if (checkout === 'portal_return') {
      billingApi.getStatus().then(setBillingStatus).catch(console.error);
    }

    searchParams.delete('checkout');
    setSearchParams(searchParams, { replace: true });
  }, [searchParams, setSearchParams, toast]);

  const currentTier = billingStatus?.effective_tier || user?.tier || 'free';

  const handleSelectTier = async (tier: string) => {
    if (tier === 'free') {
      if (isAuthenticated) {
        navigate('/dashboard');
      } else {
        navigate('/signup');
      }
      return;
    }

    if (!isAuthenticated) {
      navigate('/login', { state: { from: { pathname: '/pricing' } } });
      return;
    }

    if (!isVerified) {
      toast.error('Verify your email before upgrading your plan.');
      return;
    }

    if (tier !== 'basic' && tier !== 'pro') {
      toast.error('Unsupported paid tier selected.');
      return;
    }

    setCheckoutTier(tier);
    try {
      const result = await billingApi.createCheckoutSession(tier);
      window.location.assign(result.checkout_url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start checkout';
      toast.error(message);
    } finally {
      setCheckoutTier(null);
    }
  };

  const handleManageBilling = async () => {
    setPortalLoading(true);
    try {
      const result = await billingApi.createPortalSession();
      window.location.assign(result.portal_url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to open billing portal';
      toast.error(message);
    } finally {
      setPortalLoading(false);
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
            <Link to="/#predictions" className="nav-link">Model Views</Link>
            <Link to="/#performance" className="nav-link">Track Record</Link>
            <Link to="/#features" className="nav-link">How It Works</Link>
            <Link to="/pricing" className="nav-link">Pricing</Link>
          </nav>
          <div className="header-actions">
            <ThemeToggle />
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
          <h1>Simple pricing for investors who want structured learning tools</h1>
          <p>Choose the plan that fits how deeply you want to analyze model output</p>
        </section>

        {isAuthenticated && billingStatus?.billing_enabled && (
          <section className="pricing-billing-status">
            <div className="pricing-billing-status-content">
              <div>
                <h3>Billing</h3>
                <p>
                  Current plan: <strong>{currentTier.toUpperCase()}</strong>
                  {billingStatus.subscription_status ? (
                    <span> · Status: {billingStatus.subscription_status}</span>
                  ) : null}
                </p>
              </div>
              <button
                className="btn btn-outline"
                onClick={handleManageBilling}
                disabled={portalLoading}
              >
                {portalLoading ? 'Opening...' : 'Manage Billing'}
              </button>
            </div>
          </section>
        )}

        <section className="pricing-cards">
          {loading ? (
            <div className="pricing-loading">
              <div className="spinner" />
              <p>Loading plans...</p>
            </div>
          ) : (
            <div className="pricing-grid">
              {tiers.map((tier) => {
                const isCurrent = currentTier === tier.tier;
                const isPaidTier = tier.price > 0;
                const isCheckoutLoading = checkoutTier === tier.tier;
                return (
                  <div
                    key={tier.tier}
                    className={`pricing-card ${tier.tier === 'basic' ? 'popular' : ''}`}
                  >
                    {tier.tier === 'basic' && <div className="popular-badge">Most Popular</div>}
                    <div className="pricing-card-header">
                      <h2>{tier.name}</h2>
                      <div className="pricing-amount">
                        {tier.price === 0 ? (
                          <span className="price">Free</span>
                        ) : (
                          <>
                            <span className="currency">$</span>
                            <span className="price">{tier.price.toFixed(2)}</span>
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
                        disabled={isCurrent || isCheckoutLoading}
                      >
                        {isCurrent
                          ? 'Current Plan'
                          : isCheckoutLoading
                            ? 'Redirecting...'
                            : !isPaidTier
                              ? 'Get Started Free'
                              : 'Choose Plan'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="pricing-faq">
          <h2>Frequently Asked Questions</h2>
          <div className="faq-grid">
            <div className="faq-item">
              <h3>Can I change my plan later?</h3>
              <p>Yes. You can upgrade, downgrade, or cancel from the Stripe billing portal at any time.</p>
            </div>
            <div className="faq-item">
              <h3>What payment methods do you accept?</h3>
              <p>All major credit cards supported by Stripe.</p>
            </div>
            <div className="faq-item">
              <h3>Do paid plans include a free trial?</h3>
              <p>
                Not at this time. Paid subscriptions begin billing when activated. You can cancel at
                any time to stop future renewals.
              </p>
            </div>
            <div className="faq-item">
              <h3>What happens if I exceed my stock limit?</h3>
              <p>Upgrade to a higher tier to unlock additional stocks and limits.</p>
            </div>
          </div>
        </section>

        <section className="pricing-cta">
          <h2>Ready to learn from model-driven market context?</h2>
          <p>Use SeekingBeta as an educational decision-support reference in your research workflow.</p>
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
          <nav className="footer-links">
            <Link to="/terms">Terms</Link>
            <Link to="/privacy">Privacy</Link>
            <Link to="/refund-cancellation">Refund &amp; Cancellation</Link>
          </nav>
          <p className="footer-copyright">
            &copy; {new Date().getFullYear()} SeekingBeta. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default Pricing;
