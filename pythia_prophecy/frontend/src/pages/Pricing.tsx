import { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { tiers as tiersApi, billing as billingApi, BillingStatus } from '../api/client';
import { useToast } from '../components/Toast';
import ThemeToggle from '../components/ThemeToggle';
import { trackEvent } from '../lib/analytics';

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
      .then((data) => {
        const typedData = data as unknown as TierData[];
        setTiers(typedData);
        trackEvent('pricing_plans_loaded', { plan_count: typedData.length });
      })
      .catch((err) => {
        console.error(err);
        trackEvent('pricing_plans_load_error');
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
      trackEvent('checkout_return', { status: 'success' });
      toast.success('Checkout complete. Subscription is syncing now.');
      billingApi.getStatus().then(setBillingStatus).catch(console.error);
    } else if (checkout === 'cancelled') {
      trackEvent('checkout_return', { status: 'cancelled' });
      toast.error('Checkout canceled. No changes were made.');
    } else if (checkout === 'portal_return') {
      trackEvent('checkout_return', { status: 'portal_return' });
      billingApi.getStatus().then(setBillingStatus).catch(console.error);
    }

    searchParams.delete('checkout');
    setSearchParams(searchParams, { replace: true });
  }, [searchParams, setSearchParams, toast]);

  const currentTier = billingStatus?.effective_tier || user?.tier || 'free';

  const handleSelectTier = async (tier: string) => {
    trackEvent('pricing_plan_selected', {
      tier,
      is_authenticated: isAuthenticated,
      is_verified: isVerified,
      current_tier: currentTier,
    });

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
      trackEvent('pricing_plan_invalid', { tier });
      return;
    }

    setCheckoutTier(tier);
    try {
      const priceValue = tier === 'pro' ? 19.99 : 9.99;
      trackEvent('begin_checkout', {
        currency: 'USD',
        value: priceValue,
        plan_tier: tier,
      });
      const result = await billingApi.createCheckoutSession(tier);
      trackEvent('checkout_redirect', { plan_tier: tier });
      window.location.assign(result.checkout_url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start checkout';
      trackEvent('checkout_error', { plan_tier: tier, reason: message.slice(0, 80) });
      toast.error(message);
    } finally {
      setCheckoutTier(null);
    }
  };

  const handleManageBilling = async () => {
    trackEvent('billing_portal_open_click', { source: 'pricing' });
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
            <span className="logo-text">SeekingBeta.AI</span>
          </Link>
          <nav className="nav">
            <Link
              to="/#predictions"
              className="nav-link"
              onClick={() => trackEvent('pricing_nav_click', { destination: 'predictions' })}
            >
              Live Boards
            </Link>
            <Link
              to="/#performance"
              className="nav-link"
              onClick={() => trackEvent('pricing_nav_click', { destination: 'performance' })}
            >
              Track Record
            </Link>
            <Link
              to="/#features"
              className="nav-link"
              onClick={() => trackEvent('pricing_nav_click', { destination: 'features' })}
            >
              How It Works
            </Link>
            <Link
              to="/pricing"
              className="nav-link"
              onClick={() => trackEvent('pricing_nav_click', { destination: 'pricing' })}
            >
              Pricing
            </Link>
          </nav>
          <div className="header-actions">
            <ThemeToggle />
            {isAuthenticated ? (
              <Link
                to="/dashboard"
                className="btn btn-primary"
                onClick={() => trackEvent('pricing_dashboard_click')}
              >
                Dashboard
              </Link>
            ) : (
              <>
                <Link
                  to="/login"
                  className="btn btn-ghost"
                  onClick={() => trackEvent('pricing_login_click')}
                >
                  Log In
                </Link>
                <Link
                  to="/signup"
                  className="btn btn-primary"
                  onClick={() => trackEvent('pricing_signup_click')}
                >
                  Start Free
                </Link>
              </>
            )}
          </div>
        </div>
      </header>

      <main>
        <section className="pricing-hero">
          <h1>Pricing built for how deeply you want to work the board</h1>
          <p>Start free, then unlock more coverage, history, exports, and higher analysis limits as you go.</p>
        </section>

        {isAuthenticated && billingStatus?.billing_enabled && (
          <section className="pricing-billing-status">
            <div className="pricing-billing-status-content">
              <div>
                <h3>Subscription</h3>
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
                {portalLoading ? 'Opening...' : 'Manage Subscription'}
              </button>
            </div>
          </section>
        )}

        <section className="pricing-cards">
          {loading ? (
            <div className="pricing-loading">
              <div className="spinner" />
              <p>Loading plan access...</p>
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
                    {tier.tier === 'basic' && <div className="popular-badge">Best Starting Point</div>}
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
                          <span>watchlist</span>
                        </div>
                        <div className="highlight">
                          <strong>{tier.timeframes.length}</strong>
                          <span>board horizon{tier.timeframes.length > 1 ? 's' : ''}</span>
                        </div>
                      </div>

                      <div className="pricing-timeframes">
                        <span className="label">Board Horizons:</span>
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
                              ? 'Start Free'
                              : 'Unlock Plan'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="pricing-faq">
          <h2>Pricing FAQ</h2>
          <div className="faq-grid">
            <div className="faq-item">
              <h3>Can I change plans anytime?</h3>
              <p>Yes. You can upgrade, downgrade, or cancel from the Stripe billing portal whenever you need.</p>
            </div>
            <div className="faq-item">
              <h3>How do payments work?</h3>
              <p>Billing runs through Stripe and supports the major cards Stripe accepts.</p>
            </div>
            <div className="faq-item">
              <h3>Is there a free trial on paid plans?</h3>
              <p>
                Not right now. Paid plans begin billing when activated, and you can cancel anytime to stop future renewals.
              </p>
            </div>
            <div className="faq-item">
              <h3>What happens if I hit a limit?</h3>
              <p>Upgrade to the next tier to unlock more tickers, deeper history, and higher request limits.</p>
            </div>
          </div>
        </section>

        <section className="pricing-cta">
          <h2>Ready to work from the board instead of guesswork?</h2>
          <p>Start free, test the product in your real workflow, and upgrade only when you want more depth.</p>
          <Link
            to="/signup"
            className="btn btn-primary btn-lg"
            onClick={() => trackEvent('pricing_cta_click', { cta: 'start_free' })}
          >
            Start Free
          </Link>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="logo-icon">β</span>
            <span className="logo-text">SeekingBeta.AI</span>
          </div>
          <p className="footer-disclaimer">
            SeekingBeta.AI publishes model-generated probabilities for research and education.
            No betting, no trade execution, and no guarantee of future results.
          </p>
          <nav className="footer-links">
            <Link to="/terms" onClick={() => trackEvent('footer_link_click', { destination: 'terms' })}>
              Terms
            </Link>
            <Link to="/privacy" onClick={() => trackEvent('footer_link_click', { destination: 'privacy' })}>
              Privacy
            </Link>
            <Link
              to="/refund-cancellation"
              onClick={() => trackEvent('footer_link_click', { destination: 'refund_cancellation' })}
            >
              Refund &amp; Cancellation
            </Link>
          </nav>
          <p className="footer-copyright">
            &copy; {new Date().getFullYear()} SeekingBeta.AI. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default Pricing;
