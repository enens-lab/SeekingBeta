import { useState, useEffect, ChangeEvent, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import StockTooltip from '../components/StockTooltip';
import DashboardHeader from '../components/DashboardHeader';
import { billing, oracle, OracleData, WatchlistInsight } from '../api/client';
import { trackEvent } from '../lib/analytics';

function formatPrice(value: number | null): string {
  if (value === null) return '—';
  return `$${value.toFixed(2)}`;
}

function formatPct(value: number | null): string {
  if (value === null) return '—';
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
}

function formatMetric(value: number | null, decimals = 2): string {
  if (value === null) return '—';
  return value.toFixed(decimals);
}

function formatVolume(value: number | null): string {
  if (value === null) return '—';
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return Math.round(value).toString();
}

function MyOracle() {
  const { user, isVerified } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  const [oracleData, setOracleData] = useState<OracleData | null>(null);
  const [watchlistInsights, setWatchlistInsights] = useState<WatchlistInsight[]>([]);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [finvizEnabled, setFinvizEnabled] = useState(false);
  const [fullscreenInsight, setFullscreenInsight] = useState<WatchlistInsight | null>(null);
  const [effectiveTier, setEffectiveTier] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});

  const watchlistKey = useMemo(
    () => (oracleData?.watchlist || []).map((t) => t.toUpperCase()).join(','),
    [oracleData?.watchlist]
  );

  useEffect(() => {
    if (!isVerified) return;

    Promise.all([
      oracle.get(),
      billing.getStatus().catch(() => null),
    ])
      .then(([oracleRes, billingStatus]) => {
        setOracleData(oracleRes);
        setEffectiveTier((billingStatus?.effective_tier || user?.tier || '').toLowerCase() || null);
      })
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Failed to load oracle data'))
      .finally(() => setLoading(false));
  }, [isVerified, toast, user?.tier]);

  useEffect(() => {
    const watchlist = oracleData?.watchlist || [];
    if (!isVerified) return;
    if (!watchlist.length) {
      setWatchlistInsights([]);
      setInsightsError(null);
      setInsightsLoading(false);
      return;
    }

    setInsightsLoading(true);
    setInsightsError(null);
    oracle
      .getWatchlistInsights(Math.min(watchlist.length, 20))
      .then((payload) => {
        setWatchlistInsights(payload.insights || []);
        setFinvizEnabled(!!payload.finviz_enabled);
      })
      .catch((err) => {
        setWatchlistInsights([]);
        setFinvizEnabled(false);
        setInsightsError(err instanceof Error ? err.message : 'Could not load watchlist insights');
      })
      .finally(() => setInsightsLoading(false));
  }, [isVerified, watchlistKey]);

  useEffect(() => {
    if (!fullscreenInsight) return;

    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setFullscreenInsight(null);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = originalOverflow;
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [fullscreenInsight]);

  const handleAddStock = async (ticker: string) => {
    if (!ticker) return;
    setSaving(true);

    try {
      const updated = await oracle.addToWatchlist(ticker);
      setOracleData(updated);
      trackEvent('watchlist_ticker_add', { ticker });
      trackEvent('ticker_interaction', { ticker, action: 'watchlist_add', surface: 'oracle' });
      toast.success(`Added ${ticker.toUpperCase()} to your watchlist`);
    } catch (err) {
      trackEvent('watchlist_ticker_add_error', { ticker });
      toast.error(err instanceof Error ? err.message : 'Failed to add stock');
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveStock = async (ticker: string) => {
    setSaving(true);

    try {
      const updated = await oracle.removeFromWatchlist(ticker);
      setOracleData(updated);
      trackEvent('watchlist_ticker_remove', { ticker });
      trackEvent('ticker_interaction', { ticker, action: 'watchlist_remove', surface: 'oracle' });
      toast.success(`Removed ${ticker.toUpperCase()} from your watchlist`);
    } catch (err) {
      trackEvent('watchlist_ticker_remove_error', { ticker });
      toast.error(err instanceof Error ? err.message : 'Failed to remove stock');
    } finally {
      setSaving(false);
    }
  };

  const handleTimeframeToggle = async (timeframe: string) => {
    if (!oracleData) return;

    const currentTimeframes = oracleData.timeframes;
    let newTimeframes: string[];

    if (currentTimeframes.includes(timeframe)) {
      if (currentTimeframes.length === 1) {
        toast.error('You must have at least one horizon selected');
        return;
      }
      newTimeframes = currentTimeframes.filter((tf) => tf !== timeframe);
    } else {
      newTimeframes = [...currentTimeframes, timeframe];
    }

    setSaving(true);
    trackEvent('watchlist_timeframe_toggle', { timeframe, selected: !currentTimeframes.includes(timeframe) });

    try {
      const updated = await oracle.updateTimeframes(newTimeframes);
      setOracleData(updated);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update timeframes');
    } finally {
      setSaving(false);
    }
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  // Filter categories by search query
  const getFilteredCategories = (): Record<string, string[]> => {
    if (!oracleData?.available_categories) return {};

    const filtered: Record<string, string[]> = {};
    for (const [category, stocks] of Object.entries(oracleData.available_categories)) {
      const matchingStocks = stocks.filter((s) =>
        s.toLowerCase().includes(searchQuery.toLowerCase())
      );
      if (matchingStocks.length > 0) {
        filtered[category] = matchingStocks;
      }
    }
    return filtered;
  };

  const filteredCategories = getFilteredCategories();
  const resolvedTier = (effectiveTier || user?.tier || '').toLowerCase();
  const hasFullUniverseAccess = (oracleData?.available_stocks?.length || 0) > 15;
  const isProPlan = resolvedTier === 'pro' || hasFullUniverseAccess;

  if (!isVerified) {
    return (
      <div className="oracle-page">
        <DashboardHeader showNav={false} />
        <main className="oracle-main">
          <div className="verify-prompt">
            <h1>Verify your email to access your watchlist</h1>
            <p>Please check your inbox and click the verification link.</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="oracle-page">
      <DashboardHeader activePage="oracle" />

      <main className="oracle-main">
        <div className="oracle-header">
          <div className="oracle-title-section">
            <h1>Watchlist</h1>
            <p className="oracle-subtitle">
              Build and manage your SeekingBeta.AI watchlist and preferred horizons.
            </p>
          </div>
        </div>

        {loading ? (
          <div className="oracle-loading">
            <div className="spinner" />
            <p>Loading watchlist...</p>
          </div>
        ) : (
          <div className="oracle-content">
            {/* Horizons Section */}
            <section className="oracle-section">
              <h2>Preferred Horizons</h2>
              <p className="section-description">
                Select which model horizons you want to see
              </p>
              <div className="timeframe-grid">
                {oracleData?.available_timeframes?.map((tf) => (
                  <button
                    key={tf}
                    className={`timeframe-chip ${oracleData.timeframes.includes(tf) ? 'active' : ''}`}
                    onClick={() => handleTimeframeToggle(tf)}
                    disabled={saving}
                  >
                    {tf}
                    {oracleData.timeframes.includes(tf) && (
                      <span className="check-icon">✓</span>
                    )}
                  </button>
                ))}
              </div>
            </section>

            {/* Watchlist Section */}
            <section className="oracle-section">
              <h2>Your Watchlist</h2>
              <p className="section-description">
                Tickers currently tracked ({oracleData?.watchlist?.length || 0} selected)
              </p>

              {oracleData && oracleData.watchlist.length > 0 ? (
                <div className="watchlist-grid">
                  {oracleData.watchlist.map((ticker) => (
                    <StockTooltip key={ticker} ticker={ticker}>
                      <div className="watchlist-item">
                        <span className="ticker-name">{ticker}</span>
                        <button
                          className="remove-btn"
                          onClick={() => handleRemoveStock(ticker)}
                          disabled={saving}
                          title="Remove from watchlist"
                        >
                          ×
                        </button>
                      </div>
                    </StockTooltip>
                  ))}
                </div>
              ) : (
                <div className="empty-watchlist">
                  <p>Your watchlist is empty. Add stocks below to start tracking.</p>
                </div>
              )}
            </section>

            <section className="oracle-section">
              <h2>Watchlist Insights</h2>
              <p className="section-description">
                {finvizEnabled
                  ? 'Chart and technical snapshots for your selected tickers.'
                  : 'Snapshot cards are available. Add Finviz API credentials for richer chart data.'}
              </p>

              {insightsLoading ? (
                <div className="oracle-loading">
                  <div className="spinner" />
                  <p>Loading watchlist insights...</p>
                </div>
              ) : insightsError ? (
                <div className="empty-watchlist">
                  <p>{insightsError}</p>
                </div>
              ) : watchlistInsights.length === 0 ? (
                <div className="empty-watchlist">
                  <p>Add tickers to your watchlist to see chart insights.</p>
                </div>
              ) : (
                <div className="watchlist-insights-grid">
                  {watchlistInsights.map((insight) => (
                    <article key={insight.ticker} className="watchlist-insight-card">
                      <div className="watchlist-insight-header">
                        <div>
                          <h3>{insight.ticker}</h3>
                          <p className={`watchlist-insight-change ${(insight.change_pct || 0) >= 0 ? 'up' : 'down'}`}>
                            {formatPct(insight.change_pct)}
                          </p>
                        </div>
                        <div className="watchlist-insight-header-actions">
                          <span className={`watchlist-insight-source source-${insight.source}`}>
                            {insight.source === 'finviz' ? 'Live' : 'Cached'}
                          </span>
                          <button
                            type="button"
                            className="watchlist-insight-fullscreen-btn"
                            aria-label={`Open ${insight.ticker} chart fullscreen`}
                            title="Fullscreen chart"
                            onClick={() => {
                              trackEvent('ticker_interaction', {
                                ticker: insight.ticker,
                                action: 'watchlist_chart_fullscreen_open',
                                surface: 'oracle',
                              });
                              setFullscreenInsight(insight);
                            }}
                          >
                            <svg
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              aria-hidden="true"
                            >
                              <polyline points="15 3 21 3 21 9" />
                              <polyline points="9 21 3 21 3 15" />
                              <line x1="21" y1="3" x2="14" y2="10" />
                              <line x1="3" y1="21" x2="10" y2="14" />
                            </svg>
                          </button>
                        </div>
                      </div>

                      <a
                        href={insight.chart_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="watchlist-insight-chart-link"
                        onClick={() =>
                          trackEvent('ticker_interaction', {
                            ticker: insight.ticker,
                            action: 'watchlist_chart_open',
                            surface: 'oracle',
                          })
                        }
                      >
                        <img
                          src={insight.chart_url}
                          alt={`${insight.ticker} chart`}
                          loading="lazy"
                          className="watchlist-insight-chart"
                          referrerPolicy="no-referrer"
                        />
                      </a>

                      <dl className="watchlist-insight-metrics">
                        <div>
                          <dt>Price</dt>
                          <dd>{formatPrice(insight.price)}</dd>
                        </div>
                        <div>
                          <dt>RSI</dt>
                          <dd>{formatMetric(insight.rsi)}</dd>
                        </div>
                        <div>
                          <dt>SMA 20</dt>
                          <dd>{formatPrice(insight.sma20)}</dd>
                        </div>
                        <div>
                          <dt>SMA 50</dt>
                          <dd>{formatPrice(insight.sma50)}</dd>
                        </div>
                        <div>
                          <dt>Volume</dt>
                          <dd>{formatVolume(insight.volume)}</dd>
                        </div>
                        <div>
                          <dt>Rel Vol</dt>
                          <dd>{formatMetric(insight.rel_volume)}</dd>
                        </div>
                      </dl>

                      {insight.summary && <p className="watchlist-insight-summary">{insight.summary}</p>}
                    </article>
                  ))}
                </div>
              )}
            </section>

            {/* Add Stocks Section - Now with Categories */}
            <section className="oracle-section">
              <h2>Add to Watchlist</h2>
              <p className="section-description">
                Choose from {oracleData?.available_stocks?.length || 0} stocks available in your tier
              </p>

              {/* Search */}
              <div className="stock-search-row">
                <input
                  type="text"
                  placeholder="Search stocks..."
                  value={searchQuery}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
                  className="stock-search-input"
                />
              </div>

              {/* Categories */}
              <div className="stock-categories">
                {Object.entries(filteredCategories).map(([category, stocks]) => (
                  <div key={category} className="category">
                    <div
                      className="category-header"
                      onClick={() => toggleCategory(category)}
                    >
                      <span>{category}</span>
                      <span className="category-count">{stocks.length}</span>
                    </div>
                    <div
                      className={`category-stocks ${
                        expandedCategories[category] || searchQuery ? '' : 'collapsed'
                      }`}
                    >
                      {stocks.map((ticker) => {
                        const isInWatchlist = oracleData?.watchlist?.includes(ticker);
                        return (
                          <StockTooltip key={ticker} ticker={ticker}>
                            <button
                              className={`stock-chip ${isInWatchlist ? 'in-watchlist' : ''}`}
                              onClick={() => !isInWatchlist && handleAddStock(ticker)}
                              disabled={saving || isInWatchlist}
                            >
                              {ticker}
                              {isInWatchlist && <span className="added-icon">✓</span>}
                            </button>
                          </StockTooltip>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>

              {!isProPlan && (
                <p className="upgrade-hint">
                  <Link to="/pricing">Upgrade to Pro</Link> for access to the full universe
                </p>
              )}
            </section>

            {/* Quick Actions */}
            <div className="oracle-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  trackEvent('oracle_view_predictions_click');
                  navigate('/analysis');
                }}
              >
                View Predictions
              </button>
            </div>
          </div>
        )}
      </main>

      {fullscreenInsight && (
        <div
          className="chart-fullscreen-overlay"
          role="dialog"
          aria-modal="true"
          aria-label={`${fullscreenInsight.ticker} chart fullscreen`}
          onClick={() => setFullscreenInsight(null)}
        >
          <div className="chart-fullscreen-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="chart-fullscreen-toolbar">
              <div className="chart-fullscreen-title">
                <strong>{fullscreenInsight.ticker}</strong>
                <span>{formatPct(fullscreenInsight.change_pct)}</span>
              </div>
              <button
                type="button"
                className="chart-fullscreen-close"
                onClick={() => {
                  trackEvent('ticker_interaction', {
                    ticker: fullscreenInsight.ticker,
                    action: 'watchlist_chart_fullscreen_close',
                    surface: 'oracle',
                  });
                  setFullscreenInsight(null);
                }}
                aria-label="Close fullscreen chart"
              >
                ×
              </button>
            </div>
            <div className="chart-fullscreen-image-wrap">
              <img
                src={fullscreenInsight.chart_url}
                alt={`${fullscreenInsight.ticker} fullscreen chart`}
                className="chart-fullscreen-image"
                referrerPolicy="no-referrer"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MyOracle;
