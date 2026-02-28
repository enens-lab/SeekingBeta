import { useState, useEffect, ChangeEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import StockTooltip from '../components/StockTooltip';
import DashboardHeader from '../components/DashboardHeader';
import { oracle, OracleData } from '../api/client';

function MyOracle() {
  const { user, isVerified } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  const [oracleData, setOracleData] = useState<OracleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!isVerified) return;

    oracle.get()
      .then(setOracleData)
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Failed to load oracle data'))
      .finally(() => setLoading(false));
  }, [isVerified, toast]);

  const handleAddStock = async (ticker: string) => {
    if (!ticker) return;
    setSaving(true);

    try {
      const updated = await oracle.addToWatchlist(ticker);
      setOracleData(updated);
      toast.success(`Added ${ticker.toUpperCase()} to your watchlist`);
    } catch (err) {
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
      toast.success(`Removed ${ticker.toUpperCase()} from your watchlist`);
    } catch (err) {
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
        toast.error('You must have at least one timeframe selected');
        return;
      }
      newTimeframes = currentTimeframes.filter((tf) => tf !== timeframe);
    } else {
      newTimeframes = [...currentTimeframes, timeframe];
    }

    setSaving(true);

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
              Build and manage your SeekingBeta watchlist and preferred timeframes.
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
            {/* Timeframes Section */}
            <section className="oracle-section">
              <h2>Preferred Timeframes</h2>
              <p className="section-description">
                Select the temporal windows for your divinations
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
              {oracleData && oracleData.available_timeframes.length < 6 && (
                <p className="upgrade-hint">
                  <Link to="/pricing">Upgrade</Link> to unlock more timeframes
                </p>
              )}
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

              {user?.tier !== 'pro' && (
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
                onClick={() => navigate('/analysis')}
              >
                View Predictions
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default MyOracle;
