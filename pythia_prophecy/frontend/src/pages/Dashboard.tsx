import { useState, useEffect, ChangeEvent, ReactNode, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { oracle, predictions, OracleData, Prediction } from '../api/client';
import { useToast } from '../components/Toast';
import PredictionCard from '../components/PredictionCard';
import CompanyDetail from '../components/CompanyDetail';
import DashboardHeader from '../components/DashboardHeader';
import { trackEvent } from '../lib/analytics';

type ViewMode = 'oracle' | 'universe';

function isBullishSignal(signal?: string | null): boolean {
  const normalized = (signal || '').toLowerCase();
  return normalized === 'buy' || normalized === 'strong_buy';
}

function isBearishSignal(signal?: string | null): boolean {
  const normalized = (signal || '').toLowerCase();
  return normalized === 'sell' || normalized === 'avoid';
}

function Dashboard() {
  const { user, isVerified } = useAuth();
  const toast = useToast();
  const [oracleData, setOracleData] = useState<OracleData | null>(null);
  const [predictionData, setPredictionData] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [watchlistSaving, setWatchlistSaving] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('5d');
  const [viewMode, setViewMode] = useState<ViewMode>('oracle');
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [pageFetchError, setPageFetchError] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const pageSize = 12;

  useEffect(() => {
    if (!isVerified) return;

    oracle.get()
      .then((data) => {
        setOracleData(data);
        if (data.timeframes?.length > 0) {
          setSelectedTimeframe(data.timeframes[0]);
        }
        // If user has no watchlist, default to universe view
        if (data.watchlist?.length === 0) {
          setViewMode('universe');
        }
      })
      .catch((err) => toast.error(err instanceof Error ? err.message : 'Failed to load watchlist'))
      .finally(() => setLoading(false));
  }, [isVerified, toast]);

  // Reset to page 1 when view mode, timeframe, or search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [viewMode, selectedTimeframe, searchQuery]);

  const normalizedSearch = searchQuery.trim().toLowerCase();

  const baseTickers = useMemo(
    () =>
      viewMode === 'oracle' && oracleData?.watchlist?.length
        ? oracleData.watchlist
        : oracleData?.available_stocks || [],
    [viewMode, oracleData]
  );

  const filteredTickers = useMemo(
    () =>
      normalizedSearch
        ? baseTickers.filter((t) => t.toLowerCase().includes(normalizedSearch))
        : baseTickers,
    [baseTickers, normalizedSearch]
  );

  const totalPages = Math.max(1, Math.ceil(filteredTickers.length / pageSize));
  const safeCurrentPage = Math.min(currentPage, totalPages);
  const pageTickers = useMemo(() => {
    const start = (safeCurrentPage - 1) * pageSize;
    return filteredTickers.slice(start, start + pageSize);
  }, [filteredTickers, safeCurrentPage]);

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [currentPage, totalPages]);

  const removeWatchlistTicker = async (ticker: string) => {
    if (!oracleData || watchlistSaving) return;
    setWatchlistSaving(true);
    try {
      const updated = await oracle.removeFromWatchlist(ticker);
      setOracleData(updated);
      setPredictionData((prev) => prev.filter((p) => p.ticker !== ticker));
      if (updated.watchlist.length === 0) {
        setViewMode('universe');
      }
      trackEvent('watchlist_ticker_remove', { ticker, source: 'dashboard' });
      trackEvent('ticker_interaction', { ticker, action: 'watchlist_remove', surface: 'dashboard' });
      toast.success(`Removed ${ticker} from watchlist`);
    } catch (err) {
      trackEvent('watchlist_ticker_remove_error', { ticker, source: 'dashboard' });
      toast.error(err instanceof Error ? err.message : 'Failed to remove ticker');
    } finally {
      setWatchlistSaving(false);
    }
  };

  const clearWatchlist = async () => {
    if (!oracleData || oracleData.watchlist.length === 0 || watchlistSaving) return;
    if (!window.confirm('Clear every ticker from your watchlist board?')) return;
    setWatchlistSaving(true);
    try {
      const updated = await oracle.updateWatchlist([]);
      setOracleData(updated);
      setPredictionData([]);
      setViewMode('universe');
      trackEvent('watchlist_clear', { source: 'dashboard' });
      toast.success('Watchlist cleared');
    } catch (err) {
      trackEvent('watchlist_clear_error', { source: 'dashboard' });
      toast.error(err instanceof Error ? err.message : 'Failed to clear watchlist');
    } finally {
      setWatchlistSaving(false);
    }
  };

  useEffect(() => {
    if (!oracleData || !isVerified) return;

    if (pageTickers.length === 0) {
      setPredictionData([]);
      setPageFetchError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setPageFetchError(null);

    Promise.allSettled(
      pageTickers.map((ticker) =>
        predictions.get(ticker, selectedTimeframe)
      )
    )
      .then((results) => {
        const successful: Prediction[] = [];
        const failedTickers: string[] = [];
        results.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            successful.push(result.value);
          } else {
            failedTickers.push(pageTickers[index]);
          }
        });
        setPredictionData(successful);
        if (failedTickers.length > 0) {
          trackEvent('dashboard_prediction_partial', {
            failed_count: failedTickers.length,
            success_count: successful.length,
            timeframe: selectedTimeframe,
          });
          if (successful.length === 0) {
            setPageFetchError(
              `No model outputs available for ${failedTickers.slice(0, 3).join(', ')}${failedTickers.length > 3 ? '...' : ''}.`
            );
          } else {
            setPageFetchError(
              `Partial results: ${failedTickers.slice(0, 3).join(', ')}${failedTickers.length > 3 ? '...' : ''} unavailable.`
            );
          }
        }
        if (successful.length > 0) {
          trackEvent('dashboard_predictions_loaded', {
            count: successful.length,
            timeframe: selectedTimeframe,
            view_mode: viewMode,
          });
        }
      })
      .finally(() => setLoading(false));
  }, [oracleData, selectedTimeframe, isVerified, pageTickers]);

  if (!isVerified) {
    return (
      <div className="dashboard-page">
        <DashboardHeader showNav={false} />

        <main className="dashboard-main">
          <div className="verify-prompt">
            <div className="verify-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                <polyline points="22,6 12,13 2,6" />
              </svg>
            </div>
            <h1>Verify your email</h1>
            <p>
              Please check your inbox and click the verification link to open your dashboard.
            </p>
            <p className="verify-email">
              Sent to: <strong>{user?.email}</strong>
            </p>
          </div>
        </main>
      </div>
    );
  }

  const hasWatchlist = (oracleData?.watchlist?.length ?? 0) > 0;

  return (
    <div className="dashboard-page">
      <DashboardHeader activePage="dashboard" />

      <main className="dashboard-main">
        <div className="dashboard-controls">
          <div className="dashboard-title">
            <h1>
              {viewMode === 'oracle' && hasWatchlist ? (
                'My Board'
              ) : (
                'Full Stock Board'
              )}
            </h1>
            {hasWatchlist && oracleData && (
              <div className="view-toggle">
                <button
                  className={`toggle-btn ${viewMode === 'oracle' ? 'active' : ''}`}
                  onClick={() => {
                    trackEvent('dashboard_view_mode_change', { mode: 'oracle' });
                    setViewMode('oracle');
                  }}
                >
                  My Board ({oracleData.watchlist.length})
                </button>
                <button
                  className={`toggle-btn ${viewMode === 'universe' ? 'active' : ''}`}
                  onClick={() => {
                    trackEvent('dashboard_view_mode_change', { mode: 'universe' });
                    setViewMode('universe');
                  }}
                >
                  Full Universe ({oracleData.available_stocks.length})
                </button>
              </div>
            )}
          </div>

          {viewMode === 'oracle' && hasWatchlist && oracleData && (
            <div className="dashboard-watchlist-manager">
              <div className="watchlist-manager-header">
                <span>Tracked tickers</span>
                <button
                  type="button"
                  className="watchlist-clear-btn"
                  onClick={clearWatchlist}
                  disabled={watchlistSaving}
                >
                  Clear board
                </button>
              </div>
              <div className="watchlist-pill-list">
                {oracleData.watchlist.map((ticker) => (
                  <button
                    key={ticker}
                    type="button"
                    className="watchlist-pill"
                    onClick={() => removeWatchlistTicker(ticker)}
                    disabled={watchlistSaving}
                    title={`Remove ${ticker}`}
                  >
                    <span>{ticker}</span>
                    <span className="remove-mark">×</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="dashboard-filters">
            <div className="dashboard-search">
              <input
                type="text"
                placeholder="Search tickers..."
                value={searchQuery}
                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                  const value = e.target.value;
                  setSearchQuery(value);
                  if (value.length === 2 || value.length === 4 || value.length === 6) {
                    trackEvent('dashboard_search_update', { query_length: value.length });
                  }
                }}
                className="dashboard-search-input"
              />
              {searchQuery && (
                <button
                  className="search-clear"
                  onClick={() => setSearchQuery('')}
                >
                  ×
                </button>
              )}
            </div>
            <div className="timeframe-selector">
              <label>Horizon:</label>
              <select
                value={selectedTimeframe}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => {
                  trackEvent('dashboard_timeframe_change', { timeframe: e.target.value });
                  setSelectedTimeframe(e.target.value);
                }}
              >
                {oracleData?.available_timeframes?.map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="dashboard-stats">
          <div className="stat-card">
            <span className="stat-label">
              {viewMode === 'oracle' ? 'Watchlist' : 'Available'}
            </span>
            <span className="stat-value">
              {viewMode === 'oracle'
                ? oracleData?.watchlist?.length || 0
                : oracleData?.available_stocks?.length || 0}
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Board Horizons</span>
            <span className="stat-value">{oracleData?.available_timeframes?.length || 0}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Bullish</span>
            <span className="stat-value signal-buy">
              {predictionData.filter((p) => isBullishSignal(p.signal)).length}
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Bearish</span>
            <span className="stat-value signal-sell">
              {predictionData.filter((p) => isBearishSignal(p.signal)).length}
            </span>
          </div>
        </div>

        {!hasWatchlist && (
          <div className="empty-oracle-banner">
            <div>
              <h3>Build your board</h3>
              <p>Add the tickers you care about most so your dashboard opens to a focused board instead of the full universe.</p>
            </div>
            <Link to="/oracle" className="btn btn-primary">
              Set Up Watchlist
            </Link>
          </div>
        )}

        {loading ? (
          <div className="dashboard-loading">
            <div className="spinner" />
            <p>Loading board snapshots...</p>
          </div>
        ) : filteredTickers.length === 0 ? (
          <div className="dashboard-no-results">
            <p>No tickers match "{searchQuery}"</p>
          </div>
        ) : (
          <>
            <div className="predictions-grid">
              {predictionData.map((prediction) => (
                <PredictionCard
                  key={prediction.ticker}
                  prediction={prediction}
                  onClick={setSelectedTicker}
                />
              ))}
            </div>

            {pageFetchError && (
              <div className="dashboard-no-results">
                <p>{pageFetchError}</p>
              </div>
            )}

            {filteredTickers.length > 0 && totalPages > 1 && (
              <div className="pagination">
                <button
                  className="pagination-btn"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={safeCurrentPage === 1}
                >
                  ← Prev
                </button>
                <div className="pagination-pages">
                  {Array.from({ length: totalPages }, (_, i) => i + 1)
                    .filter((page) => {
                      // Show first, last, and pages near current
                      return page === 1 || page === totalPages ||
                        Math.abs(page - safeCurrentPage) <= 2;
                    })
                    .reduce<ReactNode[]>((acc, page, idx, arr) => {
                      // Insert ellipsis between gaps
                      if (idx > 0 && page - arr[idx - 1] > 1) {
                        acc.push(<span key={`ellipsis-${page}`} className="pagination-ellipsis">...</span>);
                      }
                      acc.push(
                        <button
                          key={page}
                          className={`pagination-num ${page === safeCurrentPage ? 'active' : ''}`}
                          onClick={() => setCurrentPage(page)}
                        >
                          {page}
                        </button>
                      );
                      return acc;
                    }, [])}
                </div>
                <button
                  className="pagination-btn"
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={safeCurrentPage === totalPages}
                >
                  Next →
                </button>
              </div>
            )}
          </>
        )}

        {oracleData && oracleData.available_stocks.length > 0 && oracleData.available_stocks.length <= 15 && (
          <div className="upgrade-banner">
            <p>
              You&apos;re viewing {oracleData.available_stocks.length} names from the full stock universe.
            </p>
            <Link to="/pricing" className="btn btn-primary">
              Unlock Full Universe
            </Link>
          </div>
        )}
      </main>

      {selectedTicker && (
        <CompanyDetail
          ticker={selectedTicker}
          onClose={() => setSelectedTicker(null)}
        />
      )}
    </div>
  );
}

export default Dashboard;
