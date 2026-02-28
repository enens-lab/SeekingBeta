import { useState, useEffect, ChangeEvent, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { oracle, predictions, OracleData, Prediction } from '../api/client';
import { useToast } from '../components/Toast';
import PredictionCard from '../components/PredictionCard';
import CompanyDetail from '../components/CompanyDetail';
import DashboardHeader from '../components/DashboardHeader';

type ViewMode = 'oracle' | 'universe';

function Dashboard() {
  const { user, isVerified } = useAuth();
  const toast = useToast();
  const [oracleData, setOracleData] = useState<OracleData | null>(null);
  const [predictionData, setPredictionData] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [watchlistSaving, setWatchlistSaving] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1d');
  const [viewMode, setViewMode] = useState<ViewMode>('oracle');
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
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

  const baseTickers = viewMode === 'oracle' && oracleData?.watchlist?.length
    ? oracleData.watchlist
    : oracleData?.available_stocks || [];

  const filteredTickers = searchQuery
    ? baseTickers.filter((t) => t.toLowerCase().includes(searchQuery.toLowerCase()))
    : baseTickers;

  const totalPages = Math.ceil(filteredTickers.length / pageSize);

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
      toast.success(`Removed ${ticker} from watchlist`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to remove ticker');
    } finally {
      setWatchlistSaving(false);
    }
  };

  const clearWatchlist = async () => {
    if (!oracleData || oracleData.watchlist.length === 0 || watchlistSaving) return;
    if (!window.confirm('Clear all tickers from your watchlist?')) return;
    setWatchlistSaving(true);
    try {
      const updated = await oracle.updateWatchlist([]);
      setOracleData(updated);
      setPredictionData([]);
      setViewMode('universe');
      toast.success('Watchlist cleared');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to clear watchlist');
    } finally {
      setWatchlistSaving(false);
    }
  };

  useEffect(() => {
    if (!oracleData || !isVerified) return;

    setLoading(true);

    const start = (currentPage - 1) * pageSize;
    const pageTickers = filteredTickers.slice(start, start + pageSize);

    Promise.allSettled(
      pageTickers.map((ticker) =>
        predictions.get(ticker, selectedTimeframe)
      )
    )
      .then((results) => {
        const successful = results
          .filter((r): r is PromiseFulfilledResult<Prediction> => r.status === 'fulfilled')
          .map((r) => r.value);
        setPredictionData(successful);
      })
      .finally(() => setLoading(false));
  }, [oracleData, selectedTimeframe, viewMode, isVerified, currentPage, filteredTickers]);

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
              Please check your inbox and click the verification link to access your dashboard.
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
                'My Watchlist'
              ) : (
                'All Predictions'
              )}
            </h1>
            {hasWatchlist && oracleData && (
              <div className="view-toggle">
                <button
                  className={`toggle-btn ${viewMode === 'oracle' ? 'active' : ''}`}
                  onClick={() => setViewMode('oracle')}
                >
                  My Watchlist ({oracleData.watchlist.length})
                </button>
                <button
                  className={`toggle-btn ${viewMode === 'universe' ? 'active' : ''}`}
                  onClick={() => setViewMode('universe')}
                >
                  All Stocks ({oracleData.available_stocks.length})
                </button>
              </div>
            )}
          </div>

          {viewMode === 'oracle' && hasWatchlist && oracleData && (
            <div className="dashboard-watchlist-manager">
              <div className="watchlist-manager-header">
                <span>Manage watchlist</span>
                <button
                  type="button"
                  className="watchlist-clear-btn"
                  onClick={clearWatchlist}
                  disabled={watchlistSaving}
                >
                  Clear all
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
                placeholder="Search stocks..."
                value={searchQuery}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
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
              <label>Timeframe:</label>
              <select
                value={selectedTimeframe}
                onChange={(e: ChangeEvent<HTMLSelectElement>) => setSelectedTimeframe(e.target.value)}
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
            <span className="stat-label">Timeframes</span>
            <span className="stat-value">{oracleData?.available_timeframes?.length || 0}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Buy Signals</span>
            <span className="stat-value signal-buy">
              {predictionData.filter((p) => p.signal === 'buy').length}
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Sell Signals</span>
            <span className="stat-value signal-sell">
              {predictionData.filter((p) => p.signal === 'sell').length}
            </span>
          </div>
        </div>

        {!hasWatchlist && (
          <div className="empty-oracle-banner">
            <div>
              <h3>Set up your Watchlist</h3>
              <p>Create a personalized watchlist to focus on the stocks that matter to you.</p>
            </div>
            <Link to="/oracle" className="btn btn-primary">
              Configure Watchlist
            </Link>
          </div>
        )}

        {loading ? (
          <div className="dashboard-loading">
            <div className="spinner" />
            <p>Loading predictions...</p>
          </div>
        ) : filteredTickers.length === 0 ? (
          <div className="dashboard-no-results">
            <p>No stocks matching "{searchQuery}"</p>
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

            {totalPages > 1 && (
              <div className="pagination">
                <button
                  className="pagination-btn"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  ← Prev
                </button>
                <div className="pagination-pages">
                  {Array.from({ length: totalPages }, (_, i) => i + 1)
                    .filter((page) => {
                      // Show first, last, and pages near current
                      return page === 1 || page === totalPages ||
                        Math.abs(page - currentPage) <= 2;
                    })
                    .reduce<ReactNode[]>((acc, page, idx, arr) => {
                      // Insert ellipsis between gaps
                      if (idx > 0 && page - arr[idx - 1] > 1) {
                        acc.push(<span key={`ellipsis-${page}`} className="pagination-ellipsis">...</span>);
                      }
                      acc.push(
                        <button
                          key={page}
                          className={`pagination-num ${page === currentPage ? 'active' : ''}`}
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
                  disabled={currentPage === totalPages}
                >
                  Next →
                </button>
              </div>
            )}
          </>
        )}

        {oracleData && oracleData.available_stocks.length < 10 && (
          <div className="upgrade-banner">
            <p>
              You have access to {oracleData.available_stocks.length} of the full universe.
            </p>
            <Link to="/pricing" className="btn btn-primary">
              Upgrade for full access
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
