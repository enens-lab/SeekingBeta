import { useState, useEffect, useRef, ChangeEvent, ReactNode, MouseEvent } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { analysis, UserFeatures, AnalysisResponse } from '../api/client';
import DashboardHeader from '../components/DashboardHeader';
import type { Chart as ChartJS } from 'chart.js';

interface ModelOption {
  value: string;
  label: string;
}

interface TaskOption {
  value: string;
  label: string;
}

interface PeriodOption {
  value: string;
  label: string;
  days: number;
}

interface HorizonOption {
  value: string;
  label: string;
}

function Analysis() {
  const { user, isVerified } = useAuth();
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<ChartJS | null>(null);

  // State
  const [userFeatures, setUserFeatures] = useState<UserFeatures | null>(null);
  const [stockCategories, setStockCategories] = useState<Record<string, string[]>>({});
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableTasks, setAvailableTasks] = useState<string[]>([]);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [model, setModel] = useState('gradient_boosting');
  const [task, setTask] = useState('classifier');
  const [period, setPeriod] = useState('1M');
  const [horizon, setHorizon] = useState('1d');

  // All possible options
  const modelIcons: Record<string, ReactNode> = {
    gradient_boosting: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
    linear_regression: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="3" y1="20" x2="21" y2="4" />
        <circle cx="6" cy="17" r="1.5" fill="currentColor" stroke="none" />
        <circle cx="10" cy="14" r="1.5" fill="currentColor" stroke="none" />
        <circle cx="14" cy="10" r="1.5" fill="currentColor" stroke="none" />
        <circle cx="18" cy="7" r="1.5" fill="currentColor" stroke="none" />
      </svg>
    ),
    random_forest: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22v-7" />
        <path d="M12 15l-4-4 2 0-2-3 3 0-1-3 2 0 0-2 0 2 2 0-1 3 3 0-2 3 2 0z" />
      </svg>
    ),
    lstm: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="6" cy="6" r="2" />
        <circle cx="6" cy="18" r="2" />
        <circle cx="18" cy="12" r="2" />
        <line x1="8" y1="6" x2="16" y2="12" />
        <line x1="8" y1="18" x2="16" y2="12" />
      </svg>
    ),
  };

  const allModels: ModelOption[] = [
    { value: 'gradient_boosting', label: 'Gradient Boosting' },
    { value: 'linear_regression', label: 'Linear Regression' },
    { value: 'random_forest', label: 'Random Forest' },
    { value: 'lstm', label: 'LSTM' },
  ];

  const allTasks: TaskOption[] = [
    { value: 'classifier', label: 'Classification (Buy/Hold/Sell)' },
    { value: 'regressor', label: 'Regression (Return %)' },
  ];

  const allPeriods: PeriodOption[] = [
    { value: '1M', label: '1 Month', days: 30 },
    { value: '3M', label: '3 Months', days: 90 },
    { value: '6M', label: '6 Months', days: 180 },
    { value: '1Y', label: '1 Year', days: 365 },
  ];

  const allHorizons: HorizonOption[] = [
    { value: '1d', label: '1 Day' },
    { value: '1w', label: '1 Week' },
  ];

  useEffect(() => {
    if (!isVerified) return;

    const loadData = async () => {
      try {
        const [featuresRes, universeRes, modelsRes] = await Promise.all([
          analysis.getUserFeatures(),
          analysis.getUniverse(),
          analysis.getModels(),
        ]);

        setUserFeatures(featuresRes);
        setStockCategories(universeRes.categories);
        setAvailableModels(modelsRes.models);
        setAvailableTasks(modelsRes.tasks);
      } catch (err) {
        console.error('Failed to load analysis data:', err);
        setError('Failed to load analysis data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [isVerified]);

  // Render chart when results change
  useEffect(() => {
    if (!results || !chartRef.current) return;

    // Dynamically import Chart.js
    import('chart.js/auto').then(({ default: Chart }) => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current?.getContext('2d');
      if (!ctx) return;

      const labels = results.results.map((r) => r.ticker);
      const data = results.results.map((r) => (r.prob_up !== null ? r.prob_up * 100 : 0));
      const colors = results.results.map((r) => {
        if (r.signal === 'buy') return 'rgba(34, 197, 94, 0.8)';
        if (r.signal === 'sell') return 'rgba(239, 68, 68, 0.8)';
        return 'rgba(107, 114, 128, 0.8)';
      });

      chartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Probability Up (%)',
              data,
              backgroundColor: colors,
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: { display: true, text: 'Probability (%)' },
            },
          },
        },
      });
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [results]);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  const toggleStock = (ticker: string) => {
    if (selectedStocks.includes(ticker)) {
      setSelectedStocks((prev) => prev.filter((s) => s !== ticker));
    } else {
      const maxStocks = userFeatures?.limits?.max_stocks_per_request || 5;
      if (selectedStocks.length >= maxStocks) {
        alert(`Maximum ${maxStocks} stocks for your tier`);
        return;
      }
      setSelectedStocks((prev) => [...prev, ticker]);
    }
  };

  const removeStock = (ticker: string) => {
    setSelectedStocks((prev) => prev.filter((s) => s !== ticker));
  };

  const runAnalysis = async (e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    if (selectedStocks.length === 0) {
      alert('Please select at least one stock');
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      const data = await analysis.run({
        tickers: selectedStocks,
        model,
        task,
        period,
        horizon,
      });
      setResults(data);

      // Refresh user features to update rate limit
      const features = await analysis.getUserFeatures();
      setUserFeatures(features);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const exportCSV = () => {
    if (!results) return;

    const headers = ['Ticker', 'Last Close', 'Prob. Up', 'Signal', 'Pred. Return'];
    const rows = results.results.map((r) => [
      r.ticker,
      r.last_close ? `$${r.last_close.toFixed(2)}` : '-',
      r.prob_up !== null ? `${(r.prob_up * 100).toFixed(1)}%` : '-',
      r.signal || '-',
      r.predicted_return !== null ? `${(r.predicted_return * 100).toFixed(2)}%` : '-',
    ]);

    const csv = [headers, ...rows].map((row) => row.map((c) => `"${c}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pythia_analysis_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter stocks by search
  const filteredCategories: Record<string, string[]> = {};
  for (const [category, stocks] of Object.entries(stockCategories)) {
    const filtered = stocks.filter((s) => s.toLowerCase().includes(searchQuery.toLowerCase()));
    if (filtered.length > 0) {
      filteredCategories[category] = filtered;
    }
  }

  if (!isVerified) {
    return (
      <div className="dashboard-page">
        <DashboardHeader showNav={false} />

        <main className="dashboard-main">
          <div className="verify-prompt">
            <div className="verify-icon">
              <svg
                width="64"
                height="64"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                <polyline points="22,6 12,13 2,6" />
              </svg>
            </div>
            <h1>Verify your email</h1>
            <p>Please check your inbox and click the verification link to access Analysis.</p>
            <p className="verify-email">
              Sent to: <strong>{user?.email}</strong>
            </p>
          </div>
        </main>
      </div>
    );
  }

  const maxStocks = userFeatures?.limits?.max_stocks_per_request || 5;
  const maxDays = userFeatures?.limits?.max_historical_days || 30;
  const requestsUsed = userFeatures?.limits?.requests_used || 0;
  const dailyLimit = userFeatures?.limits?.daily_requests;
  const showRateBanner = dailyLimit && requestsUsed >= dailyLimit * 0.8;

  return (
    <div className="dashboard-page">
      <DashboardHeader activePage="analysis" />

      <main className="analysis-main">
        {loading ? (
          <div className="dashboard-loading">
            <div className="spinner" />
            <p>Loading analysis tools...</p>
          </div>
        ) : (
          <>
            <div className="analyze-grid">
              {/* Stock Selection Panel */}
              <section className="card stock-panel">
                <h3>Select Stocks</h3>
                <div className="stock-search-row">
                  <input
                    type="text"
                    placeholder="Search stocks..."
                    value={searchQuery}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
                  />
                  <span className="stock-limit">
                    {selectedStocks.length}/{maxStocks} selected
                  </span>
                </div>

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
                        {stocks.map((ticker) => (
                          <label key={ticker} className="stock-checkbox">
                            <input
                              type="checkbox"
                              checked={selectedStocks.includes(ticker)}
                              onChange={() => toggleStock(ticker)}
                            />
                            {ticker}
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="selected-stocks">
                  <h4>Selected Stocks</h4>
                  <div className="selected-list">
                    {selectedStocks.length === 0 ? (
                      <span className="placeholder">No stocks selected</span>
                    ) : (
                      selectedStocks.map((ticker) => (
                        <span key={ticker} className="stock-chip">
                          {ticker}
                          <span className="remove" onClick={() => removeStock(ticker)}>
                            &times;
                          </span>
                        </span>
                      ))
                    )}
                  </div>
                </div>
              </section>

              {/* Configuration Panel */}
              <section className="card config-panel">
                <h3>Analysis Settings</h3>
                <form className="config-form">
                  <div className="config-row">
                    <label>Model</label>
                    <div className="model-picker">
                      {allModels.map((m) => {
                        const isDisabled = !availableModels.includes(m.value);
                        return (
                          <button
                            type="button"
                            key={m.value}
                            className={`model-option${model === m.value ? ' active' : ''}${isDisabled ? ' disabled' : ''}`}
                            onClick={() => !isDisabled && setModel(m.value)}
                            disabled={isDisabled}
                            title={m.label + (isDisabled ? ' (PRO)' : '')}
                          >
                            <span className="model-icon">{modelIcons[m.value]}</span>
                            <span className="model-label">
                              {m.label}
                              {isDisabled && <span className="pro-badge">PRO</span>}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div className="config-row">
                    <label htmlFor="selectTask">Task</label>
                    <select id="selectTask" name="selectTask" value={task} onChange={(e: ChangeEvent<HTMLSelectElement>) => setTask(e.target.value)}>
                      {allTasks.map((t) => (
                        <option
                          key={t.value}
                          value={t.value}
                          disabled={!availableTasks.includes(t.value)}
                        >
                          {t.label}
                          {!availableTasks.includes(t.value) ? ' (PRO)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="config-row">
                    <label htmlFor="selectTime">Time Period</label>
                    <select id="selectTime" name="selectTime" value={period} onChange={(e: ChangeEvent<HTMLSelectElement>) => setPeriod(e.target.value)}>
                      {allPeriods.map((p) => (
                        <option key={p.value} value={p.value} disabled={p.days > maxDays}>
                          {p.label}
                          {p.days > maxDays ? ' (PRO)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="config-row">
                    <label htmlFor="selectHorizon">Horizon</label>
                    <select id="selectHorizon" name="selectHorizon" value={horizon} onChange={(e: ChangeEvent<HTMLSelectElement>) => setHorizon(e.target.value)}>
                      {allHorizons.map((h) => (
                        <option key={h.value} value={h.value}>
                          {h.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    onClick={runAnalysis}
                    disabled={analyzing || selectedStocks.length === 0}
                  >
                    {analyzing ? 'Analyzing...' : 'Run Analysis'}
                  </button>
                </form>
              </section>
            </div>

            {/* Results Panel */}
            <section className="card results-panel">
              <div className="results-header">
                <h3>Analysis Results</h3>
                {results && userFeatures?.features?.export_csv && (
                  <button className="btn btn-outline" onClick={exportCSV}>
                    Export CSV
                  </button>
                )}
              </div>

              {error && <div className="form-error">{error}</div>}

              {!results ? (
                <div className="results-empty">
                  <p>Select stocks and run analysis to see results</p>
                </div>
              ) : (
                <div className="results-content">
                  <div className="results-chart">
                    <canvas ref={chartRef}></canvas>
                  </div>
                  <div className="table-wrap">
                    <table className="results-table">
                      <thead>
                        <tr>
                          <th>Ticker</th>
                          <th>Last Close</th>
                          <th>Prob. Up</th>
                          <th>Signal</th>
                          <th>Pred. Return</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.results.map((r) => (
                          <tr key={r.ticker}>
                            <td>
                              <strong>{r.ticker}</strong>
                            </td>
                            <td>{r.last_close ? `$${r.last_close.toFixed(2)}` : '-'}</td>
                            <td>
                              {r.prob_up !== null ? `${(r.prob_up * 100).toFixed(1)}%` : '-'}
                            </td>
                            <td className={`signal-${r.signal || 'hold'}`}>
                              {r.signal ? r.signal.toUpperCase() : '-'}
                            </td>
                            <td>
                              {r.predicted_return !== null
                                ? `${(r.predicted_return * 100).toFixed(2)}%`
                                : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>
          </>
        )}
      </main>

      {/* Rate Limit Banner */}
      {showRateBanner && (
        <div className="rate-limit-banner">
          <span>
            {requestsUsed}/{dailyLimit} requests used today
          </span>
          <Link to="/pricing" className="upgrade-link">
            Upgrade for more requests
          </Link>
        </div>
      )}

      {/* Footer with tier info */}
      <footer className="analysis-footer">
        <span>
          Tier: <strong>{userFeatures?.tier?.toUpperCase() || '-'}</strong>
        </span>
        <span>
          Requests:{' '}
          <strong>
            {dailyLimit ? `${dailyLimit - requestsUsed} remaining` : 'Unlimited'}
          </strong>
        </span>
      </footer>
    </div>
  );
}

export default Analysis;
