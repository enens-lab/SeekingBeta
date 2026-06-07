import { useState, useEffect, MouseEvent } from 'react';
import { companies } from '../api/client';

interface CompanyInfo {
  sector?: string;
  industry?: string;
  market_cap?: number;
  pe_ratio?: number;
  forward_pe?: number;
  price_to_book?: number;
  beta?: number;
  dividend_yield?: number;
  fifty_two_week_high?: number;
  fifty_two_week_low?: number;
  avg_volume?: number;
  employees?: number;
  description?: string;
  website?: string;
  exchange?: string;
  currency?: string;
  city?: string;
  state?: string;
  country?: string;
}

interface NewsArticle {
  article_id?: string;
  title: string;
  link: string;
  publisher?: string;
  published_at?: string;
}

interface CompanyData {
  company: {
    ticker: string;
    name?: string;
    asset_type?: string;
  };
  info?: CompanyInfo;
  news?: NewsArticle[];
}

interface CompanyDetailProps {
  ticker: string | null;
  onClose: () => void;
}

function CompanyDetail({ ticker, onClose }: CompanyDetailProps) {
  const [data, setData] = useState<CompanyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAllNews, setShowAllNews] = useState(false);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setShowAllNews(false);

    companies.get(ticker)
      .then((result) => setData(result as unknown as CompanyData))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load company data'))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (!ticker) return null;

  const formatMarketCap = (val?: number) => {
    if (!val) return '--';
    if (val >= 1e12) return `$${(val / 1e12).toFixed(2)}T`;
    if (val >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
    if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
    return `$${val.toLocaleString()}`;
  };

  const formatNumber = (val?: number) => {
    if (val == null) return '--';
    return val.toLocaleString();
  };

  const formatPercent = (val?: number) => {
    if (val == null) return '--';
    return `${(val * 100).toFixed(2)}%`;
  };

  const formatRatio = (val?: number) => {
    if (val == null) return '--';
    return val.toFixed(2);
  };

  const handleOverlayClick = () => {
    onClose();
  };

  const handlePanelClick = (e: MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <div className="company-detail-overlay" onClick={handleOverlayClick}>
      <div className="company-detail-panel" onClick={handlePanelClick}>
        <button className="company-detail-close" onClick={onClose}>
          &times;
        </button>

        {loading ? (
          <div className="company-detail-loading">
            <div className="spinner" />
            <p>Loading company data...</p>
          </div>
        ) : error ? (
          <div className="company-detail-error">
            <p>Company profile is temporarily unavailable for {ticker}.</p>
            <p className="company-detail-hint">Please retry in a few seconds.</p>
          </div>
        ) : data ? (
          <>
            <div className="company-detail-header">
              <div>
                <h2 className="company-detail-ticker">{data.company.ticker}</h2>
                <p className="company-detail-name">{data.company.name || 'Unknown'}</p>
              </div>
              {data.company.asset_type && (
                <span className={`company-type-badge type-${data.company.asset_type}`}>
                  {data.company.asset_type.toUpperCase()}
                </span>
              )}
            </div>

            {!data.info && (
              <div className="company-detail-hint-box">
                Fundamentals are still syncing for this ticker. Basic profile is available.
              </div>
            )}

            {data.info && (
              <div className="company-detail-info">
                {(data.info.sector || data.info.industry) && (
                  <div className="company-detail-sector">
                    {data.info.sector && <span>{data.info.sector}</span>}
                    {data.info.sector && data.info.industry && <span className="separator">/</span>}
                    {data.info.industry && <span>{data.info.industry}</span>}
                  </div>
                )}

                <div className="company-metrics-grid">
                  <div className="company-metric">
                    <span className="metric-label">Market Cap</span>
                    <span className="metric-value">{formatMarketCap(data.info.market_cap)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">P/E Ratio</span>
                    <span className="metric-value">{formatRatio(data.info.pe_ratio)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">Forward P/E</span>
                    <span className="metric-value">{formatRatio(data.info.forward_pe)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">P/B Ratio</span>
                    <span className="metric-value">{formatRatio(data.info.price_to_book)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">Beta</span>
                    <span className="metric-value">{formatRatio(data.info.beta)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">Div Yield</span>
                    <span className="metric-value">{formatPercent(data.info.dividend_yield)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">52W High</span>
                    <span className="metric-value">{data.info.fifty_two_week_high ? `$${data.info.fifty_two_week_high.toFixed(2)}` : '--'}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">52W Low</span>
                    <span className="metric-value">{data.info.fifty_two_week_low ? `$${data.info.fifty_two_week_low.toFixed(2)}` : '--'}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">Avg Volume</span>
                    <span className="metric-value">{formatNumber(data.info.avg_volume)}</span>
                  </div>
                  <div className="company-metric">
                    <span className="metric-label">Employees</span>
                    <span className="metric-value">{formatNumber(data.info.employees)}</span>
                  </div>
                </div>

                {data.info.description && (
                  <div className="company-description">
                    <h4>About</h4>
                    <p>{data.info.description}</p>
                  </div>
                )}

                {(data.info.website || data.info.exchange) && (
                  <div className="company-meta">
                    {data.info.exchange && (
                      <span className="meta-item">
                        {data.info.exchange} ({data.info.currency || 'USD'})
                      </span>
                    )}
                    {data.info.city && data.info.country && (
                      <span className="meta-item">
                        {data.info.city}{data.info.state ? `, ${data.info.state}` : ''}, {data.info.country}
                      </span>
                    )}
                    {data.info.website && (
                      <a
                        className="meta-item meta-link"
                        href={data.info.website}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {data.info.website.replace(/^https?:\/\/(www\.)?/, '')}
                      </a>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="company-news-section">
              <h4>Recent News</h4>
              {(() => {
                const validNews = (data.news || []).filter(
                  (article) => article.title && article.title.trim()
                );
                if (validNews.length === 0) {
                  return <p className="no-news-message">No recent news</p>;
                }
                const displayedNews = showAllNews ? validNews : validNews.slice(0, 5);
                const hasMore = validNews.length > 5;
                return (
                  <>
                    <div className="company-news-list">
                      {displayedNews.map((article, idx) => (
                        <a
                          key={article.article_id || idx}
                          className="company-news-item"
                          href={article.link}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <div className="news-item-content">
                            <span className="news-title">{article.title}</span>
                            <div className="news-meta">
                              {article.publisher && <span>{article.publisher}</span>}
                              {article.published_at && (
                                <span>{new Date(article.published_at).toLocaleDateString()}</span>
                              )}
                            </div>
                          </div>
                        </a>
                      ))}
                    </div>
                    {hasMore && !showAllNews && (
                      <button
                        className="load-more-news-btn"
                        onClick={() => setShowAllNews(true)}
                      >
                        Load More
                      </button>
                    )}
                  </>
                );
              })()}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

export default CompanyDetail;
