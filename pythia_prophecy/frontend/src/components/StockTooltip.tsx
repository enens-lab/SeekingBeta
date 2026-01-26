import { useState, useEffect, useRef, ReactNode } from 'react';
import { companies } from '../api/client';

interface CompanyInfo {
  sector?: string;
  industry?: string;
  market_cap?: number;
  pe_ratio?: number;
  beta?: number;
  dividend_yield?: number;
}

interface CompanyData {
  company: {
    ticker: string;
    name?: string;
    asset_type?: string;
  };
  info?: CompanyInfo;
}

const cache = new Map<string, CompanyData>();

interface StockTooltipProps {
  ticker: string;
  children: ReactNode;
}

function StockTooltip({ ticker, children }: StockTooltipProps) {
  const [visible, setVisible] = useState(false);
  const [data, setData] = useState<CompanyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hoverTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchData = async () => {
    if (cache.has(ticker)) {
      setData(cache.get(ticker)!);
      return;
    }

    setLoading(true);
    try {
      const result = await companies.get(ticker) as unknown as CompanyData;
      cache.set(ticker, result);
      setData(result);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const updatePosition = () => {
    if (!triggerRef.current) return;

    const rect = triggerRef.current.getBoundingClientRect();
    const tooltipWidth = 280;
    const tooltipHeight = 200;

    let left = rect.left + rect.width / 2 - tooltipWidth / 2;
    let top = rect.bottom + 8;

    if (left < 10) left = 10;
    if (left + tooltipWidth > window.innerWidth - 10) {
      left = window.innerWidth - tooltipWidth - 10;
    }

    if (top + tooltipHeight > window.innerHeight - 10) {
      top = rect.top - tooltipHeight - 8;
    }

    setPosition({ top, left });
  };

  const handleMouseEnter = () => {
    hoverTimeout.current = setTimeout(() => {
      setVisible(true);
      updatePosition();
      if (!cache.has(ticker)) {
        fetchData();
      } else {
        setData(cache.get(ticker)!);
      }
    }, 300);
  };

  const handleMouseLeave = () => {
    if (hoverTimeout.current) {
      clearTimeout(hoverTimeout.current);
    }
    setVisible(false);
  };

  useEffect(() => {
    return () => {
      if (hoverTimeout.current) {
        clearTimeout(hoverTimeout.current);
      }
    };
  }, []);

  const formatMarketCap = (val?: number) => {
    if (!val) return '--';
    if (val >= 1e12) return `$${(val / 1e12).toFixed(2)}T`;
    if (val >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
    if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
    return `$${val.toLocaleString()}`;
  };

  const formatRatio = (val?: number) => {
    if (val == null) return '--';
    return val.toFixed(2);
  };

  const formatPercent = (val?: number) => {
    if (val == null) return '--';
    return `${(val * 100).toFixed(2)}%`;
  };

  return (
    <div
      className="stock-tooltip-trigger"
      ref={triggerRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {visible && (
        <div
          className="stock-tooltip"
          ref={tooltipRef}
          style={{ top: position.top, left: position.left }}
        >
          {loading ? (
            <div className="stock-tooltip-loading">
              <div className="spinner-small" />
              <span>Loading...</span>
            </div>
          ) : data ? (
            <>
              <div className="stock-tooltip-header">
                <span className="stock-tooltip-ticker">{data.company.ticker}</span>
                {data.company.asset_type && (
                  <span className={`stock-tooltip-type type-${data.company.asset_type}`}>
                    {data.company.asset_type.toUpperCase()}
                  </span>
                )}
              </div>
              <div className="stock-tooltip-name">{data.company.name || 'Unknown'}</div>

              {data.info && (
                <>
                  {(data.info.sector || data.info.industry) && (
                    <div className="stock-tooltip-sector">
                      {data.info.sector}
                      {data.info.sector && data.info.industry && ' / '}
                      {data.info.industry}
                    </div>
                  )}

                  <div className="stock-tooltip-metrics">
                    <div className="stock-tooltip-metric">
                      <span className="metric-label">Mkt Cap</span>
                      <span className="metric-value">{formatMarketCap(data.info.market_cap)}</span>
                    </div>
                    <div className="stock-tooltip-metric">
                      <span className="metric-label">P/E</span>
                      <span className="metric-value">{formatRatio(data.info.pe_ratio)}</span>
                    </div>
                    <div className="stock-tooltip-metric">
                      <span className="metric-label">Beta</span>
                      <span className="metric-value">{formatRatio(data.info.beta)}</span>
                    </div>
                    <div className="stock-tooltip-metric">
                      <span className="metric-label">Div Yield</span>
                      <span className="metric-value">{formatPercent(data.info.dividend_yield)}</span>
                    </div>
                  </div>
                </>
              )}
            </>
          ) : (
            <div className="stock-tooltip-empty">
              <span className="stock-tooltip-ticker">{ticker}</span>
              <span className="stock-tooltip-no-data">No data available</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default StockTooltip;
