import { useState } from 'react';
import type { SportsMarketPick, SportsMarketSummary } from '../../api/client';
import {
  MARKETS_COLLAPSED_COUNT,
  MARKETS_DISCLAIMER,
  MARKET_RECORD_CAPTION,
  finalScoreLabel,
  formatMarketPercent,
  historyMarketBadges,
  marketAttributions,
  marketRecordLine,
  marketRecordRateLine,
  marketRowTitle,
  marketSecondaryLine,
  visibleMarketSummaries,
  visibleMarkets,
} from './marketsFormat';
import './MarketsSection.css';

type MarketsSectionProps = {
  markets?: SportsMarketPick[];
};

/** "Markets" block under an upcoming board's win-probability view. */
function MarketsSection({ markets }: MarketsSectionProps) {
  const [showAll, setShowAll] = useState(false);
  const rows = visibleMarkets(markets);
  if (!rows.length) return null;

  const shownRows = showAll ? rows : rows.slice(0, MARKETS_COLLAPSED_COUNT);
  const attributions = marketAttributions(rows);

  return (
    <section className="markets-section">
      <h3 className="markets-section-title">Markets</h3>
      <ul className="markets-list">
        {shownRows.map((pick, index) => (
          <li key={`${pick.marketId ?? pick.type}-${pick.side ?? ''}-${index}`} className="markets-row">
            <div className="markets-row-main">
              <span className="markets-row-title">{marketRowTitle(pick)}</span>
              <span className="markets-row-prob">{formatMarketPercent(pick.modelProbability)}</span>
            </div>
            <div className="markets-row-meta">{marketSecondaryLine(pick)}</div>
          </li>
        ))}
      </ul>
      {rows.length > MARKETS_COLLAPSED_COUNT ? (
        <button
          type="button"
          className="markets-toggle"
          aria-expanded={showAll}
          onClick={() => setShowAll((current) => !current)}
        >
          {showAll ? 'Show fewer' : `Show all (${rows.length})`}
        </button>
      ) : null}
      <p className="markets-footer">{MARKETS_DISCLAIMER}</p>
      {attributions.length ? <p className="markets-footer">{attributions.join(' · ')}</p> : null}
    </section>
  );
}

type HistoryMarketBadgesProps = {
  markets?: SportsMarketPick[];
};

/** Compact graded-result badges for a history (track record) row. */
export function HistoryMarketBadges({ markets }: HistoryMarketBadgesProps) {
  const badges = historyMarketBadges(markets);
  if (!badges.length) return null;

  return (
    <div className="market-result-badges">
      {badges.map((badge) => (
        <span key={badge.key} className={`market-result-badge ${badge.tone}`}>
          {badge.text}
        </span>
      ))}
    </div>
  );
}

type FinalScoreProps = {
  awayScore?: number;
  homeScore?: number;
};

export function MarketFinalScore({ awayScore, homeScore }: FinalScoreProps) {
  const label = finalScoreLabel(awayScore, homeScore);
  return label ? <span className="market-final-score">{label}</span> : null;
}

type MarketRecordProps = {
  summaries?: SportsMarketSummary[];
};

/** Live graded record per market type and season (season area). */
export function MarketRecord({ summaries }: MarketRecordProps) {
  const rows = visibleMarketSummaries(summaries);
  if (!rows.length) return null;

  return (
    <div className="market-record">
      <ul className="market-record-list">
        {rows.map((row, index) => (
          <li key={`${row.type}-${row.season ?? ''}-${index}`} className="market-record-row">
            <strong>{marketRecordLine(row)}</strong>
            <span>{marketRecordRateLine(row)}</span>
          </li>
        ))}
      </ul>
      <p className="market-record-caption">{MARKET_RECORD_CAPTION}</p>
    </div>
  );
}

type BacktestBasisCaptionProps = {
  label?: string;
};

export function BacktestBasisCaption({ label }: BacktestBasisCaptionProps) {
  return label ? <p className="backtest-basis-caption">{label}</p> : null;
}

export default MarketsSection;
