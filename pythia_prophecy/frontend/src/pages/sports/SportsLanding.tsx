import { Link } from 'react-router-dom';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import { trackEvent } from '../../lib/analytics';
import golfUpcoming from '../../data/upcoming_tournaments.json';
import golfBacktests from '../../data/historical_backtests.json';
import tennisUpcoming from '../../data/wta_upcoming_tournaments.json';
import tennisBacktests from '../../data/wta_historical_backtests.json';
import './SportsLanding.css';

type EventPrediction = {
  rank: number;
  playerName: string;
  winProbability: number;
};

type UpcomingEvent = {
  id: string;
  name: string;
  original_name?: string;
  tour: string;
  course: string;
  predictions: EventPrediction[];
};

type BacktestEvent = {
  tournament: string;
  tour: string;
  hitStatus: string;
};

const golfEvents = golfUpcoming as UpcomingEvent[];
const tennisEvents = tennisUpcoming as UpcomingEvent[];
const golfHistory = golfBacktests as BacktestEvent[];
const tennisHistory = tennisBacktests as BacktestEvent[];

const spotlightBoards = [
  {
    label: 'Golf Board',
    eyebrow: 'PGA + LPGA',
    description: 'Tournament winner probabilities with full-field rankings, course context, and calibration history.',
    event: golfEvents[0],
    accent: 'teal',
  },
  {
    label: 'Tennis Board',
    eyebrow: 'ATP + WTA',
    description: 'Singles tournament boards with win probabilities, field strength context, and tour-specific model views.',
    event: tennisEvents[0],
    accent: 'blue',
  },
];

const sportsCoverage = [
  {
    title: 'PGA Tour',
    status: 'Live now',
    summary: 'Major championships, signature events, and standard PGA tournament fields.',
  },
  {
    title: 'LPGA Tour',
    status: 'Live now',
    summary: 'Women’s major weeks and full-tournament winner boards with field-aware rankings.',
  },
  {
    title: 'ATP Singles',
    status: 'Live now',
    summary: 'Men’s hard-court, clay, and indoor tournament probability boards.',
  },
  {
    title: 'WTA Singles',
    status: 'Live now',
    summary: 'Women’s tour coverage with tournament-level ranking boards and historical backtests.',
  },
  {
    title: 'Team Sports',
    status: 'Coming next',
    summary: 'NBA, MLB, NFL, and NHL are staged as the next expansion layer once board design is finalized.',
  },
];

function SportsLanding() {
  const totalBoards = golfEvents.length + tennisEvents.length;
  const totalBacktests = golfHistory.length + tennisHistory.length;
  const totalTours = new Set(
    [...golfEvents, ...tennisEvents, ...golfHistory, ...tennisHistory].map((item) => item.tour),
  ).size;
  const totalTrackedEntrants = spotlightBoards.reduce((sum, board) => sum + (board.event?.predictions.length ?? 0), 0);

  return (
    <>
      <Header />
      <main className="sports-landing">
        <section className="sports-hero">
          <div className="sports-hero-glow"></div>
          <div className="sports-hero-grid">
            <div className="sports-hero-copy">
              <span className="sports-badge">Multi-Sport Prediction Boards</span>
              <h1 className="sports-title">Market-style sports predictions without betting, trading, or event contracts.</h1>
              <p className="sports-subtitle">
                SeekingBeta.AI brings the clarity of a prediction board to sports. Browse calibrated probabilities across
                golf and tennis, compare contenders instantly, and track how our models perform over time.
              </p>

              <div className="sports-hero-tags">
                <span>PGA</span>
                <span>LPGA</span>
                <span>ATP</span>
                <span>WTA</span>
                <span>More sports staged next</span>
              </div>

              <div className="sports-cta">
                <Link
                  to="/signup"
                  className="btn btn-primary btn-lg pulse-btn"
                  onClick={() => trackEvent('sports_landing_cta_click', { destination: 'signup' })}
                >
                  Open Sports Boards
                </Link>
                <Link
                  to="/pricing"
                  className="btn btn-outline btn-lg"
                  onClick={() => trackEvent('sports_landing_cta_click', { destination: 'pricing' })}
                >
                  View Plans
                </Link>
              </div>

              <div className="sports-disclaimer">
                Probability intelligence only. No wagering, no settlement layer, and no real-money contracts.
              </div>
            </div>

            <div className="sports-market-shell">
              <div className="market-shell-header">
                <span className="market-shell-label">Live Board Snapshot</span>
                <span className="market-shell-status">Educational probabilities</span>
              </div>
              <div className="market-shell-grid">
                <div className="market-shell-stat">
                  <strong>{totalBoards}</strong>
                  <span>active boards</span>
                </div>
                <div className="market-shell-stat">
                  <strong>{totalBacktests}</strong>
                  <span>historical board replays</span>
                </div>
                <div className="market-shell-stat">
                  <strong>{totalTours}</strong>
                  <span>live tours covered</span>
                </div>
                <div className="market-shell-stat">
                  <strong>{totalTrackedEntrants}+</strong>
                  <span>entrants in spotlight boards</span>
                </div>
              </div>
              <div className="market-shell-footer">
                Built to feel fast and scannable like a market board, but grounded in model probability instead of order flow.
              </div>
            </div>
          </div>
        </section>

        <section className="sports-preview-section" id="boards">
          <div className="section-heading">
            <span className="section-kicker">Board Preview</span>
            <h2>Browse multiple tours the same way you would scan a live board.</h2>
            <p>
              The sports experience now spans golf and tennis, with each board built around contender ranking, field context,
              and a quick view of how sharp the distribution really is.
            </p>
          </div>

          <div className="spotlight-grid">
            {spotlightBoards.map((board) => (
              <article
                key={board.label}
                className={`spotlight-card accent-${board.accent}`}
                onClick={() => trackEvent('sports_board_preview_click', { board: board.label, tour: board.event?.tour })}
              >
                <div className="spotlight-card-header">
                  <div>
                    <span className="spotlight-eyebrow">{board.eyebrow}</span>
                    <h3>{board.event?.name ?? board.label}</h3>
                  </div>
                  <span className="spotlight-tour-pill">{board.event?.tour ?? 'Live'}</span>
                </div>

                <p className="spotlight-description">{board.description}</p>

                <div className="spotlight-meta">
                  <span>{board.event?.course ?? 'Venue TBD'}</span>
                  <span>{board.event?.predictions.length ?? 0} contenders ranked</span>
                </div>

                <div className="spotlight-board">
                  {(board.event?.predictions ?? []).slice(0, 5).map((pred) => (
                    <div key={`${board.label}-${pred.rank}-${pred.playerName}`} className="spotlight-row">
                      <span className="spotlight-rank">#{pred.rank}</span>
                      <span className="spotlight-player">{pred.playerName}</span>
                      <span className="spotlight-prob">{pred.winProbability.toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="sports-coverage">
          <div className="section-heading">
            <span className="section-kicker">Coverage</span>
            <h2>Available sports and tours</h2>
            <p>We are building a broad prediction surface, starting with the sports where board-style ranking is already live and usable today.</p>
          </div>

          <div className="coverage-grid">
            {sportsCoverage.map((item) => (
              <div key={item.title} className="coverage-card">
                <div className="coverage-card-top">
                  <h3>{item.title}</h3>
                  <span className={`coverage-status ${item.status === 'Live now' ? 'live' : 'staged'}`}>{item.status}</span>
                </div>
                <p>{item.summary}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="sports-features">
          <div className="section-heading">
            <span className="section-kicker">Why It Works</span>
            <h2>Built for probability discovery, not wagers.</h2>
            <p>
              The product takes the strongest part of market-style interfaces, fast scanning and side-by-side probabilities,
              and removes the trading layer entirely.
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <h3>Scan many boards quickly</h3>
              <p>See a ranked contender board for each tournament, jump between tours, and compare probability shapes without parsing a dense spreadsheet.</p>
            </div>
            <div className="feature-card">
              <h3>Model-driven, not crowd-driven</h3>
              <p>Each board is generated from historical performance data, field context, and sport-specific features instead of trader sentiment or price action.</p>
            </div>
            <div className="feature-card">
              <h3>Track record stays visible</h3>
              <p>Historical backtests sit beside the live boards so users can inspect where the models landed, not just what the current rankings say.</p>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}

export default SportsLanding;
