import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import TeamLogo from '../../components/sports/TeamLogo';
import PlayerProfileCard from '../../components/sports/PlayerProfileCard';
import {
  sports,
  type SportsBoardsResponse,
  type SportsHistoricalBoard,
  type SportsUpcomingBoard,
} from '../../api/client';
import { trackEvent } from '../../lib/analytics';
import './SportsLanding.css';

const EMPTY_SPORTS_BOARDS: SportsBoardsResponse = {
  golf: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
  tennis: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
  mlb: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
};

type SpotlightBoard = {
  label: string;
  eyebrow: string;
  description: string;
  event?: SportsUpcomingBoard;
  accent: string;
};

function fallbackReplayEvent(backtest?: SportsHistoricalBoard): SportsUpcomingBoard | undefined {
  if (!backtest) return undefined;
  return {
    id: backtest.tournamentId || `${backtest.tour}-${backtest.tournament}`,
    name: backtest.tournament,
    tour: backtest.tour,
    course: backtest.venue || backtest.course || 'Venue TBD',
    predictions: backtest.fullField || [],
  };
}

function probabilityForSide(board: SportsUpcomingBoard, side: 'away' | 'home'): number {
  const label = side === 'away' ? board.awayTeam : board.homeTeam;
  const match = board.predictions.find((prediction) => prediction.side === side || prediction.playerName === label);
  return match?.winProbability ?? 50;
}

function SportsLanding() {
  const [sportsBoards, setSportsBoards] = useState<SportsBoardsResponse>(EMPTY_SPORTS_BOARDS);
  const [boardsLoading, setBoardsLoading] = useState(true);
  const [boardsError, setBoardsError] = useState<string | null>(null);
  const [requestedMlbDate, setRequestedMlbDate] = useState<string>('');

  useEffect(() => {
    let cancelled = false;

    const loadBoards = async () => {
      setBoardsLoading(true);
      setBoardsError(null);
      try {
        const payload = await sports.getBoards(requestedMlbDate || undefined);
        if (!cancelled) {
          setSportsBoards(payload);
        }
      } catch (error) {
        if (!cancelled) {
          setBoardsError(error instanceof Error ? error.message : 'Unable to load sports boards right now.');
        }
      } finally {
        if (!cancelled) {
          setBoardsLoading(false);
        }
      }
    };

    void loadBoards();
    return () => {
      cancelled = true;
    };
  }, [requestedMlbDate]);

  const golfEvents = sportsBoards.golf.upcoming;
  const tennisEvents = sportsBoards.tennis.upcoming;
  const mlbEvents = sportsBoards.mlb.upcoming;
  const golfHistory = sportsBoards.golf.backtests;
  const tennisHistory = sportsBoards.tennis.backtests;
  const mlbHistory = sportsBoards.mlb.backtests;

  const spotlightBoards: SpotlightBoard[] = useMemo(
    () => [
      {
        label: 'Golf Board',
        eyebrow: 'PGA + LPGA',
        description: 'Tournament outrights with ranked contenders, field depth, and major-week context.',
        event: golfEvents[0],
        accent: 'teal',
      },
      {
        label: 'Tennis Board',
        eyebrow: 'ATP + WTA',
        description: 'Singles boards with fast reads on favorites, surface context, and tour rotation.',
        event: tennisEvents[0],
        accent: 'blue',
      },
      {
        label: 'MLB Board',
        eyebrow: 'MLB',
        description: 'Same-day matchup boards with probable starters, lineup context, and bullpen pressure.',
        event: mlbEvents[0] || fallbackReplayEvent(mlbHistory[0]),
        accent: 'orange',
      },
    ],
    [golfEvents, tennisEvents, mlbEvents, mlbHistory]
  );

  const sportsCoverage = [
    {
      title: 'PGA Tour',
      status: 'Live now',
      summary: 'Majors, signature events, and standard PGA fields in one outrights board workflow.',
    },
    {
      title: 'LPGA Tour',
      status: 'Live now',
      summary: "Women's majors and LPGA stops with full-field rankings and leaderboard-style scanning.",
    },
    {
      title: 'ATP Singles',
      status: 'Live now',
      summary: "Men's draws across hard, clay, and indoor courts with tour-aware probability boards.",
    },
    {
      title: 'WTA Singles',
      status: 'Live now',
      summary: "Women's draws with tournament boards, replayable track record, and clear favorite stacks.",
    },
    {
      title: 'MLB',
      status: 'Live now',
      summary: 'Same-day MLB slates with probable starters, team detail, and matchup-level pregame probabilities.',
    },
    {
      title: 'Other Team Sports',
      status: 'Coming next',
      summary: 'NBA, NFL, and NHL are next once the team-sport board templates and inference layers are production ready.',
    },
  ];

  const totalBoards = golfEvents.length + tennisEvents.length + mlbEvents.length;
  const totalBacktests = golfHistory.length + tennisHistory.length + mlbHistory.length;
  const totalTours = new Set(
    [...golfEvents, ...tennisEvents, ...mlbEvents, ...golfHistory, ...tennisHistory, ...mlbHistory].map((item) => item.tour),
  ).size;
  const totalTrackedEntrants = spotlightBoards.reduce((sum, board) => sum + (board.event?.predictions.length ?? 0), 0);
  const mlbSelectedDate = sportsBoards.mlb.selectedDate || requestedMlbDate || '';
  const mlbSelectedLabel = (sportsBoards.mlb.availableDates || []).find((option) => option.dateKey === mlbSelectedDate)?.label || 'Next active MLB slate';

  return (
    <>
      <Header />
      <main className="sports-landing">
        <section className="sports-hero">
          <div className="sports-hero-glow" />
          <div className="sports-hero-grid">
            <div className="sports-hero-copy">
              <span className="sports-badge">Multi-Sport Prediction Boards</span>
              <h1 className="sports-title">Follow sports boards the same way you would scan a live market.</h1>
              <p className="sports-subtitle">
                Open golf outrights, tennis draws, and same-day MLB matchups in one place. Fast to scan, grounded in probability,
                and backed by track record instead of hype.
              </p>

              <div className="sports-hero-tags">
                <span>PGA</span>
                <span>LPGA</span>
                <span>ATP</span>
                <span>WTA</span>
                <span>MLB</span>
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
                Research only. No wagering, no settlement layer, and no real-money contracts.
              </div>
              {boardsError && <div className="sports-disclaimer sports-error-note">Live sports feed is refreshing: {boardsError}</div>}
            </div>

            <div className="sports-market-shell">
              <div className="market-shell-header">
                <span className="market-shell-label">Live Board Snapshot</span>
                <span className="market-shell-status">
                  {boardsLoading ? 'Refreshing boards' : 'Educational probabilities'}
                </span>
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
                Designed for the same quick read you want from a market screen, but built on model probability and replayable results.
              </div>
            </div>
          </div>
        </section>

        <section className="sports-preview-section" id="boards">
          <div className="section-heading">
            <span className="section-kicker">Board Preview</span>
            <h2>See what the models like right now across every live sport we support.</h2>
            <p>
              Each board is built for fast orientation: who is favored, where the probability clusters, and which matchup
              or tournament deserves a closer look.
            </p>
          </div>

          {boardsLoading ? (
            <div className="spotlight-empty-state">Loading the latest board rotation...</div>
          ) : (
            <div className="spotlight-grid">
              {spotlightBoards.map((board) => (
                <article
                  key={board.label}
                  className={`spotlight-card accent-${board.accent}`}
                  onClick={() => trackEvent('sports_board_preview_click', { board: board.label, tour: board.event?.tour })}
                >
                  {(() => {
                    const featuredPrediction = board.event?.predictions?.[0];
                    const remainingPredictions = (board.event?.predictions ?? []).slice(1, 5);
                    return (
                      <>
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

                  {featuredPrediction ? (
                    <div className="spotlight-featured-player">
                      <div className="spotlight-featured-label">Model favorite</div>
                      <PlayerProfileCard
                        name={featuredPrediction.playerName}
                        profile={featuredPrediction.profile}
                      />
                      <div className="spotlight-featured-prob">{featuredPrediction.winProbability.toFixed(2)}%</div>
                    </div>
                  ) : null}

                  <div className="spotlight-board">
                    {remainingPredictions.map((pred) => (
                      <div key={`${board.label}-${pred.rank}-${pred.playerName}`} className="spotlight-row">
                        <span className="spotlight-rank">#{pred.rank}</span>
                        <div className="spotlight-player">
                          <PlayerProfileCard
                            name={pred.playerName}
                            profile={pred.profile}
                            compact
                          />
                        </div>
                        <span className="spotlight-prob">{pred.winProbability.toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                      </>
                    );
                  })()}
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="mlb-slate-section">
          <div className="section-heading">
            <span className="section-kicker">MLB Same-Day Board</span>
            <h2>All MLB matchups for one live slate, not a stale rolling list.</h2>
            <p>
              Pick a date, scan every same-day game, compare team logos, probable starters, and top-line probability edges,
              then jump into the full dashboard for deeper detail.
            </p>
          </div>

          <div className="mlb-slate-shell">
            <div className="mlb-slate-toolbar">
              <div className="mlb-date-pill-row">
                {(sportsBoards.mlb.availableDates || []).map((dateOption) => (
                  <button
                    key={dateOption.dateKey}
                    className={`mlb-date-pill ${dateOption.dateKey === mlbSelectedDate ? 'active' : ''}`}
                    onClick={() => {
                      setRequestedMlbDate(dateOption.dateKey);
                      trackEvent('sports_landing_mlb_date_click', { date: dateOption.dateKey });
                    }}
                  >
                    <span>{dateOption.label}</span>
                    <strong>{dateOption.gameCount} games</strong>
                  </button>
                ))}
              </div>
              <div className="mlb-slate-copy">
                <span className="sports-runtime-pill">Selected slate</span>
                <p>{mlbSelectedLabel}</p>
              </div>
            </div>

            {boardsLoading ? (
              <div className="spotlight-empty-state">Loading the live MLB slate...</div>
            ) : mlbEvents.length === 0 ? (
              <div className="spotlight-empty-state">No live MLB slate is available for the selected date right now.</div>
            ) : (
              <div className="mlb-slate-grid">
                {mlbEvents.map((board) => {
                  const awayProb = probabilityForSide(board, 'away');
                  const homeProb = probabilityForSide(board, 'home');
                  return (
                    <article key={board.id} className="mlb-slate-card">
                      <div className="mlb-slate-card-top">
                        <span className="spotlight-tour-pill">MLB</span>
                        <span>{board.course}</span>
                      </div>

                      <div className="mlb-slate-team-row">
                        <div className="mlb-slate-team">
                          <TeamLogo
                            logoUrl={board.awayTeamDetails?.logoUrl}
                            label={board.awayTeam || 'Away'}
                            abbreviation={board.awayTeamDetails?.abbreviation}
                            primaryColor={board.awayTeamDetails?.primaryColor}
                            size="sm"
                          />
                          <div>
                            <strong>{board.awayTeam}</strong>
                            <span>{board.awayStarter || 'Starter pending'}</span>
                          </div>
                        </div>
                        <div className="mlb-slate-prob">{awayProb.toFixed(1)}%</div>
                      </div>
                      {board.awayStarter ? (
                        <div className="mlb-slate-starter">
                          <PlayerProfileCard
                            name={board.awayStarter}
                            profile={board.awayStarterProfile}
                            compact
                          />
                        </div>
                      ) : null}

                      <div className="mlb-slate-team-row">
                        <div className="mlb-slate-team">
                          <TeamLogo
                            logoUrl={board.homeTeamDetails?.logoUrl}
                            label={board.homeTeam || 'Home'}
                            abbreviation={board.homeTeamDetails?.abbreviation}
                            primaryColor={board.homeTeamDetails?.primaryColor}
                            size="sm"
                          />
                          <div>
                            <strong>{board.homeTeam}</strong>
                            <span>{board.homeStarter || 'Starter pending'}</span>
                          </div>
                        </div>
                        <div className="mlb-slate-prob">{homeProb.toFixed(1)}%</div>
                      </div>
                      {board.homeStarter ? (
                        <div className="mlb-slate-starter">
                          <PlayerProfileCard
                            name={board.homeStarter}
                            profile={board.homeStarterProfile}
                            compact
                          />
                        </div>
                      ) : null}

                      <div className="mlb-slate-foot">
                        <span>{board.homeTeamDetails?.weather || 'Weather pending'}</span>
                        <span>{board.homeTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}

            <div className="mlb-slate-cta">
              <Link
                to="/sports-dashboard"
                className="btn btn-primary"
                onClick={() => trackEvent('sports_landing_cta_click', { destination: 'sports_dashboard' })}
              >
                Open Full MLB Dashboard
              </Link>
            </div>
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
              <p>See ranked contender or matchup boards across tours, jump between slates, and compare probability shapes without parsing a dense spreadsheet.</p>
            </div>
            <div className="feature-card">
              <h3>Model-driven, not crowd-driven</h3>
              <p>Each board is generated from historical performance data, field or team context, and sport-specific features instead of trader sentiment or price action.</p>
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
