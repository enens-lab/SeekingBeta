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
  basketball: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
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
  const [requestedBasketballDate, setRequestedBasketballDate] = useState<string>('');

  useEffect(() => {
    let cancelled = false;

    const loadBoards = async () => {
      setBoardsLoading(true);
      setBoardsError(null);
      try {
        const payload = await sports.getBoards({
          mlbDate: requestedMlbDate || undefined,
          basketballDate: requestedBasketballDate || undefined,
        });
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
  }, [requestedMlbDate, requestedBasketballDate]);

  const golfEvents = sportsBoards.golf.upcoming;
  const tennisEvents = sportsBoards.tennis.upcoming;
  const basketballEvents = sportsBoards.basketball.upcoming;
  const mlbEvents = sportsBoards.mlb.upcoming;
  const golfHistory = sportsBoards.golf.backtests;
  const tennisHistory = sportsBoards.tennis.backtests;
  const basketballHistory = sportsBoards.basketball.backtests;
  const mlbHistory = sportsBoards.mlb.backtests;

  const spotlightBoards: SpotlightBoard[] = useMemo(
    () => [
      {
        label: 'Golf Board',
        eyebrow: 'PGA + LPGA',
        description: 'The next golf event, ranked from top pick down.',
        event: golfEvents[0],
        accent: 'teal',
      },
      {
        label: 'Tennis Board',
        eyebrow: 'ATP + WTA',
        description: 'The next tennis event, with the top names easy to compare.',
        event: tennisEvents[0],
        accent: 'blue',
      },
      {
        label: 'Basketball Board',
        eyebrow: 'Men + Women',
        description: 'Same-day Basketball games with team form, lineup strength, and a clear top side.',
        event: basketballEvents[0] || fallbackReplayEvent(basketballHistory[0]),
        accent: 'orange',
      },
    ],
    [golfEvents, tennisEvents, basketballEvents, basketballHistory]
  );

  const sportsCoverage = [
    {
      title: 'PGA Tour',
      status: 'Live now',
      summary: 'Majors and regular PGA events with one ranked list of likely winners.',
    },
    {
      title: 'LPGA Tour',
      status: 'Live now',
      summary: "LPGA events with a simple ranked list of likely winners.",
    },
    {
      title: 'ATP Singles',
      status: 'Live now',
      summary: "Men's tennis events with favorites ranked in one place.",
    },
    {
      title: 'WTA Singles',
      status: 'Live now',
      summary: "Women's tennis events with clear favorites and past results beside them.",
    },
    {
      title: 'Basketball',
      status: 'Live now',
      summary: "Today's men and women's Basketball games with team notes, projected rotation strength, and win numbers.",
    },
    {
      title: 'Baseball',
      status: 'Live now',
      summary: "Today's Baseball games with starter info, team notes, and win numbers.",
    },
    {
      title: 'Other Team Sports',
      status: 'Coming next',
      summary: 'NFL and NHL are next.',
    },
  ];

  const totalBoards = golfEvents.length + tennisEvents.length + basketballEvents.length + mlbEvents.length;
  const totalBacktests = golfHistory.length + tennisHistory.length + basketballHistory.length + mlbHistory.length;
  const totalTours = new Set(
    [...golfEvents, ...tennisEvents, ...basketballEvents, ...mlbEvents, ...golfHistory, ...tennisHistory, ...basketballHistory, ...mlbHistory].map((item) => item.tour),
  ).size;
  const totalTrackedEntrants = spotlightBoards.reduce((sum, board) => sum + (board.event?.predictions.length ?? 0), 0);
  const mlbSelectedDate = sportsBoards.mlb.selectedDate || requestedMlbDate || '';
  const mlbSelectedLabel = (sportsBoards.mlb.availableDates || []).find((option) => option.dateKey === mlbSelectedDate)?.label || 'Next active Baseball slate';
  const basketballSelectedDate = sportsBoards.basketball.selectedDate || requestedBasketballDate || '';
  const basketballSelectedLabel =
    (sportsBoards.basketball.availableDates || []).find((option) => option.dateKey === basketballSelectedDate)?.label ||
    'Next active Basketball slate';

  return (
    <>
      <Header />
      <main className="sports-landing">
        <section className="sports-hero">
          <div className="sports-hero-glow" />
          <div className="sports-hero-grid">
            <div className="sports-hero-copy">
              <span className="sports-badge">Sports</span>
              <h1 className="sports-title">See today&apos;s sports picks in one place.</h1>
              <p className="sports-subtitle">
                Check golf, tennis, Basketball, and Baseball on one page. See the top picks, the live matchups, and the past results.
              </p>

              <div className="sports-hero-tags">
                <span>PGA</span>
                <span>LPGA</span>
                <span>ATP</span>
                <span>WTA</span>
                <span>Basketball</span>
                <span>Baseball</span>
              </div>

              <div className="sports-cta">
                <Link
                  to="/signup"
                  className="btn btn-primary btn-lg pulse-btn"
                  onClick={() => trackEvent('sports_landing_cta_click', { destination: 'signup' })}
                >
                  Try Sports
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
                For research only. We do not place bets.
              </div>
              {boardsError && <div className="sports-disclaimer sports-error-note">Sports feed is updating: {boardsError}</div>}
            </div>

            <div className="sports-market-shell">
              <div className="market-shell-header">
                <span className="market-shell-label">Live Snapshot</span>
                <span className="market-shell-status">
                  {boardsLoading ? 'Updating' : 'Live now'}
                </span>
              </div>
              <div className="market-shell-grid">
                <div className="market-shell-stat">
                  <strong>{totalBoards}</strong>
                  <span>live boards</span>
                </div>
                <div className="market-shell-stat">
                  <strong>{totalBacktests}</strong>
                  <span>past results</span>
                </div>
                <div className="market-shell-stat">
                  <strong>{totalTours}</strong>
                  <span>tours live</span>
                </div>
                <div className="market-shell-stat">
                  <strong>{totalTrackedEntrants}+</strong>
                  <span>names on screen</span>
                </div>
              </div>
              <div className="market-shell-footer">Easy to scan. Easy to compare. Backed by past results.</div>
            </div>
          </div>
        </section>

        <section className="sports-preview-section" id="boards">
          <div className="section-heading">
            <span className="section-kicker">What&apos;s Live</span>
            <h2>See what the models like right now.</h2>
            <p>
              Start with the top names, then open the full dashboard if you want more detail.
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
                    <span>{board.event?.predictions.length ?? 0} names ranked</span>
                  </div>

                  {featuredPrediction ? (
                    <div className="spotlight-featured-player">
                      <div className="spotlight-featured-label">Top pick</div>
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
            <span className="section-kicker">Basketball Today</span>
            <h2>All Basketball games for the same day.</h2>
            <p>
              Pick a date, see every game, and open the full dashboard when you want lineup-level detail.
            </p>
          </div>

          <div className="mlb-slate-shell">
            <div className="mlb-slate-toolbar">
              <div className="mlb-date-pill-row">
                {(sportsBoards.basketball.availableDates || []).map((dateOption) => (
                  <button
                    key={dateOption.dateKey}
                    className={`mlb-date-pill ${dateOption.dateKey === basketballSelectedDate ? 'active' : ''}`}
                    onClick={() => {
                      setRequestedBasketballDate(dateOption.dateKey);
                      trackEvent('sports_landing_basketball_date_click', { date: dateOption.dateKey });
                    }}
                  >
                    <span>{dateOption.label}</span>
                    <strong>{dateOption.gameCount} games</strong>
                  </button>
                ))}
              </div>
              <div className="mlb-slate-copy">
                <span className="sports-runtime-pill">Selected day</span>
                <p>{basketballSelectedLabel}</p>
              </div>
            </div>

            {boardsLoading ? (
              <div className="spotlight-empty-state">Loading today&apos;s Basketball games...</div>
            ) : basketballEvents.length === 0 ? (
              <div className="spotlight-empty-state">No Basketball games are available for that date right now.</div>
            ) : (
              <div className="mlb-slate-grid">
                {basketballEvents.map((board) => {
                  const awayProb = probabilityForSide(board, 'away');
                  const homeProb = probabilityForSide(board, 'home');
                  return (
                    <article key={board.id} className="mlb-slate-card">
                      <div className="mlb-slate-card-top">
                        <span className="spotlight-tour-pill">{board.tour}</span>
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
                            <span>{board.awayTeamDetails?.recordPrior || 'Record pending'}</span>
                          </div>
                        </div>
                        <div className="mlb-slate-prob">{awayProb.toFixed(1)}%</div>
                      </div>

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
                            <span>{board.homeTeamDetails?.recordPrior || 'Record pending'}</span>
                          </div>
                        </div>
                        <div className="mlb-slate-prob">{homeProb.toFixed(1)}%</div>
                      </div>

                      <div className="mlb-slate-foot">
                        <span>{board.homeTeamDetails?.recentForm || 'Form pending'}</span>
                        <span>{board.homeTeamDetails?.availabilitySummary || 'Rotation mostly intact'}</span>
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
                onClick={() => trackEvent('sports_landing_cta_click', { destination: 'sports_dashboard', sport: 'basketball' })}
              >
                Open Basketball Dashboard
              </Link>
            </div>
          </div>
        </section>

        <section className="mlb-slate-section">
          <div className="section-heading">
            <span className="section-kicker">Baseball Today</span>
            <h2>All Baseball games for the same day.</h2>
            <p>
              Pick a date, see every game, and open the full dashboard when you want more detail.
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
                <span className="sports-runtime-pill">Selected day</span>
                <p>{mlbSelectedLabel}</p>
              </div>
            </div>

            {boardsLoading ? (
              <div className="spotlight-empty-state">Loading today&apos;s Baseball games...</div>
            ) : mlbEvents.length === 0 ? (
              <div className="spotlight-empty-state">No Baseball games are available for that date right now.</div>
            ) : (
              <div className="mlb-slate-grid">
                {mlbEvents.map((board) => {
                  const awayProb = probabilityForSide(board, 'away');
                  const homeProb = probabilityForSide(board, 'home');
                  return (
                    <article key={board.id} className="mlb-slate-card">
                      <div className="mlb-slate-card-top">
                        <span className="spotlight-tour-pill">Baseball</span>
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
                        <span>{board.homeTeamDetails?.availabilitySummary || 'No major roster issues'}</span>
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
                Open Baseball Dashboard
              </Link>
            </div>
          </div>
        </section>

        <section className="sports-coverage">
          <div className="section-heading">
            <span className="section-kicker">Coverage</span>
            <h2>Available sports and tours</h2>
            <p>These are the sports you can use today.</p>
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
            <span className="section-kicker">Why People Use It</span>
            <h2>Simple to read. Easy to check.</h2>
            <p>
              The goal is simple: show what stands out without making you dig through a wall of numbers.
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <h3>Read it fast</h3>
              <p>Open the page and quickly see the top names, top games, and biggest gaps.</p>
            </div>
            <div className="feature-card">
              <h3>Built on past results</h3>
              <p>The boards come from real results and sport-specific data, not crowd opinion.</p>
            </div>
            <div className="feature-card">
              <h3>Past results stay visible</h3>
              <p>You can see how the models have done before, not just what they say today.</p>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}

export default SportsLanding;
