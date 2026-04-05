import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import PlayerPortrait from './sports/PlayerPortrait';
import TeamLogo from './sports/TeamLogo';
import {
  sports,
  type SportsBoardsResponse,
  type SportsHistoricalBoard,
  type SportsUpcomingBoard,
} from '../api/client';
import { trackEvent } from '../lib/analytics';

const EMPTY_SPORTS_BOARDS: SportsBoardsResponse = {
  golf: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
  tennis: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
  basketball: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
  mlb: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
  football: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
};

type SpotlightBoard = {
  key: 'golf' | 'tennis' | 'basketball';
  eyebrow: string;
  description: string;
  event?: SportsUpcomingBoard;
  accent: 'teal' | 'blue' | 'orange';
  cta: string;
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

function probabilityForSide(board: SportsUpcomingBoard, side: 'away' | 'home'): number | null {
  const label = side === 'away' ? board.awayTeam : board.homeTeam;
  const match = board.predictions.find((prediction) => prediction.side === side || prediction.playerName === label);
  return typeof match?.winProbability === 'number' ? match.winProbability : null;
}

function updatedLabel(value?: string): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function SportsPreview() {
  const [sportsBoards, setSportsBoards] = useState<SportsBoardsResponse>(EMPTY_SPORTS_BOARDS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadBoards = async () => {
      setLoading(true);
      setError(null);
      try {
        const payload = await sports.getBoards();
        if (!cancelled) {
          setSportsBoards(payload);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Unable to load sports preview right now.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void loadBoards();
    return () => {
      cancelled = true;
    };
  }, []);

  const spotlightBoards: SpotlightBoard[] = useMemo(
    () => [
      {
        key: 'golf',
        eyebrow: 'PGA + LPGA',
        description: 'A quick look at the next golf event and who the model likes.',
        event: sportsBoards.golf.upcoming[0] || fallbackReplayEvent(sportsBoards.golf.backtests[0]),
        accent: 'teal',
        cta: 'Open golf page',
      },
      {
        key: 'tennis',
        eyebrow: 'ATP + WTA',
        description: 'A quick look at the next tennis event and the top names on the board.',
        event: sportsBoards.tennis.upcoming[0] || fallbackReplayEvent(sportsBoards.tennis.backtests[0]),
        accent: 'blue',
        cta: 'Open tennis page',
      },
      {
        key: 'basketball',
        eyebrow: 'Men + Women',
        description: 'A quick look at the next Basketball slate and the side the model likes.',
        event: sportsBoards.basketball.upcoming[0] || fallbackReplayEvent(sportsBoards.basketball.backtests[0]),
        accent: 'orange',
        cta: 'Open Basketball page',
      },
    ],
    [sportsBoards]
  );

  const mlbEvents = sportsBoards.mlb.upcoming.slice(0, 4);
  const mlbSelectedDate = sportsBoards.mlb.selectedDate;
  const mlbSelectedLabel =
    (sportsBoards.mlb.availableDates || []).find((option) => option.dateKey === mlbSelectedDate)?.label ||
    'Next active Baseball slate';
  const runtimeStamp = updatedLabel(
    sportsBoards.golf.updated_at || sportsBoards.tennis.updated_at || sportsBoards.basketball.updated_at || sportsBoards.mlb.updated_at
  );

  return (
    <section className="sports-home-preview" id="sports-preview">
      <div className="section-header sports-home-header">
        <div>
          <span className="section-kicker">Sports</span>
          <h2 className="section-title">See today&apos;s sports boards at a glance.</h2>
          <p className="section-subtitle">
            A quick look at golf, tennis, Basketball, and Baseball.
          </p>
        </div>
        <div className="sports-home-actions">
          {runtimeStamp ? <span className="sports-home-runtime">Updated {runtimeStamp}</span> : null}
          <Link
            to="/sports"
            className="btn btn-outline"
            onClick={() => trackEvent('home_sports_preview_click', { destination: 'sports' })}
          >
            Explore Sports
          </Link>
        </div>
      </div>

      {loading ? (
        <div className="sports-home-loading">
          <div className="spinner" />
          <p>Loading sports...</p>
        </div>
      ) : error ? (
        <div className="sports-home-empty">
          <h3>Sports are updating</h3>
          <p>{error}</p>
        </div>
      ) : (
        <div className="sports-home-grid">
          {spotlightBoards.map((board) => {
            const topPredictions = (board.event?.predictions || []).slice(0, 3);
            const isTeamPreview = board.key === 'basketball' && board.event?.awayTeam && board.event?.homeTeam;
            const awayProb = board.event && isTeamPreview ? probabilityForSide(board.event, 'away') : null;
            const homeProb = board.event && isTeamPreview ? probabilityForSide(board.event, 'home') : null;
            return (
              <article key={board.key} className={`sports-home-card accent-${board.accent}`}>
                <div className="sports-home-card-header">
                  <div>
                    <span className="sports-home-eyebrow">{board.eyebrow}</span>
                    <h3>{board.event?.name || `${board.key === 'golf' ? 'Golf' : 'Tennis'} board`}</h3>
                  </div>
                  <span className="sports-home-pill">{board.event?.tour || 'Live'}</span>
                </div>

                <p className="sports-home-description">{board.description}</p>

                <div className="sports-home-meta">
                  <span>{board.event?.course || 'Venue TBD'}</span>
                  <span>{board.event?.predictions?.length || 0} names</span>
                </div>

                {isTeamPreview && board.event ? (
                  <div className="sports-home-mlb-list">
                    <div className="sports-home-mlb-row">
                      <div className="sports-home-mlb-team">
                        <TeamLogo
                          logoUrl={board.event.awayTeamDetails?.logoUrl}
                          label={board.event.awayTeam || 'Away'}
                          abbreviation={board.event.awayTeamDetails?.abbreviation}
                          primaryColor={board.event.awayTeamDetails?.primaryColor}
                          size="sm"
                        />
                        <div>
                          <strong>{board.event.awayTeam}</strong>
                          <span>{board.event.awayTeamDetails?.recordPrior || board.event.awayTeamDetails?.recentForm || 'Team detail pending'}</span>
                        </div>
                      </div>
                      <span className="sports-home-prob">{awayProb !== null ? `${awayProb.toFixed(1)}%` : 'Pending'}</span>
                    </div>
                    <div className="sports-home-mlb-row">
                      <div className="sports-home-mlb-team">
                        <TeamLogo
                          logoUrl={board.event.homeTeamDetails?.logoUrl}
                          label={board.event.homeTeam || 'Home'}
                          abbreviation={board.event.homeTeamDetails?.abbreviation}
                          primaryColor={board.event.homeTeamDetails?.primaryColor}
                          size="sm"
                        />
                        <div>
                          <strong>{board.event.homeTeam}</strong>
                          <span>{board.event.homeTeamDetails?.recordPrior || board.event.homeTeamDetails?.recentForm || 'Team detail pending'}</span>
                        </div>
                      </div>
                      <span className="sports-home-prob">{homeProb !== null ? `${homeProb.toFixed(1)}%` : 'Pending'}</span>
                    </div>
                  </div>
                ) : topPredictions.length > 0 ? (
                  <div className="sports-home-player-list">
                    {topPredictions.map((prediction) => (
                      <div key={`${board.key}-${prediction.rank}-${prediction.playerName}`} className="sports-home-player-row">
                        <span className="sports-home-player-rank">#{prediction.rank}</span>
                        <PlayerPortrait imageUrl={prediction.profile?.imageUrl} label={prediction.playerName} size="sm" />
                        <div className="sports-home-player-main">
                          <strong>{prediction.playerName}</strong>
                          <span className="sports-home-player-meta">
                            {prediction.profile?.subtitle || prediction.profile?.country || 'Live board'}
                          </span>
                        </div>
                        <span className="sports-home-prob">{prediction.winProbability.toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="sports-home-empty-inline">No live board right now.</div>
                )}

                <div className="sports-home-card-footer">
                  <Link
                    to="/sports-dashboard"
                    className="sports-home-link"
                    onClick={() =>
                      trackEvent('home_sports_preview_click', { destination: 'sports_dashboard', board: board.key })
                    }
                  >
                    {board.cta}
                  </Link>
                </div>
              </article>
            );
          })}

          <article className="sports-home-card sports-home-card-mlb accent-orange">
            <div className="sports-home-card-header">
              <div>
                <span className="sports-home-eyebrow">Baseball Same-Day Slate</span>
                <h3>{mlbSelectedLabel}</h3>
              </div>
              <span className="sports-home-pill">Baseball</span>
            </div>

            <p className="sports-home-description">
              Today&apos;s Baseball games with starters and team notes.
            </p>

            <div className="sports-home-meta">
              <span>{mlbEvents.length} games</span>
              <span>Live feed</span>
            </div>

            {mlbEvents.length === 0 ? (
              <div className="sports-home-empty-inline">No Baseball games in the current window.</div>
            ) : (
              <div className="sports-home-mlb-list">
                {mlbEvents.map((board) => {
                  const awayProb = probabilityForSide(board, 'away');
                  const homeProb = probabilityForSide(board, 'home');
                  return (
                    <div key={board.id} className="sports-home-mlb-row">
                      <div className="sports-home-mlb-team">
                        <TeamLogo
                          logoUrl={board.awayTeamDetails?.logoUrl}
                          label={board.awayTeam || 'Away'}
                          abbreviation={board.awayTeamDetails?.abbreviation}
                          primaryColor={board.awayTeamDetails?.primaryColor}
                          size="sm"
                        />
                        <div>
                          <strong>{board.awayTeam}</strong>
                          <span>{awayProb !== null ? `${awayProb.toFixed(1)}%` : 'Pending'}</span>
                        </div>
                      </div>
                      <span className="sports-home-mlb-vs">at</span>
                      <div className="sports-home-mlb-team">
                        <TeamLogo
                          logoUrl={board.homeTeamDetails?.logoUrl}
                          label={board.homeTeam || 'Home'}
                          abbreviation={board.homeTeamDetails?.abbreviation}
                          primaryColor={board.homeTeamDetails?.primaryColor}
                          size="sm"
                        />
                        <div>
                          <strong>{board.homeTeam}</strong>
                          <span>{homeProb !== null ? `${homeProb.toFixed(1)}%` : 'Pending'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="sports-home-card-footer">
              <Link
                to="/sports-dashboard"
                className="sports-home-link"
                onClick={() => trackEvent('home_sports_preview_click', { destination: 'sports_dashboard', board: 'mlb' })}
              >
                Open sports dashboard
              </Link>
            </div>
          </article>
        </div>
      )}
    </section>
  );
}

export default SportsPreview;
