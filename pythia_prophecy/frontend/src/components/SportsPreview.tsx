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
  mlb: { upcoming: [], backtests: [], updated_at: '', source: 'runtime_filtered_sports_feed', selectedDate: undefined, availableDates: [] },
};

type SpotlightBoard = {
  key: 'golf' | 'tennis';
  eyebrow: string;
  description: string;
  event?: SportsUpcomingBoard;
  accent: 'teal' | 'blue';
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

function probabilityForSide(board: SportsUpcomingBoard, side: 'away' | 'home'): number {
  const label = side === 'away' ? board.awayTeam : board.homeTeam;
  const match = board.predictions.find((prediction) => prediction.side === side || prediction.playerName === label);
  return match?.winProbability ?? 50;
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
    ],
    [sportsBoards]
  );

  const mlbEvents = sportsBoards.mlb.upcoming.slice(0, 4);
  const mlbSelectedDate = sportsBoards.mlb.selectedDate;
  const mlbSelectedLabel =
    (sportsBoards.mlb.availableDates || []).find((option) => option.dateKey === mlbSelectedDate)?.label ||
    'Next active MLB slate';
  const runtimeStamp = updatedLabel(
    sportsBoards.golf.updated_at || sportsBoards.tennis.updated_at || sportsBoards.mlb.updated_at
  );

  return (
    <section className="sports-home-preview" id="sports-preview">
      <div className="section-header sports-home-header">
        <div>
          <span className="section-kicker">Sports</span>
          <h2 className="section-title">See today&apos;s sports boards at a glance.</h2>
          <p className="section-subtitle">
            A quick look at golf, tennis, and MLB.
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

                {topPredictions.length > 0 ? (
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
                <span className="sports-home-eyebrow">MLB Same-Day Slate</span>
                <h3>{mlbSelectedLabel}</h3>
              </div>
              <span className="sports-home-pill">MLB</span>
            </div>

            <p className="sports-home-description">
              Today&apos;s MLB games with starters and team notes.
            </p>

            <div className="sports-home-meta">
              <span>{mlbEvents.length} games</span>
              <span>Live feed</span>
            </div>

            {mlbEvents.length === 0 ? (
              <div className="sports-home-empty-inline">No MLB games in the current window.</div>
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
                          <span>{awayProb.toFixed(1)}%</span>
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
                          <span>{homeProb.toFixed(1)}%</span>
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
