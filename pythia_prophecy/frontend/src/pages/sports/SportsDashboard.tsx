import { useEffect, useMemo, useState } from 'react';
import DashboardHeader from '../../components/DashboardHeader';
import TeamLogo from '../../components/sports/TeamLogo';
import StarterRadarChart from '../../components/sports/StarterRadarChart';
import PlayerProfileCard from '../../components/sports/PlayerProfileCard';
import {
  sports,
  type SportsBoardCollection,
  type SportsBoardKey,
  type SportsBoardsResponse,
  type SportsHistoricalBoard,
  type SportsBoardSeasonSummary,
  type SportsLineupPlayer,
  type SportsTeamDetails,
  type SportsUpcomingBoard,
  type SportsBoardPrediction,
  type SportsRadarMetric,
} from '../../api/client';
import { trackEvent } from '../../lib/analytics';
import './SportsDashboard.css';

type SportCategory = 'Golf' | 'Tennis' | 'Basketball' | 'Baseball' | 'Football' | 'Soccer' | 'Hockey';

const SPORT_CATEGORY_TO_BACKEND_KEY: Partial<Record<SportCategory, SportsBoardKey>> = {
  Golf: 'golf',
  Tennis: 'tennis',
  Basketball: 'basketball',
  Baseball: 'mlb',
  Football: 'football',
  Soccer: 'soccer',
};

const EMPTY_COLLECTION: SportsBoardCollection = {
  upcoming: [],
  backtests: [],
  updated_at: '',
  source: 'runtime_filtered_sports_feed',
  selectedDate: undefined,
  availableDates: [],
  seasonSummary: null,
};

const EMPTY_SPORTS_BOARDS: SportsBoardsResponse = {
  golf: EMPTY_COLLECTION,
  tennis: EMPTY_COLLECTION,
  basketball: EMPTY_COLLECTION,
  mlb: EMPTY_COLLECTION,
  football: EMPTY_COLLECTION,
  soccer: EMPTY_COLLECTION,
};

function runtimeUpdatedLabel(value?: string): string | null {
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

function probabilityForSide(board: SportsUpcomingBoard, side: 'away' | 'home'): number | null {
  const label = side === 'away' ? board.awayTeam : board.homeTeam;
  const match = board.predictions.find((prediction) => prediction.side === side || prediction.playerName === label);
  return typeof match?.winProbability === 'number' ? match.winProbability : null;
}

function formatProbability(value: number | null): string {
  return value !== null ? `${value.toFixed(1)}%` : 'Pending';
}

function formatAccuracy(value?: number | null): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'Pending';
}

function trackRecordTitle(backtests: SportsHistoricalBoard[]): string {
  const years = Array.from(
    new Set(
      backtests
        .map((backtest) => Number(backtest.year))
        .filter((year) => Number.isFinite(year) && year > 0)
    )
  ).sort((a, b) => a - b);
  if (!years.length) return 'Track Record';
  if (years.length === 1) return `Track Record (${years[0]})`;
  return `Track Record (${years[0]} - ${years[years.length - 1]})`;
}

function predictedTeamFromProbabilities(board: SportsUpcomingBoard, awayProb: number | null, homeProb: number | null): string | undefined {
  if (awayProb === null && homeProb === null) return undefined;
  if (awayProb === null) return board.homeTeam;
  if (homeProb === null) return board.awayTeam;
  return awayProb > homeProb ? board.awayTeam : board.homeTeam;
}

type SeasonSummaryCardsProps = {
  summary?: SportsBoardSeasonSummary | null;
  isTeamSport: boolean;
};

function SeasonSummaryCards({ summary, isTeamSport }: SeasonSummaryCardsProps) {
  if (!summary) return null;

  if (summary.sampleSize === 0) {
    return (
      <div className="season-summary-empty">
        <strong>{summary.year} so far</strong>
        <span>No finished boards have been logged yet this year.</span>
      </div>
    );
  }

  return (
    <div className="season-summary-grid">
      <div className="season-summary-card highlight">
        <span className="season-summary-label">{summary.year} so far</span>
        <strong>{formatAccuracy(summary.topPickAccuracy)}</strong>
        <p>Top pick accuracy</p>
      </div>
      <div className="season-summary-card">
        <span className="season-summary-label">Finished boards</span>
        <strong>{summary.sampleSize}</strong>
        <p>Counted in the current year</p>
      </div>
      <div className="season-summary-card">
        <span className="season-summary-label">Top pick hits</span>
        <strong>{summary.topPickHits}</strong>
        <p>Wins from the highest-ranked side</p>
      </div>
      {!isTeamSport && summary.top3Accuracy !== null && summary.top3Accuracy !== undefined ? (
        <div className="season-summary-card">
          <span className="season-summary-label">Top 3 hit rate</span>
          <strong>{formatAccuracy(summary.top3Accuracy)}</strong>
          <p>{summary.top3Hits ?? 0} boards finished inside the top 3</p>
        </div>
      ) : null}
      {!isTeamSport && summary.top5Accuracy !== null && summary.top5Accuracy !== undefined ? (
        <div className="season-summary-card">
          <span className="season-summary-label">Top 5 hit rate</span>
          <strong>{formatAccuracy(summary.top5Accuracy)}</strong>
          <p>{summary.top5Hits ?? 0} boards finished inside the top 5</p>
        </div>
      ) : null}
    </div>
  );
}

function predictionSourceLabel(source?: string): string {
  if (!source) return 'Live model';
  if (source.includes('heuristic_fallback')) return 'Fallback scorer';
  if (source.includes('torch_model')) return 'Torch model';
  return 'Baseline model';
}

function predictionBarWidth(prediction: SportsBoardPrediction, maxProb: number): string {
  const safeMax = maxProb > 0 ? maxProb : 100;
  return `${(prediction.winProbability / safeMax) * 100}%`;
}

type MlbLineupCardProps = {
  teamName?: string;
  teamDetails?: SportsTeamDetails;
  lineup?: SportsLineupPlayer[];
  emptyLabel: string;
  summaryLabel?: string;
};

type ExpandableLineupPlayerCardProps = {
  teamName?: string;
  player: SportsLineupPlayer;
  accentColor?: string | null;
};

function ExpandableLineupPlayerCard({ teamName, player, accentColor }: ExpandableLineupPlayerCardProps) {
  const [expanded, setExpanded] = useState(false);
  const hasRadar = (player.radarMetrics?.length || 0) > 0;

  return (
    <div className={`mlb-lineup-player-shell ${expanded ? 'expanded' : ''}`}>
      <div className="mlb-lineup-row">
        <span className="mlb-lineup-slot">{player.lineupSlot || '-'}</span>
        <div className="mlb-lineup-player">
          <PlayerProfileCard
            name={player.playerName}
            profile={player.profile}
            compact
          />
          {player.performanceSummary ? (
            <div className="mlb-lineup-summary">{player.performanceSummary}</div>
          ) : null}
        </div>
        {hasRadar ? (
          <button
            type="button"
            className="lineup-details-toggle"
            onClick={() => setExpanded((current) => !current)}
          >
            {expanded ? 'Hide detail' : 'Show detail'}
          </button>
        ) : null}
      </div>
      {expanded && hasRadar ? (
        <div className="mlb-lineup-player-detail">
          <StarterRadarChart
            title={player.playerName}
            subtitle={player.performanceSummary || player.profile?.subtitle || `${teamName || 'Team'} player`}
            metrics={player.radarMetrics}
            accentColor={accentColor}
          />
        </div>
      ) : null}
    </div>
  );
}

function MlbLineupCard({ teamName, teamDetails, lineup, emptyLabel, summaryLabel = 'projected players' }: MlbLineupCardProps) {
  const players = lineup || [];

  return (
    <section className="mlb-lineup-card">
      <div className="mlb-lineup-header">
        <div className="mlb-team-detail-heading">
          <TeamLogo
            logoUrl={teamDetails?.logoUrl}
            label={teamName || 'Team'}
            abbreviation={teamDetails?.abbreviation}
            primaryColor={teamDetails?.primaryColor}
            size="sm"
          />
          <div>
            <h3>{teamName || 'Team lineup'}</h3>
            <p>{players.length > 0 ? `${players.length} ${summaryLabel}` : emptyLabel}</p>
          </div>
        </div>
      </div>
      {players.length === 0 ? (
        <div className="mlb-lineup-empty">{emptyLabel}</div>
      ) : (
        <div className="mlb-lineup-rows">
          {players.map((player) => (
            <ExpandableLineupPlayerCard
              key={`${teamName}-${player.lineupSlot || 'x'}-${player.playerId || player.playerName}`}
              teamName={teamName}
              player={player}
              accentColor={teamDetails?.primaryColor}
            />
          ))}
        </div>
      )}
    </section>
  );
}

type BasketballFeaturedPlayerCardProps = {
  teamName?: string;
  player?: SportsLineupPlayer;
  accentColor?: string | null;
};

function BasketballFeaturedPlayerCard({ teamName, player, accentColor }: BasketballFeaturedPlayerCardProps) {
  return (
    <div className="basketball-featured-player-card">
      <div className="basketball-featured-player-header">
        <div>
          <span className="basketball-featured-player-eyebrow">{teamName || 'Team'} spotlight</span>
          <h3>{player?.playerName || 'Featured player pending'}</h3>
        </div>
      </div>
      {player ? (
        <>
          <PlayerProfileCard
            name={player.playerName}
            profile={player.profile}
          />
          <StarterRadarChart
            title={player.playerName}
            subtitle={player.performanceSummary || player.profile?.subtitle || 'Projected contributor'}
            metrics={player.radarMetrics}
            accentColor={accentColor}
          />
        </>
      ) : (
        <div className="mlb-lineup-empty">Featured player detail will appear once the projected rotation is ready.</div>
      )}
    </div>
  );
}

type ExpandablePredictionCardProps = {
  player: SportsBoardPrediction;
  accentColor?: string | null;
};

function ExpandablePredictionCard({ player, accentColor }: ExpandablePredictionCardProps) {
  const [expanded, setExpanded] = useState(false);
  const hasRadar = (player.radarMetrics?.length || 0) > 0;

  return (
    <div className={`detail-player ${player.actualWinner ? 'actual-winner-highlight' : ''} ${expanded ? 'expanded' : ''}`}>
      <div className="detail-player-topline">
        <span className="dp-rank">#{player.rank}</span>
        <span className="dp-prob">{player.winProbability.toFixed(2)}%</span>
        {player.actualWinner && <span className="dp-badge">Winner</span>}
      </div>
      <PlayerProfileCard
        name={player.playerName}
        profile={player.profile}
        compact
      />
      {hasRadar ? (
        <button
          type="button"
          className="detail-player-toggle"
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? 'Hide detail' : 'Show detail'}
        </button>
      ) : null}
      {expanded && hasRadar ? (
        <StarterRadarChart
          title={player.playerName}
          subtitle={player.profile?.subtitle || 'Player profile'}
          metrics={player.radarMetrics as SportsRadarMetric[]}
          accentColor={accentColor}
        />
      ) : null}
    </div>
  );
}

type PredictionSpotlightCardProps = {
  player?: SportsBoardPrediction;
  accentColor?: string | null;
};

function PredictionSpotlightCard({ player, accentColor }: PredictionSpotlightCardProps) {
  if (!player) {
    return null;
  }

  return (
    <div className="basketball-featured-player-card">
      <div className="basketball-featured-player-header">
        <div>
          <span className="basketball-featured-player-eyebrow">Model favorite</span>
          <h3>{player.playerName}</h3>
        </div>
      </div>
      <PlayerProfileCard
        name={player.playerName}
        profile={player.profile}
      />
      <StarterRadarChart
        title={player.playerName}
        subtitle={player.profile?.subtitle || 'Player profile'}
        metrics={player.radarMetrics}
        accentColor={accentColor}
      />
    </div>
  );
}

function SportsDashboard() {
  const [sportsBoards, setSportsBoards] = useState<SportsBoardsResponse>(EMPTY_SPORTS_BOARDS);
  const [boardsLoading, setBoardsLoading] = useState(true);
  const [boardsError, setBoardsError] = useState<string | null>(null);
  const [activeSport, setActiveSport] = useState<SportCategory>('Golf');
  const [activeTab, setActiveTab] = useState<'upcoming' | 'backtest'>('upcoming');
  const [activeEventId, setActiveEventId] = useState<string>('');
  const [showAllPredictions, setShowAllPredictions] = useState(false);
  const [expandedBacktest, setExpandedBacktest] = useState<number | null>(null);
  const [playerSearchQuery, setPlayerSearchQuery] = useState('');
  const [backtestSearchQuery, setBacktestSearchQuery] = useState('');
  const [backtestFilter, setBacktestFilter] = useState<'All' | 'Hit: Top Pick' | 'Hit: Top 3' | 'Hit: Top 5' | 'Miss'>('All');
  const [tennisTourFilter, setTennisTourFilter] = useState<'All' | 'ATP' | 'WTA'>('All');
  const [golfTourFilter, setGolfTourFilter] = useState<'All' | 'PGA' | 'LPGA'>('All');
  const [requestedMlbDate, setRequestedMlbDate] = useState<string>('');
  const [requestedBasketballDate, setRequestedBasketballDate] = useState<string>('');
  const [requestedFootballDate, setRequestedFootballDate] = useState<string>('');
  const [expandedMlbMatchupId, setExpandedMlbMatchupId] = useState<string>('');
  const [expandedBasketballMatchupId, setExpandedBasketballMatchupId] = useState<string>('');
  const [expandedFootballMatchupId, setExpandedFootballMatchupId] = useState<string>('');

  useEffect(() => {
    const backendKey = SPORT_CATEGORY_TO_BACKEND_KEY[activeSport];
    if (!backendKey) {
      setBoardsLoading(false);
      setBoardsError(null);
      return;
    }

    let cancelled = false;

    const loadBoards = async () => {
      setBoardsLoading(true);
      setBoardsError(null);
      try {
        const payload = await sports.getBoards({
          sports: backendKey,
          mlbDate: backendKey === 'mlb' && requestedMlbDate ? requestedMlbDate : undefined,
          basketballDate: backendKey === 'basketball' && requestedBasketballDate ? requestedBasketballDate : undefined,
          footballDate: backendKey === 'football' && requestedFootballDate ? requestedFootballDate : undefined,
        });
        if (!cancelled) {
          setSportsBoards((prev) => ({ ...prev, [backendKey]: payload[backendKey] }));
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
  }, [activeSport, requestedMlbDate, requestedBasketballDate, requestedFootballDate]);

  const sportDataMap = useMemo(
    () => ({
      Golf: sportsBoards.golf,
      Tennis: sportsBoards.tennis,
      Basketball: sportsBoards.basketball,
      Baseball: sportsBoards.mlb,
      Football: sportsBoards.football,
      Soccer: sportsBoards.soccer,
    }),
    [sportsBoards]
  );

  const sportData = activeSport === 'Golf' || activeSport === 'Tennis' || activeSport === 'Basketball' || activeSport === 'Baseball' || activeSport === 'Football' || activeSport === 'Soccer'
    ? sportDataMap[activeSport]
    : EMPTY_COLLECTION;

  const currentUpcoming = sportData.upcoming ?? [];
  const currentBacktests = sportData.backtests ?? [];
  const currentSeasonSummary = sportData.seasonSummary ?? null;
  const currentTrackRecordTitle = useMemo(() => trackRecordTitle(currentBacktests), [currentBacktests]);

  const filteredUpcoming = useMemo(() => {
    if (activeSport === 'Tennis' && tennisTourFilter !== 'All') {
      return currentUpcoming.filter((event) => event.tour === tennisTourFilter);
    }
    if (activeSport === 'Golf' && golfTourFilter !== 'All') {
      return currentUpcoming.filter((event) => event.tour === golfTourFilter);
    }
    return currentUpcoming;
  }, [activeSport, currentUpcoming, tennisTourFilter, golfTourFilter]);

  useEffect(() => {
    if (activeSport === 'Baseball' || activeSport === 'Basketball' || activeSport === 'Football') return;
    if (!filteredUpcoming.length) {
      if (activeEventId !== '') {
        setActiveEventId('');
      }
      return;
    }
    if (!filteredUpcoming.some((event) => event.id === activeEventId)) {
      setActiveEventId(filteredUpcoming[0].id);
    }
  }, [activeSport, filteredUpcoming, activeEventId]);

  useEffect(() => {
    if (activeSport !== 'Baseball') return;
    if (!currentUpcoming.length) {
      setExpandedMlbMatchupId('');
      return;
    }
    if (!currentUpcoming.some((event) => event.id === expandedMlbMatchupId)) {
      setExpandedMlbMatchupId(currentUpcoming[0].id);
    }
  }, [activeSport, currentUpcoming, expandedMlbMatchupId]);

  useEffect(() => {
    if (activeSport !== 'Basketball') return;
    if (!currentUpcoming.length) {
      setExpandedBasketballMatchupId('');
      return;
    }
    if (!currentUpcoming.some((event) => event.id === expandedBasketballMatchupId)) {
      setExpandedBasketballMatchupId(currentUpcoming[0].id);
    }
  }, [activeSport, currentUpcoming, expandedBasketballMatchupId]);

  useEffect(() => {
    if (activeSport !== 'Football') return;
    if (!currentUpcoming.length) {
      setExpandedFootballMatchupId('');
      return;
    }
    if (!currentUpcoming.some((event) => event.id === expandedFootballMatchupId)) {
      setExpandedFootballMatchupId(currentUpcoming[0].id);
    }
  }, [activeSport, currentUpcoming, expandedFootballMatchupId]);

  const activeEvent = useMemo(
    () => filteredUpcoming.find((event) => event.id === activeEventId) || filteredUpcoming[0],
    [filteredUpcoming, activeEventId]
  );

  const sportUpdatedAt = runtimeUpdatedLabel(sportData.updated_at);
  const maxProb = activeEvent ? Math.max(...activeEvent.predictions.map((prediction) => prediction.winProbability)) : 100;
  const mlbSelectedDate = sportData.selectedDate || requestedMlbDate || '';
  const mlbSelectedLabel = (sportData.availableDates || []).find((option) => option.dateKey === mlbSelectedDate)?.label || 'Next active slate';
  const basketballSelectedDate = sportsBoards.basketball.selectedDate || requestedBasketballDate || '';
  const basketballSelectedLabel =
    (sportsBoards.basketball.availableDates || []).find((option) => option.dateKey === basketballSelectedDate)?.label ||
    'Next active slate';
  const footballSelectedDate = sportsBoards.football.selectedDate || requestedFootballDate || '';
  const footballSelectedLabel =
    (sportsBoards.football.availableDates || []).find((option) => option.dateKey === footballSelectedDate)?.label ||
    'Next active slate';
  const isTeamSport = activeSport === 'Baseball' || activeSport === 'Basketball' || activeSport === 'Football';

  const displayedPredictions = useMemo(() => {
    if (activeSport === 'Baseball') return [];
    let filtered = activeEvent?.predictions || [];

    if (playerSearchQuery.trim() !== '') {
      const query = playerSearchQuery.toLowerCase();
      filtered = filtered.filter((prediction) => prediction.playerName.toLowerCase().includes(query));
      return filtered;
    }

    return showAllPredictions ? filtered : filtered.slice(0, 15);
  }, [activeSport, activeEvent, playerSearchQuery, showAllPredictions]);

  const filteredBacktests = useMemo(() => {
    return currentBacktests.filter((backtest) => {
      const haystack = `${backtest.tournament} ${backtest.actualWinner || ''}`.toLowerCase();
      const matchesSearch = haystack.includes(backtestSearchQuery.toLowerCase());
      const matchesFilter = backtestFilter === 'All' || backtest.hitStatus === backtestFilter.replace('Hit: ', '');

      let matchesTour = true;
      if (activeSport === 'Tennis') {
        matchesTour = tennisTourFilter === 'All' || backtest.tour === tennisTourFilter;
      } else if (activeSport === 'Golf') {
        matchesTour = golfTourFilter === 'All' || backtest.tour === golfTourFilter;
      }

      return matchesSearch && matchesFilter && matchesTour;
    });
  }, [currentBacktests, backtestSearchQuery, backtestFilter, tennisTourFilter, golfTourFilter, activeSport]);

  const handleSportChange = (sport: SportCategory) => {
    if (sport !== 'Golf' && sport !== 'Tennis' && sport !== 'Basketball' && sport !== 'Baseball' && sport !== 'Football' && sport !== 'Soccer') return;

    trackEvent('sports_category_change', { sport });
    setActiveSport(sport);
    setActiveEventId('');
    setShowAllPredictions(false);
    setExpandedBacktest(null);
    setExpandedMlbMatchupId('');
    setExpandedBasketballMatchupId('');
    setExpandedFootballMatchupId('');
    setPlayerSearchQuery('');
    setBacktestSearchQuery('');
    setBacktestFilter('All');
    setTennisTourFilter('All');
    setGolfTourFilter('All');
  };

  const toggleBacktestDetails = (idx: number) => {
    if (expandedBacktest === idx) {
      setExpandedBacktest(null);
      return;
    }
    setExpandedBacktest(idx);
    trackEvent('sports_backtest_details_click', { tournament: filteredBacktests[idx]?.tournament });
  };

  const handleMlbDateChange = (value: string) => {
    setRequestedMlbDate(value);
    setExpandedMlbMatchupId('');
    trackEvent('mlb_date_selector_change', { date: value });
  };

  const handleBasketballDateChange = (value: string) => {
    setRequestedBasketballDate(value);
    setExpandedBasketballMatchupId('');
    trackEvent('basketball_date_selector_change', { date: value });
  };

  const handleFootballDateChange = (value: string) => {
    setRequestedFootballDate(value);
    setExpandedFootballMatchupId('');
    trackEvent('football_date_selector_change', { date: value });
  };

  const renderGenericUpcomingBoard = () => {
    if (!filteredUpcoming.length) {
      return (
        <div className="tournament-card no-live-board-card">
          <div className="tournament-header">
            <div className="header-left">
              <h2>No active {activeSport} boards right now</h2>
              <span className="market-status staged">Runtime filtered</span>
            </div>
          </div>
          <div className="tournament-details">
            <p>Completed events are filtered out automatically, so this section only shows boards that still belong in the current live rotation.</p>
            <p>Check the Track Record tab if you want to review finished boards while the next slate loads in.</p>
          </div>
        </div>
      );
    }

    if (!activeEvent) return null;
    const topPrediction = activeEvent.predictions[0];
    const genericAccent = activeSport === 'Golf' ? '#2ad7c4' : '#4d99ff';

    return (
      <div className="upcoming-events-container">
        <div className="event-selector">
          <div className="selector-group selector-group-primary">
            <label>Select Board:</label>
            <select
              value={activeEventId}
              onChange={(event) => {
                setActiveEventId(event.target.value);
                setShowAllPredictions(false);
                setPlayerSearchQuery('');
              }}
              className="tournament-dropdown"
            >
              {filteredUpcoming.map((event) => (
                <option key={event.id} value={event.id}>
                  [{event.tour}] {event.name}
                </option>
              ))}
            </select>
          </div>

          {activeSport === 'Tennis' && (
            <div className="selector-group tour-filter-group">
              <label>Tour:</label>
              <select
                value={tennisTourFilter}
                onChange={(event) => setTennisTourFilter(event.target.value as 'All' | 'ATP' | 'WTA')}
                className="sports-filter-dropdown"
              >
                <option value="All">All Tennis</option>
                <option value="ATP">Men&apos;s Singles (ATP)</option>
                <option value="WTA">Women&apos;s Singles (WTA)</option>
              </select>
            </div>
          )}

          {activeSport === 'Golf' && (
            <div className="selector-group tour-filter-group">
              <label>Tour:</label>
              <select
                value={golfTourFilter}
                onChange={(event) => setGolfTourFilter(event.target.value as 'All' | 'PGA' | 'LPGA')}
                className="sports-filter-dropdown"
              >
                <option value="All">All Golf</option>
                <option value="PGA">Men&apos;s (PGA)</option>
                <option value="LPGA">Women&apos;s (LPGA)</option>
              </select>
            </div>
          )}
        </div>

        <div className="tournament-card active-market">
          <div className="tournament-header">
            <div className="header-left">
              <h2>{activeEvent.name}</h2>
              <span className="market-status live">Board Live</span>
            </div>
            <div className="header-search">
              <input
                type="text"
                placeholder="Search contender..."
                value={playerSearchQuery}
                onChange={(event) => setPlayerSearchQuery(event.target.value)}
                className="sports-search-input"
              />
            </div>
          </div>
          <div className="tournament-details">
            <p><strong>{activeSport === 'Golf' ? 'Course' : 'Surface'}:</strong> {activeEvent.course}</p>
            <p><strong>Model:</strong> Tournament-aware probability ranker</p>
          </div>

          {topPrediction?.radarMetrics?.length ? (
            <div className="basketball-featured-grid generic-featured-grid">
              <PredictionSpotlightCard
                player={topPrediction}
                accentColor={genericAccent}
              />
            </div>
          ) : null}

          {displayedPredictions.length === 0 ? (
            <div className="no-results-message">No contenders found matching &quot;{playerSearchQuery}&quot;</div>
          ) : (
            <div className="prediction-leaderboard">
              <div className="leaderboard-header">
                <span>Rank</span>
                <span>Player</span>
                <span>Win Probability</span>
              </div>
              {displayedPredictions.map((prediction) => (
                <div key={`${prediction.rank}-${prediction.playerName}`} className="leaderboard-row">
                  <span className="player-rank">#{prediction.rank}</span>
                  <div className="leaderboard-player-cell">
                    <PlayerProfileCard
                      name={prediction.playerName}
                      profile={prediction.profile}
                      compact
                    />
                  </div>
                  <div className="probability-container">
                    <span className="prob-value">{prediction.winProbability.toFixed(2)}%</span>
                    <div className="prob-bar-bg">
                      <div className="prob-bar-fill" style={{ width: predictionBarWidth(prediction, maxProb) }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {!showAllPredictions && playerSearchQuery === '' && activeEvent.predictions.length > 15 && (
            <div className="view-all-container">
              <button className="btn btn-outline" onClick={() => setShowAllPredictions(true)}>
                View All Players ({activeEvent.predictions.length})
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMlbUpcomingBoard = () => {
    const availableDates = sportData.availableDates || [];

    return (
      <div className="upcoming-events-container">
        <div className="event-selector event-selector-mlb">
          <div className="selector-group selector-group-primary">
            <label>Select Baseball slate:</label>
            <select
              value={mlbSelectedDate}
              onChange={(event) => handleMlbDateChange(event.target.value)}
              className="tournament-dropdown"
            >
              {availableDates.map((dateOption) => (
                <option key={dateOption.dateKey} value={dateOption.dateKey}>
                  {dateOption.label} | {dateOption.gameCount} games
                </option>
              ))}
            </select>
          </div>
          <div className="mlb-slate-summary">
            <span className="mlb-slate-badge">Single-day board</span>
            <p>Showing every regular-season matchup for {mlbSelectedLabel}.</p>
          </div>
        </div>

        {!currentUpcoming.length ? (
          <div className="tournament-card no-live-board-card">
            <div className="tournament-header">
              <div className="header-left">
                <h2>No Baseball games found for this slate</h2>
                <span className="market-status staged">Live schedule filtered</span>
              </div>
            </div>
            <div className="tournament-details">
              <p>We only show boards for the selected regular-season date. If Baseball is off today, pick the next active slate from the selector above.</p>
            </div>
          </div>
        ) : (
          <div className="mlb-board-list">
            {currentUpcoming.map((board) => {
              const awayProb = probabilityForSide(board, 'away');
              const homeProb = probabilityForSide(board, 'home');
              const predictedTeam = predictedTeamFromProbabilities(board, awayProb, homeProb);
              const expanded = expandedMlbMatchupId === board.id;

              return (
                <article key={board.id} className={`mlb-matchup-card ${expanded ? 'expanded' : ''}`}>
                  <div className="mlb-matchup-top">
                    <div className="mlb-matchup-copy">
                      <div className="mlb-matchup-meta">
                        <span className="spotlight-tour-pill">Baseball</span>
                        <span>{board.course}</span>
                        <span>{predictionSourceLabel(board.predictionSource)}</span>
                      </div>
                      <h2>{board.name}</h2>
                      <p className="mlb-matchup-subtitle">
                        Probable starters: {board.awayStarter || 'Starter pending'} vs {board.homeStarter || 'Starter pending'}
                      </p>
                    </div>
                    <button
                      className="btn btn-outline btn-sm"
                      onClick={() => setExpandedMlbMatchupId(expanded ? '' : board.id)}
                    >
                      {expanded ? 'Hide details' : 'Show details'}
                    </button>
                  </div>

                  <div className="mlb-matchup-body">
                    <div className="mlb-team-column">
                      <div className="mlb-team-header">
                        <TeamLogo
                          logoUrl={board.awayTeamDetails?.logoUrl}
                          label={board.awayTeam || 'Away'}
                          abbreviation={board.awayTeamDetails?.abbreviation}
                          primaryColor={board.awayTeamDetails?.primaryColor}
                          size="md"
                        />
                        <div>
                          <div className="mlb-team-name">{board.awayTeam}</div>
                          <div className="mlb-team-subtext">{board.awayStarter || 'Starter pending'}</div>
                        </div>
                      </div>
                      <div className="mlb-probability-stack">
                        <strong>{formatProbability(awayProb)}</strong>
                        <span>away win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.awayTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.awayTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
                      {board.awayStarter ? (
                        <div className="mlb-starter-inline">
                          <PlayerProfileCard
                            name={board.awayStarter}
                            profile={board.awayStarterProfile}
                            compact
                          />
                        </div>
                      ) : null}
                    </div>

                    <div className="mlb-matchup-middle">
                      <div className="mlb-edge-pill">Model edge: {predictedTeam || 'TBD'}</div>
                      <div className="mlb-vs-marker">vs</div>
                      <div className="mlb-middle-notes">
                        <span>{board.homeTeamDetails?.venue || board.course}</span>
                        <span>{board.homeTeamDetails?.weather || 'Weather pending'}</span>
                      </div>
                    </div>

                    <div className="mlb-team-column align-right">
                      <div className="mlb-team-header team-header-right">
                        <div>
                          <div className="mlb-team-name">{board.homeTeam}</div>
                          <div className="mlb-team-subtext">{board.homeStarter || 'Starter pending'}</div>
                        </div>
                        <TeamLogo
                          logoUrl={board.homeTeamDetails?.logoUrl}
                          label={board.homeTeam || 'Home'}
                          abbreviation={board.homeTeamDetails?.abbreviation}
                          primaryColor={board.homeTeamDetails?.primaryColor}
                          size="md"
                        />
                      </div>
                      <div className="mlb-probability-stack">
                        <strong>{formatProbability(homeProb)}</strong>
                        <span>home win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.homeTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.homeTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
                      {board.homeStarter ? (
                        <div className="mlb-starter-inline">
                          <PlayerProfileCard
                            name={board.homeStarter}
                            profile={board.homeStarterProfile}
                            compact
                            align="right"
                          />
                        </div>
                      ) : null}
                    </div>
                  </div>

                  {expanded && (
                    <div className="mlb-expanded-panel">
                      <div className="mlb-starter-profile-grid">
                        <div className="mlb-starter-profile-card">
                          <PlayerProfileCard
                            name={board.awayStarter || `${board.awayTeam || 'Away'} starter`}
                            profile={board.awayStarterProfile}
                          />
                        </div>
                        <div className="mlb-starter-profile-card">
                          <PlayerProfileCard
                            name={board.homeStarter || `${board.homeTeam || 'Home'} starter`}
                            profile={board.homeStarterProfile}
                            align="right"
                          />
                        </div>
                      </div>

                      <div className="mlb-team-details-grid">
                        <div className="mlb-team-detail-card">
                          <div className="mlb-team-detail-heading">
                            <TeamLogo
                              logoUrl={board.awayTeamDetails?.logoUrl}
                              label={board.awayTeam || 'Away'}
                              abbreviation={board.awayTeamDetails?.abbreviation}
                              primaryColor={board.awayTeamDetails?.primaryColor}
                              size="sm"
                            />
                            <div>
                              <h3>{board.awayTeam}</h3>
                              <p>{board.awayTeamDetails?.recordPrior || 'Record pending'}</p>
                            </div>
                          </div>
                          <ul className="mlb-detail-list">
                            <li><span>Recent form</span><strong>{board.awayTeamDetails?.recentForm || 'Pending'}</strong></li>
                            <li><span>Bullpen</span><strong>{board.awayTeamDetails?.bullpenSummary || 'Pending'}</strong></li>
                            <li><span>Availability</span><strong>{board.awayTeamDetails?.availabilitySummary || 'Pending'}</strong></li>
                            <li><span>Lineup continuity</span><strong>{board.awayTeamDetails?.lineupContinuity || 'Pending'}</strong></li>
                          </ul>
                        </div>

                        <div className="mlb-team-detail-card">
                          <div className="mlb-team-detail-heading">
                            <TeamLogo
                              logoUrl={board.homeTeamDetails?.logoUrl}
                              label={board.homeTeam || 'Home'}
                              abbreviation={board.homeTeamDetails?.abbreviation}
                              primaryColor={board.homeTeamDetails?.primaryColor}
                              size="sm"
                            />
                            <div>
                              <h3>{board.homeTeam}</h3>
                              <p>{board.homeTeamDetails?.recordPrior || 'Record pending'}</p>
                            </div>
                          </div>
                          <ul className="mlb-detail-list">
                            <li><span>Recent form</span><strong>{board.homeTeamDetails?.recentForm || 'Pending'}</strong></li>
                            <li><span>Bullpen</span><strong>{board.homeTeamDetails?.bullpenSummary || 'Pending'}</strong></li>
                            <li><span>Availability</span><strong>{board.homeTeamDetails?.availabilitySummary || 'Pending'}</strong></li>
                            <li><span>Lineup continuity</span><strong>{board.homeTeamDetails?.lineupContinuity || 'Pending'}</strong></li>
                          </ul>
                        </div>
                      </div>

                      <div className="mlb-lineup-grid">
                        <MlbLineupCard
                          teamName={board.awayTeam}
                          teamDetails={board.awayTeamDetails}
                          lineup={board.awayLineup}
                          emptyLabel="Projected away lineup is still populating."
                          summaryLabel="projected bats"
                        />
                        <MlbLineupCard
                          teamName={board.homeTeam}
                          teamDetails={board.homeTeamDetails}
                          lineup={board.homeLineup}
                          emptyLabel="Projected home lineup is still populating."
                          summaryLabel="projected bats"
                        />
                      </div>

                      <div className="mlb-radar-grid">
                        <StarterRadarChart
                          title={board.awayStarter || `${board.awayTeam || 'Away'} starter`}
                          subtitle={`${board.awayTeamDetails?.abbreviation || 'AWY'} probable starter`}
                          metrics={board.awayStarterRadar}
                          accentColor={board.awayTeamDetails?.primaryColor || '#4d99ff'}
                        />
                        <StarterRadarChart
                          title={board.homeStarter || `${board.homeTeam || 'Home'} starter`}
                          subtitle={`${board.homeTeamDetails?.abbreviation || 'HME'} probable starter`}
                          metrics={board.homeStarterRadar}
                          accentColor={board.homeTeamDetails?.primaryColor || '#24d3b9'}
                        />
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderBasketballUpcomingBoard = () => {
    const availableDates = sportsBoards.basketball.availableDates || [];

    return (
      <div className="upcoming-events-container">
        <div className="event-selector event-selector-mlb">
          <div className="selector-group selector-group-primary">
            <label>Select Basketball slate:</label>
            <select
              value={basketballSelectedDate}
              onChange={(event) => handleBasketballDateChange(event.target.value)}
              className="tournament-dropdown"
            >
              {availableDates.map((dateOption) => (
                <option key={dateOption.dateKey} value={dateOption.dateKey}>
                  {dateOption.label} | {dateOption.gameCount} games
                </option>
              ))}
            </select>
          </div>
          <div className="mlb-slate-summary">
            <span className="mlb-slate-badge">Single-day board</span>
            <p>Showing every Basketball matchup for {basketballSelectedLabel}.</p>
          </div>
        </div>

        {!currentUpcoming.length ? (
          <div className="tournament-card no-live-board-card">
            <div className="tournament-header">
              <div className="header-left">
                <h2>No Basketball games found for this slate</h2>
                <span className="market-status staged">Live schedule filtered</span>
              </div>
            </div>
            <div className="tournament-details">
              <p>We only show boards for the selected date. If there are no games today, pick the next active slate from the selector above.</p>
            </div>
          </div>
        ) : (
          <div className="mlb-board-list">
            {currentUpcoming.map((board) => {
              const awayProb = probabilityForSide(board, 'away');
              const homeProb = probabilityForSide(board, 'home');
              const predictedTeam = predictedTeamFromProbabilities(board, awayProb, homeProb);
              const expanded = expandedBasketballMatchupId === board.id;

              return (
                <article key={board.id} className={`mlb-matchup-card ${expanded ? 'expanded' : ''}`}>
                  <div className="mlb-matchup-top">
                    <div className="mlb-matchup-copy">
                      <div className="mlb-matchup-meta">
                        <span className="spotlight-tour-pill">{board.tour}</span>
                        <span>{board.course}</span>
                        <span>{predictionSourceLabel(board.predictionSource)}</span>
                      </div>
                      <h2>{board.name}</h2>
                      <p className="mlb-matchup-subtitle">
                        Expected rotation strength and availability heading into tip-off.
                      </p>
                    </div>
                    <button
                      className="btn btn-outline btn-sm"
                      onClick={() => setExpandedBasketballMatchupId(expanded ? '' : board.id)}
                    >
                      {expanded ? 'Hide details' : 'Show details'}
                    </button>
                  </div>

                  <div className="mlb-matchup-body">
                    <div className="mlb-team-column">
                      <div className="mlb-team-header">
                        <TeamLogo
                          logoUrl={board.awayTeamDetails?.logoUrl}
                          label={board.awayTeam || 'Away'}
                          abbreviation={board.awayTeamDetails?.abbreviation}
                          primaryColor={board.awayTeamDetails?.primaryColor}
                          size="md"
                        />
                        <div>
                          <div className="mlb-team-name">{board.awayTeam}</div>
                          <div className="mlb-team-subtext">{board.awayTeamDetails?.recordPrior || 'Record pending'}</div>
                        </div>
                      </div>
                      <div className="mlb-probability-stack">
                        <strong>{formatProbability(awayProb)}</strong>
                        <span>away win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.awayTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.awayTeamDetails?.availabilitySummary || 'Rotation stable'}</span>
                      </div>
                      {board.awayFeaturedPlayer ? (
                        <div className="basketball-featured-inline">
                          <PlayerProfileCard
                            name={board.awayFeaturedPlayer.playerName}
                            profile={board.awayFeaturedPlayer.profile}
                            compact
                          />
                        </div>
                      ) : null}
                    </div>

                    <div className="mlb-matchup-middle">
                      <div className="mlb-edge-pill">Model edge: {predictedTeam || 'TBD'}</div>
                      <div className="mlb-vs-marker">vs</div>
                      <div className="mlb-middle-notes">
                        <span>{board.homeTeamDetails?.venue || board.course}</span>
                        <span>{board.homeTeamDetails?.lineupContinuity || 'Continuity pending'}</span>
                      </div>
                    </div>

                    <div className="mlb-team-column align-right">
                      <div className="mlb-team-header team-header-right">
                        <div>
                          <div className="mlb-team-name">{board.homeTeam}</div>
                          <div className="mlb-team-subtext">{board.homeTeamDetails?.recordPrior || 'Record pending'}</div>
                        </div>
                        <TeamLogo
                          logoUrl={board.homeTeamDetails?.logoUrl}
                          label={board.homeTeam || 'Home'}
                          abbreviation={board.homeTeamDetails?.abbreviation}
                          primaryColor={board.homeTeamDetails?.primaryColor}
                          size="md"
                        />
                      </div>
                      <div className="mlb-probability-stack">
                        <strong>{formatProbability(homeProb)}</strong>
                        <span>home win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.homeTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.homeTeamDetails?.availabilitySummary || 'Rotation stable'}</span>
                      </div>
                      {board.homeFeaturedPlayer ? (
                        <div className="basketball-featured-inline">
                          <PlayerProfileCard
                            name={board.homeFeaturedPlayer.playerName}
                            profile={board.homeFeaturedPlayer.profile}
                            compact
                            align="right"
                          />
                        </div>
                      ) : null}
                    </div>
                  </div>

                  {expanded && (
                    <div className="mlb-expanded-panel">
                      <div className="basketball-featured-grid">
                        <BasketballFeaturedPlayerCard
                          teamName={board.awayTeam}
                          player={board.awayFeaturedPlayer}
                          accentColor={board.awayTeamDetails?.primaryColor || '#4d99ff'}
                        />
                        <BasketballFeaturedPlayerCard
                          teamName={board.homeTeam}
                          player={board.homeFeaturedPlayer}
                          accentColor={board.homeTeamDetails?.primaryColor || '#24d3b9'}
                        />
                      </div>

                      <div className="mlb-team-details-grid">
                        <div className="mlb-team-detail-card">
                          <div className="mlb-team-detail-heading">
                            <TeamLogo
                              logoUrl={board.awayTeamDetails?.logoUrl}
                              label={board.awayTeam || 'Away'}
                              abbreviation={board.awayTeamDetails?.abbreviation}
                              primaryColor={board.awayTeamDetails?.primaryColor}
                              size="sm"
                            />
                            <div>
                              <h3>{board.awayTeam}</h3>
                              <p>{board.awayTeamDetails?.recordPrior || 'Record pending'}</p>
                            </div>
                          </div>
                          <ul className="mlb-detail-list">
                            <li><span>Recent form</span><strong>{board.awayTeamDetails?.recentForm || 'Pending'}</strong></li>
                            <li><span>Availability</span><strong>{board.awayTeamDetails?.availabilitySummary || 'Pending'}</strong></li>
                            <li><span>Lineup continuity</span><strong>{board.awayTeamDetails?.lineupContinuity || 'Pending'}</strong></li>
                          </ul>
                        </div>

                        <div className="mlb-team-detail-card">
                          <div className="mlb-team-detail-heading">
                            <TeamLogo
                              logoUrl={board.homeTeamDetails?.logoUrl}
                              label={board.homeTeam || 'Home'}
                              abbreviation={board.homeTeamDetails?.abbreviation}
                              primaryColor={board.homeTeamDetails?.primaryColor}
                              size="sm"
                            />
                            <div>
                              <h3>{board.homeTeam}</h3>
                              <p>{board.homeTeamDetails?.recordPrior || 'Record pending'}</p>
                            </div>
                          </div>
                          <ul className="mlb-detail-list">
                            <li><span>Recent form</span><strong>{board.homeTeamDetails?.recentForm || 'Pending'}</strong></li>
                            <li><span>Availability</span><strong>{board.homeTeamDetails?.availabilitySummary || 'Pending'}</strong></li>
                            <li><span>Lineup continuity</span><strong>{board.homeTeamDetails?.lineupContinuity || 'Pending'}</strong></li>
                          </ul>
                        </div>
                      </div>

                      <div className="mlb-lineup-grid">
                        <MlbLineupCard
                          teamName={board.awayTeam}
                          teamDetails={board.awayTeamDetails}
                          lineup={board.awayLineup}
                          emptyLabel="Projected away rotation is still populating."
                          summaryLabel="projected rotation"
                        />
                        <MlbLineupCard
                          teamName={board.homeTeam}
                          teamDetails={board.homeTeamDetails}
                          lineup={board.homeLineup}
                          emptyLabel="Projected home rotation is still populating."
                          summaryLabel="projected rotation"
                        />
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderFootballUpcomingBoard = () => {
    const availableDates = sportsBoards.football.availableDates || [];

    return (
      <div className="upcoming-events-container">
        <div className="event-selector event-selector-mlb">
          <div className="selector-group selector-group-primary">
            <label>Select Football slate:</label>
            <select
              value={footballSelectedDate}
              onChange={(event) => handleFootballDateChange(event.target.value)}
              className="tournament-dropdown"
            >
              {availableDates.map((dateOption) => (
                <option key={dateOption.dateKey} value={dateOption.dateKey}>
                  {dateOption.label} | {dateOption.gameCount} games
                </option>
              ))}
            </select>
          </div>
          <div className="mlb-slate-summary">
            <span className="mlb-slate-badge">Single-day board</span>
            <p>Showing every Football matchup for {footballSelectedLabel}.</p>
          </div>
        </div>

        {!currentUpcoming.length ? (
          <div className="tournament-card no-live-board-card">
            <div className="tournament-header">
              <div className="header-left">
                <h2>No Football games found for this slate</h2>
                <span className="market-status staged">Seasonal schedule filtered</span>
              </div>
            </div>
            <div className="tournament-details">
              <p>Football boards are date-based. In the offseason this view will be empty, and it will repopulate as soon as the next regular-season slate is available.</p>
            </div>
          </div>
        ) : (
          <div className="mlb-board-list">
            {currentUpcoming.map((board) => {
              const awayProb = probabilityForSide(board, 'away');
              const homeProb = probabilityForSide(board, 'home');
              const predictedTeam = predictedTeamFromProbabilities(board, awayProb, homeProb);
              const expanded = expandedFootballMatchupId === board.id;

              return (
                <article key={board.id} className={`mlb-matchup-card ${expanded ? 'expanded' : ''}`}>
                  <div className="mlb-matchup-top">
                    <div className="mlb-matchup-copy">
                      <div className="mlb-matchup-meta">
                        <span className="spotlight-tour-pill">Football</span>
                        <span>{board.course}</span>
                        <span>{predictionSourceLabel(board.predictionSource)}</span>
                      </div>
                      <h2>{board.name}</h2>
                      <p className="mlb-matchup-subtitle">
                        Probable quarterbacks: {board.awayStarter || 'QB pending'} vs {board.homeStarter || 'QB pending'}
                      </p>
                    </div>
                    <button
                      className="btn btn-outline btn-sm"
                      onClick={() => setExpandedFootballMatchupId(expanded ? '' : board.id)}
                    >
                      {expanded ? 'Hide details' : 'Show details'}
                    </button>
                  </div>

                  <div className="mlb-matchup-body">
                    <div className="mlb-team-column">
                      <div className="mlb-team-header">
                        <TeamLogo
                          logoUrl={board.awayTeamDetails?.logoUrl}
                          label={board.awayTeam || 'Away'}
                          abbreviation={board.awayTeamDetails?.abbreviation}
                          primaryColor={board.awayTeamDetails?.primaryColor}
                          size="md"
                        />
                        <div>
                          <div className="mlb-team-name">{board.awayTeam}</div>
                          <div className="mlb-team-subtext">{board.awayStarter || 'QB pending'}</div>
                        </div>
                      </div>
                      <div className="mlb-probability-stack">
                        <strong>{formatProbability(awayProb)}</strong>
                        <span>away win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.awayTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.awayTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
                      {board.awayStarter ? (
                        <div className="mlb-starter-inline">
                          <PlayerProfileCard
                            name={board.awayStarter}
                            profile={board.awayStarterProfile}
                            compact
                          />
                        </div>
                      ) : null}
                    </div>

                    <div className="mlb-matchup-middle">
                      <div className="mlb-edge-pill">Model edge: {predictedTeam || 'TBD'}</div>
                      <div className="mlb-vs-marker">vs</div>
                      <div className="mlb-middle-notes">
                        <span>{board.homeTeamDetails?.venue || board.course}</span>
                        <span>{board.homeTeamDetails?.weather || 'Weather pending'}</span>
                      </div>
                    </div>

                    <div className="mlb-team-column align-right">
                      <div className="mlb-team-header team-header-right">
                        <div>
                          <div className="mlb-team-name">{board.homeTeam}</div>
                          <div className="mlb-team-subtext">{board.homeStarter || 'QB pending'}</div>
                        </div>
                        <TeamLogo
                          logoUrl={board.homeTeamDetails?.logoUrl}
                          label={board.homeTeam || 'Home'}
                          abbreviation={board.homeTeamDetails?.abbreviation}
                          primaryColor={board.homeTeamDetails?.primaryColor}
                          size="md"
                        />
                      </div>
                      <div className="mlb-probability-stack">
                        <strong>{formatProbability(homeProb)}</strong>
                        <span>home win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.homeTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.homeTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
                      {board.homeStarter ? (
                        <div className="mlb-starter-inline">
                          <PlayerProfileCard
                            name={board.homeStarter}
                            profile={board.homeStarterProfile}
                            compact
                            align="right"
                          />
                        </div>
                      ) : null}
                    </div>
                  </div>

                  {expanded && (
                    <div className="mlb-expanded-panel">
                      <div className="mlb-starter-profile-grid">
                        <div className="mlb-starter-profile-card">
                          <PlayerProfileCard
                            name={board.awayStarter || `${board.awayTeam || 'Away'} QB`}
                            profile={board.awayStarterProfile}
                          />
                        </div>
                        <div className="mlb-starter-profile-card">
                          <PlayerProfileCard
                            name={board.homeStarter || `${board.homeTeam || 'Home'} QB`}
                            profile={board.homeStarterProfile}
                            align="right"
                          />
                        </div>
                      </div>

                      <div className="mlb-team-details-grid">
                        <div className="mlb-team-detail-card">
                          <div className="mlb-team-detail-heading">
                            <TeamLogo
                              logoUrl={board.awayTeamDetails?.logoUrl}
                              label={board.awayTeam || 'Away'}
                              abbreviation={board.awayTeamDetails?.abbreviation}
                              primaryColor={board.awayTeamDetails?.primaryColor}
                              size="sm"
                            />
                            <div>
                              <h3>{board.awayTeam}</h3>
                              <p>{board.awayTeamDetails?.recordPrior || 'Record pending'}</p>
                            </div>
                          </div>
                          <ul className="mlb-detail-list">
                            <li><span>Recent form</span><strong>{board.awayTeamDetails?.recentForm || 'Pending'}</strong></li>
                            <li><span>Offense</span><strong>{board.awayTeamDetails?.bullpenSummary || 'Pending'}</strong></li>
                            <li><span>Availability</span><strong>{board.awayTeamDetails?.availabilitySummary || 'Pending'}</strong></li>
                            <li><span>Depth</span><strong>{board.awayTeamDetails?.lineupContinuity || 'Pending'}</strong></li>
                          </ul>
                        </div>

                        <div className="mlb-team-detail-card">
                          <div className="mlb-team-detail-heading">
                            <TeamLogo
                              logoUrl={board.homeTeamDetails?.logoUrl}
                              label={board.homeTeam || 'Home'}
                              abbreviation={board.homeTeamDetails?.abbreviation}
                              primaryColor={board.homeTeamDetails?.primaryColor}
                              size="sm"
                            />
                            <div>
                              <h3>{board.homeTeam}</h3>
                              <p>{board.homeTeamDetails?.recordPrior || 'Record pending'}</p>
                            </div>
                          </div>
                          <ul className="mlb-detail-list">
                            <li><span>Recent form</span><strong>{board.homeTeamDetails?.recentForm || 'Pending'}</strong></li>
                            <li><span>Offense</span><strong>{board.homeTeamDetails?.bullpenSummary || 'Pending'}</strong></li>
                            <li><span>Availability</span><strong>{board.homeTeamDetails?.availabilitySummary || 'Pending'}</strong></li>
                            <li><span>Depth</span><strong>{board.homeTeamDetails?.lineupContinuity || 'Pending'}</strong></li>
                          </ul>
                        </div>
                      </div>

                      <div className="mlb-lineup-grid">
                        <MlbLineupCard
                          teamName={board.awayTeam}
                          teamDetails={board.awayTeamDetails}
                          lineup={board.awayLineup}
                          emptyLabel="Projected away skill players are still populating."
                          summaryLabel="featured players"
                        />
                        <MlbLineupCard
                          teamName={board.homeTeam}
                          teamDetails={board.homeTeamDetails}
                          lineup={board.homeLineup}
                          emptyLabel="Projected home skill players are still populating."
                          summaryLabel="featured players"
                        />
                      </div>

                      <div className="mlb-radar-grid">
                        <StarterRadarChart
                          title={board.awayStarter || `${board.awayTeam || 'Away'} QB`}
                          subtitle={`${board.awayTeamDetails?.abbreviation || 'AWY'} quarterback form`}
                          metrics={board.awayStarterRadar}
                          accentColor={board.awayTeamDetails?.primaryColor || '#4d99ff'}
                        />
                        <StarterRadarChart
                          title={board.homeStarter || `${board.homeTeam || 'Home'} QB`}
                          subtitle={`${board.homeTeamDetails?.abbreviation || 'HME'} quarterback form`}
                          metrics={board.homeStarterRadar}
                          accentColor={board.homeTeamDetails?.primaryColor || '#24d3b9'}
                        />
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="dashboard-page sports-dashboard">
      <DashboardHeader activePage="sports" />

      <main className="dashboard-main">
        <div className="dashboard-title sports-dashboard-title">
          <div>
            <h1>Sports Prediction Boards</h1>
            <p className="subtitle">
              Scan live boards and past results across golf, tennis, Basketball, Baseball, and Football without any betting or trading layer.
            </p>
          </div>
          {sportUpdatedAt && (
            <div className="sports-runtime-meta">
              <span className="sports-runtime-pill">Runtime feed</span>
              <span className="sports-runtime-stamp">Updated {sportUpdatedAt}</span>
            </div>
          )}
        </div>

        <div className="sports-navigation">
          <div className="sports-tabs">
            <button className={`sport-tab ${activeSport === 'Golf' ? 'active' : ''}`} onClick={() => handleSportChange('Golf')}>
              Golf
            </button>
            <button className={`sport-tab ${activeSport === 'Tennis' ? 'active' : ''}`} onClick={() => handleSportChange('Tennis')}>
              Tennis
            </button>
            <button className={`sport-tab ${activeSport === 'Basketball' ? 'active' : ''}`} onClick={() => handleSportChange('Basketball')}>
              Basketball
            </button>
            <button className={`sport-tab ${activeSport === 'Baseball' ? 'active' : ''}`} onClick={() => handleSportChange('Baseball')}>
              Baseball
            </button>
            <button className={`sport-tab ${activeSport === 'Football' ? 'active' : ''}`} onClick={() => handleSportChange('Football')}>
              Football
            </button>
            <button className={`sport-tab ${activeSport === 'Soccer' ? 'active' : ''}`} onClick={() => handleSportChange('Soccer')}>
              Soccer
            </button>
            <button className={`sport-tab ${activeSport === 'Hockey' ? 'active' : ''} disabled-tab`} onClick={() => handleSportChange('Hockey')}>
              Hockey <span className="badge-tbd">TBD</span>
            </button>
          </div>
        </div>

        <div className="sports-content">
          {(activeSport === 'Golf' || activeSport === 'Tennis' || activeSport === 'Basketball' || activeSport === 'Baseball' || activeSport === 'Football' || activeSport === 'Soccer') ? (
            <div className="pga-market-container">
              <div className="pga-tabs">
                <button className={`toggle-btn ${activeTab === 'upcoming' ? 'active' : ''}`} onClick={() => setActiveTab('upcoming')}>
                  Live Boards
                </button>
                <button className={`toggle-btn ${activeTab === 'backtest' ? 'active' : ''}`} onClick={() => setActiveTab('backtest')}>
                  Track Record
                </button>
              </div>

              {boardsLoading ? (
                <div className="tournament-card">
                  <div className="dashboard-loading compact-loading">
                    <div className="spinner" />
                    <p>Loading live sports boards...</p>
                  </div>
                </div>
              ) : boardsError ? (
                <div className="tournament-card no-live-board-card">
                  <div className="tournament-header">
                    <div className="header-left">
                      <h2>Live sports feed is temporarily unavailable</h2>
                      <span className="market-status staged">Retry shortly</span>
                    </div>
                  </div>
                  <div className="tournament-details">
                    <p>{boardsError}</p>
                    <p>We're hiding stale boards instead of showing old matchups, so this section only fills when the runtime feed is healthy.</p>
                  </div>
                </div>
              ) : activeTab === 'upcoming' ? (
                activeSport === 'Baseball'
                  ? renderMlbUpcomingBoard()
                  : activeSport === 'Basketball'
                    ? renderBasketballUpcomingBoard()
                    : activeSport === 'Football'
                      ? renderFootballUpcomingBoard()
                      : renderGenericUpcomingBoard()
              ) : (
                <div className="backtest-container">
                  <div className="backtest-header-area">
                    <div>
                      <h2>{currentTrackRecordTitle}</h2>
                      <p className="backtest-desc">
                        {activeSport === 'Baseball'
                          ? "Review historical Baseball game boards and compare the model's top side against the actual winner."
                          : activeSport === 'Basketball'
                            ? "Review historical Basketball game boards and compare the model's top side against the actual winner."
                            : activeSport === 'Football'
                              ? "Review historical Football game boards and compare the model's top side against the actual winner."
                            : "See how often the board's highest-ranked names landed the eventual winner, Top 3, or Top 5."}
                      </p>
                      <SeasonSummaryCards summary={currentSeasonSummary} isTeamSport={isTeamSport} />
                    </div>
                    <div className="backtest-filters">
                      {activeSport === 'Tennis' && (
                        <select
                          value={tennisTourFilter}
                          onChange={(event) => setTennisTourFilter(event.target.value as 'All' | 'ATP' | 'WTA')}
                          className="sports-filter-dropdown"
                        >
                          <option value="All">All Tennis</option>
                          <option value="ATP">Men&apos;s (ATP)</option>
                          <option value="WTA">Women&apos;s (WTA)</option>
                        </select>
                      )}
                      {activeSport === 'Golf' && (
                        <select
                          value={golfTourFilter}
                          onChange={(event) => setGolfTourFilter(event.target.value as 'All' | 'PGA' | 'LPGA')}
                          className="sports-filter-dropdown"
                        >
                          <option value="All">All Golf</option>
                          <option value="PGA">Men&apos;s (PGA)</option>
                          <option value="LPGA">Women&apos;s (LPGA)</option>
                        </select>
                      )}
                      <input
                        type="text"
                        placeholder={isTeamSport ? 'Search matchup or winner...' : 'Search tournament or winner...'}
                        value={backtestSearchQuery}
                        onChange={(event) => setBacktestSearchQuery(event.target.value)}
                        className="sports-search-input"
                      />
                      <select
                        value={backtestFilter}
                        onChange={(event) => setBacktestFilter(event.target.value as typeof backtestFilter)}
                        className="sports-filter-dropdown"
                      >
                        <option value="All">All Results</option>
                        <option value="Hit: Top Pick">Hit: Top Pick Only</option>
                        <option value="Hit: Top 3">Hit: Top 3</option>
                        <option value="Hit: Top 5">Hit: Top 5</option>
                        <option value="Miss">Misses Only</option>
                      </select>
                    </div>
                  </div>

                  {filteredBacktests.length === 0 ? (
                    <div className="no-results-message">No historical results found matching your filters.</div>
                  ) : (
                    <div className="backtest-table">
                      <div className="backtest-header">
                        <span>Year</span>
                        <span>{isTeamSport ? 'Matchup' : 'Tournament'}</span>
                        <span>{isTeamSport ? 'Predicted Side' : 'Top Predicted Picks'}</span>
                        <span>Actual Winner</span>
                        <span>Result</span>
                        <span>Details</span>
                      </div>
                      {filteredBacktests.map((backtest: SportsHistoricalBoard, idx) => (
                        <div key={`${backtest.tournament}-${backtest.year}-${idx}`} className="backtest-row-container">
                          <div className={`backtest-row ${backtest.hitStatus !== 'Miss' ? 'hit' : 'miss'}`}>
                            <span>{backtest.year}</span>
                            <div className="tournament-info-col">
                              <strong>{backtest.tournament}</strong>
                              <div className="tour-label">{isTeamSport ? backtest.venue : backtest.tour}</div>
                            </div>
                            <div className="top-picks-col">
                              {isTeamSport ? (
                                <>
                                  <strong>{backtest.predictedWinner}</strong> <small>({((backtest.prob || 0) * 100).toFixed(1)}%)</small><br />
                                  <small>{backtest.awayTeam} at {backtest.homeTeam}</small>
                                </>
                              ) : (
                                <>
                                  <strong>1. {backtest.predictedWinner}</strong> <small>({((backtest.prob || 0) * 100).toFixed(1)}%)</small><br />
                                  <small>2. {backtest.predictedTop3?.[1]} | 3. {backtest.predictedTop3?.[2]}</small><br />
                                  <small>4. {backtest.predictedTop5?.[3]} | 5. {backtest.predictedTop5?.[4]}</small>
                                </>
                              )}
                            </div>
                            <span>{backtest.actualWinner}</span>
                            <span className={`result-badge ${backtest.hitStatus.toLowerCase().replace(' ', '-')}`}>
                              {backtest.hitStatus !== 'Miss' ? `Hit: ${backtest.hitStatus}` : 'Miss'}
                            </span>
                            <button className="btn-text details-toggle" onClick={() => toggleBacktestDetails(idx)}>
                              {expandedBacktest === idx ? 'Hide Board' : 'View Board'}
                            </button>
                          </div>

                          {expandedBacktest === idx && (
                            <div className="backtest-details-panel">
                              {isTeamSport ? (
                                <>
                                  <h4>{activeSport} Game Details</h4>
                                  <div className="mlb-backtest-summary">
                                    <div className="mlb-backtest-summary-column">
                                      <span className="mlb-summary-label">Predicted side</span>
                                      <strong>{backtest.predictedWinner}</strong>
                                      <span>{((backtest.prob || 0) * 100).toFixed(1)}% top-side confidence</span>
                                    </div>
                                    <div className="mlb-backtest-summary-column">
                                      <span className="mlb-summary-label">Actual winner</span>
                                      <strong>{backtest.actualWinner || 'Unknown'}</strong>
                                      <span>{backtest.venue || 'Ballpark'}</span>
                                    </div>
                                  </div>

                                  {activeSport === 'Baseball' || activeSport === 'Football' ? (
                                    <>
                                      <div className="mlb-starter-profile-grid">
                                        <div className="mlb-starter-profile-card">
                                          <PlayerProfileCard
                                            name={backtest.awayStarter || `${backtest.awayTeam || 'Away'} ${activeSport === 'Football' ? 'QB' : 'starter'}`}
                                            profile={backtest.awayStarterProfile}
                                          />
                                        </div>
                                        <div className="mlb-starter-profile-card">
                                          <PlayerProfileCard
                                            name={backtest.homeStarter || `${backtest.homeTeam || 'Home'} ${activeSport === 'Football' ? 'QB' : 'starter'}`}
                                            profile={backtest.homeStarterProfile}
                                            align="right"
                                          />
                                        </div>
                                      </div>

                                      <div className="mlb-radar-grid">
                                        <StarterRadarChart
                                          title={backtest.awayStarter || `${backtest.awayTeam || 'Away'} ${activeSport === 'Football' ? 'QB' : 'starter'}`}
                                          subtitle={`${backtest.awayTeamDetails?.abbreviation || 'AWY'} ${activeSport === 'Football' ? 'quarterback profile' : 'game profile'}`}
                                          metrics={backtest.awayStarterRadar}
                                          accentColor={backtest.awayTeamDetails?.primaryColor || '#4d99ff'}
                                        />
                                        <StarterRadarChart
                                          title={backtest.homeStarter || `${backtest.homeTeam || 'Home'} ${activeSport === 'Football' ? 'QB' : 'starter'}`}
                                          subtitle={`${backtest.homeTeamDetails?.abbreviation || 'HME'} ${activeSport === 'Football' ? 'quarterback profile' : 'game profile'}`}
                                          metrics={backtest.homeStarterRadar}
                                          accentColor={backtest.homeTeamDetails?.primaryColor || '#24d3b9'}
                                        />
                                      </div>
                                    </>
                                  ) : activeSport === 'Basketball' ? (
                                    <div className="basketball-featured-grid">
                                      <BasketballFeaturedPlayerCard
                                        teamName={backtest.awayTeam}
                                        player={backtest.awayFeaturedPlayer}
                                        accentColor={backtest.awayTeamDetails?.primaryColor || '#4d99ff'}
                                      />
                                      <BasketballFeaturedPlayerCard
                                        teamName={backtest.homeTeam}
                                        player={backtest.homeFeaturedPlayer}
                                        accentColor={backtest.homeTeamDetails?.primaryColor || '#24d3b9'}
                                      />
                                    </div>
                                  ) : null}

                                  <div className="mlb-lineup-grid">
                                    <MlbLineupCard
                                      teamName={backtest.awayTeam}
                                      teamDetails={backtest.awayTeamDetails}
                                      lineup={backtest.awayLineup}
                                      emptyLabel={
                                        activeSport === 'Baseball'
                                          ? 'Historical away lineup details were not available.'
                                          : activeSport === 'Football'
                                            ? 'Historical away skill-player details were not available.'
                                            : 'Historical away rotation details were not available.'
                                      }
                                      summaryLabel={
                                        activeSport === 'Baseball'
                                          ? 'recorded bats'
                                          : activeSport === 'Football'
                                            ? 'featured players'
                                            : 'rotation players'
                                      }
                                    />
                                    <MlbLineupCard
                                      teamName={backtest.homeTeam}
                                      teamDetails={backtest.homeTeamDetails}
                                      lineup={backtest.homeLineup}
                                      emptyLabel={
                                        activeSport === 'Baseball'
                                          ? 'Historical home lineup details were not available.'
                                          : activeSport === 'Football'
                                            ? 'Historical home skill-player details were not available.'
                                            : 'Historical home rotation details were not available.'
                                      }
                                      summaryLabel={
                                        activeSport === 'Baseball'
                                          ? 'recorded bats'
                                          : activeSport === 'Football'
                                            ? 'featured players'
                                            : 'rotation players'
                                      }
                                    />
                                  </div>
                                </>
                              ) : (
                                <>
                                  <h4>Full Field Ranking</h4>
                                  <div className="details-grid">
                                    {(backtest.fullField || []).map((player) => (
                                      <ExpandablePredictionCard
                                        key={`${player.rank}-${player.playerName}`}
                                        player={player}
                                        accentColor={activeSport === 'Golf' ? '#2ad7c4' : '#4d99ff'}
                                      />
                                    ))}
                                  </div>
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="tbd-container">
              <h2>{activeSport} Boards Are In Development</h2>
              <p>We are actively building the data and model stack for {activeSport}.</p>
              <p>Check back soon for live probability boards and track record views.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default SportsDashboard;
