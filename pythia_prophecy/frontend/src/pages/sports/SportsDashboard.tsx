import { useEffect, useMemo, useState } from 'react';
import DashboardHeader from '../../components/DashboardHeader';
import TeamLogo from '../../components/sports/TeamLogo';
import StarterRadarChart from '../../components/sports/StarterRadarChart';
import {
  sports,
  type SportsBoardCollection,
  type SportsBoardsResponse,
  type SportsHistoricalBoard,
  type SportsUpcomingBoard,
  type SportsBoardPrediction,
} from '../../api/client';
import { trackEvent } from '../../lib/analytics';
import './SportsDashboard.css';

type SportCategory = 'Golf' | 'Tennis' | 'NBA' | 'MLB' | 'NFL' | 'NHL';

const EMPTY_COLLECTION: SportsBoardCollection = {
  upcoming: [],
  backtests: [],
  updated_at: '',
  source: 'runtime_filtered_sports_feed',
  selectedDate: undefined,
  availableDates: [],
};

const EMPTY_SPORTS_BOARDS: SportsBoardsResponse = {
  golf: EMPTY_COLLECTION,
  tennis: EMPTY_COLLECTION,
  mlb: EMPTY_COLLECTION,
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

function probabilityForSide(board: SportsUpcomingBoard, side: 'away' | 'home'): number {
  const label = side === 'away' ? board.awayTeam : board.homeTeam;
  const match = board.predictions.find((prediction) => prediction.side === side || prediction.playerName === label);
  return match?.winProbability ?? 50;
}

function predictionBarWidth(prediction: SportsBoardPrediction, maxProb: number): string {
  const safeMax = maxProb > 0 ? maxProb : 100;
  return `${(prediction.winProbability / safeMax) * 100}%`;
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
  const [expandedMlbMatchupId, setExpandedMlbMatchupId] = useState<string>('');

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

  const sportDataMap = useMemo(
    () => ({
      Golf: sportsBoards.golf,
      Tennis: sportsBoards.tennis,
      MLB: sportsBoards.mlb,
    }),
    [sportsBoards]
  );

  const sportData = activeSport === 'Golf' || activeSport === 'Tennis' || activeSport === 'MLB'
    ? sportDataMap[activeSport]
    : EMPTY_COLLECTION;

  const currentUpcoming = sportData.upcoming ?? [];
  const currentBacktests = sportData.backtests ?? [];

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
    if (activeSport === 'MLB') return;
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
    if (activeSport !== 'MLB') return;
    if (!currentUpcoming.length) {
      setExpandedMlbMatchupId('');
      return;
    }
    if (!currentUpcoming.some((event) => event.id === expandedMlbMatchupId)) {
      setExpandedMlbMatchupId(currentUpcoming[0].id);
    }
  }, [activeSport, currentUpcoming, expandedMlbMatchupId]);

  const activeEvent = useMemo(
    () => filteredUpcoming.find((event) => event.id === activeEventId) || filteredUpcoming[0],
    [filteredUpcoming, activeEventId]
  );

  const sportUpdatedAt = runtimeUpdatedLabel(sportData.updated_at);
  const maxProb = activeEvent ? Math.max(...activeEvent.predictions.map((prediction) => prediction.winProbability)) : 100;
  const mlbSelectedDate = sportData.selectedDate || requestedMlbDate || '';
  const mlbSelectedLabel = (sportData.availableDates || []).find((option) => option.dateKey === mlbSelectedDate)?.label || 'Next active slate';

  const displayedPredictions = useMemo(() => {
    if (activeSport === 'MLB') return [];
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
    if (sport !== 'Golf' && sport !== 'Tennis' && sport !== 'MLB') return;

    trackEvent('sports_category_change', { sport });
    setActiveSport(sport);
    setActiveEventId('');
    setShowAllPredictions(false);
    setExpandedBacktest(null);
    setExpandedMlbMatchupId('');
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
                  <span className="player-name">{prediction.playerName}</span>
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
            <label>Select MLB slate:</label>
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
                <h2>No MLB games found for this slate</h2>
                <span className="market-status staged">Live schedule filtered</span>
              </div>
            </div>
            <div className="tournament-details">
              <p>We only show boards for the selected regular-season date. If MLB is off today, pick the next active slate from the selector above.</p>
            </div>
          </div>
        ) : (
          <div className="mlb-board-list">
            {currentUpcoming.map((board) => {
              const awayProb = probabilityForSide(board, 'away');
              const homeProb = probabilityForSide(board, 'home');
              const predictedSide = awayProb > homeProb ? 'away' : 'home';
              const predictedTeam = predictedSide === 'away' ? board.awayTeam : board.homeTeam;
              const expanded = expandedMlbMatchupId === board.id;

              return (
                <article key={board.id} className={`mlb-matchup-card ${expanded ? 'expanded' : ''}`}>
                  <div className="mlb-matchup-top">
                    <div className="mlb-matchup-copy">
                      <div className="mlb-matchup-meta">
                        <span className="spotlight-tour-pill">MLB</span>
                        <span>{board.course}</span>
                        <span>{board.predictionSource === 'heuristic_fallback' ? 'Fallback scorer' : 'Baseline model'}</span>
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
                        <strong>{awayProb.toFixed(1)}%</strong>
                        <span>away win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.awayTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.awayTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
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
                        <strong>{homeProb.toFixed(1)}%</strong>
                        <span>home win probability</span>
                      </div>
                      <div className="mlb-context-chip-grid">
                        <span className="mlb-context-chip">{board.homeTeamDetails?.recentForm || 'Form pending'}</span>
                        <span className="mlb-context-chip">{board.homeTeamDetails?.availabilitySummary || 'Roster stable'}</span>
                      </div>
                    </div>
                  </div>

                  {expanded && (
                    <div className="mlb-expanded-panel">
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

  return (
    <div className="dashboard-page sports-dashboard">
      <DashboardHeader activePage="sports" />

      <main className="dashboard-main">
        <div className="dashboard-title sports-dashboard-title">
          <div>
            <h1>Sports Prediction Boards</h1>
            <p className="subtitle">
              Scan live-looking probability boards and model replays across golf, tennis, and baseball without any betting or trading layer.
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
            <button className={`sport-tab ${activeSport === 'NBA' ? 'active' : ''} disabled-tab`} onClick={() => handleSportChange('NBA')}>
              NBA <span className="badge-tbd">TBD</span>
            </button>
            <button className={`sport-tab ${activeSport === 'MLB' ? 'active' : ''}`} onClick={() => handleSportChange('MLB')}>
              MLB
            </button>
            <button className={`sport-tab ${activeSport === 'NFL' ? 'active' : ''} disabled-tab`} onClick={() => handleSportChange('NFL')}>
              NFL <span className="badge-tbd">TBD</span>
            </button>
            <button className={`sport-tab ${activeSport === 'NHL' ? 'active' : ''} disabled-tab`} onClick={() => handleSportChange('NHL')}>
              NHL <span className="badge-tbd">TBD</span>
            </button>
          </div>
        </div>

        <div className="sports-content">
          {(activeSport === 'Golf' || activeSport === 'Tennis' || activeSport === 'MLB') ? (
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
                activeSport === 'MLB' ? renderMlbUpcomingBoard() : renderGenericUpcomingBoard()
              ) : (
                <div className="backtest-container">
                  <div className="backtest-header-area">
                    <div>
                      <h2>Track Record (2024 - 2026)</h2>
                      <p className="backtest-desc">
                        {activeSport === 'MLB'
                          ? "Review historical MLB game boards and compare the model's top side against the actual winner."
                          : "See how often the board's highest-ranked names landed the eventual winner, Top 3, or Top 5."}
                      </p>
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
                        placeholder={activeSport === 'MLB' ? 'Search matchup or winner...' : 'Search tournament or winner...'}
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
                        <span>{activeSport === 'MLB' ? 'Matchup' : 'Tournament'}</span>
                        <span>{activeSport === 'MLB' ? 'Predicted Side' : 'Top Predicted Picks'}</span>
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
                              <div className="tour-label">{activeSport === 'MLB' ? backtest.venue : backtest.tour}</div>
                            </div>
                            <div className="top-picks-col">
                              {activeSport === 'MLB' ? (
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
                              <h4>Full Field Ranking</h4>
                              <div className="details-grid">
                                {(backtest.fullField || []).map((player) => (
                                  <div key={`${player.rank}-${player.playerName}`} className={`detail-player ${player.actualWinner ? 'actual-winner-highlight' : ''}`}>
                                    <span className="dp-rank">#{player.rank}</span>
                                    <span className="dp-name">{player.playerName}</span>
                                    <span className="dp-prob">{player.winProbability.toFixed(2)}%</span>
                                    {player.actualWinner && <span className="dp-badge">Winner</span>}
                                  </div>
                                ))}
                              </div>
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
