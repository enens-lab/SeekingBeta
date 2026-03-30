import { useState, useMemo } from 'react';
import DashboardHeader from '../../components/DashboardHeader';
import { trackEvent } from '../../lib/analytics';
import upcomingTournamentsPGA from '../../data/upcoming_tournaments.json';
import historicalBacktestsPGA from '../../data/historical_backtests.json';
import upcomingTournamentsWTA from '../../data/wta_upcoming_tournaments.json';
import historicalBacktestsWTA from '../../data/wta_historical_backtests.json';
import upcomingTournamentsMLB from '../../data/mlb_upcoming_tournaments.json';
import historicalBacktestsMLB from '../../data/mlb_historical_backtests.json';
import './SportsDashboard.css';

type SportCategory = 'Golf' | 'Tennis' | 'NBA' | 'MLB' | 'NFL' | 'NHL';
type BoardPrediction = {
  rank: number;
  playerName: string;
  winProbability: number;
};
type AvailabilitySummary = {
  ilAdds14?: number;
  ilActivations14?: number;
  rosterMoves14?: number;
};
type SportsBoard = {
  id: string;
  name: string;
  tour: string;
  course: string;
  predictions: BoardPrediction[];
  awayStarter?: string;
  homeStarter?: string;
  awayAvailability?: AvailabilitySummary;
  homeAvailability?: AvailabilitySummary;
};

const sportDataMap = {
  Golf: {
    upcoming: upcomingTournamentsPGA as any[],
    backtests: historicalBacktestsPGA as any[],
  },
  Tennis: {
    upcoming: upcomingTournamentsWTA as any[],
    backtests: historicalBacktestsWTA as any[],
  },
  MLB: {
    upcoming: upcomingTournamentsMLB as any[],
    backtests: historicalBacktestsMLB as any[],
  },
} as const;

function SportsDashboard() {
  const [activeSport, setActiveSport] = useState<SportCategory>('Golf');
  const [pgaTab, setPgaTab] = useState<'upcoming' | 'backtest'>('upcoming');
  
  const sportData = sportDataMap[activeSport as keyof typeof sportDataMap];
  const currentUpcoming = sportData?.upcoming ?? [];
  const currentBacktests = sportData?.backtests ?? [];

  const [activeEventId, setActiveEventId] = useState<string>(currentUpcoming[0]?.id || '');
  const [showAllPredictions, setShowAllPredictions] = useState(false);
  const [expandedBacktest, setExpandedBacktest] = useState<number | null>(null);
  
  // Search and filter states
  const [playerSearchQuery, setPlayerSearchQuery] = useState('');
  const [backtestSearchQuery, setBacktestSearchQuery] = useState('');
  const [backtestFilter, setBacktestFilter] = useState<'All' | 'Hit: Top Pick' | 'Hit: Top 3' | 'Hit: Top 5' | 'Miss'>('All');
  const [tennisTourFilter, setTennisTourFilter] = useState<'All' | 'ATP' | 'WTA'>('All');
  const [golfTourFilter, setGolfTourFilter] = useState<'All' | 'PGA' | 'LPGA'>('All');

  const handleSportChange = (sport: SportCategory) => {
    if (sport !== 'Golf' && sport !== 'Tennis' && sport !== 'MLB') return;
    
    trackEvent('sports_category_change', { sport });
    setActiveSport(sport);
    
    // Reset tournament-specific state when switching sports
    const nextUpcoming = sportDataMap[sport as keyof typeof sportDataMap]?.upcoming ?? [];
    setActiveEventId(nextUpcoming[0]?.id || '');
    setShowAllPredictions(false);
    setExpandedBacktest(null);
    setPlayerSearchQuery('');
    setBacktestSearchQuery('');
    setBacktestFilter('All');
    setTennisTourFilter('All');
    setGolfTourFilter('All');
  };

  // Filter current upcoming by tour
  const filteredUpcoming = useMemo(() => {
    if (activeSport === 'Tennis' && tennisTourFilter !== 'All') {
      return (currentUpcoming as any[]).filter(t => t.tour === tennisTourFilter);
    }
    if (activeSport === 'Golf' && golfTourFilter !== 'All') {
      return (currentUpcoming as any[]).filter(t => t.tour === golfTourFilter);
    }
    return currentUpcoming;
  }, [activeSport, tennisTourFilter, golfTourFilter, currentUpcoming]);

  // Adjust active event if filter changed
  useMemo(() => {
    if (!filteredUpcoming.find(t => t.id === activeEventId)) {
      setActiveEventId(filteredUpcoming[0]?.id || '');
    }
  }, [filteredUpcoming, activeEventId]);

  const activeEvent = (filteredUpcoming.find(t => t.id === activeEventId) || filteredUpcoming[0]) as SportsBoard | undefined;

  const maxProb = activeEvent ? Math.max(...activeEvent.predictions.map((p: BoardPrediction) => p.winProbability)) : 100;
  
  const displayedPredictions = useMemo(() => {
    let filtered = activeEvent?.predictions || [];
    
    if (playerSearchQuery.trim() !== '') {
      const query = playerSearchQuery.toLowerCase();
      filtered = filtered.filter((p: BoardPrediction) => p.playerName.toLowerCase().includes(query));
      return filtered;
    }
    
    return showAllPredictions ? filtered : filtered.slice(0, 15);
  }, [activeEvent, playerSearchQuery, showAllPredictions]);

  const filteredBacktests = useMemo(() => {
    return (currentBacktests as any[]).filter(bt => {
      const matchesSearch = bt.tournament.toLowerCase().includes(backtestSearchQuery.toLowerCase()) || 
                            bt.actualWinner.toLowerCase().includes(backtestSearchQuery.toLowerCase());
      const matchesFilter = backtestFilter === 'All' || bt.hitStatus === backtestFilter.replace('Hit: ', '');
      
      let matchesTour = true;
      if (activeSport === 'Tennis') {
        matchesTour = tennisTourFilter === 'All' || bt.tour === tennisTourFilter;
      } else if (activeSport === 'Golf') {
        matchesTour = golfTourFilter === 'All' || bt.tour === golfTourFilter;
      }
      
      return matchesSearch && matchesFilter && matchesTour;
    });
  }, [currentBacktests, backtestSearchQuery, backtestFilter, tennisTourFilter, golfTourFilter, activeSport]);

  const toggleBacktestDetails = (idx: number) => {
    if (expandedBacktest === idx) {
      setExpandedBacktest(null);
    } else {
      setExpandedBacktest(idx);
      trackEvent('sports_backtest_details_click', { tournament: filteredBacktests[idx].tournament });
    }
  };

  return (
    <div className="dashboard-page sports-dashboard">
      <DashboardHeader activePage="sports" />

      <main className="dashboard-main">
        <div className="dashboard-title">
          <h1>Sports Prediction Boards</h1>
          <p className="subtitle">Scan probability boards and model replays across golf, tennis, and baseball without any betting or trading layer.</p>
        </div>

        <div className="sports-navigation">
          <div className="sports-tabs">
            <button 
              className={`sport-tab ${activeSport === 'Golf' ? 'active' : ''}`}
              onClick={() => handleSportChange('Golf')}
            >
              Golf
            </button>
            <button 
              className={`sport-tab ${activeSport === 'Tennis' ? 'active' : ''}`}
              onClick={() => handleSportChange('Tennis')}
            >
              Tennis
            </button>
            <button 
              className={`sport-tab ${activeSport === 'NBA' ? 'active' : ''} disabled-tab`}
              onClick={() => handleSportChange('NBA')}
            >
              NBA <span className="badge-tbd">TBD</span>
            </button>
            <button 
              className={`sport-tab ${activeSport === 'MLB' ? 'active' : ''}`}
              onClick={() => handleSportChange('MLB')}
            >
              MLB
            </button>
            <button 
              className={`sport-tab ${activeSport === 'NFL' ? 'active' : ''} disabled-tab`}
              onClick={() => handleSportChange('NFL')}
            >
              NFL <span className="badge-tbd">TBD</span>
            </button>
            <button 
              className={`sport-tab ${activeSport === 'NHL' ? 'active' : ''} disabled-tab`}
              onClick={() => handleSportChange('NHL')}
            >
              NHL <span className="badge-tbd">TBD</span>
            </button>
          </div>
        </div>

        <div className="sports-content">
          {(activeSport === 'Golf' || activeSport === 'Tennis' || activeSport === 'MLB') ? (
            <div className="pga-market-container">
              <div className="pga-tabs">
                <button 
                  className={`toggle-btn ${pgaTab === 'upcoming' ? 'active' : ''}`}
                  onClick={() => setPgaTab('upcoming')}
                >
                  Live Boards
                </button>
                <button 
                  className={`toggle-btn ${pgaTab === 'backtest' ? 'active' : ''}`}
                  onClick={() => setPgaTab('backtest')}
                >
                  Track Record
                </button>
              </div>

              {pgaTab === 'upcoming' && (
                <div className="upcoming-events-container">
                  <div className="event-selector">
                    <div className="selector-group">
                      <label>{activeSport === 'MLB' ? 'Select Matchup: ' : 'Select Board: '}</label>
                      <select 
                        value={activeEventId} 
                        onChange={(e) => {
                          setActiveEventId(e.target.value);
                          setShowAllPredictions(false);
                          setPlayerSearchQuery('');
                        }}
                        className="tournament-dropdown"
                      >
                        {filteredUpcoming.map(t => (
                          <option key={t.id} value={t.id}>
                            [{t.tour}] {t.name}
                          </option>
                        ))}
                      </select>
                    </div>

                    {activeSport === 'Tennis' && (
                      <div className="selector-group tour-filter-group">
                        <label>Tour: </label>
                        <select 
                          value={tennisTourFilter}
                          onChange={(e) => setTennisTourFilter(e.target.value as any)}
                          className="sports-filter-dropdown"
                        >
                          <option value="All">All Tennis</option>
                          <option value="ATP">Men's Singles (ATP)</option>
                          <option value="WTA">Women's Singles (WTA)</option>
                        </select>
                      </div>
                    )}

                    {activeSport === 'Golf' && (
                      <div className="selector-group tour-filter-group">
                        <label>Tour: </label>
                        <select 
                          value={golfTourFilter}
                          onChange={(e) => setGolfTourFilter(e.target.value as any)}
                          className="sports-filter-dropdown"
                        >
                          <option value="All">All Golf</option>
                          <option value="PGA">Men's (PGA)</option>
                          <option value="LPGA">Women's (LPGA)</option>
                        </select>
                      </div>
                    )}
                  </div>
                  
                  {activeSport === 'MLB' && currentUpcoming.length === 0 ? (
                    <div className="tournament-card no-live-board-card">
                      <div className="tournament-header">
                        <div className="header-left">
                          <h2>MLB board rollout in progress</h2>
                          <span className="market-status staged">Track record live</span>
                        </div>
                      </div>
                      <div className="tournament-details">
                        <p><strong>Status:</strong> Historical MLB model replays are live now.</p>
                        <p><strong>Next step:</strong> Pregame MLB board exports will appear here once the daily inference job is enabled.</p>
                      </div>
                    </div>
                  ) : activeEvent ? (
                    <div className="tournament-card active-market">
                      <div className="tournament-header">
                        <div className="header-left">
                          <h2>{activeEvent.name}</h2>
                          <span className="market-status live">Board Live</span>
                        </div>
                        <div className="header-search">
                          <input 
                            type="text" 
                            placeholder={`Search contender...`} 
                            value={playerSearchQuery}
                            onChange={(e) => setPlayerSearchQuery(e.target.value)}
                            className="sports-search-input"
                          />
                        </div>
                      </div>
                      <div className="tournament-details">
                        <p><strong>{activeSport === 'Golf' ? 'Course' : activeSport === 'Tennis' ? 'Surface' : 'Venue'}:</strong> {activeEvent.course}</p>
                        <p><strong>Model:</strong> {activeSport === 'MLB' ? 'Home-win classifier with team, starter, lineup, bullpen, and Statcast context' : 'Tournament-aware probability ranker'}</p>
                        {activeSport === 'MLB' && (
                          <>
                            <p><strong>Probable starters:</strong> {activeEvent.awayStarter || 'TBD'} vs {activeEvent.homeStarter || 'TBD'}</p>
                            <p>
                              <strong>Availability pulse:</strong>{' '}
                              Away IL adds (14d): {activeEvent.awayAvailability?.ilAdds14 ?? 0}, Home IL adds (14d): {activeEvent.homeAvailability?.ilAdds14 ?? 0}
                            </p>
                          </>
                        )}
                      </div>

                      {displayedPredictions.length === 0 ? (
                        <div className="no-results-message">
                          No contenders found matching "{playerSearchQuery}"
                        </div>
                      ) : (
                        <div className="prediction-leaderboard">
                          <div className="leaderboard-header">
                            <span>Rank</span>
                            <span>{activeSport === 'MLB' ? 'Side' : 'Player'}</span>
                            <span>Win Probability</span>
                          </div>
                          {displayedPredictions.map((pred: BoardPrediction) => (
                            <div key={pred.rank} className="leaderboard-row">
                              <span className="player-rank">#{pred.rank}</span>
                              <span className="player-name">{pred.playerName}</span>
                              <div className="probability-container">
                                <span className="prob-value">{pred.winProbability.toFixed(2)}%</span>
                                <div className="prob-bar-bg">
                                  <div 
                                    className="prob-bar-fill" 
                                    style={{ width: `${(pred.winProbability / maxProb) * 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {!showAllPredictions && playerSearchQuery === '' && activeEvent.predictions.length > 15 && (
                        <div className="view-all-container">
                          <button 
                            className="btn btn-outline"
                            onClick={() => setShowAllPredictions(true)}
                          >
                            View All Players ({activeEvent.predictions.length})
                          </button>
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
              )}

              {pgaTab === 'backtest' && (
                <div className="backtest-container">
                  <div className="backtest-header-area">
                    <div>
                      <h2>Track Record (2024 - 2025)</h2>
                      <p className="backtest-desc">
                        {activeSport === 'MLB'
                          ? 'Review historical MLB game boards and compare the model’s top side against the actual winner.'
                          : 'See how often the board&apos;s highest-ranked names landed the eventual winner, Top 3, or Top 5.'}
                      </p>
                    </div>
                    <div className="backtest-filters">
                      {activeSport === 'Tennis' && (
                        <select 
                          value={tennisTourFilter}
                          onChange={(e) => setTennisTourFilter(e.target.value as any)}
                          className="sports-filter-dropdown"
                        >
                          <option value="All">All Tennis</option>
                          <option value="ATP">Men's (ATP)</option>
                          <option value="WTA">Women's (WTA)</option>
                        </select>
                      )}
                      {activeSport === 'Golf' && (
                        <select 
                          value={golfTourFilter}
                          onChange={(e) => setGolfTourFilter(e.target.value as any)}
                          className="sports-filter-dropdown"
                        >
                          <option value="All">All Golf</option>
                          <option value="PGA">Men's (PGA)</option>
                          <option value="LPGA">Women's (LPGA)</option>
                        </select>
                      )}
                      <input 
                        type="text" 
                        placeholder={activeSport === 'MLB' ? 'Search matchup or winner...' : 'Search tournament or winner...'} 
                        value={backtestSearchQuery}
                        onChange={(e) => setBacktestSearchQuery(e.target.value)}
                        className="sports-search-input"
                      />
                      <select 
                        value={backtestFilter}
                        onChange={(e) => setBacktestFilter(e.target.value as any)}
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
                    <div className="no-results-message">
                      No historical results found matching your filters.
                    </div>
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
                      {filteredBacktests.map((bt, idx) => (
                        <div key={idx} className="backtest-row-container">
                          <div className={`backtest-row ${bt.hitStatus !== 'Miss' ? 'hit' : 'miss'}`}>
                            <span>{bt.year}</span>
                            <div className="tournament-info-col">
                              <strong>{bt.tournament}</strong>
                              <div className="tour-label">{activeSport === 'MLB' ? bt.venue : bt.tour}</div>
                            </div>
                            <div className="top-picks-col">
                              {activeSport === 'MLB' ? (
                                <>
                                  <strong>{bt.predictedWinner}</strong> <small>({(bt.prob * 100).toFixed(1)}%)</small><br />
                                  <small>{bt.awayTeam} at {bt.homeTeam}</small>
                                </>
                              ) : (
                                <>
                                  <strong>1. {bt.predictedWinner}</strong> <small>({(bt.prob * 100).toFixed(1)}%)</small><br />
                                  <small>2. {bt.predictedTop3?.[1]} | 3. {bt.predictedTop3?.[2]}</small><br />
                                  <small>4. {bt.predictedTop5?.[3]} | 5. {bt.predictedTop5?.[4]}</small>
                                </>
                              )}
                            </div>
                            <span>{bt.actualWinner}</span>
                            <span className={`result-badge ${bt.hitStatus.toLowerCase().replace(' ', '-')}`}>
                              {bt.hitStatus !== 'Miss' ? `Hit: ${bt.hitStatus}` : 'Miss'}
                            </span>
                            <button 
                              className="btn-text details-toggle"
                              onClick={() => toggleBacktestDetails(idx)}
                            >
                              {expandedBacktest === idx ? 'Hide Board' : 'View Board'}
                            </button>
                          </div>
                          
                          {expandedBacktest === idx && (
                            <div className="backtest-details-panel">
                              <h4>Full Field Ranking</h4>
                              <div className="details-grid">
                                {bt.fullField?.map((player: any) => (
                                  <div key={player.rank} className={`detail-player ${player.actualWinner ? 'actual-winner-highlight' : ''}`}>
                                    <span className="dp-rank">#{player.rank}</span>
                                    <span className="dp-name">{player.playerName}</span>
                                    <span className="dp-prob">{player.winProbability.toFixed(2)}%</span>
                                    {player.actualWinner && <span className="dp-badge">Winner</span>}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}                        </div>
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
