import { useState } from 'react';
import DashboardHeader from '../../components/DashboardHeader';
import { trackEvent } from '../../lib/analytics';
import upcomingTournaments from '../../data/upcoming_tournaments.json';
import historicalBacktests from '../../data/historical_backtests.json';
import './SportsDashboard.css';

type SportCategory = 'PGA' | 'Tennis' | 'NBA' | 'MLB' | 'NFL' | 'NHL';

function SportsDashboard() {
  const [activeSport, setActiveSport] = useState<SportCategory>('PGA');
  const [pgaTab, setPgaTab] = useState<'upcoming' | 'backtest'>('upcoming');
  const [activeEventId, setActiveEventId] = useState<string>(upcomingTournaments[0]?.id || '');
  const [showAllPredictions, setShowAllPredictions] = useState(false);
  const [expandedBacktest, setExpandedBacktest] = useState<number | null>(null);

  const handleSportChange = (sport: SportCategory) => {
    trackEvent('sports_category_change', { sport });
    setActiveSport(sport);
  };

  const activeEvent = upcomingTournaments.find(t => t.id === activeEventId) || upcomingTournaments[0];
  const maxProb = activeEvent ? Math.max(...activeEvent.predictions.map(p => p.winProbability)) : 100;
  
  const displayedPredictions = showAllPredictions 
    ? activeEvent?.predictions 
    : activeEvent?.predictions.slice(0, 15);

  const toggleBacktestDetails = (idx: number) => {
    if (expandedBacktest === idx) {
      setExpandedBacktest(null);
    } else {
      setExpandedBacktest(idx);
      trackEvent('sports_backtest_details_click', { tournament: historicalBacktests[idx].tournament });
    }
  };

  return (
    <div className="dashboard-page sports-dashboard">
      <DashboardHeader activePage="sports" />

      <main className="dashboard-main">
        <div className="dashboard-title">
          <h1>Sports Prediction Markets</h1>
          <p className="subtitle">Educational probability models for major sporting events.</p>
        </div>

        <div className="sports-navigation">
          <div className="sports-tabs">
            <button 
              className={`sport-tab ${activeSport === 'PGA' ? 'active' : ''}`}
              onClick={() => handleSportChange('PGA')}
            >
              PGA Golf
            </button>
            <button 
              className={`sport-tab ${activeSport === 'Tennis' ? 'active' : ''} disabled-tab`}
              onClick={() => handleSportChange('Tennis')}
            >
              Tennis (WTA) <span className="badge-tbd">TBD</span>
            </button>
            <button 
              className={`sport-tab ${activeSport === 'NBA' ? 'active' : ''} disabled-tab`}
              onClick={() => handleSportChange('NBA')}
            >
              NBA <span className="badge-tbd">TBD</span>
            </button>
            <button 
              className={`sport-tab ${activeSport === 'MLB' ? 'active' : ''} disabled-tab`}
              onClick={() => handleSportChange('MLB')}
            >
              MLB <span className="badge-tbd">TBD</span>
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
          {activeSport === 'PGA' ? (
            <div className="pga-market-container">
              <div className="pga-tabs">
                <button 
                  className={`toggle-btn ${pgaTab === 'upcoming' ? 'active' : ''}`}
                  onClick={() => setPgaTab('upcoming')}
                >
                  Upcoming Tournaments
                </button>
                <button 
                  className={`toggle-btn ${pgaTab === 'backtest' ? 'active' : ''}`}
                  onClick={() => setPgaTab('backtest')}
                >
                  Historical Backtesting
                </button>
              </div>

              {pgaTab === 'upcoming' && (
                <div className="upcoming-events-container">
                  <div className="event-selector">
                    <label>Select Tournament: </label>
                    <select 
                      value={activeEventId} 
                      onChange={(e) => {
                        setActiveEventId(e.target.value);
                        setShowAllPredictions(false);
                      }}
                      className="tournament-dropdown"
                    >
                      {upcomingTournaments.map(t => (
                        <option key={t.id} value={t.id}>{t.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  {activeEvent && (
                    <div className="tournament-card active-market">
                      <div className="tournament-header">
                        <h2>{activeEvent.name}</h2>
                        <span className="market-status live">Market Live</span>
                      </div>
                      <div className="tournament-details">
                        <p><strong>Course:</strong> {activeEvent.course}</p>
                        <p><strong>Model:</strong> Tournament-Aware Softmax Ranker</p>
                      </div>

                      <div className="prediction-leaderboard">
                        <div className="leaderboard-header">
                          <span>Rank</span>
                          <span>Player</span>
                          <span>Win Probability</span>
                        </div>
                        {displayedPredictions?.map((pred) => (
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
                      
                      {!showAllPredictions && activeEvent.predictions.length > 15 && (
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
                  )}
                </div>
              )}

              {pgaTab === 'backtest' && (
                <div className="backtest-container">
                  <h2>Model Track Record (2024 - 2025)</h2>
                  <p className="backtest-desc">Comparing the model's top predicted picks against the actual tournament winner. We evaluate Top Pick (1st), Top 3, and Top 5 probabilities.</p>
                  
                  <div className="backtest-table">
                    <div className="backtest-header">
                      <span>Year</span>
                      <span>Tournament</span>
                      <span>Top Predicted Picks</span>
                      <span>Actual Winner</span>
                      <span>Result</span>
                      <span>Details</span>
                    </div>
                    {historicalBacktests.map((bt, idx) => (
                      <div key={idx} className="backtest-row-container">
                        <div className={`backtest-row ${bt.hitStatus !== 'Miss' ? 'hit' : 'miss'}`}>
                          <span>{bt.year}</span>
                          <span>{bt.tournament}</span>
                          <div className="top-picks-col">
                            <strong>1. {bt.predictedWinner}</strong> <small>({(bt.prob * 100).toFixed(1)}%)</small><br />
                            <small>2. {bt.predictedTop3?.[1]} | 3. {bt.predictedTop3?.[2]}</small><br />
                            <small>4. {bt.predictedTop5?.[3]} | 5. {bt.predictedTop5?.[4]}</small>
                          </div>
                          <span>{bt.actualWinner}</span>
                          <span className={`result-badge ${bt.hitStatus.toLowerCase().replace(' ', '-')}`}>
                            {bt.hitStatus !== 'Miss' ? `Hit: ${bt.hitStatus}` : 'Miss'}
                          </span>
                          <button 
                            className="btn-text details-toggle"
                            onClick={() => toggleBacktestDetails(idx)}
                          >
                            {expandedBacktest === idx ? 'Hide Details' : 'View Details'}
                          </button>
                        </div>
                        
                        {expandedBacktest === idx && (
                          <div className="backtest-details-panel">
                            <h4>Full Model Field Ranking</h4>
                            <div className="details-grid">
                              {bt.fullField?.map(player => (
                                <div key={player.rank} className={`detail-player ${player.actualWinner ? 'actual-winner-highlight' : ''}`}>
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
                </div>
              )}
            </div>
          ) : (
            <div className="tbd-container">
              <h2>{activeSport} Prediction Models</h2>
              <p>We are actively developing proprietary deep learning models for {activeSport}.</p>
              <p>Check back soon for educational probability signals.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default SportsDashboard;
