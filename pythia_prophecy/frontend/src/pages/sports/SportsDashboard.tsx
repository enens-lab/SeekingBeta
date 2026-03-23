import { useState } from 'react';
import DashboardHeader from '../../components/DashboardHeader';
import { trackEvent } from '../../lib/analytics';
import './SportsDashboard.css';

type SportCategory = 'PGA' | 'Tennis' | 'NBA' | 'MLB' | 'NFL' | 'NHL';

interface TournamentPrediction {
  playerName: string;
  winProbability: number;
  rank: number;
}

// Mock data for the demonstration of 2026 PGA tournament
const mockPGAPredictions: TournamentPrediction[] = [
  { playerName: 'Scottie Scheffler', winProbability: 16.4, rank: 1 },
  { playerName: 'Xander Schauffele', winProbability: 11.2, rank: 2 },
  { playerName: 'Rory McIlroy', winProbability: 9.8, rank: 3 },
  { playerName: 'Collin Morikawa', winProbability: 7.5, rank: 4 },
  { playerName: 'Ludvig Aberg', winProbability: 6.2, rank: 5 },
  { playerName: 'Viktor Hovland', winProbability: 5.1, rank: 6 },
  { playerName: 'Patrick Cantlay', winProbability: 4.3, rank: 7 },
  { playerName: 'Wyndham Clark', winProbability: 3.8, rank: 8 },
  { playerName: 'Tommy Fleetwood', winProbability: 3.1, rank: 9 },
  { playerName: 'Max Homa', winProbability: 2.5, rank: 10 },
];

const mockBacktests = [
  { year: 2025, tournament: 'THE PLAYERS Championship', predictedWinner: 'Scottie Scheffler', actualWinner: 'Scottie Scheffler', hit: true },
  { year: 2024, tournament: 'Masters Tournament', predictedWinner: 'Scottie Scheffler', actualWinner: 'Scottie Scheffler', hit: true },
  { year: 2024, tournament: 'PGA Championship', predictedWinner: 'Rory McIlroy', actualWinner: 'Xander Schauffele', hit: false },
  { year: 2023, tournament: 'U.S. Open', predictedWinner: 'Wyndham Clark', actualWinner: 'Wyndham Clark', hit: true },
];

function SportsDashboard() {
  const [activeSport, setActiveSport] = useState<SportCategory>('PGA');
  const [pgaTab, setPgaTab] = useState<'upcoming' | 'backtest'>('upcoming');

  const handleSportChange = (sport: SportCategory) => {
    trackEvent('sports_category_change', { sport });
    setActiveSport(sport);
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
                <div className="tournament-card active-market">
                  <div className="tournament-header">
                    <h2>April 2026 Masters Tournament</h2>
                    <span className="market-status live">Market Live</span>
                  </div>
                  <div className="tournament-details">
                    <p><strong>Course:</strong> Augusta National Golf Club, Augusta, GA</p>
                    <p><strong>Model:</strong> Tournament-Aware Softmax Ranker</p>
                  </div>

                  <div className="prediction-leaderboard">
                    <div className="leaderboard-header">
                      <span>Rank</span>
                      <span>Player</span>
                      <span>Win Probability</span>
                    </div>
                    {mockPGAPredictions.map((pred) => (
                      <div key={pred.rank} className="leaderboard-row">
                        <span className="player-rank">#{pred.rank}</span>
                        <span className="player-name">{pred.playerName}</span>
                        <div className="probability-container">
                          <span className="prob-value">{pred.winProbability.toFixed(1)}%</span>
                          <div className="prob-bar-bg">
                            <div 
                              className="prob-bar-fill" 
                              style={{ width: `${Math.min(100, pred.winProbability * 3)}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {pgaTab === 'backtest' && (
                <div className="backtest-container">
                  <h2>Model Track Record (2023 - 2025)</h2>
                  <p className="backtest-desc">Comparing the model's top predicted pick against the actual tournament winner.</p>
                  
                  <div className="backtest-table">
                    <div className="backtest-header">
                      <span>Year</span>
                      <span>Tournament</span>
                      <span>Predicted Top Pick</span>
                      <span>Actual Winner</span>
                      <span>Result</span>
                    </div>
                    {mockBacktests.map((bt, idx) => (
                      <div key={idx} className={`backtest-row ${bt.hit ? 'hit' : 'miss'}`}>
                        <span>{bt.year}</span>
                        <span>{bt.tournament}</span>
                        <span>{bt.predictedWinner}</span>
                        <span>{bt.actualWinner}</span>
                        <span className="result-badge">
                          {bt.hit ? '✅ Hit' : '❌ Miss'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="tbd-container">
              <div className="tbd-icon">🚧</div>
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
