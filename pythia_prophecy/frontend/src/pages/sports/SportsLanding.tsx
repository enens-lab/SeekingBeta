import { Link } from 'react-router-dom';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import { trackEvent } from '../../lib/analytics';
import mastersPredictions from '../../data/masters_preview.json';
import './SportsLanding.css';

function SportsLanding() {
  const top5Predictions = mastersPredictions.slice(0, 5);
  // Scale max width visually for the highest probability
  const maxProb = Math.max(...top5Predictions.map(p => p.winProbability));

  return (
    <>
      <Header />
      <main className="sports-landing">
        <section className="sports-hero">
          <div className="sports-hero-glow"></div>
          <div className="sports-hero-content">
            <span className="sports-badge">New Beta Feature</span>
            <h1 className="sports-title">Advanced Sports Prediction Markets</h1>
            <p className="sports-subtitle">
              SeekingBeta brings predictive modeling to the sports world. 
              Our tournament-aware system evaluates entire Golf fields (PGA & LPGA) to output realistic, mathematically sound win probabilities.
            </p>
            <div className="sports-cta">
              <Link
                to="/signup"
                className="btn btn-primary btn-lg pulse-btn"
                onClick={() => trackEvent('sports_landing_cta_click')}
              >
                Access Full Predictions
              </Link>
            </div>
            <div className="tech-stack-row">
              <span>Data-Driven</span>
              <span className="dot">•</span>
              <span>35,000+ Tournament Rows</span>
              <span className="dot">•</span>
              <span>Proven Track Record</span>
            </div>
          </div>
        </section>

        <section className="sports-preview-section">
          <div className="preview-container">
            <div className="preview-header">
              <h2>April 2026 Masters Tournament • Early Preview</h2>
              <p>Model Win Probability Rankings (Top 5 Displayed)</p>
            </div>
            
            <div className="preview-leaderboard">
              <div className="preview-leaderboard-header">
                <span>Rank</span>
                <span>Player</span>
                <span>Calibrated Probability</span>
              </div>
              {top5Predictions.map((pred) => (
                <div key={pred.rank} className="preview-row">
                  <span className="preview-rank">#{pred.rank}</span>
                  <span className="preview-name">{pred.playerName}</span>
                  <div className="preview-prob-container">
                    <span className="preview-prob-value">{pred.winProbability.toFixed(2)}%</span>
                    <div className="preview-prob-bg">
                      <div 
                        className="preview-prob-fill" 
                        style={{ width: `${(pred.winProbability / maxProb) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="preview-footer">
              <p>View the full 90+ player field, exact odds, and value disparities.</p>
              <Link to="/signup" className="btn btn-outline">Unlock Free Access</Link>
            </div>
          </div>
        </section>

        <section className="sports-features">
          <div className="features-grid">
            <div className="feature-card">
              <h3>Tournament-Aware Ranking</h3>
              <p>Unlike standard predictive models, our system knows there is only one winner. It ranks players dynamically against the specific field strength they are facing.</p>
            </div>
            <div className="feature-card">
              <h3>Field-Relative Skill</h3>
              <p>We leverage advanced relative scoring metrics, neutralizing statistical inflation from easy courses to identify the true elite players.</p>
            </div>
            <div className="feature-card">
              <h3>Course Volatility</h3>
              <p>By analyzing player driving distance and accuracy alongside historical course difficulty, the model autonomously discovers true "course fit."</p>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}

export default SportsLanding;
