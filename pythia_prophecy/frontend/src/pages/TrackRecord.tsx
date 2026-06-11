import { Link } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import PerformanceComparison from '../components/PerformanceComparison';

function TrackRecord() {
  return (
    <>
      <Header />
      <main className="track-record-page">
        <section className="track-record-intro">
          <div className="track-record-intro-inner">
            <span className="section-kicker">Track Record</span>
            <h1>Check the record before you rely on the picks.</h1>
            <p>
              Every rating SeekingBeta.AI publishes is backed by a model that we score against a
              passive S&amp;P 500 (SPY) benchmark. The scorecard below is generated from our backtest
              artifacts and refreshes as new data is processed — it is the same data we hold ourselves
              to internally.
            </p>
            <p className="track-record-intro-note">
              Past performance does not guarantee future results. These figures are for educational and
              research use only and are not investment advice. See the{' '}
              <Link to="/methodology">methodology</Link> for how the numbers are computed and what
              assumptions go into them.
            </p>
          </div>
        </section>

        <PerformanceComparison />

        <section className="track-record-howto">
          <div className="track-record-howto-inner">
            <h2>How to read this</h2>
            <ul>
              <li>
                <strong>Model Return vs. SPY Return.</strong> Both start from the same notional equity on
                the first backtest date, so the gap between the two lines is the model&apos;s alpha over
                simply holding the index.
              </li>
              <li>
                <strong>Hit rate</strong> is the share of closed trades that finished net-positive after
                the transaction-cost assumption shown in the methodology.
              </li>
              <li>
                <strong>Sharpe &amp; max drawdown</strong> describe the risk taken to earn that return —
                a high return with a deep drawdown is not the same as a steady one.
              </li>
              <li>
                <strong>Horizons differ.</strong> The Core 5-Day and Jackpot 20-Day models are scored
                separately because they target different holding periods; switch tabs above to compare.
              </li>
            </ul>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}

export default TrackRecord;
