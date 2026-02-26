import { Link } from 'react-router-dom';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Start with a watchlist. Get signal guidance in minutes.</h2>
        <p>SeekingBeta helps new investors build confidence with model-backed reference signals.</p>
        <Link to="/signup" className="btn btn-primary btn-lg">
          Create Free Account
        </Link>
      </div>
    </section>
  );
}

export default CallToAction;
