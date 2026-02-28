import { Link } from 'react-router-dom';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Build a watchlist and study model behavior in minutes.</h2>
        <p>SeekingBeta is built to help investors learn from model-generated reference ratings and market context.</p>
        <Link to="/signup" className="btn btn-primary btn-lg">
          Create Free Account
        </Link>
      </div>
    </section>
  );
}

export default CallToAction;
