import { Link } from 'react-router-dom';

function CallToAction() {
  return (
    <section className="cta-section">
      <div className="cta-content">
        <h2>Ready to enhance your market analysis?</h2>
        <p>Join traders using data-driven signals to inform their decisions.</p>
        <Link to="/signup" className="btn btn-primary btn-lg">
          Get Started Free
        </Link>
      </div>
    </section>
  );
}

export default CallToAction;
