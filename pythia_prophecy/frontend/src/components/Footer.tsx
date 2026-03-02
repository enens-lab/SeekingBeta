import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="site-footer">
      <div className="footer-content">
        <div className="footer-brand">
          <span className="logo-icon">β</span>
          <span className="logo-text">SeekingBeta</span>
        </div>
        <p className="footer-disclaimer">
          SeekingBeta is for informational and research purposes only.
          Past performance does not guarantee future results.
          Always do your own research and consult a financial advisor.
        </p>
        <nav className="footer-links">
          <Link to="/methodology" onClick={() => trackEvent('footer_link_click', { destination: 'methodology' })}>
            Methodology
          </Link>
          <Link to="/terms" onClick={() => trackEvent('footer_link_click', { destination: 'terms' })}>
            Terms
          </Link>
          <Link to="/privacy" onClick={() => trackEvent('footer_link_click', { destination: 'privacy' })}>
            Privacy
          </Link>
          <Link
            to="/refund-cancellation"
            onClick={() => trackEvent('footer_link_click', { destination: 'refund_cancellation' })}
          >
            Refund &amp; Cancellation
          </Link>
        </nav>
        <p className="footer-copyright">
          &copy; {currentYear} SeekingBeta. All rights reserved.
        </p>
      </div>
    </footer>
  );
}

export default Footer;
