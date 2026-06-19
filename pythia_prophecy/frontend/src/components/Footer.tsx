import { Link } from 'react-router-dom';
import { trackEvent } from '../lib/analytics';

function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="site-footer">
      <div className="footer-content">
        <div className="footer-brand">
          <span className="logo-icon">β</span>
          <span className="logo-text">SeekingBeta.AI</span>
        </div>
        <p className="footer-disclaimer">
          SeekingBeta.AI publishes model-generated probabilities for research and education.
          No betting, no trade execution, and no guarantee of future results.
        </p>
        <nav className="footer-links">
          <Link to="/methodology" onClick={() => trackEvent('footer_link_click', { destination: 'methodology' })}>
            Methodology
          </Link>
          <Link to="/support" onClick={() => trackEvent('footer_link_click', { destination: 'support' })}>
            Support
          </Link>
          <a
            href="https://discord.gg/ckTC8JhWU9"
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => trackEvent('discord_click', { source: 'footer' })}
          >
            Discord
          </a>
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
          <Link
            to="/delete-account"
            onClick={() => trackEvent('footer_link_click', { destination: 'delete_account' })}
          >
            Delete Account
          </Link>
        </nav>
        <p className="footer-copyright">
          &copy; {currentYear} SeekingBeta.AI. All rights reserved.
        </p>
      </div>
    </footer>
  );
}

export default Footer;
