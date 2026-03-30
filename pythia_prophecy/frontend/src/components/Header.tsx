import { MouseEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';
import { trackEvent } from '../lib/analytics';
import { useLocation } from 'react-router-dom';

function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const scrollToSection = (e: MouseEvent<HTMLAnchorElement>, sectionId: string) => {
    trackEvent('landing_nav_click', { destination: sectionId });
    if (location.pathname !== '/') {
      return;
    }
    e.preventDefault();
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="site-header">
      <div className="header-content">
        <Link to="/" className="logo">
          <span className="logo-icon">β</span>
          <span className="logo-text">SeekingBeta.AI</span>
        </Link>

        <nav className="nav">
          <div className="nav-dropdown">
            <span className="nav-link nav-dropdown-trigger">Boards ▾</span>
            <div className="nav-dropdown-content">
              <Link to="/dashboard" className="nav-link">Stocks</Link>
              <Link to="/sports" className="nav-link">Sports</Link>
            </div>
          </div>
          <a
            href="/#performance"
            className="nav-link"
            onClick={(e) => scrollToSection(e, 'performance')}
          >
            Track Record
          </a>
          <a
            href="/#features"
            className="nav-link"
            onClick={(e) => scrollToSection(e, 'features')}
          >
            How It Works
          </a>
          <a
            href="/#beta-testers"
            className="nav-link"
            onClick={(e) => scrollToSection(e, 'beta-testers')}
          >
            Founding Beta
          </a>
          <Link
            to="/methodology"
            className="nav-link"
            onClick={() => trackEvent('landing_nav_click', { destination: 'methodology' })}
          >
            Methodology
          </Link>
          <a
            href="https://discord.gg/ckTC8JhWU9"
            className="nav-link"
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => trackEvent('discord_click', { source: 'header_nav' })}
          >
            Discord
          </a>
          <Link
            to="/pricing"
            className="nav-link"
            onClick={() => trackEvent('landing_nav_click', { destination: 'pricing' })}
          >
            Pricing
          </Link>
        </nav>

        <div className="header-actions">
          <ThemeToggle />
          {isAuthenticated ? (
            <>
              <Link
                to="/dashboard"
                className="btn btn-ghost"
                onClick={() => trackEvent('header_dashboard_click')}
              >
                Dashboard
              </Link>
              <button
                className="btn btn-ghost"
                onClick={() => {
                  trackEvent('header_logout_click');
                  handleLogout();
                }}
              >
                Log Out
              </button>
            </>
          ) : (
            <>
              <Link
                to="/login"
                className="btn btn-ghost"
                onClick={() => trackEvent('header_login_click')}
              >
                Log In
              </Link>
              <Link
                to="/signup"
                className="btn btn-primary"
                onClick={() => trackEvent('header_signup_click')}
              >
                Start Free
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
