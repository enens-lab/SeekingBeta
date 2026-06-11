import { MouseEvent, useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';
import { trackEvent } from '../lib/analytics';

function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname, location.hash]);

  const handleLogout = () => {
    logout();
    setMobileMenuOpen(false);
    navigate('/');
  };

  const closeMobileMenu = () => setMobileMenuOpen(false);

  const scrollToSection = (e: MouseEvent<HTMLAnchorElement>, sectionId: string) => {
    trackEvent('landing_nav_click', { destination: sectionId });
    if (location.pathname !== '/') {
      closeMobileMenu();
      return;
    }
    e.preventDefault();
    closeMobileMenu();
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className={`site-header${mobileMenuOpen ? ' mobile-menu-open' : ''}`}>
      <div className="header-content">
        <Link to="/" className="logo" onClick={closeMobileMenu}>
          <span className="logo-icon">β</span>
          <span className="logo-text">SeekingBeta.AI</span>
        </Link>

        <nav className="nav desktop-nav">
          <div className="nav-dropdown">
            <span className="nav-link nav-dropdown-trigger">Boards ▾</span>
            <div className="nav-dropdown-content">
              <Link to="/dashboard" className="nav-link">Stocks</Link>
              <Link to="/sports" className="nav-link">Sports</Link>
            </div>
          </div>
          <Link
            to="/track-record"
            className="nav-link"
            onClick={() => trackEvent('landing_nav_click', { destination: 'track_record' })}
          >
            Track Record
          </Link>
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
            Early Access
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

        <div className="header-actions desktop-header-actions">
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

        <button
          type="button"
          className="mobile-nav-toggle"
          aria-expanded={mobileMenuOpen}
          aria-controls="site-mobile-nav"
          aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          onClick={() => setMobileMenuOpen((open) => !open)}
        >
          <span></span>
          <span></span>
          <span></span>
        </button>
      </div>

      <div
        id="site-mobile-nav"
        className={`mobile-nav-panel${mobileMenuOpen ? ' open' : ''}`}
      >
        <div className="mobile-nav-group">
          <span className="mobile-nav-group-label">Boards</span>
          <div className="mobile-nav-links">
            <Link to="/dashboard" className="nav-link" onClick={closeMobileMenu}>
              Stocks
            </Link>
            <Link to="/sports" className="nav-link" onClick={closeMobileMenu}>
              Sports
            </Link>
          </div>
        </div>

        <div className="mobile-nav-group">
          <span className="mobile-nav-group-label">Explore</span>
          <div className="mobile-nav-links">
            <Link
              to="/track-record"
              className="nav-link"
              onClick={() => {
                trackEvent('landing_nav_click', { destination: 'track_record' });
                closeMobileMenu();
              }}
            >
              Track Record
            </Link>
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
              Early Access
            </a>
            <Link
              to="/methodology"
              className="nav-link"
              onClick={() => {
                trackEvent('landing_nav_click', { destination: 'methodology' });
                closeMobileMenu();
              }}
            >
              Methodology
            </Link>
            <a
              href="https://discord.gg/ckTC8JhWU9"
              className="nav-link"
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => {
                trackEvent('discord_click', { source: 'header_nav_mobile' });
                closeMobileMenu();
              }}
            >
              Discord
            </a>
            <Link
              to="/pricing"
              className="nav-link"
              onClick={() => {
                trackEvent('landing_nav_click', { destination: 'pricing' });
                closeMobileMenu();
              }}
            >
              Pricing
            </Link>
          </div>
        </div>

        <div className="mobile-nav-footer">
          <ThemeToggle />
          {isAuthenticated ? (
            <div className="mobile-nav-actions">
              <Link
                to="/dashboard"
                className="btn btn-ghost"
                onClick={() => {
                  trackEvent('header_dashboard_click');
                  closeMobileMenu();
                }}
              >
                Dashboard
              </Link>
              <button
                className="btn btn-primary"
                onClick={() => {
                  trackEvent('header_logout_click');
                  handleLogout();
                }}
              >
                Log Out
              </button>
            </div>
          ) : (
            <div className="mobile-nav-actions">
              <Link
                to="/login"
                className="btn btn-ghost"
                onClick={() => {
                  trackEvent('header_login_click');
                  closeMobileMenu();
                }}
              >
                Log In
              </Link>
              <Link
                to="/signup"
                className="btn btn-primary"
                onClick={() => {
                  trackEvent('header_signup_click');
                  closeMobileMenu();
                }}
              >
                Start Free
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
