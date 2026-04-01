import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';
import { trackEvent } from '../lib/analytics';

type ActivePage = 'dashboard' | 'oracle' | 'analysis' | 'profile' | 'sports';

interface DashboardHeaderProps {
  activePage?: ActivePage;
  showNav?: boolean;
}

function DashboardHeader({ activePage, showNav = true }: DashboardHeaderProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname, location.hash]);

  const handleLogout = () => {
    trackEvent('dashboard_logout_click');
    logout();
    setMobileMenuOpen(false);
    navigate('/');
  };

  return (
    <header className={`dashboard-header${mobileMenuOpen ? ' mobile-menu-open' : ''}`}>
      <div className="dashboard-header-main">
        <Link to="/" className="logo" onClick={() => setMobileMenuOpen(false)}>
          <span className="logo-icon">β</span>
          <span className="logo-text">SeekingBeta.AI</span>
        </Link>

        {showNav && (
          <nav className="dashboard-nav">
            <span className="nav-group-label">Boards:</span>
            <Link
              to="/dashboard"
              className={`nav-link${(activePage === 'dashboard' || activePage === 'oracle' || activePage === 'analysis') ? ' active' : ''}`}
              onClick={() => trackEvent('dashboard_nav_click', { destination: 'dashboard' })}
            >
              Stocks
            </Link>
            <Link
              to="/sports-dashboard"
              className={`nav-link${activePage === 'sports' ? ' active' : ''}`}
              onClick={() => trackEvent('dashboard_nav_click', { destination: 'sports' })}
            >
              Sports
            </Link>
            <div className="nav-divider"></div>
            {activePage !== 'sports' && (
              <>
                <Link
                  to="/oracle"
                  className={`nav-link${activePage === 'oracle' ? ' active-sub' : ''}`}
                  onClick={() => trackEvent('dashboard_nav_click', { destination: 'oracle' })}
                >
                  Watchlist
                </Link>
                <Link
                  to="/analysis"
                  className={`nav-link${activePage === 'analysis' ? ' active-sub' : ''}`}
                  onClick={() => trackEvent('dashboard_nav_click', { destination: 'analysis' })}
                >
                  Analysis
                </Link>
              </>
            )}
            <Link
              to="/profile"
              className={`nav-link${activePage === 'profile' ? ' active' : ''}`}
              onClick={() => trackEvent('dashboard_nav_click', { destination: 'profile' })}
            >
              Profile
            </Link>
          </nav>
        )}

        {showNav && (
          <div className="dashboard-user">
            <span className="user-name">
              {user?.first_name} {user?.last_name}
            </span>
            <span className={`tier-badge tier-${user?.tier}`}>
              {user?.tier}
            </span>
          </div>
        )}

        <div className="header-actions dashboard-desktop-actions">
          <ThemeToggle />
          {showNav && user?.tier !== 'pro' && (
            <Link
              to="/pricing"
              className="btn btn-ghost"
              onClick={() => trackEvent('dashboard_upgrade_click')}
            >
              Unlock More
            </Link>
          )}
          <button className="btn btn-ghost" onClick={handleLogout}>Log Out</button>
        </div>

        {showNav && (
          <button
            type="button"
            className="dashboard-mobile-toggle"
            aria-expanded={mobileMenuOpen}
            aria-controls="dashboard-mobile-nav"
            aria-label={mobileMenuOpen ? 'Close dashboard navigation menu' : 'Open dashboard navigation menu'}
            onClick={() => setMobileMenuOpen((open) => !open)}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
        )}
      </div>

      {showNav && (
        <div
          id="dashboard-mobile-nav"
          className={`dashboard-mobile-panel${mobileMenuOpen ? ' open' : ''}`}
        >
          <div className="dashboard-mobile-meta">
            <div className="dashboard-mobile-user">
              <strong>{user?.first_name} {user?.last_name}</strong>
              <span className={`tier-badge tier-${user?.tier}`}>{user?.tier}</span>
            </div>
            <ThemeToggle />
          </div>

          <div className="dashboard-mobile-links">
            <Link
              to="/dashboard"
              className={`nav-link${(activePage === 'dashboard' || activePage === 'oracle' || activePage === 'analysis') ? ' active' : ''}`}
              onClick={() => {
                trackEvent('dashboard_nav_click', { destination: 'dashboard_mobile' });
                setMobileMenuOpen(false);
              }}
            >
              Stocks
            </Link>
            <Link
              to="/sports-dashboard"
              className={`nav-link${activePage === 'sports' ? ' active' : ''}`}
              onClick={() => {
                trackEvent('dashboard_nav_click', { destination: 'sports_mobile' });
                setMobileMenuOpen(false);
              }}
            >
              Sports
            </Link>
            {activePage !== 'sports' && (
              <>
                <Link
                  to="/oracle"
                  className={`nav-link${activePage === 'oracle' ? ' active-sub' : ''}`}
                  onClick={() => {
                    trackEvent('dashboard_nav_click', { destination: 'oracle_mobile' });
                    setMobileMenuOpen(false);
                  }}
                >
                  Watchlist
                </Link>
                <Link
                  to="/analysis"
                  className={`nav-link${activePage === 'analysis' ? ' active-sub' : ''}`}
                  onClick={() => {
                    trackEvent('dashboard_nav_click', { destination: 'analysis_mobile' });
                    setMobileMenuOpen(false);
                  }}
                >
                  Analysis
                </Link>
              </>
            )}
            <Link
              to="/profile"
              className={`nav-link${activePage === 'profile' ? ' active' : ''}`}
              onClick={() => {
                trackEvent('dashboard_nav_click', { destination: 'profile_mobile' });
                setMobileMenuOpen(false);
              }}
            >
              Profile
            </Link>
          </div>

          <div className="dashboard-mobile-actions">
            {user?.tier !== 'pro' && (
              <Link
                to="/pricing"
                className="btn btn-outline"
                onClick={() => {
                  trackEvent('dashboard_upgrade_click');
                  setMobileMenuOpen(false);
                }}
              >
                Unlock More
              </Link>
            )}
            <button className="btn btn-primary" onClick={handleLogout}>Log Out</button>
          </div>
        </div>
      )}
    </header>
  );
}

export default DashboardHeader;
