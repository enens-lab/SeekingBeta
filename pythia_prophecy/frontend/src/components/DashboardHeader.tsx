import { Link, useNavigate } from 'react-router-dom';
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
  const { user, logout } = useAuth();

  const handleLogout = () => {
    trackEvent('dashboard_logout_click');
    logout();
    navigate('/');
  };

  return (
    <header className="dashboard-header">
      <Link to="/" className="logo">
        <span className="logo-icon">β</span>
        <span className="logo-text">SeekingBeta.AI</span>
      </Link>

      {showNav && (
        <nav className="dashboard-nav">
          <span className="nav-group-label">Markets:</span>
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

      <div className="header-actions">
        <ThemeToggle />
        {showNav && user?.tier !== 'pro' && (
          <Link
            to="/pricing"
            className="btn btn-ghost"
            onClick={() => trackEvent('dashboard_upgrade_click')}
          >
            Upgrade
          </Link>
        )}
        <button className="btn btn-ghost" onClick={handleLogout}>Log Out</button>
      </div>
    </header>
  );
}

export default DashboardHeader;
