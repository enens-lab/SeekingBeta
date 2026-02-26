import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

type ActivePage = 'dashboard' | 'oracle' | 'analysis';

interface DashboardHeaderProps {
  activePage?: ActivePage;
  showNav?: boolean;
}

function DashboardHeader({ activePage, showNav = true }: DashboardHeaderProps) {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <header className="dashboard-header">
      <Link to="/" className="logo">
        <span className="logo-icon">β</span>
        <span className="logo-text">SeekingBeta</span>
      </Link>

      {showNav && (
        <nav className="dashboard-nav">
          <Link
            to="/dashboard"
            className={`nav-link${activePage === 'dashboard' ? ' active' : ''}`}
          >
            Dashboard
          </Link>
          <Link
            to="/oracle"
            className={`nav-link${activePage === 'oracle' ? ' active' : ''}`}
          >
            My Oracle
          </Link>
          <Link
            to="/analysis"
            className={`nav-link${activePage === 'analysis' ? ' active' : ''}`}
          >
            Analysis
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
        {showNav && user?.tier !== 'pro' && (
          <Link to="/pricing" className="btn btn-ghost">Upgrade</Link>
        )}
        <button className="btn btn-ghost" onClick={handleLogout}>Log Out</button>
      </div>
    </header>
  );
}

export default DashboardHeader;
